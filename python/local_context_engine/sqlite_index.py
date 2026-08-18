"""SQLite FTS5-backed index for high-performance local retrieval.

This is the P0 high-performance backend for Local Context Engine:

- SQLite FTS5 full-text search instead of pure-Python BM25
- disk-persistent index, WAL mode, incremental updates
- precomputed token counts stored with each chunk
- same interface as ``LocalIndex``, so the engine can swap backends

FTS5 is bundled with Python's ``sqlite3`` on most platforms (including the
WSL Python 3.12 used by this repo). If FTS5 is unavailable, the engine can
fall back to ``LocalIndex``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import Chunk, SourceKind
from .text_utils import tokenize

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE NOT NULL,
    source TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL,
    timestamp REAL,
    priority REAL,
    tokens INTEGER NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='rowid',
    tokenize = 'unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SqliteIndex:
    """A SQLite FTS5 index exposing the same API as LocalIndex."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._revision = self._load_revision()

    # ------------------------------------------------------------------
    # Properties / lifecycle
    # ------------------------------------------------------------------
    @property
    def revision(self) -> int:
        return self._revision

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SqliteIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------
    def add(self, chunk: Chunk) -> None:
        self._add(chunk)
        self._conn.commit()

    def add_many(self, chunks: Iterable[Chunk]) -> int:
        count = 0
        for chunk in chunks:
            self._add(chunk)
            count += 1
        self._conn.commit()
        return count

    def remove(self, chunk_id: str) -> None:
        cur = self._conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
        if cur.rowcount:
            self._bump_revision()
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM chunks")
        self._bump_revision()
        self._conn.commit()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"])

    def get(self, chunk_id: str) -> Optional[Chunk]:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        return self._row_to_chunk(row) if row else None

    def all_chunks(self) -> List[Chunk]:
        rows = self._conn.execute("SELECT * FROM chunks ORDER BY rowid").fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def chunks_by_kind(self, kind: SourceKind) -> List[Chunk]:
        if isinstance(kind, str):
            kind = SourceKind(kind)
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE kind = ? ORDER BY rowid", (kind.value,)
        ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 20) -> List[Chunk]:
        """FTS5 search with LIKE fallback for empty/special queries."""
        fts_query = self._build_fts_query(query)
        if fts_query:
            try:
                rows = self._conn.execute(
                    """
                    SELECT c.*, f.rank AS fts_rank
                    FROM chunks_fts f
                    JOIN chunks c ON c.rowid = f.rowid
                    WHERE chunks_fts MATCH ?
                    ORDER BY f.rank
                    LIMIT ?
                    """,
                    (fts_query, top_k),
                ).fetchall()
                if rows:
                    return [self._row_to_chunk(row) for row in rows]
            except sqlite3.OperationalError:
                # FTS syntax issue -> fall through to LIKE.
                pass

        # Deterministic fallback: substring match.
        pattern = f"%{query}%"
        rows = self._conn.execute(
            """
            SELECT * FROM chunks
            WHERE content LIKE ? ESCAPE '\\'
            ORDER BY rowid
            LIMIT ?
            """,
            (pattern, top_k),
        ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    # ------------------------------------------------------------------
    # Persistence (SQLite is already persistent)
    # ------------------------------------------------------------------
    def save(self, path: Optional[str | Path] = None) -> None:
        if path is not None and str(path) != self.db_path:
            raise ValueError("SqliteIndex is bound to its db_path; use a new instance to copy.")
        self._conn.commit()

    @classmethod
    def load(cls, path: str | Path) -> "SqliteIndex":
        return cls(path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _add(self, chunk: Chunk) -> None:
        metadata_json = json.dumps(chunk.metadata, ensure_ascii=False)
        self._conn.execute(
            """
            INSERT INTO chunks (id, source, kind, content, metadata, timestamp, priority, tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source = excluded.source,
                kind = excluded.kind,
                content = excluded.content,
                metadata = excluded.metadata,
                timestamp = excluded.timestamp,
                priority = excluded.priority,
                tokens = excluded.tokens
            """,
            (
                chunk.id,
                chunk.source,
                chunk.kind.value if isinstance(chunk.kind, SourceKind) else str(chunk.kind),
                chunk.content,
                metadata_json,
                chunk.timestamp,
                chunk.priority,
                chunk.tokens,
            ),
        )
        self._bump_revision()

    def _bump_revision(self) -> None:
        self._revision += 1
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES('revision', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(self._revision),),
        )

    def _load_revision(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'revision'"
        ).fetchone()
        return int(row["value"]) if row else 0

    @staticmethod
    def _build_fts_query(query: str) -> Optional[str]:
        tokens = tokenize(query)
        if not tokens:
            return None
        # OR keeps recall high; ranking still prefers documents with more terms.
        return " OR ".join(f'"{t}"' for t in tokens)

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            source=row["source"],
            kind=SourceKind(row["kind"]),
            content=row["content"],
            metadata=json.loads(row["metadata"] or "{}"),
            timestamp=row["timestamp"],
            priority=row["priority"] or 0.0,
            tokens=row["tokens"] or 0,
        )
