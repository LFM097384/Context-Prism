"""Local in-memory / JSON-persistent RAG index.

This is intentionally dependency-free: it uses BM25 (a standard lexical RAG
retrieval) plus pluggable metadata boosts. A real deployment can replace this
with SQLite + FTS5, LanceDB, or any embedding store while keeping the same
interface.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import Chunk, SourceKind
from .text_utils import term_frequencies, tokenize


class LocalIndex:
    """A small BM25 index over Chunk objects."""

    def __init__(self) -> None:
        self._chunks: Dict[str, Chunk] = {}
        self._postings: Dict[str, Dict[str, int]] = {}
        self._doc_lengths: Dict[str, int] = {}
        self._avgdl: float = 0.0
        self._k1 = 1.5
        self._b = 0.75
        self._revision = 0

    @property
    def revision(self) -> int:
        """Monotonic version counter, incremented on any mutation."""
        return self._revision

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------
    def add(self, chunk: Chunk) -> None:
        if chunk.id in self._chunks:
            self.remove(chunk.id)
        self._chunks[chunk.id] = chunk
        tokens = tokenize(chunk.content)
        freqs = term_frequencies(tokens)
        self._doc_lengths[chunk.id] = sum(freqs.values())
        for token, count in freqs.items():
            self._postings.setdefault(token, {})[chunk.id] = count
        self._recompute_avgdl()
        self._revision += 1

    def add_many(self, chunks: Iterable[Chunk]) -> int:
        count = 0
        for chunk in chunks:
            self.add(chunk)
            count += 1
        return count

    def remove(self, chunk_id: str) -> None:
        chunk = self._chunks.pop(chunk_id, None)
        if chunk is None:
            return
        self._doc_lengths.pop(chunk_id, None)
        for postings in self._postings.values():
            postings.pop(chunk_id, None)
        self._postings = {t: p for t, p in self._postings.items() if p}
        self._recompute_avgdl()
        self._revision += 1

    def clear(self) -> None:
        self._chunks.clear()
        self._postings.clear()
        self._doc_lengths.clear()
        self._avgdl = 0.0
        self._revision += 1

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._chunks)

    def get(self, chunk_id: str) -> Optional[Chunk]:
        return self._chunks.get(chunk_id)

    def all_chunks(self) -> List[Chunk]:
        return list(self._chunks.values())

    def chunks_by_kind(self, kind: SourceKind) -> List[Chunk]:
        return [c for c in self._chunks.values() if c.kind == kind]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 20) -> List[Chunk]:
        """Return the top-k BM25 hits for a query."""
        if not self._chunks:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores: Dict[str, float] = {}
        n = len(self._chunks)
        for token in set(query_tokens):
            postings = self._postings.get(token, {})
            df = len(postings)
            if df == 0:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for doc_id, freq in postings.items():
                doc_len = self._doc_lengths.get(doc_id, 0)
                denom = freq + self._k1 * (
                    1.0 - self._b + self._b * doc_len / max(self._avgdl, 1.0)
                )
                score = idf * (freq * (self._k1 + 1.0)) / max(denom, 1e-9)
                scores[doc_id] = scores.get(doc_id, 0.0) + score
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [self._chunks[doc_id] for doc_id, _ in ranked[:top_k]]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "saved_at": time.time(),
            "chunks": [
                {
                    "id": c.id,
                    "source": c.source,
                    "kind": c.kind.value,
                    "content": c.content,
                    "metadata": c.metadata,
                    "timestamp": c.timestamp,
                    "priority": c.priority,
                    "tokens": c.tokens,
                }
                for c in self._chunks.values()
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LocalIndex":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        index = cls()
        for item in data.get("chunks", []):
            chunk = Chunk(
                id=item["id"],
                source=item["source"],
                kind=SourceKind(item["kind"]),
                content=item["content"],
                metadata=item.get("metadata", {}),
                timestamp=item.get("timestamp"),
                priority=item.get("priority", 0.0),
                tokens=item.get("tokens", 0),
            )
            index.add(chunk)
        return index

    def _recompute_avgdl(self) -> None:
        if not self._doc_lengths:
            self._avgdl = 0.0
            return
        self._avgdl = sum(self._doc_lengths.values()) / len(self._doc_lengths)
