"""The Local Context Engine orchestrator.

This is the main entry point used before calling any LLM API:

    engine = LocalContextEngine()
    engine.add_file("notes.md")
    engine.add_history_jsonl("history.jsonl")
    window = engine.build_context(query="...", max_tokens=4000)
    messages = window.as_messages()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .cache import TTLCache
from .compression import CompressionResult, build_omitted_note, compress_items
from .index import LocalIndex
from .ingest import (
    chunk_text,
    ingest_directory,
    ingest_file,
    ingest_history_jsonl,
    ingest_text,
    ingest_trajectory_jsonl,
)
from .models import (
    Chunk,
    ContextSection,
    ContextWindow,
    RetrievedItem,
    SourceKind,
    estimate_tokens,
)
from .prioritization import PriorityConfig, prioritize
from .retrieval import RetrievalConfig, retrieve
from .sqlite_index import SqliteIndex


class LocalContextEngine:
    """Provider-agnostic dynamic context window builder."""

    def __init__(
        self,
        index: Optional[LocalIndex | SqliteIndex] = None,
        storage_path: Optional[str | Path] = None,
        backend: str = "auto",
        cache_size: int = 128,
        cache_ttl: float = 60.0,
    ) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        self.retrieval_cache = TTLCache(maxsize=cache_size, ttl_seconds=cache_ttl)
        self.compression_cache = TTLCache(maxsize=cache_size, ttl_seconds=cache_ttl)
        if index is not None:
            self.index = index
        elif self.storage_path is not None:
            self.index = self._open_index(self.storage_path, backend)
        elif backend == "sqlite":
            self.index = SqliteIndex(":memory:")
        else:
            self.index = LocalIndex()

    @staticmethod
    def _open_index(path: Path, backend: str) -> LocalIndex | SqliteIndex:
        if backend == "sqlite":
            return SqliteIndex(path)
        if backend == "json":
            if path.exists():
                return LocalIndex.load(path)
            return LocalIndex()
        # auto
        if path.exists():
            if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"} or _looks_like_sqlite(path):
                return SqliteIndex(path)
            return LocalIndex.load(path)
        if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            return SqliteIndex(path)
        return LocalIndex()

    # ------------------------------------------------------------------
    # Ingest API
    # ------------------------------------------------------------------
    def add_chunk(self, chunk: Chunk) -> None:
        self.index.add(chunk)

    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        return self.index.add_many(chunks)

    def add_text(
        self,
        content: str,
        source: str,
        kind: SourceKind,
        chunk_size: int = 800,
        overlap: int = 80,
        metadata: Optional[Dict] = None,
        timestamp: Optional[float] = None,
        priority: float = 0.0,
    ) -> int:
        chunks = chunk_text(
            content,
            source,
            kind,
            chunk_size=chunk_size,
            overlap=overlap,
            metadata=metadata,
            timestamp=timestamp,
            priority=priority,
        )
        return self.add_chunks(chunks)

    def add_file(self, path: str | Path, kind: Optional[SourceKind] = None) -> int:
        return self.add_chunks(ingest_file(path, kind=kind))

    def add_directory(
        self,
        directory: str | Path,
        kind: Optional[SourceKind] = None,
        extensions: Optional[Iterable[str]] = None,
        ignore: Optional[Iterable[str]] = None,
    ) -> int:
        return self.add_chunks(ingest_directory(directory, kind=kind, extensions=extensions, ignore=ignore))

    def add_history_jsonl(self, path: str | Path, source_name: Optional[str] = None) -> int:
        return self.add_chunks(ingest_history_jsonl(path, source_name=source_name))

    def add_trajectory_jsonl(self, path: str | Path, source_name: Optional[str] = None) -> int:
        return self.add_chunks(ingest_trajectory_jsonl(path, source_name=source_name))

    def add_file_incremental(
        self,
        path: str | Path,
        kind: Optional[SourceKind] = None,
        state_path: Optional[str | Path] = None,
    ) -> Tuple[int, int, int]:
        """Ingest a file only when its mtime/size changed.

        Returns ``(added_count, changed_count, skipped_count)``.
        """
        path = Path(path)
        state_path = Path(state_path) if state_path else self._default_state_path()
        state = self._load_state(state_path)
        stat = path.stat()
        key = str(path.resolve())
        previous = state.get(key)
        if previous is not None and previous.get("mtime") == stat.st_mtime and previous.get("size") == stat.st_size:
            return 0, 0, 1

        count = self.add_file(path, kind=kind)
        state[key] = {"mtime": stat.st_mtime, "size": stat.st_size}
        self._save_state(state_path, state)
        return count, 1 if count else 0, 0

    def add_directory_incremental(
        self,
        directory: str | Path,
        kind: Optional[SourceKind] = None,
        extensions: Optional[Iterable[str]] = None,
        ignore: Optional[Iterable[str]] = None,
        state_path: Optional[str | Path] = None,
    ) -> Tuple[int, int, int]:
        """Ingest only new/changed files in a directory.

        Returns ``(added_count, changed_count, skipped_count)``.
        """
        directory = Path(directory)
        state_path = Path(state_path) if state_path else self._default_state_path()
        state = self._load_state(state_path)
        allowed = set(extensions or [".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".txt", ".json", ".jsonl", ".csv", ".html", ".css"])
        ignored = set(ignore or [])
        current_keys = set()
        total_added = 0
        total_changed = 0
        total_skipped = 0

        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if any(part in ignored for part in path.parts):
                continue
            if allowed and path.suffix.lower() not in allowed:
                continue
            key = str(path.resolve())
            current_keys.add(key)
            stat = path.stat()
            previous = state.get(key)
            if previous is not None and previous.get("mtime") == stat.st_mtime and previous.get("size") == stat.st_size:
                total_skipped += 1
                continue
            try:
                count = self.add_file(path, kind=kind)
            except Exception as exc:
                print(f"[ingest] skipped {path}: {exc}")
                continue
            state[key] = {"mtime": stat.st_mtime, "size": stat.st_size}
            total_added += count
            total_changed += 1 if count else 0

        # Drop state entries for files that no longer exist (optional cleanup).
        for key in list(state.keys()):
            if key not in current_keys and Path(key).exists() is False:
                del state[key]

        self._save_state(state_path, state)
        return total_added, total_changed, total_skipped

    def clear_caches(self) -> None:
        self.retrieval_cache.clear()
        self.compression_cache.clear()

    def _default_state_path(self) -> Path:
        if self.storage_path is not None:
            return self.storage_path.with_name(self.storage_path.name + ".state.json")
        return Path(".lce_incremental_state.json")

    @staticmethod
    def _load_state(path: Path) -> Dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def _save_state(path: Path, state: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Optional[str | Path] = None) -> None:
        target = Path(path) if path else self.storage_path
        if target is None:
            raise ValueError("No storage path configured; pass one to save().")
        self.index.save(target)

    def load(self, path: Optional[str | Path] = None, backend: str = "auto") -> None:
        target = Path(path) if path else self.storage_path
        if target is None:
            raise ValueError("No storage path configured; pass one to load().")
        self.index = self._open_index(target, backend)
        self.storage_path = target

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------
    def build_context(
        self,
        query: str,
        max_tokens: int = 8000,
        reserved_output_tokens: int = 1024,
        retrieval_config: Optional[RetrievalConfig] = None,
        priority_config: Optional[PriorityConfig] = None,
        include_preferences: bool = True,
    ) -> ContextWindow:
        """Assemble a dynamic context window for a query.

        The engine always reserves room for the model's output, then fills the
        remaining budget with preferences first, followed by retrieved history,
        trajectory, code and files. Lower-priority items are compressed or
        omitted with a visible note.
        """
        budget = max(0, max_tokens - reserved_output_tokens)
        if budget <= 0:
            raise ValueError("max_tokens must be greater than reserved_output_tokens")

        # 1. Retrieve and prioritize (with retrieval cache).
        retrieval_key = self._retrieval_cache_key(query, retrieval_config)
        retrieved = self.retrieval_cache.get(retrieval_key)
        if retrieved is None:
            retrieved = retrieve(self.index, query, retrieval_config)
            self.retrieval_cache.set(retrieval_key, retrieved)
        prioritized = prioritize(retrieved, priority_config)

        # 2. Always-available user preferences section.
        pref_section = None
        omitted_count = 0
        omitted_tokens = 0
        if include_preferences:
            pref_chunks = self.index.chunks_by_kind(SourceKind.PREFERENCE)
            if pref_chunks:
                pref_items = [
                    RetrievedItem(chunk=c, score=10.0 - i * 0.001, reason="always-include preference")
                    for i, c in enumerate(pref_chunks)
                ]
                pref_key = self._compression_cache_key(query, budget, pref_items)
                pref_result = self.compression_cache.get(pref_key)
                if pref_result is None:
                    pref_result = compress_items(pref_items, budget, query=query)
                    self.compression_cache.set(pref_key, pref_result)
                pref_section = self._section_from_result(
                    "User Preferences", pref_result, kind=SourceKind.PREFERENCE
                )
                omitted_count += pref_result.omitted_count
                omitted_tokens += pref_result.omitted_tokens

        used = pref_section.tokens if pref_section else 0
        remaining = max(0, budget - used)

        # 3. Compress the remaining retrieved context into the leftover budget.
        non_pref_items = [item for item in prioritized if item.chunk.kind != SourceKind.PREFERENCE]
        non_pref_key = self._compression_cache_key(query, remaining, non_pref_items)
        non_pref_result = self.compression_cache.get(non_pref_key)
        if non_pref_result is None:
            non_pref_result = compress_items(non_pref_items, remaining, query=query)
            self.compression_cache.set(non_pref_key, non_pref_result)
        omitted_count += non_pref_result.omitted_count
        omitted_tokens += non_pref_result.omitted_tokens

        sections: List[ContextSection] = []
        if pref_section:
            sections.append(pref_section)

        # 4. Group packed items into named sections.
        sections.extend(self._group_into_sections(non_pref_result))

        # 5. Add an omitted-memory note if there is room.
        note = build_omitted_note(omitted_count, omitted_tokens)
        if note:
            note_tokens = estimate_tokens(note)
            used_after_sections = sum(s.tokens for s in sections)
            if used_after_sections + note_tokens <= budget:
                sections.append(
                    ContextSection(
                        title="Omitted Memory",
                        content=note,
                        tokens=note_tokens,
                        compressed=True,
                    )
                )

        total_tokens = sum(s.tokens for s in sections)
        return ContextWindow(
            query=query,
            sections=sections,
            max_tokens=max_tokens,
            reserved_output_tokens=reserved_output_tokens,
            token_budget=budget,
            total_tokens=total_tokens,
            omitted_count=omitted_count,
            omitted_tokens=omitted_tokens,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _retrieval_cache_key(self, query: str, config: Optional[RetrievalConfig]) -> tuple:
        return (
            "retrieve",
            self.index.revision,
            query.strip().lower(),
            self._retrieval_config_fingerprint(config),
        )

    @staticmethod
    def _retrieval_config_fingerprint(config: Optional[RetrievalConfig]) -> Optional[tuple]:
        if config is None:
            return None
        return (
            config.relevance_weight,
            config.recency_weight,
            config.source_weight,
            config.priority_weight,
            config.recency_half_life_days,
            config.top_k,
            tuple(sorted((k.value, v) for k, v in config.source_weights.items())),
        )

    def _compression_cache_key(self, query: str, budget: int, items: List[RetrievedItem]) -> tuple:
        # Include index revision so a changed index cannot return stale results.
        item_sig = tuple((item.chunk.id, round(item.score, 6)) for item in items)
        return ("compress", self.index.revision, query.strip().lower(), budget, item_sig)

    @staticmethod
    def _section_from_result(
        title: str,
        result: CompressionResult,
        kind: SourceKind,
    ) -> ContextSection:
        packed = result.packed
        content = LocalContextEngine._render_packed(packed)
        tokens = sum(p.tokens for p in packed)
        return ContextSection(
            title=title,
            content=content,
            tokens=tokens,
            items=[p.item for p in packed],
            compressed=any(p.compressed for p in packed),
        )

    @staticmethod
    def _group_into_sections(result: CompressionResult) -> List[ContextSection]:
        """Group packed items by SourceKind into stable, human-readable sections."""
        titles = {
            SourceKind.HISTORY: "Relevant History",
            SourceKind.TRAJECTORY: "Agent Trajectory",
            SourceKind.CODE: "Relevant Code",
            SourceKind.FILE: "Relevant Files",
        }
        grouped: Dict[SourceKind, List] = {kind: [] for kind in titles}
        for packed in result.packed:
            kind = packed.item.chunk.kind
            if kind in grouped:
                grouped[kind].append(packed)

        sections: List[ContextSection] = []
        for kind in titles:
            packed = grouped[kind]
            if not packed:
                continue
            content = LocalContextEngine._render_packed(packed)
            tokens = sum(p.tokens for p in packed)
            sections.append(
                ContextSection(
                    title=titles[kind],
                    content=content,
                    tokens=tokens,
                    items=[p.item for p in packed],
                    compressed=any(p.compressed for p in packed),
                )
            )
        return sections

    @staticmethod
    def _render_packed(packed) -> str:
        parts: List[str] = []
        for p in packed:
            chunk = p.item.chunk
            source_label = chunk.source
            if chunk.metadata.get("path"):
                source_label = chunk.metadata["path"]
            parts.append(f"[{chunk.kind.value} | {source_label}]\n{p.content}")
        return "\n\n".join(parts)


def _looks_like_sqlite(path: Path) -> bool:
    """Detect a SQLite database by its 16-byte file header."""
    try:
        with path.open("rb") as fh:
            return fh.read(16) == b"SQLite format 3\x00"
    except OSError:
        return False
