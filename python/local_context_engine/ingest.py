"""Ingest helpers for turning raw local data into Chunks."""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from xml.etree import ElementTree as ET

from .models import Chunk, SourceKind


def chunk_text(
    content: str,
    source: str,
    kind: SourceKind,
    chunk_size: int = 800,
    overlap: int = 80,
    metadata: Optional[Dict] = None,
    timestamp: Optional[float] = None,
    priority: float = 0.0,
) -> List[Chunk]:
    """Split long text into overlapping chunks.

    A chunk is a unit that can be retrieved independently. Overlap keeps
    context continuity across boundaries.
    """
    if not content.strip():
        return []
    if isinstance(kind, str):
        kind = SourceKind(kind)
    chunks: List[Chunk] = []
    text = content.strip()
    step = max(1, chunk_size - overlap)
    start = 0
    idx = 0
    while start < len(text):
        piece = text[start : start + chunk_size]
        chunks.append(
            Chunk(
                id=f"{source}:{idx}:{start}",
                source=source,
                kind=kind,
                content=piece,
                metadata=dict(metadata or {}),
                timestamp=timestamp,
                priority=priority,
            )
        )
        idx += 1
        if len(piece) < chunk_size:
            break
        start += step
    return chunks


def ingest_text(
    content: str,
    source: str,
    kind: SourceKind,
    chunk_size: int = 800,
    overlap: int = 80,
    metadata: Optional[Dict] = None,
    timestamp: Optional[float] = None,
    priority: float = 0.0,
) -> List[Chunk]:
    """Alias for chunk_text."""
    return chunk_text(
        content,
        source,
        kind,
        chunk_size=chunk_size,
        overlap=overlap,
        metadata=metadata,
        timestamp=timestamp,
        priority=priority,
    )


def ingest_file(path: str | Path, kind: Optional[SourceKind] = None) -> List[Chunk]:
    """Ingest a text file, choosing SourceKind by extension when not given."""
    path = Path(path)
    text = _read_text_file(path)
    if kind is None:
        kind = _kind_for_path(path)
    return chunk_text(
        text,
        source=str(path),
        kind=kind,
        metadata={"path": str(path), "extension": path.suffix},
        timestamp=path.stat().st_mtime,
    )


def ingest_directory(
    directory: str | Path,
    kind: Optional[SourceKind] = None,
    extensions: Optional[Iterable[str]] = None,
    ignore: Optional[Iterable[str]] = None,
) -> List[Chunk]:
    """Recursively ingest text files from a directory."""
    directory = Path(directory)
    allowed = set(extensions or [".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".txt", ".json", ".jsonl", ".csv", ".html", ".css"])
    ignored = set(ignore or [])
    chunks: List[Chunk] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(directory).as_posix()
        if any(part in ignored for part in path.parts):
            continue
        if allowed and path.suffix.lower() not in allowed:
            continue
        try:
            chunks.extend(ingest_file(path, kind=kind))
        except Exception as exc:  # keep going on unreadable files
            print(f"[ingest] skipped {path}: {exc}")
    return chunks


def ingest_history_jsonl(path: str | Path, source_name: Optional[str] = None) -> List[Chunk]:
    """Ingest a JSONL conversation log.

    Each line may look like:
      {"role": "user", "content": "...", "timestamp": 1234567890, "session": "abc"}
    """
    path = Path(path)
    source = source_name or str(path)
    chunks: List[Chunk] = []
    with path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = record.get("content") or record.get("text") or record.get("message")
            if not content:
                continue
            timestamp = record.get("timestamp")
            if isinstance(timestamp, str):
                try:
                    timestamp = float(timestamp)
                except ValueError:
                    timestamp = None
            chunks.append(
                Chunk(
                    id=f"{source}:history:{idx}",
                    source=source,
                    kind=SourceKind.HISTORY,
                    content=f"{record.get('role', 'unknown')}: {content}",
                    metadata=record,
                    timestamp=timestamp,
                    priority=1.0 if record.get("important") else 0.0,
                )
            )
    return chunks


def ingest_trajectory_jsonl(path: str | Path, source_name: Optional[str] = None) -> List[Chunk]:
    """Ingest an agent trajectory / tool-call log.

    Each line may look like:
      {"step": 1, "tool": "read_file", "input": "...", "output": "...", "timestamp": 1234567890}
    """
    path = Path(path)
    source = source_name or str(path)
    chunks: List[Chunk] = []
    with path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            tool = record.get("tool") or record.get("action") or "tool"
            content_parts = []
            if record.get("input"):
                content_parts.append(f"input: {record['input']}")
            if record.get("output"):
                content_parts.append(f"output: {record['output']}")
            if record.get("thought") or record.get("reasoning"):
                content_parts.append(f"reasoning: {record.get('thought') or record.get('reasoning')}")
            if not content_parts:
                content_parts.append(json.dumps(record, ensure_ascii=False))
            timestamp = record.get("timestamp")
            if isinstance(timestamp, str):
                try:
                    timestamp = float(timestamp)
                except ValueError:
                    timestamp = None
            chunks.append(
                Chunk(
                    id=f"{source}:trajectory:{idx}",
                    source=source,
                    kind=SourceKind.TRAJECTORY,
                    content=f"[{tool}] " + " | ".join(content_parts),
                    metadata=record,
                    timestamp=timestamp,
                    priority=2.0 if record.get("important") else 0.0,
                )
            )
    return chunks


def _read_text_file(path: Path) -> str:
    """Read a text file; supports .docx via the standard library."""
    if path.suffix.lower() == ".docx":
        return _read_docx(path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Unsupported binary or non-UTF-8 file: {path}") from exc


def _read_docx(path: Path) -> str:
    """Extract plain text from a .docx file using only the standard library."""
    word_ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    paragraphs = []
    for para in root.iter(f"{word_ns}p"):
        text = "".join(
            node.text or ""
            for node in para.iter(f"{word_ns}t")
        )
        if text.strip():
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _kind_for_path(path: Path) -> SourceKind:
    suffix = path.suffix.lower()
    if suffix in {".py", ".ts", ".tsx", ".js", ".jsx", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".sql"}:
        return SourceKind.CODE
    if suffix in {".md", ".txt", ".rst", ".doc", ".docx", ".pdf"}:
        return SourceKind.FILE
    if suffix in {".json", ".jsonl"}:
        # JSONL is commonly used for history/trajectory; caller can override.
        return SourceKind.FILE
    return SourceKind.FILE
