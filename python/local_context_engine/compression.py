"""Compression strategies for fitting more context into a finite window.

The prototype uses deterministic local compression:
- full inclusion for high-priority chunks
- extractive sentence compression for chunks that only partially fit
- a compact "omitted memory" note so the model knows what was left out
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from .models import RetrievedItem, estimate_tokens
from .text_utils import split_sentences, term_frequencies, tokenize


@dataclass
class PackedItem:
    item: RetrievedItem
    content: str
    tokens: int
    compressed: bool = False


@dataclass
class CompressionResult:
    packed: List[PackedItem]
    omitted_count: int = 0
    omitted_tokens: int = 0
    omitted_ids: List[str] = None

    def __post_init__(self) -> None:
        if self.omitted_ids is None:
            self.omitted_ids = []


def compress_items(
    items: Iterable[RetrievedItem],
    budget_tokens: int,
    query: str = "",
    min_chunk_tokens: int = 24,
) -> CompressionResult:
    """Greedily pack items into a token budget, compressing the overflow."""
    ordered = sorted(items, key=lambda item: item.score, reverse=True)
    packed: List[PackedItem] = []
    omitted_ids: List[str] = []
    omitted_tokens = 0
    used = 0
    for item in ordered:
        content = item.chunk.content
        tokens = item.chunk.tokens or estimate_tokens(content)
        if used + tokens <= budget_tokens:
            packed.append(PackedItem(item=item, content=content, tokens=tokens, compressed=False))
            used += tokens
            continue

        remaining = budget_tokens - used
        if remaining >= min_chunk_tokens:
            compressed, compressed_tokens = compress_chunk_to_fit(
                content, remaining, query=query
            )
            if compressed_tokens > 0:
                packed.append(
                    PackedItem(item=item, content=compressed, tokens=compressed_tokens, compressed=True)
                )
                used += compressed_tokens
                continue

        omitted_ids.append(item.chunk.id)
        omitted_tokens += tokens

    return CompressionResult(
        packed=packed,
        omitted_count=len(omitted_ids),
        omitted_tokens=omitted_tokens,
        omitted_ids=omitted_ids,
    )


def compress_chunk_to_fit(content: str, max_tokens: int, query: str = "") -> Tuple[str, int]:
    """Compress a single chunk to fit max_tokens.

    First tries extractive sentence ranking; if that is still too big, falls
    back to hard truncation at the token boundary.
    """
    if max_tokens <= 0:
        return "", 0
    if estimate_tokens(content) <= max_tokens:
        return content, estimate_tokens(content)

    extractive = _extractive_compress(content, max_tokens, query)
    if extractive:
        return extractive, estimate_tokens(extractive)

    return truncate_text(content, max_tokens)


def truncate_text(content: str, max_tokens: int) -> Tuple[str, int]:
    """Truncate text to an approximate token budget while keeping whole lines."""
    if max_tokens <= 0:
        return "", 0
    if estimate_tokens(content) <= max_tokens:
        return content, estimate_tokens(content)
    parts = content.splitlines(keepends=True)
    out: List[str] = []
    used = 0
    for part in parts:
        part_tokens = estimate_tokens(part)
        if used + part_tokens > max_tokens:
            # If a single long line still exceeds the remaining budget, cut it.
            remaining = max_tokens - used
            if remaining > 8 and part:
                cut = _cut_by_tokens(part, remaining)
                out.append(cut)
                used += estimate_tokens(cut)
            break
        out.append(part)
        used += part_tokens
    text = "".join(out).rstrip()
    if not text:
        # last resort: hard character cut
        text = _cut_by_tokens(content, max_tokens)
    return text, estimate_tokens(text)


def _extractive_compress(content: str, max_tokens: int, query: str = "") -> str:
    """Pick the most informative sentences under a token budget."""
    sentences = split_sentences(content)
    if len(sentences) <= 1:
        return ""
    query_freqs = term_frequencies(tokenize(query)) if query else {}
    scored = []
    for idx, sentence in enumerate(sentences):
        freqs = term_frequencies(tokenize(sentence))
        overlap = sum(freqs.get(t, 0) * qf for t, qf in query_freqs.items())
        length_bonus = min(1.0, len(sentence) / 200.0)
        position_bonus = 1.0 if idx == 0 else 0.6
        score = overlap + length_bonus * 0.5 + position_bonus
        scored.append((score, idx, sentence))
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen: List[Tuple[int, str]] = []
    used = 0
    for _, idx, sentence in scored:
        sentence_tokens = estimate_tokens(sentence)
        if used + sentence_tokens > max_tokens:
            continue
        chosen.append((idx, sentence))
        used += sentence_tokens
    if not chosen:
        return ""
    chosen.sort(key=lambda x: x[0])
    return "\n".join(s for _, s in chosen)


def build_omitted_note(omitted_count: int, omitted_tokens: int, max_tokens: int = 80) -> str:
    """A small note that tells the model older/less relevant context was omitted."""
    if omitted_count <= 0:
        return ""
    note = (
        f"[Local Context Engine] {omitted_count} lower-priority item(s) "
        f"(~{omitted_tokens} tokens) were omitted to stay within the context budget. "
        "Ask the user or use retrieval if any omitted detail is needed."
    )
    if estimate_tokens(note) > max_tokens:
        note = (
            f"[Local Context Engine] {omitted_count} lower-priority item(s) "
            f"(~{omitted_tokens} tokens) omitted for budget."
        )
    return note


def _cut_by_tokens(text: str, max_tokens: int) -> str:
    """Cut text by approximate tokens, preserving CJK characters."""
    if max_tokens <= 0:
        return ""
    out = []
    used = 0
    for ch in text:
        cost = 1 if "\u4e00" <= ch <= "\u9fff" else 0.25
        if used + cost > max_tokens:
            break
        out.append(ch)
        used += cost
    return "".join(out).rstrip()
