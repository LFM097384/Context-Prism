"""Lightweight text utilities used by retrieval and compression."""

from __future__ import annotations

import math
import re
from typing import Iterable, List

# Matches ASCII words/numbers plus individual CJK characters.
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    """Split text into simple tokens for BM25 / frequency scoring."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences on common punctuation and newlines."""
    parts = re.split(r"(?<=[。！？!?.])\s*|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def term_frequencies(tokens: Iterable[str]) -> dict:
    """Return a dict of token -> frequency."""
    freqs: dict = {}
    for token in tokens:
        freqs[token] = freqs.get(token, 0) + 1
    return freqs


def cosine_similarity(a: dict, b: dict) -> float:
    """Cosine similarity between two sparse term-frequency dicts."""
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
