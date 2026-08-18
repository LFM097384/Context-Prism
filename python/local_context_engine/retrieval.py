"""Hybrid retrieval: BM25 relevance + recency + source priority + user priority.

The final score is a weighted combination so the engine can favor not just
lexical relevance but also what matters for the current agent/user.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .index import LocalIndex
from .models import Chunk, RetrievedItem, SourceKind


@dataclass
class RetrievalConfig:
    """Tunable weights for the retrieval stage."""

    relevance_weight: float = 1.0
    recency_weight: float = 0.25
    source_weight: float = 0.2
    priority_weight: float = 0.3
    recency_half_life_days: float = 7.0
    top_k: int = 40
    source_weights: Dict[SourceKind, float] = field(default_factory=lambda: {
        SourceKind.PREFERENCE: 1.5,
        SourceKind.TRAJECTORY: 1.1,
        SourceKind.HISTORY: 1.0,
        SourceKind.CODE: 1.0,
        SourceKind.FILE: 0.9,
    })

    def __post_init__(self) -> None:
        # Accept string keys too.
        normalized: Dict[SourceKind, float] = {}
        for key, value in self.source_weights.items():
            if isinstance(key, str):
                key = SourceKind(key)
            normalized[key] = value
        self.source_weights = normalized


def retrieve(
    index: LocalIndex,
    query: str,
    config: Optional[RetrievalConfig] = None,
    now: Optional[float] = None,
) -> List[RetrievedItem]:
    """Retrieve chunks from the index and rank them with the hybrid score."""
    config = config or RetrievalConfig()
    now = now if now is not None else time.time()
    chunks = index.search(query, top_k=config.top_k)
    items: List[RetrievedItem] = []
    for chunk in chunks:
        score = config.relevance_weight * _bm25_rank_score(chunk, chunks)
        age_days = max(0.0, (now - (chunk.timestamp or now)) / 86400.0)
        recency = math.exp(-age_days / config.recency_half_life_days)
        source_boost = config.source_weights.get(chunk.kind, 1.0)
        score += config.recency_weight * recency
        score += config.source_weight * (source_boost - 1.0)
        score += config.priority_weight * (chunk.priority / 10.0)
        reason = f"bm25={_bm25_rank_score(chunk, chunks):.3f}"
        if recency > 0.8:
            reason += ", recent"
        if chunk.priority > 0:
            reason += f", priority={chunk.priority}"
        items.append(RetrievedItem(chunk=chunk, score=score, reason=reason))
    items.sort(key=lambda item: item.score, reverse=True)
    return items


def _bm25_rank_score(chunk: Chunk, ranked_chunks: List[Chunk]) -> float:
    """Normalize a chunk's position into a 0..1 relevance score.

    The index already returns BM25 scores; here we use the rank as a simple
    monotonic normalization so different corpora are comparable.
    """
    try:
        rank = ranked_chunks.index(chunk)
    except ValueError:
        return 0.0
    if not ranked_chunks:
        return 0.0
    return max(0.0, 1.0 - rank / max(len(ranked_chunks), 1))
