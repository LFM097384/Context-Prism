"""Prioritization heuristics for deciding what must stay in the window."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .models import Chunk, RetrievedItem, SourceKind


@dataclass
class PriorityConfig:
    """Weights applied after retrieval to re-order candidates."""

    kind_base: Dict[SourceKind, float] = field(default_factory=lambda: {
        SourceKind.PREFERENCE: 5.0,
        SourceKind.TRAJECTORY: 3.0,
        SourceKind.HISTORY: 2.5,
        SourceKind.CODE: 2.0,
        SourceKind.FILE: 1.5,
    })
    recency_weight: float = 0.2
    priority_weight: float = 1.0
    length_penalty_weight: float = 0.1
    max_length_tokens: int = 800


def prioritize(
    items: Iterable[RetrievedItem],
    config: Optional[PriorityConfig] = None,
) -> List[RetrievedItem]:
    """Combine retrieval score with source/recency/importance heuristics."""
    config = config or PriorityConfig()
    ranked: List[RetrievedItem] = []
    for item in items:
        chunk = item.chunk
        kind_base = config.kind_base.get(chunk.kind, 1.0)
        recency = _recency_score(chunk)
        length_penalty = min(1.0, _approx_tokens(chunk.content) / config.max_length_tokens)
        priority = chunk.priority / 10.0
        score = (
            item.score
            + kind_base * 0.1
            + recency * config.recency_weight
            + priority * config.priority_weight
            - length_penalty * config.length_penalty_weight
        )
        ranked.append(RetrievedItem(chunk=chunk, score=score, reason=item.reason))
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked


def _recency_score(chunk: Chunk, half_life_days: float = 7.0) -> float:
    import math
    import time

    now = time.time()
    age_days = max(0.0, (now - (chunk.timestamp or now)) / 86400.0)
    return math.exp(-age_days / half_life_days)


def _approx_tokens(text: str) -> int:
    from .models import estimate_tokens

    return estimate_tokens(text)
