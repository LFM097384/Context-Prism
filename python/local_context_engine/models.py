"""Core data structures for the Local Context Engine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SourceKind(str, Enum):
    """The broad categories of context that can be indexed."""

    HISTORY = "history"          # chat history, transcripts, conversation logs
    CODE = "code"                # source code files / snippets
    FILE = "file"                # documents, notes, markdown, plain text
    PREFERENCE = "preference"    # user preferences, style rules, instructions
    TRAJECTORY = "trajectory"    # agent actions / tool calls / trajectories


@dataclass
class Chunk:
    """A single retrievable unit of local context."""

    id: str
    source: str
    kind: SourceKind
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    priority: float = 0.0  # user/domain-level importance, 0..10
    tokens: int = 0  # precomputed token estimate; 0 means compute lazily

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = time.time()
        if isinstance(self.kind, str):
            self.kind = SourceKind(self.kind)
        if self.tokens <= 0:
            self.tokens = estimate_tokens(self.content)


@dataclass
class RetrievedItem:
    """A chunk with its retrieval / priority score."""

    chunk: Chunk
    score: float = 0.0
    reason: str = ""


@dataclass
class ContextSection:
    """One named section in the final context window."""

    title: str
    content: str
    tokens: int = 0
    items: List[RetrievedItem] = field(default_factory=list)
    compressed: bool = False


@dataclass
class ContextWindow:
    """The assembled dynamic context ready to be sent to any LLM."""

    query: str
    sections: List[ContextSection] = field(default_factory=list)
    max_tokens: int = 0
    reserved_output_tokens: int = 0
    token_budget: int = 0
    total_tokens: int = 0
    omitted_count: int = 0
    omitted_tokens: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def text(self) -> str:
        """Render the context as a single text block (provider-agnostic)."""
        parts: List[str] = []
        for section in self.sections:
            if not section.content:
                continue
            header = f"### {section.title}"
            if section.compressed:
                header += " [compressed]"
            parts.append(header)
            parts.append(section.content)
        if parts:
            return "\n\n".join(parts)
        return ""

    def as_messages(self, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Render as a minimal OpenAI-style message list.

        The context is placed in the system message; the query stays as the
        user message. This shape is easy to translate to Anthropic or other
        APIs.
        """
        system_content = system_prompt or "You are a helpful assistant."
        context_text = self.text
        if context_text:
            system_content = f"{system_content}\n\n# Local Context\n\n{context_text}"
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": self.query},
        ]

    def report(self) -> Dict[str, Any]:
        """Return a JSON-serializable report for debugging / observability."""
        return {
            "query": self.query,
            "max_tokens": self.max_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "token_budget": self.token_budget,
            "total_tokens": self.total_tokens,
            "omitted_count": self.omitted_count,
            "omitted_tokens": self.omitted_tokens,
            "sections": [
                {
                    "title": s.title,
                    "tokens": s.tokens,
                    "compressed": s.compressed,
                    "items": [
                        {
                            "id": item.chunk.id,
                            "source": item.chunk.source,
                            "kind": item.chunk.kind.value,
                            "score": round(item.score, 4),
                            "tokens": item.chunk.tokens or estimate_tokens(item.chunk.content),
                            "reason": item.reason,
                        }
                        for item in s.items
                    ],
                }
                for s in self.sections
            ],
        }


def estimate_tokens(text: str) -> int:
    """Local token estimate.

    English text is approximated as ~4 characters per token; CJK characters
    are counted as one token each. This keeps the prototype dependency-free
    and deterministic.
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    if other == 0:
        return cjk
    return cjk + (other + 3) // 4
