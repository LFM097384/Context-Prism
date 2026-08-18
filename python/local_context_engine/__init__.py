"""Local Context Engine - a provider-agnostic dynamic context window builder.

The engine sits in front of any LLM API and decides what to put into the
context window: retrieval, compression and prioritization over local memory,
code, files, user preferences and agent trajectories.
"""

from .engine import LocalContextEngine
from .models import Chunk, ContextWindow, SourceKind
from .providers import call_deepseek, to_deepseek_payload, to_openai_payload
from .sqlite_index import SqliteIndex

__all__ = [
    "LocalContextEngine",
    "Chunk",
    "ContextWindow",
    "SourceKind",
    "SqliteIndex",
    "call_deepseek",
    "to_deepseek_payload",
    "to_openai_payload",
]

__version__ = "0.2.0"
