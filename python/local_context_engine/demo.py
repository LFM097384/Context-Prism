"""A self-contained end-to-end demo for the Local Context Engine."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .engine import LocalContextEngine
from .models import SourceKind
from .providers import to_deepseek_payload


def run_demo(index_path: str | Path = ".lce_demo_index.json", backend: str = "auto", quiet: bool = False) -> dict:
    """Run the demo with synthetic local context and print the built window."""
    engine = LocalContextEngine(storage_path=index_path, backend=backend)

    # 1. User preferences.
    engine.add_text(
        content=(
            "User preferences: Always reply in concise Chinese unless the user writes in another language. "
            "Use the project's existing style and avoid inventing APIs. The user values local-first, "
            "privacy-preserving solutions. Keep code examples small and runnable."
        ),
        source="user/preferences.md",
        kind=SourceKind.PREFERENCE,
        priority=10,
    )

    # 2. Conversation history.
    history_lines = [
        {"role": "user", "content": "Can you help me build a local RAG prototype?", "timestamp": 1700000000, "session": "demo"},
        {"role": "assistant", "content": "Sure. I would start with a small BM25 index and add compression later.", "timestamp": 1700000010, "session": "demo"},
        {"role": "user", "content": "What context sources should the engine support?", "timestamp": 1700000100, "session": "demo"},
        {"role": "assistant", "content": "History, code, files, user preferences, and agent trajectories.", "timestamp": 1700000110, "session": "demo"},
        {"role": "user", "content": "It must fit inside a dynamic token budget before calling the model.", "timestamp": 1700000200, "session": "demo"},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
        for line in history_lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        history_path = fh.name
    engine.add_history_jsonl(history_path, source_name="demo/history.jsonl")

    # 3. Code.
    engine.add_text(
        content=(
            "def bm25_score(tokens, doc_freqs, doc_len, avgdl, n_docs, k1=1.5, b=0.75):\n"
            "    score = 0.0\n"
            "    for term, freq in doc_freqs.items():\n"
            "        df = len(postings[term])\n"
            "        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))\n"
            "        score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * doc_len / avgdl))\n"
            "    return score\n"
        ),
        source="demo/bm25.py",
        kind=SourceKind.CODE,
        priority=6,
    )

    # 4. Files.
    engine.add_text(
        content=(
            "Architecture note: Local Context Engine should sit between the application and the LLM API. "
            "It retrieves from local stores, compresses long histories, prioritizes preferences and "
            "trajectory, then renders a dynamic context window for any provider."
        ),
        source="demo/architecture.md",
        kind=SourceKind.FILE,
        priority=4,
    )

    # 5. Agent trajectory.
    trajectory_lines = [
        {"step": 1, "tool": "search_index", "input": "dynamic context window", "output": "found 3 chunks", "timestamp": 1700000300},
        {"step": 2, "tool": "compress_history", "input": "long conversation", "output": "kept 5 turns, summarized 20 turns", "timestamp": 1700000310},
        {"step": 3, "tool": "assemble_context", "input": "query + preferences + retrieval", "output": "window ready: 2048 tokens", "timestamp": 1700000320},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
        for line in trajectory_lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        trajectory_path = fh.name
    engine.add_trajectory_jsonl(trajectory_path, source_name="demo/trajectory.jsonl")

    engine.save()

    # Build a context window.
    query = "How should I assemble a dynamic context window with local RAG?"
    window = engine.build_context(query=query, max_tokens=1200, reserved_output_tokens=200)

    if not quiet:
        print("=" * 70)
        print("Local Context Engine Demo")
        print("=" * 70)
        print(f"\nQuery: {query}\n")
        print(window.text)
        print(f"\n---\nTokens: {window.total_tokens}/{window.token_budget} "
              f"(omitted {window.omitted_count} items, ~{window.omitted_tokens} tokens)")
        print("\nMessages preview:")
        messages = window.as_messages()
        print(json.dumps(messages, ensure_ascii=False, indent=2))

        print("\nDeepSeek payload preview:")
        payload = to_deepseek_payload(window, model="deepseek-chat")
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    return window.report()


if __name__ == "__main__":
    run_demo()
