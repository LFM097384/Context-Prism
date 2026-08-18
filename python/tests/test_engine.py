"""Unit tests for the Local Context Engine testbed."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local_context_engine.compression import compress_items
from local_context_engine.engine import LocalContextEngine
from local_context_engine.index import LocalIndex
from local_context_engine.models import Chunk, RetrievedItem, SourceKind, estimate_tokens
from local_context_engine.providers import to_deepseek_payload, to_openai_payload
from local_context_engine.retrieval import retrieve
from local_context_engine.sqlite_index import SqliteIndex
from local_context_engine.text_utils import tokenize


class TokenizerTests(unittest.TestCase):
    def test_estimate_tokens_english_and_cjk(self):
        self.assertGreater(estimate_tokens("hello world"), 0)
        self.assertEqual(estimate_tokens(""), 0)
        # A Chinese character counts as roughly one token.
        self.assertEqual(estimate_tokens("你好世界"), 4)

    def test_tokenize_cjk_and_ascii(self):
        tokens = tokenize("Hello 世界 RAG")
        self.assertIn("hello", tokens)
        self.assertIn("世界", tokens)
        self.assertIn("rag", tokens)


class RetrievalTests(unittest.TestCase):
    def test_bm25_finds_relevant_chunk(self):
        index = LocalIndex()
        index.add_many(
            [
                Chunk(id="1", source="a.md", kind=SourceKind.FILE, content="The cat sat on the mat.", priority=0),
                Chunk(id="2", source="b.md", kind=SourceKind.FILE, content="Python RAG dynamic context window compression."),
                Chunk(id="3", source="c.md", kind=SourceKind.FILE, content="The dog ran in the park.", priority=0),
            ]
        )
        hits = index.search("dynamic context window", top_k=3)
        self.assertEqual(hits[0].id, "2")
        retrieved = retrieve(index, "dynamic context window")
        self.assertEqual(retrieved[0].chunk.id, "2")


class CompressionTests(unittest.TestCase):
    def test_compress_items_fits_budget(self):
        items = [
            RetrievedItem(
                chunk=Chunk(id="1", source="a", kind=SourceKind.HISTORY, content="A" * 500, priority=1),
                score=0.9,
            ),
            RetrievedItem(
                chunk=Chunk(id="2", source="b", kind=SourceKind.FILE, content="B" * 500, priority=0),
                score=0.5,
            ),
            RetrievedItem(
                chunk=Chunk(id="3", source="c", kind=SourceKind.CODE, content="C" * 500, priority=0),
                score=0.1,
            ),
        ]
        result = compress_items(items, budget_tokens=80, query="")
        self.assertLessEqual(sum(p.tokens for p in result.packed), 80)
        self.assertGreater(result.omitted_count, 0)


class EngineTests(unittest.TestCase):
    def test_end_to_end_build_context(self):
        engine = LocalContextEngine()
        engine.add_text(
            content="User prefers concise Chinese and local-first solutions.",
            source="pref.md",
            kind=SourceKind.PREFERENCE,
            priority=10,
        )
        engine.add_text(
            content="History: the user asked about local RAG and dynamic context windows.",
            source="history.log",
            kind=SourceKind.HISTORY,
        )
        engine.add_text(
            content="Code: def assemble_context(query, budget): return []",
            source="engine.py",
            kind=SourceKind.CODE,
        )
        window = engine.build_context(
            query="How does assemble_context work?",
            max_tokens=600,
            reserved_output_tokens=100,
        )
        self.assertGreater(window.total_tokens, 0)
        self.assertLessEqual(window.total_tokens, window.token_budget)
        self.assertTrue(any("User Preferences" == s.title for s in window.sections))
        self.assertIn("assemble_context", window.text)
        messages = window.as_messages()
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "index.json"
            engine = LocalContextEngine(storage_path=index_path)
            engine.add_text("persisted chunk about local context", "p.txt", SourceKind.FILE)
            engine.save()

            loaded = LocalContextEngine(storage_path=index_path)
            self.assertEqual(len(loaded.index), 1)

    def test_ingest_history_and_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "history.jsonl"
            history_path.write_text(
                json.dumps({"role": "user", "content": "hello context engine"}) + "\n",
                encoding="utf-8",
            )
            trajectory_path = Path(tmp) / "trajectory.jsonl"
            trajectory_path.write_text(
                json.dumps({"step": 1, "tool": "search", "input": "hello", "output": "world"}) + "\n",
                encoding="utf-8",
            )
            engine = LocalContextEngine()
            self.assertEqual(engine.add_history_jsonl(history_path), 1)
            self.assertEqual(engine.add_trajectory_jsonl(trajectory_path), 1)
            self.assertEqual(len(engine.index), 2)


class SqliteIndexTests(unittest.TestCase):
    def test_sqlite_fts5_search_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "index.db"
            engine = LocalContextEngine(storage_path=db_path, backend="sqlite")
            engine.add_text(
                "Python RAG dynamic context window compression.",
                "doc.md",
                SourceKind.FILE,
            )
            engine.add_text(
                "The cat sat on the mat.",
                "cat.md",
                SourceKind.FILE,
            )
            engine.save()

            window = engine.build_context("dynamic context window", max_tokens=400, reserved_output_tokens=80)
            self.assertIn("dynamic context window", window.text)

            loaded = LocalContextEngine(storage_path=db_path, backend="auto")
            self.assertEqual(len(loaded.index), 2)
            hits = loaded.index.search("dynamic context window", top_k=1)
            self.assertGreaterEqual(len(hits), 1)


class ProviderTests(unittest.TestCase):
    def _make_window(self):
        engine = LocalContextEngine()
        engine.add_text(
            "User prefers concise Chinese and DeepSeek compatibility.",
            "pref.md",
            SourceKind.PREFERENCE,
            priority=10,
        )
        return engine.build_context(
            "Build a DeepSeek compatible context window",
            max_tokens=600,
            reserved_output_tokens=150,
        )

    def test_deepseek_chat_payload(self):
        window = self._make_window()
        payload = to_deepseek_payload(window, model="deepseek-chat")
        self.assertEqual(payload["model"], "deepseek-chat")
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertIn("temperature", payload)
        self.assertIn("max_tokens", payload)
        self.assertEqual(payload["max_tokens"], window.reserved_output_tokens)

    def test_deepseek_reasoner_omits_temperature(self):
        window = self._make_window()
        payload = to_deepseek_payload(window, model="deepseek-reasoner")
        self.assertEqual(payload["model"], "deepseek-reasoner")
        self.assertNotIn("temperature", payload)

    def test_openai_payload(self):
        window = self._make_window()
        payload = to_openai_payload(window, model="gpt-4o-mini")
        self.assertEqual(payload["model"], "gpt-4o-mini")
        self.assertIn("temperature", payload)


class IncrementalIngestTests(unittest.TestCase):
    def test_file_incremental_skips_unchanged_and_ingests_changed(self):
        import os
        import time

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "note.md"
            state_path = Path(tmp) / "state.json"
            path.write_text("first version", encoding="utf-8")
            engine = LocalContextEngine()

            added, changed, skipped = engine.add_file_incremental(path, state_path=state_path)
            self.assertEqual((added, changed, skipped), (1, 1, 0))

            added, changed, skipped = engine.add_file_incremental(path, state_path=state_path)
            self.assertEqual((added, changed, skipped), (0, 0, 1))

            path.write_text("second version with more words", encoding="utf-8")
            future = time.time() + 5
            os.utime(path, (future, future))
            added, changed, skipped = engine.add_file_incremental(path, state_path=state_path)
            self.assertEqual((added, changed, skipped), (1, 1, 0))
            self.assertEqual(len(engine.index), 1)  # same chunk id, replaced


class CacheTests(unittest.TestCase):
    def test_retrieval_cache_is_used(self):
        engine = LocalContextEngine(cache_size=16)
        engine.add_text("cache me about local RAG", "a.md", SourceKind.FILE)
        engine.build_context("local RAG", max_tokens=300, reserved_output_tokens=60)
        self.assertEqual(len(engine.retrieval_cache), 1)
        engine.clear_caches()
        self.assertEqual(len(engine.retrieval_cache), 0)


if __name__ == "__main__":
    unittest.main()
