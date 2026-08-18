import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, utimesSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { LocalIndex } from "../engine/index.js";
import { SqliteIndex } from "../engine/sqlite_index.js";
import { LocalContextEngine } from "../engine/engine.js";
import { Chunk, SourceKind, estimateTokens } from "../engine/models.js";
import { compressItems } from "../engine/compression.js";
import { toDeepSeekPayload } from "../engine/providers.js";

test("estimateTokens counts CJK and English", () => {
  assert.equal(estimateTokens(""), 0);
  assert.equal(estimateTokens("你好世界"), 4);
  assert.ok(estimateTokens("hello world") > 0);
});

test("memory BM25 index finds relevant chunk", () => {
  const index = new LocalIndex();
  index.addMany([
    new Chunk({ id: "1", source: "a.md", kind: SourceKind.FILE, content: "The cat sat on the mat." }),
    new Chunk({ id: "2", source: "b.md", kind: SourceKind.FILE, content: "Python RAG dynamic context window compression." }),
    new Chunk({ id: "3", source: "c.md", kind: SourceKind.FILE, content: "The dog ran in the park." }),
  ]);
  const hits = index.search("dynamic context window", 3);
  assert.equal(hits[0].id, "2");
});

test("SQLite FTS5 index searches and reloads", () => {
  const dir = mkdtempSync(join(tmpdir(), "lce-js-"));
  const dbPath = join(dir, "index.db");
  const engine = new LocalContextEngine({ storagePath: dbPath, backend: "sqlite" });
  engine.addText({
    content: "Python RAG dynamic context window compression.",
    source: "doc.md",
    kind: SourceKind.FILE,
  });
  engine.addText({
    content: "The cat sat on the mat.",
    source: "cat.md",
    kind: SourceKind.FILE,
  });
  const window = engine.buildContext({ query: "dynamic context window", maxTokens: 400, reservedOutputTokens: 80 });
  assert.match(window.text, /dynamic context window/);

  const loaded = new LocalContextEngine({ storagePath: dbPath, backend: "auto" });
  assert.equal(loaded.index.size, 2);
  const hits = loaded.index.search("dynamic context window", 1);
  assert.ok(hits.length >= 1);
});

test("compression fits token budget", () => {
  const items = [
    { chunk: new Chunk({ id: "1", source: "a", kind: SourceKind.HISTORY, content: "A".repeat(500) }), score: 0.9, reason: "" },
    { chunk: new Chunk({ id: "2", source: "b", kind: SourceKind.FILE, content: "B".repeat(500) }), score: 0.5, reason: "" },
    { chunk: new Chunk({ id: "3", source: "c", kind: SourceKind.CODE, content: "C".repeat(500) }), score: 0.1, reason: "" },
  ];
  const result = compressItems(items, 80, "");
  const used = result.packed.reduce((sum, packed) => sum + packed.tokens, 0);
  assert.ok(used <= 80);
  assert.ok(result.omittedCount > 0);
});

test("engine end-to-end build context", () => {
  const engine = new LocalContextEngine();
  engine.addText({
    content: "User prefers concise Chinese and local-first solutions.",
    source: "pref.md",
    kind: SourceKind.PREFERENCE,
    priority: 10,
  });
  engine.addText({
    content: "History: the user asked about local RAG and dynamic context windows.",
    source: "history.log",
    kind: SourceKind.HISTORY,
  });
  engine.addText({
    content: "Code: function assembleContext(query, budget) { return []; }",
    source: "engine.js",
    kind: SourceKind.CODE,
  });
  const window = engine.buildContext({ query: "How does assembleContext work?", maxTokens: 600, reservedOutputTokens: 100 });
  assert.ok(window.totalTokens > 0);
  assert.ok(window.totalTokens <= window.tokenBudget);
  assert.ok(window.sections.some((section) => section.title === "User Preferences"));
  assert.match(window.text, /assembleContext/);
  const messages = window.asMessages();
  assert.equal(messages[0].role, "system");
  assert.equal(messages[1].role, "user");
});

test("DeepSeek payload is OpenAI-compatible and reasoner omits temperature", () => {
  const engine = new LocalContextEngine();
  engine.addText({
    content: "User prefers concise Chinese and DeepSeek compatibility.",
    source: "pref.md",
    kind: SourceKind.PREFERENCE,
    priority: 10,
  });
  const window = engine.buildContext({ query: "Build a DeepSeek compatible context window", maxTokens: 600, reservedOutputTokens: 150 });

  const chatPayload = toDeepSeekPayload(window, { model: "deepseek-chat" });
  assert.equal(chatPayload.model, "deepseek-chat");
  assert.equal(chatPayload.messages[0].role, "system");
  assert.equal(chatPayload.messages[1].role, "user");
  assert.equal(chatPayload.max_tokens, window.reservedOutputTokens);
  assert.ok("temperature" in chatPayload);

  const reasonerPayload = toDeepSeekPayload(window, { model: "deepseek-reasoner" });
  assert.equal(reasonerPayload.model, "deepseek-reasoner");
  assert.ok(!("temperature" in reasonerPayload));
});

test("incremental file ingest skips unchanged files", () => {
  const dir = mkdtempSync(join(tmpdir(), "lce-inc-"));
  const filePath = join(dir, "note.md");
  const statePath = join(dir, "state.json");
  writeFileSync(filePath, "first version", "utf8");
  const engine = new LocalContextEngine();

  let result = engine.addFileIncremental({ path: filePath, statePath });
  assert.deepEqual(result, { added: 1, changed: 1, skipped: 0 });

  result = engine.addFileIncremental({ path: filePath, statePath });
  assert.deepEqual(result, { added: 0, changed: 0, skipped: 1 });

  writeFileSync(filePath, "second version with more words", "utf8");
  const future = new Date(Date.now() + 5000);
  utimesSync(filePath, future, future);
  result = engine.addFileIncremental({ path: filePath, statePath });
  assert.deepEqual(result, { added: 1, changed: 1, skipped: 0 });
  assert.equal(engine.index.size, 1);
});
