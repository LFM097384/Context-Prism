# ContextPrism Architecture

ContextPrism sits between a DSH agent and any LLM API. It builds a dynamic
context window from project-local data before each model request.

## Pipeline

1. **Ingest** — files, directories, chat history JSONL, and agent trajectory JSONL are split into chunks and stored in a project-level index.
2. **Retrieve** — BM25 (or SQLite FTS5) finds candidate chunks; hybrid scoring adds recency, source weight, and user priority.
3. **Prioritize** — user preferences always come first; then history, trajectory, code, and files are ranked by score.
4. **Compress** — high-priority chunks stay whole; overflow is extractively compressed or truncated; omitted items are summarized in an "Omitted Memory" note.
5. **Assemble** — sections are packed into a token-budgeted `ContextWindow`, rendered as text or provider-specific messages.

## Storage

- Index: `<workspace>/.lce/index.db` (SQLite FTS5) or `.json` (BM25 fallback).
- Incremental state: `<workspace>/.lce/index.db.state.json`.

## Key Modules

| Module | Responsibility |
| --- | --- |
| `engine/engine.js` | Orchestrator |
| `engine/index.js` | In-memory BM25 |
| `engine/sqlite_index.js` | SQLite FTS5 |
| `engine/retrieval.js` | Hybrid scoring |
| `engine/prioritization.js` | Ranking heuristics |
| `engine/compression.js` | Budget packing |
| `engine/providers.js` | DeepSeek / OpenAI payloads |
