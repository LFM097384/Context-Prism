# Changelog

## [0.7.0] - 2026-08-18

- Add DSH Web client footer action that opens the ContextPrism dashboard.
- Add host HTTP route `/context-prism/dashboard` serving the live HTML dashboard.

## [0.6.0] - 2026-08-18

- Add real-time file watching with debounced re-index (`fileWatch` / `watchDebounceMs`).
- Add A/B retrieval evaluation tool (`context_prism_evaluate`).

## [0.5.0] - 2026-08-18

- Add standalone HTML dashboard generation (`context_prism_dashboard`).

## [0.4.0] - 2026-08-18

- Add local semantic retrieval (hash n-gram embedding + cosine similarity).
- Add optional LLM summarization with extractive fallback.
- Add `context_prism_status` and `context_prism_summarize` tools.
- Add `llmSummarization` / `summarizationModel` / `summaryMaxTokens` config.

## [0.3.0] - 2026-08-18

- Add automatic session history / trajectory capture.
- Add automatic project file incremental indexing.
- Add automatic context injection before each LLM request.
- Add `autoInject` / `autoIndexFiles` / `autoMaxTokens` / `autoReservedTokens` / `autoIndexIntervalMs` config.

## [0.2.0] - 2026-08-18

- Add DSH workspace / project-level index storage (`.lce/index.db`).
- Add in-process JavaScript port of ContextPrism core.
- Add `context_prism_build` and `context_prism_ingest` DSH tools.
- Add SQLite FTS5 + BM25 retrieval, compression, prioritization, caches, incremental ingest.
- Add DeepSeek / OpenAI compatible payload adapters.
- Add automatic session history / trajectory capture.
- Add automatic project file incremental indexing.
- Add automatic context injection before each LLM request.

## [0.1.0] - 2026-08-18

- Initial Local Context Engine prototype.
