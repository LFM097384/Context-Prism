# ContextPrism Usage

## DSH Tools

After installing ContextPrism in a DSH profile, the agent can call:

### `context_prism_ingest`

```json
{
  "path": "docs",
  "kind": "file",
  "index": ".lce/index.db",
  "backend": "auto",
  "incremental": true
}
```

### `context_prism_build`

```json
{
  "query": "How should I assemble a dynamic context window with local RAG?",
  "maxTokens": 4000,
  "reservedOutputTokens": 800,
  "index": ".lce/index.db",
  "backend": "auto"
}
```

## JavaScript API

```js
import { LocalContextEngine } from "./engine/engine.js";
import { SourceKind } from "./engine/models.js";

const engine = new LocalContextEngine({ storagePath: ".lce/index.db", backend: "sqlite" });
engine.addText({ content: "User prefers concise Chinese.", source: "pref.md", kind: SourceKind.PREFERENCE, priority: 10 });
engine.addHistoryJsonl("history.jsonl");
engine.addTrajectoryJsonl("trajectory.jsonl");

const window = engine.buildContext({ query: "Summarize the project context", maxTokens: 4000, reservedOutputTokens: 800 });
console.log(window.text);
```

## Tests

```bash
npm test
```
