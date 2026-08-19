/**
 * ContextPrism DSH plugin — in-process JS port.
 *
 * Registers two model-facing tools:
 *   - context_prism_build: build a dynamic context window before calling an LLM
 *   - context_prism_ingest: ingest files/history/trajectory into the local index
 *
 * This version uses the JavaScript/TypeScript port of ContextPrism directly
 * inside the DSH process, so it no longer needs Python or WSL.
 */

import z from "@deepseek-ai/schemastery";
import { defineTool } from "@deepseek-ai/dsh-tools";
import { mkdirSync, statSync, watch, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { LocalContextEngine } from "../engine/engine.js";
import { Chunk, SourceKind } from "../engine/models.js";
import { RetrievalConfig, retrieve } from "../engine/retrieval.js";
import { summarizeText } from "../engine/summarizer.js";

/** Cordis plugin name. */
const name = "context-prism";
/** Services required by this plugin. */
const inject = ["tools", "workspaceRegistry", "webServer"];

/** Plugin configuration schema. */
const Config = z.object({
  backend: z.string().default("auto"),
  defaultIndex: z.string().default(".lce/index.db"),
  cacheSize: z.number().default(128),
  cacheTtl: z.number().default(60),
  autoInject: z.boolean().default(true),
  autoIndexFiles: z.boolean().default(true),
  autoMaxTokens: z.number().default(4000),
  autoReservedTokens: z.number().default(800),
  autoIndexIntervalMs: z.number().default(60000),
  fileWatch: z.boolean().default(true),
  watchDebounceMs: z.number().default(2000),
  llmSummarization: z.boolean().default(false),
  summarizationModel: z.string().default("deepseek-chat"),
  summaryMaxTokens: z.number().default(200),
});

const engines = new Map();
const sessionStates = new Map();
const fileScanTimes = new Map();
const watchers = new Map();
const watchTimers = new Map();
let lastProjectRoot = null;

function apply(ctx, config = {}) {
  const cfg = {
    backend: "auto",
    defaultIndex: ".lce/index.db",
    cacheSize: 128,
    cacheTtl: 60,
    autoInject: true,
    autoIndexFiles: true,
    autoMaxTokens: 4000,
    autoReservedTokens: 800,
    autoIndexIntervalMs: 60000,
    fileWatch: true,
    watchDebounceMs: 2000,
    llmSummarization: false,
    summarizationModel: "deepseek-chat",
    summaryMaxTokens: 200,
    ...config,
  };

  ctx.tools.register(
    defineTool({
      name: "context_prism_build",
      description:
        "Build a dynamic context window from local history, code, files, user preferences, and agent trajectories. " +
        "Use before sending a query to any LLM to retrieve, compress, and prioritize context into a token budget.",
      parameters: {
        query: { type: "string", required: true, description: "The current user/agent query." },
        maxTokens: { type: "number", description: "Total context window size (default 8000)." },
        reservedOutputTokens: { type: "number", description: "Tokens reserved for model output (default 1024)." },
        index: { type: "string", description: "Local index path (default .lce/index.db under the DSH workspace)." },
        backend: { type: "string", enum: ["auto", "json", "sqlite"], description: "Index backend." },
      },
      output: {
        schema: {
          type: "object",
          additionalProperties: false,
          properties: {
            output: { type: "string", required: true },
            report: { type: "object", required: true, additionalProperties: true },
          },
        },
        render: (_args, value) => [{ type: "text", text: value.output }],
      },
      async execute(toolArgs, exec) {
        const projectRoot = await resolveProjectRoot(cfg, exec, ctx);
        const index = resolveIndex(cfg, projectRoot, toolArgs.index);
        const backend = toolArgs.backend || cfg.backend;
        const engine = getEngine(cfg, index, backend);
        const window = engine.buildContext({
          query: toolArgs.query,
          maxTokens: toolArgs.maxTokens ?? 8000,
          reservedOutputTokens: toolArgs.reservedOutputTokens ?? 1024,
        });
        const output =
          window.text +
          `\n\n---\nTokens: ${window.totalTokens}/${window.tokenBudget} ` +
          `(omitted ${window.omittedCount} items, ~${window.omittedTokens} tokens)`;
        return { output, report: window.report() };
      },
      presentCall(toolArgs) {
        return {
          card: "generic",
          title: "Build local context window",
          kind: "read",
          rawInput: toolArgs.query,
        };
      },
    }),
  );

  ctx.tools.register(
    defineTool({
      name: "context_prism_ingest",
      description:
        "Ingest local files, directories, chat history JSONL, or agent trajectory JSONL into the Local Context Engine index.",
      parameters: {
        path: { type: "string", required: true, description: "File or directory to ingest." },
        kind: { type: "string", enum: ["history", "code", "file", "preference", "trajectory"], description: "Force source kind." },
        index: { type: "string", description: "Local index path (default .lce/index.db under the DSH workspace)." },
        backend: { type: "string", enum: ["auto", "json", "sqlite"], description: "Index backend." },
        incremental: { type: "boolean", description: "Only ingest new/changed files." },
      },
      output: {
        schema: {
          type: "object",
          additionalProperties: false,
          properties: {
            output: { type: "string", required: true },
          },
        },
        render: (_args, value) => [{ type: "text", text: value.output }],
      },
      async execute(toolArgs, exec) {
        const projectRoot = await resolveProjectRoot(cfg, exec, ctx);
        const targetPath = resolve(projectRoot, toolArgs.path);
        const index = resolveIndex(cfg, projectRoot, toolArgs.index);
        const backend = toolArgs.backend || cfg.backend;
        const engine = getEngine(cfg, index, backend);
        const kind = toolArgs.kind ? SourceKind[toolArgs.kind.toUpperCase()] || toolArgs.kind : null;
        let count = 0;

        if (toolArgs.incremental) {
          const result = isDirectory(targetPath)
            ? engine.addDirectoryIncremental({ directory: targetPath, kind, extensions: null })
            : engine.addFileIncremental({ path: targetPath, kind });
          count = result.added;
        } else if (isDirectory(targetPath)) {
          count = engine.addDirectory(targetPath, { kind });
        } else if (kind === SourceKind.TRAJECTORY || /trajectory/i.test(targetPath)) {
          count = engine.addTrajectoryJsonl(targetPath);
        } else if (targetPath.toLowerCase().endsWith(".jsonl")) {
          count = engine.addHistoryJsonl(targetPath);
        } else {
          count = engine.addFile(targetPath, kind);
        }

        return { output: `Ingested ${count} chunks into ${index}` };
      },
      presentCall(toolArgs) {
        return {
          card: "generic",
          title: "Ingest into Local Context Engine",
          kind: "write",
          rawInput: toolArgs.path,
        };
      },
    }),
  );

  ctx.tools.register(
    defineTool({
      name: "context_prism_status",
      description:
        "Show the current ContextPrism project index status: chunk counts by source kind, sample sources, and storage location.",
      parameters: {
        index: { type: "string", description: "Local index path (default .lce/index.db under the DSH workspace)." },
        backend: { type: "string", enum: ["auto", "json", "sqlite"], description: "Index backend." },
      },
      output: {
        schema: {
          type: "object",
          additionalProperties: false,
          properties: {
            output: { type: "string", required: true },
            report: { type: "object", required: true, additionalProperties: true },
          },
        },
        render: (_args, value) => [{ type: "text", text: value.output }],
      },
      async execute(toolArgs, exec) {
        const projectRoot = await resolveProjectRoot(cfg, exec, ctx);
        const index = resolveIndex(cfg, projectRoot, toolArgs.index);
        const backend = toolArgs.backend || cfg.backend;
        const engine = getEngine(cfg, index, backend);
        const chunks = engine.index.allChunks();
        const byKind = {};
        for (const chunk of chunks) {
          byKind[chunk.kind] = (byKind[chunk.kind] || 0) + 1;
        }
        const sources = [...new Set(chunks.map((chunk) => chunk.source))].slice(0, 20);
        const report = {
          index,
          backend,
          totalChunks: chunks.length,
          byKind,
          sources,
        };
        const output = [
          `ContextPrism index: ${index}`,
          `Backend: ${backend}`,
          `Total chunks: ${chunks.length}`,
          `By kind: ${JSON.stringify(byKind)}`,
          `Sample sources:\n${sources.map((source) => `- ${source}`).join("\n") || "(empty)"}`,
        ].join("\n");
        return { output, report };
      },
      presentCall() {
        return { card: "generic", title: "ContextPrism status", kind: "read", rawInput: null };
      },
    }),
  );

  ctx.tools.register(
    defineTool({
      name: "context_prism_dashboard",
      description:
        "Generate a standalone HTML dashboard for the current ContextPrism project index and save it under .lce/dashboard.html.",
      parameters: {
        index: { type: "string", description: "Local index path (default .lce/index.db under the DSH workspace)." },
        backend: { type: "string", enum: ["auto", "json", "sqlite"], description: "Index backend." },
      },
      output: {
        schema: {
          type: "object",
          additionalProperties: false,
          properties: {
            output: { type: "string", required: true },
            path: { type: "string", required: true },
          },
        },
        render: (_args, value) => [{ type: "text", text: value.output }],
      },
      async execute(toolArgs, exec) {
        const projectRoot = await resolveProjectRoot(cfg, exec, ctx);
        const index = resolveIndex(cfg, projectRoot, toolArgs.index);
        const backend = toolArgs.backend || cfg.backend;
        const engine = getEngine(cfg, index, backend);
        const html = buildDashboardHtml(engine, index, backend);
        const dashboardPath = `${projectRoot}/.lce/dashboard.html`;
        mkdirSync(`${projectRoot}/.lce`, { recursive: true });
        writeFileSync(dashboardPath, html, "utf8");
        const chunks = engine.index.allChunks();
        return {
          output: `Dashboard written to ${dashboardPath}\nTotal chunks: ${chunks.length}\nOpen it in a browser to view the panel.`,
          path: dashboardPath,
        };
      },
      presentCall() {
        return { card: "generic", title: "Generate ContextPrism dashboard", kind: "read", rawInput: null };
      },
    }),
  );

  ctx.tools.register(
    defineTool({
      name: "context_prism_evaluate",
      description:
        "Run a small A/B retrieval evaluation: compare semantic + BM25 vs BM25-only hit rate on sample queries.",
      parameters: {
        queries: { type: "string", description: "Optional JSON array of {query, expected} pairs." },
        index: { type: "string", description: "Local index path (default .lce/index.db under the DSH workspace)." },
        backend: { type: "string", enum: ["auto", "json", "sqlite"], description: "Index backend." },
      },
      output: {
        schema: {
          type: "object",
          additionalProperties: false,
          properties: {
            output: { type: "string", required: true },
            report: { type: "object", required: true, additionalProperties: true },
          },
        },
        render: (_args, value) => [{ type: "text", text: value.output }],
      },
      async execute(toolArgs, exec) {
        const projectRoot = await resolveProjectRoot(cfg, exec, ctx);
        const index = resolveIndex(cfg, projectRoot, toolArgs.index);
        const backend = toolArgs.backend || cfg.backend;
        const engine = getEngine(cfg, index, backend);
        let queries = [
          { query: "How should I assemble a dynamic context window with local RAG?", expected: "RAG" },
          { query: "What are the user preferences for replies?", expected: "preference" },
          { query: "What did the agent do with search_index?", expected: "search_index" },
        ];
        if (toolArgs.queries) {
          try {
            queries = JSON.parse(toolArgs.queries);
          } catch {
            throw new Error("queries must be a valid JSON array of {query, expected}");
          }
        }
        const rows = [];
        let semanticHits = 0;
        let bm25Hits = 0;
        for (const item of queries) {
          const query = item.query || "";
          const expected = item.expected || "";
          const withSemantic = retrieve(engine.index, query, new RetrievalConfig({ useSemantic: true, topK: 10 }));
          const withoutSemantic = retrieve(engine.index, query, new RetrievalConfig({ useSemantic: false, topK: 10 }));
          const hit = (items) =>
            items.some((entry) => JSON.stringify(entry.chunk).toLowerCase().includes(String(expected).toLowerCase()));
          const semanticOk = hit(withSemantic);
          const bm25Ok = hit(withoutSemantic);
          if (semanticOk) semanticHits += 1;
          if (bm25Ok) bm25Hits += 1;
          rows.push({ query, expected, semanticOk, bm25Ok });
        }
        const report = {
          total: queries.length,
          semanticHits,
          bm25Hits,
          semanticHitRate: queries.length ? semanticHits / queries.length : 0,
          bm25HitRate: queries.length ? bm25Hits / queries.length : 0,
          rows,
        };
        const output = [
          "ContextPrism A/B retrieval evaluation",
          `Total queries: ${queries.length}`,
          `Semantic + BM25 hit rate: ${report.semanticHitRate.toFixed(2)} (${semanticHits}/${queries.length})`,
          `BM25-only hit rate: ${report.bm25HitRate.toFixed(2)} (${bm25Hits}/${queries.length})`,
          "",
          ...rows.map(
            (row) =>
              `- ${row.query} => semantic:${row.semanticOk ? "hit" : "miss"} bm25:${row.bm25Ok ? "hit" : "miss"}`,
          ),
        ].join("\n");
        return { output, report };
      },
      presentCall() {
        return { card: "generic", title: "ContextPrism A/B evaluation", kind: "read", rawInput: null };
      },
    }),
  );

  ctx.tools.register(
    defineTool({
      name: "context_prism_summarize",
      description:
        "Summarize a block of text using the configured LLM (DeepSeek by default), with local extractive fallback.",
      parameters: {
        text: { type: "string", required: true, description: "Text to summarize." },
        maxTokens: { type: "number", description: "Maximum summary token count (default 200)." },
      },
      output: {
        schema: {
          type: "object",
          additionalProperties: false,
          properties: {
            output: { type: "string", required: true },
          },
        },
        render: (_args, value) => [{ type: "text", text: value.output }],
      },
      async execute(toolArgs) {
        const summary = await summarizeText(toolArgs.text, {
          maxTokens: toolArgs.maxTokens ?? cfg.summaryMaxTokens,
          model: cfg.summarizationModel,
          enabled: cfg.llmSummarization,
        });
        return { output: summary };
      },
      presentCall(toolArgs) {
        return { card: "generic", title: "Summarize with ContextPrism", kind: "read", rawInput: toolArgs.text };
      },
    }),
  );

  if (ctx.webServer?.register) {
    const route = {
      kind: "prefix",
      path: "/context-prism/dashboard",
      handler: async (req, res) => {
        try {
          const projectRoot = lastProjectRoot || process.cwd();
          const index = resolveIndex(cfg, projectRoot, null);
          const engine = getEngine(cfg, index, cfg.backend);
          const html = buildDashboardHtml(engine, index, cfg.backend);
          res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
          res.end(html);
        } catch (error) {
          res.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
          res.end(`ContextPrism dashboard error: ${error.message}`);
        }
      },
    };
    if (ctx.effect) {
      ctx.effect(() => ctx.webServer.register(route), "context-prism: dashboard route");
    } else {
      ctx.webServer.register(route);
    }
  }

  if (cfg.autoInject) {
    ctx.on("agent/pre-step", async ({ agent, signal }, next) => {
      const decision = await next();
      if (!decision || decision.kind === "reject") return decision;
      try {
        const projectRoot = await resolveProjectRoot(cfg, { agent }, ctx);
        const index = resolveIndex(cfg, projectRoot, null);
        const engine = getEngine(cfg, index, cfg.backend);
        const sessionId = agent.id || agent.session.id || "default";
        const state = getSessionState(sessionId);

        const capture = captureSessionEvents(engine, agent, state);
        if (cfg.autoIndexFiles) ensureProjectFilesIndexed(engine, projectRoot, cfg);

        if (capture.newUserMessages > 0 || !state.hasInjected) {
          const query = latestUserQuery(agent.session.events || []) || "ContextPrism context";
          const window = cfg.llmSummarization
            ? await engine.buildContextAsync({
                query,
                maxTokens: cfg.autoMaxTokens,
                reservedOutputTokens: cfg.autoReservedTokens,
                summarizer: (text, maxTokens) =>
                  summarizeText(text, {
                    maxTokens,
                    model: cfg.summarizationModel,
                    enabled: cfg.llmSummarization,
                  }),
              })
            : engine.buildContext({
                query,
                maxTokens: cfg.autoMaxTokens,
                reservedOutputTokens: cfg.autoReservedTokens,
              });
          if (window.text) {
            const { createUserMessage } = await import("@deepseek-ai/dsh-llm");
            const contextMessage = createUserMessage({
              content: [{ type: "text", text: `[ContextPrism auto-injected context]\n\n${window.text}` }],
              source: { kind: "context-prism", form: "auto-injection" },
            });
            state.hasInjected = true;
            return { kind: "enter", messages: [...decision.messages, contextMessage] };
          }
        }
      } catch (error) {
        console.error("[ContextPrism] auto-context error:", error);
      }
      return decision;
    });
  }
}

function getSessionState(sessionId) {
  if (!sessionStates.has(sessionId)) {
    sessionStates.set(sessionId, { lastCapturedSeq: 0, hasInjected: false });
  }
  return sessionStates.get(sessionId);
}

function captureSessionEvents(engine, agent, state) {
  const events = agent.session?.events || [];
  const sessionId = agent.id || agent.session?.id || "default";
  const toAdd = [];
  let newUserMessages = 0;
  let lastSeq = state.lastCapturedSeq;

  for (const event of events) {
    const seq = event.seq ?? 0;
    if (seq < state.lastCapturedSeq) continue;
    const timestamp = event.time ? event.time / 1000 : Date.now() / 1000;
    if (event.type === "user/message") {
      const text = eventText(event);
      if (text) {
        toAdd.push(
          new Chunk({
            id: `${sessionId}:user:${seq}`,
            source: `session:${sessionId}`,
            kind: SourceKind.HISTORY,
            content: `user: ${text}`,
            timestamp,
            priority: 1,
          }),
        );
        newUserMessages += 1;
      }
    } else if (event.type === "assistant/message") {
      const text = eventText(event);
      if (text) {
        toAdd.push(
          new Chunk({
            id: `${sessionId}:assistant:${seq}`,
            source: `session:${sessionId}`,
            kind: SourceKind.HISTORY,
            content: `assistant: ${text}`,
            timestamp,
            priority: 1,
          }),
        );
      }
    } else if (event.type === "tool/call" || event.type === "tool/result") {
      toAdd.push(
        new Chunk({
          id: `${sessionId}:${event.type}:${seq}`,
          source: `session:${sessionId}`,
          kind: SourceKind.TRAJECTORY,
          content: `[${event.type}] ${JSON.stringify(event.data ?? {})}`,
          timestamp,
          priority: 0.5,
        }),
      );
    }
    lastSeq = Math.max(lastSeq, seq);
  }

  if (toAdd.length > 0) engine.addChunks(toAdd);
  state.lastCapturedSeq = lastSeq;
  return { newUserMessages };
}

function eventText(event) {
  const message = event.data?.message;
  if (!message) return "";
  if (typeof message === "string") return message;
  if (typeof message.content === "string") return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .map((block) => block?.text || block?.content || "")
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

function latestUserQuery(events) {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];
    if (event.type === "user/message") {
      const text = eventText(event);
      if (text) return text;
    }
  }
  return "";
}

function ensureProjectFilesIndexed(engine, projectRoot, cfg) {
  const now = Date.now();
  const last = fileScanTimes.get(projectRoot) || 0;
  if (now - last < cfg.autoIndexIntervalMs) return;
  fileScanTimes.set(projectRoot, now);
  try {
    engine.addDirectoryIncremental({
      directory: projectRoot,
      extensions: [".md", ".txt", ".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".jsonl", ".csv", ".html", ".css"],
      ignore: ["node_modules", ".git", ".lce", "dist", "build", ".venv", "__pycache__", ".next", "coverage", ".turbo"],
    });
  } catch (error) {
    console.error("[ContextPrism] file indexing error:", error);
  }
  if (cfg.fileWatch) ensureFileWatcher(engine, projectRoot, cfg);
}

function ensureFileWatcher(engine, projectRoot, cfg) {
  if (watchers.has(projectRoot)) return;
  try {
    const watcher = watch(projectRoot, { recursive: true }, (_event, filename) => {
      if (!filename) return;
      const name = String(filename);
      if (shouldIgnoreWatchPath(name)) return;
      scheduleWatchReindex(engine, projectRoot, cfg);
    });
    watchers.set(projectRoot, watcher);
  } catch (error) {
    console.error("[ContextPrism] file watch not available:", error.message);
  }
}

function shouldIgnoreWatchPath(name) {
  return /(^|[\\/])(node_modules|\.git|\.lce|dist|build|\.venv|__pycache__|\.next|coverage|\.turbo)([\\/]|$)/.test(name);
}

function scheduleWatchReindex(engine, projectRoot, cfg) {
  if (watchTimers.has(projectRoot)) clearTimeout(watchTimers.get(projectRoot));
  const timer = setTimeout(() => {
    watchTimers.delete(projectRoot);
    try {
      engine.addDirectoryIncremental({
        directory: projectRoot,
        extensions: [".md", ".txt", ".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".jsonl", ".csv", ".html", ".css"],
        ignore: ["node_modules", ".git", ".lce", "dist", "build", ".venv", "__pycache__", ".next", "coverage", ".turbo"],
      });
    } catch (error) {
      console.error("[ContextPrism] watch reindex error:", error);
    }
  }, cfg.watchDebounceMs);
  watchTimers.set(projectRoot, timer);
}

async function resolveProjectRoot(cfg, exec, ctx) {
  const cwd = exec.agent?.session.header.cwd ?? process.cwd();
  let root = cwd;
  if (ctx.workspaceRegistry?.resolveByPath) {
    try {
      const workspace = await ctx.workspaceRegistry.resolveByPath(cwd);
      if (workspace?.path) root = workspace.path;
    } catch {
      // Not a registered workspace yet; fall back to the session cwd.
    }
  }
  lastProjectRoot = root;
  return root;
}

function resolveIndex(cfg, projectRoot, index) {
  const value = index || cfg.defaultIndex;
  return resolve(projectRoot, value);
}

function getEngine(cfg, index, backend) {
  const key = `${backend}:${index}`;
  if (!engines.has(key)) {
    engines.set(
      key,
      new LocalContextEngine({
        storagePath: index,
        backend,
        cacheSize: cfg.cacheSize,
        cacheTtl: cfg.cacheTtl,
      }),
    );
  }
  return engines.get(key);
}

function isDirectory(path) {
  try {
    return statSync(path).isDirectory();
  } catch {
    return false;
  }
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function buildDashboardHtml(engine, index, backend) {
  const chunks = engine.index.allChunks();
  const byKind = {};
  for (const chunk of chunks) {
    byKind[chunk.kind] = (byKind[chunk.kind] || 0) + 1;
  }
  const rows = chunks
    .slice()
    .sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0))
    .slice(0, 200)
    .map(
      (chunk) =>
        `<tr><td>${escapeHtml(chunk.kind)}</td><td>${escapeHtml(chunk.source)}</td><td>${chunk.tokens}</td><td>${escapeHtml(chunk.content.slice(0, 160))}</td></tr>`,
    )
    .join("\n");
  return `<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8" />
<title>ContextPrism Dashboard</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;color:#1f2328;background:#f6f8fa}
h1{font-size:20px}
.card{background:#fff;border:1px solid #d0d7de;border-radius:12px;padding:16px;margin-bottom:16px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid #d0d7de;padding:6px 8px;text-align:left;vertical-align:top}
th{background:#f0f3f6}
code{background:#eff1f3;padding:2px 4px;border-radius:4px}
</style>
</head>
<body>
<h1>ContextPrism Dashboard</h1>
<div class="card"><strong>Index:</strong> <code>${escapeHtml(index)}</code><br/>
<strong>Backend:</strong> <code>${escapeHtml(backend)}</code><br/>
<strong>Total chunks:</strong> ${chunks.length}</div>
<div class="card"><h2>By Kind</h2><pre>${escapeHtml(JSON.stringify(byKind, null, 2))}</pre></div>
<div class="card"><h2>Recent Chunks</h2><table><thead><tr><th>Kind</th><th>Source</th><th>Tokens</th><th>Preview</th></tr></thead><tbody>${rows}</tbody></table></div>
</body>
</html>`;
}

export { Config, apply, inject, name };
