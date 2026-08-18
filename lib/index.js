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
import { statSync } from "node:fs";
import { resolve } from "node:path";
import { LocalContextEngine } from "../engine/engine.js";
import { Chunk, SourceKind } from "../engine/models.js";

/** Cordis plugin name. */
const name = "context-prism";
/** Services required by this plugin. */
const inject = ["tools", "workspaceRegistry"];

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
});

const engines = new Map();
const sessionStates = new Map();
const fileScanTimes = new Map();

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
          const window = engine.buildContext({
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
}

async function resolveProjectRoot(cfg, exec, ctx) {
  const cwd = exec.agent?.session.header.cwd ?? process.cwd();
  if (ctx.workspaceRegistry?.resolveByPath) {
    try {
      const workspace = await ctx.workspaceRegistry.resolveByPath(cwd);
      if (workspace?.path) return workspace.path;
    } catch {
      // Not a registered workspace yet; fall back to the session cwd.
    }
  }
  return cwd;
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

export { Config, apply, inject, name };
