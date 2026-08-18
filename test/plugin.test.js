import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { apply } from "../lib/index.js";

function makeContext(workspacePath = null) {
  const tools = [];
  const ctx = {
    tools: {
      register(definition) {
        tools.push(definition);
        return () => {};
      },
    },
    workspaceRegistry: {
      async resolveByPath(path) {
        if (workspacePath) return { path: workspacePath };
        return { path };
      },
    },
    on() {},
  };
  return { ctx, tools };
}

test("DSH plugin registers both tools", () => {
  const { ctx, tools } = makeContext();
  apply(ctx, { backend: "json", defaultIndex: ".lce_test_index.json" });
  const names = tools.map((tool) => tool.name).sort();
  assert.deepEqual(names, ["context_prism_build", "context_prism_ingest"]);
});

test("DSH plugin context_prism_ingest and context_prism_build work in-process", async () => {
  const dir = mkdtempSync(join(tmpdir(), "lce-plugin-"));
  const indexPath = join(dir, "index.json");
  const { ctx, tools } = makeContext(dir);
  apply(ctx, { backend: "json", defaultIndex: indexPath });

  const ingest = tools.find((tool) => tool.name === "context_prism_ingest");
  const build = tools.find((tool) => tool.name === "context_prism_build");
  assert.ok(ingest);
  assert.ok(build);

  const filePath = join(dir, "note.md");
  writeFileSync(filePath, "User prefers concise local-first context.", "utf8");
  const exec = { agent: { session: { header: { cwd: dir } } } };

  const ingestResult = await ingest.execute(
    { path: filePath, kind: "preference", index: indexPath, backend: "json" },
    exec,
  );
  assert.match(ingestResult.output, /Ingested 1 chunks/);

  const buildResult = await build.execute(
    { query: "local-first context", maxTokens: 300, reservedOutputTokens: 60, index: indexPath, backend: "json" },
    exec,
  );
  assert.match(buildResult.output, /local-first/);
  assert.ok(buildResult.report.totalTokens > 0);
});

test("plugin uses workspace-level default index under .lce", async () => {
  const dir = mkdtempSync(join(tmpdir(), "lce-workspace-"));
  const { ctx, tools } = makeContext(dir);
  apply(ctx, { backend: "sqlite", defaultIndex: ".lce/index.db" });
  const build = tools.find((tool) => tool.name === "context_prism_build");
  const exec = { agent: { session: { header: { cwd: dir } } } };

  const result = await build.execute(
    { query: "hello", maxTokens: 200, reservedOutputTokens: 40 },
    exec,
  );
  assert.ok(result.output.includes("Tokens:"));
  assert.ok(existsSync(join(dir, ".lce", "index.db")));
});
