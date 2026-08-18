// Ingest helpers for turning raw local data into Chunks (JS port).

import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, extname } from "node:path";
import { Chunk, SourceKind } from "./models.js";

const DEFAULT_EXTENSIONS = new Set([
  ".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".txt", ".json", ".jsonl", ".csv", ".html", ".css",
]);

export function chunkText({ content, source, kind, chunkSize = 800, overlap = 80, metadata = {}, timestamp = null, priority = 0 }) {
  if (!content || !content.trim()) return [];
  const text = content.trim();
  const step = Math.max(1, chunkSize - overlap);
  const chunks = [];
  let start = 0;
  let idx = 0;
  while (start < text.length) {
    const piece = text.slice(start, start + chunkSize);
    chunks.push(
      new Chunk({
        id: `${source}:${idx}:${start}`,
        source,
        kind,
        content: piece,
        metadata: { ...metadata },
        timestamp,
        priority,
      }),
    );
    idx += 1;
    if (piece.length < chunkSize) break;
    start += step;
  }
  return chunks;
}

export function ingestText({ content, source, kind, chunkSize = 800, overlap = 80, metadata = {}, timestamp = null, priority = 0 }) {
  return chunkText({ content, source, kind, chunkSize, overlap, metadata, timestamp, priority });
}

export function ingestFile(filePath, kind = null) {
  const text = readFileSync(filePath, "utf8");
  const resolvedKind = kind || kindForPath(filePath);
  const stat = statSync(filePath);
  return chunkText({
    content: text,
    source: filePath,
    kind: resolvedKind,
    metadata: { path: filePath, extension: extname(filePath) },
    timestamp: stat.mtimeMs / 1000,
  });
}

export function ingestDirectory(directory, { kind = null, extensions = null, ignore = [] } = {}) {
  const allowed = new Set(extensions || DEFAULT_EXTENSIONS);
  const ignored = new Set(ignore || []);
  const chunks = [];
  walk(directory, (filePath) => {
    if (ignored.size > 0 && filePath.split(/[\\/]/).some((part) => ignored.has(part))) return;
    if (allowed.size > 0 && !allowed.has(extname(filePath).toLowerCase())) return;
    try {
      chunks.push(...ingestFile(filePath, kind));
    } catch {
      // skip unreadable files
    }
  });
  return chunks;
}

export function ingestHistoryJsonl(filePath, sourceName = null) {
  const source = sourceName || filePath;
  const lines = readFileSync(filePath, "utf8").split("\n");
  const chunks = [];
  for (let idx = 0; idx < lines.length; idx += 1) {
    const line = lines[idx].trim();
    if (!line) continue;
    let record;
    try {
      record = JSON.parse(line);
    } catch {
      continue;
    }
    const content = record.content || record.text || record.message;
    if (!content) continue;
    let timestamp = record.timestamp;
    if (typeof timestamp === "string") timestamp = Number(timestamp) || null;
    chunks.push(
      new Chunk({
        id: `${source}:history:${idx}`,
        source,
        kind: SourceKind.HISTORY,
        content: `${record.role || "unknown"}: ${content}`,
        metadata: record,
        timestamp,
        priority: record.important ? 1 : 0,
      }),
    );
  }
  return chunks;
}

export function ingestTrajectoryJsonl(filePath, sourceName = null) {
  const source = sourceName || filePath;
  const lines = readFileSync(filePath, "utf8").split("\n");
  const chunks = [];
  for (let idx = 0; idx < lines.length; idx += 1) {
    const line = lines[idx].trim();
    if (!line) continue;
    let record;
    try {
      record = JSON.parse(line);
    } catch {
      continue;
    }
    const tool = record.tool || record.action || "tool";
    const contentParts = [];
    if (record.input) contentParts.push(`input: ${record.input}`);
    if (record.output) contentParts.push(`output: ${record.output}`);
    if (record.thought || record.reasoning) contentParts.push(`reasoning: ${record.thought || record.reasoning}`);
    if (contentParts.length === 0) contentParts.push(JSON.stringify(record));
    let timestamp = record.timestamp;
    if (typeof timestamp === "string") timestamp = Number(timestamp) || null;
    chunks.push(
      new Chunk({
        id: `${source}:trajectory:${idx}`,
        source,
        kind: SourceKind.TRAJECTORY,
        content: `[${tool}] ${contentParts.join(" | ")}`,
        metadata: record,
        timestamp,
        priority: record.important ? 2 : 0,
      }),
    );
  }
  return chunks;
}

export function kindForPath(filePath) {
  const suffix = extname(filePath).toLowerCase();
  if ([".py", ".ts", ".tsx", ".js", ".jsx", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".sql"].includes(suffix)) {
    return SourceKind.CODE;
  }
  return SourceKind.FILE;
}

function walk(directory, visit) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const full = join(directory, entry.name);
    if (entry.isDirectory()) {
      walk(full, visit);
    } else if (entry.isFile()) {
      visit(full);
    }
  }
}

export function relativePath(directory, filePath) {
  return relative(directory, filePath);
}
