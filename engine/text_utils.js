// Lightweight text utilities used by retrieval and compression.

const TOKEN_RE = /[a-zA-Z0-9_]+|[\u4e00-\u9fff]+/g;

export function tokenize(text) {
  return Array.from(String(text).matchAll(TOKEN_RE), (m) => m[0].toLowerCase());
}

export function splitSentences(text) {
  return String(text)
    .split(/(?<=[。！？!?.])\s*|\n+/)
    .map((part) => part.trim())
    .filter(Boolean);
}

export function termFrequencies(tokens) {
  const freqs = {};
  for (const token of tokens) {
    freqs[token] = (freqs[token] || 0) + 1;
  }
  return freqs;
}

export function cosineSimilarity(a, b) {
  if (!a || !b) return 0;
  const common = Object.keys(a).filter((key) => key in b);
  let dot = 0;
  for (const key of common) dot += a[key] * b[key];
  const normA = Math.sqrt(Object.values(a).reduce((sum, v) => sum + v * v, 0));
  const normB = Math.sqrt(Object.values(b).reduce((sum, v) => sum + v * v, 0));
  if (normA === 0 || normB === 0) return 0;
  return dot / (normA * normB);
}
