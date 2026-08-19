// Lightweight local semantic retrieval helpers.
//
// This is a dependency-free "semantic-ish" embedding: it hashes token and
// character n-grams into a fixed-dimension vector and uses cosine similarity.
// It is not a transformer model, but it captures lexical overlap and fuzzy
// morphology better than raw BM25, and it runs fully offline.

import { tokenize } from "./text_utils.js";

const DEFAULT_DIMENSIONS = 256;

function hashString(value) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export function embedText(text, dimensions = DEFAULT_DIMENSIONS) {
  const vector = new Float64Array(dimensions);
  const tokens = tokenize(text);
  const freqs = new Map();
  for (const token of tokens) {
    freqs.set(token, (freqs.get(token) || 0) + 1);
  }

  for (const [token, freq] of freqs) {
    const h = hashString(token) % dimensions;
    const sign = hashString(`sign:${token}`) % 2 === 0 ? 1 : -1;
    vector[h] += sign * freq;
  }

  // Character 2-grams add a little fuzzy/typographical robustness.
  const normalized = String(text).toLowerCase().replace(/\s+/g, " ");
  for (let i = 0; i < normalized.length - 1; i += 1) {
    const gram = normalized.slice(i, i + 2);
    const h = hashString(`ng:${gram}`) % dimensions;
    const sign = hashString(`sign:${gram}`) % 2 === 0 ? 1 : -1;
    vector[h] += sign * 0.5;
  }

  return normalize(vector);
}

export function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function normalize(vector) {
  let norm = 0;
  for (let i = 0; i < vector.length; i += 1) {
    norm += vector[i] * vector[i];
  }
  norm = Math.sqrt(norm);
  if (norm === 0) return vector;
  for (let i = 0; i < vector.length; i += 1) {
    vector[i] /= norm;
  }
  return vector;
}
