// Hybrid retrieval: BM25 relevance + recency + source priority + user priority.

import { SourceKind, RetrievedItem } from "./models.js";

const DEFAULT_SOURCE_WEIGHTS = {
  [SourceKind.PREFERENCE]: 1.5,
  [SourceKind.TRAJECTORY]: 1.1,
  [SourceKind.HISTORY]: 1.0,
  [SourceKind.CODE]: 1.0,
  [SourceKind.FILE]: 0.9,
};

export class RetrievalConfig {
  constructor({
    relevanceWeight = 1.0,
    recencyWeight = 0.25,
    sourceWeight = 0.2,
    priorityWeight = 0.3,
    recencyHalfLifeDays = 7.0,
    topK = 40,
    sourceWeights = DEFAULT_SOURCE_WEIGHTS,
  } = {}) {
    this.relevanceWeight = relevanceWeight;
    this.recencyWeight = recencyWeight;
    this.sourceWeight = sourceWeight;
    this.priorityWeight = priorityWeight;
    this.recencyHalfLifeDays = recencyHalfLifeDays;
    this.topK = topK;
    this.sourceWeights = { ...DEFAULT_SOURCE_WEIGHTS, ...sourceWeights };
  }
}

export function retrieve(index, query, config = null) {
  const cfg = config || new RetrievalConfig();
  const now = Date.now() / 1000;
  const chunks = index.search(query, cfg.topK);
  const items = [];
  for (const chunk of chunks) {
    let score = cfg.relevanceWeight * bm25RankScore(chunk, chunks);
    const ageDays = Math.max(0, (now - (chunk.timestamp || now)) / 86400);
    const recency = Math.exp(-ageDays / cfg.recencyHalfLifeDays);
    const sourceBoost = cfg.sourceWeights[chunk.kind] ?? 1.0;
    score += cfg.recencyWeight * recency;
    score += cfg.sourceWeight * (sourceBoost - 1.0);
    score += cfg.priorityWeight * (chunk.priority / 10.0);
    const reasons = [`bm25=${bm25RankScore(chunk, chunks).toFixed(3)}`];
    if (recency > 0.8) reasons.push("recent");
    if (chunk.priority > 0) reasons.push(`priority=${chunk.priority}`);
    items.push(new RetrievedItem({ chunk, score, reason: reasons.join(", ") }));
  }
  items.sort((a, b) => b.score - a.score);
  return items;
}

function bm25RankScore(chunk, rankedChunks) {
  const rank = rankedChunks.indexOf(chunk);
  if (rank < 0 || rankedChunks.length === 0) return 0;
  return Math.max(0, 1 - rank / rankedChunks.length);
}
