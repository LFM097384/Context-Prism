// Prioritization heuristics for deciding what must stay in the window.

import { SourceKind, RetrievedItem, estimateTokens } from "./models.js";

const DEFAULT_KIND_BASE = {
  [SourceKind.PREFERENCE]: 5.0,
  [SourceKind.TRAJECTORY]: 3.0,
  [SourceKind.HISTORY]: 2.5,
  [SourceKind.CODE]: 2.0,
  [SourceKind.FILE]: 1.5,
};

export class PriorityConfig {
  constructor({
    kindBase = DEFAULT_KIND_BASE,
    recencyWeight = 0.2,
    priorityWeight = 1.0,
    lengthPenaltyWeight = 0.1,
    maxLengthTokens = 800,
  } = {}) {
    this.kindBase = { ...DEFAULT_KIND_BASE, ...kindBase };
    this.recencyWeight = recencyWeight;
    this.priorityWeight = priorityWeight;
    this.lengthPenaltyWeight = lengthPenaltyWeight;
    this.maxLengthTokens = maxLengthTokens;
  }
}

export function prioritize(items, config = null) {
  const cfg = config || new PriorityConfig();
  const ranked = [];
  for (const item of items) {
    const chunk = item.chunk;
    const kindBase = cfg.kindBase[chunk.kind] ?? 1.0;
    const recency = recencyScore(chunk);
    const lengthPenalty = Math.min(1, estimateTokens(chunk.content) / cfg.maxLengthTokens);
    const priority = chunk.priority / 10.0;
    const score =
      item.score +
      kindBase * 0.1 +
      recency * cfg.recencyWeight +
      priority * cfg.priorityWeight -
      lengthPenalty * cfg.lengthPenaltyWeight;
    ranked.push(new RetrievedItem({ chunk, score, reason: item.reason }));
  }
  ranked.sort((a, b) => b.score - a.score);
  return ranked;
}

function recencyScore(chunk, halfLifeDays = 7.0) {
  const now = Date.now() / 1000;
  const ageDays = Math.max(0, (now - (chunk.timestamp || now)) / 86400);
  return Math.exp(-ageDays / halfLifeDays);
}
