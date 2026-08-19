// Optional LLM summarization with local extractive fallback.

import { compressChunkToFit } from "./compression.js";
import { callDeepSeek } from "./providers.js";

export async function summarizeText(
  text,
  { maxTokens = 200, apiKey = null, model = "deepseek-chat", enabled = true } = {},
) {
  if (!enabled) return fallbackSummary(text, maxTokens);
  const key = apiKey || process.env.DEEPSEEK_API_KEY;
  if (!key) return fallbackSummary(text, maxTokens);

  const payload = {
    model,
    messages: [
      {
        role: "user",
        content:
          `Summarize the following context into at most ${maxTokens} tokens. ` +
          `Keep key facts, decisions, names, and numbers. Do not add new information.\n\n${text}`,
      },
    ],
    max_tokens: maxTokens,
    temperature: 0.3,
  };

  try {
    const response = await callDeepSeek(payload, { apiKey: key });
    const summary = response?.choices?.[0]?.message?.content;
    if (summary && summary.trim()) return summary.trim();
  } catch (error) {
    console.error("[ContextPrism] LLM summarization failed, using extractive fallback:", error.message);
  }
  return fallbackSummary(text, maxTokens);
}

export async function mapReduceSummarize(
  chunks,
  { maxTokens = 400, apiKey = null, model = "deepseek-chat", enabled = true } = {},
) {
  if (!enabled || chunks.length === 0) return "";
  const text = chunks.map((chunk) => chunk.content).join("\n\n");
  if (!apiKey && !process.env.DEEPSEEK_API_KEY) return fallbackSummary(text, maxTokens);
  return summarizeText(text, { maxTokens, apiKey, model, enabled });
}

function fallbackSummary(text, maxTokens) {
  const [summary] = compressChunkToFit(text, maxTokens);
  return summary || "";
}
