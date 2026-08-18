// Provider adapters (JS port).

export function toOpenAIPayload(window, { model = "gpt-4o-mini", systemPrompt = null, temperature = 0.7, maxTokens = null, stream = false, extraBody = null } = {}) {
  const payload = {
    model,
    messages: window.asMessages(systemPrompt),
    temperature,
    stream,
  };
  if (maxTokens !== null && maxTokens !== undefined) {
    payload.max_tokens = maxTokens;
  } else if (window.reservedOutputTokens) {
    payload.max_tokens = window.reservedOutputTokens;
  }
  if (extraBody) Object.assign(payload, extraBody);
  return payload;
}

export function toDeepSeekPayload(window, { model = "deepseek-chat", systemPrompt = null, temperature = 0.7, maxTokens = null, stream = false, extraBody = null } = {}) {
  const payload = {
    model,
    messages: window.asMessages(systemPrompt),
    stream,
  };
  if (maxTokens !== null && maxTokens !== undefined) {
    payload.max_tokens = maxTokens;
  } else if (window.reservedOutputTokens) {
    payload.max_tokens = window.reservedOutputTokens;
  }
  if (!String(model).toLowerCase().includes("reasoner")) {
    payload.temperature = temperature;
  }
  if (extraBody) Object.assign(payload, extraBody);
  return payload;
}

export function toDeepSeekMessages(window, systemPrompt = null) {
  return window.asMessages(systemPrompt);
}

export async function callDeepSeek(payload, { apiKey = null, baseUrl = "https://api.deepseek.com/chat/completions", timeoutMs = 60000 } = {}) {
  const key = apiKey || process.env.DEEPSEEK_API_KEY;
  if (!key) {
    throw new Error("DEEPSEEK_API_KEY is not set. Pass apiKey= or set the environment variable.");
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(baseUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${key}`,
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`DeepSeek API error ${response.status}: ${detail}`);
    }
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}
