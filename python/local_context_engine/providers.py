"""Provider adapters.

The engine itself is provider-agnostic. These helpers translate a
``ContextWindow`` into API payloads for specific providers. DeepSeek uses an
OpenAI-compatible chat completions format, so the payloads share the same
shape; DeepSeek's ``deepseek-reasoner`` model has stricter parameter rules and
is handled separately.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .models import ContextWindow


def to_openai_payload(
    window: ContextWindow,
    model: str = "gpt-4o-mini",
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert a context window to an OpenAI-compatible chat payload."""
    payload: Dict[str, Any] = {
        "model": model,
        "messages": window.as_messages(system_prompt=system_prompt),
        "temperature": temperature,
        "stream": stream,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    elif window.reserved_output_tokens:
        payload["max_tokens"] = window.reserved_output_tokens
    if extra_body:
        payload.update(extra_body)
    return payload


def to_deepseek_payload(
    window: ContextWindow,
    model: str = "deepseek-chat",
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert a context window to a DeepSeek API payload.

    DeepSeek is OpenAI-compatible. For ``deepseek-reasoner``, unsupported
    sampling parameters are omitted because the API rejects them.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": window.as_messages(system_prompt=system_prompt),
        "stream": stream,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    elif window.reserved_output_tokens:
        payload["max_tokens"] = window.reserved_output_tokens

    if "reasoner" not in model.lower():
        payload["temperature"] = temperature

    if extra_body:
        payload.update(extra_body)
    return payload


def to_deepseek_messages(
    window: ContextWindow,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Return just the message list in DeepSeek-compatible format."""
    return window.as_messages(system_prompt=system_prompt)


def call_deepseek(
    payload: Dict[str, Any],
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com/chat/completions",
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Send a DeepSeek-compatible payload using only the standard library.

    The API key is read from ``DEEPSEEK_API_KEY`` when not passed explicitly.
    This function is provided for integration; it is never called by the engine
    itself.
    """
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. Pass api_key= or set the environment variable."
        )
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API error {exc.code}: {detail}") from exc
