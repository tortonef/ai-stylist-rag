"""Minimal Ollama client (standard library only): structured JSON output, thinking disabled.

Same client as brand-voice-copywriter; only the output schema differs.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


class LLMError(RuntimeError):
    """Ollama is unreachable or returned output we cannot use."""


def parse_answer(text: str) -> str:
    """Parse the model's JSON answer into the answer text, or raise LLMError."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"Model output is not valid JSON ({e.msg}): {text[:200]!r}") from e
    if not isinstance(obj, dict) or not isinstance(obj.get("answer"), str):
        raise LLMError(f"Model output needs a string 'answer': {text[:200]!r}")
    return obj["answer"].strip()


def _request(url: str, body: dict | None, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat(messages: list[dict], cfg: dict) -> str:
    """Send one chat request and return the answer text."""
    o = cfg["ollama"]
    body = {
        "model": o["model"],
        "messages": messages,
        "stream": False,
        "think": False,
        "format": ANSWER_SCHEMA,
        "options": {"temperature": o["temperature"]},
    }
    try:
        data = _request(f"{o['host']}/api/chat", body, o["timeout_s"])
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        raise LLMError(f"Ollama returned HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"Cannot reach Ollama at {o['host']}: {e}") from e
    return parse_answer(data.get("message", {}).get("content", ""))


def unavailable_reason(cfg: dict) -> str | None:
    """None if Ollama is running and has the configured model, otherwise a short explanation."""
    o = cfg["ollama"]
    try:
        tags = _request(f"{o['host']}/api/tags", None, 2)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return f"Ollama is not running at {o['host']}."
    names = {m.get("name") for m in tags.get("models", [])}
    if o["model"] not in names and f"{o['model']}:latest" not in names:
        return f"Ollama is running but the model {o['model']} is not pulled (run: ollama pull {o['model']})."
    return None
