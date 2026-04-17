from __future__ import annotations

import inspect
import os
import re
from typing import Any, Optional

from sarvamai import SarvamAI


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_ANSWER_MARKER_RE = re.compile(r"(?im)^\s*(final\s*answer|answer|response)\s*:\s*")


class SarvamLLM:
    def __init__(self, api_subscription_key: str, model: Optional[str] = None):
        self._client = SarvamAI(api_subscription_key=api_subscription_key)
        self._model = model or os.getenv("SARVAM_MODEL") or "sarvam-m"

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate a response from Sarvam using the official SDK."""

        messages = [
            {
                "role": "system",
                "content": (
                    "Return only the final customer reply. "
                    "No analysis, no reasoning, no <think> tags, no extra preface."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        chat = getattr(self._client, "chat", None)
        if chat is None:
            raise RuntimeError("SarvamAI client missing 'chat' attribute")

        completions = getattr(chat, "completions", None)
        if completions is None:
            raise RuntimeError("SarvamAI client missing 'chat.completions' attribute")

        # sarvamai==0.1.12 exposes chat.completions as a bound method that does NOT accept `model`.
        # Some other SDK shapes may expose chat.completions.create(...) and may accept `model`.
        call_fn = completions if callable(completions) else getattr(completions, "create", None)
        if not callable(call_fn):
            raise RuntimeError("Unsupported SarvamAI SDK: chat.completions is not callable")

        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            sig = inspect.signature(call_fn)
            if "model" in sig.parameters:
                kwargs["model"] = self._model
        except Exception:
            # If signature inspection fails, fall back to the most compatible call.
            pass

        resp: Any = call_fn(**kwargs)
        text = _extract_text(resp)
        return strip_think_tags(text)


def strip_think_tags(text: str) -> str:
    """Remove model 'thinking' content and return only the final answer text."""

    if not isinstance(text, str):
        return str(text).strip()

    cleaned = _THINK_BLOCK_RE.sub("", text)
    # In case the model returns unmatched tags.
    cleaned = cleaned.replace("</think>", "").replace("<think>", "")
    cleaned = cleaned.strip()

    # If the model includes an explicit marker, keep only what follows the LAST marker.
    matches = list(_ANSWER_MARKER_RE.finditer(cleaned))
    if matches:
        cleaned = cleaned[matches[-1].end() :].strip()

    return cleaned


def _extract_text(resp: Any) -> str:
    """Extract assistant text from various response shapes."""

    # OpenAI-like: resp.choices[0].message.content
    choices = getattr(resp, "choices", None)
    if choices and len(choices) > 0:
        choice0 = choices[0]
        msg = getattr(choice0, "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
        # Sometimes choice has .text
        text = getattr(choice0, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

    # Dict-like
    if isinstance(resp, dict):
        try:
            c = resp["choices"][0]
            if "message" in c and "content" in c["message"]:
                return str(c["message"]["content"]).strip()
            if "text" in c:
                return str(c["text"]).strip()
        except Exception:
            pass

    return str(resp).strip()
