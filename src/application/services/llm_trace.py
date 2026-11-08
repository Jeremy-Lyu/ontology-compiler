"""Shared helpers for LLM trace normalization.

This module centralizes lightweight trace fields used across bootstrap,
agent clients, and monitor wrappers so we keep prompt hashing and
call identification consistent.
"""
from __future__ import annotations

import hashlib
from typing import Any


def hash_text(text: str) -> str:
    """Return a stable short hash for prompt / message content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def messages_to_text(messages: list[dict[str, str]]) -> str:
    """Flatten chat messages into a deterministic trace string."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def extract_prompt_text(*args: Any, **kwargs: Any) -> str:
    """Best-effort extraction of prompt-like text from call args."""
    parts: list[str] = []
    for arg in args:
        if isinstance(arg, str):
            parts.append(arg)
        elif isinstance(arg, list):
            parts.append(str(arg))
    for key in ("prompt", "messages", "input", "text"):
        value = kwargs.get(key)
        if value is not None:
            parts.append(str(value))
    return "\n".join(parts)
