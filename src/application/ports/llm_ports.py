"""LLM bounded-context port definitions.

Port for calling LLM providers. Split from `all_ports.py` during P2
port classification (§9.8).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class LLMPort(ABC):
    """Interface for calling an LLM provider."""

    @abstractmethod
    async def call(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call the LLM with a prompt and return text."""
        raise NotImplementedError

    @abstractmethod
    async def call_with_retry(self, prompt: str, *, max_retries: int = 3) -> str:
        """Call the LLM with retry logic."""
        raise NotImplementedError


__all__ = [
    "LLMPort",
]
