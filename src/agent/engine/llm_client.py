"""Agent LLM Client — pluggable chat interface for the Agent loop.

Wraps the existing application-layer LLMPort with a chat-style interface
expected by the Agent engine. Supports both:
  - String-based prompt (for models that take a single text input)
  - Chat messages format (list[dict]) converted to prompt internally

The AgentLLMClient is intentionally decoupled from specific providers
(DeepSeek, Dify, OpenAI, etc.) — inject any LLMPort implementation.
"""
from __future__ import annotations

import logging
from typing import Any

from src.application.bootstrap.llm_gateway import get_llm_gateway
from src.application.services.llm_trace import hash_text

logger = logging.getLogger(__name__)


class AgentLLMClient:
    """LLM client adapter for the Agent engine.

    Wraps a :class:`~src.application.ports.llm_port.LLMPort` and provides
    a higher-level ``chat()`` interface that the Agent loop uses.

    Args:
        llm_port: Any concrete implementation of LLMPort.
        temperature: Sampling temperature (default 0.7 for agent reasoning).
        max_tokens: Token cap per response.
    """

    def __init__(
        self,
        llm_port: Any | None = None,  # LLMPort — typed loosely to avoid hard dep at import time
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        self._llm = llm_port or get_llm_gateway().get_primary_client()
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def chat(self, messages: list[dict[str, str]]) -> str:
        """Send a chat-format conversation to the LLM and return the reply.

        Formats the messages list as a structured prompt text compatible with
        the string-based LLMPort interface.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.

        Returns:
            The model's reply as a plain string.
        """
        prompt = self._messages_to_prompt(messages)
        prompt_hash = hash_text(prompt)
        logger.debug("AgentLLMClient.chat: prompt length=%d chars hash=%s", len(prompt), prompt_hash)
        gateway = get_llm_gateway()
        active_llm = self._llm or gateway.get_primary_client()
        response = await active_llm.call(
            prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        logger.debug("AgentLLMClient.chat: response length=%d chars hash=%s", len(response), prompt_hash)
        return response

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        """Convert chat messages to a single prompt string.

        Format::

            [System]
            <system content>

            [User]
            <user content>

            [Assistant]
            <assistant content>

            [Observation]
            <tool content>
        """
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            label = {
                "system": "[System]",
                "user": "[User]",
                "assistant": "[Assistant]",
                "tool": "[Observation]",
            }.get(role, f"[{role.capitalize()}]")
            parts.append(f"{label}\n{content}")
        return "\n\n".join(parts)
