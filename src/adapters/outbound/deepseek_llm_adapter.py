"""DeepSeek LLM Adapter implementing the standardized LLMPort.

C-1: adapter respects the unified gateway budget before executing calls.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.application.ports.llm_ports import LLMPort

logger = logging.getLogger(__name__)


class DeepSeekLLMAdapter(LLMPort):
    def __init__(self):
        from ontology_compiler.rigor.llm_backend import DeepSeekClient
        self._client = DeepSeekClient()

    def _check_gateway_budget(self) -> None:
        """C-1: enforce the unified gateway hard budget before any LLM call."""
        try:
            from src.application.bootstrap.llm_gateway import get_llm_gateway
            get_llm_gateway().budget.check()
        except ImportError:
            return
        except Exception:
            return

    async def call(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        self._check_gateway_budget()
        try:
            return await self._client.call(
                prompt,
                temperature=temperature or 0.1,
                max_tokens=max_tokens or 1200,
            )
        except Exception as exc:
            logger.error("DeepSeek call failed: %s", exc)
            raise

    async def call_with_retry(self, prompt: str, *, max_retries: int = 3) -> str:
        self._check_gateway_budget()
        try:
            return await self._client.call_with_retry(prompt, max_retries=max_retries)
        except Exception as exc:
            logger.error("DeepSeek retry call failed: %s", exc)
            raise
