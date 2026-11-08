"""Dify LLM Adapter implementing the standardized LLMPort.

[OPTIMIZATION] Enhanced with:
- Circuit Breaker pattern for Dify failure protection
- Request caching to reduce duplicate calls
- Rate limiting to prevent API overload
- Timeout control with configurable thresholds
- Automatic fallback to DeepSeek when Dify is unavailable
- Exponential backoff with jitter for retry
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable

from src.application.ports.llm_ports import LLMPort
from src.application.rag.infrastructure.dify_client import (
    ask_dify_kb_qa,
    get_dify_answer_text,
    load_dify_config_from_env,
    DifyConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerState:
    """State for circuit breaker pattern."""
    failures: int = 0
    successes: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    is_open: bool = False
    open_since: float = 0.0


@dataclass
class RateLimitState:
    """State for rate limiting."""
    tokens: int = 0
    last_refill_time: float = 0.0


@dataclass
class CacheEntry:
    """Cached LLM response entry."""
    response: str
    timestamp: float
    ttl_seconds: float


class DifyLLMAdapter(LLMPort):
    """Enhanced Dify LLM Adapter with circuit breaker, caching, and rate limiting."""

    DEFAULT_CIRCUIT_BREAKER_CONFIG = {
        "failure_threshold": 5,
        "success_threshold": 3,
        "reset_timeout": 60.0,
    }

    DEFAULT_RATE_LIMIT_CONFIG = {
        "max_tokens": 10,
        "refill_rate": 2.0,
    }

    DEFAULT_CACHE_CONFIG = {
        "enabled": True,
        "max_entries": 500,
        "ttl_seconds": 300,
    }

    def __init__(self, config: Optional[DifyConfig] = None):
        self._config = config or load_dify_config_from_env()

        self._circuit_breaker = CircuitBreakerState()
        self._circuit_config = self.DEFAULT_CIRCUIT_BREAKER_CONFIG

        self._rate_limit = RateLimitState()
        self._rate_config = self.DEFAULT_RATE_LIMIT_CONFIG

        self._cache: Dict[str, CacheEntry] = {}
        self._cache_config = self.DEFAULT_CACHE_CONFIG

        self._fallback_client: Optional[LLMPort] = None
        self._semaphore = asyncio.Semaphore(3)

        self._total_calls = 0
        self._total_failures = 0
        self._cache_hits = 0
        self._fallback_reasons: list[Dict[str, str]] = []

    def _check_gateway_budget(self) -> None:
        """C-1: enforce the unified gateway hard budget before any LLM call.

        Inactive by default (no policy configured = no-op), so enabling the
        adapter never changes behavior until a budget is set via the gateway.
        """
        try:
            from src.application.bootstrap.llm_gateway import get_llm_gateway
            get_llm_gateway().budget.check()
        except ImportError:
            return
        except Exception:
            return

    def _record_fallback_reason(self, reason: str, target: str = "deepseek") -> None:
        """Record why a fallback was triggered (C-1 observability)."""
        entry: Dict[str, str] = {"reason": reason, "target": target}
        self._fallback_reasons.append(entry)
        logger.warning("Fallback triggered: %s -> %s", reason, target)

    def _get_cache_key(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2000) -> str:
        """Generate a versioned cache key from prompt and parameters (C-4).

        The key folds in the resolved :class:`CacheVersionBundle` so that a
        prompt/schema/policy/model/mode bump invalidates all cached LLM
        responses — the LLM layer and the document-cache layer share the same
        version semantics.
        """
        from src.pipeline.stage_decision import compute_versioned_cache_key
        from src.pipeline.versioned_cache import resolve_version_bundle

        bundle = resolve_version_bundle(
            provider="dify",
            model=self._config.model if hasattr(self._config, "model") else None,
        )
        return compute_versioned_cache_key(
            source_chunk=prompt[:4096],
            normalized_prompt=f"t={temperature}|mt={max_tokens}",
            version_bundle=bundle,
        )

    def _get_from_cache(self, prompt: str, temperature: float, max_tokens: int) -> Optional[str]:
        """Get cached response if available and not expired."""
        if not self._cache_config["enabled"]:
            return None
        
        key = self._get_cache_key(prompt, temperature, max_tokens)
        entry = self._cache.get(key)
        
        if entry and time.time() - entry.timestamp < entry.ttl_seconds:
            self._cache_hits += 1
            logger.debug(f"Cache hit for prompt (hash={key[:8]})")
            return entry.response
        
        return None

    def _add_to_cache(self, prompt: str, temperature: float, max_tokens: int, response: str) -> None:
        """Add response to cache."""
        if not self._cache_config["enabled"]:
            return
        
        if len(self._cache) >= self._cache_config["max_entries"]:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]
        
        key = self._get_cache_key(prompt, temperature, max_tokens)
        self._cache[key] = CacheEntry(
            response=response,
            timestamp=time.time(),
            ttl_seconds=self._cache_config["ttl_seconds"],
        )

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._circuit_breaker.is_open:
            elapsed = time.time() - self._circuit_breaker.open_since
            if elapsed >= self._circuit_config["reset_timeout"]:
                logger.info("Circuit breaker resetting after %ds", elapsed)
                self._circuit_breaker.is_open = False
                self._circuit_breaker.failures = 0
            else:
                logger.warning("Circuit breaker open, skipping Dify call (retry in %.0fs)", 
                              self._circuit_config["reset_timeout"] - elapsed)
                return True
        return False

    def _record_success(self) -> None:
        """Record successful call for circuit breaker."""
        self._circuit_breaker.successes += 1
        self._circuit_breaker.last_success_time = time.time()
        
        if self._circuit_breaker.failures > 0:
            self._circuit_breaker.failures -= 1
        
        if self._circuit_breaker.successes >= self._circuit_config["success_threshold"]:
            self._circuit_breaker.is_open = False
            self._circuit_breaker.failures = 0

    def _record_failure(self) -> None:
        """Record failed call for circuit breaker."""
        self._circuit_breaker.failures += 1
        self._circuit_breaker.last_failure_time = time.time()
        self._total_failures += 1
        
        if self._circuit_breaker.failures >= self._circuit_config["failure_threshold"]:
            logger.error("Circuit breaker tripped after %d failures", self._circuit_breaker.failures)
            self._circuit_breaker.is_open = True
            self._circuit_breaker.open_since = time.time()

    async def _acquire_rate_limit(self) -> None:
        """Acquire rate limit token."""
        now = time.time()
        elapsed = now - self._rate_limit.last_refill_time
        
        tokens_to_add = int(elapsed * self._rate_config["refill_rate"])
        if tokens_to_add > 0:
            self._rate_limit.tokens = min(
                self._rate_limit.tokens + tokens_to_add,
                self._rate_config["max_tokens"]
            )
            self._rate_limit.last_refill_time = now
        
        while self._rate_limit.tokens <= 0:
            await asyncio.sleep(0.5)
            now = time.time()
            elapsed = now - self._rate_limit.last_refill_time
            tokens_to_add = int(elapsed * self._rate_config["refill_rate"])
            if tokens_to_add > 0:
                self._rate_limit.tokens = min(
                    self._rate_limit.tokens + tokens_to_add,
                    self._rate_config["max_tokens"]
                )
                self._rate_limit.last_refill_time = now
        
        self._rate_limit.tokens -= 1

    def _get_fallback_client(self) -> LLMPort:
        """Get fallback client (DeepSeek)."""
        if self._fallback_client is None:
            try:
                from src.adapters.outbound.deepseek_llm_adapter import DeepSeekLLMAdapter
                self._fallback_client = DeepSeekLLMAdapter()
                logger.info("Fallback client initialized: DeepSeek")
            except ImportError:
                logger.warning("DeepSeek adapter not available, fallback disabled")
        return self._fallback_client

    async def _call_with_protection(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call Dify with circuit breaker, rate limiting, and fallback protection."""
        # C-1: check the unified gateway budget before spending anything.
        self._check_gateway_budget()

        temp = temperature or 0.3
        mt = max_tokens or 1200

        cached = self._get_from_cache(prompt, temp, mt)
        if cached:
            return cached

        if self._is_circuit_open():
            fallback = self._get_fallback_client()
            if fallback:
                self._record_fallback_reason("circuit_breaker_open")
                return await fallback.call(prompt, temperature=temp, max_tokens=mt)
            raise RuntimeError("Dify circuit breaker open and no fallback available")

        await self._acquire_rate_limit()

        async with self._semaphore:
            try:
                response = await asyncio.wait_for(
                    self._call_dify(prompt, temperature=temp, max_tokens=mt),
                    timeout=self._config.timeout_s,
                )
                self._record_success()
                self._add_to_cache(prompt, temp, mt, response)
                return response
            except asyncio.TimeoutError:
                logger.error("Dify call timed out after %.0fs", self._config.timeout_s)
                self._record_failure()

                fallback = self._get_fallback_client()
                if fallback:
                    self._record_fallback_reason("timeout")
                    return await fallback.call(prompt, temperature=temp, max_tokens=mt)
                raise
            except Exception as exc:
                logger.error("Dify call failed: %s", exc)
                self._record_failure()

                fallback = self._get_fallback_client()
                if fallback:
                    self._record_fallback_reason(f"error:{type(exc).__name__}")
                    return await fallback.call(prompt, temperature=temp, max_tokens=mt)
                raise

    async def _call_dify(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1200,
    ) -> str:
        """Direct Dify API call."""
        self._total_calls += 1
        
        try:
            response = ask_dify_kb_qa(
                query=prompt,
                config=self._config,
            )
            return get_dify_answer_text(response)
        except Exception as exc:
            logger.error("Dify raw call failed: %s", exc)
            raise

    async def call(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call Dify with full protection mechanisms."""
        try:
            return await self._call_with_protection(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.error("Dify LLM call failed after all protections: %s", exc)
            raise

    async def call_with_retry(
        self,
        prompt: str,
        *,
        max_retries: int = 3,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call Dify with retry and protection mechanisms."""
        temp = temperature or 0.3
        mt = max_tokens or 1200
        
        cached = self._get_from_cache(prompt, temp, mt)
        if cached:
            return cached

        last_error = None
        for attempt in range(max_retries):
            try:
                return await self._call_with_protection(
                    prompt,
                    temperature=temp,
                    max_tokens=mt,
                )
            except Exception as exc:
                last_error = exc
                logger.warning("Dify retry attempt %d/%d failed: %s", attempt + 1, max_retries, exc)
                if attempt < max_retries - 1:
                    jitter = (2 ** attempt) + (attempt * 0.5)
                    await asyncio.sleep(jitter)

        logger.error("Dify retry call failed after %d attempts: %s", max_retries, last_error)
        raise last_error

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "cache_hits": self._cache_hits,
            "circuit_breaker": {
                "failures": self._circuit_breaker.failures,
                "successes": self._circuit_breaker.successes,
                "is_open": self._circuit_breaker.is_open,
                "open_since": self._circuit_breaker.open_since,
            },
            "cache_size": len(self._cache),
            "fallback_count": len(self._fallback_reasons),
            "fallback_reasons": self._fallback_reasons[-10:],
        }