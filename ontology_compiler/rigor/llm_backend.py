"""
RIGOR LLM Backend — Unified LLM client stack.

Consolidates BaseLLMClient (base_llm_client.py), DifyClient (dify_client.py),
DeepSeekClient (deepseek_client.py), and UnifiedLLMClient (llm_client.py)
into a single module.

Public API is identical to the four original modules.

Enhancements (2026-07-07):
- Circuit Breaker for Dify API protection
- Response caching to reduce redundant calls
- Rate limiting to prevent API overload
- Enhanced timeout control with configurable thresholds
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..config import get_config
from core.contracts import CacheVersionBundle
from core.errors import ConfigError, RetryPolicy, TransientInfraError, with_transient_retry_async

# Default endpoint; override via env or config if needed for proxies / on-prem.
_DEFAULT_DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1/chat/completions"


# ---------------------------------------------------------------------------
# Circuit Breaker — Prevents repeated calls to failing services
# ---------------------------------------------------------------------------

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker.
    
    P7 Tuning (2026-07-12): Adjusted defaults to be more resilient:
    - failure_threshold: increased from 5 to 10 (less sensitive to transient failures)
    - recovery_timeout_s: decreased from 60 to 30 (faster recovery)
    - half_open_attempts: increased from 2 to 3 (more chances to test recovery)
    - half_open_delay_s: added 2 second delay before HALF_OPEN attempts
    """
    failure_threshold: int = 10
    recovery_timeout_s: float = 30.0
    half_open_attempts: int = 3
    half_open_delay_s: float = 2.0


class CircuitBreaker:
    """Circuit breaker pattern implementation.

    States:
        CLOSED: Normal operation, calls pass through
        OPEN: Too many failures, calls are blocked
        HALF_OPEN: Recovery period, limited calls allowed to test health

    P7 Enhancement (2026-07-12): Added delay in HALF_OPEN state to give service
    time to recover before testing, preventing immediate re-tripping.
    """

    class TrippedError(Exception):
        """Raised when circuit breaker is in OPEN state."""

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self._state = "CLOSED"
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_attempts = 0
        self._logger = logging.getLogger("RIGOR.CircuitBreaker")

    def _transition_to_open(self):
        self._state = "OPEN"
        self._last_failure_time = time.time()
        self._failure_count = 0
        self._logger.warning("Circuit breaker OPEN - blocking calls for %ds",
                           int(self.config.recovery_timeout_s))

    def _transition_to_closed(self):
        self._state = "CLOSED"
        self._failure_count = 0
        self._half_open_attempts = 0
        self._logger.info("Circuit breaker CLOSED - normal operation resumed")

    def _transition_to_half_open(self):
        self._state = "HALF_OPEN"
        self._half_open_attempts = 0
        self._logger.info("Circuit breaker HALF_OPEN - testing recovery")

    def _check_state(self):
        """Check and possibly transition circuit breaker state."""
        if self._state == "OPEN":
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.config.recovery_timeout_s:
                self._transition_to_half_open()
            else:
                raise self.TrippedError(
                    f"Circuit breaker tripped, try again in {int(self.config.recovery_timeout_s - elapsed)}s"
                )
        return self._state

    def record_success(self):
        """Record a successful call."""
        self._check_state()
        if self._state == "HALF_OPEN":
            self._half_open_attempts += 1
            if self._half_open_attempts >= self.config.half_open_attempts:
                self._transition_to_closed()
        else:
            self._failure_count = 0

    def record_failure(self):
        """Record a failed call."""
        self._failure_count += 1
        if self._failure_count >= self.config.failure_threshold:
            self._transition_to_open()

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with circuit breaker protection.
        
        P7 Enhancement: Added delay in HALF_OPEN state to allow service recovery.
        """
        state = self._check_state()
        
        if state == "HALF_OPEN" and self._half_open_attempts >= self.config.half_open_attempts:
            raise self.TrippedError("Circuit breaker still testing")
        
        if state == "HALF_OPEN" and self._half_open_attempts == 0:
            await asyncio.sleep(self.config.half_open_delay_s)
            self._logger.debug(f"Waited {self.config.half_open_delay_s}s before first HALF_OPEN attempt")
        
        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except self.TrippedError:
            raise
        except Exception as e:
            self.record_failure()
            raise


# ---------------------------------------------------------------------------
# Response Cache — Reduces redundant LLM calls
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A cached response entry."""
    value: str
    timestamp: float
    ttl_s: float = 300.0

    def is_expired(self) -> bool:
        return time.time() - self.timestamp >= self.ttl_s


class LLMCache:
    """Simple in-memory cache for LLM responses.

    P1 / C-4 standardization: cache keys are built from the prompt plus a
    structured :class:`CacheVersionBundle` (prompt_version / schema_version /
    policy_version / model_family / extraction_mode). A version bump in any
    dimension invalidates all cached results — never silently reuse a stale
    response generated under different prompt / schema / policy / model / mode.

    For backward compatibility, *version_bundle* also accepts a plain string
    (legacy flat bundle). It is converted into a ``CacheVersionBundle`` with
    the string placed in ``model_family`` so the cache still keys on it.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_s: float = 300.0,
        version_bundle: CacheVersionBundle | str | None = None,
    ):
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._default_ttl_s = default_ttl_s
        # Normalise to CacheVersionBundle
        if version_bundle is None or version_bundle == "":
            self._version_bundle = CacheVersionBundle()
        elif isinstance(version_bundle, CacheVersionBundle):
            self._version_bundle = version_bundle
        else:
            # Legacy flat string — stash in model_family so it still keys.
            self._version_bundle = CacheVersionBundle(model_family=str(version_bundle))
        self._logger = logging.getLogger("RIGOR.LLMCache")

    def _get_key(self, prompt: str, **kwargs) -> str:
        """Generate a cache key from prompt and parameters.

        P1 / C-4: uses the unified ``CacheVersionBundle.to_fingerprint()``
        so every cache layer shares the same version semantics.
        """
        key_data = {
            "prompt": prompt,
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
        }
        if not self._version_bundle.is_empty():
            key_data["version_bundle"] = self._version_bundle.to_fingerprint()
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def get(self, prompt: str, **kwargs) -> str | None:
        """Get cached response if available and not expired."""
        key = self._get_key(prompt, **kwargs)
        entry = self._cache.get(key)
        if entry and not entry.is_expired():
            self._logger.debug("Cache hit for prompt (hash=%s)", key[:8])
            return entry.value
        if entry:
            self._logger.debug("Cache expired for prompt (hash=%s)", key[:8])
            del self._cache[key]
        return None

    def set(self, prompt: str, value: str, **kwargs) -> None:
        """Cache a response."""
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache.keys(),
                           key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]

        key = self._get_key(prompt, **kwargs)
        ttl = kwargs.get("cache_ttl_s", self._default_ttl_s)
        self._cache[key] = CacheEntry(value=value, timestamp=time.time(), ttl_s=ttl)
        self._logger.debug("Cache set for prompt (hash=%s, ttl=%ds)", key[:8], int(ttl))

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._logger.info("Cache cleared")

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


# ---------------------------------------------------------------------------
# Rate Limiter — Controls API call frequency
# ---------------------------------------------------------------------------

@dataclass
class RateLimitConfig:
    """Configuration for rate limiter."""
    max_calls: int = 10
    time_window_s: float = 60.0
    min_interval_s: float = 1.0


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        self._tokens = self.config.max_calls
        self._last_refill_time = time.time()
        self._logger = logging.getLogger("RIGOR.RateLimiter")

    def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill_time
        
        tokens_to_add = (elapsed / self.config.time_window_s) * self.config.max_calls
        self._tokens = min(self._tokens + tokens_to_add, self.config.max_calls)
        self._last_refill_time = now

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        self._refill_tokens()
        
        if self._tokens >= 1:
            self._tokens -= 1
            return
        
        wait_time = self.config.min_interval_s
        self._logger.debug("Rate limited, waiting %.1fs", wait_time)
        await asyncio.sleep(wait_time)
        
        self._refill_tokens()
        if self._tokens >= 1:
            self._tokens -= 1
            return
        
        raise RuntimeError("Rate limit exceeded")


# ---------------------------------------------------------------------------
# LLM Request Batcher — Merges multiple requests into batch calls
# ---------------------------------------------------------------------------

@dataclass
class BatchConfig:
    """Configuration for request batching."""
    max_batch_size: int = 10
    max_wait_ms: int = 500
    min_batch_size: int = 2
    enabled: bool = True


@dataclass
class BatchRequest:
    """A single request that can be batched."""
    prompt: str
    temperature: float | None = None
    max_tokens: int | None = None
    kwargs: dict = field(default_factory=dict)
    future: asyncio.Future = field(default_factory=lambda: asyncio.Future())


class LLMBatchClient:
    """LLM client with request batching support.

    Groups multiple small requests into batch calls to reduce API overhead.
    Uses a background task to collect requests and batch them periodically.

    Features:
    - Automatic request collection and batching
    - Configurable batch size and timeout
    - Transparent fallback to single calls when batching is disabled
    - Supports all LLM providers
    """

    def __init__(self, client: BaseLLMClient, config: BatchConfig | None = None):
        self.client = client
        self.config = config or BatchConfig()
        self._pending_requests: list[BatchRequest] = []
        self._batch_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger("RIGOR.LLMBatchClient")

    async def call(self, prompt: str, temperature: float | None = None, max_tokens: int | None = None, **kwargs) -> str:
        """Call LLM with optional batching."""
        if not self.config.enabled:
            return await self.client.call(prompt, temperature, max_tokens, **kwargs)

        request_kwargs = kwargs.copy()
        request_kwargs.pop("temperature", None)
        request_kwargs.pop("max_tokens", None)

        request = BatchRequest(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            kwargs=request_kwargs,
        )

        async with self._lock:
            self._pending_requests.append(request)
            self._start_batch_task()

        return await request.future

    def _start_batch_task(self):
        """Start the batch processing task if not already running."""
        if self._batch_task is None or self._batch_task.done():
            self._batch_task = asyncio.create_task(self._process_batches())

    async def _process_batches(self):
        """Background task that processes pending requests in batches."""
        while True:
            await asyncio.sleep(self.config.max_wait_ms / 1000.0)
            
            async with self._lock:
                if not self._pending_requests:
                    break
                
                batch = self._pending_requests[:self.config.max_batch_size]
                self._pending_requests = self._pending_requests[self.config.max_batch_size:]

            if len(batch) >= self.config.min_batch_size:
                await self._execute_batch(batch)
            else:
                await self._execute_individual(batch)

    async def _execute_batch(self, batch: list[BatchRequest]):
        """Execute a batch of requests."""
        self._logger.info(f"Executing batch of {len(batch)} requests")
        
        try:
            results = await self._batch_call(batch)
            
            for req, result in zip(batch, results):
                if not req.future.done():
                    req.future.set_result(result)
            
            self._logger.debug(f"Batch completed successfully: {len(results)} results")
        except Exception as e:
            self._logger.error(f"Batch execution failed: {e}")
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

    async def _execute_individual(self, batch: list[BatchRequest]):
        """Execute requests individually when batch size is below threshold."""
        for req in batch:
            try:
                result = await self.client.call(
                    req.prompt,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    **req.kwargs
                )
                if not req.future.done():
                    req.future.set_result(result)
            except Exception as e:
                if not req.future.done():
                    req.future.set_exception(e)

    async def _batch_call(self, batch: list[BatchRequest]) -> list[str]:
        """Make a single LLM call that processes multiple prompts."""
        combined_prompt = self._build_combined_prompt(batch)
        
        response = await self.client.call(
            combined_prompt,
            temperature=batch[0].temperature if batch[0].temperature else 0.1,
            max_tokens=max(req.max_tokens or 4000 for req in batch) * len(batch),
        )
        
        return self._parse_combined_response(response, len(batch))

    def _build_combined_prompt(self, batch: list[BatchRequest]) -> str:
        """Build a combined prompt for batch processing."""
        prompt_parts = []
        prompt_parts.append("""You are an ontology extraction expert. Process the following multiple prompts and return results in JSON format.
        
For each prompt, provide a separate JSON response. Return a JSON array with one element per prompt.

Format requirements:
- Return ONLY a JSON array
- Each element corresponds to the prompt at the same index
- Do NOT include any preamble or explanation
""")
        
        for i, req in enumerate(batch):
            prompt_parts.append(f"\n--- PROMPT {i + 1} ---\n")
            prompt_parts.append(req.prompt)
            prompt_parts.append(f"\n--- END PROMPT {i + 1} ---")
        
        prompt_parts.append("\n\nReturn JSON array:")
        
        return "\n".join(prompt_parts)

    def _parse_combined_response(self, response: str, expected_count: int) -> list[str]:
        """Parse the combined response into individual results."""
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return [json.dumps(item) for item in data[:expected_count]]
        except json.JSONDecodeError:
            pass
        
        return self._fallback_parse(response, expected_count)

    def _fallback_parse(self, response: str, expected_count: int) -> list[str]:
        """Fallback parsing when JSON is malformed."""
        results = []
        start = 0
        
        for _ in range(expected_count):
            start_idx = response.find("{", start)
            end_idx = response.find("}", start_idx)
            
            if start_idx == -1 or end_idx == -1:
                break
            
            results.append(response[start_idx:end_idx + 1])
            start = end_idx + 1
        
        while len(results) < expected_count:
            results.append("{}")
        
        return results

    async def call_with_retry(self, prompt: str, policy: RetryPolicy | None = None, **kwargs) -> str:
        """Call LLM with retry and optional batching."""
        if not self.config.enabled:
            return await self.client.call_with_retry(prompt, policy=policy, **kwargs)
        
        request_kwargs = kwargs.copy()
        temperature = request_kwargs.pop("temperature", None)
        max_tokens = request_kwargs.pop("max_tokens", None)
        
        request = BatchRequest(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            kwargs=request_kwargs,
        )

        async with self._lock:
            self._pending_requests.append(request)
            self._start_batch_task()

        return await request.future

    async def close(self):
        """Shutdown the batch client."""
        if self._batch_task and not self._batch_task.done():
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass


_INTERNAL_DOMAINS = {"internal.example.com", "corp.example.com", "example.com"}

def _should_use_proxy(url: str) -> bool:
    """Determine if proxy should be used based on URL domain.

    LLM 调用（DIFY / DeepSeek）统一不走代理：
    - DIFY 是 内网服务，无需代理
    - DeepSeek 外部 API 走 企业代理会被 governor 拦截
    - 如需启用代理，设置环境变量 LLM_ENABLE_PROXY=true
    """
    if os.environ.get("LLM_ENABLE_PROXY", "").lower() == "true":
        import urllib.parse
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.split(':')[0].lower()
            for internal_domain in _INTERNAL_DOMAINS:
                if domain.endswith(internal_domain):
                    return False
            return True
        except:
            return False
    return False

def _httpx_async_client(timeout: float | int, url: str = ""):
    """Create httpx AsyncClient with proper proxy handling."""
    import httpx

    use_proxy = _should_use_proxy(url)
    http_proxy = os.environ.get("HTTP_PROXY", "") if use_proxy else ""
    https_proxy = os.environ.get("HTTPS_PROXY", "") if use_proxy else ""
    
    if http_proxy or https_proxy:
        transport = httpx.AsyncHTTPTransport(
            proxy=httpx.Proxy(
                url=http_proxy or https_proxy,
            )
        )
        return httpx.AsyncClient(timeout=timeout, transport=transport, trust_env=False)
    else:
        return httpx.AsyncClient(timeout=timeout, trust_env=False)


# ---------------------------------------------------------------------------
# BaseLLMClient  (was rigor/base_llm_client.py)
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """
    Abstract base class for LLM API clients.

    Provides shared infrastructure:
    - Config and logger initialisation
    - LLM call usage tracking
    - Retry loop with exponential backoff + jitter
    - Response validation and error formatting
    """

    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(self._logger_name)
        self._check_api_key()

    @property
    @abstractmethod
    def _logger_name(self) -> str:
        """Logger name used by this client."""

    @property
    @abstractmethod
    def api_key(self) -> str | None:
        """Return the configured API key (may be None/empty)."""

    @abstractmethod
    async def call(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Issue a single API call and return the response text."""

    def _check_api_key(self) -> None:
        if not self.api_key:
            raise ConfigError(
                f"{self.__class__.__name__} API key not configured. "
                "Set the corresponding environment variable before starting."
            )

    def _track_llm_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """R7: Track LLM usage with token counts for cost analysis.

        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
        """
        try:
            usage = getattr(self.config, "runtime_llm_usage", None)
            if not isinstance(usage, dict):
                return

            # Update counters
            usage["llm_calls"] = float(usage.get("llm_calls", 0.0)) + 1.0
            usage["prompt_tokens"] = int(usage.get("prompt_tokens", 0)) + prompt_tokens
            usage["completion_tokens"] = int(usage.get("completion_tokens", 0)) + completion_tokens
            usage["total_tokens"] = int(usage.get("total_tokens", 0)) + prompt_tokens + completion_tokens

            # Cost estimation (per 1K tokens, in USD)
            # DeepSeek: $0.07/1K prompt, $0.07/1K completion (deepseek-chat)
            # Dify: Variable, use placeholder
            prompt_cost = prompt_tokens * 0.00007  # $0.07 per 1K
            completion_cost = completion_tokens * 0.00007  # $0.07 per 1K

            usage["estimated_cost"] = float(usage.get("estimated_cost", 0.0)) + prompt_cost + completion_cost

            # Log at debug level
            self.logger.debug(
                f"R7 LLM usage: +{prompt_tokens} prompt / +{completion_tokens} completion tokens, "
                f"total={usage['total_tokens']}, est_cost=${usage['estimated_cost']:.4f}"
            )
        except Exception:
            pass

    def _is_retryable_error(self, exc: Exception) -> bool:
        return isinstance(exc, TransientInfraError)

    async def call_with_retry(
        self,
        prompt: str,
        policy: RetryPolicy | None = None,
        **kwargs: Any,
    ) -> str:
        logger = self.logger
        client_name = self.__class__.__name__

        async def _once():
            return await self.call(prompt, **kwargs)

        try:
            return await with_transient_retry_async(_once, policy=policy)
        except TransientInfraError:
            logger.error("%s call failed after all retry attempts", client_name)
            raise
        except Exception as exc:
            logger.error("%s call failed: %s", client_name, exc)
            raise

    def validate_response(self, response: str) -> bool:
        if not response:
            return False
        try:
            json.loads(response)
            return True
        except json.JSONDecodeError:
            return len(response) > 0

    def handle_error(self, error: Exception) -> str:
        if isinstance(error, asyncio.TimeoutError):
            return f"{self.__class__.__name__} request timed out"
        if isinstance(error, ValueError):
            return "Invalid API configuration"
        return f"{self.__class__.__name__} error: {error}"


# ---------------------------------------------------------------------------
# DifyClient  (was rigor/dify_client.py)
# ---------------------------------------------------------------------------

class DifyAPIError(Exception):
    """Raised when Dify API returns an error response."""

    def __init__(self, message: str, *, status_code: int | None = None, response_text: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class DifyClient(BaseLLMClient):
    """Communicates with DIFY API."""

    @property
    def _logger_name(self) -> str:
        return "RIGOR.DIFY"

    @property
    def api_key(self) -> str | None:
        return self._api_key

    def _is_retryable_error(self, exc: Exception) -> bool:
        if isinstance(exc, DifyAPIError):
            retryable_statuses = {429, 502, 503, 504}
            status = getattr(exc, "status_code", None)
            return (status in retryable_statuses) or (status is None)
        import httpx
        retryable = (
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.PoolTimeout,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.TimeoutException,
            asyncio.TimeoutError,
            httpx.HTTPError,
            ConnectionError,
            OSError,
        )
        return isinstance(exc, retryable)

    def __init__(self, config=None, api_key_override: str = "", model_override: str = "", 
                 max_tokens_override: int = 0, temperature_override: float = None, 
                 timeout_override: int = 0, is_pro: bool = False):
        self.config = config or get_config()
        self._api_key = api_key_override or self.config.dify_api_key
        super().__init__(config)
        
        self.base_url = self.config.dify_base_url
        self.username = self.config.dify_username
        self.model = model_override or self.config.dify_model
        self.max_tokens = max_tokens_override or self.config.dify_max_tokens
        self.temperature = temperature_override if temperature_override is not None else self.config.dify_temperature
        self.app_type = self.config.dify_app_type
        self.timeout = timeout_override or int(getattr(self.config, "dify_timeout_seconds", 300) or 300)
        self._is_pro = is_pro
        
        self._circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=getattr(self.config, "dify_circuit_failure_threshold", 5),
                recovery_timeout_s=getattr(self.config, "dify_circuit_recovery_s", 60.0),
                half_open_attempts=getattr(self.config, "dify_circuit_half_open_attempts", 2),
            )
        )
        
        # 2026-08-18: unified LLM_CACHE_* env vars take precedence over the
        # legacy per-provider defaults. This way a single env bump changes
        # cache behavior for all providers (Dify, DeepSeek, OpenAI).
        _unified_ttl_s = float(os.environ.get("LLM_CACHE_TTL_S", "0") or 0)
        _unified_max_size = int(os.environ.get("LLM_CACHE_MAX_SIZE", "0") or 0)
        _cache_ttl_s = _unified_ttl_s or getattr(self.config, "dify_cache_ttl_s", 300.0)
        _cache_max_size = _unified_max_size or getattr(self.config, "dify_cache_max_size", 1000)

        # P1 / C-4: build a structured CacheVersionBundle so the LLM cache
        # shares the same 5-dimension version semantics as the document cache
        # and stage decision logic. Legacy flat-string bundle is still
        # accepted (stored in model_family) for backward compatibility.
        _legacy_bundle = (
            getattr(self.config, "dify_cache_version_bundle", None)
            or os.getenv("LLM_CACHE_VERSION_BUNDLE", "")
        )
        if _legacy_bundle:
            cache_bundle: CacheVersionBundle | str = _legacy_bundle
        else:
            cache_bundle = CacheVersionBundle(
                prompt_version=os.getenv("PROMPT_VERSION", ""),
                schema_version=os.getenv("SCHEMA_VERSION", ""),
                policy_version=os.getenv("POLICY_VERSION", os.getenv("SKILL_VERSION", "")),
                model_family=f"dify:{self.model}",
                extraction_mode=os.getenv("PIPELINE_EXECUTION_MODE", "production"),
            )
        self._cache = LLMCache(
            max_size=_cache_max_size,
            default_ttl_s=_cache_ttl_s,
            version_bundle=cache_bundle,
        )
        self.logger.info(
            "DIFY LLM cache configured — max_size=%d ttl_s=%.0f bundle=%s",
            _cache_max_size, _cache_ttl_s,
            cache_bundle.to_fingerprint() if isinstance(cache_bundle, CacheVersionBundle) else cache_bundle,
        )

        self._rate_limiter = RateLimiter(
            RateLimitConfig(
                max_calls=getattr(self.config, "dify_rate_limit_max_calls", 5),
                time_window_s=getattr(self.config, "dify_rate_limit_window_s", 60.0),
                min_interval_s=getattr(self.config, "dify_min_call_interval_s", 2.0),
            )
        )
        
        # HARD CONSTRAINT (from archived dify_client.py): large JSON payloads sent via
        # httpx streaming sometimes get their HTTP/1.1 connection reset mid-stream by
        # the corporate Dify reverse proxy. urllib.request sends the entire body
        # in a single-shot (no streaming), which is 100 % unaffected.
        self._LARGE_PAYLOAD_THRESHOLD_BYTES = int(os.environ.get(
            "DIFY_LARGE_PAYLOAD_THRESHOLD_BYTES", "4096"
        ))

        if is_pro:
            self.logger.info(f"DifyClient PRO initialized - Model: {self.model}, API Key: {'configured' if self._api_key else 'not configured'}")
            self.logger.info(f"DifyClient PRO protection: CircuitBreaker(failure_threshold={self._circuit_breaker.config.failure_threshold}), "
                            f"RateLimiter(max_calls={self._rate_limiter.config.max_calls}/min), Cache(max_size={self._cache._max_size}), "
                            f"LargePayloadThreshold={self._LARGE_PAYLOAD_THRESHOLD_BYTES}B")

    # ------------------------------------------------------------------
    # Payload helpers — shared between httpx streaming and urllib paths
    # ------------------------------------------------------------------

    def _build_dify_payload(self, prompt: str) -> tuple[str, dict]:
        """Build (url, payload_dict) for the configured app_type.

        For the urllib large-payload path we use response_mode=blocking so the
        server returns a single JSON document (no SSE streaming). For the httpx
        small-payload path the caller overrides this to streaming later.
        """
        if self.app_type == "chat":
            url = f"{self.base_url}/chat-messages"
            payload = {
                "inputs": {},
                "query": prompt,
                "response_mode": "blocking",
                "user": self.username,
                "auto_generate_name": False,
            }
        elif self.app_type == "workflow":
            url = f"{self.base_url}/workflows/run"
            payload = {
                "inputs": {"query": prompt},
                "response_mode": "blocking",
                "user": self.username,
            }
        else:
            # completion-messages fallback
            url = f"{self.base_url}/completion-messages"
            payload = {
                "inputs": {},
                "query": prompt,
                "response_mode": "blocking",
                "user": self.username,
            }
        return url, payload

    def _dify_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Connection": "close",
        }

    @staticmethod
    def _is_transient_dify_error(status_code: int, err_text: str) -> bool:
        if status_code in {429, 502, 503, 504}:
            return True
        lowered = (err_text or "").lower()
        return any(token in lowered for token in ("no_tokens", "no available tokens", "server_error"))

    @staticmethod
    def _extract_text_from_outputs(outputs) -> Optional[str]:
        if isinstance(outputs, str) and outputs.strip():
            return outputs.strip()
        if isinstance(outputs, dict):
            for k in ("answer", "text", "result", "output"):
                v = outputs.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            for v in outputs.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return None

    def _get_dify_answer_text(self, resp_obj) -> str:
        """Extract answer text from a parsed Dify JSON response.

        Mirrors the robust parsing chain in archived dify_client.get_dify_answer_text:
        answer → output → message.content → data.answer → data.outputs → metadata.
        """
        import json as _json

        if not isinstance(resp_obj, dict):
            raise DifyAPIError(f"dify response is not a dict, got {type(resp_obj).__name__}")

        def _first_nonempty(*vals):
            for v in vals:
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return None

        def _coerce(v):
            if isinstance(v, str):
                return v.strip() if v.strip() else None
            if isinstance(v, list):
                parts = []
                for item in v:
                    if isinstance(item, str) and item.strip():
                        parts.append(item.strip())
                    elif isinstance(item, dict):
                        t = _first_nonempty(item.get("text"), item.get("content"), item.get("answer"))
                        if t:
                            parts.append(t)
                joined = "\n".join(parts).strip()
                return joined if joined else None
            if isinstance(v, dict):
                return _first_nonempty(v.get("text"), v.get("content"), v.get("answer"))
            return None

        # Common chat-messages shape
        ans = _coerce(resp_obj.get("answer")) or _first_nonempty(resp_obj.get("output"), resp_obj.get("text"))
        if ans:
            return ans

        msg = resp_obj.get("message")
        if isinstance(msg, dict):
            ans = _coerce(msg.get("content")) or _coerce(msg.get("answer")) or _first_nonempty(msg.get("text"))
            if ans:
                return ans

        data = resp_obj.get("data")
        if isinstance(data, dict):
            ans = _coerce(data.get("answer")) or _first_nonempty(data.get("output"), data.get("text"))
            if ans:
                return ans
            outputs = data.get("outputs")
            if isinstance(outputs, dict):
                ans = self._extract_text_from_outputs(outputs)
                if ans:
                    return ans
            if isinstance(outputs, str):
                return outputs.strip()

        # Event-like shape (workflow SSE wrapped as blocking JSON)
        ev = resp_obj.get("event")
        if isinstance(ev, str) and ev in {"workflow_finished", "node_finished", "message"}:
            out = resp_obj.get("outputs")
            ans = self._extract_text_from_outputs(out)
            if ans:
                return ans
            data_obj = resp_obj.get("data")
            if isinstance(data_obj, dict):
                ans = self._extract_text_from_outputs(data_obj.get("outputs"))
                if ans:
                    return ans

        meta = resp_obj.get("metadata")
        if isinstance(meta, dict):
            ans = _first_nonempty(meta.get("answer"), meta.get("output"), meta.get("text"), meta.get("result"))
            if ans:
                return ans
            ann = meta.get("annotation_reply")
            if isinstance(ann, dict):
                ans = _coerce(ann.get("content")) or _coerce(ann.get("answer")) or _coerce(ann.get("text"))
                if ans:
                    return ans

        top_keys = sorted([k for k in resp_obj.keys() if isinstance(k, str)])
        data_keys = sorted([k for k in data.keys()]) if isinstance(data, dict) else []
        meta_keys = sorted([k for k in meta.keys()]) if isinstance(meta, dict) else []
        ans_val = resp_obj.get("answer")
        raise DifyAPIError(
            "dify response missing answer; "
            f"top_keys={top_keys}; data_keys={data_keys}; meta_keys={meta_keys}; "
            f"answer_type={type(ans_val).__name__}; answer_len={len(ans_val) if isinstance(ans_val, str) else None}"
        )

    # ------------------------------------------------------------------
    # Large-payload transport: urllib.request (single-shot, no streaming)
    # ------------------------------------------------------------------

    def _call_dify_urllib_blocking_sync(
        self,
        *,
        url: str,
        payload_bytes: bytes,
        headers: dict,
    ) -> str:
        """Synchronous (blocking) HTTP POST via urllib.request.

        Must be wrapped with asyncio.to_thread() at the call site so it does
        not block the asyncio event loop. This intentionally does NOT use
        httpx / aiohttp / requests — the corporate reverse proxy has
        been observed to reset HTTP/1.1 connections mid-stream when httpx's
        HTTP/2 + keep-alive pool reuses an idle server-side closed socket.
        urllib.request sends the entire body in one shot and is unaffected.
        """
        import time as _time
        import json as _json
        import urllib.parse
        import urllib.request
        import urllib.error

        start = _time.time()
        parsed_url = urllib.parse.urlparse(url)

        # Proxy handling aligned with _should_use_proxy: DIFY is an internal
        # (*.internal.example.com) service — do NOT use the corporate HTTP proxy.
        use_proxy = _should_use_proxy(url)
        if use_proxy:
            # Let urllib pick up HTTP_PROXY / HTTPS_PROXY from the environment.
            proxy_handler = urllib.request.ProxyHandler()
        else:
            # Explicitly disable proxy for internal DIFY endpoint.
            proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy_handler)

        max_attempts = 3
        last_exc: Optional[BaseException] = None
        resp_body_raw: Optional[bytes] = None
        resp_status: Optional[int] = None
        resp_headers: Optional[dict] = None

        for attempt in range(1, max_attempts + 1):
            try:
                req = urllib.request.Request(url, data=payload_bytes, headers=headers, method="POST")
                with opener.open(req, timeout=self.timeout) as _resp:
                    resp_status = getattr(_resp, "status", None) or _resp.getcode()
                    resp_headers = dict(_resp.headers.items())
                    resp_body_raw = _resp.read()
                break  # success
            except urllib.error.HTTPError as exc:
                last_exc = exc
                try:
                    resp_status = exc.code
                    resp_body_raw = exc.read()
                    resp_headers = dict(exc.headers.items()) if exc.headers else {}
                except Exception:
                    pass
                break  # HTTP-level errors do not retry
            except (
                urllib.error.URLError,
                ConnectionResetError,
                BrokenPipeError,
                TimeoutError,
                OSError,
            ) as exc:
                last_exc = exc
                if attempt < max_attempts:
                    wait_s = 1.5 * attempt
                    self.logger.warning(
                        "dify urllib transient attempt=%s/%s err=%s — retrying in %ss",
                        attempt, max_attempts, exc, wait_s,
                    )
                    _time.sleep(wait_s)
                    continue
                raise TransientInfraError(
                    f"dify urllib transient error after {max_attempts} attempts: {exc}"
                ) from exc
            except TransientInfraError:
                raise
            except Exception as exc:
                last_exc = exc
                break

        if resp_status is None:
            raise DifyAPIError(f"dify urllib request failed (no response): {last_exc}")

        text = (resp_body_raw or b"").decode("utf-8", errors="replace")
        total_ms = (_time.time() - start) * 1000

        if resp_status >= 400:
            err_msg = text[:500]
            # try to extract a structured error message
            try:
                err_obj = _json.loads(text)
                if isinstance(err_obj, dict):
                    err_dict = err_obj.get("error")
                    if isinstance(err_dict, dict):
                        m = err_dict.get("message")
                        if isinstance(m, str) and m.strip():
                            err_msg = m.strip()
            except Exception:
                pass
            self.logger.error(
                "DIFY urllib HTTP error - status=%s elapsed=%.0fms body=%s",
                resp_status, total_ms, err_msg,
            )
            if self._is_transient_dify_error(resp_status, err_msg):
                raise TransientInfraError(f"DIFY urllib transient HTTP {resp_status}: {err_msg}")
            raise DifyAPIError(
                f"DIFY API error HTTP {resp_status}: {err_msg}",
                status_code=resp_status,
                response_text=err_msg,
            )

        # If blocking response came back as SSE content-type, parse line by line.
        ctype = ""
        if resp_headers:
            for hk, hv in resp_headers.items():
                if hk.lower() == "content-type" and isinstance(hv, str):
                    ctype = hv.lower()
                    break
        if "text/event-stream" in ctype:
            parsed = self._parse_sse_lines_sync(text.splitlines())
            answer = self._get_dify_answer_text(parsed)
        else:
            try:
                data_obj = _json.loads(text)
            except Exception as exc:
                raise DifyAPIError(f"dify urllib invalid json response: {exc}") from exc

            # Some Dify deployments return an event-like JSON even in blocking
            # mode: event present but answer missing. Re-parse as SSE shape.
            if isinstance(data_obj, dict):
                a = data_obj.get("answer")
                ev = data_obj.get("event")
                if isinstance(ev, str) and ev in {"workflow_started", "node_started", "node_finished", "workflow_finished", "message"} and (not isinstance(a, str) or not a.strip()):
                    self.logger.info("dify urllib blocking response was event-shape, re-parsing as SSE-like chain")
                    # The blocking response is a single event; try data.outputs
                    pass
            answer = self._get_dify_answer_text(data_obj)

        self.logger.info(
            "DIFY urllib call OK - url_host=%s prompt_payload=%dB response=%dchars elapsed=%.0fms",
            parsed_url.hostname, len(payload_bytes), len(answer), total_ms,
        )
        return answer

    @staticmethod
    def _parse_sse_lines_sync(lines: list) -> dict:
        """Parse Dify SSE line list (synchronous) and return the normalized dict.

        For DIFY chat-messages with response_mode=streaming, each SSE chunk
        carries an INCREMENTAL ``answer`` substring (NOT the full text) — we
        must concatenate them. For workflow / blocking-fallback events the
        answer is emitted once (as ``answer`` / ``text`` / ``outputs.*``).
        Both strategies are combined below:

        1. First try: accumulate incremental ``answer`` tokens (chat streaming).
        2. If nothing accumulated: fall back to single-event extraction
           (workflow_finished / outputs / metadata shapes — matches archived
           dify_client._parse_sse_answer_lines).
        """
        import json as _json

        answer_chunks: list[str] = []
        single_answer: Optional[str] = None
        last_event: dict = {}
        for raw in lines:
            line = (raw or "").strip()
            if not line or not line.startswith("data:"):
                continue
            data_part = line[len("data:"):].strip()
            if not data_part or data_part == "[DONE]":
                continue
            try:
                evt = _json.loads(data_part)
            except Exception:
                continue
            if not isinstance(evt, dict):
                continue
            last_event = evt

            # Chat streaming: incremental "answer" deltas
            inc = evt.get("answer")
            if isinstance(inc, str) and inc:
                answer_chunks.append(inc)
                continue

            # Workflow / single-event extraction (only when no deltas seen)
            if not answer_chunks and single_answer is None:
                t = evt.get("text")
                if isinstance(t, str) and t.strip():
                    single_answer = t.strip()
                    continue
                out = evt.get("outputs")
                extracted = DifyClient._extract_text_from_outputs(out)
                if extracted:
                    single_answer = extracted
                    continue
                data_obj = evt.get("data")
                if isinstance(data_obj, dict):
                    extracted = DifyClient._extract_text_from_outputs(data_obj.get("outputs"))
                    if extracted:
                        single_answer = extracted
                        continue

        answer = ("".join(answer_chunks) if answer_chunks else single_answer)
        if answer:
            out = dict(last_event)
            out["answer"] = answer
            return out
        keys = sorted([k for k in last_event.keys() if isinstance(k, str)]) if isinstance(last_event, dict) else []
        raise DifyAPIError(f"dify sse response missing answer; last_event_keys={keys}")

    async def call(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs,
    ) -> str:
        if not self.api_key:
            self.logger.error("DIFY API key not configured")
            raise ValueError("DIFY_API_KEY environment variable not set")

        use_cache = kwargs.get("use_cache", True)

        if use_cache:
            cached_response = self._cache.get(prompt, temperature=temperature, max_tokens=max_tokens)
            if cached_response:
                self.logger.info(f"DIFY API cache hit - Prompt: {len(prompt)} chars, Response: {len(cached_response)} characters")
                return cached_response

        await self._rate_limiter.acquire()

        # Build payload once, measure its size, then route the transport.
        # _build_dify_payload defaults response_mode to "blocking" (for the
        # urllib path). The httpx streaming branch overrides it later.
        import json as _json_base
        import urllib.parse as _urlparse
        url, payload_dict = self._build_dify_payload(prompt)
        payload_bytes = _json_base.dumps(payload_dict, ensure_ascii=False).encode("utf-8")
        payload_size = len(payload_bytes)
        parsed_url = _urlparse.urlparse(url)
        is_large_payload = payload_size >= self._LARGE_PAYLOAD_THRESHOLD_BYTES
        transport = "urllib(blocking)" if is_large_payload else "httpx(streaming)"
        self.logger.info(
            "DIFY API call start - host=%s prompt=%dchars payload=%dB threshold=%dB transport=%s",
            parsed_url.hostname, len(prompt), payload_size,
            self._LARGE_PAYLOAD_THRESHOLD_BYTES, transport,
        )

        async def _make_api_call() -> str:
            import time as _time
            # Import httpx eagerly here (even for the urllib path) so the
            # `except httpx.*` handlers below never see an unbound name.
            import httpx  # noqa: F401  (used by except clauses)

            start_time = _time.time()
            answer: str
            try:
                if is_large_payload:
                    # ------------------------------------------------------------------
                    # LARGE PAYLOAD path: urllib.request (single-shot blocking)
                    # Wrapped in asyncio.to_thread so we do NOT block the loop.
                    # This avoids the corporate-proxy HTTP/2 connection-reset bug.
                    # ------------------------------------------------------------------
                    headers = self._dify_headers()
                    answer = await asyncio.to_thread(
                        self._call_dify_urllib_blocking_sync,
                        url=url,
                        payload_bytes=payload_bytes,
                        headers=headers,
                    )
                else:
                    # ------------------------------------------------------------------
                    # SMALL PAYLOAD path: existing httpx async streaming.
                    # For small JSON bodies the proxy never resets mid-stream;
                    # streaming keeps the connection alive and avoids read timeouts
                    # on long generations.
                    # ------------------------------------------------------------------
                    import httpx

                    # Override response_mode to streaming for the small-payload path.
                    stream_payload = dict(payload_dict)
                    stream_payload["response_mode"] = "streaming"
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    }
                    async with _httpx_async_client(self.timeout, url) as client:
                        connect_start = _time.time()
                        async with client.stream("POST", url, headers=headers, json=stream_payload) as response:
                            connect_end = _time.time()
                            response.raise_for_status()
                            chunk_answer = ""
                            chunk_count = 0
                            async for chunk in response.aiter_lines():
                                if chunk.startswith("data: "):
                                    data_str = chunk[6:]
                                    if data_str.strip() == "[DONE]":
                                        break
                                    try:
                                        data = _json_base.loads(data_str)
                                    except _json_base.JSONDecodeError:
                                        continue
                                    content = data.get("answer", "")
                                    if isinstance(content, str) and content:
                                        chunk_answer += content
                                        chunk_count += 1
                                        if chunk_count % 10 == 0:
                                            elapsed_ms = (_time.time() - connect_start) * 1000
                                            self.logger.debug(
                                                "DIFY streaming: chunks=%d received=%dchars elapsed=%.0fms",
                                                chunk_count, len(chunk_answer), elapsed_ms,
                                            )
                            answer = chunk_answer
                            if not answer:
                                # Some Dify endpoints tuck text into data.outputs
                                # even in the last SSE chunk. Call the unified parser
                                # on the aggregated events shape by running it on
                                # last-chunk data if present; otherwise the unified
                                # empty-answer handler raises consistently below.
                                self.logger.warning("DIFY httpx streaming answer empty, checking outputs")
                                # If we have absolutely nothing, let the common
                                # post-call block below raise DifyAPIError.
                    total_ms = (_time.time() - start_time) * 1000
                    self.logger.info(
                        "DIFY httpx streaming done - response=%dchars chunks=%d elapsed=%.0fms",
                        len(answer), chunk_count, total_ms,
                    )

                # Common post-call: validate answer, track usage.
                if not answer or not isinstance(answer, str) or not answer.strip():
                    raise DifyAPIError(
                        f"DIFY returned empty answer from {transport}; "
                        f"payload={payload_size}B prompt_len={len(prompt)}"
                    )

                total_time_ms = (_time.time() - start_time) * 1000
                prompt_tokens = len(prompt) // 4
                completion_tokens = len(answer) // 4
                self._track_llm_usage(prompt_tokens, completion_tokens)
                self.logger.info(
                    "DIFY API call OK - transport=%s response=%dchars elapsed=%.0fms "
                    "prompt_tokens=%d completion_tokens=%d",
                    transport, len(answer), total_time_ms, prompt_tokens, completion_tokens,
                )
                return answer
            except httpx.HTTPStatusError as e:
                status = int(e.response.status_code)
                total_time_ms = (_time.time() - start_time) * 1000
                err_text = (e.response.text or "")[:500]
                self.logger.error(
                    "DIFY httpx HTTP error - status=%s elapsed=%.0fms body=%s",
                    status, total_time_ms, err_text,
                )
                if self._is_transient_dify_error(status, err_text):
                    raise TransientInfraError(f"DIFY httpx transient HTTP {status}: {err_text}") from e
                raise DifyAPIError(
                    f"DIFY API error HTTP {status}: {err_text}",
                    status_code=status,
                    response_text=err_text,
                ) from e
            except (asyncio.TimeoutError, httpx.TimeoutException, httpx.ReadError) as e:
                total_time_ms = (_time.time() - start_time) * 1000
                self.logger.error(
                    "DIFY httpx read/timeout error - elapsed=%.0fms err=%s: %s",
                    total_time_ms, type(e).__name__, e,
                )
                raise TransientInfraError(f"DIFY httpx read/timeout error: {e}") from e
            except (TransientInfraError, DifyAPIError):
                raise
            except Exception as e:
                total_time_ms = (_time.time() - start_time) * 1000
                self.logger.error(
                    "DIFY API call failed - transport=%s elapsed=%.0fms err=%s: %r",
                    transport, total_time_ms, type(e).__name__, e, exc_info=True,
                )
                raise DifyAPIError(f"DIFY API call failed ({transport}): {type(e).__name__}: {e!r}") from e

        try:
            answer = await self._circuit_breaker.execute(_make_api_call)
        except CircuitBreaker.TrippedError as e:
            self.logger.warning(f"DIFY circuit breaker tripped: {e}")
            raise TransientInfraError(f"DIFY circuit breaker tripped: {e}") from e

        if use_cache and answer:
            self._cache.set(prompt, answer, temperature=temperature, max_tokens=max_tokens)

        return answer


# ---------------------------------------------------------------------------
# DeepSeekClient  (was rigor/deepseek_client.py)
# ---------------------------------------------------------------------------

class DeepSeekClient(BaseLLMClient):
    """Communicates with DeepSeek API."""

    @property
    def _logger_name(self) -> str:
        return "RIGOR.DeepSeek"

    @property
    def api_key(self) -> str | None:
        return self.config.deepseek_api_key

    def __init__(self, config=None):
        super().__init__(config)
        self.model = self.config.deepseek_model
        self.timeout = self.config.deepseek_timeout

        # 2026-08-18: unified LLM_CACHE_* env vars take precedence over the
        # legacy per-provider defaults. See DifyClient.__init__ for rationale.
        _unified_ttl_s = float(os.environ.get("LLM_CACHE_TTL_S", "0") or 0)
        _unified_max_size = int(os.environ.get("LLM_CACHE_MAX_SIZE", "0") or 0)
        _cache_ttl_s = _unified_ttl_s or getattr(self.config, "deepseek_cache_ttl_s", 300.0)
        _cache_max_size = _unified_max_size or getattr(self.config, "deepseek_cache_max_size", 1000)

        # P1 / C-4: structured CacheVersionBundle — see DifyClient notes.
        _legacy_bundle = (
            getattr(self.config, "deepseek_cache_version_bundle", None)
            or os.getenv("LLM_CACHE_VERSION_BUNDLE", "")
        )
        if _legacy_bundle:
            cache_bundle: CacheVersionBundle | str = _legacy_bundle
        else:
            cache_bundle = CacheVersionBundle(
                prompt_version=os.getenv("PROMPT_VERSION", ""),
                schema_version=os.getenv("SCHEMA_VERSION", ""),
                policy_version=os.getenv("POLICY_VERSION", os.getenv("SKILL_VERSION", "")),
                model_family=f"deepseek:{self.model}",
                extraction_mode=os.getenv("PIPELINE_EXECUTION_MODE", "production"),
            )
        self._cache = LLMCache(
            max_size=_cache_max_size,
            default_ttl_s=_cache_ttl_s,
            version_bundle=cache_bundle,
        )
        base_url = getattr(self.config, "deepseek_base_url", None) or os.getenv("DEEPSEEK_BASE_URL")
        if base_url:
            base_url = base_url.rstrip("/")
            if not base_url.endswith("/chat/completions"):
                self._api_url = f"{base_url}/v1/chat/completions" if not base_url.endswith("/v1") else f"{base_url}/chat/completions"
            else:
                self._api_url = base_url
        else:
            self._api_url = _DEFAULT_DEEPSEEK_API_URL

    def _is_retryable_error(self, exc: Exception) -> bool:
        import httpx
        if isinstance(exc, httpx.HTTPStatusError):
            try:
                status = int(exc.response.status_code)
            except Exception:
                status = 0
            return status in {429, 502, 503, 504}
        retryable = (
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.PoolTimeout,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.TimeoutException,
            asyncio.TimeoutError,
            httpx.HTTPError,
            ConnectionError,
            OSError,
        )
        return isinstance(exc, retryable)

    async def call(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = True,
        **kwargs,
    ) -> str:
        """Call DeepSeek API. Uses streaming by default to avoid read timeouts.

        Streaming keeps the connection alive via incremental SSE chunks,
        preventing the 180s timeout that occurs when waiting for a full
        response from long generations. Falls back to non-streaming on error.
        """
        if not self.api_key:
            self.logger.error("DeepSeek API key not configured")
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")

        use_cache = kwargs.get("use_cache", True)
        if use_cache:
            cached_response = self._cache.get(prompt, temperature=temperature, max_tokens=max_tokens)
            if cached_response is not None:
                self.logger.info(f"DeepSeek API cache hit - Prompt: {len(prompt)} chars, Response: {len(cached_response)} characters")
                return cached_response

        if stream:
            try:
                answer = await self._call_streaming(prompt, temperature, max_tokens, **kwargs)
            except Exception as e:
                self.logger.warning("Streaming failed, falling back to non-streaming: %s", e)
                answer = await self._call_non_streaming(prompt, temperature, max_tokens, **kwargs)
        else:
            answer = await self._call_non_streaming(prompt, temperature, max_tokens, **kwargs)

        if use_cache and answer:
            self._cache.set(prompt, answer, temperature=temperature, max_tokens=max_tokens)
        return answer

    async def _call_streaming(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> str:
        """Stream-based call: receives SSE chunks to avoid read timeouts.

        Timeout strategy: connect=30s (fast fail), read from self.timeout
        (default 300s — generous, only triggers if NO chunk arrives for
        5 min — not total time). Previously read=300.0 was hard-coded
        which ignored DEEPSEEK_TIMEOUT_SECONDS / per-client overrides.
        """
        import httpx
        import json as _json

        _read_timeout = float(self.timeout) if self.timeout else 300.0
        streaming_timeout = httpx.Timeout(
            connect=30.0,
            read=_read_timeout,
            write=30.0,
            pool=30.0,
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert in enterprise ontology design and semantic modeling."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            **kwargs,
        }

        async with _httpx_async_client(streaming_timeout, self._api_url) as client:
            async with client.stream("POST", self._api_url, headers=headers, json=payload) as response:
                status = response.status_code
                if status in {429, 502, 503, 504}:
                    raise TransientInfraError(f"DeepSeek API transient error: {status}")
                if status != 200:
                    body = await response.aread()
                    try:
                        err = _json.loads(body).get('error', {}).get('message', 'Unknown error')
                    except Exception:
                        err = f"HTTP {status}"
                    raise RuntimeError(f"DeepSeek API error: {err}")

                content_parts: list[str] = []
                reasoning_parts: list[str] = []
                prompt_tokens = 0
                completion_tokens = 0
                chunk_count = 0

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = _json.loads(data_str)
                    except _json.JSONDecodeError:
                        continue

                    chunk_count += 1

                    if "usage" in chunk and chunk["usage"]:
                        prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                        completion_tokens = chunk["usage"].get("completion_tokens", 0)

                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            content_parts.append(delta["content"])
                        if delta.get("reasoning_content"):
                            reasoning_parts.append(delta["reasoning_content"])

        content = "".join(content_parts)
        if not content:
            content = "".join(reasoning_parts)
            if content:
                self.logger.debug("Using reasoning_content as content (content was empty)")

        if prompt_tokens or completion_tokens:
            self._track_llm_usage(prompt_tokens, completion_tokens)

        self.logger.debug(
            "DeepSeek streaming: %d chars, %d chunks, %d/%d tokens",
            len(content), chunk_count, prompt_tokens, completion_tokens,
        )
        return content

    async def _call_non_streaming(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> str:
        """Original non-streaming call (fallback path)."""
        try:
            import httpx
            async with _httpx_async_client(self.timeout, self._api_url) as client:
                response = await client.post(
                    self._api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are an expert in enterprise ontology design and semantic modeling."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        **kwargs,
                    },
                )
                result = response.json()
                status = response.status_code
                if status in {429, 502, 503, 504}:
                    self.logger.error("DeepSeek API transient error: %d", status)
                    raise TransientInfraError(f"DeepSeek API transient error: {status}")
                if status != 200:
                    error_msg = result.get('error', {}).get('message', 'Unknown error')
                    self.logger.error("DeepSeek API error: %s", error_msg)
                    raise RuntimeError(f"DeepSeek API error: {error_msg}")
                message = result['choices'][0]['message']
                content = message.get('content', '')

                if not content:
                    reasoning_content = message.get('reasoning_content', '')
                    if reasoning_content:
                        content = reasoning_content
                        self.logger.debug("Using reasoning_content as content (original content was empty)")

                usage = result.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                self._track_llm_usage(prompt_tokens, completion_tokens)

                self.logger.debug("DeepSeek response received: %d chars", len(content))
                return content
        except (httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.PoolTimeout,
                httpx.ConnectTimeout, httpx.ReadTimeout, httpx.TimeoutException,
                asyncio.TimeoutError, httpx.HTTPError, ConnectionError, OSError) as e:
            self.logger.error("DeepSeek API transient connection error: %s", type(e).__name__)
            raise TransientInfraError(f"DeepSeek API transient error: {type(e).__name__}: {e}") from e
        except TransientInfraError:
            raise
        except Exception as e:
            self.logger.error("DeepSeek API call failed: %s", e)
            raise


# ---------------------------------------------------------------------------
# UnifiedLLMClient  (was rigor/llm_client.py)
# ---------------------------------------------------------------------------

class UnifiedLLMClient:
    """Routes LLM calls to the appropriate provider based on config.

    F5 Fix (2026-05-18): Added fallback provider support. When the primary provider
    raises TransientInfraError after exhausting retries, the call is retried on each
    fallback provider in turn. Fallbacks are opt-in via config.llm_fallback_providers
    (comma-separated string, e.g. "dify" or "deepseek,dify").

    Smart routing (2026-07-06): Added task complexity-based model routing.
    When complexity="complex", automatically routes to DIFY PRO (e.g., DeepSeek V3.1) 
    for better reasoning performance. Complex tasks include:
    - Long documents (>15k chars)
    - Cross-file association analysis
    - Quality assessment failure retry
    - Schema validation repeated failures

    Dual DIFY Model (2026-07-07): Supports two DIFY models:
    - Standard: Fast model for simple extraction tasks
    - PRO: DeepSeek V3.1 or other advanced model for complex tasks

    Request Batching (2026-07-10): Added support for LLM request batching to reduce
    API overhead when processing multiple small requests concurrently.

    P7 Fix (2026-07-12): Fixed _MIN_CALL_INTERVAL_S to read from config instead of
    hardcoded value, ensuring consistency with DIFY_RATE_LIMIT_MAX_CALLS configuration.
    """

    _PROVIDER_CLASSES: dict[str, type] = {
        "dify": DifyClient,
        "deepseek": DeepSeekClient,
    }

    _COMPLEXITY_THRESHOLD_CHARS = 15000

    def __init__(self, config=None, enable_batching: bool = True, provider: str | None = None, fallback_providers: str | None = None):
        self.config = config or get_config()
        self.logger = logging.getLogger("RIGOR.LLM")
        selected_provider = (provider or getattr(self.config, "llm_provider", None) or os.environ.get("LLM_PROVIDER", "dify") or "dify").lower()
        self.provider = selected_provider
        self.client = self._build_client(self.provider)
        self._last_call_time = 0.0
        
        self._MIN_CALL_INTERVAL_S = getattr(self.config, "dify_min_call_interval_s", 3.0)
        self.logger.info(f"LLM call interval configured: {self._MIN_CALL_INTERVAL_S}s")

        # Request batching
        self._enable_batching = enable_batching and getattr(self.config, "llm_batching_enabled", True)
        self._batch_client = LLMBatchClient(
            self.client,
            BatchConfig(
                max_batch_size=getattr(self.config, "llm_batch_max_size", 10),
                max_wait_ms=getattr(self.config, "llm_batch_max_wait_ms", 500),
                min_batch_size=getattr(self.config, "llm_batch_min_size", 2),
                enabled=self._enable_batching,
            )
        )
        if self._enable_batching:
            self.logger.info("LLM request batching enabled")

        # Build ordered fallback list (skip primary)
        fallback_cfg = fallback_providers if fallback_providers is not None else (getattr(self.config, "llm_fallback_providers", "") or os.environ.get("LLM_FALLBACK_PROVIDERS", "") or "")
        self._fallback_clients: list[tuple[str, BaseLLMClient]] = []
        for name in [p.strip().lower() for p in fallback_cfg.split(",") if p.strip()]:
            if name == self.provider:
                continue
            try:
                fb_client = self._build_client(name)
                self._fallback_clients.append((name, fb_client))
                self.logger.info("Registered fallback LLM provider: %s", name)
            except Exception as exc:
                self.logger.warning("Skipping fallback provider %s (init failed): %s", name, exc)

        # DIFY PRO client for complex tasks (uses DIFY PRO config)
        self._pro_client = None
        if self.provider == "dify":
            pro_api_key = getattr(self.config, "dify_pro_api_key", "")
            pro_model = getattr(self.config, "dify_pro_model", "")
            if pro_api_key and pro_model:
                try:
                    pro_max_tokens = getattr(self.config, "dify_pro_max_tokens", 4000)
                    pro_temp = getattr(self.config, "dify_pro_temperature", 0.1)
                    pro_timeout = getattr(self.config, "dify_pro_timeout_seconds", 300)
                    self._pro_client = DifyClient(
                        config=self.config,
                        api_key_override=pro_api_key,
                        model_override=pro_model,
                        max_tokens_override=pro_max_tokens,
                        temperature_override=pro_temp,
                        timeout_override=pro_timeout,
                        is_pro=True
                    )
                    self.logger.info("DIFY PRO client initialized for complex tasks - Model: %s", pro_model)
                except Exception as exc:
                    self.logger.warning("Failed to initialize DIFY PRO client: %s", exc)
            elif "deepseek" in [f[0] for f in self._fallback_clients]:
                try:
                    pro_model = getattr(self.config, "deepseek_pro_model", "")
                    self._pro_client = self._build_client("deepseek", override_model=pro_model)
                    self.logger.info("DeepSeek PRO client initialized for complex tasks")
                except Exception as exc:
                    self.logger.warning("Failed to initialize DeepSeek PRO client: %s", exc)
        elif self.provider == "deepseek":
            # DeepSeek primary: build a DeepSeek PRO client for complex tasks
            # when DEEPSEEK_PRO_MODEL differs from DEEPSEEK_MODEL. If they are
            # identical (common config), reuse the primary client to avoid an
            # unused duplicate.
            try:
                pro_model = getattr(self.config, "deepseek_pro_model", "")
                if pro_model and pro_model != getattr(self.config, "deepseek_model", ""):
                    self._pro_client = self._build_client("deepseek", override_model=pro_model)
                    self.logger.info("DeepSeek PRO client initialized for complex tasks - Model: %s", pro_model)
            except Exception as exc:
                self.logger.warning("Failed to initialize DeepSeek PRO client: %s", exc)

    def _build_client(self, provider: str, override_model: str = "") -> BaseLLMClient:
        cls = self._PROVIDER_CLASSES.get(provider)
        if cls is None:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        if override_model:
            import copy
            pro_config = copy.deepcopy(self.config)
            object.__setattr__(pro_config, "deepseek_model", override_model)
            return cls(pro_config)
        
        return cls(self.config)

    def _detect_complexity(self, prompt: str, **kwargs) -> str:
        """Detect task complexity based on prompt length and other factors."""
        complexity = kwargs.get("complexity", "auto").lower()
        
        if complexity != "auto":
            return complexity
        
        if len(prompt) > self._COMPLEXITY_THRESHOLD_CHARS:
            return "complex"
        
        if kwargs.get("is_cross_file", False):
            return "complex"
        
        if kwargs.get("is_retry", False):
            return "complex"
        
        if kwargs.get("is_rigor_refinement", False):
            return "complex"
        
        return "standard"

    def _get_client_for_complexity(self, complexity: str) -> BaseLLMClient:
        """Get the appropriate client based on complexity level."""
        if complexity == "complex" and self._pro_client:
            return self._pro_client
        return self.client

    async def call(self, prompt: str, temperature: float = None, max_tokens: int = None, **kwargs) -> str:
        complexity = self._detect_complexity(prompt, **kwargs)
        client = self._get_client_for_complexity(complexity)
        client_name = "DIFY PRO" if hasattr(client, '_is_pro') and client._is_pro else self.provider
        
        if complexity == "complex":
            self.logger.info("Complex task detected (prompt_len=%d), using %s", len(prompt), client_name)
        else:
            self.logger.debug("Standard task (prompt_len=%d), using %s", len(prompt), client_name)
        
        await self._enforce_call_interval()
        
        try:
            if complexity == "standard" and self._enable_batching:
                return await self._batch_client.call(prompt, temperature, max_tokens, **kwargs)
            return await client.call(prompt, temperature, max_tokens, **kwargs)
        except CircuitBreaker.TrippedError as cb_exc:
            self.logger.warning("Circuit breaker OPEN for %s, attempting immediate fallback: %s", client_name, cb_exc)
            if complexity == "complex" and client != self.client:
                try:
                    return await self.client.call(prompt, temperature, max_tokens, **kwargs)
                except Exception:
                    pass
            return await self._call_fallbacks(prompt, temperature, max_tokens, cb_exc, **kwargs)
        except TransientInfraError as primary_exc:
            if complexity == "complex" and client != self.client:
                self.logger.warning("%s transient error, trying standard %s", client_name, self.provider)
                return await self.client.call(prompt, temperature, max_tokens, **kwargs)
            self.logger.warning("Provider %s transient error, trying fallbacks: %s", self.provider, primary_exc)
            return await self._call_fallbacks(prompt, temperature, max_tokens, primary_exc, **kwargs)
        except Exception as e:
            self.logger.error("LLM call failed with %s: %s", client_name, e)
            raise

    async def _enforce_call_interval(self):
        import time
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self._MIN_CALL_INTERVAL_S:
            wait_time = self._MIN_CALL_INTERVAL_S - elapsed
            await asyncio.sleep(wait_time)
        self._last_call_time = time.time()

    async def call_with_retry(self, prompt: str, policy: RetryPolicy | None = None, **kwargs) -> str:
        complexity = self._detect_complexity(prompt, **kwargs)
        client = self._get_client_for_complexity(complexity)
        client_name = "DIFY PRO" if hasattr(client, '_is_pro') and client._is_pro else self.provider
        
        if complexity == "complex":
            self.logger.info("Complex task detected (with retry) (prompt_len=%d), using %s", len(prompt), client_name)
        else:
            self.logger.debug("Standard task (with retry) (prompt_len=%d), using %s", len(prompt), client_name)
        
        await self._enforce_call_interval()
        
        try:
            if complexity == "standard" and self._enable_batching:
                return await self._batch_client.call_with_retry(prompt, policy=policy, **kwargs)
            return await client.call_with_retry(prompt, policy=policy, **kwargs)
        except CircuitBreaker.TrippedError as cb_exc:
            self.logger.warning("Circuit breaker OPEN for %s, attempting immediate fallback with retry: %s", client_name, cb_exc)
            if complexity == "complex" and client != self.client:
                try:
                    return await self.client.call_with_retry(prompt, policy=policy, **kwargs)
                except Exception:
                    pass
            return await self._call_fallbacks_with_retry(prompt, policy, cb_exc, **kwargs)
        except TransientInfraError as primary_exc:
            if complexity == "complex" and client != self.client:
                self.logger.warning("%s exhausted retries, falling back to standard %s", client_name, self.provider)
                return await self.client.call_with_retry(prompt, policy=policy, **kwargs)
            self.logger.warning("Provider %s exhausted retries, trying fallbacks: %s", self.provider, primary_exc)
            return await self._call_fallbacks_with_retry(prompt, policy, primary_exc, **kwargs)
        except Exception as e:
            self.logger.error("LLM call with retry failed: %s", e)
            if self._fallback_clients:
                self.logger.warning("Non-transient error from %s, trying fallbacks anyway: %s", self.provider, e)
                return await self._call_fallbacks_with_retry(prompt, policy, e, **kwargs)
            raise

    async def _call_fallbacks(
        self,
        prompt: str,
        temperature: float | None,
        max_tokens: int | None,
        last_exc: Exception,
        **kwargs,
    ) -> str:
        """Try each fallback provider in order; raise last exception if all fail."""
        for name, fb_client in self._fallback_clients:
            try:
                self.logger.info("Trying fallback provider: %s", name)
                result = await fb_client.call(prompt, temperature, max_tokens, **kwargs)
                self.logger.info("Fallback provider %s succeeded", name)
                return result
            except Exception as exc:
                self.logger.warning("Fallback provider %s failed: %s", name, exc)
                last_exc = exc
        raise last_exc

    async def _call_fallbacks_with_retry(
        self,
        prompt: str,
        policy: RetryPolicy | None,
        last_exc: Exception,
        **kwargs,
    ) -> str:
        """Try each fallback provider (with retry) in order; raise last exception if all fail."""
        for name, fb_client in self._fallback_clients:
            try:
                self.logger.info("Trying fallback provider with retry: %s", name)
                result = await fb_client.call_with_retry(prompt, policy=policy, **kwargs)
                self.logger.info("Fallback provider %s succeeded", name)
                return result
            except Exception as exc:
                self.logger.warning("Fallback provider %s failed: %s", name, exc)
                last_exc = exc
        raise last_exc


# ---------------------------------------------------------------------------
# D-1 unified client factory (2026-08-15)
# ---------------------------------------------------------------------------
#
# Every production caller constructs the Rigor LLM client through
# ``get_llm_client`` so circuit-breaker / rate-limiter / LLM-cache state is
# shared per provider signature instead of being duplicated per call site.
# The factory lives here (dependency direction: ontology_compiler must not
# import from src.application); ``src.application.bootstrap.llm_bootstrap``
# re-exports it as its public API.

_llm_client_cache: dict[str, "UnifiedLLMClient"] = {}


def _llm_client_cache_key(provider: str | None, fallback_providers: str | None) -> str:
    return f"{provider or ''}|{fallback_providers or ''}"


def get_llm_client(provider: str | None = None, fallback_providers: str | None = None) -> "UnifiedLLMClient":
    """Get (and cache) the unified Rigor LLM client.

    Reads the same ``LLM_PROVIDER`` / ``LLM_FALLBACK_PROVIDERS`` env / config
    that ``UnifiedLLMClient()`` uses. Subsequent calls with the same provider
    signature return the cached instance. Call ``reset_llm_client_cache()`` in
    tests to force a fresh client.
    """
    key = _llm_client_cache_key(provider, fallback_providers)
    client = _llm_client_cache.get(key)
    if client is None:
        client = UnifiedLLMClient(provider=provider, fallback_providers=fallback_providers)
        _llm_client_cache[key] = client
    return client


def reset_llm_client_cache() -> None:
    """Clear the cached LLM clients (test isolation hook)."""
    _llm_client_cache.clear()
