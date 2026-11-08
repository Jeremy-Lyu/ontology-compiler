"""Unified LLM Configuration Bootstrap.

P2-5 Implementation (2026-06-16):
Centralizes all LLM client initialization and configuration.
Provides a single entry point for LLM setup across the application.

Benefits:
- Single source of truth for LLM configuration
- Consistent client initialization across modules
- Easy testing with mock clients
- Configuration validation at startup
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict

from src.application.services.llm_trace import extract_prompt_text, hash_text

logger = logging.getLogger(__name__)


def _wrap_client_for_monitoring(client: Any, provider: str, model: str) -> Any:
    """包装LLM客户端以添加监控功能

    拦截所有LLM调用并记录到LLMMonitor，无需修改每个阶段代码。
    """
    try:
        from src.application.services.llm_monitor import get_llm_monitor, TruncationReason

        monitor = get_llm_monitor()
        provider_rates = {
            "deepseek": 0.0005,
            "dify": 0.001,
            "openai": 0.002,
            "azure": 0.002,
            "local": 0.0,
        }
        monitor.set_provider_cost_rate(provider.lower(), provider_rates.get(provider.lower(), 0.0))

        class MonitoredClient:
            def __init__(self, wrapped):
                self._wrapped = wrapped
                self._provider = provider
                self._model = model

            def __getattr__(self, name):
                attr = getattr(self._wrapped, name)
                if callable(attr):
                    def wrapped_call(*args, **kwargs):
                        return self._monitor_call(name, attr, *args, **kwargs)
                    return wrapped_call
                return attr

            def _check_budget(self):
                """C-1: enforce the gateway hard budget before spending.

                Inactive by default (all limits 0 => no-op), so enabling the
                wrapper never changes behavior until a policy is configured.
                The gateway owns the ledger; the wrapper reads it lazily to
                avoid import cycles.
                """
                try:
                    from src.application.bootstrap.llm_gateway import get_llm_gateway

                    get_llm_gateway().budget.check()
                except ImportError:
                    return

            def _monitor_call(self, method_name: str, method: Callable, *args, **kwargs):
                self._check_budget()
                start_time = time.time()
                prompt_text = extract_prompt_text(*args, **kwargs)
                prompt_hash = hash_text(prompt_text) if prompt_text else ""
                try:
                    result = method(*args, **kwargs)

                    duration_ms = int((time.time() - start_time) * 1000)
                    usage = self._extract_usage(result)
                    truncation_reason = self._extract_truncation_reason(usage)

                    monitor.record_call(
                        provider=self._provider,
                        model=self._model,
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                        duration_ms=duration_ms,
                        success=True,
                        truncation_reason=truncation_reason,
                        call_id=prompt_hash,
                        prompt_hash=prompt_hash,
                    )

                    return result
                except Exception as e:
                    duration_ms = int((time.time() - start_time) * 1000)
                    monitor.record_call(
                        provider=self._provider,
                        model=self._model,
                        duration_ms=duration_ms,
                        success=False,
                        truncation_reason=TruncationReason.ERROR,
                        error_message=str(e),
                        call_id=prompt_hash,
                        prompt_hash=prompt_hash,
                    )
                    raise

            def _extract_usage(self, result) -> Dict[str, Any]:
                """从LLM结果中提取使用信息"""
                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

                if hasattr(result, "usage"):
                    usage_obj = result.usage
                    if hasattr(usage_obj, "prompt_tokens"):
                        usage["prompt_tokens"] = usage_obj.prompt_tokens
                    if hasattr(usage_obj, "completion_tokens"):
                        usage["completion_tokens"] = usage_obj.completion_tokens
                    if hasattr(usage_obj, "total_tokens"):
                        usage["total_tokens"] = usage_obj.total_tokens

                elif isinstance(result, dict):
                    usage_info = result.get("usage", {})
                    usage["prompt_tokens"] = usage_info.get("prompt_tokens", 0)
                    usage["completion_tokens"] = usage_info.get("completion_tokens", 0)
                    usage["total_tokens"] = usage_info.get("total_tokens", 0)

                return usage

            def _extract_truncation_reason(self, usage: Dict[str, Any]) -> TruncationReason:
                raw = usage.get("truncation_reason", "none")
                try:
                    return TruncationReason(raw)
                except Exception:
                    return TruncationReason.NONE

        return MonitoredClient(client)
    except ImportError:
        logger.warning("LLM Monitor not available, skipping client wrapping")
        return client


class LLMProviderType(Enum):
    """Supported LLM provider types."""
    DEEPSEEK = "deepseek"
    DIFY = "dify"
    OPENAI = "openai"
    AZURE = "azure"
    LOCAL = "local"


@dataclass
class LLMClientConfig:
    """Configuration for a single LLM client."""
    provider: LLMProviderType
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.3
    max_tokens: int = 2000
    extra_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMBootstrapConfig:
    """Complete LLM bootstrap configuration."""
    primary_provider: LLMProviderType = LLMProviderType.DIFY
    fallback_provider: LLMProviderType | None = None
    deepseek_config: LLMClientConfig | None = None
    dify_config: LLMClientConfig | None = None
    openai_config: LLMClientConfig | None = None
    azure_config: LLMClientConfig | None = None
    local_config: LLMClientConfig | None = None
    enable_usage_tracking: bool = True
    enable_retry: bool = True
    retry_policy: dict[str, Any] = field(default_factory=lambda: {
        "max_attempts": 3,
        "base_delay_s": 2.0,
        "max_delay_s": 30.0,
    })


class LLMBootstrap:
    """Unified LLM configuration and client initialization.

    This class serves as the single entry point for all LLM-related
    configuration and client creation in the application.

    Usage:
        # Initialize at application startup
        bootstrap = LLMBootstrap.from_config()
        bootstrap.validate()

        # Get configured clients
        deepseek_client = bootstrap.get_client(LLMProviderType.DEEPSEEK)
        dify_client = bootstrap.get_client(LLMProviderType.DIFY)

        # Or use the unified provider
        provider = bootstrap.get_unified_provider()
        response = provider.complete(LLMRequest(prompt="Hello"))
    """

    _instance: "LLMBootstrap | None" = None

    def __init__(self, config: LLMBootstrapConfig | None = None) -> None:
        """Initialize the bootstrap with optional configuration.

        Args:
            config: Optional bootstrap configuration. If not provided,
                reads from CompilerConfig environment variables.
        """
        self._config = config or self._load_from_env()
        self._clients: dict[LLMProviderType, Any] = {}
        self._validated = False

    @classmethod
    def get_instance(cls) -> "LLMBootstrap":
        """Get the singleton instance of LLMBootstrap.

        Creates a new instance if one doesn't exist.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None

    def _load_from_env(self) -> LLMBootstrapConfig:
        """Load configuration from environment via CompilerConfig."""
        from ontology_compiler.config import get_config

        compiler_config = get_config()

        # Determine primary provider
        provider_str = compiler_config.llm_provider or "dify"
        try:
            primary_provider = LLMProviderType(provider_str.lower())
        except ValueError:
            logger.warning(f"Unknown LLM provider '{provider_str}', using DIFY")
            primary_provider = LLMProviderType.DIFY

        # Build DeepSeek config
        deepseek_config = LLMClientConfig(
            provider=LLMProviderType.DEEPSEEK,
            api_key=compiler_config.deepseek_api_key,
            base_url="https://api.deepseek.com/v1",
            model=compiler_config.deepseek_model or "deepseek-chat",
            timeout=compiler_config.deepseek_timeout or 60,
            max_retries=3,
            temperature=0.3,
            max_tokens=2000,
        )

        # Build Dify config
        dify_config = LLMClientConfig(
            provider=LLMProviderType.DIFY,
            api_key=compiler_config.dify_api_key,
            base_url=compiler_config.dify_base_url,
            model=None,  # Dify uses app_type instead
            timeout=compiler_config.dify_timeout_seconds or 180,
            max_retries=compiler_config.dify_max_retries or 3,
            extra_config={
                "username": compiler_config.dify_username,
                "app_type": compiler_config.dify_app_type,
            },
        )

        return LLMBootstrapConfig(
            primary_provider=primary_provider,
            deepseek_config=deepseek_config,
            dify_config=dify_config,
            enable_usage_tracking=True,
            enable_retry=True,
        )

    def validate(self) -> bool:
        """Validate the configuration and check API keys.

        Returns:
            True if configuration is valid.

        Raises:
            ConfigError: If required configuration is missing.
        """
        from src.domain.errors import ConfigError

        errors: list[str] = []

        # Check primary provider configuration
        primary_config = self._get_provider_config(self._config.primary_provider)
        if primary_config is None:
            errors.append(f"Primary provider {self._config.primary_provider.value} has no configuration")
        elif not primary_config.api_key:
            # Some providers (like local) don't need API keys
            if self._config.primary_provider not in (LLMProviderType.LOCAL,):
                env_var = f"{self._config.primary_provider.value.upper()}_API_KEY"
                errors.append(f"Missing API key for {self._config.primary_provider.value}: set {env_var}")

        if errors:
            raise ConfigError(
                "LLM Bootstrap validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        self._validated = True
        logger.info(f"LLM Bootstrap validated: primary={self._config.primary_provider.value}")
        return True

    def _get_provider_config(self, provider: LLMProviderType) -> LLMClientConfig | None:
        """Get configuration for a specific provider."""
        config_map = {
            LLMProviderType.DEEPSEEK: self._config.deepseek_config,
            LLMProviderType.DIFY: self._config.dify_config,
            LLMProviderType.OPENAI: self._config.openai_config,
            LLMProviderType.AZURE: self._config.azure_config,
            LLMProviderType.LOCAL: self._config.local_config,
        }
        return config_map.get(provider)

    def get_client(self, provider: LLMProviderType) -> Any:
        """Get or create a client for the specified provider.

        Args:
            provider: The LLM provider type.

        Returns:
            The configured client for that provider.

        Raises:
            ConfigError: If provider is not configured.
        """
        if provider in self._clients:
            return self._clients[provider]

        config = self._get_provider_config(provider)
        if config is None:
            from src.domain.errors import ConfigError
            raise ConfigError(f"No configuration for LLM provider: {provider.value}")

        client = self._create_client(config)
        self._clients[provider] = client
        return client

    def _create_client(self, config: LLMClientConfig) -> Any:
        """Create a client instance from configuration.
        
        如果启用了使用跟踪，会自动包装客户端以添加监控功能。
        """
        if config.provider == LLMProviderType.DEEPSEEK:
            client = self._create_deepseek_client(config)
        elif config.provider == LLMProviderType.DIFY:
            client = self._create_dify_client(config)
        elif config.provider == LLMProviderType.OPENAI:
            client = self._create_openai_client(config)
        elif config.provider == LLMProviderType.AZURE:
            client = self._create_azure_client(config)
        elif config.provider == LLMProviderType.LOCAL:
            client = self._create_local_client(config)
        else:
            raise NotImplementedError(f"Client creation for {config.provider} not implemented")

        if self._config.enable_usage_tracking:
            model = config.model or config.extra_config.get("model", "unknown")
            client = _wrap_client_for_monitoring(client, config.provider.value, model)

        return client

    def _create_deepseek_client(self, config: LLMClientConfig) -> Any:
        """Create DeepSeek client."""
        from ontology_compiler.rigor.llm_backend import DeepSeekClient
        return DeepSeekClient(config=config.extra_config.get("compiler_config"))

    def _create_dify_client(self, config: LLMClientConfig) -> Any:
        """Create Dify client."""
        from ontology_compiler.rigor.llm_backend import DifyClient
        return DifyClient(config=config.extra_config.get("compiler_config"))

    def _create_openai_client(self, config: LLMClientConfig) -> Any:
        """Create OpenAI client."""
        try:
            import openai
            return openai.OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout,
            )
        except ImportError:
            raise RuntimeError("openai package required for OpenAI backend")

    def _create_azure_client(self, config: LLMClientConfig) -> Any:
        """Create Azure OpenAI client."""
        try:
            import openai
            return openai.AzureOpenAI(
                api_key=config.api_key,
                azure_endpoint=config.base_url,
                api_version=config.extra_config.get("api_version", "2024-02-01"),
            )
        except ImportError:
            raise RuntimeError("openai package required for Azure backend")

    def _create_local_client(self, config: LLMClientConfig) -> Any:
        """Create local/mock client for testing."""
        # Return a mock client that echoes prompts
        class LocalMockClient:
            def __init__(self, config: LLMClientConfig):
                self.config = config

            def complete(self, prompt: str, **kwargs: Any) -> str:
                return f"[LOCAL MOCK] {prompt[:100]}..."

            async def call(self, prompt: str, **kwargs: Any) -> str:
                return self.complete(prompt, **kwargs)

        return LocalMockClient(config)

    def get_unified_provider(self) -> Any:
        """Get the unified LLM provider using primary configuration."""
        from ontology_compiler.rigor.llm_provider import UnifiedLLMProvider, LLMBackend

        backend_map = {
            LLMProviderType.DEEPSEEK: LLMBackend.DEEPSEEK,
            LLMProviderType.OPENAI: LLMBackend.OPENAI,
            LLMProviderType.AZURE: LLMBackend.AZURE,
        }

        backend = backend_map.get(self._config.primary_provider, LLMBackend.DEEPSEEK)
        primary_config = self._get_provider_config(self._config.primary_provider)

        return UnifiedLLMProvider(
            backend=backend,
            api_key=primary_config.api_key if primary_config else None,
        )

    def get_rigor_client(self) -> Any:
        """Get the Rigor LLM client (UnifiedLLMClient).

        Delegates to the factory in ``ontology_compiler.rigor.llm_backend``
        (D-1) so every caller shares one assembly point and caches by provider
        signature instead of spinning up fresh clients.
        """
        from ontology_compiler.rigor.llm_backend import get_llm_client
        return get_llm_client()

    @property
    def config(self) -> LLMBootstrapConfig:
        """Get the bootstrap configuration."""
        return self._config

    @property
    def is_validated(self) -> bool:
        """Check if bootstrap has been validated."""
        return self._validated


# Convenience function for application startup
def initialize_llm_bootstrap() -> LLMBootstrap:
    """Initialize and validate LLM bootstrap at application startup.

    Returns:
        The validated LLMBootstrap instance.

    Raises:
        ConfigError: If configuration is invalid.
    """
    bootstrap = LLMBootstrap.get_instance()
    bootstrap.validate()
    return bootstrap


_llm_client_cache: dict[str, Any] = {}


def _llm_client_cache_key(provider: str | None, fallback_providers: str | None) -> str:
    """Cache key for an LLM client signature.

    2026-08-15 (D-1): direct ``UnifiedLLMClient()`` instantiations across
    pipeline/extractor/RAG are consolidated here. Every caller gets the same
    client for the same provider signature, so circuit-breaker / rate-limiter /
    cache state is shared instead of being duplicated per call site.
    """
    return f"{provider or ''}|{fallback_providers or ''}"


def get_llm_client(provider: str | None = None, fallback_providers: str | None = None) -> Any:
    """Get (and cache) the Rigor LLM client — the only client assembly point.

    Canonical implementation lives in ``ontology_compiler.rigor.llm_backend``
    (dependency direction: ontology_compiler must not import from
    src.application); this module re-exports it as the application-facing API.
    """
    from ontology_compiler.rigor.llm_backend import get_llm_client as _canonical

    return _canonical(provider=provider, fallback_providers=fallback_providers)


def reset_llm_client_cache() -> None:
    """Clear the cached LLM clients (test isolation hook)."""
    from ontology_compiler.rigor.llm_backend import reset_llm_client_cache as _reset

    _reset()


__all__ = [
    "LLMProviderType",
    "LLMClientConfig",
    "LLMBootstrapConfig",
    "LLMBootstrap",
    "initialize_llm_bootstrap",
    "get_llm_client",
    "reset_llm_client_cache",
]