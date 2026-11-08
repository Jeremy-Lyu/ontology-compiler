"""
Configuration for Ontology Compiler

Centralized configuration for compiler/runtime components.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict

from core.utils.file_helpers import load_json_object

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent
_DEFAULT_ENV_FILES = [
    str(_PROJECT_ROOT / "config" / "env.env"),
    str(_PROJECT_ROOT / "env.local.env"),
    str(_PROJECT_ROOT / "env.env"),
]
_DEFAULT_RUNTIME_SETTINGS: Dict[str, Any] = {
    "harness": {
        "enabled": True,
    },
    "hooks": {
        "enabled": False,
        "deny_exit_code": 2,
        "timeout_seconds": 10,
        "execution_mode": "exec",
        "allow_shell": False,
        "commands": [],
        "event_commands": {
            "PreToolUse": [],
            "PostToolUse": []
        },
    },
    "env": {
        "files": list(_DEFAULT_ENV_FILES),
    },
    "pipeline": {
        "policy_path": "config/pipeline_policy.json",
        "default_profile": "local",
        "default_run_id_prefix": "TBH-",
    },
    "paths": {
        "outputs_root": "outputs",
        "outputs_ontology_root": "outputs/ontology",
        "outputs_capsules_root": "outputs/capsules",
        "outputs_logs_root": "outputs/logs",
    },
}

_RUNTIME_SETTINGS_CACHE: Dict[str, Any] | None = None
_RUNTIME_SETTINGS_SOURCES: list[str] = []
_LOADED_ENV_FILES: list[str] = []


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge dictionaries (override wins)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_runtime_settings(
    settings_paths: list[str] | None = None,
    *,
    force_reload: bool = False,
) -> Dict[str, Any]:
    """Load runtime settings with layered overrides.

    Priority: base defaults -> project settings.json -> local settings.local.json
    """
    global _RUNTIME_SETTINGS_CACHE, _RUNTIME_SETTINGS_SOURCES

    if _RUNTIME_SETTINGS_CACHE is not None and not force_reload:
        return copy.deepcopy(_RUNTIME_SETTINGS_CACHE)

    merged = copy.deepcopy(_DEFAULT_RUNTIME_SETTINGS)
    sources = ["base:defaults"]

    candidate_paths = settings_paths or ["config/settings.json", "config/settings.local.json"]
    for rel in candidate_paths:
        path = Path(rel)
        if not path.exists():
            continue
        try:
            payload = load_json_object(path)
            merged = _deep_merge_dict(merged, payload)
            sources.append(str(path))
        except (FileNotFoundError, ValueError):
            continue

    _RUNTIME_SETTINGS_CACHE = merged
    _RUNTIME_SETTINGS_SOURCES = sources
    return copy.deepcopy(merged)


def get_runtime_settings_sources() -> list[str]:
    """Return loaded runtime settings sources."""
    return list(_RUNTIME_SETTINGS_SOURCES)


def get_runtime_paths(*, force_reload: bool = False) -> Dict[str, str]:
    settings = load_runtime_settings(force_reload=force_reload)
    paths = settings.get("paths") if isinstance(settings, dict) else None
    if not isinstance(paths, dict):
        return {}
    return {str(k): str(v) for k, v in paths.items() if k and v}


def load_env_file(env_path: str | list[str] = None) -> Dict[str, str]:
    """Load environment variables from .env file(s).

    Later files override earlier ones.
    """
    env_vars: Dict[str, str] = {}

    paths: list[str]
    if env_path is None:
        paths = _DEFAULT_ENV_FILES
    elif isinstance(env_path, list):
        paths = env_path
    else:
        paths = [env_path]

    for p in paths:
        if not p:
            continue
        path_obj = Path(p)
        if not path_obj.exists():
            continue
        with open(path_obj, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    k = key.strip()
                    v = value.strip()
                    env_vars[k] = v
                    os.environ[k] = v

    return env_vars


def load_runtime_env(*, force_reload: bool = False) -> Dict[str, str]:
    """Load runtime env files based on runtime settings."""
    global _LOADED_ENV_FILES

    settings = load_runtime_settings(force_reload=force_reload)
    env_files = settings.get("env", {}).get("files") or list(_DEFAULT_ENV_FILES)
    if not isinstance(env_files, list):
        env_files = list(_DEFAULT_ENV_FILES)

    _LOADED_ENV_FILES = [str(p) for p in env_files]
    return load_env_file(_LOADED_ENV_FILES)


def get_loaded_env_files() -> list[str]:
    """Return configured env file loading order."""
    return list(_LOADED_ENV_FILES)


# Initialize runtime settings and env at module import.
load_runtime_settings()
_load_env_vars = load_runtime_env()


@dataclass(frozen=True)
class CompilerConfig:
    """Compiler configuration (immutable baseline; runtime counters bypass frozen)."""

    # Paths
    layer0_schema_path: str = "ontology_layer0_schema.json"
    local_ontology_path: str = "ontology_local.json"
    output_dir: str = "outputs/ontology"
    checkpoint_dir: str = "checkpoints"
    master_ontology_path: str = "outputs/ontology/master_ontology_v2.json"

    # Quality gates
    quality_gate_bronze: float = 0.6
    quality_gate_silver: float = 0.8
    quality_gate_gold: float = 0.95
    min_ingest_quality: str = "SILVER"

    # Validation
    validate_completeness: bool = True
    validate_consistency: bool = True
    validate_provenance: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: str = "compiler.log"
    enable_execution_log: bool = True

    # Performance
    max_batch_size: int = 100
    cache_enabled: bool = True
    cache_ttl: int = 3600

    # DeepSeek — None means "read from env at instantiation" (see __post_init__)
    deepseek_api_key: str | None = None
    deepseek_base_url: str | None = None
    deepseek_model: str | None = None
    deepseek_pro_model: str | None = None
    deepseek_timeout: int | None = None

    # DIFY LLM Configuration — None means "read from env at instantiation"
    dify_base_url: str | None = None
    dify_api_key: str | None = None
    dify_username: str | None = None
    dify_model: str | None = None
    dify_max_tokens: int | None = None
    dify_temperature: float | None = None
    dify_app_type: str | None = None
    dify_timeout_seconds: int | None = None
    dify_max_retries: int | None = None
    
    # DIFY PRO LLM Configuration — For complex tasks (e.g., DeepSeek V3.1)
    dify_pro_api_key: str | None = None
    dify_pro_model: str | None = None
    dify_pro_max_tokens: int | None = None
    dify_pro_temperature: float | None = None
    dify_pro_timeout_seconds: int | None = None

    # LLM Provider Selection — None means "read from env at instantiation"
    llm_provider: str | None = None
    llm_fallback_providers: str | None = None

    # Ontology RAG settings — None means "read from env at instantiation"
    ontology_rag_mode: str | None = None
    ontology_rag_llm_provider: str | None = None
    ontology_rag_strict_b: bool | None = None
    ontology_rag_require_l1: bool | None = None

    # RIGOR
    rigor_max_iterations: int = 5
    rigor_convergence_threshold: float = 0.85
    rigor_sample_size: int = 100
    # R1: Skip RIGOR if SemanticMD coverage exceeds threshold (0.0-1.0, default 0.75)
    # Set to 0.0 to always run RIGOR, 1.0 to skip RIGOR entirely
    skip_rigor_if_coverage_gt: float = 0.75
    # [R6 - RIGOR_THRESHOLD_CONFIG]: Additional thresholds loaded from YAML
    rigor_confidence_threshold: float = 0.85
    rigor_min_improvement: float = 0.05
    # Runtime tracking for loaded profile
    _rigor_profile: str = "default"
    _rigor_thresholds: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Resolve None fields from environment variables at instantiation time.

        Fields left as None (= not explicitly passed to constructor) are read
        from the current os.environ, so each CompilerConfig() reflects the live
        environment.  Explicitly-passed values (including empty string / False)
        are preserved as-is.
        """
        _set = object.__setattr__

        def _s(field: str, env_key: str, default: str) -> None:
            if getattr(self, field) is None:
                _set(self, field, os.getenv(env_key, default))

        def _i(field: str, env_key: str, default: int, fallback_key: str | None = None) -> None:
            if getattr(self, field) is None:
                raw = os.getenv(env_key) or (os.getenv(fallback_key) if fallback_key else None) or str(default)
                _set(self, field, int(raw))

        def _f(field: str, env_key: str, default: float) -> None:
            if getattr(self, field) is None:
                _set(self, field, float(os.getenv(env_key, str(default))))

        def _b(field: str, env_key: str, default_str: str) -> None:
            if getattr(self, field) is None:
                _set(self, field, os.getenv(env_key, default_str).strip().lower() in {"1", "true", "yes"})

        _s("deepseek_api_key",        "DEEPSEEK_API_KEY",        "")
        _s("deepseek_base_url",       "DEEPSEEK_BASE_URL",      "")
        _s("deepseek_model",          "DEEPSEEK_MODEL",         "deepseek-chat")
        _s("deepseek_pro_model",      "DEEPSEEK_PRO_MODEL",     "")
        _i("deepseek_timeout",        "DEEPSEEK_TIMEOUT",        120, "DEEPSEEK_TIMEOUT_SECONDS")
        _s("dify_base_url",           "DIFY_BASE_URL",           "")
        _s("dify_api_key",            "DIFY_API_KEY",            "")
        _s("dify_username",           "DIFY_USERNAME",           "")
        _s("dify_model",              "DIFY_MODEL",              "dify-model")
        _i("dify_max_tokens",         "DIFY_MAX_TOKENS",         3000)
        _f("dify_temperature",        "DIFY_TEMPERATURE",        0.1)
        _s("dify_app_type",           "DIFY_APP_TYPE",           "chat")
        _i("dify_timeout_seconds",    "DIFY_TIMEOUT_SECONDS",    300)
        _i("dify_max_retries",        "DIFY_MAX_RETRIES",        3)
        
        _s("dify_pro_api_key",        "DIFY_PRO_API_KEY",        "")
        _s("dify_pro_model",          "DIFY_PRO_MODEL",          "")
        _i("dify_pro_max_tokens",     "DIFY_PRO_MAX_TOKENS",     4000)
        _f("dify_pro_temperature",    "DIFY_PRO_TEMPERATURE",    0.1)
        _i("dify_pro_timeout_seconds","DIFY_PRO_TIMEOUT_SECONDS",300)
        
        _s("llm_provider",            "LLM_PROVIDER",            "dify")
        _s("llm_fallback_providers",   "LLM_FALLBACK_PROVIDERS",   "")
        _s("ontology_rag_mode",       "ONTOLOGY_RAG_MODE",       "a")
        _s("ontology_rag_llm_provider", "ONTOLOGY_RAG_LLM_PROVIDER", "")
        _b("ontology_rag_strict_b",   "ONTOLOGY_RAG_STRICT_B",   "")
        _b("ontology_rag_require_l1", "ONTOLOGY_RAG_REQUIRE_L1", "1")

        # [R6 - RIGOR_THRESHOLD_CONFIG]: Load domain-specific RIGOR thresholds from YAML
        self._load_rigor_thresholds(_set)

    def _load_rigor_thresholds(self, _set) -> None:
        """Load RIGOR thresholds from config/rigor_thresholds.yaml.
        
        [R6 - RIGOR_THRESHOLD_CONFIG]:
        - Reads domain profile from RIGOR_DOMAIN env var (default: 'default')
        - Overrides hardcoded defaults with YAML values
        - Falls back to hardcoded values if YAML is missing/invalid
        """
        if yaml is None:
            return  # PyYAML not available, use hardcoded defaults
        
        threshold_path = Path("config/rigor_thresholds.yaml")
        if not threshold_path.exists():
            return  # Use hardcoded defaults
        
        try:
            with open(threshold_path, "r", encoding="utf-8") as f:
                thresholds = yaml.safe_load(f)
            
            if not isinstance(thresholds, dict):
                return
            
            # Select profile: RIGOR_DOMAIN env var -> 'default'
            domain = os.getenv("RIGOR_DOMAIN", "default").lower()
            profile = thresholds.get(domain) or thresholds.get("default", {})
            
            if not isinstance(profile, dict):
                return
            
            # Map YAML keys to CompilerConfig field names
            field_map = {
                "max_iterations": "rigor_max_iterations",
                "convergence_threshold": "rigor_convergence_threshold",
                "sample_size": "rigor_sample_size",
                "skip_if_coverage_gt": "skip_rigor_if_coverage_gt",
                "confidence_threshold": "rigor_confidence_threshold",
                "min_improvement": "rigor_min_improvement",
            }
            
            for yaml_key, field_name in field_map.items():
                if yaml_key in profile and hasattr(self, field_name):
                    value = profile[yaml_key]
                    # Type conversion based on existing field type
                    current = getattr(self, field_name)
                    if isinstance(current, int):
                        _set(self, field_name, int(value))
                    elif isinstance(current, float):
                        _set(self, field_name, float(value))
                    else:
                        _set(self, field_name, value)
            
            # Store full profile for runtime access
            _set(self, "_rigor_profile", domain)
            _set(self, "_rigor_thresholds", profile)
            
        except Exception as e:
            # Log but don't fail - use hardcoded defaults
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to load RIGOR thresholds from {threshold_path}: {e}. "
                "Using hardcoded defaults."
            )

    runtime_llm_usage: Dict[str, float] = field(
        default_factory=lambda: {
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "total_tokens": 0.0,
            "llm_calls": 0.0,
            "cost": 0.0,
        }
    )

    def reset_runtime_llm_usage(self) -> None:
        """Reset LLM counters; bypasses frozen restriction via object.__setattr__."""
        object.__setattr__(
            self,
            "runtime_llm_usage",
            {
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "total_tokens": 0.0,
                "llm_calls": 0.0,
                "cost": 0.0,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paths": {
                "layer0_schema": self.layer0_schema_path,
                "local_ontology": self.local_ontology_path,
                "output_dir": self.output_dir,
                "checkpoint_dir": self.checkpoint_dir,
            },
            "quality_gates": {
                "bronze": self.quality_gate_bronze,
                "silver": self.quality_gate_silver,
                "gold": self.quality_gate_gold,
                "min_ingest": self.min_ingest_quality,
            },
            "validation": {
                "completeness": self.validate_completeness,
                "consistency": self.validate_consistency,
                "provenance": self.validate_provenance,
            },
            "logging": {
                "level": self.log_level,
                "file": self.log_file,
                "execution_log": self.enable_execution_log,
            },
            "performance": {
                "max_batch_size": self.max_batch_size,
                "cache_enabled": self.cache_enabled,
                "cache_ttl": self.cache_ttl,
            },
            "deepseek": {
                "model": self.deepseek_model,
                "timeout": self.deepseek_timeout,
            },
            "dify": {
                "base_url": self.dify_base_url,
                "api_key": self.dify_api_key,
                "username": self.dify_username,
                "model": self.dify_model,
                "max_tokens": self.dify_max_tokens,
                "temperature": self.dify_temperature,
                "app_type": self.dify_app_type,
            },
            "llm": {
                "provider": self.llm_provider,
            },
            "rigor": {
                "max_iterations": self.rigor_max_iterations,
                "convergence_threshold": self.rigor_convergence_threshold,
                "sample_size": self.rigor_sample_size,
            },
        }


_CONFIG_OVERRIDES: dict = {}


def get_config() -> CompilerConfig:
    """Return a CompilerConfig built fresh from the current environment.

    Each call re-reads os.getenv so that test monkeypatching and runtime
    env changes (e.g. load_runtime_env()) are reflected immediately.
    Any values set via update_config() are applied on top as overrides.
    """
    base = CompilerConfig()
    if _CONFIG_OVERRIDES:
        valid = {k: v for k, v in _CONFIG_OVERRIDES.items() if hasattr(base, k)}
        return dataclasses.replace(base, **valid)
    return base


def update_config(**kwargs) -> None:
    """Override specific config fields for the lifetime of the process.

    Overrides are applied on top of the live environment on each get_config() call.
    """
    valid_keys = {f.name for f in dataclasses.fields(CompilerConfig)}
    for k, v in kwargs.items():
        if k in valid_keys:
            _CONFIG_OVERRIDES[k] = v


DEFAULT_CONFIG = get_config()
