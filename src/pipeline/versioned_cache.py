"""Versioned cache key resolution for C-4.

Resolves a :class:`CacheVersionBundle` from environment + pipeline context so
that every cache layer (document cache, LLM adapter cache) shares the same
version dimensions. A version bump in any dimension invalidates all cached
results — this is the mechanism behind "cache hit 不会错误复用陈旧结果".

Version dimensions (per C-4 spec):
- ``prompt_version``  — prompt template revision (``PROMPT_VERSION``)
- ``schema_version``  — ontology schema revision (``SCHEMA_VERSION``)
- ``policy_version``  — extraction/governance policy revision
                       (``POLICY_VERSION`` or ``SKILL_VERSION``)
- ``model_family``    — LLM model family (``DEEPSEEK_MODEL`` /
                        ``DIFY_MODEL`` / ``LLM_PROVIDER``)
- ``extraction_mode`` — pipeline execution mode (production/degraded/offline)
"""
from __future__ import annotations

import os
from typing import Any, Optional

from core.contracts import CacheVersionBundle


def _env(*keys: str, default: str = "") -> str:
    """Return the first non-empty env value among *keys*."""
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return default


def resolve_model_family(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Resolve the model-family dimension for the version bundle.

    Prefers an explicit ``LLM_MODEL_FAMILY`` env, then the caller-supplied
    *model* label, then the provider-configured model env vars, and finally
    the provider name. All resolve to a non-empty string so the dimension is
    always present in the cache key.
    """
    explicit = _env("LLM_MODEL_FAMILY")
    if explicit:
        return explicit
    if model:
        return str(model)
    configured = _env("DEEPSEEK_MODEL", "DIFY_MODEL")
    if configured:
        return configured
    if provider:
        return str(provider)
    return _env("LLM_PROVIDER", default="unknown")


def resolve_version_bundle(
    *,
    prompt_version: Optional[str] = None,
    schema_version: Optional[str] = None,
    policy_version: Optional[str] = None,
    model_family: Optional[str] = None,
    extraction_mode: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> CacheVersionBundle:
    """Resolve a :class:`CacheVersionBundle` from explicit args + env.

    Explicit args take precedence; env vars fill the gaps. This is the single
    entry point used by the document cache and the LLM adapter so both layers
    share identical version semantics.
    """
    return CacheVersionBundle(
        prompt_version=prompt_version or _env("PROMPT_VERSION"),
        schema_version=schema_version or _env("SCHEMA_VERSION"),
        policy_version=policy_version or _env("POLICY_VERSION", "SKILL_VERSION"),
        model_family=model_family or resolve_model_family(provider=provider, model=model),
        extraction_mode=extraction_mode or _env(
            "PIPELINE_EXECUTION_MODE", default="production"
        ),
    )


def resolve_version_bundle_from_context(context: Any) -> CacheVersionBundle:
    """Resolve a version bundle from a :class:`PipelineContext`-like object.

    Reads ``prompt_version`` / ``schema_version`` / ``skill_version`` /
    ``execution_mode`` attributes (falling back to env) so the orchestrator
    can build a bundle without duplicating env-lookup logic.
    """
    prompt_version = getattr(context, "prompt_version", "") or None
    schema_version = getattr(context, "schema_version", "") or None
    skill_version = getattr(context, "skill_version", "") or None
    execution_mode = getattr(context, "execution_mode", "") or None

    return resolve_version_bundle(
        prompt_version=prompt_version,
        schema_version=schema_version,
        policy_version=skill_version,
        extraction_mode=execution_mode,
    )


__all__ = [
    "resolve_version_bundle",
    "resolve_version_bundle_from_context",
    "resolve_model_family",
]
