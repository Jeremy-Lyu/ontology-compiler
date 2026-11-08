"""Unified cache-key version bundle (C-4 / P1 standardization).

Defines the single canonical :class:`CacheVersionBundle` and the single
canonical cache-key function :func:`compute_versioned_cache_key` used by
*every* cache layer across the codebase — document cache, LLM adapter
cache, chunk cache, artifact index, etc.

This lives in the neutral ``core.contracts`` layer so both
``ontology_compiler`` and the application layer can depend on it without
crossing the application import boundary (Phase 2 dependency 收口).

Version dimensions (C-4 spec):
    - ``prompt_version``   — prompt template revision
    - ``schema_version``   — ontology schema revision
    - ``policy_version``   — extraction / governance policy revision
    - ``model_family``     — LLM model family
    - ``extraction_mode``  — pipeline execution mode (production / degraded / offline)

A change in *any* dimension invalidates all cached results — this is the
mechanism behind "cache hit 不会错误复用陈旧结果".
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional


# Unit separator — prevents ambiguity when concatenating variable-length
# string parts (e.g. "ab" + "c" vs "a" + "bc").
_UNIT_SEP = "\x1f"


@dataclass(frozen=True)
class CacheVersionBundle:
    """All dimensions that must match for a cache hit to be valid (C-4).

    If *any* of these differs from the cached run, the result is considered
    stale and the stage must re-run. This is the explicit version key that
    prevents "cache hit 不会错误复用陈旧结果".
    """

    prompt_version: str = ""
    schema_version: str = ""
    policy_version: str = ""
    model_family: str = ""
    extraction_mode: str = ""

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, str]:
        """Return a plain dict suitable for JSON serialisation."""
        return {
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "model_family": self.model_family,
            "extraction_mode": self.extraction_mode,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "CacheVersionBundle":
        """Build a bundle from a (possibly partial / None) dict.

        Missing keys default to empty string so the bundle is always
        well-formed — an empty bundle is valid and just means "no version
        information available" (treated as stale by :func:`version_bundles_match`).
        """
        if not data:
            return cls()
        return cls(
            prompt_version=str(data.get("prompt_version", "")),
            schema_version=str(data.get("schema_version", "")),
            policy_version=str(data.get("policy_version", "")),
            model_family=str(data.get("model_family", "")),
            extraction_mode=str(data.get("extraction_mode", "")),
        )

    def is_empty(self) -> bool:
        """True when *no* version dimension has been set.

        An empty bundle typically means "first run" or "caller didn't
        supply version info". Cache layers should treat an empty cached
        bundle as stale so the next run stamps proper version info.
        """
        return not any(
            v
            for v in (
                self.prompt_version,
                self.schema_version,
                self.policy_version,
                self.model_family,
                self.extraction_mode,
            )
        )

    def to_fingerprint(self) -> str:
        """Return a short stable hash of this bundle.

        Useful when you need a single string to mix into another hash
        (e.g. chunk cache keys) instead of the full structured bundle.
        """
        parts = [
            self.prompt_version or "",
            self.schema_version or "",
            self.policy_version or "",
            self.model_family or "",
            self.extraction_mode or "",
        ]
        return hashlib.sha256(
            _UNIT_SEP.join(parts).encode("utf-8")
        ).hexdigest()[:16]


def version_bundles_match(
    current: CacheVersionBundle,
    cached: Optional[CacheVersionBundle],
) -> bool:
    """True only when every version dimension matches exactly (C-4).

    A missing / empty cached bundle (first run, or pre-C-4 legacy entry)
    never matches — the stage must run so it gets stamped with proper
    version info.
    """
    if cached is None:
        return False
    if cached.is_empty():
        return False
    return current == cached


def compute_versioned_cache_key(
    source_chunk: str,
    normalized_prompt: str,
    version_bundle: CacheVersionBundle,
) -> str:
    """Compute the canonical C-4 versioned cache key.

    ``key = sha256(source_chunk + normalized_prompt + prompt_version +
    schema_version + policy_version + model_family + extraction_mode)``

    This is the *single* cache-key function used by every cache layer
    (document cache, LLM adapter cache, chunk cache, …) so that a version
    bump invalidates all layers consistently.
    """
    parts = [
        source_chunk or "",
        normalized_prompt or "",
        version_bundle.prompt_version or "",
        version_bundle.schema_version or "",
        version_bundle.policy_version or "",
        version_bundle.model_family or "",
        version_bundle.extraction_mode or "",
    ]
    payload = _UNIT_SEP.join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "CacheVersionBundle",
    "compute_versioned_cache_key",
    "version_bundles_match",
]
