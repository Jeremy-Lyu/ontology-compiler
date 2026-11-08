"""Neutral cross-layer contracts.

Holds shared contract primitives that both `ontology_compiler` and the
application layer can depend on without crossing the application import
boundary (Phase 2 dependency收口).
"""

from .ports import *  # noqa: F401,F403
from .version_keys import *  # noqa: F401,F403

__all__ = [
    "StoredArtifact",
    "StoredArtifactSet",
    "PipelineSummaryPaths",
    "HookResult",
    "ArtifactStorePort",
    "CapsuleQueryPort",
    "HookRunnerPort",
    "IdentityRegistryPort",
    "EvidencePlanePort",
    # C-4 / P1 unified cache key
    "CacheVersionBundle",
    "compute_versioned_cache_key",
    "version_bundles_match",
]