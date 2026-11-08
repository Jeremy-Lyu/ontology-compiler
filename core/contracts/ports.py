"""Neutral application-port contracts (compiler-facing).

Holds the port interfaces and their small DTOs that `ontology_compiler`
depends on. These live in the neutral `core/contracts` layer so that
`ontology_compiler` no longer imports across the application boundary
(Phase 2 / P2-04: broker compiler's direct `src.application.ports.all_ports`
imports into a neutral contract layer).

`src/application/ports/all_ports.py` re-exports these names so that existing
application-layer imports keep working unchanged and class identity is shared
with the canonical definitions here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class StoredArtifact:
    """Description of one stored artifact."""

    name: str
    path: str
    payload: Any | None = None


@dataclass(frozen=True)
class StoredArtifactSet:
    """Paths returned after storing a run artifact set."""

    run_dir: str
    manifest_path: str
    artifacts: list[StoredArtifact] = field(default_factory=list)


@dataclass(frozen=True)
class PipelineSummaryPaths:
    """Canonical and legacy locations for orchestrator pipeline summaries."""

    summary_path: Path
    canonical_latest_path: Path
    legacy_latest_path: Path


@dataclass(frozen=True)
class HookResult:
    """Result of a hook execution."""
    allowed: bool
    exit_code: int
    reason: str
    output: str


# =============================================================================
# Port Interfaces
# =============================================================================

class ArtifactStorePort(ABC):
    """Interface for persisting run artifacts and manifests."""

    @abstractmethod
    def write_run_artifact_set(
        self,
        *,
        run_id: str,
        manifest: dict[str, Any],
        artifacts: dict[str, Any],
    ) -> StoredArtifactSet:
        """Persist a manifest plus arbitrary JSON/text artifacts for one run."""
        raise NotImplementedError

    @abstractmethod
    def write_artifacts(
        self,
        *,
        run_id: str,
        artifacts: dict[str, Any],
    ) -> list[StoredArtifact]:
        """Persist additional artifacts into an existing run directory."""
        raise NotImplementedError

    @abstractmethod
    def build_pipeline_summary_paths(self, *, run_id: str) -> PipelineSummaryPaths:
        """Return canonical and legacy paths for orchestrator pipeline summaries."""
        raise NotImplementedError

    @abstractmethod
    def write_pipeline_summary(
        self,
        summary: dict[str, Any],
        *,
        paths: PipelineSummaryPaths,
        legacy_warning_label: str,
    ) -> None:
        """Persist orchestrator summary to canonical outputs and best-effort legacy path."""
        raise NotImplementedError


class CapsuleQueryPort(ABC):
    """Port for querying historical capsule experience."""

    @abstractmethod
    async def query_capsules(self, query: str, limit: int = 5) -> list[Any]:
        """Search for relevant capsules based on text query."""
        raise NotImplementedError


class HookRunnerPort(ABC):
    """Interface for running lifecycle hooks (e.g. PreToolUse, PostToolUse)."""

    @abstractmethod
    def run(self, event: str, payload: dict[str, Any]) -> HookResult:
        """Execute all registered hooks for the given event name."""
        raise NotImplementedError


class IdentityRegistryPort(ABC):
    """Port for identity and DID registry operations.

    Provides an abstract interface for managing decentralized identities (DIDs)
    and their associated roles, trust levels, and scopes.
    """

    @abstractmethod
    def load(self) -> dict[str, Any]:
        """Load identity registry data.

        Returns:
            Dictionary containing schema_version and identities list.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, payload: dict[str, Any]) -> None:
        """Save identity registry data.

        Args:
            payload: Registry data to persist.
        """
        raise NotImplementedError

    @abstractmethod
    def invalidate_cache(self) -> None:
        """Force next load() to re-read from disk."""
        raise NotImplementedError

    @abstractmethod
    def get_identity(self, subject_id: str) -> dict[str, Any] | None:
        """Get identity record by subject_id.

        Args:
            subject_id: The subject identifier.

        Returns:
            Identity record or None if not found.
        """
        raise NotImplementedError

    @abstractmethod
    def check_scope(self, subject_id: str, scope: str) -> bool:
        """Check if identity has the specified scope.

        Args:
            subject_id: The subject identifier.
            scope: The scope (e.g., "l1", "l2", "publish").

        Returns:
            True if identity has the scope.
        """
        raise NotImplementedError


class EvidencePlanePort(ABC):
    """Port for evidence lookup and indexing over generated L1/L2 artifacts.

    Provides an abstract interface for building evidence indexes and tracing
    entities back to their source files and runs.
    """

    @abstractmethod
    def build_index(self, *, runs_root: Path) -> dict[str, Any]:
        """Scan all runs and build evidence index.

        Args:
            runs_root: Root directory containing run artifacts.

        Returns:
            Dictionary mapping entity URIs to their evidence entries.
        """
        raise NotImplementedError

    @abstractmethod
    def trace_to_source(self, entity_uri: str, *, runs_root: Path) -> list[dict[str, Any]]:
        """Trace an entity URI to its source evidence.

        Args:
            entity_uri: The entity URI to trace.
            runs_root: Root directory containing run artifacts.

        Returns:
            List of evidence entries for the entity.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_fingerprint(self, run_id: str, expected_fingerprint: str, *, runs_root: Path) -> bool:
        """Validate run content fingerprint for cache consistency.

        Args:
            run_id: The run identifier.
            expected_fingerprint: Expected fingerprint hash.
            runs_root: Root directory containing run artifacts.

        Returns:
            True if fingerprint matches.
        """
        raise NotImplementedError

    def trace_to_source_with_locator(
        self,
        entity_uri: str,
        *,
        runs_root: Path,
        source_text: str = "",
        file_path: Optional[Path] = None,
    ) -> list[dict[str, Any]]:
        """Trace an entity URI to its source evidence with fine-grained position mapping.

        [Phase 2 - EVIDENCE_BRIDGE]: Bridges inter-run evidence tracing with intra-document
        position mapping to provide precise char_offset, line_range, and context information.

        Args:
            entity_uri: The entity URI to trace.
            runs_root: Root directory containing run artifacts.
            source_text: Optional source text for fine-grained position mapping.
                If provided, EvidenceLocator will be used to compute precise positions.
            file_path: Optional file path for EvidenceLocator context.

        Returns:
            List of evidence entries, potentially enriched with fine-grained position info.
        """
        # Default implementation: fall back to trace_to_source
        return self.trace_to_source(entity_uri, runs_root=runs_root)


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
]