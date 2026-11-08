"""Artifact bounded-context port definitions.

Ports and DTOs for persisting/querying run artifacts, capsules and
storage. The neutral compiler-facing contracts live in
`core.contracts.ports`; they are re-exported here so application-layer
imports share the canonical class identity. Split from `all_ports.py`
during P2 port classification (§9.8).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from core.contracts.ports import (
    ArtifactStorePort,
    CapsuleQueryPort,
    PipelineSummaryPaths,
    StoredArtifact,
    StoredArtifactSet,
)


class StoragePort(ABC):
    """Port for ontology artifact storage."""

    @abstractmethod
    def write_json_artifact(
        self,
        run_id: str,
        artifact_name: str,
        data: dict[str, Any]
    ) -> Path:
        """Write JSON artifact and return path."""
        raise NotImplementedError

    @abstractmethod
    def write_markdown_artifact(
        self,
        run_id: str,
        artifact_name: str,
        content: str
    ) -> Path:
        """Write Markdown artifact and return path."""
        raise NotImplementedError

    @abstractmethod
    def read_artifact(
        self,
        run_id: str,
        artifact_name: str
    ) -> dict[str, Any] | str | None:
        """Read artifact by name."""
        raise NotImplementedError


__all__ = [
    "ArtifactStorePort",
    "CapsuleQueryPort",
    "PipelineSummaryPaths",
    "StoredArtifact",
    "StoredArtifactSet",
    "StoragePort",
]
