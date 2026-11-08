"""Folder-analysis bounded-context port definitions.

Ports and DTOs for the folder analysis workflow (including session
replay support). Split from `all_ports.py` during P2 port
classification (§9.8).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class FolderAnalysisRequest:
    """Request for folder analysis."""
    folder_path: str
    domain: str
    run_id: str
    topic_id: str
    profile: str = "local"
    source_profile: str | None = None
    max_docs: int | None = None
    include_archived: bool = False
    with_capsules: bool = False
    # Additional legacy parameters
    legacy_params: dict[str, Any] | None = None


@dataclass
class FolderAnalysisResult:
    """Result from folder analysis."""
    finalization_bundle: dict[str, Any]
    analysis_context: dict[str, Any]
    success: bool
    error_message: str | None = None


class FolderAnalysisServicePort:
    """Port for folder analysis service operations."""

    async def analyze_folder_once(
        self,
        request: FolderAnalysisRequest,
        return_finalization_bundle: bool = True
    ) -> FolderAnalysisResult:
        """Analyze a folder and return analysis results."""
        raise NotImplementedError

    async def save_capsule_experience(
        self,
        topic_id: str,
        domain: str,
        run_id: str,
        write_stage_context: dict[str, Any],
        capsule_result: dict[str, Any]
    ) -> None:
        """Save capsule experience from analysis."""
        raise NotImplementedError


class SessionReplayStorePort(ABC):
    """Port for loading persisted replay sessions by session_id."""

    @abstractmethod
    def load(self, session_id: str) -> dict[str, Any]:
        """Load a replay session payload for the given session id."""
        raise NotImplementedError


__all__ = [
    "FolderAnalysisRequest",
    "FolderAnalysisResult",
    "FolderAnalysisServicePort",
    "SessionReplayStorePort",
]
