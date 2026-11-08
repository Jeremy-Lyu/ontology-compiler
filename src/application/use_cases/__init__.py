"""Application layer use cases.

BATCH 0 Consolidation (2026-06-29):
- Unified exports for canonical use cases
- Marked deprecated modules
- Stage files consolidated into pipeline_use_cases.py

BATCH 5 CONSOLIDATION (2026-07-12):
- Pipeline use cases consolidated into pipeline_use_cases.py
- Replay use cases consolidated into replay_use_cases.py
- Summary use cases consolidated into summary_use_cases.py

Migration note:
- New grouping packages (`pipeline/`, `replay/`, `summary/`) provide clearer
  boundaries while keeping imports stable.

Phase 4 (2026-08-12):
- `analyze_folder_once` is a legacy module whose archived adapter dependency
  (`src.adapters.folder_analysis_service_adapter`) no longer exists; eagerly
  importing it crashed the whole package (import-time `ModuleNotFoundError`).
  The import is now guarded so the remaining (pipeline/replay/summary) use-case
  exports stay importable. `FolderAnalysisRequest`/`HookRunner` are removed from
  `__all__`; they remain reachable only via the legacy module itself, which
  still fails to import and is slated for retirement (Phase 6).
"""
import logging

from src.application.services.unified_fingerprint import folder_fingerprint
from core.utils.file_helpers import slug_text

# Document processing
from .discover_folder_documents import DiscoverFolderDocumentsRequest, DiscoverFolderDocumentsUseCase

# Legacy folder-analysis entry — archived adapter dependency, guarded (Phase 4).
try:
    from .analyze_folder_once import FolderAnalysisRequest, HookRunner
except ImportError:  # pragma: no cover - legacy module with archived dependency
    logger = logging.getLogger(__name__)
    logger.warning(
        "analyze_folder_once is unavailable (archived folder_analysis_service_adapter); "
        "FolderAnalysisRequest/HookRunner re-exports disabled"
    )
    FolderAnalysisRequest = None  # type: ignore[assignment]
    HookRunner = None  # type: ignore[assignment]

# Pipeline use cases (consolidated)
# NOTE (Phase 6 retirement, 2026-08-13; archived 2026-08-26): the legacy
# stage-execution stack (`pipeline_use_cases` / `_stage_execution` /
# `stage_loop_with_hooks_use_case`) had zero external consumers and has been
# ARCHIVED together with its characterization test to
# `archived/legacy_stage_stack_20260826/`. Do not re-add callers here; see
# docs/CODE_CLEANUP_CANDIDATES.md for the retirement ledger.

# Replay use cases (consolidated)
from .replay_use_cases import (
    ReplayRequest,
    ReplayHandlerResult,
    ReplayOutputAssembler,
    OrchestratorReplayHandler,
    replay_pipeline_session,
    build_replay_sse_event_name,
    build_replay_output_metadata,
    build_replay_summary_sse_payload,
)

# Summary use cases (consolidated)
from .summary_use_cases import (
    build_pipeline_run_payload,
    build_pipeline_summary,
    build_pipeline_summary_paths,
    write_pipeline_summary_artifacts,
)

__all__ = [
    # Core folder analysis
    "folder_fingerprint",
    "slug_text",
    # Document processing
    "DiscoverFolderDocumentsRequest",
    "DiscoverFolderDocumentsUseCase",
    # Replay use cases
    "ReplayRequest",
    "ReplayHandlerResult",
    "ReplayOutputAssembler",
    "OrchestratorReplayHandler",
    "replay_pipeline_session",
    "build_replay_sse_event_name",
    "build_replay_output_metadata",
    "build_replay_summary_sse_payload",
    # Summary use cases
    "build_pipeline_run_payload",
    "build_pipeline_summary",
    "build_pipeline_summary_paths",
    "write_pipeline_summary_artifacts",
]
