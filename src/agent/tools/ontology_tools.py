"""Ontology-specific Agent Tools (U4.1 → U4.4).

Wraps the existing Application Use Cases as Agent-native Tools that conform
to the AgentTool Protocol and the U3.5 FolderAnalysisToolPort interface.

Available tools
---------------
QueryCapsuleTool            — query historical capsule experience (U4.4)
FolderAnalysisTool          — wraps ``analyze_folder_once``, full folder analysis
DiscoverDocumentsTool       — document discovery without LLM analysis
SelectiveFolderAnalysisTool — top-N file analysis for large folders (U4.3)
ExtractFragmentsTool        — single-document deep inspection (U4.3)

Dynamic strategy (U4.3)
-----------------------
When ``discover_documents`` returns > LARGE_FOLDER_THRESHOLD files, the Agent
should switch from ``analyze_folder`` to ``selective_folder_analysis`` with an
appropriate ``max_files`` cap, or call ``extract_fragments`` on individual key
files for deep inspection before running a partial analysis.

Memory strategy (U4.4)
----------------------
Always call ``query_capsule`` first when the user provides a domain / topic
keyword.  Use the returned summaries to decide whether a fresh analysis is
necessary or whether the existing capsule insights are sufficient.
"""
from __future__ import annotations

import logging
from typing import Any

from src.agent.tools.base import AgentTool, ToolResult  # noqa: F401 (re-export for callers)
from src.agent.tools.capsule_tools import QueryCapsuleTool  # noqa: F401 (re-export for callers)
from src.application.ports.artifact_ports import CapsuleQueryPort
from src.domain.models.folder_analysis_result import FolderAnalysisResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Dynamic strategy threshold (U4.3)
# ──────────────────────────────────────────────────────────────────────────────

#: When a folder has more files than this, prefer ``selective_folder_analysis``
#: or ``extract_fragments`` over full ``analyze_folder`` to stay within budget.
LARGE_FOLDER_THRESHOLD: int = 20


# ──────────────────────────────────────────────────────────────────────────────
# Tool: analyze_folder
# ──────────────────────────────────────────────────────────────────────────────


class FolderAnalysisTool:
    """Analyze a document folder and extract ontology entities.

    Thin wrapper around ``analyze_folder_once`` conforming to the
    :class:`AgentTool` Protocol.

    .. tip::
        For folders with more than LARGE_FOLDER_THRESHOLD files prefer
        ``selective_folder_analysis`` to avoid token-budget overruns.
    """

    name: str = "analyze_folder"
    description: str = (
        "Analyze a document folder and extract ontology entities (nodes, relationships, "
        "instances). Returns a structured FolderAnalysisResult containing quality grade, "
        "artifact paths, LLM usage, and a domain-level entity graph. "
        f"CAUTION: use selective_folder_analysis instead when folder has >{LARGE_FOLDER_THRESHOLD} files."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "folder_path": {
                "type": "string",
                "description": "Absolute or relative path to the document folder to analyze.",
            },
            "domain": {
                "type": "string",
                "description": "Domain label applied to the extraction (e.g. SUPPLY_CHAIN).",
                "default": "SUPPLY_CHAIN",
            },
            "fast_mode": {
                "type": "boolean",
                "description": "Skip RIGOR refinement for speed. Recommended for agent loops.",
                "default": True,
            },
            "max_files": {
                "type": "integer",
                "description": "Maximum number of files to process (None = all).",
            },
        },
        "required": ["folder_path"],
    }

    def __init__(self, run_id_prefix: str = "agent") -> None:
        self._run_id_prefix = run_id_prefix

    async def run(
        self,
        folder_path: str,
        domain: str = "SUPPLY_CHAIN",
        fast_mode: bool = True,
        max_files: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute folder analysis and return ToolResult."""
        import uuid
        from src.application.use_cases.analyze_folder_once import analyze_folder_once, FolderAnalysisRequest
        from src.adapters.outbound.hooks.hook_runner_adapter import HookRunnerAdapter

        run_id = f"{self._run_id_prefix}-{uuid.uuid4().hex[:8]}"
        logger.info(
            "FolderAnalysisTool.run: folder=%s domain=%s fast=%s run_id=%s",
            folder_path, domain, fast_mode, run_id,
        )
        try:
            result: FolderAnalysisResult = await analyze_folder_once(
                FolderAnalysisRequest(
                    folder_path=folder_path,
                    domain=domain,
                    fast_mode=fast_mode,
                    max_files=max_files,
                    run_id=run_id,
                    hook_runner=HookRunnerAdapter(),
                )
            )
            return ToolResult(
                success=result.status.value == "SUCCESS",
                content=result.to_dict(),
                tool_name=self.name,
                error="; ".join(result.errors) if result.errors else None,
                metadata={
                    "run_id": result.run_id,
                    "quality_grade": result.quality.grade,
                    "status": result.status.value,
                },
            )
        except Exception as exc:
            logger.exception("FolderAnalysisTool failed: %s", exc)
            return ToolResult(
                success=False,
                content=None,
                tool_name=self.name,
                error=str(exc),
            )


# ──────────────────────────────────────────────────────────────────────────────
# Tool: selective_folder_analysis  (U4.3 — dynamic strategy)
# ──────────────────────────────────────────────────────────────────────────────


class SelectiveFolderAnalysisTool:
    """Analyze only the top-N most important files in a large folder.

    Use when ``discover_documents`` reports more than ``LARGE_FOLDER_THRESHOLD``
    files.  Runs ``analyze_folder_once`` with a hard ``max_files`` cap so the
    agent stays within its token budget while still extracting meaningful
    ontology entities from the core documents.
    """

    name: str = "selective_folder_analysis"
    description: str = (
        f"Analyze only the top-N files in a folder (recommended when file count > {LARGE_FOLDER_THRESHOLD}). "
        "Requires an explicit max_files cap. Preferred over analyze_folder for large or "
        "mixed-content repositories. Always runs in fast_mode to keep token budget low."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "folder_path": {
                "type": "string",
                "description": "Path to the document folder.",
            },
            "max_files": {
                "type": "integer",
                "description": (
                    "Maximum number of files to process. "
                    f"Recommended: 5–{LARGE_FOLDER_THRESHOLD} for large folders."
                ),
                "default": 10,
            },
            "domain": {
                "type": "string",
                "description": "Domain label (e.g. SUPPLY_CHAIN).",
                "default": "SUPPLY_CHAIN",
            },
        },
        "required": ["folder_path"],
    }

    def __init__(self, run_id_prefix: str = "agent") -> None:
        self._run_id_prefix = run_id_prefix

    async def run(
        self,
        folder_path: str,
        max_files: int = 10,
        domain: str = "SUPPLY_CHAIN",
        **kwargs: Any,
    ) -> ToolResult:
        """Execute selective (top-N) folder analysis in fast_mode."""
        import uuid
        from src.application.use_cases.analyze_folder_once import analyze_folder_once, FolderAnalysisRequest
        from src.adapters.outbound.hooks.hook_runner_adapter import HookRunnerAdapter

        effective_max = max(1, int(max_files))
        run_id = f"{self._run_id_prefix}-sel-{uuid.uuid4().hex[:8]}"
        logger.info(
            "SelectiveFolderAnalysisTool.run: folder=%s max_files=%d run_id=%s",
            folder_path, effective_max, run_id,
        )
        try:
            result: FolderAnalysisResult = await analyze_folder_once(
                FolderAnalysisRequest(
                    folder_path=folder_path,
                    domain=domain,
                    fast_mode=True,   # selective always uses fast mode
                    max_files=effective_max,
                    run_id=run_id,
                    hook_runner=HookRunnerAdapter(),
                )
            )
            return ToolResult(
                success=result.status.value == "SUCCESS",
                content=result.to_dict(),
                tool_name=self.name,
                error="; ".join(result.errors) if result.errors else None,
                metadata={
                    "run_id": result.run_id,
                    "quality_grade": result.quality.grade,
                    "status": result.status.value,
                    "files_analyzed": effective_max,
                    "strategy": "selective",
                },
            )
        except Exception as exc:
            logger.exception("SelectiveFolderAnalysisTool failed: %s", exc)
            return ToolResult(
                success=False,
                content=None,
                tool_name=self.name,
                error=str(exc),
            )


# ──────────────────────────────────────────────────────────────────────────────
# Tool: discover_documents
# ──────────────────────────────────────────────────────────────────────────────


class DiscoverDocumentsTool:
    """Discover readable documents in a folder without running LLM analysis.

    Always call this first. The result includes a ``strategy_hint`` field that
    tells the agent which analysis tool to use based on the file count.
    """

    name: str = "discover_documents"
    description: str = (
        "Scan a folder and return the list of readable document paths "
        "(pdf, docx, xlsx, md, etc.) without performing LLM analysis. "
        f"If document_count > {LARGE_FOLDER_THRESHOLD} in the result, "
        "use selective_folder_analysis instead of analyze_folder."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "folder_path": {
                "type": "string",
                "description": "Path to the folder to scan.",
            },
            "max_files": {
                "type": "integer",
                "description": "Cap on the number of files returned.",
            },
        },
        "required": ["folder_path"],
    }

    async def run(
        self,
        folder_path: str,
        max_files: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        from src.application.use_cases.discover_folder_documents import (
            DiscoverFolderDocumentsRequest,
            DiscoverFolderDocumentsUseCase,
        )
        from src.adapters.outbound.parsers.file_parser_document_reader import (
            FileParserDocumentReader,
        )

        logger.info("DiscoverDocumentsTool.run: folder=%s", folder_path)
        try:
            use_case = DiscoverFolderDocumentsUseCase(FileParserDocumentReader())
            docs = use_case.execute(
                DiscoverFolderDocumentsRequest(
                    folder_path=folder_path,
                    max_files=max_files,
                )
            )
            doc_count = len(docs)
            content = {
                "folder_path": folder_path,
                "document_count": doc_count,
                "documents": docs,
                "strategy_hint": (
                    "selective_folder_analysis"
                    if doc_count > LARGE_FOLDER_THRESHOLD
                    else "analyze_folder"
                ),
            }
            return ToolResult(
                success=True,
                content=content,
                tool_name=self.name,
                metadata={
                    "document_count": doc_count,
                    "large_folder": doc_count > LARGE_FOLDER_THRESHOLD,
                },
            )
        except Exception as exc:
            logger.exception("DiscoverDocumentsTool failed: %s", exc)
            return ToolResult(
                success=False,
                content=None,
                tool_name=self.name,
                error=str(exc),
            )


# ──────────────────────────────────────────────────────────────────────────────
# Tool: extract_fragments  (U4.3)
# ──────────────────────────────────────────────────────────────────────────────


class ExtractFragmentsTool:
    """Extract text fragments from a single document for deep inspection.

    Use to read raw content of a specific file before deciding whether to
    include it in a selective analysis, or to identify which documents are
    most ontology-relevant.

    Returns a list of ``{"source": ..., "text": ...}`` objects (one per
    paragraph / slide / page depending on the document type).
    Truncates at ``max_fragments`` to guard token budget.
    """

    name: str = "extract_fragments"
    description: str = (
        "Extract raw text fragments (paragraphs, slides, pages) from a single document. "
        "Use this before selective_folder_analysis to identify which files contain "
        "the most ontology-relevant content, or for deep inspection of a key document."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the document file to extract.",
            },
            "max_fragments": {
                "type": "integer",
                "description": "Maximum number of fragments to return (default 50).",
                "default": 50,
            },
        },
        "required": ["file_path"],
    }

    async def run(
        self,
        file_path: str,
        max_fragments: int = 50,
        **kwargs: Any,
    ) -> ToolResult:
        """Extract and return text fragments from one document."""
        from src.application.use_cases.extract_document_fragments import (
            ExtractDocumentFragmentsRequest,
            ExtractDocumentFragmentsUseCase,
        )
        from src.adapters.outbound.parsers.file_parser_document_reader import (
            FileParserDocumentReader,
        )

        logger.info(
            "ExtractFragmentsTool.run: file=%s max_fragments=%d",
            file_path, max_fragments,
        )
        try:
            use_case = ExtractDocumentFragmentsUseCase(FileParserDocumentReader())
            fragments = use_case.execute(
                ExtractDocumentFragmentsRequest(file_path=file_path)
            )
            truncated = fragments[:max_fragments]
            serialised = [{"source": f.source, "text": f.text} for f in truncated]
            content = {
                "file_path": file_path,
                "total_fragments": len(fragments),
                "returned_fragments": len(truncated),
                "fragments": serialised,
            }
            return ToolResult(
                success=True,
                content=content,
                tool_name=self.name,
                metadata={
                    "total_fragments": len(fragments),
                    "truncated": len(fragments) > max_fragments,
                },
            )
        except Exception as exc:
            logger.exception("ExtractFragmentsTool failed: %s", exc)
            return ToolResult(
                success=False,
                content=None,
                tool_name=self.name,
                error=str(exc),
            )


# ──────────────────────────────────────────────────────────────────────────────
# Default tool registry
# ──────────────────────────────────────────────────────────────────────────────


def default_ontology_tools(
    run_id_prefix: str = "agent",
    capsule_storage_dir: str | None = None,
    capsule_query_port: CapsuleQueryPort | None = None,
) -> list:
    """Return the default tool registry for the Ontology Agent (U4.4).

    Ordered to guide system-prompt readability:
    1. query_capsule             — U4.4: check historical experience first
    2. discover_documents        — drives strategy based on file count
    3. analyze_folder            — full analysis for small folders (≤ threshold)
    4. selective_folder_analysis — partial analysis for large folders (> threshold)
    5. extract_fragments         — deep single-document inspection
    """

    effective_storage_dir = capsule_storage_dir or "outputs/capsules"

    return [
        QueryCapsuleTool(storage_dir=effective_storage_dir, capsule_query_port=capsule_query_port),
        DiscoverDocumentsTool(),
        FolderAnalysisTool(run_id_prefix=run_id_prefix),
        SelectiveFolderAnalysisTool(run_id_prefix=run_id_prefix),
        ExtractFragmentsTool(),
    ]
