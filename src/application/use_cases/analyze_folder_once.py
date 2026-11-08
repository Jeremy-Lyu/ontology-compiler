"""Application use-case for folder-level ontology analysis."""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

import json
import logging

from src.application.ports.document_ports import DocumentReaderPort
from src.application.ports.folder_analysis_ports import FolderAnalysisServicePort
from src.application.ports.observability_ports import HookResult, HookRunnerPort
from src.adapters.folder_analysis_service_adapter import (
    FolderAnalysisServiceAdapter,
    legacy_analyze_folder_once,
    legacy_save_capsule_experience,
)

from src.domain.errors import HookDeniedError
from core.utils.file_helpers import detect_source_profile_from_paths, slug_text
from src.application.use_cases.discover_folder_documents import (
    DiscoverFolderDocumentsRequest,
    DiscoverFolderDocumentsUseCase,
)
from src.application.use_cases.folder_analysis_orchestration import (
    build_analysis_application_finalization_bundle,
    build_analysis_application_orchestration_bundle,
    build_analysis_exec_completion_event,
    build_analysis_exec_error_event,
    build_analysis_exec_progress_event,
    build_analysis_post_hook_payload_inputs,
    build_analysis_pre_hook_payload_inputs,
)
from src.application.use_cases.folder_analysis_core import (
    build_folder_analysis_run_context,
    folder_fingerprint,
)
from src.application.services.incremental_analysis_service import (
    IncrementalAnalysisService,
    IncrementalAnalysisResult,
)
from src.application.services.patch_integration_service import (
    PatchIntegrationService,
)
from src.domain.models.folder_analysis_result import FolderAnalysisResult

# Module-level aliases — exposed so tests can monkeypatch them.
save_capsule_experience = legacy_save_capsule_experience

logger = logging.getLogger(__name__)

ExecEventCallback = Callable[[dict[str, Any]], None]


class _AllowAllHookRunner(HookRunnerPort):
    """Default hook runner that allows all events."""

    def run(self, event: str, payload: dict[str, Any]) -> HookResult:
        return HookResult(allowed=True, exit_code=0, reason="hooks disabled", output="")


class HookRunner(_AllowAllHookRunner):
    """Compatibility hook runner symbol for tests and injection."""


def _build_hook_payload(
    *,
    stage: str,
    tool: str,
    event: str,
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
    is_error: bool = False,
) -> dict[str, Any]:
    return {
        "ts": datetime.now().isoformat(),
        "stage": stage,
        "tool": tool,
        "event": event,
        "input": input_data or {},
        "output": output_data or {},
        "is_error": is_error,
    }

def _build_default_document_reader() -> DocumentReaderPort:
    reader_cls = import_module(
        "src.adapters.outbound.parsers.file_parser_document_reader"
    ).FileParserDocumentReader
    return reader_cls()

def _emit_exec_event(
    *,
    exec_event_callback: ExecEventCallback | None,
    event: dict[str, Any],
) -> None:
    if exec_event_callback is not None:
        exec_event_callback(event)

@dataclass(frozen=True)
class FolderAnalysisRequest:
    """Request DTO for folder analysis."""
    folder_path: str
    run_id: str | None = None
    publish_shared: bool = True
    generate_report: bool = True
    minimal_output: bool = False
    with_capsules: bool = False
    domain: str = "SUPPLY_CHAIN"
    input_type: str = "auto"
    misc_batch: bool = False
    fast_mode: bool = False
    smoke_mode: bool = False
    max_files: int | None = None
    include_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None
    progress_callback: Callable[[int, int, str], None] | None = None
    document_reader: DocumentReaderPort | None = None
    exec_event_callback: ExecEventCallback | None = None
    hook_runner: HookRunnerPort | None = None
    incremental_service: IncrementalAnalysisService | None = None
    skip_if_unchanged: bool = True  # B-1: Enable incremental skip by default
    force_reanalyze: bool = False
    
    # B-2: Patch integration options
    patch_service: PatchIntegrationService | None = None
    publish_as_patch: bool = False  # If True, publish results as atomic patch
    patch_dry_run: bool = True      # If True, validate patch without applying
    patch_baseline_l1: Path | None = None  # Optional baseline for patch comparison
    patch_baseline_l2: Path | None = None  # Optional baseline for patch comparison

async def _analyze_folder_once(
    request: FolderAnalysisRequest,
    *,
    folder_analysis_service: FolderAnalysisServicePort | None = None,
) -> FolderAnalysisResult:
    """Application entrypoint for folder analysis.

    This keeps the legacy compiler/service behavior intact while moving more
    deterministic orchestration into the application layer.
    """

    folder_path = request.folder_path
    run_id = request.run_id
    fast_mode = request.fast_mode
    smoke_mode = request.smoke_mode
    max_files = request.max_files
    input_type = request.input_type
    domain = request.domain
    publish_shared = request.publish_shared
    generate_report = request.generate_report
    minimal_output = request.minimal_output
    with_capsules = request.with_capsules
    misc_batch = request.misc_batch
    include_patterns = request.include_patterns
    exclude_patterns = request.exclude_patterns
    progress_callback = request.progress_callback
    document_reader = request.document_reader
    exec_event_callback = request.exec_event_callback
    hook_runner = request.hook_runner
    incremental_service = request.incremental_service or IncrementalAnalysisService()
    skip_if_unchanged = request.skip_if_unchanged
    force_reanalyze = request.force_reanalyze

    # B-1: Check incremental analysis
    if skip_if_unchanged:
        incremental_check: IncrementalAnalysisResult = incremental_service.check_should_skip(
            folder_path,
            force_reanalyze=force_reanalyze,
        )
        
        if incremental_check.should_skip and incremental_check.manifest_path:
            logger.info(f"Incremental skip: {incremental_check.reason}")
            logger.info(f"Using cached manifest: {incremental_check.manifest_path}")
            
            # Load cached result from manifest
            with open(incremental_check.manifest_path, "r", encoding="utf-8") as f:
                cached_manifest = json.load(f)
            
            # Build a result from the cached manifest
            source_root = Path(folder_path)
            resolved_run_id = run_id or f"TBH-{slug_text(source_root.name) or 'tbh'}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            return FolderAnalysisResult(
                status="skipped",
                run_id=cached_manifest.get("run_id", resolved_run_id),
                folder=folder_path,
                fast_mode=fast_mode,
                smoke_mode=smoke_mode,
                compile={},
                report={},
                manifest=str(incremental_check.manifest_path),
                llm_usage=cached_manifest.get("llm_usage", {}),
                summary=cached_manifest.get("summary", {}),
                quality=cached_manifest.get("quality", {}),
                artifacts=cached_manifest.get("artifacts", {}),
                run_dir=cached_manifest.get("artifacts", {}).get("run_dir", ""),
                incremental_info={
                    "status": incremental_check.status,
                    "reason": incremental_check.reason,
                    "previous_fingerprint": incremental_check.previous_fingerprint,
                    "current_fingerprint": incremental_check.current_fingerprint,
                },
            )
        elif incremental_check.status in ("changed", "stale", "new", "forced"):
            logger.info(f"Incremental analysis: {incremental_check.reason}")
    
    run_context = build_folder_analysis_run_context(
        folder_path=folder_path,
        run_id=run_id,
        fast_mode=fast_mode,
        smoke_mode=smoke_mode,
        max_files=max_files,
    )
    effective_max_files = run_context["effective_max_files"]

    discover_documents = DiscoverFolderDocumentsUseCase(
        document_reader or _build_default_document_reader()
    )
    readable_documents = discover_documents.execute(
        DiscoverFolderDocumentsRequest(
            folder_path=folder_path,
            max_files=effective_max_files,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
    )
    if not readable_documents:
        raise ValueError("no readable documents found in folder")

    source_profile = detect_source_profile_from_paths(Path(path) for path in readable_documents)
    if input_type != "auto":
        source_profile["primary_type"] = input_type

    orchestration_bundle = build_analysis_application_orchestration_bundle(
        folder_path=folder_path,
        run_id=run_id,
        fast_mode=fast_mode,
        smoke_mode=smoke_mode,
        max_files=max_files,
        misc_batch=misc_batch,
        domain=domain,
        input_type=input_type,
        publish_shared=publish_shared,
        generate_report=generate_report,
        minimal_output=minimal_output,
        with_capsules=with_capsules,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        progress_callback=progress_callback,
        source_profile=source_profile,
        discovered_documents=readable_documents,
        started_at=datetime.now().isoformat(),
        source_fingerprint=folder_fingerprint(Path(folder_path)),
    )
    analysis_context = orchestration_bundle["analysis_context"]

    effective_hook_runner = hook_runner or HookRunner()

    pre_hook_inputs = build_analysis_pre_hook_payload_inputs(
        folder_path=analysis_context.folder_path,
        run_id=analysis_context.run_id,
        domain=domain,
        fast_mode=analysis_context.fast_mode,
        smoke_mode=analysis_context.smoke_mode,
        max_files=analysis_context.effective_max_files,
    )
    pre_hook = effective_hook_runner.run(
        "PreToolUse",
        _build_hook_payload(**pre_hook_inputs),
    )
    if not pre_hook.allowed:
        raise HookDeniedError(pre_hook.output or f"hook denied analyze_folder_once: {folder_path}")

    _emit_exec_event(
        exec_event_callback=exec_event_callback,
        event=build_analysis_exec_progress_event(
            analysis_context=analysis_context,
            stage="discover_documents",
            message="discovered readable documents",
            current=analysis_context.discovered_document_count,
            total=analysis_context.effective_max_files,
        ),
    )
    _emit_exec_event(
        exec_event_callback=exec_event_callback,
        event=build_analysis_exec_progress_event(
            analysis_context=analysis_context,
            stage="detect_source_profile",
            message="detected source profile",
        ),
    )
    _emit_exec_event(
        exec_event_callback=exec_event_callback,
        event=build_analysis_exec_progress_event(
            analysis_context=analysis_context,
            stage="legacy_analysis",
            message="delegating to legacy analysis service",
        ),
    )

    try:
        # Always go through the port/adapter layer; default adapter wraps the legacy function.
        # Pass the module-level legacy_analyze_folder_once so tests can still monkeypatch it.
        service = folder_analysis_service or FolderAnalysisServiceAdapter(legacy_analyze_func=legacy_analyze_folder_once)
        legacy_request = dict(orchestration_bundle["legacy_request"])
        from src.application.ports.folder_analysis_ports import FolderAnalysisRequest as PortRequest
        port_request = PortRequest(
            folder_path=folder_path,
            domain=domain,
            run_id=run_id or "auto",
            topic_id=analysis_context.topic_id,
            profile=orchestration_bundle.get("profile", "local"),
            source_profile=orchestration_bundle.get("source_profile"),
            max_docs=max_files,
            include_archived=False,
            with_capsules=with_capsules,
            legacy_params=legacy_request,
        )
        port_result = await service.analyze_folder_once(port_request, return_finalization_bundle=True)
        if not port_result.success:
            raise RuntimeError(f"Folder analysis failed: {port_result.error_message}")
        legacy_finalization_bundle = port_result.finalization_bundle
        application_finalization_bundle = build_analysis_application_finalization_bundle(
            finalization_bundle=legacy_finalization_bundle,
            analysis_context=analysis_context,
        )
        application_result_context = application_finalization_bundle["application_result_context"]
        analysis_response_legacy = application_finalization_bundle["response"]
        analysis_response = FolderAnalysisResult.from_legacy_dict(analysis_response_legacy)

        if with_capsules:
            try:
                write_stage_context = legacy_finalization_bundle["write_result_bundle"]["write_stage_bundle"]["write_stage_context"]
                await save_capsule_experience(
                    topic_id=analysis_context.topic_id,
                    domain=write_stage_context["domain"],
                    run_id=analysis_context.run_id,
                    source_folder=folder_path,
                    l1=write_stage_context.get("l1") or {},
                    l2=write_stage_context.get("l2") or {},
                    quality=write_stage_context.get("quality") or {},
                    status=legacy_finalization_bundle.get("final_status", "UNKNOWN"),
                )
            except Exception as _cap_exc:
                logger.warning(f"Capsule experience save failed (non-fatal): {_cap_exc}")

        post_hook = effective_hook_runner.run(
            "PostToolUse",
            _build_hook_payload(
                **build_analysis_post_hook_payload_inputs(
                    folder_path=folder_path,
                    run_id=analysis_context.run_id,
                    final_result=legacy_finalization_bundle.get("final_result") or {},
                )
            ),
        )
        if not post_hook.allowed:
            raise HookDeniedError(post_hook.output or f"hook denied analyze_folder_once post step: {folder_path}")
        _emit_exec_event(
            exec_event_callback=exec_event_callback,
            event=build_analysis_exec_progress_event(
                analysis_context=analysis_context,
                stage="finalize_response",
                message="finalized application response",
            ),
        )
        _emit_exec_event(
            exec_event_callback=exec_event_callback,
            event=build_analysis_exec_completion_event(
                application_result_context=application_result_context,
            ),
        )
        # B-1: Record successful analysis in cache for future incremental runs
        if hasattr(analysis_response, 'manifest') and analysis_response.manifest:
            try:
                manifest_path = Path(analysis_response.manifest)
                if manifest_path.exists():
                    incremental_service.record_analysis_complete(
                        folder_path=folder_path,
                        run_id=analysis_response.run_id,
                        manifest_path=manifest_path,
                        metadata={
                            "status": analysis_response.status,
                            "has_summary": bool(analysis_response.summary),
                            "discovered_docs": len(readable_documents),
                        }
                    )
            except Exception as cache_exc:
                logger.warning(f"Failed to record analysis cache: {cache_exc}")
        
        # B-2: Publish compilation results as atomic patch (if enabled)
        if request.publish_as_patch and hasattr(analysis_response, 'run_dir') and analysis_response.run_dir:
            try:
                patch_service = request.patch_service or PatchIntegrationService()
                run_dir_path = Path(analysis_response.run_dir)
                
                if run_dir_path.exists():
                    patch_result = patch_service.publish_compilation_as_patch(
                        run_dir=run_dir_path,
                        baseline_l1_path=request.patch_baseline_l1,
                        baseline_l2_path=request.patch_baseline_l2,
                        reason=f"Compilation result for {analysis_response.run_id}",
                        requested_by="compilation-pipeline",
                        dry_run=request.patch_dry_run,
                    )
                    
                    # Attach patch result to response
                    analysis_response.patch_result = {
                        "status": patch_result.status,
                        "patch_id": patch_result.patch_id,
                        "operations_applied": patch_result.operations_applied,
                        "dry_run": patch_result.dry_run,
                        "rollback_available": patch_result.rollback_available,
                    }
                    
                    logger.info(
                        f"Patch publication {patch_result.status}: {patch_result.operations_applied} ops "
                        f"(dry_run={patch_result.dry_run})"
                    )
            except Exception as patch_exc:
                logger.warning(f"Patch publication failed: {patch_exc}")
                analysis_response.patch_result = {"status": "FAILED", "error": str(patch_exc)}
        
        return analysis_response
    except Exception as exc:
        _emit_exec_event(
            exec_event_callback=exec_event_callback,
            event=build_analysis_exec_error_event(
                analysis_context=analysis_context,
                error=exc,
                stage="legacy_analysis",
            ),
        )
        raise


async def analyze_folder_once(
    request: FolderAnalysisRequest | None = None,
    *,
    folder_path: str | None = None,
    run_id: str | None = None,
    publish_shared: bool = True,
    generate_report: bool = True,
    minimal_output: bool = False,
    with_capsules: bool = False,
    domain: str = "SUPPLY_CHAIN",
    input_type: str = "auto",
    misc_batch: bool = False,
    fast_mode: bool = False,
    smoke_mode: bool = False,
    max_files: int | None = None,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    document_reader: DocumentReaderPort | None = None,
    exec_event_callback: ExecEventCallback | None = None,
    hook_runner: HookRunnerPort | None = None,
    folder_analysis_service: FolderAnalysisServicePort | None = None,
    incremental_service: IncrementalAnalysisService | None = None,
    skip_if_unchanged: bool = True,  # B-1: Enable incremental skip by default
    force_reanalyze: bool = False,
    # B-2: Patch integration options
    patch_service: PatchIntegrationService | None = None,
    publish_as_patch: bool = False,
    patch_dry_run: bool = True,
    patch_baseline_l1: Path | None = None,
    patch_baseline_l2: Path | None = None,
) -> FolderAnalysisResult:
    """Application entrypoint for folder analysis.

    Supports both:
    - Passing a request DTO: ``analyze_folder_once(FolderAnalysisRequest(...))``
    - Passing keyword args: ``analyze_folder_once(folder_path=..., ...)``
    """

    if request is None:
        if folder_path is None:
            raise TypeError("folder_path is required")
        request = FolderAnalysisRequest(
            folder_path=folder_path,
            run_id=run_id,
            publish_shared=publish_shared,
            generate_report=generate_report,
            minimal_output=minimal_output,
            with_capsules=with_capsules,
            domain=domain,
            input_type=input_type,
            misc_batch=misc_batch,
            fast_mode=fast_mode,
            smoke_mode=smoke_mode,
            max_files=max_files,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            progress_callback=progress_callback,
            document_reader=document_reader,
            exec_event_callback=exec_event_callback,
            hook_runner=hook_runner,
            incremental_service=incremental_service,
            skip_if_unchanged=skip_if_unchanged,
            force_reanalyze=force_reanalyze,
            # B-2: Patch integration
            patch_service=patch_service,
            publish_as_patch=publish_as_patch,
            patch_dry_run=patch_dry_run,
            patch_baseline_l1=patch_baseline_l1,
            patch_baseline_l2=patch_baseline_l2,
        )
    return await _analyze_folder_once(request, folder_analysis_service=folder_analysis_service)
