"""Stage-execution core extracted from MainPipeline (LOC ratchet §3.1).

Phase 2 of the main_pipeline LOC reduction: the C-4 stage-decision loop and
the pipeline-output collector moved verbatim from MainPipeline methods using
the established ``pipeline``-first-parameter pattern (same as
``asset_packet.generate_asset_packet``). Behavior stays locked by
tests/pipeline/test_main_pipeline_characterization.py, which goes through the
delegating methods on MainPipeline.

Contents:
- :func:`run_pipeline_stages` — the orchestration loop over StageDecisions
- :func:`resolve_actual_stage_decision` — authoritative C-4 decision
- :func:`compute_stage_decision_trace` — manifest-trace wrapper
- :func:`collect_pipeline_output` — quality-gate input collection
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover - type hints only, avoids circular import
    from .main_pipeline import MainPipeline

logger = logging.getLogger(__name__)


async def run_pipeline_stages(pipeline: "MainPipeline", stage_enums) -> None:
    """Run *stage_enums* through per-stage StageDecision gating.

    Moved verbatim from ``MainPipeline._run_pipeline_stages``.
    """
    from .stage_decision import StageDecision

    trace: List[Dict[str, Any]] = []
    # Tracks whether the *previous* stage actually executed (as opposed to
    # being skipped/reused). Used as a conservative upstream-change signal
    # for the next stage's decision input (prevents reusing artifacts when
    # the pipeline just re-ran an upstream stage).
    last_stage_executed: bool = True

    for stage in stage_enums:
        decision_result = resolve_actual_stage_decision(
            pipeline, stage, upstream_changed=last_stage_executed
        )
        trace.append(decision_result.to_dict())
        decision = decision_result.decision

        stage_result: "StageResult"
        current_stage_executed: bool

        if decision == StageDecision.RUN:
            # Legacy _should_skip_stage is preserved as a compatibility
            # safety net (it handled offline-mode EMERGENT_REASONING/GUARD
            # skipping before StageDecision existed). When it fires we
            # synthesise a skipped result so downstream still sees shape.
            if pipeline._should_skip_stage(stage):
                stage_result = pipeline._make_skipped_stage_result(
                    stage,
                    reason=f"legacy _should_skip_stage in {pipeline._execution_mode.value} mode",
                    decision=StageDecision.SKIP_UNCHANGED,
                )
                current_stage_executed = False
            else:
                # PipelineStage lives in main_pipeline (circular-import safe:
                # compare by enum value instead of importing the class).
                if getattr(stage, "value", None) == "extract":
                    pipeline._filter_unprocessed_files()
                stage_result = await pipeline._execute_pipeline_stage(stage)
                current_stage_executed = True
        elif decision == StageDecision.SKIP_UNCHANGED:
            stage_result = pipeline._make_skipped_stage_result(
                stage,
                reason=decision_result.reason,
                decision=StageDecision.SKIP_UNCHANGED,
            )
            current_stage_executed = False
        elif decision == StageDecision.REUSE_ARTIFACT:
            # C-4: replay the cached StageResult so downstream stages see
            # the exact same shape they would after a real execution.
            #
            # Two sources are considered, in priority order:
            #   1. ``_stage_results[stage.value]`` — same-run idempotent
            #      re-entry. Tag in-place with reuse markers so the trace
            #      is explicit.
            #   2. Cross-run artifact bundle loaded from sibling run dirs
            #      (``_find_prev_run_bundle`` → ``_load_stage_result_from_bundle``).
            #      This is the "replay result on unchanged input" path.
            #
            # If neither source is available we fall back to a skipped
            # result with a clear warning. This is a defensive safety net
            # because the decision function should not have returned
            # REUSE_ARTIFACT when no cached signals are present.
            existing = pipeline._stage_results.get(stage.value)
            if existing is not None and getattr(existing, "success", False):
                logger.info(
                    f"♻️  StageDecision.REUSE_ARTIFACT (same run): {stage.value} "
                    f"— {decision_result.reason}"
                )
                existing_data = dict(existing.data or {})
                existing_data["reused_artifact"] = True
                existing_data["stage_decision"] = StageDecision.REUSE_ARTIFACT.value
                existing_data["reuse_reason"] = decision_result.reason
                stage_result = existing
                stage_result.data = existing_data
                stage_result.message = (
                    f"[REUSE_ARTIFACT] {stage_result.message}"
                    if stage_result.message
                    else f"[REUSE_ARTIFACT] {decision_result.reason}"
                )
            else:
                prev_bundle = find_prev_run_bundle(pipeline, stage)
                if prev_bundle is not None:
                    logger.info(
                        f"♻️  StageDecision.REUSE_ARTIFACT (cross-run): "
                        f"{stage.value} replaying run_id="
                        f"{prev_bundle.get('run_id')} — {decision_result.reason}"
                    )
                    stage_result = load_stage_result_from_bundle(
                        pipeline, stage, prev_bundle, reuse_reason=decision_result.reason
                    )
                    # Tag warnings so operators see reuse in summary logs.
                    pipeline._warnings.append(
                        f"Reused stage {stage.value} artifact from "
                        f"previous run ({prev_bundle.get('run_id')}): "
                        f"{decision_result.reason}"
                    )
                else:
                    logger.warning(
                        f"[StageDecision] REUSE_ARTIFACT for {stage.value}: "
                        "decision function returned reuse but no cached "
                        "or cross-run bundle could be loaded. Safely "
                        "falling back to SKIP_UNCHANGED."
                    )
                    stage_result = pipeline._make_skipped_stage_result(
                        stage,
                        reason=(
                            f"REUSE_ARTIFACT decision issued but bundle "
                            f"unavailable at runtime. Decision reason: "
                            f"{decision_result.reason}"
                        ),
                        decision=StageDecision.REUSE_ARTIFACT,
                    )
            current_stage_executed = False
        elif decision == StageDecision.DEGRADE:
            stage_result = pipeline._make_degraded_stage_result(
                stage, reason=decision_result.reason
            )
            current_stage_executed = False
        elif decision == StageDecision.BLOCK:
            # Hard precondition failure: synthesise a failing result and
            # let _process_stage_result decide if the pipeline should be
            # halted (PREFLIGHT / GUARD block → halt).
            logger.error(
                f"🚫 StageDecision.BLOCK for {stage.value}: {decision_result.reason}"
            )
            from .stages.common import StageResult as _SR

            stage_result = _SR(
                stage=stage.value,
                success=False,
                message=f"[BLOCK] {decision_result.reason}",
                data={
                    "stage_decision": StageDecision.BLOCK.value,
                    "block_reason": decision_result.reason,
                    "errors": [f"stage {stage.value} blocked: {decision_result.reason}"],
                },
                elapsed_seconds=0.0,
            )
            current_stage_executed = False
        else:  # pragma: no cover - defensive fallback for unknown enum
            logger.warning(
                f"[StageDecision] Unknown decision {decision!r} for {stage.value}; "
                "falling back to RUN"
            )
            stage_result = await pipeline._execute_pipeline_stage(stage)
            current_stage_executed = True

        pipeline._stage_results[stage.value] = pipeline._sanitize_stage_result_paths(stage_result)
        last_stage_executed = current_stage_executed
        if await pipeline._process_stage_result(stage, pipeline._stage_results[stage.value]):
            break

    if trace:
        if not isinstance(pipeline.context.metadata, dict):
            pipeline.context.metadata = {}
        pipeline.context.metadata["stage_decision_trace"] = trace


def resolve_actual_stage_decision(
    pipeline: "MainPipeline",
    stage,
    *,
    upstream_changed: bool = True,
) -> "StageDecisionResult":
    """Compute the authoritative C-4 StageDecision for *stage*.

    Resolution order (from most-reliable signal to conservative fallback):
      1. **Within-run cache** (``_stage_results``): handles idempotent
         re-entry inside the same invocation. The cached result carries
         ``source_hash`` / ``version_bundle`` inside its ``.data`` dict.
      2. **Cross-run artifact registry** (``outputs/.../runs/*/artifacts``):
         scans sibling runs for the *same stage*, filtering bundles whose
         ``metadata.source_hash`` matches the current run and whose
         previous execution was a success. This is the "reuse previous
         result on unchanged input" path.
      3. **RUN** fallback when neither cache is available or any
         version/signature dimension mismatches.

    Conservatism rules (C-4: never reuse stale results, RUN on ambiguity):
      - Cross-run lookup requires an exact ``source_hash`` match; any
        partial signal is discarded.
      - ``cached_version_bundle`` populated only when the bundle stored a
        non-empty version map; otherwise we force RUN so a prompt/schema
        change won't silently replay old output.
      - Up to 50 sibling artifact dirs are scanned; if the output tree is
        larger than that we truncate and fall back to RUN (rather than
        OOM-thrashing the directory walk).

    Moved verbatim from ``MainPipeline._resolve_actual_stage_decision``.
    """
    from .stage_decision import (
        CacheVersionBundle,
        StageDecisionInput,
        StageDecisionResult,
        compute_stage_decision,
    )
    from .versioned_cache import resolve_version_bundle_from_context

    vb = resolve_version_bundle_from_context(pipeline.context)

    existing = pipeline._stage_results.get(stage.value)
    existing_data: Dict[str, Any] = getattr(existing, "data", None) or {}

    cached_success: bool
    has_cached_artifact: bool
    cached_source_hash: str
    cached_vb: CacheVersionBundle

    if existing is not None and getattr(existing, "success", False):
        # Case 1: same-run idempotent re-entry → use memory cache.
        cached_success = True
        has_cached_artifact = True
        cached_source_hash = str(
            existing_data.get("source_hash") or existing_data.get("run_id") or ""
        )
        cached_vb = CacheVersionBundle.from_dict(
            existing_data.get("version_bundle")
        )
    else:
        # Case 2: cross-run lookup → scan sibling run artifacts.
        prev_bundle = find_prev_run_bundle(pipeline, stage)
        if prev_bundle is None:
            cached_success = False
            has_cached_artifact = False
            cached_source_hash = ""
            cached_vb = CacheVersionBundle()
        else:
            md = prev_bundle.get("metadata") or {}
            cached_success = bool(md.get("success", False))
            has_cached_artifact = cached_success
            cached_source_hash = str(md.get("source_hash") or "")
            cached_vb = CacheVersionBundle.from_dict(md.get("version_bundle"))

    retry_failed_chunks: bool = True

    inp = StageDecisionInput(
        stage=stage.value,
        execution_mode=pipeline._execution_mode.value,
        source_hash=pipeline._run_id,
        version_bundle=vb,
        cached_success=cached_success,
        cached_source_hash=cached_source_hash,
        cached_version_bundle=None if cached_vb.is_empty() else cached_vb,
        has_cached_artifact=has_cached_artifact,
        upstream_changed=upstream_changed,
        retry_failed_chunks=retry_failed_chunks,
    )
    return compute_stage_decision(inp)


def compute_stage_decision_trace(pipeline: "MainPipeline", stage) -> Optional[Dict[str, Any]]:
    """Record the C-4 StageDecision into the manifest trace.

    Delegates to :func:`resolve_actual_stage_decision` so the *recorded*
    reason is 100 % identical to the *acted-upon* reason. Maintained as a
    thin wrapper because the original calling convention (``Optional[Dict]``
    with graceful ``None`` on error) is relied on by the manifest writer.
    """
    try:
        return resolve_actual_stage_decision(pipeline, stage).to_dict()
    except Exception as exc:  # pragma: no cover - observability must not break the run
        logger.debug(f"[StageDecision] trace unavailable for {stage.value}: {exc}")
        return None


def collect_pipeline_output(pipeline: "MainPipeline") -> Optional[Dict[str, Any]]:
    """收集 pipeline 输出数据用于验证。

    Prefers the persisted ``run_manifest.json`` (written by ``save_result``).
    When it is not available yet (quality gate runs inside ``run()`` before
    ``save_result``), falls back to an in-memory view assembled from the
    current stage results so the quality gate can still run on real data.

    Moved verbatim from ``MainPipeline._collect_pipeline_output``.
    """
    try:
        output_dir = Path(pipeline.context.output_dir)
        manifest_file = output_dir / "run_manifest.json"

        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.debug(f"Failed to collect pipeline output: {e}")

    # Fallback: assemble from live stage results (extract/merge carry the
    # node/relation payloads used by the quality gate).
    try:
        nodes: list[Any] = []
        instances: list[Any] = []
        relationships: list[Any] = []
        metadata: Dict[str, Any] = {}

        # Prefer the authoritative master written by the merge stage. It is
        # the single ID-consistent source (fragment IDs are remapped here),
        # so mixing it with per-stage fragments produces dangling-endpoint
        # false positives in the quality gate. Fall back to stage mixing
        # only when no merge master is available (e.g. extract-only tests).
        master_path = None
        merge_result = pipeline._stage_results.get("merge")
        if merge_result and isinstance(merge_result.data, dict):
            master_path = merge_result.data.get("master_path")
        if master_path:
            try:
                with open(master_path, "r", encoding="utf-8") as _mf:
                    _master = json.load(_mf)
                nodes = _master.get("nodes") or []
                instances = _master.get("instances") or []
                relationships = _master.get("relationships") or []
                _meta = _master.get("metadata")
                if isinstance(_meta, dict):
                    metadata.update(_meta)
            except Exception as _exc:
                logger.debug(f"Master read failed, falling back to stage mixing: {_exc}")
                nodes, instances, relationships = [], [], []

        if not nodes and not relationships:
            # No authoritative master — mix per-stage payloads (legacy path,
            # kept for extract-only/test scenarios).
            for stage_result in pipeline._stage_results.values():
                stage_data = stage_result.data or {}
                for key, value in stage_data.items():
                    if isinstance(value, list) and value:
                        if all(isinstance(item, dict) and (item.get("meta_id") or item.get("id")) for item in value[:3]):
                            nodes.extend(value)
                    if isinstance(value, list) and key in ("relationships", "relations"):
                        relationships.extend(value)
                    elif isinstance(value, list) and key == "instances":
                        instances.extend(value)
                meta = stage_data.get("metadata")
                if isinstance(meta, dict):
                    metadata.update(meta)
        if isinstance(pipeline.context.metadata, dict):
            for key in ("classification_metrics", "quality_gate"):
                if key in pipeline.context.metadata:
                    metadata.setdefault(key, pipeline.context.metadata[key])

        # B1 (2026-08-15): schema alias normalization + view assembly are
        # in the pure pipeline_output_mapper; this function keeps only the
        # I/O selection (master vs stage mixing + context metadata).
        from src.application.transformers.pipeline_output_mapper import (
            assemble_pipeline_output_view,
        )

        return assemble_pipeline_output_view(
            source_path=pipeline.context.source_path,
            nodes=nodes,
            instances=instances,
            relationships=relationships,
            metadata=metadata,
            stats={"entity_count": pipeline._stats.entity_count, "relation_count": pipeline._stats.relation_count},
        )
    except Exception as e:
        logger.debug(f"Failed to assemble pipeline output fallback: {e}")
        return None


# ----------------------------------------------------------------------
# Cross-run C-4 helpers (P0-3 batch 5, 2026-09-09)
#
# Moved verbatim from MainPipeline._collect_sibling_artifact_roots /
# _find_prev_run_bundle / _load_stage_result_from_bundle. Their only
# consumers were resolve_actual_stage_decision / run_pipeline_stages in
# this module, so the call sites now use the module functions directly
# (no MainPipeline delegates remain; no tests referenced them).
# ----------------------------------------------------------------------
def collect_sibling_artifact_roots(pipeline: "MainPipeline", limit: int = 50) -> list[Path]:
    """Return artifact directories from sibling runs under ``output_dir/runs``.

    The current run's own artifact directory is excluded so we never
    self-reference an in-progress bundle.
    """
    try:
        artifact_root = Path(pipeline._run_paths.artifact_root)
        # build_pipeline_run_paths yields: output_dir/runs/<run_id>/artifacts
        runs_root = artifact_root.parent.parent  # → output_dir/runs
        if not runs_root.is_dir():
            return []
        siblings: list[Path] = []
        for run_dir in runs_root.iterdir():
            if not run_dir.is_dir():
                continue
            if run_dir.name == pipeline._run_id:  # exclude self
                continue
            sib = run_dir / "artifacts"
            if sib.is_dir():
                siblings.append(sib)
                if len(siblings) >= limit:
                    break
        return siblings
    except OSError as exc:
        logger.debug(f"[ArtifactRegistry] sibling scan failed: {exc}")
        return []


def find_prev_run_bundle(pipeline: "MainPipeline", stage) -> Optional[Dict[str, Any]]:
    """Scan sibling runs for the newest successful bundle matching our source hash.

    Returns the raw bundle dict or None. Only bundles with:
      - ``metadata.success == True``
      - ``metadata.source_hash == pipeline._run_id`` (same source document)
    are considered eligible for REUSE_ARTIFACT consideration. Ties break
    via the bundle's ``created_at`` timestamp (newest wins).
    """
    from .artifact_registry import ArtifactRegistry

    roots = collect_sibling_artifact_roots(pipeline)
    if not roots:
        return None
    try:
        bundle_paths = pipeline._artifact_registry.iter_bundle_paths(
            roots, stage=stage.value
        )
    except Exception as exc:  # pragma: no cover - never let lookup break pipeline
        logger.debug(f"[ArtifactRegistry] iter failed: {exc}")
        return None

    eligible: list[tuple[str, Dict[str, Any]]] = []
    for path in bundle_paths:
        bundle = ArtifactRegistry.load_bundle_from_path(path)
        if bundle is None:
            continue
        md = bundle.get("metadata") or {}
        if not md.get("success", False):
            continue
        bundle_source = str(md.get("source_hash") or "")
        if bundle_source and bundle_source == pipeline._run_id:
            created_at = str(bundle.get("created_at") or "")
            eligible.append((created_at, bundle))

    if not eligible:
        return None
    eligible.sort(key=lambda x: x[0], reverse=True)  # newest first
    return eligible[0][1]


def load_stage_result_from_bundle(
    pipeline: "MainPipeline", stage, bundle: Dict[str, Any], *, reuse_reason: str
) -> "StageResult":
    """Rehydrate a StageResult-shaped contract from a stored artifact bundle.

    ``_register_stage_bundle`` writes enough metadata fields to reproduce
    the *shape* downstream stages expect: success flag, ``data`` dict,
    message, elapsed timing, output/artifact file lists, run_dir and
    fragment_path. The original StageResult object is not pickled — we
    intentionally rebuild from plain JSON so the contract stays
    introspectable/versioned.
    """
    from .stages.common import StageResult
    from .stage_decision import StageDecision as _SD

    md = bundle.get("metadata") or {}
    data: Dict[str, Any] = dict(md.get("stage_result_data") or {})
    data["reused_artifact"] = True
    data["reused_from_run_id"] = bundle.get("run_id") or ""
    data["reused_from_stage"] = bundle.get("stage") or ""
    data["stage_decision"] = _SD.REUSE_ARTIFACT.value
    data["reuse_reason"] = reuse_reason
    data["source_hash"] = pipeline._run_id

    output_files: list[str] = list(md.get("stage_result_output_files") or [])
    artifact_paths: list[str] = list(md.get("stage_result_artifact_paths") or [])

    result = StageResult(
        stage=stage.value,
        success=True,
        message=(
            f"[REUSE_ARTIFACT] {md.get('stage_result_message') or reuse_reason}"
        ),
        data=data,
        elapsed_seconds=float(md.get("stage_result_elapsed_seconds") or 0.0),
        run_dir=str(md.get("stage_result_run_dir") or ""),
        fragment_path=str(md.get("stage_result_fragment_path") or ""),
    )
    # StageResult dataclass does not necessarily declare these attrs on
    # every consumer version; set them via object.__dict__ to be safe.
    if not hasattr(result, "output_files"):
        try:
            result.output_files = output_files  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass
    else:
        result.output_files = output_files  # type: ignore[attr-defined]
    if not hasattr(result, "artifact_paths"):
        try:
            result.artifact_paths = artifact_paths  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass
    else:
        result.artifact_paths = artifact_paths  # type: ignore[attr-defined]
    return result
