from __future__ import annotations

"""Run regression samples from config/regression_samples.json.

The runner executes selected samples, evaluates them against manifest thresholds,
and emits a pass/fail summary for each sample and the whole batch.
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Windows compatibility: ProactorEventLoop is unreliable in this environment —
# its constructor may succeed but asyncio.run() then fails with
# "'_Win32ProactorEventLoop' object has no attribute '_selector'" (a known
# Winsock LSP-pollution symptom, see scripts/run_batch.py 模式 B). Use
# SelectorEventLoop instead, matching scripts/ontology_cli.py.
if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.main_pipeline import MainPipeline, PipelineContext
from src.pipeline.run_context import build_pipeline_run_paths
from scripts.regression_artifact_diagnosis import source_breakdown, trace_relation_type

# Phase B: 确定性 RAG 检索评估最新摘要（scripts/retrieval_eval.py 输出）
DEFAULT_RETRIEVAL_EVAL = REPO_ROOT / "outputs" / "runs" / "ontology" / "rag" / "eval" / "retrieval_eval_latest.json"


@dataclass
class RegressionSample:
    sample_id: str
    input_path: str
    expected: dict[str, Any]
    group_id: str
    name: str
    golden_semantic: dict[str, Any] | None = None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_retrieval_eval_summary(path: Path) -> dict[str, Any]:
    """Phase B: load the deterministic RAG retrieval eval summary (if any).

    Produced by scripts/retrieval_eval.py against the current master and merged
    into the batch summary/baseline so RAG Recall@K is visible in regression.
    Returns {} when the eval output is absent so regression stays self-contained.
    """
    payload = _read_json(path)
    if not payload:
        return {}
    return {
        "corpus": payload.get("corpus", {}),
        "golden": payload.get("golden", {}),
        "retrievers_summary": payload.get("retrievers_summary", {}),
    }


def _compute_evidence_coverage(audit_data: dict[str, Any]) -> float:
    """Derive evidence coverage from audit statistics when not explicitly provided.

    evidence_coverage = nodes_with_evidence / total_nodes
    """
    total = int(audit_data.get("total_nodes", 0) or 0)
    low_evidence = int(audit_data.get("low_evidence_count", 0) or 0)
    if total <= 0:
        return 0.0
    with_evidence = max(0, total - low_evidence)
    return round(with_evidence / total, 4)


def _compute_related_to_share(audit_data: dict[str, Any]) -> float:
    """Derive RELATED_TO share from audit relationship stats."""
    total_rels = int(audit_data.get("total_relationships", 0) or 0)
    if total_rels <= 0:
        return 0.0
    rel_stats = audit_data.get("relationship_stats") or audit_data.get("relationship_type_distribution") or {}
    related_to = 0
    if isinstance(rel_stats, dict):
        for k, v in rel_stats.items():
            if str(k).upper() == "RELATED_TO":
                related_to = int(v)
                break
    return round(related_to / total_rels, 4)


def _compute_classification_consistency(audit_data: dict[str, Any]) -> float:
    """Derive classification consistency from audit data.

    Uses shallow_nodes and dangling_relationships as proxy indicators.
    A higher node quality means higher classification consistency.
    """
    total_nodes = int(audit_data.get("total_nodes", 0) or 0)
    shallow = int(audit_data.get("shallow_nodes", 0) or 0)
    if total_nodes <= 0:
        return 0.0
    return round(max(0.0, 1.0 - shallow / total_nodes), 4)


def _collect_observed_entities(payload: dict[str, Any], stage_keys: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """Collect observed entity dicts from stage results for golden matching.

    Phase B 双层评估口径 (2026-08-29):
    - ``stage_keys=None`` (default) keeps the legacy behavior of scanning ALL
      stage results — used by the legacy ``golden_*`` mixed metrics so the
      existing baseline stays comparable.
    - ``stage_keys=("extract",)`` restricts collection to the raw extraction
      output (Layer 1), answering "did extraction find the golden entity?".
    """
    entities: list[dict[str, Any]] = []
    stage_results = payload.get("stage_results", {}) or {}
    for stage_name, stage_result in stage_results.items():
        if stage_keys is not None and stage_name not in stage_keys:
            continue
        if not isinstance(stage_result, dict):
            continue
        data = stage_result.get("data", {}) or {}
        if not isinstance(data, dict):
            continue
        for key in ("nodes", "entities", "instances"):
            bucket = data.get(key)
            if isinstance(bucket, list):
                entities.extend(item for item in bucket if isinstance(item, dict))
    return entities


def _collect_observed_relations(payload: dict[str, Any], stage_keys: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """Collect observed relation dicts from stage results for golden matching.

    See :func:`_collect_observed_entities` for the stage_keys semantics
    (None = legacy all-stage scan; ("extract",) = Layer-1 extraction only).
    """
    relations: list[dict[str, Any]] = []
    stage_results = payload.get("stage_results", {}) or {}
    for stage_name, stage_result in stage_results.items():
        if stage_keys is not None and stage_name not in stage_keys:
            continue
        if not isinstance(stage_result, dict):
            continue
        data = stage_result.get("data", {}) or {}
        if not isinstance(data, dict):
            continue
        for key in ("relationships", "relations", "edges"):
            bucket = data.get(key)
            if isinstance(bucket, list):
                relations.extend(item for item in bucket if isinstance(item, dict))
    return relations


def _load_master_observed(master_path: str | Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Load merged-master entities/relations for the merge-layer evaluation.

    Phase B 双层评估口径 (2026-08-29): the merge stage's ``StageResult.data``
    carries only counts, so Layer 2 reads the authoritative merged master file
    (``master_ontology_unified.json``) via ``stage_results["merge"].data.master_path``.

    口径 note: in regression mode the master starts as a copy of the baseline
    master, so merge-layer recall measures the *full post-merge state*
    (baseline-inherited entities count too).  Use ``retention`` (merge recall /
    extract recall) to isolate what THIS run's merge preserved from THIS run's
    extraction.
    """
    if not master_path:
        return None
    try:
        master = json.loads(Path(master_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(master, dict):
        return None
    entities: list[dict[str, Any]] = []
    for key in ("nodes", "instances"):
        bucket = master.get(key)
        if isinstance(bucket, list):
            entities.extend(item for item in bucket if isinstance(item, dict))
    relations: list[dict[str, Any]] = []
    rel_bucket = master.get("relationships")
    if isinstance(rel_bucket, list):
        relations.extend(item for item in rel_bucket if isinstance(item, dict))
    return entities, relations


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    return _read_json(manifest_path)


def _load_samples(manifest_path: Path) -> list[RegressionSample]:
    manifest = _load_manifest(manifest_path)
    samples: list[RegressionSample] = []
    for group in manifest.get("sample_groups", []):
        group_id = str(group.get("group_id", ""))
        group_name = str(group.get("name", group_id))
        for sample in group.get("samples", []):
            sample_id = str(sample.get("sample_id") or sample.get("id") or "")
            input_path = str(sample.get("input_path") or sample.get("source_path") or "")
            if not sample_id or not input_path:
                continue
            samples.append(
                RegressionSample(
                    sample_id=sample_id,
                    input_path=input_path,
                    expected=sample.get("expected", {}) or {},
                    group_id=group_id,
                    name=group_name,
                    golden_semantic=sample.get("golden_semantic"),
                )
            )
    return samples


def _evaluate_sample(result_payload: dict[str, Any], expected: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    stage_results = result_payload.get("stage_results", {}) or {}
    relationship_result = stage_results.get("relationship", {}) if isinstance(stage_results, dict) else {}
    dedup_result = stage_results.get("dedup", {}) if isinstance(stage_results, dict) else {}
    audit_result = stage_results.get("audit", {}) if isinstance(stage_results, dict) else {}

    relationship_data = relationship_result.get("data", {}) if isinstance(relationship_result, dict) else {}
    dedup_data = dedup_result.get("data", {}) if isinstance(dedup_result, dict) else {}
    audit_data = audit_result.get("data", {}) if isinstance(audit_result, dict) else {}
    extract_data = {}
    if isinstance(stage_results.get("extract"), dict):
        extract_data = stage_results["extract"].get("data", {}) or {}

    observed = {
        "success": bool(result_payload.get("success", False)),
        "status": str(result_payload.get("status", "")),
        "relation_count": int(
            audit_data.get("relation_count")
            or audit_data.get("total_relationships")
            or relationship_data.get("relation_count")
            or relationship_data.get("total_relationships")
            or 0
        ),
        "entity_count": int(
            audit_data.get("entity_count")
            or audit_data.get("total_nodes")
            or 0
        ),
        "removed_nodes": int(dedup_data.get("removed_nodes", 0) or 0),
        "removed_relationships": int(dedup_data.get("removed_relationships", 0) or 0),
        "evidence_coverage": float(
            audit_data.get("evidence_coverage")
            or _compute_evidence_coverage(audit_data)
        ),
        "quality_score": float(audit_data.get("quality_score", 0.0) or 0.0),
        "related_to_share": float(audit_data.get("related_to_share", 0.0) or _compute_related_to_share(audit_data)),
        "mismerge_rate": float(dedup_data.get("mismerge_rate", 0.0) or 0.0),
        "classification_consistency": float(
            audit_data.get("classification_consistency", 0.0)
            or _compute_classification_consistency(audit_data)
        ),
        "rollback_success": float(audit_data.get("rollback_success", 0.0) or 0.0),
    }

    checks: list[dict[str, Any]] = []
    gate_ok = True

    def add_check(name: str, passed: bool, observed_value: Any, threshold_value: Any, comparator: str) -> None:
        nonlocal gate_ok
        gate_ok = gate_ok and passed
        checks.append({
            "name": name,
            "passed": passed,
            "observed": observed_value,
            "threshold": threshold_value,
            "comparator": comparator,
        })

    add_check("pipeline_success", observed["success"], observed["success"], True, "==")

    min_classification = float(thresholds.get("classification_consistency_min", 0.0) or 0.0)
    add_check(
        "classification_consistency",
        observed["classification_consistency"] >= min_classification,
        observed["classification_consistency"],
        min_classification,
        ">=",
    )

    min_relationship_precision = float(thresholds.get("relationship_precision_min", 0.0) or 0.0)
    add_check(
        "quality_score",
        observed["quality_score"] >= min_relationship_precision,
        observed["quality_score"],
        min_relationship_precision,
        ">=",
    )

    max_related_to_share = float(thresholds.get("related_to_share_max", 1.0) or 1.0)
    add_check(
        "related_to_share",
        observed["related_to_share"] <= max_related_to_share,
        observed["related_to_share"],
        max_related_to_share,
        "<=",
    )

    max_mismerge_rate = float(thresholds.get("dedup_mismerge_rate_max", 1.0) or 1.0)
    add_check(
        "mismerge_rate",
        observed["mismerge_rate"] <= max_mismerge_rate,
        observed["mismerge_rate"],
        max_mismerge_rate,
        "<=",
    )

    min_evidence_coverage = float(thresholds.get("evidence_coverage_min", 0.0) or 0.0)
    add_check(
        "evidence_coverage",
        observed["evidence_coverage"] >= min_evidence_coverage,
        observed["evidence_coverage"],
        min_evidence_coverage,
        ">=",
    )

    min_rollback_success = float(thresholds.get("rollback_success_min", 0.0) or 0.0)
    add_check(
        "rollback_success",
        observed["rollback_success"] >= min_rollback_success,
        observed["rollback_success"],
        min_rollback_success,
        ">=",
    )

    expected_min_relations = expected.get("min_relations")
    if expected_min_relations is not None:
        add_check(
            "expected_min_relations",
            observed["relation_count"] >= int(expected_min_relations),
            observed["relation_count"],
            int(expected_min_relations),
            ">=",
        )

    expected_max_removed_nodes = expected.get("max_removed_nodes")
    if expected_max_removed_nodes is not None:
        add_check(
            "expected_max_removed_nodes",
            observed["removed_nodes"] <= int(expected_max_removed_nodes),
            observed["removed_nodes"],
            int(expected_max_removed_nodes),
            "<=",
        )

    expected_min_quality = expected.get("min_quality_score")
    if expected_min_quality is not None:
        add_check(
            "expected_min_quality_score",
            observed["quality_score"] >= float(expected_min_quality),
            observed["quality_score"],
            float(expected_min_quality),
            ">=",
        )

    # E (2026-08-15): golden semantic gate — recall/precision vs the golden set.
    # Phase B 双层评估口径 (2026-08-29): in addition to the legacy mixed-bucket
    # metrics (kept for baseline comparability), evaluate the golden set against
    #   Layer 1 (extract): raw extraction output  -> "can extraction find it?"
    #   Layer 2 (merge):   post-merge master file  -> "did merge preserve it?"
    # plus a retention derivative (merge recall / extract recall) that isolates
    # merge-stage loss from extraction-stage miss.
    golden = result_payload.get("golden_semantic")
    golden_metrics_out = None
    layered_golden: dict[str, Any] | None = None
    if golden:
        from scripts.golden_semantic_eval import load_golden_set, golden_metrics
        golden_measure = load_golden_set(golden)
        observed_entities = _collect_observed_entities(result_payload)
        observed_relations = _collect_observed_relations(result_payload)
        golden_metrics_out = golden_metrics(golden_measure, observed_entities, observed_relations)
        add_check(
            "golden_entity_recall",
            golden_metrics_out["entity_recall"] >= 0.8,
            golden_metrics_out["entity_recall"],
            0.8,
            ">=",
        )
        add_check(
            "golden_relation_recall",
            golden_metrics_out["relation_recall"] >= 0.6,
            golden_metrics_out["relation_recall"],
            0.6,
            ">=",
        )
        add_check(
            "golden_no_false_positives",
            not golden_metrics_out["false_positives"],
            len(golden_metrics_out["false_positives"]),
            0,
            "==",
        )

        # --- Layer 1: extract (raw extraction output only) -------------------
        extract_entities = _collect_observed_entities(result_payload, stage_keys=("extract",))
        extract_relations = _collect_observed_relations(result_payload, stage_keys=("extract",))
        extract_layer = golden_metrics(golden_measure, extract_entities, extract_relations)

        # --- Layer 2: merge (post-merge master file) -------------------------
        merge_stage_data = {}
        if isinstance(stage_results.get("merge"), dict):
            merge_stage_data = stage_results["merge"].get("data", {}) or {}
        master_observed = _load_master_observed(merge_stage_data.get("master_path"))
        merge_layer: dict[str, Any] | None = None
        retention: dict[str, Any] | None = None
        if master_observed is not None:
            master_entities, master_relations = master_observed
            merge_layer = golden_metrics(golden_measure, master_entities, master_relations)
            # Retention: fraction of golden found at extract layer that survived
            # into the merged master.  Guarded against div-by-zero; when the
            # extract layer missed everything, retention is undefined (None).
            if float(extract_layer["entity_recall"]) > 0:
                entity_retention = round(
                    float(merge_layer["entity_recall"]) / float(extract_layer["entity_recall"]), 4
                )
            else:
                entity_retention = None
            if float(extract_layer["relation_recall"]) > 0:
                relation_retention = round(
                    float(merge_layer["relation_recall"]) / float(extract_layer["relation_recall"]), 4
                )
            else:
                relation_retention = None
            retention = {
                "entity_retention": entity_retention,
                "relation_retention": relation_retention,
            }

        layered_golden = {
            "extract": extract_layer,
            "merge": merge_layer,
            "retention": retention,
            "master_path": merge_stage_data.get("master_path"),
            "口径": {
                "extract": "本次运行抽取产物（纯增量，无 baseline 继承）",
                "merge": "合并后 master 全量状态（regression 模式含 baseline 继承实体）",
                "retention": "merge 层 recall / extract 层 recall，隔离 merge 阶段损耗",
            },
        }

        # Config-gated checks: enforce only when the manifest/config declares a
        # threshold; otherwise the layered metrics stay observational so the
        # first rollout never breaks CI (same policy as the RAG gate rollout).
        layer_thresholds = [
            ("extract_entity_recall", float(extract_layer["entity_recall"]), thresholds.get("extract_entity_recall_min")),
            ("extract_relation_recall", float(extract_layer["relation_recall"]), thresholds.get("extract_relation_recall_min")),
        ]
        if merge_layer is not None:
            layer_thresholds += [
                ("merge_entity_recall", float(merge_layer["entity_recall"]), thresholds.get("merge_entity_recall_min")),
                ("merge_relation_recall", float(merge_layer["relation_recall"]), thresholds.get("merge_relation_recall_min")),
            ]
        if retention is not None and retention.get("entity_retention") is not None:
            layer_thresholds.append(
                ("entity_retention", float(retention["entity_retention"]), thresholds.get("entity_retention_min"))
            )
        for name, value, threshold in layer_thresholds:
            if threshold is None:
                continue
            add_check(name, value >= float(threshold), value, float(threshold), ">=")

    return {
        "passed": gate_ok,
        "observed": observed,
        "checks": checks,
        "golden": golden_metrics_out,
        "layered_golden": layered_golden,
        "quality_observability": _build_quality_observability(result_payload),
    }


def _build_quality_observability(result_payload: dict[str, Any]) -> dict[str, Any]:
    """P1-5 (Gate D): warn-only observability block (drift/evidence/dead-relations).

    Never affects pass/fail: no add_check calls, no threshold reads. Compares
    the extract layer against the merged master file and reports evidence +
    dead-relation stats of the master.
    """
    from scripts.quality_observability import (
        build_quality_observability,
        load_master_entities_relations,
    )

    try:
        stage_results = result_payload.get("stage_results", {}) or {}
        merge_data = {}
        if isinstance(stage_results.get("merge"), dict):
            merge_data = stage_results["merge"].get("data", {}) or {}
        master_entities, master_relations = load_master_entities_relations(merge_data.get("master_path"))
        extract_entities = _collect_observed_entities(result_payload, stage_keys=("extract",))
        extract_relations = _collect_observed_relations(result_payload, stage_keys=("extract",))
        return build_quality_observability(extract_entities, extract_relations, master_entities, master_relations)
    except Exception as exc:  # pragma: no cover - observability must not break the run
        return {"error": f"{type(exc).__name__}: {exc}"}


async def _run_sample(
    sample: RegressionSample,
    output_dir: Path,
    domain: str,
    manifest_thresholds: dict[str, Any],
    *,
    max_cost_usd: float = 0.0,
    max_calls: int = 0,
    max_total_tokens: int = 0,
    degraded_offline: bool = False,
    production_override: bool = False,
) -> dict[str, Any]:
    run_id = f"REG-{sample.sample_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # B-8: when --production is passed, run the pipeline in PRODUCTION mode so
    # LLM-dependent stages actually execute and produce a meaningful baseline.
    regression_mode = not production_override
    ctx = PipelineContext(source_path=sample.input_path, domain=domain, run_id=run_id, output_dir=str(output_dir), regression_mode=regression_mode)
    if regression_mode:
        baseline_master = REPO_ROOT / "outputs" / "runs" / "ontology" / "master_ontology_unified.json"
        sample_master = output_dir / run_id / "regression_master" / "master_ontology_unified.json"
        sample_master.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(baseline_master, sample_master)
        ctx.metadata["regression_master_path"] = str(sample_master)
    # C-1: propagate governance caps into PipelineContext so MainPipeline.__init__
    # (_arm_governance_from_context) picks them up without coupling to argparse.
    ctx.metadata["governance"] = {
        "max_cost_usd": max_cost_usd,
        "max_calls": max_calls,
        "max_total_tokens": max_total_tokens,
        "degraded_offline": degraded_offline,
    }
    ctx.metadata["regression_manifest"] = {
        "sample_id": sample.sample_id,
        "group_id": sample.group_id,
        "group_name": sample.name,
        "expected": sample.expected,
        "source_path": sample.input_path,
        "thresholds": manifest_thresholds,
        "regression_mode": True,
    }
    # C-3 / C-1 治理层：每个 sample 运行前清空 LLM 监控累计，避免样本之间 calls/costs 串项
    try:
        from src.application.services.llm_monitor import get_llm_monitor
        get_llm_monitor().reset()
    except Exception:
        pass
    # P1-3 (2026-09-01):同样清空 evidence-gate 决策计数，避免样本之间 evidence_signals 串项
    try:
        from src.application.rag.evidence_gate import get_evidence_gate
        get_evidence_gate().reset_decision_counts()
    except Exception:
        pass
    pipeline = MainPipeline(ctx)
    result = await pipeline.run()
    run_paths = build_pipeline_run_paths(ctx.output_dir, run_id)

    # C-3: collect cost / token / latency summary from the LLM monitor so
    # the baseline payload carries stable aggregation fields for comparison.
    llm_cost_summary = _collect_llm_cost_summary()

    payload = {
        "sample_id": sample.sample_id,
        "group_id": sample.group_id,
        "run_id": run_id,
        "source_path": sample.input_path,
        "status": result.status.value,
        "success": result.success,
        "output_files": result.output_files,
        "stage_results": {k: v.to_dict() for k, v in result.stage_results.items()},
        "run_root": str(run_paths.run_root),
        "artifact_root": str(run_paths.artifact_root),
        "audit_root": str(run_paths.audit_root),
        "regression_root": str(run_paths.regression_root),
        "golden_semantic": sample.golden_semantic,
        "llm_cost": llm_cost_summary,
    }
    evaluation = _evaluate_sample(payload, sample.expected, manifest_thresholds)
    payload["evaluation"] = evaluation
    _attach_artifact_diagnostics(payload, run_paths, run_id, sample.sample_id)
    payload["pass_fail"] = "PASS" if evaluation["passed"] else "FAIL"
    return payload


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _attach_artifact_diagnostics(payload: dict[str, Any], run_paths: Any, run_id: str, sample_id: str) -> None:
    payload["diagnostics"] = {"source_breakdown": {}, "merged_master_source_breakdown": {}, "relation_traces": {}}
    artifact_root = Path(run_paths.artifact_root)
    extract_stage = _load_json_if_exists(artifact_root / f"{run_id}-extract-bundle.json")
    merge_stage = _load_json_if_exists(artifact_root / f"{run_id}-merge-bundle.json")
    relationship_stage = _load_json_if_exists(artifact_root / f"{run_id}-relationship-bundle.json")
    if extract_stage:
        payload["diagnostics"]["source_breakdown"] = source_breakdown(extract_stage)
    master_path = (((merge_stage.get("metadata") or {}).get("stage_result_data") or {}).get("master_path"))
    if master_path:
        master = _load_json_if_exists(Path(master_path))
        if master:
            payload["diagnostics"]["merged_master_source_breakdown"] = source_breakdown({"metadata": {"stage_result_data": master}})
    if sample_id == "ebom-mbom-2023":
        payload["diagnostics"]["relation_traces"]["DERIVED_FROM"] = trace_relation_type(extract_stage, relationship_stage, "DERIVED_FROM")


def _render_batch_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"- Executed at: `{summary['executed_at']}`",
        f"- Overall: **{summary['overall_status']}**",
        f"- Passed: `{summary['passed_count']}`",
        f"- Failed: `{summary['failed_count']}`",
        "",
        "## Sample Results",
        "",
    ]
    for item in summary.get("results", []):
        lines.extend([
            f"### {item['sample_id']} ({item['pass_fail']})",
            f"- Group: `{item['group_id']}`",
            f"- Run ID: `{item['run_id']}`",
            f"- Source: `{item['source_path']}`",
            f"- Status: `{item['status']}`",
            f"- Quality: `{item.get('quality_score', 'N/A')}`",
            f"- Evidence: `{item.get('evidence_coverage', 'N/A')}`",
        ])
        # Chunk statistics if available in stage results
        stage_results = item.get("stage_results", {})
        for stage_name, stage_data in stage_results.items():
            if not isinstance(stage_data, dict):
                continue
            data = stage_data.get("data", {}) if isinstance(stage_data.get("data"), dict) else {}
            chunk_stats = data.get("chunking_summary") or data.get("chunk_stats")
            if chunk_stats and isinstance(chunk_stats, dict):
                lines.extend([
                    f"- **Chunking** (`{stage_name}`):",
                    f"  - Strategy: `{chunk_stats.get('strategy_used', 'N/A')}`",
                    f"  - Total chunks: `{chunk_stats.get('total_chunks', 'N/A')}`",
                    f"  - Avg chunk chars: `{chunk_stats.get('avg_chunk_chars', 'N/A')}`",
                    f"  - Oversized: `{chunk_stats.get('oversized_chunks', 0)}`",
                    f"  - Pre-split: `{chunk_stats.get('pre_split_triggered', False)}`",
                    f"  - Quality score: `{chunk_stats.get('chunk_quality_score', 'N/A')}`",
                ])
                break
        lines.extend([
            f"- Evidence coverage: `{item['evaluation']['observed']['evidence_coverage']}`",
            f"- Quality score: `{item['evaluation']['observed']['quality_score']}`",
        ])
        # C-3: LLM cost / token / latency per sample
        llm_cost = item.get("llm_cost") or {}
        if llm_cost:
            lines.extend([
                f"- **LLM Cost**: calls=`{llm_cost.get('total_calls', 0)}` tokens=`{llm_cost.get('total_tokens', 0)}` cost=`${llm_cost.get('total_cost_estimate', 0):.4f}` latency=`{llm_cost.get('total_duration_ms', 0)}ms`",
            ])
        lines.append("")
        for check in item["evaluation"]["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            lines.append(f"- [{mark}] {check['name']}: observed={check['observed']} threshold={check['threshold']} comparator={check['comparator']}")
        lines.append("")
    return "\n".join(lines)


def _collect_evidence_signals() -> dict[str, Any]:
    """P1-3: snapshot RAG evidence-gate decision telemetry (best effort).

    The gate records ANSWER / RETRIEVE_MORE / ESCALATE counts plus the last
    decision; regression runs without RAG answering legitimately report empty
    counts.
    """
    try:
        from src.application.rag.evidence_gate import get_evidence_gate
        return get_evidence_gate().get_evidence_signals()
    except Exception:
        return {}


def _collect_llm_cost_summary() -> dict[str, Any]:
    """C-3: snapshot LLM cost, token, and latency totals.

    P1-3 (2026-09-01): also snapshots C-5 telemetry — ``route_decisions``
    (routing tier counts recorded by ``LLMGateway.execute_routed``) and
    ``evidence_signals`` (RAG evidence gate decisions) — so the baseline
    carries them alongside the cost fields.
    """
    try:
        from src.application.services.llm_monitor import get_llm_monitor
        summary = get_llm_monitor().get_summary()
        stages = summary.get("stage_metrics") or {}
        return {
            "total_calls": summary.get("total_calls", 0),
            "total_tokens": summary.get("total_tokens", 0),
            "total_prompt_tokens": sum(s.get("total_prompt_tokens", 0) for s in stages.values()),
            "total_completion_tokens": sum(s.get("total_completion_tokens", 0) for s in stages.values()),
            "total_cost_estimate": round(summary.get("total_cost_estimate", 0.0), 6),
            "total_duration_ms": summary.get("total_duration_ms", 0),
            "unknown_price_calls": summary.get("unknown_price_calls", 0),
            "stage_metrics": stages,
            "route_decisions": summary.get("route_decisions") or {},
            "evidence_signals": _collect_evidence_signals(),
        }
    except Exception:
        return {}


def _render_retrieval_eval_lines(summary: dict[str, Any]) -> list[str]:
    """Phase B: render the deterministic RAG retrieval eval summary (if present)."""
    eval_data = summary.get("retrieval_eval") or {}
    if not eval_data:
        return []
    lines = ["## RAG Retrieval Eval (Phase B)", ""]
    corpus = eval_data.get("corpus") or {}
    golden = eval_data.get("golden") or {}
    lines.extend([
        f"- Corpus chunks: `{corpus.get('chunk_count', 'N/A')}` ({corpus.get('source', 'N/A')})",
        f"- Golden entities: `{golden.get('entities_total', 'N/A')}`, queries: `{golden.get('queries_total', 'N/A')}`, not-in-master: `{golden.get('not_in_master_count', 'N/A')}`",
    ])
    for rname, rsum in (eval_data.get("retrievers_summary") or {}).items():
        lines.append(
            f"- **{rname}**: Hit@1=`{rsum.get('hit@1', 'N/A')}` Hit@5=`{rsum.get('hit@5', 'N/A')}` "
            f"Recall@5=`{rsum.get('recall@5', 'N/A')}` MRR=`{rsum.get('mrr', 'N/A')}`"
        )
    lines.append("")
    return lines


def _aggregate_quality_observability(results: list[dict[str, Any]]) -> dict[str, Any]:
    """P1-5 (Gate D): batch-level mean of per-sample quality_observability blocks."""
    from scripts.quality_observability import aggregate_quality_observability

    blocks = [
        (item.get("evaluation") or {}).get("quality_observability") or {}
        for item in results
    ]
    blocks = [b for b in blocks if isinstance(b, dict) and "error" not in b]
    if not blocks:
        return {"metrics": {}, "口径": "无有效 quality_observability 块"}
    return aggregate_quality_observability(blocks)


def _render_quality_observability_lines(summary: dict[str, Any]) -> list[str]:
    """Render the batch-level quality observability section (warn-only)."""
    obs = summary.get("quality_observability_summary") or {}
    metrics = obs.get("metrics") or {}
    if not metrics:
        return []
    lines = ["## Quality Observability (P1-5, warn-only)", ""]
    lines.append("口径：extract→master drift / merged master evidence / dead-relations / RELATED_TO share；不进 gate")
    lines.append("")
    lines.append("| metric | mean | n |")
    lines.append("|---|---|---|")
    for key in ("missing_ratio", "real_missing_ratio", "ghost_missing_count", "evidence_coverage", "dead_ratio", "source_cq_coverage", "related_to_share"):
        entry = metrics.get(key)
        if entry:
            lines.append(f"| {key} | {entry['mean']} | {entry['n']} |")
    recurring = obs.get("recurring_missing_in_master") or {}
    if recurring:
        top = sorted(recurring.items(), key=lambda kv: -kv[1])[:10]
        lines.append("")
        lines.append(f"- recurring missing_in_master: {', '.join(f'{k}×{v}' for k, v in top)}")
    real_recurring = obs.get("recurring_real_missing_in_master") or {}
    if real_recurring:
        top = sorted(real_recurring.items(), key=lambda kv: -kv[1])[:10]
        lines.append(f"- recurring real_missing_in_master: {', '.join(f'{k}×{v}' for k, v in top)}")
    dead_types = obs.get("dead_by_type_total") or {}
    if dead_types:
        top = sorted(dead_types.items(), key=lambda kv: -kv[1])[:10]
        lines.append(f"- dead_by_type_total: {', '.join(f'{k}×{v}' for k, v in top)}")
    lines.append("")
    return lines


def _aggregate_layered_golden(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Phase B 双层评估口径: batch-level mean of per-sample layered golden metrics.

    Aggregates extract-layer and merge-layer entity/relation recall plus the
    retention derivative across samples that carry ``layered_golden``.  Samples
    without golden_semantic (or without a merge master) are skipped per metric.
    """
    sums: dict[str, list[float]] = {}
    for item in results:
        layered = ((item.get("evaluation") or {}).get("layered_golden")) or {}
        extract_layer = layered.get("extract") or {}
        merge_layer = layered.get("merge") or {}
        retention = layered.get("retention") or {}
        pairs = [
            ("extract_entity_recall", extract_layer.get("entity_recall")),
            ("extract_relation_recall", extract_layer.get("relation_recall")),
            ("merge_entity_recall", merge_layer.get("entity_recall") if merge_layer else None),
            ("merge_relation_recall", merge_layer.get("relation_recall") if merge_layer else None),
            ("entity_retention", retention.get("entity_retention")),
            ("relation_retention", retention.get("relation_retention")),
        ]
        for key, value in pairs:
            if isinstance(value, (int, float)):
                sums.setdefault(key, []).append(float(value))
    metrics = {
        key: {"mean": round(sum(vals) / len(vals), 4), "n": len(vals)}
        for key, vals in sums.items()
    }
    return {"metrics": metrics, "口径": "extract=抽取层纯增量 / merge=合并后master全量 / retention=merge与extract recall之比"}


def _render_layered_golden_lines(summary: dict[str, Any]) -> list[str]:
    """Render the batch-level dual-layer golden section for the MD report."""
    layered = summary.get("layered_golden_summary") or {}
    metrics = layered.get("metrics") or {}
    if not metrics:
        return []
    lines = ["## Golden Dual-Layer Eval (Phase B)", ""]
    lines.append("口径：extract=抽取层纯增量 / merge=合并后master全量（regression 模式含 baseline 继承）/ retention=merge与extract recall之比")
    lines.append("")
    lines.append("| metric | mean | n |")
    lines.append("|---|---|---|")
    for key in (
        "extract_entity_recall",
        "extract_relation_recall",
        "merge_entity_recall",
        "merge_relation_recall",
        "entity_retention",
        "relation_retention",
    ):
        entry = metrics.get(key)
        if entry:
            lines.append(f"| {key} | {entry['mean']} | {entry['n']} |")
    lines.append("")
    return lines


def _render_batch_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Regression Summary", "",
        f"- Manifest: `{summary['manifest']}`",
        f"- Executed at: `{summary['executed_at']}`",
        f"- Overall: **{summary['overall_status']}**",
        f"- Passed: `{summary['passed_count']}`",
        f"- Failed: `{summary['failed_count']}`", "",
    ]
    lines.extend(_render_retrieval_eval_lines(summary))
    lines.extend(_render_layered_golden_lines(summary))
    lines.extend(_render_quality_observability_lines(summary))
    for item in summary.get("results", []):
        evaluation = item["evaluation"]
        lines.extend([f"## {item['sample_id']} ({item['pass_fail']})", f"- Source: `{item['source_path']}`", ""])
        for check in evaluation.get("checks", []):
            mark = "PASS" if check["passed"] else "FAIL"
            lines.append(f"- [{mark}] {check['name']}: observed={check['observed']} threshold={check['threshold']}")
        layered = evaluation.get("layered_golden")
        if layered:
            extract_layer = layered.get("extract") or {}
            merge_layer = layered.get("merge") or {}
            retention = layered.get("retention") or {}
            lines.append("- Layered golden (Phase B):")
            lines.append(
                f"  - extract: entity_recall=`{extract_layer.get('entity_recall', 'N/A')}` "
                f"relation_recall=`{extract_layer.get('relation_recall', 'N/A')}`"
            )
            if merge_layer:
                lines.append(
                    f"  - merge: entity_recall=`{merge_layer.get('entity_recall', 'N/A')}` "
                    f"relation_recall=`{merge_layer.get('relation_recall', 'N/A')}`"
                )
            else:
                lines.append("  - merge: `N/A` (master file not found at merge stage master_path)")
            if retention:
                lines.append(
                    f"  - retention: entity=`{retention.get('entity_retention', 'N/A')}` "
                    f"relation=`{retention.get('relation_retention', 'N/A')}`"
                )
        lines.append("")
    return "\n".join(lines)


def _baseline_payload(summary: dict[str, Any]) -> dict[str, Any]:
    """Strip a regression run down to the stable fields used as a baseline."""
    # C-3: aggregate per-sample cost / token / latency into a batch-level summary.
    batch_cost = _aggregate_batch_cost(summary.get("results", []))
    # P1-3 (2026-09-01): batch-level C-5 telemetry (route decisions + evidence
    # gate signals) so the baseline carries them for before/after comparison.
    return {
        "manifest": summary.get("manifest"),
        "executed_at": summary.get("executed_at"),
        "count": summary.get("count", 0),
        "passed_count": summary.get("passed_count", 0),
        "failed_count": summary.get("failed_count", 0),
        "overall_status": summary.get("overall_status"),
        "thresholds": summary.get("thresholds", {}),
        "retrieval_eval": summary.get("retrieval_eval", {}),
        "layered_golden_summary": summary.get("layered_golden_summary", {}),
        "quality_observability_summary": summary.get("quality_observability_summary", {}),
        "batch_cost": batch_cost,
        "route_decisions": _merge_route_decisions(summary.get("results", [])),
        "evidence_signals": _merge_evidence_signals(summary.get("results", [])),
        "results": summary.get("results", []),
    }


def _merge_route_decisions(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """P1-3: merge per-sample route_decisions into batch-level tier counts."""
    merged: dict[str, dict[str, int]] = {}
    for item in results:
        route = (item.get("llm_cost") or {}).get("route_decisions") or {}
        for kind, tiers in route.items():
            bucket = merged.setdefault(str(kind), {})
            for tier, count in (tiers or {}).items():
                bucket[str(tier)] = bucket.get(str(tier), 0) + int(count or 0)
    return merged


def _merge_evidence_signals(results: list[dict[str, Any]]) -> dict[str, Any]:
    """P1-3: merge per-sample evidence gate decision counts at batch level."""
    merged_counts: dict[str, int] = {}
    for item in results:
        signals = (item.get("llm_cost") or {}).get("evidence_signals") or {}
        counts = signals.get("decision_counts") or {}
        for action, count in counts.items():
            merged_counts[str(action)] = merged_counts.get(str(action), 0) + int(count or 0)
    return {"decision_counts": merged_counts}


def _aggregate_batch_cost(results: list[dict[str, Any]]) -> dict[str, Any]:
    """C-3: roll up per-sample LLM cost / token / latency into batch totals."""
    total_calls = 0
    total_tokens = 0
    total_cost = 0.0
    total_duration_ms = 0
    unknown_price = 0
    for item in results:
        cost = item.get("llm_cost") or {}
        total_calls += int(cost.get("total_calls", 0) or 0)
        total_tokens += int(cost.get("total_tokens", 0) or 0)
        total_cost += float(cost.get("total_cost_estimate", 0.0) or 0.0)
        total_duration_ms += int(cost.get("total_duration_ms", 0) or 0)
        unknown_price += int(cost.get("unknown_price_calls", 0) or 0)
    return {
        "total_calls": total_calls,
        "total_tokens": total_tokens,
        "total_cost_estimate": round(total_cost, 6),
        "total_duration_ms": total_duration_ms,
        "unknown_price_calls": unknown_price,
    }


def _write_baseline(summary: dict[str, Any], baseline_path: Path) -> None:
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(_baseline_payload(summary), ensure_ascii=False, indent=2), encoding="utf-8")


def _run_compare(before_path: Path, after_path: Path, output_path: Path, emit_html: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compare_cmd = [
        sys.executable,
        "scripts/compare_pipeline_results.py",
        "--before",
        str(before_path),
        "--after",
        str(after_path),
        "--output",
        str(output_path),
    ]
    if emit_html:
        compare_cmd.append("--html")
    subprocess.run(compare_cmd, check=False)


def _render_batch_html(summary: dict[str, Any]) -> str:
    rows = []
    for item in summary.get("results", []):
        rows.append(
            "<tr>"
            f"<td>{item['sample_id']}</td>"
            f"<td>{item['group_id']}</td>"
            f"<td>{item['pass_fail']}</td>"
            f"<td>{item['status']}</td>"
            f"<td>{item['evaluation']['observed']['quality_score']}</td>"
            f"<td>{item['evaluation']['observed']['evidence_coverage']}</td>"
            f"<td>{item['run_id']}</td>"
            "</tr>"
        )
    rows_html = "\n".join(rows)
    return "<html><body><h1>Regression Summary</h1><p>Overall: {}</p><table>{}</table></body></html>".format(summary["overall_status"], rows_html)



def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run regression samples from manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_regression_samples.py --write-baseline\n"
            "  python scripts/run_regression_samples.py --sample-id S1 --html\n"
            "  python scripts/run_regression_samples.py --compare-only\n"
            "  python scripts/run_regression_samples.py --compare-only --before BASE.json --after CANDIDATE.json\n"
        ),
    )
    parser.add_argument("--manifest", default="config/regression_samples.json", help="Path to regression manifest")
    parser.add_argument("--output-dir", default="outputs/runs/ontology/pipeline", help="Pipeline output root")
    parser.add_argument("--domain", default="SUPPLY_CHAIN", help="Pipeline domain")
    parser.add_argument("--sample-id", action="append", help="Optional sample IDs to run")
    parser.add_argument("--html", action="store_true", help="Also write HTML summary")
    parser.add_argument("--baseline-output", default="outputs/runs/ontology/baselines/regression_summary_baseline.json", help="Path to write baseline summary JSON")
    parser.add_argument("--write-baseline", action="store_true", help="Write the current summary to the baseline output path")
    parser.add_argument("--compare-only", action="store_true", help="Skip pipeline execution and only compare summaries")
    parser.add_argument("--before", help="Baseline summary to compare against in compare-only mode")
    parser.add_argument("--after", help="Candidate summary to compare in compare-only mode")
    # C-1: LLM governance caps — same semantics as ontology_cli.py / run_batch.py.
    # Default 0 means "no cap" so unconfigured runs are 100% backward compatible.
    parser.add_argument("--max-cost-usd", type=float, default=0.0, help="Hard cap on cumulative LLM spend (USD, 0 = unlimited)")
    parser.add_argument("--max-calls", type=int, default=0, help="Hard cap on total LLM API calls (0 = unlimited)")
    parser.add_argument("--max-total-tokens", type=int, default=0, help="Hard cap on total LLM tokens (prompt+completion, 0 = unlimited)")
    parser.add_argument("--degraded-offline", action="store_true", help="Force DEGRADE mode for LLM-heavy stages (offline/no-LLM fallback)")
    parser.add_argument("--production", action="store_true", help="Override regression_mode=OFFLINE and run LLM-dependent stages (needed for --write-baseline)")
    parser.add_argument("--retrieval-eval", default=str(DEFAULT_RETRIEVAL_EVAL), help="Path to retrieval_eval_latest.json to merge into the summary (Phase B)")
    parser.add_argument(
        "--enforce-rag-gate",
        action="store_true",
        help="If the retrieval gate in retrieval_eval_latest.json is FAIL, flip the batch overall_status to FAIL (hard gate)",
    )
    parser.add_argument(
        "--skip-governed-rebuild",
        action="store_true",
        help="Skip governed RAG packet rebuild after every master merge (sets RAG_GOVERNED_REBUILD=0 env for subprocesses). "
             "Useful for CI matrix runs where only the last rank should publish the retrieval asset.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.skip_governed_rebuild:
        os.environ["RAG_GOVERNED_REBUILD"] = "0"
        print("[RAG] --skip-governed-rebuild ON -> merge/hook will not rebuild governed packet.")

    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path)
    manifest_thresholds = manifest.get("acceptance", {}) or {}
    samples = _load_samples(manifest_path)
    if args.sample_id:
        selected = set(args.sample_id)
        samples = [s for s in samples if s.sample_id in selected]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = output_dir / "regression_reports"
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    summary_path = run_dir / f"regression_summary_{stamp}.json"
    markdown_path = run_dir / f"regression_summary_{stamp}.md"
    html_path = run_dir / f"regression_summary_{stamp}.html"
    latest_summary = run_dir / "latest_regression_summary.json"
    latest_markdown = run_dir / "latest_regression_summary.md"
    latest_html = run_dir / "latest_regression_summary.html"

    summary: dict[str, Any] = {
        "manifest": str(manifest_path),
        "executed_at": datetime.now().isoformat(),
        "count": 0,
        "passed_count": 0,
        "failed_count": 0,
        "overall_status": "PASS",
        "thresholds": manifest_thresholds,
        "results": [],
    }

    baseline_output = Path(args.baseline_output)
    manifest_compare = manifest.get("compare_with") if isinstance(manifest, dict) else None
    manifest_before = Path(str(manifest_compare.get("before"))) if isinstance(manifest_compare, dict) and manifest_compare.get("before") else baseline_output
    manifest_after = Path(str(manifest_compare.get("after"))) if isinstance(manifest_compare, dict) and manifest_compare.get("after") else latest_summary

    if args.compare_only:
        before_path = Path(args.before) if args.before else manifest_before
        after_path = Path(args.after) if args.after else manifest_after
        if not after_path.exists():
            raise FileNotFoundError(f"Compare target not found: {after_path}")
        if not before_path.exists():
            raise FileNotFoundError(f"Baseline not found: {before_path}")
        compare_output = run_dir / f"compare_{stamp}.json"
        _run_compare(before_path, after_path, compare_output, args.html)
        print(compare_output)
        return 0

    results = []

    # ============================================================
    # C-1: LLM HARD BUDGET GATE — arm BEFORE any LLM-dependent
    # _run_sample executes. Caps are enforced inside the gateway
    # *before* any provider is contacted, so costs never silently
    # accumulate. Default 0 == unlimited, fully backward-compatible.
    # ============================================================
    max_cost_usd = float(getattr(args, "max_cost_usd", 0.0) or 0.0)
    max_calls = int(getattr(args, "max_calls", 0) or 0)
    max_total_tokens = int(getattr(args, "max_total_tokens", 0) or 0)
    degraded_offline = bool(getattr(args, "degraded_offline", False))

    budget_any_active = max_cost_usd > 0 or max_calls > 0 or max_total_tokens > 0
    if budget_any_active or degraded_offline:
        try:
            from src.application.bootstrap.llm_gateway import get_llm_gateway
            gateway = get_llm_gateway()
            if budget_any_active:
                gateway.set_budget_policy(
                    max_cost_usd=max_cost_usd,
                    max_calls=max_calls,
                    max_total_tokens=max_total_tokens,
                )
                print(
                    f"[GOV] [BudgetGate] Armed: max_cost=${max_cost_usd:.4f}, "
                    f"max_calls={max_calls}, max_tokens={max_total_tokens}"
                )
            if degraded_offline:
                print("[GOV] [DegradedMode] Forcing DEGRADE mode for LLM-dependent stages")
        except Exception as _exc:  # pragma: no cover - best-effort arming
            print(f"[GOV] [WARN] Could not arm LLM governance: {_exc!r}", file=sys.stderr)

    # [WINDOWS FIX] Winsock LSP pollution makes socket.socketpair() fail with
    # WinError 10038 inside BaseSelectorEventLoop._make_self_pipe, which blows
    # up the very first asyncio.run(). Reuse the scripts/ontology_cli.py patch:
    # pre-create the socketpair (TCP-loopback fallback, AF_INET bypasses the
    # LSP) and hand it to the event loop; if even that fails, skip the
    # self-pipe entirely (single-threaded pipelines are safe without it).
    try:
        from scripts.ontology_cli import _patch_make_self_pipe, _precreate_socketpair

        pre_ssock = pre_csock = None
        try:
            pre_ssock, pre_csock = _precreate_socketpair()
        except OSError:
            pass
        _patch_make_self_pipe(pre_ssock, pre_csock)
    except Exception as _exc:  # pragma: no cover - best-effort availability
        print(f"[WIN] Could not apply asyncio socketpair patch: {_exc!r}", file=sys.stderr)

    for sample in samples:
        results.append(
            asyncio.run(
                _run_sample(
                    sample,
                    output_dir,
                    args.domain,
                    manifest_thresholds,
                    max_cost_usd=max_cost_usd,
                    max_calls=max_calls,
                    max_total_tokens=max_total_tokens,
                    degraded_offline=degraded_offline,
                    production_override=args.production,
                )
            )
        )

    passed_count = sum(1 for r in results if r["pass_fail"] == "PASS")
    failed_count = len(results) - passed_count
    overall_status = "PASS" if failed_count == 0 else "FAIL"
    # Phase B: attach deterministic RAG retrieval eval summary (best-effort).
    retrieval_eval = _load_retrieval_eval_summary(Path(args.retrieval_eval))
    if retrieval_eval:
        print(f"[RAG] [Eval] Merged retrieval eval summary from: {args.retrieval_eval}")
    else:
        print(f"[RAG] [Eval] No retrieval eval summary found at: {args.retrieval_eval} (skipping merge)")
    summary = {
        "manifest": str(manifest_path),
        "executed_at": datetime.now().isoformat(),
        "count": len(results),
        "passed_count": passed_count,
        "failed_count": failed_count,
        "overall_status": overall_status,
        "thresholds": manifest_thresholds,
        "retrieval_eval": retrieval_eval,
        "layered_golden_summary": _aggregate_layered_golden(results),
        "quality_observability_summary": _aggregate_quality_observability(results),
        "results": results,
    }
    # Phase B: Retrieval hard gate -- enforce FAIL when retrieval_eval_latest gate
    # is FAIL (e.g. Chinese aliases regression, chunker / embedder / adapter
    # change that degraded retrieval quality).
    rag_gate = ((retrieval_eval or {}).get("gate") or {}).get("passed") if retrieval_eval else None
    rag_gate_enforced = False
    if args.enforce_rag_gate:
        if retrieval_eval and rag_gate is False:
            summary["overall_status"] = "FAIL"
            summary["rag_gate_forced_fail"] = True
            rag_gate_enforced = True
            print("[RAG] [Gate] FORCE-FAIL batch: retrieval gate FAIL + --enforce-rag-gate")
        elif not retrieval_eval:
            summary["overall_status"] = "FAIL"
            summary["rag_gate_missing_eval"] = True
            rag_gate_enforced = True
            print("[RAG] [Gate] FORCE-FAIL batch: retrieval eval missing + --enforce-rag-gate")
        else:
            print(f"[RAG] [Gate] --enforce-rag-gate OK (gate.passed={rag_gate})")
    else:
        if retrieval_eval and rag_gate is False:
            print("[RAG] [Gate] WARNING: retrieval gate FAIL (--enforce-rag-gate OFF -> no batch flip)")

    summary_path = run_dir / f"regression_summary_{stamp}.json"
    markdown_path = run_dir / f"regression_summary_{stamp}.md"
    html_path = run_dir / f"regression_summary_{stamp}.html"
    latest_summary = run_dir / "latest_regression_summary.json"
    latest_markdown = run_dir / "latest_regression_summary.md"
    latest_html = run_dir / "latest_regression_summary.html"

    baseline_path = None
    if manifest_before.exists():
        baseline_path = manifest_before

    if args.write_baseline:
        baseline_output = Path(args.baseline_output)
        _write_baseline(summary, baseline_output)
        summary["baseline_output"] = str(baseline_output)

    _ensure_parent(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _ensure_parent(markdown_path).write_text(_render_batch_markdown(summary), encoding="utf-8")
    if args.html:
        _ensure_parent(html_path).write_text(_render_batch_html(summary), encoding="utf-8")
        summary["html_report"] = str(html_path)

    if baseline_path and baseline_path.exists():
        compare_output = run_dir / f"compare_{stamp}.json"
        compare_cmd = [
            sys.executable,
            "scripts/compare_pipeline_results.py",
            "--before",
            str(baseline_path),
            "--after",
            str(summary_path),
            "--output",
            str(compare_output),
            "--html",
        ]
        subprocess.run(compare_cmd, check=False)
        summary["compare_report"] = str(compare_output)
    elif baseline_path:
        summary["compare_report"] = None
        summary["compare_warning"] = f"Baseline not found: {baseline_path}"

    _ensure_parent(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _ensure_parent(markdown_path).write_text(_render_batch_markdown(summary), encoding="utf-8")
    if args.html:
        _ensure_parent(html_path).write_text(_render_batch_html(summary), encoding="utf-8")
    _ensure_parent(latest_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _ensure_parent(latest_markdown).write_text(_render_batch_markdown(summary), encoding="utf-8")
    if args.html:
        _ensure_parent(latest_html).write_text(_render_batch_html(summary), encoding="utf-8")

    print(summary_path)
    print(f"PASS={passed_count} FAIL={failed_count}")
    return 0 if summary["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
