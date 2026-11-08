"""Main Pipeline - 主线生产管道

按照架构审计报告和重构方案，这是默认生产路径：
- discover/scan → preflight → extract → normalize → validate → dedup → relationship/merge → audit/quality_gate → export/publish

核心设计原则：
1. 默认路径唯一：生产默认入口只有一个
2. 增强能力可插拔：质量增强、严格检查、实验对比都必须显式启用
3. 职责严格分离：流程、规则、结构契约、能力定义分开治理
4. 输出前统一收口：最终产物在导出/发布前统一过 schema、quality、governance

边界（Phase 3）：
- pipeline 只负责编排与产物生成；知识消费由 `src.application.rag` 承担。
- 本模块不直接导入 RAG 编排模块，只通过资产包/检索产物服务消费 RAG 侧产物。

契约校验链集成（CONTRACT_GOVERNANCE_SPEC.md 第6节）：
- 配置加载 → 契约识别 → schema 验证 → skill 绑定 → pipeline 执行 → 输出校验 → 质量门禁

参考文档：
- PROJECT_ARCHITECTURE_AUDIT_REPORT_20260708.md
- PIPELINE_REFACTOR_ACTION_PLAN.md
- CONTRACT_GOVERNANCE_SPEC.md
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

from src.domain.models.core import (
    ArtifactEnvelope as CoreArtifactEnvelope,
    PipelineExecutionResult as CorePipelineExecutionResult,
    TraceContext,
    ValidationReport,
)

from .stages import (
    StageResult,
)
from .stages.common import (
    normalize_entity,
    resolve_stage_fragment,
)
from . import governance_arming
from . import status_quality
from . import runflow
from . import pipeline_factory
from . import continuity_artifacts
from . import skill_bootstrap
from . import pipeline_init
from . import stage_handlers
from . import stage_skills
from . import stage_plumbing
from . import trace_provenance
from . import stage_result_builders

from src.infrastructure.config.contract_validation import (
    get_validation_chain,
)
from src.pipeline.skill_integration import (
    get_skill_pipeline_bridge,
)
from src.pipeline.pipeline_skill_router import (
    get_pipeline_skill_router,
)
from skills.skill_contract import (
    UnifiedSkillContract,
)
from ontology_compiler.services.data_continuity import (
    get_continuity_service,
)
from ontology_compiler.services.asset_packet_service import (
    get_asset_packet_service,
)
from ontology_compiler.services.asset_index_service import (
    get_asset_index_service,
)
from src.infrastructure.services.retrieval_asset_packet_service import (
    get_retrieval_asset_packet_service,
)
from src.infrastructure.services.pipeline_result_artifact_service import (
    get_pipeline_result_artifact_service,
)

logger = logging.getLogger(__name__)


# Phase 1 LOC reduction (docs/CODE_CLEANUP_CANDIDATES.md §3.1): pure
# result-payload helpers live in result_helpers.py; MainPipeline keeps thin
# delegating methods so tests and callers stay stable.
from .result_helpers import (  # noqa: F401  (re-exported for compatibility)
    _config_fingerprint,
    build_contract_artifacts as _build_contract_artifacts_impl,
    build_resolved_config_snapshot as _build_resolved_config_snapshot_impl,
    build_validation_report,
    build_manifest_payload,
    save_result_bundle,
    build_core_pipeline_result,
    build_artifact_envelope,
)


from .pipeline_models import (
    PipelineContext,
    PipelineExecutionMode,
    PipelineExecutionResult,
    PipelineQualityStatus,
    PipelineStage,
    PipelineStats,
    PipelineStatus,
)


class MainPipeline:
    """主线生产管道 - 标准、稳定、默认的生产任务执行路径

    遵循架构审计报告建议的主线阶段：
    discover/scan → preflight → extract → normalize → validate → dedup → relationship/merge → audit/quality_gate → export/publish

    技能集成：
    - SkillPipelineBridge: 统一管理 skill 和 pipeline 的交互
    - SkillContractValidator: 在各阶段执行契约验证
    """

    @staticmethod
    def _normalize_output_dir(output_dir: str, run_id: str) -> str:
        # P0-3 batch 17: see pipeline_init.normalize_output_dir.
        return pipeline_init.normalize_output_dir(output_dir, run_id)

    def __init__(self, context: PipelineContext):
        # P0-3 batch 17 (2026-09-09): construction state moved verbatim to
        # pipeline_init (init_context_state / init_run_state / init_services /
        # init_execution_state); order preserved exactly.
        pipeline_init.init_context_state(self, context)
        pipeline_init.init_run_state(self)
        pipeline_init.init_services(self)
        pipeline_init.init_execution_state(self)

    @staticmethod
    def _generate_run_id() -> str:
        # P0-3 batch 17: see pipeline_init.generate_run_id.
        return pipeline_init.generate_run_id()

    @staticmethod
    def _compute_file_hash(source_path: str) -> str:
        # P0-3 batch 17: see pipeline_init.compute_file_hash.
        return pipeline_init.compute_file_hash(source_path)

    def _determine_execution_mode(self) -> PipelineExecutionMode:
        # P0-3 batch 17: see pipeline_init.determine_execution_mode.
        return pipeline_init.determine_execution_mode(self)

    def _set_execution_mode(self, mode: PipelineExecutionMode, reason: str = "") -> None:
        # P0-3 batch 17: instance-level seam (guard-failure tests patch with
        # Mock); see pipeline_init.set_execution_mode.
        pipeline_init.set_execution_mode(self, mode, reason)

    def _arm_governance_from_context(self) -> None:
        # P0-3 batch 8 (2026-09-09): body moved verbatim to
        # governance_arming.arm_governance_from_context
        # (pipeline-first-parameter pattern); this delegate keeps the
        # __init__ call site stable.
        return governance_arming.arm_governance_from_context(self)

    async def run(self, stages: Optional[List[str]] = None) -> PipelineExecutionResult:
        """执行主线 pipeline

        契约校验链（CONTRACT_GOVERNANCE_SPEC.md 第6节）：
        配置加载 → 契约识别 → schema 验证 → skill 绑定 → pipeline 执行 → 输出校验 → 质量门禁
        """
        start_time = time.time()
        self._log_pipeline_start()
        self._stats.total_stages = len(self._stage_order)

        self._initialize_skill_integration()
        self._begin_run_continuity()

        if not self._ensure_contract_validation():
            result = self._build_failed_contract_result()
            self._end_run_continuity("FAILED")
            self._log_summary(result)
            return result

        stage_enums = self._resolve_stage_enums(stages)
        await self._run_pipeline_stages(stage_enums)
        self._collect_classification_metrics()

        return self._finalize_pipeline(start_time)

    def _finalize_pipeline(self, start_time: float) -> PipelineExecutionResult:
        self._stats.total_latency_ms = (time.time() - start_time) * 1000
        self._run_post_execution_validation()

        status = self._derive_pipeline_status()
        execution_reason = self._derive_execution_reason()
        quality_status, quality_reason, validation_passed, quality_score = self._derive_quality_status(status)
        result = self._build_pipeline_result(status, execution_reason, quality_status, quality_reason, validation_passed, quality_score)

        if isinstance(self.context.metadata, dict):
            self.context.metadata.setdefault("quality_gate", {})
            self.context.metadata["quality_gate"].update({
                "quality_status": quality_status.value,
                "quality_reason": quality_reason,
                "validation_passed": validation_passed,
                "quality_score": quality_score,
            })

        self._end_run_continuity(status.value)
        # NOTE: keep calling through the instance method — the characterization
        # tests patch `pipeline._generate_asset_packet` as an instance-level
        # seam; a direct module call here would bypass the mock.
        self._generate_asset_packet(result)
        self._log_summary(result)
        return result

    def _log_pipeline_start(self) -> None:
        # P0-3 batch 12 (2026-09-09): see runflow.log_pipeline_start.
        runflow.log_pipeline_start(self)

    def _ensure_contract_validation(self) -> bool:
        # P0-3 batch 12: pre-run contract gate; characterization tests patch
        # _run_contract_validation_chain at instance level (seam preserved).
        # See runflow.ensure_contract_validation.
        return runflow.ensure_contract_validation(self)

    def _build_failed_contract_result(self) -> PipelineExecutionResult:
        # P0-3 batch 12: see runflow.build_failed_contract_result.
        return runflow.build_failed_contract_result(self)

    def _resolve_stage_enums(self, stages: Optional[List[str]]) -> List[PipelineStage]:
        # P0-3 batch 12: see runflow.resolve_stage_enums.
        return runflow.resolve_stage_enums(self, stages)

    async def _run_pipeline_stages(self, stage_enums: List[PipelineStage]) -> None:
        """C-4 StageDecision 驱动的阶段编排循环（实现见 stage_execution_core）。"""
        from .stage_execution_core import run_pipeline_stages as _run

        await _run(self, stage_enums)
    # ------------------------------------------------------------------
    # C-4 StageDecision helpers
    # ------------------------------------------------------------------
    def _resolve_actual_stage_decision(
        self,
        stage: PipelineStage,
        *,
        upstream_changed: bool = True,
    ) -> "StageDecisionResult":
        """计算权威 C-4 StageDecision（实现见 stage_execution_core）。"""
        from .stage_execution_core import resolve_actual_stage_decision as _resolve

        return _resolve(self, stage, upstream_changed=upstream_changed)

    # P0-3 batch 5 (2026-09-09): cross-run C-4 helpers
    # (_collect_sibling_artifact_roots / _find_prev_run_bundle /
    # _load_stage_result_from_bundle) moved verbatim to stage_execution_core;
    # stage_execution_core now calls the module functions directly. No tests
    # referenced these three methods (verified repo-wide 2026-09-09).

    def _compute_stage_decision_trace(self, stage: PipelineStage) -> Optional[Dict[str, Any]]:
        """Record the C-4 StageDecision into the manifest trace."""
        from .stage_execution_core import compute_stage_decision_trace as _trace

        return _trace(self, stage)
    def _make_skipped_stage_result(
        self,
        stage: PipelineStage,
        *,
        reason: str,
        decision: "StageDecision",
    ) -> "StageResult":
        """P1-3 (2026-10-16): 实现下沉至 stage_result_builders（纯函数）。"""
        return stage_result_builders.make_skipped_stage_result(
            stage, reason=reason, decision=decision,
            run_id=self._run_id, warnings=self._warnings,
        )

    def _make_degraded_stage_result(
        self, stage: PipelineStage, *, reason: str
    ) -> "StageResult":
        """P1-3 (2026-10-16): 实现下沉至 stage_result_builders（纯函数）。"""
        return stage_result_builders.make_degraded_stage_result(
            stage, reason=reason, run_id=self._run_id, warnings=self._warnings,
        )

    def _should_skip_stage(self, stage: PipelineStage) -> bool:
        """P1-3 (2026-10-16): 判定逻辑下沉至 stage_result_builders。"""
        reason = stage_result_builders.skip_reason_for_mode(self._execution_mode, stage)
        if reason:
            logger.info(f"⏭️  {reason}")
            self._warnings.append(reason)
            return True
        return False

    async def _execute_pipeline_stage(self, stage: PipelineStage) -> StageResult:
        return await self._execute_stage(stage)

    async def _process_stage_result(self, stage: PipelineStage, stage_result: StageResult) -> bool:
        """处理阶段结果；返回 True 表示应停止后续阶段执行"""
        if stage_result.success:
            await self._handle_successful_stage(stage, stage_result)
            return False
        return self._handle_failed_stage(stage, stage_result)

    async def _handle_successful_stage(self, stage: PipelineStage, stage_result: StageResult) -> None:
        self._stats.completed_stages += 1
        self._collect_stats(stage_result)
        self._collect_output_files(stage_result)
        await self._apply_stage_skills(stage, stage_result)
        self._validate_stage_output(stage, stage_result)
        self._register_stage_artifacts(stage, stage_result)

    def _handle_failed_stage(self, stage: PipelineStage, stage_result: StageResult) -> bool:
        """处理失败阶段；返回 True 表示严重失败应停止 pipeline"""
        self._stats.failed_stages += 1
        stage_errors = stage_result.data.get("errors", []) if stage_result.data else []
        self._errors.extend(stage_errors)

        if stage == PipelineStage.PREFLIGHT:
            logger.error(f"🔴 Critical stage {stage.value} failed, stopping pipeline")
            return True

        if stage == PipelineStage.GUARD and self._should_continue_guard_after_failure(stage_result):
            self._set_execution_mode(PipelineExecutionMode.OFFLINE, "guard_returned_degraded_empty_output")
            logger.warning("🟡 Guard returned degraded/empty output; continuing pipeline for offline validation")
            return False

        if stage == PipelineStage.GUARD:
            logger.error(f"🔴 Critical stage {stage.value} failed, stopping pipeline")
            return True

        return False

    def _should_continue_guard_after_failure(self, stage_result: StageResult) -> bool:
        stage_data = stage_result.data or {}
        quality_level = str(stage_data.get("quality_level", "")).upper()
        has_structure = bool(stage_data.get("nodes") or stage_data.get("instances") or stage_data.get("relationships") or stage_data.get("fragment_path"))
        return quality_level == "DEGRADED" and not has_structure

    def _collect_classification_metrics(self) -> None:
        # P0-3 batch 12: B2 use-case wiring; see runflow.collect_classification_metrics.
        runflow.collect_classification_metrics(self)

    def _store_classification_metrics(self, classification_summary) -> None:
        # P0-3 batch 12: see runflow.store_classification_metrics.
        runflow.store_classification_metrics(self, classification_summary)

    def _derive_pipeline_status(self) -> PipelineStatus:
        # P0-3 batch 11 (2026-09-09): see status_quality.derive_pipeline_status.
        return status_quality.derive_pipeline_status(self)

    def _extract_stage_quality(self) -> dict[str, Any]:
        # P0-3 batch 11: diagnostics seam (characterization tests call
        # directly); see status_quality.extract_stage_quality.
        return status_quality.extract_stage_quality(self)

    def _derive_quality_status(self, pipeline_status: PipelineStatus) -> tuple[PipelineQualityStatus, str, bool, float]:
        # P0-3 batch 11: A-2 gate-only derivation; characterization tests call
        # directly; see status_quality.derive_quality_status.
        return status_quality.derive_quality_status(self, pipeline_status)

    def _derive_execution_reason(self) -> str:
        # P0-3 batch 11: see status_quality.derive_execution_reason.
        return status_quality.derive_execution_reason(self)

    def _build_pipeline_result(
        self,
        status: PipelineStatus,
        execution_reason: str,
        quality_status: PipelineQualityStatus,
        quality_reason: str,
        validation_passed: bool,
        quality_score: float,
    ) -> PipelineExecutionResult:
        # P0-3 batch 11: see status_quality.build_pipeline_result.
        return status_quality.build_pipeline_result(
            self, status, execution_reason, quality_status, quality_reason, validation_passed, quality_score
        )

    def _run_contract_validation_chain(self) -> bool:
        """运行契约校验链（CONTRACT_GOVERNANCE_SPEC.md 第6节）"""
        # B3 (2026-08-15): chain execution + report chip assembly migrated to
        # the pipeline_validation use case; this method wires chips into state.
        try:
            from src.application.use_cases.pipeline_validation_use_case import (
                run_contract_validation,
            )

            outcome = run_contract_validation(
                get_validation_chain(),
                source_path=self.context.source_path,
                domain=self.context.domain,
                run_id=self._run_id,
            )
            self._errors.extend(outcome.error_chips)
            self._warnings.extend(outcome.warning_chips)
            return outcome.valid
        except Exception as e:
            logger.error(f"Failed to run contract validation chain: {e}")
            self._errors.append(f"Contract validation error: {e}")
            return False

    def _initialize_skill_integration(self):
        # P0-3 batch 16 (2026-09-09): see skill_bootstrap.initialize_skill_integration.
        skill_bootstrap.initialize_skill_integration(self)

    def _build_default_skill_contract(self) -> UnifiedSkillContract:
        # P0-3 batch 9 (2026-09-09): see stage_skills.build_default_skill_contract.
        return stage_skills.build_default_skill_contract(self)

    async def _apply_stage_skills(self, stage: PipelineStage, stage_result: StageResult):
        # P0-3 batch 9: instance-level seam (characterization tests patch with
        # AsyncMock); see stage_skills.apply_stage_skills.
        return await stage_skills.apply_stage_skills(self, stage, stage_result)

    def _get_skills_for_stage(self, stage: PipelineStage) -> List[str]:
        # P0-3 batch 9: see stage_skills.get_skills_for_stage.
        return stage_skills.get_skills_for_stage(self, stage)

    def _merge_skill_result(self, stage_result: StageResult, skill_result):
        # P0-3 batch 9: see stage_skills.merge_skill_result.
        return stage_skills.merge_skill_result(self, stage_result, skill_result)

    def _validate_stage_output(self, stage: PipelineStage, stage_result: StageResult):
        # P0-3 batch 9: instance-level seam (characterization tests patch with
        # Mock); see stage_skills.validate_stage_output.
        return stage_skills.validate_stage_output(self, stage, stage_result)

    def _collect_validation_entities(self, stage_data: Dict[str, Any]) -> Dict[str, Any]:
        # P0-3 batch 9: see stage_skills.collect_validation_entities.
        return stage_skills.collect_validation_entities(self, stage_data)

    def _collect_validation_relations(self, stage_data: Dict[str, Any]) -> list[Dict[str, Any]]:
        # P0-3 batch 9: see stage_skills.collect_validation_relations.
        return stage_skills.collect_validation_relations(self, stage_data)

    def _record_stage_validation_violations(self, stage: PipelineStage, violations: list[Dict[str, Any]]) -> None:
        # P0-3 batch 9: see stage_skills.record_stage_validation_violations.
        return stage_skills.record_stage_validation_violations(self, stage, violations)

    def _run_post_execution_validation(self):
        # P0-3 batch 16: instance-level seam (run-flow tests patch with
        # Mock); see skill_bootstrap.run_post_execution_validation.
        skill_bootstrap.run_post_execution_validation(self)

    def _validate_pipeline_output(self, output_data: Dict[str, Any]) -> None:
        # P0-3 batch 16: B3 use-case wiring; see skill_bootstrap.validate_pipeline_output.
        skill_bootstrap.validate_pipeline_output(self, output_data)

    def _run_quality_gate(self, output_data: Dict[str, Any]) -> None:
        # P0-3 batch 16: B1 use-case wiring; see skill_bootstrap.run_quality_gate.
        skill_bootstrap.run_quality_gate(self, output_data)

    def _collect_pipeline_output(self) -> Optional[Dict[str, Any]]:
        """收集 pipeline 输出用于验证（实现见 stage_execution_core）。"""
        from .stage_execution_core import collect_pipeline_output as _collect

        return _collect(self)

    async def _execute_stage(self, stage: PipelineStage) -> StageResult:
        """执行单个阶段"""
        logger.info(f"\n▶️  Executing stage: {stage.value}")
        start_time = time.time()

        try:
            result = await self._stage_handler(stage)
            latency_ms = (time.time() - start_time) * 1000

            if result.success:
                logger.info(f"✅ Stage {stage.value} completed ({latency_ms:.1f}ms)")
            else:
                logger.error(f"❌ Stage {stage.value} failed ({latency_ms:.1f}ms)")
                stage_errors = result.data.get("errors", []) if result.data else []
                if stage_errors:
                    for error in stage_errors[:5]:
                        logger.error(f"   - {error}")

            return result

        except Exception as e:
            elapsed_seconds = time.time() - start_time
            logger.error(f"💥 Stage {stage.value} exception: {e} ({elapsed_seconds*1000:.1f}ms)")
            return StageResult(
                stage=stage.value,
                success=False,
                message=str(e),
                data={"errors": [str(e)]},
                elapsed_seconds=elapsed_seconds,
            )

    async def _stage_handler(self, stage: PipelineStage) -> StageResult:
        """阶段处理器分发（测试经类级 patch 此方法注入）"""
        handlers = {
            PipelineStage.PREFLIGHT: self._handle_preflight,
            PipelineStage.EXTRACT: self._handle_extract,
            PipelineStage.MERGE: self._handle_merge,
            PipelineStage.DEDUP: self._handle_dedup,
            PipelineStage.GRAPH_FUSION: self._handle_graph_fusion,
            PipelineStage.RELATIONSHIP: self._handle_relationship,
            PipelineStage.AUDIT: self._handle_audit,
            PipelineStage.DCS_AUDIT: self._handle_dcs_audit,
            PipelineStage.GUARD: self._handle_guard,
            PipelineStage.EMERGENT_REASONING: self._handle_emergent_reasoning,
            PipelineStage.EXPORT: self._handle_export,
            PipelineStage.DOCUMENT_REPORT: self._handle_document_report,
            PipelineStage.SUMMARIZE: self._handle_summarize,
        }

        handler = handlers.get(stage)
        if not handler:
            raise ValueError(f"Unknown stage: {stage}")

        return await handler()

    def _normalize_artifact_path(self, artifact_path: str) -> str:
        # P0-3 batch 7 (2026-09-09): see stage_plumbing.normalize_artifact_path.
        return stage_plumbing.normalize_artifact_path(self, artifact_path)

    def _sanitize_stage_result_paths(self, stage_result: StageResult) -> StageResult:
        # P0-3 batch 7: see stage_plumbing.sanitize_stage_result.
        return stage_plumbing.sanitize_stage_result(self, stage_result)

    def _normalize_stage_result_paths(self, stage_result: StageResult) -> None:
        # P0-3 batch 7: see stage_plumbing.normalize_stage_result_paths.
        stage_plumbing.normalize_stage_result_paths(self, stage_result)

    def _build_entity_layer_index(self, stage_result: StageResult) -> dict[str, str]:
        # P0-3 batch 7: see stage_plumbing.build_entity_layer_index.
        return stage_plumbing.build_entity_layer_index(self, stage_result)

    def _rewrite_stage_relationships(self, stage_result: StageResult) -> None:
        # P0-3 batch 7: see stage_plumbing.rewrite_stage_relationships.
        stage_plumbing.rewrite_stage_relationships(self, stage_result)

    def _normalize_stage_entities(self, stage_result: StageResult) -> None:
        # P0-3 batch 7: see stage_plumbing.normalize_stage_entities.
        stage_plumbing.normalize_stage_entities(self, stage_result)

    def _normalize_entity_record(self, entity: Any) -> Any:
        # P0-3 batch 7: see stage_plumbing.normalize_entity_record.
        return stage_plumbing.normalize_entity_record(self, entity)

    def _infer_entity_layer(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        # P0-3 batch 7: see stage_plumbing.infer_entity_layer.
        return stage_plumbing.infer_entity_layer(self, entity)

    def _build_entity_profile_summary(self, entities: list[dict[str, Any]]) -> dict[str, Any]:
        # P0-3 batch 7: see stage_plumbing.build_entity_profile_summary.
        return stage_plumbing.build_entity_profile_summary(self, entities)

    def _project_structured_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        # P0-3 batch 7: see stage_plumbing.project_structured_metadata.
        return stage_plumbing.project_structured_metadata(self, metadata)

    def _is_layer_pair_blocked(self, rel: Dict[str, Any]) -> bool:
        # P0-3 batch 7: see stage_plumbing.is_layer_pair_blocked.
        return stage_plumbing.is_layer_pair_blocked(self, rel)

    def _lookup_entity_layer(self, entity_id: str) -> str:
        # P0-3 batch 7: see stage_plumbing.lookup_entity_layer.
        return stage_plumbing.lookup_entity_layer(self, entity_id)

    def _rewrite_related_to(self, rel: Dict[str, Any]) -> Dict[str, Any]:
        # P0-3 batch 7: see stage_plumbing.rewrite_related_to.
        return stage_plumbing.rewrite_related_to(self, rel)

    def _build_stage_args(self) -> argparse.Namespace:
        # P0-3 batch 7: see stage_plumbing.build_stage_args.
        return stage_plumbing.build_stage_args(self)

    async def _handle_preflight(self) -> StageResult:
        # P0-3 batch 18 (2026-09-09): see stage_handlers.handle_preflight.
        return await stage_handlers.handle_preflight(self)

    async def _handle_health(self) -> StageResult:
        # P0-3 batch 18: see stage_handlers.handle_health.
        return await stage_handlers.handle_health(self)

    async def _handle_extract(self) -> StageResult:
        # P0-3 batch 18: see stage_handlers.handle_extract.
        return await stage_handlers.handle_extract(self)

    def _resolve_stage_fragment_arg(self) -> str:
        # P0-3 batch 7: see stage_plumbing.resolve_stage_fragment_arg.
        return stage_plumbing.resolve_stage_fragment_arg(self)

    async def _handle_merge(self) -> StageResult:
        # P0-3 batch 18: see stage_handlers.handle_merge.
        return await stage_handlers.handle_merge(self)

    async def _handle_dedup(self) -> StageResult:
        # P0-3 batch 18: see stage_handlers.handle_dedup.
        return await stage_handlers.handle_dedup(self)

    async def _handle_graph_fusion(self) -> StageResult:
        # P0-3 batch 18: see stage_handlers.handle_graph_fusion.
        return await stage_handlers.handle_graph_fusion(self)

    async def _handle_relationship(self) -> StageResult:
        # P0-3 batch 18: see stage_handlers.handle_relationship.
        return await stage_handlers.handle_relationship(self)

    async def _handle_audit(self) -> StageResult:
        # P0-3 batch 18: see stage_handlers.handle_audit.
        return await stage_handlers.handle_audit(self)

    async def _handle_dcs_audit(self) -> StageResult:
        # P0-3 batch 18: see stage_handlers.handle_dcs_audit.
        return await stage_handlers.handle_dcs_audit(self)

    async def _handle_guard(self) -> StageResult:
        # P0-3 batch 18: see stage_handlers.handle_guard.
        return await stage_handlers.handle_guard(self)

    async def _handle_emergent_reasoning(self) -> StageResult:
        # P0-3 batch 18: see stage_handlers.handle_emergent_reasoning.
        return await stage_handlers.handle_emergent_reasoning(self)

    async def _handle_export(self) -> StageResult:
        # P0-3 batch 18: see stage_handlers.handle_export.
        return await stage_handlers.handle_export(self)

    async def _handle_document_report(self) -> StageResult:
        return await stage_handlers.handle_document_report(self)

    async def _handle_summarize(self) -> StageResult:
        # P0-3 batch 18: see stage_handlers.handle_summarize.
        return await stage_handlers.handle_summarize(self)

    def _collect_stats(self, result: StageResult):
        # P0-3 batch 7: see stage_plumbing.collect_stats.
        stage_plumbing.collect_stats(self, result)

    def _collect_output_files(self, result: StageResult):
        # P0-3 batch 7: see stage_plumbing.collect_output_files.
        stage_plumbing.collect_output_files(self, result)

    def _log_summary(self, result: PipelineExecutionResult):
        # P0-3 batch 7: see stage_plumbing.log_summary.
        stage_plumbing.log_summary(self, result)

    def _build_trace_context(self, stage_id: str = "", task_id: str = "", artifact_id: str = "") -> TraceContext:
        # P0-3 batch 6 (2026-09-09): assembly moved verbatim to
        # trace_provenance.build_trace_context (pipeline-first-parameter
        # pattern); this delegate keeps the call sites stable.
        return trace_provenance.build_trace_context(self, stage_id, task_id, artifact_id)

    def _build_provenance(self) -> Dict[str, Any]:
        # P0-3 batch 6: see _build_trace_context.
        return trace_provenance.build_provenance(self)

    def _build_run_metadata(self) -> Dict[str, Any]:
        # P0-3 batch 6: see _build_trace_context.
        return trace_provenance.build_run_metadata(self)

    def _schema_version(self) -> str:
        # P0-3 batch 6: see _build_trace_context.
        return trace_provenance.schema_version(self)

    def _skill_version(self) -> str:
        # P0-3 batch 6: see _build_trace_context.
        return trace_provenance.skill_version(self)

    def _prompt_version(self) -> str:
        # P0-3 batch 6: see _build_trace_context.
        return trace_provenance.prompt_version(self)

    def _build_core_pipeline_result(self, result: PipelineExecutionResult) -> CorePipelineExecutionResult:
        # P0-3 batch 10 (2026-09-09): see result_helpers.build_core_pipeline_result.
        return build_core_pipeline_result(self, result)

    def _build_validation_report(self, result: PipelineExecutionResult) -> ValidationReport:
        """P0-3 batch 3 (2026-09-09): assembly moved verbatim to
        result_helpers.build_validation_report (pipeline-first-parameter
        pattern); this delegate keeps the call sites and any instance-level
        patches stable."""
        return build_validation_report(self, result)

    def _build_artifact_envelope(self, result: PipelineExecutionResult) -> CoreArtifactEnvelope:
        # P0-3 batch 10 (2026-09-09): see result_helpers.build_artifact_envelope.
        # Chains through self._build_core_pipeline_result so the bridge
        # characterization seam (direct calls) keeps working.
        core_result = self._build_core_pipeline_result(result)
        return build_artifact_envelope(self, result, core_result)

    def _begin_run_continuity(self):
        # P0-3 batch 15 (2026-09-09): instance-level seam (run-flow tests
        # patch with Mock); see continuity_artifacts.begin_run_continuity.
        continuity_artifacts.begin_run_continuity(self)

    def _load_regression_baseline_if_needed(self) -> None:
        # P0-3 batch 15: see continuity_artifacts.load_regression_baseline_if_needed.
        continuity_artifacts.load_regression_baseline_if_needed(self)

    def _load_regression_baseline(self) -> Dict[str, Any]:
        # P0-3 batch 15: see continuity_artifacts.load_regression_baseline.
        return continuity_artifacts.load_regression_baseline(self)

    def _end_run_continuity(self, status: str):
        # P0-3 batch 15: instance-level seam (run-flow tests patch with
        # Mock); see continuity_artifacts.end_run_continuity.
        continuity_artifacts.end_run_continuity(self, status)

    def _filter_unprocessed_files(self):
        # P0-3 batch 15: see continuity_artifacts.filter_unprocessed_files.
        continuity_artifacts.filter_unprocessed_files(self)

    def _filter_single_processed_file(self, source_path: Path) -> None:
        # P0-3 batch 15: see continuity_artifacts.filter_single_processed_file.
        continuity_artifacts.filter_single_processed_file(self, source_path)

    def _filter_processed_directory_files(self, source_path: Path) -> None:
        # P0-3 batch 15: see continuity_artifacts.filter_processed_directory_files.
        continuity_artifacts.filter_processed_directory_files(self, source_path)

    def _register_stage_artifacts(self, stage: PipelineStage, stage_result: StageResult):
        # P0-3 batch 15: instance-level seam (run-flow tests patch with
        # Mock); see continuity_artifacts.register_stage_artifacts.
        continuity_artifacts.register_stage_artifacts(self, stage, stage_result)

    def _register_stage_bundle(self, stage: PipelineStage, stage_result: StageResult, stage_data: Dict[str, Any], output_files: List[str]) -> str:
        # P0-3 batch 15: see continuity_artifacts.register_stage_bundle.
        return continuity_artifacts.register_stage_bundle(self, stage, stage_result, stage_data, output_files)

    def _record_stage_artifact_outputs(self, stage: PipelineStage, stage_result: StageResult, stage_data: Dict[str, Any], output_files: List[str]) -> None:
        # P0-3 batch 15: see continuity_artifacts.record_stage_artifact_outputs.
        continuity_artifacts.record_stage_artifact_outputs(self, stage, stage_result, stage_data, output_files)

    def _mark_source_file_processed_if_needed(self, stage_result: StageResult) -> None:
        # P0-3 batch 15: see continuity_artifacts.mark_source_file_processed_if_needed.
        continuity_artifacts.mark_source_file_processed_if_needed(self, stage_result)

    def save_result(self, result: PipelineExecutionResult, output_dir: Optional[str] = None):
        """保存兼容结果和标准化契约结果。

        P0-3 batch 4 (2026-09-09): the orchestration moved verbatim to
        result_helpers.save_result_bundle (pipeline-first-parameter pattern);
        this delegate keeps the public entry stable for run_main_pipeline.
        """
        return save_result_bundle(self, result, output_dir)

    def _build_contract_artifacts(self, save_dir: str, validation_report: ValidationReport) -> Dict[str, str]:
        return _build_contract_artifacts_impl(
            save_dir, validation_report.model_dump(exclude_none=True)
        )

    def _build_resolved_config_snapshot(self) -> Dict[str, Any]:
        """Build a resolved-config snapshot for run replayability (A-3)."""
        return _build_resolved_config_snapshot_impl(self.context)

    def _build_manifest_payload(self, result: PipelineExecutionResult, result_dict: Dict[str, Any], contract_artifacts: Dict[str, str]) -> Dict[str, Any]:
        """P0-3 batch 4 (2026-09-09): assembly moved verbatim to
        result_helpers.build_manifest_payload; this delegate keeps the
        characterization-test seam stable."""
        return build_manifest_payload(self, result, result_dict, contract_artifacts)

    def _generate_asset_packet(self, result: PipelineExecutionResult):
        """生成标准资产包（Phase C - Asset Packaging）——委托给 asset_packet 模块。

        逻辑已移出 MainPipeline（Phase 3 / P0-3 batch 2 2026-09-09）：去重与
        证据提取在 asset_packet.py 内直接使用 result_helpers 实现（MainPipeline
        的 _deduplicate_* / _extract_evidence_* 薄委托已删除）；本委托方法由
        tests/pipeline/test_asset_packet_characterization.py 锁定，保留为
        instance-level seam（run 流程经 self 调用，测试在实例上 mock）。
        """
        from .asset_packet import generate_asset_packet

        generate_asset_packet(self, result)


def _auto_detect_domain(source_path: str) -> str:
    # P0-3 batch 13 (2026-09-09): see pipeline_factory.auto_detect_domain.
    return pipeline_factory.auto_detect_domain(source_path)


def create_main_pipeline(
    source_path: str,
    domain: str = "AUTO",
    **kwargs,
) -> MainPipeline:
    # P0-3 batch 13: re-export alias (router + src.pipeline.__init__ import
    # by name); implementation lives in pipeline_factory.
    return pipeline_factory.create_main_pipeline(source_path, domain, **kwargs)


async def run_main_pipeline(
    source_path: str,
    domain: str = "SUPPLY_CHAIN",
    **kwargs,
) -> PipelineExecutionResult:
    # P0-3 batch 13: re-export alias; see pipeline_factory.run_main_pipeline.
    return await pipeline_factory.run_main_pipeline(source_path, domain, **kwargs)
