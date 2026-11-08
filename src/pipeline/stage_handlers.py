"""Per-stage handler implementations extracted from MainPipeline (P0-3 batch 18).

All thirteen ``_handle_*`` stage handlers, moved verbatim from MainPipeline
methods using the established ``pipeline``-first-parameter pattern (same as
stage_skills, runflow, status_quality, stage_plumbing). The dispatch table in
``MainPipeline._stage_handler`` intentionally stays on the orchestrator (the
run-flow characterization tests patch ``_stage_handler`` at class level as
the injection seam).

Contents: handle_preflight / handle_health / handle_extract / handle_merge /
handle_dedup / handle_graph_fusion / handle_relationship / handle_audit /
handle_dcs_audit / handle_guard / handle_emergent_reasoning / handle_export /
handle_summarize.

No tests referenced the ``_handle_*`` methods directly (verified repo-wide
2026-09-09); the MainPipeline delegates preserve the bound-method references
used by the dispatch table.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .stages import (
    run_audit,
    run_dcs_audit,
    run_dedup,
    run_document_report,
    run_emergent_reasoning,
    run_export,
    run_extract,
    run_graph_fusion,
    run_guard,
    run_merge,
    run_preflight,
    run_relationship,
    run_summarize,
)

if TYPE_CHECKING:  # pragma: no cover - type hints only, avoids circular import
    from .main_pipeline import MainPipeline
    from .stages import StageResult

logger = logging.getLogger(__name__)


async def handle_preflight(pipeline: "MainPipeline") -> "StageResult":
    """预检查阶段"""
    try:
        from src.utils.file_lock import cleanup_stale_locks
        output_dir = Path(pipeline.context.output_dir)
        cleaned = cleanup_stale_locks(output_dir, stale_timeout=300.0)
        if cleaned > 0:
            logger.info(f"[Preflight] Cleaned {cleaned} stale lock files")
    except ImportError:
        pass

    args = pipeline._build_stage_args()
    return run_preflight(args)


async def handle_health(pipeline: "MainPipeline") -> "StageResult":
    """健康检查阶段"""
    from .stages import run_health

    args = pipeline._build_stage_args()
    return run_health(args)


async def handle_extract(pipeline: "MainPipeline") -> "StageResult":
    """文档抽取阶段"""
    args = pipeline._build_stage_args()
    return await run_extract(args)


async def handle_merge(pipeline: "MainPipeline") -> "StageResult":
    """本体合并阶段"""
    args = pipeline._build_stage_args()
    args.fragment = pipeline._resolve_stage_fragment_arg()
    return run_merge(args)


async def handle_dedup(pipeline: "MainPipeline") -> "StageResult":
    """去重阶段"""
    args = pipeline._build_stage_args()
    return run_dedup(args)


async def handle_graph_fusion(pipeline: "MainPipeline") -> "StageResult":
    """图代数拓扑融合阶段（步骤2）"""
    args = pipeline._build_stage_args()
    return await run_graph_fusion(args)


async def handle_relationship(pipeline: "MainPipeline") -> "StageResult":
    """关系构建阶段"""
    args = pipeline._build_stage_args()
    args.fragment = pipeline._resolve_stage_fragment_arg()
    return await run_relationship(args)


async def handle_audit(pipeline: "MainPipeline") -> "StageResult":
    """审计阶段"""
    args = pipeline._build_stage_args()
    return await run_audit(args)


async def handle_dcs_audit(pipeline: "MainPipeline") -> "StageResult":
    """维度覆盖度审计阶段（步骤3）"""
    args = pipeline._build_stage_args()
    return await run_dcs_audit(args)


async def handle_guard(pipeline: "MainPipeline") -> "StageResult":
    """质量门禁阶段"""
    args = pipeline._build_stage_args()
    return await run_guard(args)


async def handle_emergent_reasoning(pipeline: "MainPipeline") -> "StageResult":
    """涌现推理计算阶段（步骤4）"""
    args = pipeline._build_stage_args()
    return await run_emergent_reasoning(args)


async def handle_export(pipeline: "MainPipeline") -> "StageResult":
    """导出阶段"""
    args = pipeline._build_stage_args()
    return run_export(args)


async def handle_document_report(pipeline: "MainPipeline") -> "StageResult":
    """文档级MD报告生成阶段"""
    args = pipeline._build_stage_args()
    return run_document_report(args)


async def handle_summarize(pipeline: "MainPipeline") -> "StageResult":
    """汇总阶段"""
    args = pipeline._build_stage_args()
    return run_summarize(args)
