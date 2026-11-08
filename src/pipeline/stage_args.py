"""Stage-args builder for pipeline stage invocations (G7, B-7, 2026-08-17).

Extracted from ``MainPipeline._build_stage_args``: building the
``argparse.Namespace`` passed to stage functions is configuration assembly
(a pure-ish transformation over the run context), not orchestration. Kept
separate so stage invocation shape is decoupled from the orchestrator.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def set_stage_args_source_path(args: argparse.Namespace, source_path: str) -> None:
    source_path = Path(source_path)
    if source_path.is_file():
        args.file = str(source_path)
        args.folder = None
    else:
        args.file = None
        args.folder = str(source_path)


def set_stage_args_flags(
    args: argparse.Namespace,
    *,
    fast_mode: bool,
    enable_rigor_extract: bool,
    strict_governance: bool = False,
) -> None:
    args.skip_dedup = False
    args.no_backup = False
    args.strict_governance = strict_governance
    args.max_cross_layer_ratio = 0.8
    args.max_relationship_issues = 100
    args.min_master_entities = 1
    args.strict_preflight = False
    args.json = False
    args.run_dir = ""
    args.fragment = ""
    args.fast_mode = fast_mode
    args.enable_rigor = enable_rigor_extract


def build_stage_args(
    *,
    context: Any,
    run_paths: Any,
    run_id: str,
    raw_context_output_dir: Optional[str],
    project_root: Path,
) -> argparse.Namespace:
    """Build the Namespace used by stage functions (G7).

    ``context`` carries domain/source_path/metadata/regression_mode/fast_mode/
    enable_rigor_extract; ``run_paths`` carries run_root/stage_root/file_root/
    artifact_root. ``project_root`` locates the master ontology path.
    """
    from argparse import Namespace

    args = Namespace()
    args.stage = "full"
    args.domain = context.domain
    args.output_dir = str(run_paths.run_root)
    args.run_root = str(run_paths.run_root)
    args.stage_root = str(run_paths.stage_root)
    args.file_root = str(run_paths.file_root)
    args.artifact_root = str(run_paths.artifact_root)
    metadata = getattr(context, "metadata", {}) or {}
    regression_master_path = metadata.get("regression_master_path") if isinstance(metadata, dict) else getattr(metadata, "regression_master_path", None)
    args.master_path = str(Path(regression_master_path) if regression_master_path else project_root / "outputs" / "runs" / "ontology" / "master" / "master_ontology_unified.json")
    args.run_id = run_id

    if raw_context_output_dir != getattr(context, "output_dir", None):
        logger.info(
            "[DEBUG-PATH] build_stage_args run_id=%s output_dir=%s source=%s",
            run_id,
            args.output_dir,
            getattr(context, "source_path", ""),
        )

    set_stage_args_source_path(args, str(getattr(context, "source_path", "")))

    metadata = getattr(context, "metadata", None)
    strict_governance = bool(getattr(metadata, "strict_governance", False)) if metadata else False
    set_stage_args_flags(
        args,
        fast_mode=bool(getattr(context, "fast_mode", False)),
        enable_rigor_extract=bool(getattr(context, "enable_rigor_extract", False)),
        strict_governance=strict_governance,
    )
    args.enable_llm_resolver = bool(getattr(context, "enable_llm_conflict_resolver", False))

    if getattr(context, "regression_mode", False):
        args.strict_governance = True
        args.no_backup = False
        args.fast_mode = False
    return args