from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineRunPaths:
    run_root: Path
    stage_root: Path
    file_root: Path
    artifact_root: Path

    def file_root_for_index(self, file_index: int) -> Path:
        return self.stage_root / "files" / f"{file_index:04d}"

    @property
    def regression_root(self) -> Path:
        return self.artifact_root / "regression"

    @property
    def audit_root(self) -> Path:
        return self.artifact_root / "audit"


def _contains_repeated_run_id(path: Path, run_id: str) -> bool:
    parts = [part for part in path.parts if part == run_id]
    return len(parts) > 1


def _contains_files_zero_zero_zero_zero(path: Path) -> bool:
    parts = list(path.parts)
    return parts.count("files") > 1 or parts.count("0000") > 1


def build_pipeline_run_paths(output_dir: str | Path, run_id: str, stage_name: str = "extract", file_index: int = 0) -> PipelineRunPaths:
    base = Path(output_dir)
    if _contains_repeated_run_id(base, run_id) or _contains_files_zero_zero_zero_zero(base):
        raise ValueError(f"Refusing to derive nested pipeline paths from unstable output_dir: {base}")
    run_root = base / run_id if base.name != run_id else base
    stage_root = run_root / stage_name
    file_root = stage_root / "files" / f"{file_index:04d}"
    artifact_root = run_root / "artifacts"
    return PipelineRunPaths(
        run_root=run_root,
        stage_root=stage_root,
        file_root=file_root,
        artifact_root=artifact_root,
    )
