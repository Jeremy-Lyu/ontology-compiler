"""Filesystem adapter implementing `ArtifactStorePort`."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from src.application.ports.artifact_ports import (
    ArtifactStorePort,
    PipelineSummaryPaths,
    StoredArtifact,
    StoredArtifactSet,
)


class OutputsArtifactStore(ArtifactStorePort):
    """Persist artifacts under canonical ontology output roots."""

    def __init__(
        self,
        outputs_root: str | Path = "outputs/ontology/runs",
        pipeline_root: str | Path = "outputs/ontology/pipeline",
        legacy_root: str | Path = "outputs/ontology",
    ):
        self._outputs_root = Path(outputs_root)
        self._pipeline_root = Path(pipeline_root)
        self._legacy_root = Path(legacy_root)

    def write_run_artifact_set(
        self,
        *,
        run_id: str,
        manifest: dict[str, Any],
        artifacts: dict[str, Any],
    ) -> StoredArtifactSet:
        run_dir = self._outputs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = run_dir / "analysis_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        stored_artifacts = self.write_artifacts(run_id=run_id, artifacts=artifacts)

        return StoredArtifactSet(
            run_dir=str(run_dir),
            manifest_path=str(manifest_path),
            artifacts=stored_artifacts,
        )

    def write_artifacts(
        self,
        *,
        run_id: str,
        artifacts: dict[str, Any],
    ) -> list[StoredArtifact]:
        run_dir = self._outputs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        stored_artifacts: list[StoredArtifact] = []
        for name, payload in artifacts.items():
            artifact_path = run_dir / name
            self._write_payload(artifact_path, payload)
            stored_artifacts.append(StoredArtifact(name=name, path=str(artifact_path), payload=payload))
        return stored_artifacts

    def build_pipeline_summary_paths(self, *, run_id: str) -> PipelineSummaryPaths:
        summary_path = self._outputs_root / run_id / "PIPELINE_RUN_SUMMARY.json"
        canonical_latest_path = self._pipeline_root / "PIPELINE_RUN_SUMMARY_LATEST.json"
        legacy_latest_path = self._legacy_root / "PIPELINE_RUN_SUMMARY_LATEST.json"
        return PipelineSummaryPaths(
            summary_path=summary_path,
            canonical_latest_path=canonical_latest_path,
            legacy_latest_path=legacy_latest_path,
        )

    def write_pipeline_summary(
        self,
        summary: dict[str, Any],
        *,
        paths: PipelineSummaryPaths,
        legacy_warning_label: str,
    ) -> None:
        payload = json.dumps(summary, ensure_ascii=False, indent=2)

        paths.summary_path.parent.mkdir(parents=True, exist_ok=True)
        paths.summary_path.write_text(payload, encoding="utf-8")

        paths.canonical_latest_path.parent.mkdir(parents=True, exist_ok=True)
        paths.canonical_latest_path.write_text(payload, encoding="utf-8")

        try:
            paths.legacy_latest_path.parent.mkdir(parents=True, exist_ok=True)
            paths.legacy_latest_path.write_text(payload, encoding="utf-8")
        except Exception as exc:
            print(f"[WARN] Failed to {legacy_warning_label} legacy latest summary path: {exc}", file=sys.stderr)

    @staticmethod
    def _write_payload(path: Path, payload: Any) -> None:
        if isinstance(payload, (dict, list)):
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return
        path.write_text(str(payload), encoding="utf-8")
