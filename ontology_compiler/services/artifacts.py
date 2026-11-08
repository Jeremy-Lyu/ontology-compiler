"""Artifact management functions for folder analysis.

[BATCH 4 CONSOLIDATION 2026-06-29]:
- This module provides **generic artifact I/O operations** (JSON/text writing, validation)
- For **layered artifact models and storage**, see layer_artifacts.py

Boundary Alignment:
- artifacts.py: Generic I/O, schema validation, auxiliary artifact relocation
- layer_artifacts.py: Layered data models (L1/L2/L2.5), LayerArtifactStore, incremental updates
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from core.contracts.ports import ArtifactStorePort
from core.errors import PipelineError, SchemaValidationError


def write_json_artifact(
    *,
    run_dir: Path,
    filename: str,
    payload: Dict[str, Any],
    artifact_store: ArtifactStorePort | None = None,
) -> str:
    """Write a JSON artifact to disk or via the artifact store."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if artifact_store is not None:
        stored = artifact_store.write_artifacts(run_id=run_dir.name, artifacts={filename: payload})
        return stored[0].path
    path = run_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(path)


def write_text_artifact(
    *,
    run_dir: Path,
    filename: str,
    content: str,
    artifact_store: ArtifactStorePort | None = None,
) -> str:
    """Write a text artifact to disk or via the artifact store."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if artifact_store is not None:
        stored = artifact_store.write_artifacts(run_id=run_dir.name, artifacts={filename: content})
        return stored[0].path
    path = run_dir / filename
    path.write_text(content, encoding="utf-8")
    return str(path)

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover
    Draft202012Validator = None  # type: ignore[assignment]


def relocate_minimal_aux_artifacts(*, run_dir: Path, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Relocate auxiliary artifacts into `run_dir/aux/`.

    This keeps the run folder top-level clean while preserving debugging/traceability.
    Only used in minimal output mode.

    Args:
        run_dir: Target run directory.
        artifacts: Dictionary of artifact paths.

    Returns:
        Updated artifact paths dictionary.
    """
    import logging

    logger = logging.getLogger(__name__)

    aux_dir = run_dir / "aux"
    aux_dir.mkdir(parents=True, exist_ok=True)

    keys_to_move = {
        "supplement": "tbh_supplement.json",
        "layered": "layer2_instances.json",
        "analysis": "TBH_ANALYSIS.md",
    }

    updated: Dict[str, Any] = dict(artifacts)
    for key, filename in keys_to_move.items():
        src = run_dir / filename
        if not src.exists() or not src.is_file():
            continue
        dst = aux_dir / filename
        try:
            shutil.move(str(src), str(dst))
            updated[key] = str(dst)
        except Exception as exc:
            logger.exception(f"failed to relocate {filename} to aux: {exc}")

    # Handle compilations directory if it exists
    comp_dir = artifacts.get("compilations_dir")
    if comp_dir:
        comp_path = Path(comp_dir)
        if comp_path.exists() and comp_path.is_dir():
            # To avoid merge complexity; keep both by nesting.
            dst_dir = aux_dir / f"compilations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(str(comp_dir), str(dst_dir))
            updated["compilations_dir"] = str(dst_dir)

    return updated


def write_layered_artifacts(
    *,
    run_dir: Path,
    l1: Dict[str, Any],
    l2: Dict[str, Any],
    manifest: Dict[str, Any],
    artifact_store: ArtifactStorePort | None = None,
) -> Dict[str, str]:
    """Write layered artifacts (L1, L2, manifest) to run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)

    if artifact_store is not None:
        stored = artifact_store.write_run_artifact_set(
            run_id=run_dir.name,
            manifest=manifest,
            artifacts={
                "L1_DOMAIN_SANDBOX.json": l1,
                "L2_OPERATIONAL_INSTANCES.json": l2,
            },
        )
        artifact_map = {artifact.name: artifact.path for artifact in stored.artifacts}
        return {
            "l1": artifact_map["L1_DOMAIN_SANDBOX.json"],
            "l2": artifact_map["L2_OPERATIONAL_INSTANCES.json"],
            "manifest": stored.manifest_path,
        }

    return {
        "l1": write_json_artifact(run_dir=run_dir, filename="L1_DOMAIN_SANDBOX.json", payload=l1),
        "l2": write_json_artifact(run_dir=run_dir, filename="L2_OPERATIONAL_INSTANCES.json", payload=l2),
        "manifest": write_json_artifact(run_dir=run_dir, filename="analysis_manifest.json", payload=manifest),
    }


def _validate_with_schema(payload: Dict[str, Any], schema_filename: str, payload_name: str) -> None:
    if Draft202012Validator is None:
        return
    schema_path = Path(__file__).parent.parent.parent / "schemas" / schema_filename
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    except PipelineError:
        raise
    except Exception as exc:
        raise SchemaValidationError(f"{payload_name} schema bootstrap failed: {exc}") from exc

    if errors:
        first = errors[0]
        where = ".".join(str(p) for p in first.path) or "<root>"
        raise SchemaValidationError(f"{payload_name} schema validation failed at {where}: {first.message}")


def validate_layered_payloads(
    *,
    l1: Dict[str, Any],
    l2: Dict[str, Any],
    manifest: Dict[str, Any],
) -> None:
    """Validate L1, L2, and manifest payloads against their schemas.

    Args:
        l1: L1 domain sandbox payload.
        l2: L2 operational instances payload.
        manifest: Analysis manifest payload.

    Raises:
        SchemaValidationError: If any payload fails validation.
    """
    _validate_with_schema(l1, "l1_domain_sandbox.schema.json", "L1_DOMAIN_SANDBOX")
    _validate_with_schema(l2, "l2_operational_instances.schema.json", "L2_OPERATIONAL_INSTANCES")
    _validate_with_schema(manifest, "analysis_manifest.schema.json", "analysis_manifest")


__all__ = [
    "write_json_artifact",
    "write_text_artifact",
    "relocate_minimal_aux_artifacts",
    "write_layered_artifacts",
    "validate_layered_payloads",
]
