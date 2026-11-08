"""Patch integration service for ontology pipeline.

B-2 Implementation 2026-05-09:
Integrates OntologyPatchEngine into the core folder analysis pipeline,
enabling atomic patch-based publication of compilation results.

H6 Integration 2026-05-26:
Added AnalysisResult to Patch conversion support, bridging DataAnalysisOrchestrator
with OntologyPatchEngine via AnalysisResultPatchAdapter.

P2-4 Refactoring 2026-06-16:
Decoupled from direct OntologyPatchEngine dependency.
Now uses PatchApplicationPort interface for architecture compliance.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from src.application.ports.quality_ports import PatchApplicationPort, PatchOperation as PortPatchOperation

if TYPE_CHECKING:
    from ontology_compiler.services.patch_engine import OntologyPatchEngine

logger = logging.getLogger(__name__)


@dataclass
class PatchResult:
    """Result of patch application."""

    status: str  # SUCCESS, DRY_RUN_SUCCESS, FAILED
    patch_id: str
    operations_applied: int
    dry_run: bool
    rollback_available: bool
    rollback_simulation_ok: bool | None = None
    error_message: str | None = None


class PatchIntegrationService:
    """Application wrapper for patch construction and publication.

    The service keeps patch-building logic separate from port invocation so the
    orchestration boundary stays explicit.
    """

    def __init__(
        self,
        patch_port: PatchApplicationPort | None = None,
        patch_engine: "OntologyPatchEngine | None" = None,
    ) -> None:
        if patch_port is not None:
            self._patch_port = patch_port
        else:
            from ontology_compiler.services.patch_engine import OntologyPatchEngine
            from src.adapters.ontology_patch_adapter import OntologyPatchAdapter

            engine = patch_engine or OntologyPatchEngine()
            self._patch_port = OntologyPatchAdapter(engine)

        self._patch_history: list[dict[str, Any]] = []

    @property
    def patch_port(self) -> PatchApplicationPort:
        return self._patch_port

    def _add_patch_operation(
        self,
        operations: list[PortPatchOperation],
        op: str,
        target_artifact: str,
        target_id: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        operations.append(
            PortPatchOperation(
                op=op,
                entity_type=target_artifact,
                entity_id=target_id,
                payload=payload,
            )
        )

    def _build_operations_from_compilation(
        self,
        *,
        l1_before: dict[str, Any],
        l1_after: dict[str, Any],
        l2_before: dict[str, Any],
        l2_after: dict[str, Any],
    ) -> list[PortPatchOperation]:
        operations: list[PortPatchOperation] = []
        self._build_l1_object_operations(operations, l1_before, l1_after)
        self._build_l1_relationship_operations(operations, l1_before, l1_after)
        self._build_l2_instance_operations(operations, l2_before, l2_after)
        return operations

    def _build_l1_object_operations(
        self,
        operations: list[PortPatchOperation],
        l1_before: dict[str, Any],
        l1_after: dict[str, Any],
    ) -> None:
        before_objects = {
            obj.get("meta_id"): obj
            for obj in (l1_before.get("sandbox") or {}).get("objects", [])
            if obj.get("meta_id")
        }
        after_objects = {
            obj.get("meta_id"): obj
            for obj in (l1_after.get("sandbox") or {}).get("objects", [])
            if obj.get("meta_id")
        }

        for obj_id, obj in after_objects.items():
            if obj_id not in before_objects:
                self._add_patch_operation(operations, "ADD", "L1_OBJECTS", obj_id, obj)
        for obj_id in before_objects:
            if obj_id not in after_objects:
                self._add_patch_operation(operations, "DELETE", "L1_OBJECTS", obj_id)
            elif before_objects[obj_id] != after_objects[obj_id]:
                self._add_patch_operation(operations, "MODIFY", "L1_OBJECTS", obj_id, after_objects[obj_id])

    def _build_l1_relationship_operations(
        self,
        operations: list[PortPatchOperation],
        l1_before: dict[str, Any],
        l1_after: dict[str, Any],
    ) -> None:
        before_rels = self._index_relationships((l1_before.get("sandbox") or {}).get("relationships", []))
        after_rels = self._index_relationships((l1_after.get("sandbox") or {}).get("relationships", []))
        for rel_key, rel in after_rels.items():
            if rel_key not in before_rels:
                self._add_patch_operation(operations, "ADD", "L1_RELATIONSHIPS", rel_key, rel)
            elif before_rels[rel_key] != rel:
                self._add_patch_operation(operations, "MODIFY", "L1_RELATIONSHIPS", rel_key, rel)
        for rel_key in before_rels:
            if rel_key not in after_rels:
                self._add_patch_operation(operations, "DELETE", "L1_RELATIONSHIPS", rel_key)

    def _build_l2_instance_operations(
        self,
        operations: list[PortPatchOperation],
        l2_before: dict[str, Any],
        l2_after: dict[str, Any],
    ) -> None:
        before_instances = {
            inst.get("id"): inst
            for inst in (l2_before or {}).get("instances", [])
            if inst.get("id")
        }
        after_instances = {
            inst.get("id"): inst
            for inst in (l2_after or {}).get("instances", [])
            if inst.get("id")
        }
        for inst_id, inst in after_instances.items():
            if inst_id not in before_instances:
                self._add_patch_operation(operations, "ADD", "L2_INSTANCES", inst_id, inst)
        for inst_id in before_instances:
            if inst_id not in after_instances:
                self._add_patch_operation(operations, "DELETE", "L2_INSTANCES", inst_id)
            elif before_instances[inst_id] != after_instances[inst_id]:
                self._add_patch_operation(operations, "MODIFY", "L2_INSTANCES", inst_id, after_instances[inst_id])

    def build_patch_from_compilation(
        self,
        *,
        l1_before: dict[str, Any],
        l1_after: dict[str, Any],
        l2_before: dict[str, Any],
        l2_after: dict[str, Any],
        reason: str,
        source_refs: list[str] | None = None,
        requested_by: str = "system-pipeline",
    ) -> dict[str, Any]:
        patch_ops = self._build_operations_from_compilation(
            l1_before=l1_before,
            l1_after=l1_after,
            l2_before=l2_before,
            l2_after=l2_after,
        )
        patch = self._patch_port.build_patch(
            operations=patch_ops,
            reason=reason,
            source_refs=source_refs,
            requested_by=requested_by,
        )
        logger.info(f"Built patch with {len(patch_ops)} operations: {reason}")
        return patch

    def _index_relationships(self, relationships: list[dict]) -> dict[str, dict]:
        indexed = {}
        for rel in relationships:
            source = rel.get("source") or rel.get("subject", "")
            predicate = rel.get("predicate") or rel.get("type", "")
            target = rel.get("target") or rel.get("object", "")
            indexed[f"{source}::{predicate}::{target}"] = rel
        return indexed

    def _load_run_artifacts(self, run_dir: Path) -> tuple[Path, Path, Path] | None:
        l1_path = run_dir / "L1_DOMAIN_SANDBOX.json"
        l2_path = run_dir / "L2_OPERATIONAL_INSTANCES.json"
        manifest_path = run_dir / "analysis_manifest.json"

        required = [l1_path, l2_path, manifest_path]
        if not all(p.exists() for p in required):
            missing = [p.name for p in required if not p.exists()]
            logger.warning("Patch for %s missing required files: %s", run_dir, missing)
            return None

        return l1_path, l2_path, manifest_path

    def _read_json_file(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_baseline_states(
        self,
        *,
        baseline_l1_path: Path | None,
        baseline_l2_path: Path | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        l1_before = (
            self._read_json_file(baseline_l1_path)
            if baseline_l1_path and baseline_l1_path.exists()
            else {"sandbox": {"objects": [], "relationships": []}}
        )
        l2_before = (
            self._read_json_file(baseline_l2_path)
            if baseline_l2_path and baseline_l2_path.exists()
            else {"instances": []}
        )
        return l1_before, l2_before

    def _read_runtime_states(self, run_artifacts: tuple[Path, Path, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
        l1_path, l2_path, _manifest_path = run_artifacts
        return self._read_json_file(l1_path), self._read_json_file(l2_path)

    def _build_patch_operation_payloads(self, patch: dict[str, Any]) -> list[PortPatchOperation]:
        operations = patch.get("operations", [])
        return [
            PortPatchOperation(
                op=op.get("op", "ADD"),
                entity_type=op.get("target_artifact", "L1_OBJECTS"),
                entity_id=op.get("target_id", ""),
                payload=op.get("payload"),
            )
            for op in operations
        ]

    def _prepare_patch_application(
        self,
        *,
        patch: dict[str, Any],
        run_dir: Path,
        dry_run: bool,
        simulate_rollback: bool,
    ) -> tuple[PatchResult | None, tuple[Path, Path, Path] | None, list[PortPatchOperation] | None, list[Any] | None]:
        run_artifacts = self._load_run_artifacts(run_dir)
        if run_artifacts is None:
            return (
                PatchResult(
                    status="FAILED",
                    patch_id=patch.get("generated_at", "unknown"),
                    operations_applied=0,
                    dry_run=dry_run,
                    rollback_available=False,
                    error_message=f"Missing required files in {run_dir}",
                ),
                None,
                None,
                None,
            )

        patch_ops = self._build_patch_operation_payloads(patch)
        operations = patch.get("operations", [])
        return None, run_artifacts, patch_ops, operations

    def _record_patch_history(
        self,
        *,
        patch: dict[str, Any],
        result: dict[str, Any],
        run_dir: Path,
        dry_run: bool,
    ) -> None:
        self._patch_history.append(
            {"patch": patch, "result": result, "run_dir": str(run_dir), "dry_run": dry_run}
        )

    def _build_patch_result(
        self,
        *,
        patch: dict[str, Any],
        operations_applied: int,
        dry_run: bool,
        rollback_available: bool,
        rollback_simulation_ok: bool | None = None,
        error_message: str | None = None,
        status: str = "SUCCESS",
    ) -> PatchResult:
        return PatchResult(
            status=status,
            patch_id=patch.get("generated_at", "unknown"),
            operations_applied=operations_applied,
            dry_run=dry_run,
            rollback_available=rollback_available,
            rollback_simulation_ok=rollback_simulation_ok,
            error_message=error_message,
        )

    def _build_empty_patch_rejection(self, patch: dict[str, Any]) -> PatchResult:
        return self._build_patch_result(
            patch=patch,
            operations_applied=0,
            dry_run=True,
            rollback_available=False,
            error_message="Empty patch: no changes detected between baseline and current",
            status="FAILED",
        )

    def _build_excessive_operations_rejection(self, patch: dict[str, Any], op_count: int, max_operations: int) -> PatchResult:
        return self._build_patch_result(
            patch=patch,
            operations_applied=op_count,
            dry_run=True,
            rollback_available=False,
            error_message=f"Excessive operations ({op_count} > {max_operations}): potential runaway modification blocked",
            status="FAILED",
        )

    def _build_critical_deletion_rejection(self, patch: dict[str, Any], op_count: int, critical_deletions: list[dict[str, Any]]) -> PatchResult:
        return self._build_patch_result(
            patch=patch,
            operations_applied=op_count,
            dry_run=True,
            rollback_available=False,
            error_message=f"Critical entity deletion blocked: {len(critical_deletions)} protected entities",
            status="FAILED",
        )

    def _validate_patch_safety(self, patch: dict[str, Any], run_dir: Path) -> tuple[bool, PatchResult | None]:
        operations = patch.get("operations", [])
        op_count = len(operations)

        if op_count == 0:
            logger.warning(f"Patch for {run_dir}: no operations detected (empty patch)")
            return False, self._build_empty_patch_rejection(patch)

        MAX_REASONABLE_OPERATIONS = 1000
        if op_count > MAX_REASONABLE_OPERATIONS:
            logger.error(
                f"Patch for {run_dir}: excessive operations ({op_count} > {MAX_REASONABLE_OPERATIONS}). This may indicate a runaway process or incorrect baseline."
            )
            return False, self._build_excessive_operations_rejection(patch, op_count, MAX_REASONABLE_OPERATIONS)

        critical_prefixes = ("META_", "SYSTEM_", "CORE_")
        critical_deletions = [
            op
            for op in operations
            if op.get("op") == "DELETE" and any(str(op.get("target_id", "")).startswith(p) for p in critical_prefixes)
        ]
        if critical_deletions:
            logger.error(
                f"Patch for {run_dir}: attempted deletion of {len(critical_deletions)} critical entities: {[op.get('target_id') for op in critical_deletions[:5]]}"
            )
            return False, self._build_critical_deletion_rejection(patch, op_count, critical_deletions)

        return True, None

    def _apply_compiled_patch(
        self,
        *,
        patch: dict[str, Any],
        run_dir: Path,
        dry_run: bool,
    ) -> PatchResult:
        ok, rejected = self._validate_patch_safety(patch, run_dir)
        if not ok and rejected is not None:
            rejected.dry_run = dry_run
            return rejected

        return self.apply_patch_to_run(patch=patch, run_dir=run_dir, dry_run=dry_run, simulate_rollback=True)

    def apply_patch_to_run(
        self,
        *,
        patch: dict[str, Any],
        run_dir: Path,
        dry_run: bool = True,
        simulate_rollback: bool = True,
    ) -> PatchResult:
        run_artifacts = self._load_run_artifacts(run_dir)
        if run_artifacts is None:
            return self._build_patch_result(
                patch=patch,
                operations_applied=0,
                dry_run=dry_run,
                rollback_available=False,
                error_message=f"Missing required files in {run_dir}",
                status="FAILED",
            )

        l1_path, l2_path, manifest_path = run_artifacts

        try:
            self._patch_port.set_paths(l1_path, l2_path, manifest_path)
            patch_ops = self._build_patch_operation_payloads(patch)
            operations = patch.get("operations", [])

            if len(patch_ops) != len(operations):
                logger.warning(
                    "Patch %s: normalized %d raw operations to %d port operations",
                    patch.get("generated_at", "unknown"),
                    len(operations),
                    len(patch_ops),
                )

            result = self._patch_port.apply_patch(
                operations=patch_ops,
                dry_run=dry_run,
                simulate_rollback=simulate_rollback,
                reason=patch.get("reason", ""),
                source_refs=patch.get("source_refs"),
                requested_by=patch.get("requested_by", "system-pipeline"),
            )
            self._record_patch_history(patch=patch, result=result, run_dir=run_dir, dry_run=dry_run)

            status = "DRY_RUN_SUCCESS" if dry_run else "SUCCESS"
            if result.get("status") == "FAILED":
                status = "FAILED"

            return self._build_patch_result(
                patch=patch,
                operations_applied=result.get("operations", len(operations)),
                dry_run=dry_run,
                rollback_available=True,
                rollback_simulation_ok=result.get("rollback_simulation_ok"),
                error_message=result.get("error_message"),
                status=status,
            )
        except Exception as e:
            logger.error(f"Patch application failed: {e}")
            return self._build_patch_result(
                patch=patch,
                operations_applied=0,
                dry_run=dry_run,
                rollback_available=False,
                error_message=str(e),
                status="FAILED",
            )

    def publish_compilation_as_patch(
        self,
        *,
        run_dir: Path,
        baseline_l1_path: Path | None = None,
        baseline_l2_path: Path | None = None,
        reason: str = "Compilation result publication",
        requested_by: str = "compilation-pipeline",
        dry_run: bool = True,
    ) -> PatchResult:
        run_artifacts = self._load_run_artifacts(run_dir)
        if run_artifacts is None:
            return self._build_patch_result(
                patch={"generated_at": ""},
                operations_applied=0,
                dry_run=dry_run,
                rollback_available=False,
                error_message=f"Missing L1/L2 files in {run_dir}",
                status="FAILED",
            )

        l1_after, l2_after = self._read_runtime_states(run_artifacts)
        l1_before, l2_before = self._load_baseline_states(
            baseline_l1_path=baseline_l1_path,
            baseline_l2_path=baseline_l2_path,
        )

        patch = self.build_patch_from_compilation(
            l1_before=l1_before,
            l1_after=l1_after,
            l2_before=l2_before,
            l2_after=l2_after,
            reason=reason,
            source_refs=[str(run_dir)],
            requested_by=requested_by,
        )
        return self._apply_compiled_patch(patch=patch, run_dir=run_dir, dry_run=dry_run)

    def get_patch_history(self) -> list[dict[str, Any]]:
        return list(self._patch_history)

    def save_patch_to_file(
        self,
        *,
        patch: dict[str, Any],
        output_path: Path,
    ) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(patch, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved patch to {output_path}")

    def build_patch_from_analysis_result(
        self,
        *,
        analysis_result: Any,
        domain: str = "SUPPLY_CHAIN",
        reason: str = "Data analysis findings converted to ontology patch",
        source_refs: list[str] | None = None,
        requested_by: str = "data-analysis-pipeline",
        use_llm: bool = True,
    ) -> dict[str, Any]:
        logger.warning(
            "build_patch_from_analysis_result is deprecated. "
            "The data_analysis subsystem has been removed. "
            "Use build_patch_from_compilation instead."
        )
        return self._patch_port.build_patch(
            operations=[],
            reason=f"{reason} (data_analysis subsystem removed)",
            source_refs=source_refs or [],
            requested_by=requested_by,
        )
