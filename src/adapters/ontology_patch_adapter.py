"""Adapter implementing PatchApplicationPort using OntologyPatchEngine.

P2-4 Implementation (2026-06-16):
Decouples PatchIntegrationService from direct OntologyPatchEngine dependency,
enabling architecture compliance with port/adapter pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ontology_compiler.services.patch_engine import OntologyPatchEngine
from src.application.ports.quality_ports import (
    PatchApplicationPort,
    PatchOperation,
)

logger = logging.getLogger(__name__)


class OntologyPatchAdapter(PatchApplicationPort):
    """Adapter that wraps OntologyPatchEngine to implement PatchApplicationPort.

    This adapter translates the port's abstract interface to the concrete
    OntologyPatchEngine implementation, enabling dependency inversion.

    Usage:
        # Create adapter with engine
        engine = OntologyPatchEngine()
        adapter = OntologyPatchAdapter(engine)

        # Use through port interface
        result = adapter.apply_patch(operations, dry_run=True)
    """

    def __init__(
        self,
        patch_engine: OntologyPatchEngine | None = None,
        l1_path: Path | None = None,
        l2_path: Path | None = None,
        manifest_path: Path | None = None,
    ) -> None:
        """Initialize the adapter with optional engine and paths.

        Args:
            patch_engine: Optional OntologyPatchEngine instance.
            l1_path: Default L1 ontology file path.
            l2_path: Default L2 ontology file path.
            manifest_path: Default manifest file path.
        """
        self._engine = patch_engine or OntologyPatchEngine()
        self._l1_path = l1_path
        self._l2_path = l2_path
        self._manifest_path = manifest_path
        self._patch_history: list[dict[str, Any]] = []

    def set_paths(
        self,
        l1_path: Path,
        l2_path: Path,
        manifest_path: Path,
    ) -> None:
        """Set the ontology file paths for patch application."""
        self._l1_path = l1_path
        self._l2_path = l2_path
        self._manifest_path = manifest_path

    def apply_patch(
        self,
        operations: list[PatchOperation],
        dry_run: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Apply patch operations to ontology.

        Args:
            operations: List of PatchOperation objects.
            dry_run: If True, only validate without writing.
            **kwargs: Additional options (simulate_rollback, enable_rollback, etc.)

        Returns:
            Result dictionary with status and operation details.

        Raises:
            ValueError: If paths are not configured.
        """
        if not all([self._l1_path, self._l2_path, self._manifest_path]):
            raise ValueError(
                "Ontology paths not configured. Call set_paths() before apply_patch()."
            )

        # Convert PatchOperation to engine format
        engine_operations = [
            {
                "op": op.op,
                "target_artifact": op.entity_type,
                "target_id": op.entity_id,
                "payload": op.payload,
                "previous": op.previous,
            }
            for op in operations
        ]

        # Build patch using engine
        patch = self._engine.build_patch(
            operations=engine_operations,
            reason=kwargs.get("reason", "Port adapter invocation"),
            source_refs=kwargs.get("source_refs"),
            requested_by=kwargs.get("requested_by", "patch-adapter"),
        )

        # Apply through engine
        result = self._engine.apply_patch(
            patch=patch,
            l1_path=self._l1_path,
            l2_path=self._l2_path,
            manifest_path=self._manifest_path,
            dry_run=dry_run,
            simulate_rollback=kwargs.get("simulate_rollback", True),
            enable_rollback=kwargs.get("enable_rollback", False),
            use_lock=kwargs.get("use_lock", True),
        )

        # Record in history
        self._patch_history.append({
            "patch_id": patch.get("generated_at"),
            "operations_count": len(operations),
            "dry_run": dry_run,
            "result": result,
        })

        return result

    def rollback(self, patch_id: str) -> bool:
        """Rollback a previously applied patch.

        Note: OntologyPatchEngine uses file-level backup rollback.
        This method searches for backup files matching the patch timestamp.

        Args:
            patch_id: The patch identifier (timestamp from generated_at).

        Returns:
            True if rollback succeeded, False otherwise.
        """
        if not self._l1_path or not self._l2_path:
            logger.warning("Cannot rollback: paths not configured")
            return False

        # Find backup files matching patch_id timestamp
        # Format: generated_at = "2026-06-16T12:34:56.789"
        # Backup format: .backup.20260616_123456
        try:
            timestamp_part = patch_id.split("T")[0].replace("-", "") + "_" + patch_id.split("T")[1][:8].replace(":", "")
        except (IndexError, ValueError):
            logger.warning(f"Invalid patch_id format: {patch_id}")
            return False

        backup_l1 = self._l1_path.with_suffix(f"{self._l1_path.suffix}.backup.{timestamp_part}")
        backup_l2 = self._l2_path.with_suffix(f"{self._l2_path.suffix}.backup.{timestamp_part}")

        if not backup_l1.exists() or not backup_l2.exists():
            logger.warning(f"Backup files not found for patch: {patch_id}")
            return False

        try:
            import shutil
            shutil.copy2(backup_l1, self._l1_path)
            shutil.copy2(backup_l2, self._l2_path)
            logger.info(f"Rollback successful for patch: {patch_id}")
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def build_patch(
        self,
        operations: list[PatchOperation],
        reason: str,
        source_refs: list[str] | None = None,
        requested_by: str = "patch-adapter",
    ) -> dict[str, Any]:
        """Build a patch dictionary from operations.

        This is a convenience method for creating patches without applying them.

        Args:
            operations: List of PatchOperation objects.
            reason: Reason for the patch.
            source_refs: Source references.
            requested_by: Requester identity.

        Returns:
            Patch dictionary ready for application.
        """
        engine_operations = [
            {
                "op": op.op,
                "target_artifact": op.entity_type,
                "target_id": op.entity_id,
                "payload": op.payload,
                "previous": op.previous,
            }
            for op in operations
        ]

        return self._engine.build_patch(
            operations=engine_operations,
            reason=reason,
            source_refs=source_refs,
            requested_by=requested_by,
        )

    def get_patch_history(self) -> list[dict[str, Any]]:
        """Get history of applied patches."""
        return list(self._patch_history)


__all__ = ["OntologyPatchAdapter"]