from __future__ import annotations

import copy
import json
import logging
import shutil
try:
    import fcntl  # R7.5: File locking for Unix-like systems
except ImportError:
    fcntl = None  # Windows doesn't have fcntl
import os
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from core.utils.file_helpers import load_json_object
from core.errors import DataContractError
from ontology_compiler.services.merge.identity_registry import IdentityRegistry
from ontology_compiler.services.artifacts import validate_layered_payloads

# NOTE (Phase 2 / P2-03, 2026-08-13): former `EntityVersionService` module no
# longer exists; see merge_validator.py for the same cleanup rationale. Call
# sites stay guard-protected for duck-typed external recorders.

logger = logging.getLogger(__name__)


class PatchTransactionLock:
    """File-based transaction lock for preventing concurrent patch operations (R7.5).

    Uses lock files to ensure only one patch operation modifies ontology files at a time.
    Supports both Unix (fcntl) and Windows (file existence) locking mechanisms.
    """

    def __init__(self, lock_path: Path, timeout_seconds: int = 30):
        self.lock_path = lock_path
        self.timeout_seconds = timeout_seconds
        self._lock_file: Any = None
        self._acquired = False

    def acquire(self) -> bool:
        """Acquire the lock with timeout."""
        import time
        start_time = time.time()

        while time.time() - start_time < self.timeout_seconds:
            try:
                # Try to create lock file exclusively
                self._lock_file = open(self.lock_path, "x")
                self._lock_file.write(json.dumps({
                    "pid": os.getpid(),
                    "timestamp": datetime.now().isoformat(),
                    "lock_type": "patch_transaction"
                }))

                # On Unix, also use fcntl for additional safety
                if fcntl:
                    try:
                        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except (OSError, IOError):
                        pass  # Lock failed but file creation succeeded

                self._acquired = True
                logger.debug(f"Acquired patch lock: {self.lock_path}")
                return True

            except FileExistsError:
                # Lock already held, check if stale
                try:
                    with open(self.lock_path, "r") as f:
                        lock_data = json.load(f)
                        lock_time = datetime.fromisoformat(lock_data.get("timestamp", "2000-01-01"))
                        # If lock is older than timeout, consider it stale
                        if (datetime.now() - lock_time).total_seconds() > self.timeout_seconds * 2:
                            logger.warning(f"Removing stale lock: {self.lock_path}")
                            self.lock_path.unlink()
                            continue
                except (json.JSONDecodeError, FileNotFoundError, ValueError):
                    pass

                time.sleep(0.1)  # Brief wait before retry

        logger.error(f"Failed to acquire patch lock within {self.timeout_seconds}s: {self.lock_path}")
        return False

    def release(self) -> None:
        """Release the lock."""
        if not self._acquired:
            return

        try:
            if self._lock_file:
                if fcntl:
                    try:
                        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                    except (OSError, IOError):
                        pass
                self._lock_file.close()
        except Exception as e:
            logger.warning(f"Error releasing lock file: {e}")
        finally:
            try:
                if self.lock_path.exists():
                    self.lock_path.unlink()
            except Exception as e:
                logger.warning(f"Error removing lock file: {e}")

            self._acquired = False
            logger.debug(f"Released patch lock: {self.lock_path}")

    def __enter__(self) -> "PatchTransactionLock":
        if not self.acquire():
            raise ConcurrentPatchError(f"Could not acquire lock: {self.lock_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class ConcurrentPatchError(Exception):
    """Raised when concurrent patch operation is detected."""
    pass


class OntologyPatchEngine:
    """Generate and apply ontology patches with dry-run validation.

    R7.5: Added transaction locking to prevent concurrent modification conflicts.
    """

    def __init__(self, version_service: "Any | None" = None) -> None:
        self.identity_registry = IdentityRegistry()
        self._version_service = version_service
        self._lock_timeout = 30  # seconds

    def build_patch(
        self,
        *,
        operations: list[dict[str, Any]],
        reason: str,
        source_refs: list[str] | None = None,
        requested_by: str = "system-architect",
    ) -> dict[str, Any]:
        self.identity_registry.require_role(subject_id=requested_by, allowed_roles=["verified-user", "auto-domain-architect"])
        return {
            "schema_version": "v1.0",
            "generated_at": datetime.now().isoformat(),
            "requested_by": requested_by,
            "reason": reason,
            "source_refs": source_refs or [],
            "operations": operations,
        }

    def apply_patch(
        self,
        *,
        patch: dict[str, Any],
        l1_path: Path,
        l2_path: Path,
        manifest_path: Path,
        dry_run: bool = True,
        simulate_rollback: bool = False,
        enable_rollback: bool = False,
        use_lock: bool = True,  # R7.5: Enable transaction locking
    ) -> dict[str, Any]:
        """Apply ontology patch with optional file-level rollback.

        P1-1 Enhancement (2026-05-30):
        - Added enable_rollback parameter for automatic file backup and recovery
        - Creates .backup files before writing changes
        - Automatically restores from backup on failure if enable_rollback=True

        R7.5 Enhancement (2026-06-04):
        - Added use_lock parameter for transaction-level locking
        - Prevents concurrent patch operations on the same ontology files

        Args:
            patch: Patch operations dictionary.
            l1_path: Path to L1 ontology file.
            l2_path: Path to L2 ontology file.
            manifest_path: Path to manifest file.
            dry_run: If True, only simulate without writing.
            simulate_rollback: If True, simulate rollback without actual backup.
            enable_rollback: If True, create file backups and auto-restore on failure.
            use_lock: If True (default), acquire transaction lock to prevent concurrent modifications.

        Returns:
            Result dictionary with status, operations count, and rollback info.
        """
        self._validate_patch_schema(patch)

        # R7.5: Acquire transaction lock if enabled (not for dry_run)
        if use_lock and not dry_run:
            lock_path = l1_path.parent / ".patch.lock"
            lock = PatchTransactionLock(lock_path, timeout_seconds=self._lock_timeout)
            if not lock.acquire():
                raise ConcurrentPatchError(
                    f"Another patch operation is in progress. "
                    f"Lock file: {lock_path}"
                )
        else:
            lock = None

        try:
            l1 = self._load_json(l1_path)
            l2 = self._load_json(l2_path)
            manifest = self._load_json(manifest_path)

            new_l1 = copy.deepcopy(l1)
            new_l2 = copy.deepcopy(l2)
            rollback_log: list[dict[str, Any]] = []

            for operation in patch.get("operations") or []:
                self._apply_single(operation=operation, l1=new_l1, l2=new_l2, rollback_log=rollback_log)

            validate_layered_payloads(l1=new_l1, l2=new_l2, manifest=manifest)

            rollback_simulation_ok: bool | None = None
            if simulate_rollback:
                rollback_simulation_ok = self._simulate_rollback(
                    l1_before=l1,
                    l2_before=l2,
                    l1_after=new_l1,
                    l2_after=new_l2,
                    rollback_log=rollback_log,
                )

            result = {
                "status": "DRY_RUN_SUCCESS" if dry_run else "SUCCESS",
                "dry_run": dry_run,
                "operations": len(patch.get("operations") or []),
                "rollback_log": rollback_log,
                "rollback_simulation_ok": rollback_simulation_ok,
            }
            if dry_run:
                return result

            # File-level backup for rollback support
            backup_l1_path = None
            backup_l2_path = None
            if enable_rollback:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_l1_path = l1_path.with_suffix(f"{l1_path.suffix}.backup.{timestamp}")
                backup_l2_path = l2_path.with_suffix(f"{l2_path.suffix}.backup.{timestamp}")
                shutil.copy2(l1_path, backup_l1_path)
                shutil.copy2(l2_path, backup_l2_path)
                logger.info(f"Created backups: {backup_l1_path.name}, {backup_l2_path.name}")

            try:
                self._write_json(l1_path, new_l1)
                self._write_json(l2_path, new_l2)

                # Register entity versions for ADD/MODIFY operations
                if self._version_service is not None:
                    patch_id = patch.get("schema_version", "unknown") + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
                    for operation in patch.get("operations") or []:
                        op = str(operation.get("op") or "").upper()
                        if op in {"ADD", "MODIFY"}:
                            payload = operation.get("payload") or {}
                            entity_id = str(payload.get("meta_id") or operation.get("target_id") or "")
                            if entity_id:
                                artifact = str(operation.get("target_artifact") or "")
                                entity_type = "OntologyRelationship" if artifact == "L1_RELATIONSHIPS" else "OntologyEntity"
                                try:
                                    self._version_service.register_entity_version(
                                        entity_id=entity_id,
                                        entity_type=entity_type,
                                        entity_data=payload,
                                        patch_id=patch_id,
                                    )
                                except Exception as _ve:
                                    logger.warning(f"version tracking failed for {entity_id}: {_ve}")

                result["backup_files"] = [str(backup_l1_path), str(backup_l2_path)] if enable_rollback else []
                return result

            except Exception as e:
                if enable_rollback and backup_l1_path and backup_l2_path:
                    logger.error(f"Patch application failed, restoring from backup: {e}")
                    shutil.copy2(backup_l1_path, l1_path)
                    shutil.copy2(backup_l2_path, l2_path)
                    result["status"] = "ROLLBACK_SUCCESS"
                    result["error"] = str(e)
                    result["backup_files"] = [str(backup_l1_path), str(backup_l2_path)]
                    return result
                raise

        finally:
            # R7.5: Release transaction lock
            if lock is not None:
                lock.release()

    def merge_patch_with_rollback(
        self,
        *,
        patch: dict[str, Any],
        l1_path: Path,
        l2_path: Path,
        manifest_path: Path,
    ) -> dict[str, Any]:
        backup_l1 = self._load_json(l1_path)
        backup_l2 = self._load_json(l2_path)
        try:
            return self.apply_patch(
                patch=patch,
                l1_path=l1_path,
                l2_path=l2_path,
                manifest_path=manifest_path,
                dry_run=False,
            )
        except Exception:
            self._write_json(l1_path, backup_l1)
            self._write_json(l2_path, backup_l2)
            raise

    def _validate_patch_schema(self, patch: dict[str, Any]) -> None:
        if not isinstance(patch, dict):
            raise DataContractError("patch must be object")

        operations = patch.get("operations")
        if not isinstance(operations, list) or not operations:
            raise DataContractError("patch.operations must be non-empty list")

        for idx, op in enumerate(operations):
            if not isinstance(op, dict):
                raise DataContractError(f"operation[{idx}] must be object")
            op_name = str(op.get("op") or "").upper()
            if op_name not in {"ADD", "DELETE", "MODIFY"}:
                raise DataContractError(f"operation[{idx}] invalid op: {op_name}")
            target_artifact = str(op.get("target_artifact") or "")
            if target_artifact not in {"L1_OBJECTS", "L1_RELATIONSHIPS", "L2_INSTANCES"}:
                raise DataContractError(f"operation[{idx}] invalid target_artifact: {target_artifact}")

            payload = op.get("payload")
            if op_name in {"ADD", "MODIFY"} and not isinstance(payload, dict):
                raise DataContractError(f"operation[{idx}] payload must be object for {op_name}")

            payload_meta_id = str(payload.get("meta_id") or "") if isinstance(payload, dict) else ""
            target_id = str(op.get("target_id") or payload_meta_id)
            if op_name in {"DELETE", "MODIFY"} and not target_id:
                raise DataContractError(f"operation[{idx}] target_id/meta_id required for {op_name}")

    def _simulate_rollback(
        self,
        *,
        l1_before: dict[str, Any],
        l2_before: dict[str, Any],
        l1_after: dict[str, Any],
        l2_after: dict[str, Any],
        rollback_log: list[dict[str, Any]],
    ) -> bool:
        probe_l1 = copy.deepcopy(l1_after)
        probe_l2 = copy.deepcopy(l2_after)
        probe_rollback: list[dict[str, Any]] = []

        for operation in reversed(rollback_log):
            self._apply_single(operation=operation, l1=probe_l1, l2=probe_l2, rollback_log=probe_rollback)

        return probe_l1 == l1_before and probe_l2 == l2_before

    def _apply_single(
        self,
        *,
        operation: dict[str, Any],
        l1: dict[str, Any],
        l2: dict[str, Any],
        rollback_log: list[dict[str, Any]],
    ) -> None:
        op = str(operation.get("op") or "").upper()
        target_artifact = str(operation.get("target_artifact") or "")
        payload = operation.get("payload") or {}
        target_id = str(operation.get("target_id") or payload.get("meta_id") or "")

        if target_artifact == "L1_OBJECTS":
            items = (l1.get("sandbox") or {}).setdefault("objects", [])
        elif target_artifact == "L1_RELATIONSHIPS":
            items = (l1.get("sandbox") or {}).setdefault("relationships", [])
        elif target_artifact == "L2_INSTANCES":
            items = l2.setdefault("instances", [])
        else:
            raise DataContractError(f"unsupported target_artifact: {target_artifact}")

        if op == "ADD":
            items.append(payload)
            rollback_log.append({"op": "DELETE", "target_artifact": target_artifact, "target_id": target_id})
            return

        idx = self._find_item_index(items, target_id)
        if idx is None:
            raise DataContractError(f"target not found for {op}: {target_artifact}:{target_id}")

        if op == "DELETE":
            old = items.pop(idx)
            rollback_log.append({"op": "ADD", "target_artifact": target_artifact, "payload": old, "target_id": target_id})
            return

        if op == "MODIFY":
            old = copy.deepcopy(items[idx])
            items[idx].update(payload)
            rollback_log.append({"op": "MODIFY", "target_artifact": target_artifact, "payload": old, "target_id": target_id})
            return

        raise DataContractError(f"unsupported patch op: {op}")

    def _find_item_index(self, items: list[dict[str, Any]], target_id: str) -> int | None:
        for idx, item in enumerate(items):
            meta_id = str(item.get("meta_id") or "")
            rel_key = "::".join([
                str(item.get("source") or ""),
                str(item.get("predicate") or item.get("type") or ""),
                str(item.get("target") or ""),
            ])
            if target_id and (meta_id == target_id or rel_key == target_id):
                return idx
        return None

    def _load_json(self, path: Path) -> dict[str, Any]:
        payload = load_json_object(path)
        if not isinstance(payload, dict):
            raise DataContractError(f"patch engine payload must be object: {path}")
        return payload

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


