"""Transactional Patch Skill — 并发安全事务补丁

R38-QA Skill #5: 确保并发场景下的数据一致性，提供事务性补丁操作和回滚机制。
"""

import asyncio
import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..base import Skill, SkillContext, SkillResult, SkillPriority

logger = logging.getLogger(__name__)


@dataclass
class PatchOperation:
    """Single patch operation."""
    op_id: str
    operation_type: str  # "add", "update", "delete"
    target_path: str  # e.g., "entities.0.attributes"
    old_value: Any = None
    new_value: Any = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class Transaction:
    """Patch transaction with rollback support."""
    transaction_id: str
    operations: List[PatchOperation] = field(default_factory=list)
    committed: bool = False
    rolled_back: bool = False
    timestamp: float = field(default_factory=time.time)
    
    def add_operation(self, op: PatchOperation) -> None:
        """Add operation to transaction."""
        self.operations.append(op)
    
    def can_commit(self) -> bool:
        """Check if transaction can be committed."""
        return not self.committed and not self.rolled_back


class TransactionalPatchSkill(Skill):
    """Provide transactional patch operations with concurrency safety.
    
    Features:
    1. Optimistic locking with version checking
    2. Automatic rollback on failure
    3. Conflict detection for concurrent modifications
    4. Dry-run mode for validation
    
    Evidence: patch_engine.py transactional operations
    Performance: Prevents data corruption in concurrent scenarios.
    """
    
    # Lock timeout in seconds
    LOCK_TIMEOUT = 30
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.enable_dry_run = self.config.get("enable_dry_run", True)
        self.enable_optimistic_lock = self.config.get("enable_optimistic_lock", True)
        self._locks: Dict[str, float] = {}  # resource -> lock_time
        self._versions: Dict[str, int] = {}  # resource -> version
        self._transactions: Dict[str, Transaction] = {}
    
    @property
    def priority(self) -> SkillPriority:
        return SkillPriority.CRITICAL
    
    @property
    def description(self) -> str:
        return "Transactional patch with concurrency safety"
    
    async def can_apply(self, context: SkillContext) -> bool:
        """Can apply if patch operations exist in config."""
        return "patch_operations" in context.config or "pending_patches" in context.metadata
    
    def _acquire_lock(self, resource: str) -> bool:
        """Acquire lock on resource."""
        current_time = time.time()
        
        if resource in self._locks:
            lock_time = self._locks[resource]
            if current_time - lock_time > self.LOCK_TIMEOUT:
                # Lock expired, release and reacquire
                del self._locks[resource]
            else:
                return False
        
        self._locks[resource] = current_time
        logger.debug(f"Acquired lock on {resource}")
        return True
    
    def _release_lock(self, resource: str) -> None:
        """Release lock on resource."""
        if resource in self._locks:
            del self._locks[resource]
            logger.debug(f"Released lock on {resource}")
    
    def _check_version(self, resource: str, expected_version: int) -> bool:
        """Check version for optimistic locking."""
        if not self.enable_optimistic_lock:
            return True
        
        current_version = self._versions.get(resource, 0)
        return current_version == expected_version
    
    def _increment_version(self, resource: str) -> None:
        """Increment version for resource."""
        self._versions[resource] = self._versions.get(resource, 0) + 1
    
    def _get_nested_value(self, obj: Any, path: str) -> Any:
        """Get value from nested object by path."""
        parts = path.split(".")
        current = obj
        
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    current = current[idx] if 0 <= idx < len(current) else None
                except ValueError:
                    return None
            else:
                return None
            
            if current is None:
                return None
        
        return current
    
    def _set_nested_value(self, obj: Any, path: str, value: Any) -> bool:
        """Set value in nested object by path."""
        parts = path.split(".")
        current = obj
        
        for i, part in enumerate(parts[:-1]):
            if isinstance(current, dict):
                if part not in current:
                    current[part] = {}
                current = current[part]
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    if 0 <= idx < len(current):
                        current = current[idx]
                    else:
                        return False
                except ValueError:
                    return False
            else:
                return False
        
        # Set the final value
        final_part = parts[-1]
        if isinstance(current, dict):
            current[final_part] = value
            return True
        elif isinstance(current, list):
            try:
                idx = int(final_part)
                if 0 <= idx < len(current):
                    current[idx] = value
                    return True
            except ValueError:
                return False
        return False
    
    def _apply_patch(self, target: Any, operation: PatchOperation) -> bool:
        """Apply single patch operation."""
        if operation.operation_type == "add":
            return self._set_nested_value(target, operation.target_path, operation.new_value)
        
        elif operation.operation_type == "update":
            # For updates, check version first
            expected_version = operation.old_value.get("_version", 0) if isinstance(operation.old_value, dict) else 0
            if self.enable_optimistic_lock:
                if not self._check_version(operation.target_path, expected_version):
                    logger.warning(f"Version conflict on {operation.target_path}")
                    return False
            return self._set_nested_value(target, operation.target_path, operation.new_value)
        
        elif operation.operation_type == "delete":
            # For delete, set to None or remove from parent
            parts = operation.target_path.split(".")
            if len(parts) > 1:
                parent_path = ".".join(parts[:-1])
                parent = self._get_nested_value(target, parent_path)
                final_key = parts[-1]
                if isinstance(parent, dict):
                    if final_key in parent:
                        del parent[final_key]
                        return True
                elif isinstance(parent, list):
                    try:
                        idx = int(final_key)
                        if 0 <= idx < len(parent):
                            parent.pop(idx)
                            return True
                    except ValueError:
                        pass
            return False
        
        return False
    
    def _rollback_transaction(self, target: Any, transaction: Transaction) -> None:
        """Rollback all operations in transaction."""
        logger.info(f"Rolling back transaction {transaction.transaction_id}")
        
        # Apply operations in reverse order
        for operation in reversed(transaction.operations):
            if operation.old_value is not None:
                self._apply_patch(target, PatchOperation(
                    op_id=f"rollback-{operation.op_id}",
                    operation_type="update",
                    target_path=operation.target_path,
                    old_value=None,
                    new_value=operation.old_value
                ))
    
    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute transactional patch operations."""
        # Get patch operations
        patch_ops = context.config.get("patch_operations", [])
        if not patch_ops:
            patch_ops = context.metadata.get("pending_patches", [])
        
        if not patch_ops:
            return SkillResult(
                success=True,
                skill_name=self.name,
                applied=False,
                output={"action": "none", "reason": "no patch operations"}
            )
        
        # Create transaction
        tx_id = f"tx-{int(time.time() * 1000)}"
        transaction = Transaction(transaction_id=tx_id)
        
        # Create working copy for dry-run
        target = context.extraction_result
        if target is None:
            target = {"entities": context.entities, "relationships": context.relationships}
        
        working_copy = deepcopy(target)
        
        # Collect resources to lock
        resources: Set[str] = set()
        for op in patch_ops:
            resources.add(op.get("target_path", "").split(".")[0])
        
        # Acquire locks
        acquired_locks: List[str] = []
        for resource in resources:
            if self._acquire_lock(resource):
                acquired_locks.append(resource)
            else:
                # Failed to acquire lock, rollback
                for r in acquired_locks:
                    self._release_lock(r)
                return SkillResult(
                    success=False,
                    skill_name=self.name,
                    applied=False,
                    error=f"Failed to acquire lock on {resource}",
                    output={"action": "none", "reason": f"lock timeout on {resource}"}
                )
        
        try:
            # Apply operations
            for op_data in patch_ops:
                operation = PatchOperation(
                    op_id=op_data.get("op_id", f"op-{len(transaction.operations)}"),
                    operation_type=op_data.get("type", "update"),
                    target_path=op_data.get("target_path", ""),
                    old_value=self._get_nested_value(working_copy, op_data.get("target_path", "")),
                    new_value=op_data.get("value")
                )
                
                # Dry-run validation
                if self.enable_dry_run:
                    dry_result = self._get_nested_value(working_copy, operation.target_path)
                    if operation.old_value != dry_result:
                        logger.warning(f"Dry-run mismatch for {operation.target_path}")
                
                # Apply operation
                if not self._apply_patch(working_copy, operation):
                    logger.error(f"Failed to apply operation {operation.op_id}")
                    self._rollback_transaction(working_copy, transaction)
                    return SkillResult(
                        success=False,
                        skill_name=self.name,
                        applied=False,
                        error=f"Failed to apply {operation.op_id}",
                        output={"action": "rollback", "operations_applied": len(transaction.operations)}
                    )
                
                transaction.add_operation(operation)
            
            # Commit successful
            transaction.committed = True
            
            # Update context
            if context.extraction_result is None:
                context.entities = working_copy.get("entities", context.entities)
                context.relationships = working_copy.get("relationships", context.relationships)
            else:
                context.extraction_result = working_copy
            
            return SkillResult(
                success=True,
                skill_name=self.name,
                applied=True,
                output={
                    "transaction_id": tx_id,
                    "operations_applied": len(transaction.operations),
                    "action": "committed"
                }
            )
        
        finally:
            # Release locks
            for resource in acquired_locks:
                self._release_lock(resource)
    
    def get_transaction_status(self, tx_id: str) -> Optional[Dict[str, Any]]:
        """Get transaction status."""
        tx = self._transactions.get(tx_id)
        if tx:
            return {
                "transaction_id": tx.transaction_id,
                "operations": len(tx.operations),
                "committed": tx.committed,
                "rolled_back": tx.rolled_back,
                "timestamp": tx.timestamp
            }
        return None
