"""Incremental merge validator and decision engine for ontology layer outputs.

P1-8 Refactoring (2026-05-16):
- Merged auto_merge.py into this module as MergeDecisionEngine class
- Unified validation and merge execution in single module
- Maintains backward compatibility with existing API
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ontology_compiler.config import get_config

# R11: Migrate to EntityDeduplicationService
from ontology_compiler.services.merge.entity_dedup_service import EntityDeduplicationService

# NOTE (Phase 2 / P2-03, 2026-08-13): the former
# `src.application.services.entity_version_service.EntityVersionService` no
# longer exists anywhere in the active tree. The old TYPE_CHECKING-only import
# was a dangling reference; call sites are protected by `is not None` guards so
# external duck-typed version recorders remain compatible. The annotation is
# therefore a plain blob type (`Any | None`) rather than a phantom class name.

logger = logging.getLogger(__name__)


# =============================================================================
# Merge Decision Engine (from auto_merge.py)
# =============================================================================

class MergeAction(str, Enum):
    """Merge action types."""

    SKIP = "skip"  # Skip the object (duplicate)
    MERGE = "merge"  # Merge as new object
    OVERWRITE = "overwrite"  # Overwrite existing object
    RENAME = "rename"  # Rename to avoid conflict
    MANUAL = "manual"  # Require manual review


@dataclass
class MergeDecision:
    """Decision for a single object merge."""

    meta_id: str
    meta_name: str
    action: MergeAction
    confidence: float  # 0.0 to 1.0
    reason: str
    new_meta_id: Optional[str] = None  # For RENAME action
    source_object: Dict[str, Any] = field(default_factory=dict)
    target_object: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta_id": self.meta_id,
            "meta_name": self.meta_name,
            "action": self.action.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "new_meta_id": self.new_meta_id,
        }


@dataclass
class MergeResult:
    """Result of automatic merge execution."""

    merged_count: int
    skipped_count: int
    overwritten_count: int
    renamed_count: int
    manual_review_count: int
    decisions: List[MergeDecision] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "merged_count": self.merged_count,
            "skipped_count": self.skipped_count,
            "overwritten_count": self.overwritten_count,
            "renamed_count": self.renamed_count,
            "manual_review_count": self.manual_review_count,
            "decisions": [d.to_dict() for d in self.decisions],
            "warnings": self.warnings,
        }


class MergeDecisionEngine:
    """Unified merge decision engine for incremental ontology updates.

    P1-8 Refactoring: Combines validation and auto-merge execution from:
    - merge_validator.py (validation logic)
    - auto_merge.py (execution logic)

    Usage:
        engine = MergeDecisionEngine()
        result = engine.execute_merge(
            l1_objects=l1_objects,
            master_nodes=master_nodes,
            confidence_threshold=0.8,
        )
    """

    def __init__(self) -> None:
        self._confidence_threshold: float = 0.8
        self._priorities: Dict[str, float] = {
            "duplicate_exact_match": 1.0,  # Exact meta_id match
            "conflict_high_confidence": 0.9,  # Class conflict with high confidence
            "conflict_low_confidence": 0.7,  # Class conflict with low confidence
            "new_object": 1.0,  # New object (no conflict)
        }

    def set_confidence_threshold(self, threshold: float) -> "MergeDecisionEngine":
        """Set the minimum confidence for automatic actions."""
        self._confidence_threshold = max(0.0, min(1.0, threshold))
        return self

    def set_priority(self, key: str, value: float) -> "MergeDecisionEngine":
        """Set priority for a specific merge scenario."""
        self._priorities[key] = max(0.0, min(1.0, value))
        return self

    def execute_merge(
        self,
        *,
        l1_objects: List[Dict[str, Any]],
        master_nodes: List[Dict[str, Any]],
        confidence_threshold: Optional[float] = None,
        auto_overwrite: bool = False,
        auto_rename: bool = True,
    ) -> MergeResult:
        """Execute automatic merge based on validation report.

        Args:
            l1_objects: L1 sandbox objects to merge
            master_nodes: Existing master ontology nodes
            confidence_threshold: Minimum confidence for auto-actions (default: 0.8)
            auto_overwrite: Allow automatic overwrites (default: False)
            auto_rename: Allow automatic renaming (default: True)

        Returns:
            MergeResult with decisions and statistics
        """
        if confidence_threshold is not None:
            self.set_confidence_threshold(confidence_threshold)

        result = MergeResult(
            merged_count=0,
            skipped_count=0,
            overwritten_count=0,
            renamed_count=0,
            manual_review_count=0,
        )

        # Build master lookup indices
        master_by_id: Dict[str, Dict[str, Any]] = {}
        master_by_name: Dict[str, Dict[str, Any]] = {}
        for node in master_nodes:
            mid = str(node.get("meta_id") or node.get("node_id") or node.get("id") or "").strip()
            mname = str(node.get("meta_name") or node.get("name") or "").strip().lower()
            if mid:
                master_by_id[mid] = node
            if mname:
                master_by_name[mname] = node

        # Process each L1 object
        for obj in l1_objects:
            decision = self._decide_merge(
                obj=obj,
                master_by_id=master_by_id,
                master_by_name=master_by_name,
                auto_overwrite=auto_overwrite,
                auto_rename=auto_rename,
            )
            result.decisions.append(decision)

            # Update statistics
            if decision.action == MergeAction.SKIP:
                result.skipped_count += 1
            elif decision.action == MergeAction.MERGE:
                result.merged_count += 1
            elif decision.action == MergeAction.OVERWRITE:
                result.overwritten_count += 1
            elif decision.action == MergeAction.RENAME:
                result.renamed_count += 1
            elif decision.action == MergeAction.MANUAL:
                result.manual_review_count += 1

        # Generate warnings
        if result.manual_review_count > 0:
            result.warnings.append(
                f"{result.manual_review_count} object(s) require manual review "
                f"(confidence < {self._confidence_threshold})"
            )
        if result.overwritten_count > 0 and not auto_overwrite:
            result.warnings.append(
                f"{result.overwritten_count} object(s) would be overwritten "
                "(enable auto_overwrite to confirm)"
            )

        return result

    def _compute_object_fingerprint(self, obj: Dict[str, Any]) -> str:
        """Compute content fingerprint for an object.

        H9 Fix: Used to detect content changes even when meta_id matches.
        Phase 3 (2026-05-16): Added relationships to fingerprint calculation.
        R11 Migration (2026-05-29): Delegates to EntityDeduplicationService.

        Excludes run-specific metadata like timestamps.
        """
        # R11: Use unified deduplication service with LLM conflict resolver
        dedup_service = EntityDeduplicationService(use_llm_conflict_resolver=True)
        return dedup_service.compute_object_fingerprint(obj)

    def _decide_merge(
        self,
        obj: Dict[str, Any],
        master_by_id: Dict[str, Dict[str, Any]],
        master_by_name: Dict[str, Dict[str, Any]],
        auto_overwrite: bool,
        auto_rename: bool,
    ) -> MergeDecision:
        """Decide merge action for a single object."""
        obj_id = str(obj.get("meta_id") or "").strip()
        obj_name = str(obj.get("meta_name") or "").strip()
        obj_class = str(obj.get("meta_class") or "").strip()

        # Case 1: Exact meta_id duplicate
        # H9 Fix: Compare content fingerprint to detect updates
        if obj_id and obj_id in master_by_id:
            master_obj = master_by_id[obj_id]
            obj_fp = self._compute_object_fingerprint(obj)
            master_fp = self._compute_object_fingerprint(master_obj)

            if obj_fp == master_fp:
                # Content identical - safe to skip
                confidence = self._priorities["duplicate_exact_match"]
                return MergeDecision(
                    meta_id=obj_id,
                    meta_name=obj_name,
                    action=MergeAction.SKIP,
                    confidence=confidence,
                    reason=f"Exact meta_id '{obj_id}' with identical content",
                    source_object=obj,
                    target_object=master_obj,
                )
            else:
                # Content changed - needs overwrite or manual review
                confidence = self._priorities["duplicate_exact_match"]
                if auto_overwrite:
                    return MergeDecision(
                        meta_id=obj_id,
                        meta_name=obj_name,
                        action=MergeAction.OVERWRITE,
                        confidence=confidence,
                        reason=f"Exact meta_id '{obj_id}' with content changes",
                        source_object=obj,
                        target_object=master_obj,
                    )
                else:
                    return MergeDecision(
                        meta_id=obj_id,
                        meta_name=obj_name,
                        action=MergeAction.MANUAL,
                        confidence=confidence,
                        reason=f"Exact meta_id '{obj_id}' but content differs - manual review needed",
                        source_object=obj,
                        target_object=master_obj,
                    )

        # Case 2: Name collision with different class
        name_key = obj_name.lower()
        if name_key and name_key in master_by_name:
            master_obj = master_by_name[name_key]
            master_class = str(
                master_obj.get("meta_class") or master_obj.get("class") or ""
            ).strip()

            if master_class and obj_class and master_class != obj_class:
                # Class conflict
                confidence = self._priorities["conflict_high_confidence"]
                if auto_rename:
                    return MergeDecision(
                        meta_id=obj_id,
                        meta_name=obj_name,
                        action=MergeAction.RENAME,
                        confidence=confidence,
                        reason=f"Class conflict: '{obj_class}' vs '{master_class}'",
                        new_meta_id=obj_id,  # Keep same ID, different name
                        source_object=obj,
                        target_object=master_obj,
                    )
                else:
                    return MergeDecision(
                        meta_id=obj_id,
                        meta_name=obj_name,
                        action=MergeAction.MANUAL,
                        confidence=confidence,
                        reason=f"Class conflict: '{obj_class}' vs '{master_class}' - manual resolution needed",
                        source_object=obj,
                        target_object=master_obj,
                    )

        # Case 3: New object - safe to merge
        confidence = self._priorities["new_object"]
        return MergeDecision(
            meta_id=obj_id,
            meta_name=obj_name,
            action=MergeAction.MERGE,
            confidence=confidence,
            reason="New object (no conflicts detected)",
            source_object=obj,
            target_object={},
        )


# =============================================================================
# Legacy AutoMergeEngine alias for backward compatibility
# =============================================================================

AutoMergeEngine = MergeDecisionEngine


# =============================================================================
# Validation Functions
# =============================================================================


def _load_master_nodes(master_ontology_path: str) -> List[Dict[str, Any]]:
    """Load node records from master_ontology JSON file.

    Supports three shapes:
    - Flat list of node dicts
    - {"nodes": [...]} or {"objects": [...]}
    - {"domains": {"SUPPLY_CHAIN": {"nodes": [...], ...}, ...}}

    Returns an empty list on any error (non-fatal).
    """
    try:
        p = Path(master_ontology_path)
        if not p.exists():
            logger.debug(f"merge_validator: master_ontology not found at {p}, skipping.")
            return []
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.debug(f"merge_validator: failed to load master_ontology: {exc}")
        return []

    nodes: List[Dict[str, Any]] = []
    if isinstance(data, list):
        nodes = data
    elif isinstance(data, dict):
        for key in ("nodes", "objects", "classes"):
            if isinstance(data.get(key), list):
                nodes = data[key]
                break
        if not nodes:
            # Nested domain buckets
            for bucket in (data.get("domains") or {}).values():
                if isinstance(bucket, dict):
                    for key in ("nodes", "objects"):
                        if isinstance(bucket.get(key), list):
                            nodes.extend(bucket[key])
                            break
    return [n for n in nodes if isinstance(n, dict)]


def validate_incremental_merge(
    *,
    l1: Dict[str, Any],
    master_ontology_path: str | None = None,
    auto_merge: bool = False,
    confidence_threshold: float = 0.8,
    auto_overwrite: bool = False,
    auto_rename: bool = True,
    version_service: "Any | None" = None,
) -> Dict[str, Any]:
    """Check L1 sandbox objects against the master ontology for duplicates and conflicts.

    Args:
        l1: L1 domain sandbox payload (as produced by extract_l1_domain_sandbox).
        master_ontology_path: Path to the master ontology JSON file. If None, uses config default.
        auto_merge: If True, performs automatic merge based on confidence scores (P2-13).
        confidence_threshold: Minimum confidence for auto merge actions (0.0-1.0).
        auto_overwrite: Allow automatic overwrites during auto merge.
        auto_rename: Allow automatic renaming during auto merge.

    Returns:
        A report dict:
        {
            "duplicates": [{"meta_id": ..., "meta_name": ...}],   # exact meta_id matches
            "conflicts":  [{"meta_name": ..., "l1_meta_class": ..., "master_meta_class": ...}],
            "new_objects_count": int,    # objects not found in master
            "checked_count": int,        # total L1 objects examined
            "master_node_count": int,    # total master nodes loaded
            "warnings": [...],           # human-readable summary lines
            "auto_merge": {...},         # auto merge results (if auto_merge=True)
        }
    """
    if master_ontology_path is None:
        config = get_config()
        master_ontology_path = config.master_ontology_path
    report: Dict[str, Any] = {
        "duplicates": [],
        "conflicts": [],
        "new_objects_count": 0,
        "checked_count": 0,
        "master_node_count": 0,
        "warnings": [],
    }

    try:
        sandbox = (l1.get("sandbox") or {})
        l1_objects: List[Dict[str, Any]] = [
            o for o in (sandbox.get("objects") or []) if isinstance(o, dict)
        ]
        report["checked_count"] = len(l1_objects)

        if not l1_objects:
            return report

        master_nodes = _load_master_nodes(master_ontology_path)
        report["master_node_count"] = len(master_nodes)

        # Build lookup indices from master
        master_by_id: Dict[str, Dict[str, Any]] = {}
        master_by_name: Dict[str, Dict[str, Any]] = {}
        for node in master_nodes:
            mid = str(node.get("meta_id") or node.get("node_id") or node.get("id") or "").strip()
            mname = str(node.get("meta_name") or node.get("name") or "").strip().lower()
            if mid:
                master_by_id[mid] = node
            if mname:
                master_by_name[mname] = node

        new_count = 0
        for obj in l1_objects:
            obj_id = str(obj.get("meta_id") or "").strip()
            obj_name = str(obj.get("meta_name") or "").strip()
            obj_class = str(obj.get("meta_class") or "").strip()

            # Check exact meta_id duplicate
            if obj_id and obj_id in master_by_id:
                report["duplicates"].append({
                    "meta_id": obj_id,
                    "meta_name": obj_name,
                })
                continue

            # Check name collision with different class (conflict)
            name_key = obj_name.lower()
            if name_key and name_key in master_by_name:
                master_node = master_by_name[name_key]
                master_class = str(
                    master_node.get("meta_class") or master_node.get("class") or ""
                ).strip()
                if master_class and obj_class and master_class != obj_class:
                    report["conflicts"].append({
                        "meta_name": obj_name,
                        "l1_meta_class": obj_class,
                        "master_meta_class": master_class,
                    })
                    continue

            new_count += 1

        report["new_objects_count"] = new_count

        # Build human-readable warnings
        warnings: List[str] = []
        if report["duplicates"]:
            dup_names = [d["meta_name"] or d["meta_id"] for d in report["duplicates"][:5]]
            warnings.append(
                f"{len(report['duplicates'])} duplicate(s) found in master_ontology "
                f"(e.g. {', '.join(dup_names)}). Consider skipping re-ingestion."
            )
        if report["conflicts"]:
            conf_names = [c["meta_name"] for c in report["conflicts"][:5]]
            warnings.append(
                f"{len(report['conflicts'])} class conflict(s) detected "
                f"(e.g. {', '.join(conf_names)}). Review meta_class assignments."
            )
        if new_count == 0 and l1_objects:
            warnings.append(
                "All L1 objects already exist in master_ontology. "
                "This run adds no new ontology coverage."
            )
        report["warnings"] = warnings

    except Exception as exc:
        logger.warning(f"validate_incremental_merge: unexpected error (non-fatal): {exc}")
        report["warnings"] = [f"merge validation error: {exc}"]

    # P2-13: Auto merge mode (P1-8: now uses MergeDecisionEngine from this module)
    if auto_merge:
        try:
            engine = MergeDecisionEngine()
            merge_result = engine.execute_merge(
                l1_objects=l1_objects,
                master_nodes=master_nodes,
                confidence_threshold=confidence_threshold,
                auto_overwrite=auto_overwrite,
                auto_rename=auto_rename,
            )
            report["auto_merge"] = merge_result.to_dict()
            report["auto_merge"]["enabled"] = True

            # C-2: Register entity versions for merged/overwritten objects
            if version_service is not None:
                for decision in merge_result.decisions:
                    if decision.action in {MergeAction.MERGE, MergeAction.OVERWRITE}:
                        entity_id = str(decision.meta_id or decision.meta_name or "")
                        if entity_id:
                            try:
                                version_service.register_entity_version(
                                    entity_id=entity_id,
                                    entity_type="OntologyEntity",
                                    entity_data=decision.source_object,
                                    metadata={"merge_action": decision.action.value, "reason": decision.reason},
                                )
                            except Exception as _ve:
                                logger.warning(f"version tracking failed for {entity_id}: {_ve}")
        except Exception as exc:
            logger.warning(f"validate_incremental_merge: auto_merge failed (non-fatal): {exc}")
            report["auto_merge"] = {"enabled": False, "error": str(exc)}

    return report
