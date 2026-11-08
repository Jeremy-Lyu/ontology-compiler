"""Change detection service based on entity fingerprints.

P8 Optimization (2026-06-09): Implements fingerprint-based change detection
to identify modified, added, and deleted entities between versions.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ChangeDetector:
    """
    Detects changes between entity sets using fingerprint-based comparison.
    
    Features:
    - Entity fingerprint generation using content hashing
    - Change detection (added, modified, deleted)
    - Change type classification (metadata, content, relationships)
    - Change impact assessment
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize change detector.
        
        Args:
            cache_dir: Directory for storing fingerprint history
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_fingerprint(self, entity: Dict[str, Any]) -> str:
        """
        Generate a unique fingerprint for an entity.

        Fingerprint is based on:
        - meta_name (primary identifier)
        - meta_class
        - description
        - attributes (sorted keys)
        - relationships (sorted by target)

        Args:
            entity: Entity dictionary or Pydantic BaseModel

        Returns:
            SHA256 hex digest fingerprint
        """
        def _get(obj, key, default=""):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        components = [
            _get(entity, "meta_name", ""),
            _get(entity, "meta_class", ""),
            _get(entity, "description", ""),
        ]

        attributes = _get(entity, "attributes", {})
        if isinstance(attributes, dict):
            for key in sorted(attributes.keys()):
                components.append(f"{key}={str(attributes[key])}")

        relationships = _get(entity, "relationships", [])
        if isinstance(relationships, list):
            for rel in sorted(relationships, key=lambda x: _get(x, "target", "")):
                components.append(f"{_get(rel, 'type', '')}->{_get(rel, 'target', '')}")

        content = "|".join(components)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def detect_changes(
        self,
        previous_entities: List[Dict[str, Any]],
        current_entities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Detect changes between two sets of entities.
        
        Args:
            previous_entities: List of entities from previous version
            current_entities: List of entities from current version
            
        Returns:
            Change report with added, modified, deleted, and unchanged entities
        """
        def _get_meta_name(e):
            if isinstance(e, dict):
                return e.get("meta_name", "")
            return getattr(e, "meta_name", "")

        previous_map = {_get_meta_name(e): e for e in previous_entities}
        current_map = {_get_meta_name(e): e for e in current_entities}
        
        previous_names = set(previous_map.keys())
        current_names = set(current_map.keys())
        
        added = [current_map[name] for name in current_names - previous_names]
        deleted = [previous_map[name] for name in previous_names - current_names]
        
        modified: List[Dict[str, Any]] = []
        unchanged: List[Dict[str, Any]] = []
        
        for name in previous_names & current_names:
            prev_entity = previous_map[name]
            curr_entity = current_map[name]
            
            prev_fp = self.generate_fingerprint(prev_entity)
            curr_fp = self.generate_fingerprint(curr_entity)
            
            if prev_fp != curr_fp:
                change_type, details = self._classify_change(prev_entity, curr_entity)
                modified.append({
                    "entity": curr_entity,
                    "change_type": change_type,
                    "change_details": details,
                    "previous_fingerprint": prev_fp,
                    "current_fingerprint": curr_fp,
                })
            else:
                unchanged.append(curr_entity)
        
        impact_assessment = self._assess_impact(added, modified, deleted)
        
        return {
            "added": added,
            "modified": modified,
            "deleted": deleted,
            "unchanged": unchanged,
            "stats": {
                "total_previous": len(previous_entities),
                "total_current": len(current_entities),
                "added_count": len(added),
                "modified_count": len(modified),
                "deleted_count": len(deleted),
                "unchanged_count": len(unchanged),
                "change_ratio": (len(added) + len(modified) + len(deleted)) / max(len(previous_entities), 1),
            },
            "impact": impact_assessment,
            "detected_at": datetime.now().isoformat(),
        }

    def _classify_change(
        self,
        previous: Dict[str, Any],
        current: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Classify the type of change between two entity versions.

        Change types:
        - metadata: meta_class, domain, category changed
        - content: description, definition changed
        - attributes: attribute values changed
        - relationships: relationship added/removed/modified
        - mixed: multiple change types

        Args:
            previous: Previous entity state (dict or Pydantic BaseModel)
            current: Current entity state (dict or Pydantic BaseModel)

        Returns:
            (change_type, details) tuple
        """
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        changes: List[str] = []
        details: Dict[str, Any] = {}

        if _get(previous, "meta_class") != _get(current, "meta_class"):
            changes.append("metadata")
            details["meta_class"] = {
                "previous": _get(previous, "meta_class"),
                "current": _get(current, "meta_class"),
            }

        if _get(previous, "domain") != _get(current, "domain"):
            changes.append("metadata")
            details["domain"] = {
                "previous": _get(previous, "domain"),
                "current": _get(current, "domain"),
            }

        if _get(previous, "description") != _get(current, "description"):
            changes.append("content")
            details["description"] = {
                "previous": _get(previous, "description"),
                "current": _get(current, "description"),
            }

        prev_attrs = _get(previous, "attributes", {})
        curr_attrs = _get(current, "attributes", {})
        if prev_attrs != curr_attrs:
            changes.append("attributes")
            details["attributes"] = {
                "added": [k for k in curr_attrs if k not in prev_attrs],
                "removed": [k for k in prev_attrs if k not in curr_attrs],
                "modified": [k for k in curr_attrs if k in prev_attrs and prev_attrs[k] != curr_attrs[k]],
            }

        prev_rels = _get(previous, "relationships", [])
        curr_rels = _get(current, "relationships", [])
        prev_rel_keys = set((_get(r, "type"), _get(r, "target")) for r in prev_rels)
        curr_rel_keys = set((_get(r, "type"), _get(r, "target")) for r in curr_rels)
        if prev_rel_keys != curr_rel_keys:
            changes.append("relationships")
            details["relationships"] = {
                "added": len(curr_rel_keys - prev_rel_keys),
                "removed": len(prev_rel_keys - curr_rel_keys),
            }

        if len(changes) == 0:
            change_type = "unchanged"
        elif len(changes) == 1:
            change_type = changes[0]
        else:
            change_type = "mixed"

        return change_type, details

    def _assess_impact(
        self,
        added: List[Dict[str, Any]],
        modified: List[Dict[str, Any]],
        deleted: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Assess the impact of detected changes.
        
        Impact levels:
        - low: Only metadata changes, no structural impact
        - medium: Content or attribute changes affecting data quality
        - high: Relationship or entity changes affecting ontology structure
        - critical: Core entities deleted or modified
        
        Args:
            added: List of added entities
            modified: List of modified entities
            deleted: List of deleted entities
            
        Returns:
            Impact assessment dictionary
        """
        def _get_mc(obj):
            if isinstance(obj, dict):
                return obj.get("meta_class")
            return getattr(obj, "meta_class", None)

        critical_entities = ["System", "Capability", "Process", "Data", "Actor", "Asset"]

        critical_deleted = sum(1 for e in deleted if _get_mc(e) in critical_entities)
        critical_modified = sum(1 for m in modified if _get_mc(m.get("entity")) in critical_entities)
        relationship_changes = sum(1 for m in modified if m.get("change_type") in ["relationships", "mixed"])
        
        if critical_deleted > 0 or (critical_modified > 0 and relationship_changes > 0):
            level = "critical"
        elif relationship_changes > 0 or len(deleted) > 0:
            level = "high"
        elif len(modified) > len(added):
            level = "medium"
        elif len(added) > 0:
            level = "low"
        else:
            level = "none"
        
        return {
            "level": level,
            "critical_deleted_count": critical_deleted,
            "critical_modified_count": critical_modified,
            "relationship_changes_count": relationship_changes,
            "added_count": len(added),
            "modified_count": len(modified),
            "deleted_count": len(deleted),
            "recommendations": self._generate_recommendations(level, added, modified, deleted),
        }

    def _generate_recommendations(
        self,
        level: str,
        added: List[Dict[str, Any]],
        modified: List[Dict[str, Any]],
        deleted: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Generate recommendations based on change impact.
        
        Args:
            level: Impact level
            added: List of added entities
            modified: List of modified entities
            deleted: List of deleted entities
            
        Returns:
            List of recommendation strings
        """
        recommendations: List[str] = []
        
        if level == "critical":
            recommendations.append("CRITICAL: Review deleted/modified core entities immediately")
            recommendations.append("Run full ontology validation to check structural integrity")
            recommendations.append("Verify no orphaned relationships exist")
        
        elif level == "high":
            recommendations.append("Review relationship changes for consistency")
            recommendations.append("Run ontology validation to check for broken references")
            recommendations.append("Update any downstream dependencies")
        
        elif level == "medium":
            recommendations.append("Review content/attribute changes for accuracy")
            recommendations.append("Check quality scores for impacted entities")
            recommendations.append("Consider re-running RIGOR for modified entities")
        
        elif level == "low":
            recommendations.append("New entities detected - verify classification")
            recommendations.append("Check for duplicate entities")
        
        return recommendations

    def save_fingerprints(
        self,
        entities: List[Dict[str, Any]],
        version: str,
        context: str = "",
    ) -> None:
        """
        Save entity fingerprints to cache for future comparison.
        
        Args:
            entities: List of entities to fingerprint
            version: Version identifier
            context: Optional context (domain, project, etc.)
        """
        if not self.cache_dir:
            logger.warning("No cache directory configured, skipping save")
            return
        
        fingerprints = {}
        for entity in entities:
            if isinstance(entity, dict):
                name = entity.get("meta_name", "")
            else:
                name = getattr(entity, "meta_name", "")
            if name:
                fingerprints[name] = {
                    "fingerprint": self.generate_fingerprint(entity),
                    "entity": entity,
                    "version": version,
                    "saved_at": datetime.now().isoformat(),
                }
        
        cache_file = self.cache_dir / f"fingerprints_{context}_{version}.json"
        import json
        
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(fingerprints, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(fingerprints)} fingerprints to {cache_file}")

    def load_fingerprints(
        self,
        version: str,
        context: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load previously saved fingerprints from cache.
        
        Args:
            version: Version identifier
            context: Optional context (domain, project, etc.)
            
        Returns:
            Dictionary mapping entity name to fingerprint data
        """
        if not self.cache_dir:
            return {}
        
        cache_file = self.cache_dir / f"fingerprints_{context}_{version}.json"
        
        if not cache_file.exists():
            logger.debug(f"Cache file not found: {cache_file}")
            return {}
        
        import json
        
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load fingerprints: {e}")
            return {}

    def compare_with_version(
        self,
        current_entities: List[Dict[str, Any]],
        previous_version: str,
        context: str = "",
    ) -> Dict[str, Any]:
        """
        Compare current entities with a previously saved version.
        
        Args:
            current_entities: Current entities to compare
            previous_version: Version identifier to compare against
            context: Optional context (domain, project, etc.)
            
        Returns:
            Change report
        """
        saved_data = self.load_fingerprints(previous_version, context)
        previous_entities = [data["entity"] for data in saved_data.values()]
        
        return self.detect_changes(previous_entities, current_entities)

    @staticmethod
    def calculate_entity_hash(entity: Dict[str, Any]) -> str:
        """Calculate SHA256 hash of entity content."""
        import json
        content = json.dumps(entity, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def group_changes_by_type(changes: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Group changes by their change type."""
        result: Dict[str, List[Dict[str, Any]]] = {
            "metadata": [],
            "content": [],
            "attributes": [],
            "relationships": [],
            "mixed": [],
        }
        
        for modified in changes.get("modified", []):
            change_type = modified.get("change_type", "mixed")
            if change_type in result:
                result[change_type].append(modified)
        
        return result