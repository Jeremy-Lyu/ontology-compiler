"""Schema Registry — Ontology schema single source of truth loader.

[W1 / Plan A 2026-06-20]: Introduce schemas/ontology_schema.yaml as single source of truth,
unifying scattered hardcoded main/subclass lists across the project, supporting legacy alias normalization.

Design goals:
- Main classes, subclasses, relation types defined in one place
- Legacy artifacts (including L1_Domain / Equipment_Infrastructure etc.) can still be read
- Provide query APIs like normalize_subclass / is_valid_main_class
- Version management with migration support for schema evolution

[VERSION-MIGRATION]: Added schema version management and incremental migration support.
- Version number tracking (semver: major.minor.patch)
- Migration scripts registry for automatic field transformation
- Entity schema compatibility checking
- Automatic field migration for legacy entities

Typical usage:
    from core.schema_registry import get_schema_registry

    reg = get_schema_registry()
    reg.is_valid_main_class("Capability")              # True
    reg.is_valid_subclass("Capability", "Domain")      # True
    reg.normalize_subclass("L1_Domain")                # "Domain"
    reg.list_subclasses("Asset")                       # ["Product_Definition", ...]
    
    # Version management
    reg.version                                        # "1.2.0"
    reg.is_compatible("1.1.0")                        # True
    migrated_entity = reg.migrate_entity(legacy_entity, "1.1.0")
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

import yaml


_DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schemas" / "ontology_schema.yaml"
)


class SchemaRegistry:
    """Load and query ontology schema single source of truth."""

    def __init__(self, schema_path: Optional[Path] = None) -> None:
        self.schema_path = Path(schema_path) if schema_path else _DEFAULT_SCHEMA_PATH
        self._data: Dict = self._load()

    def _load(self) -> Dict:
        if not self.schema_path.exists():
            raise FileNotFoundError(
                f"Ontology schema file not found: {self.schema_path}"
            )
        with self.schema_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        data.setdefault("main_classes", [])
        data.setdefault("subclasses", {})
        data.setdefault("legacy_subclass_aliases", {})
        data.setdefault("legacy_main_class_aliases", {})
        data.setdefault("relation_types", [])
        data.setdefault("relation_families", {})
        data.setdefault("legacy_relation_type_aliases", {})
        return data

    def normalize_main_class(self, main_class: str) -> str:
        """Normalize legacy alias (e.g., scenario) to standard main class name (BusinessScenario).

        Returns original if not a legacy alias, caller can continue with validation.
        Supports case-insensitive matching.
        Also handles subclass names used as main class input (e.g., L1_Domain -> Capability).
        """
        if not main_class:
            return main_class

        main_class = main_class.strip()

        if main_class in self._data["legacy_main_class_aliases"]:
            return self._data["legacy_main_class_aliases"][main_class]

        for valid_class in self._data["main_classes"]:
            if valid_class.lower() == main_class.lower():
                return valid_class

        if main_class in self._data["legacy_subclass_aliases"]:
            normalized_subclass = self._data["legacy_subclass_aliases"][main_class]
            for mc, subclasses in self._data["subclasses"].items():
                if normalized_subclass in subclasses:
                    return mc

        return main_class

    def is_valid_main_class(self, name: str) -> bool:
        normalized = self.normalize_main_class(name)
        return normalized in self._data["main_classes"]

    @property
    def legacy_main_class_aliases(self) -> Dict[str, str]:
        return dict(self._data["legacy_main_class_aliases"])

    @property
    def main_classes(self) -> List[str]:
        return list(self._data["main_classes"])

    def list_subclasses(self, main_class: str) -> List[str]:
        return list(self._data["subclasses"].get(main_class, []))

    def is_valid_subclass(self, main_class: str, subclass: str) -> bool:
        normalized = self.normalize_subclass(subclass, main_class)
        return normalized in self._data["subclasses"].get(main_class, [])

    def normalize_subclass(self, subclass: str, main_class: str = "") -> str:
        """Normalize legacy alias (e.g., L1_Domain) to standard subclass name (Domain).

        Returns original if not a legacy alias, caller can continue with validation.
        Supports case-insensitive matching.
        """
        if not subclass:
            return subclass

        subclass = subclass.strip()

        if subclass in self._data["legacy_subclass_aliases"]:
            return self._data["legacy_subclass_aliases"][subclass]

        if main_class:
            main_class = self.normalize_main_class(main_class)
            for valid_subclass in self._data["subclasses"].get(main_class, []):
                if valid_subclass.lower() == subclass.lower():
                    return valid_subclass

        return subclass

    @property
    def legacy_aliases(self) -> Dict[str, str]:
        return dict(self._data["legacy_subclass_aliases"])

    @property
    def relation_types(self) -> List[str]:
        return list(self._data["relation_types"])

    def normalize_relation_type(self, name: str) -> str:
        """Normalize relation type, supports case-insensitive matching and legacy aliases."""
        if not name:
            return name
        name = name.strip()
        
        if name in self._data["legacy_relation_type_aliases"]:
            return self._data["legacy_relation_type_aliases"][name]
        
        for alias, normalized in self._data["legacy_relation_type_aliases"].items():
            if alias.lower() == name.lower():
                return normalized
        
        for valid_type in self._data["relation_types"]:
            if valid_type.lower() == name.lower():
                return valid_type
        return name

    def is_valid_relation_type(self, name: str) -> bool:
        if not name:
            return False
        name = name.strip()
        
        if name in self._data["legacy_relation_type_aliases"]:
            return True
        
        for alias in self._data["legacy_relation_type_aliases"]:
            if alias.lower() == name.lower():
                return True
        
        for valid_type in self._data["relation_types"]:
            if valid_type.lower() == name.lower():
                return True
        return False

    @property
    def version(self) -> str:
        return str(self._data.get("version", "0.0.0"))

    @property
    def relation_families(self) -> Dict:
        return dict(self._data.get("relation_families", {}))

    def get_relation_family(self, relation_type: str) -> Optional[str]:
        """Get the family name for a given relation type."""
        normalized_type = self.normalize_relation_type(relation_type)
        for family_name, family_data in self._data.get("relation_families", {}).items():
            if normalized_type in family_data.get("types", []):
                return family_name
        return None

    def get_family_priority(self, family_name: str) -> int:
        """Get the priority of a relation family."""
        family = self._data.get("relation_families", {}).get(family_name, {})
        return family.get("priority", 999)

    def get_family_types(self, family_name: str) -> List[str]:
        """Get all relation types in a family."""
        family = self._data.get("relation_families", {}).get(family_name, {})
        return list(family.get("types", []))

    def get_family_description(self, family_name: str) -> str:
        """Get the description of a relation family."""
        family = self._data.get("relation_families", {}).get(family_name, {})
        return family.get("description", "")

    # ------------------------------------------------------------------ #
    # [VERSION-MIGRATION] Version Management and Migration
    # ------------------------------------------------------------------ #

    def _parse_version(self, version_str: str) -> tuple[int, int, int]:
        """Parse semantic version string to tuple (major, minor, patch)."""
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version_str.strip())
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return (0, 0, 0)

    def is_compatible(self, entity_version: str) -> bool:
        """Check if an entity from a specific schema version is compatible with current version.
        
        Compatibility rules:
        - Same major version: compatible (backward compatible changes)
        - Different major version: incompatible (breaking changes)
        
        Args:
            entity_version: The schema version the entity was created with
        
        Returns:
            True if compatible, False otherwise
        """
        current = self._parse_version(self.version)
        entity = self._parse_version(entity_version)
        
        return current[0] == entity[0]

    def get_version_diff(self, from_version: str, to_version: str = None) -> dict[str, Any]:
        """Get the difference between two versions."""
        to_version = to_version or self.version
        from_v = self._parse_version(from_version)
        to_v = self._parse_version(to_version)
        
        is_compatible = from_v[0] == to_v[0]
        needs_migration = from_v < to_v
        
        return {
            "from_version": from_version,
            "to_version": to_version,
            "is_compatible": is_compatible,
            "needs_migration": needs_migration,
            "major_changed": from_v[0] != to_v[0],
            "minor_changed": from_v[1] != to_v[1],
            "patch_changed": from_v[2] != to_v[2],
        }

    def _migrations(self) -> dict[str, Callable[[dict], dict]]:
        """Registry of migration functions for each version upgrade.
        
        Key: version string to migrate TO (e.g., "1.1.0", "1.2.0")
        Value: Migration function that transforms an entity dict
        """
        return {
            "1.1.0": self._migrate_to_1_1_0,
            "1.2.0": self._migrate_to_1_2_0,
        }

    def _migrate_to_1_1_0(self, entity: dict) -> dict:
        """Migrate entity from version 1.0.x to 1.1.0.
        
        Changes in 1.1.0:
        - Added layer_scope field (spec/impl)
        - Added meta_subclass field
        """
        migrated = entity.copy()
        
        if "layer_scope" not in migrated or not migrated["layer_scope"]:
            layer = str(migrated.get("layer", "")).lower()
            if layer.startswith("l0") or layer.startswith("l1"):
                migrated["layer_scope"] = "spec"
            elif layer.startswith("l3"):
                migrated["layer_scope"] = "impl"
            else:
                migrated["layer_scope"] = "spec"
        
        if "meta_subclass" not in migrated or not migrated["meta_subclass"]:
            migrated["meta_subclass"] = ""
        
        return migrated

    def _migrate_to_1_2_0(self, entity: dict) -> dict:
        """Migrate entity from version 1.1.x to 1.2.0.
        
        Changes in 1.2.0:
        - Added description field (alias for meta_description)
        - Added status field with default value "draft"
        """
        migrated = self._migrate_to_1_1_0(entity)
        
        if "description" not in migrated or not migrated["description"]:
            migrated["description"] = migrated.get("meta_description", "")
        
        if "status" not in migrated or not migrated["status"]:
            migrated["status"] = "draft"
        
        return migrated

    def migrate_entity(self, entity: dict, entity_version: str) -> dict:
        """Migrate an entity from its original schema version to the current version.
        
        Args:
            entity: The entity dict to migrate
            entity_version: The schema version the entity was created with
        
        Returns:
            Migrated entity dict with all current version fields
        """
        if not entity:
            return entity
        
        current_version = self.version
        
        if entity_version == current_version:
            return entity.copy()
        
        from_v = self._parse_version(entity_version)
        to_v = self._parse_version(current_version)
        
        if from_v >= to_v:
            return entity.copy()
        
        migrations = self._migrations()
        migrated = entity.copy()
        
        sorted_versions = sorted(migrations.keys(), key=self._parse_version)
        
        for target_version in sorted_versions:
            target_v = self._parse_version(target_version)
            if from_v < target_v <= to_v:
                migrated = migrations[target_version](migrated)
        
        return migrated

    def validate_entity_compatibility(self, entity: dict) -> tuple[bool, list[str]]:
        """Validate entity compatibility with current schema.
        
        Checks:
        - Required fields exist (id/code/name/layer/type/status)
        - Valid main_class and subclass
        - Valid relation types (if any)
        
        Args:
            entity: Entity dict to validate
        
        Returns:
            (is_valid, [errors])
        """
        errors = []
        
        required_fields = ["meta_id", "meta_name", "meta_class", "layer"]
        for field in required_fields:
            if not entity.get(field):
                errors.append(f"Missing required field: {field}")
        
        main_class = entity.get("meta_class", "")
        if main_class and not self.is_valid_main_class(main_class):
            errors.append(f"Invalid main_class: {main_class}")
        
        subclass = entity.get("meta_subclass", "")
        if subclass and main_class:
            if not self.is_valid_subclass(main_class, subclass):
                errors.append(f"Invalid subclass '{subclass}' for main_class '{main_class}'")
        
        relationships = entity.get("relationships", [])
        for rel in relationships:
            if isinstance(rel, dict):
                rel_type = rel.get("type") or rel.get("relation_type") or rel.get("predicate")
                if rel_type and not self.is_valid_relation_type(rel_type):
                    errors.append(f"Invalid relation type: {rel_type}")
        
        return (len(errors) == 0, errors)

    def get_entity_schema_requirements(self, main_class: str) -> dict[str, Any]:
        """Get schema requirements for a specific entity type.
        
        Args:
            main_class: The entity main class
        
        Returns:
            Dict with required_fields, optional_fields, and valid_subclasses
        """
        normalized_class = self.normalize_main_class(main_class)
        
        return {
            "main_class": normalized_class,
            "required_fields": ["meta_id", "meta_name", "meta_class", "layer"],
            "optional_fields": [
                "meta_subclass", "description", "meta_description", 
                "status", "layer_scope", "domain", "parent_id",
                "properties", "relationships", "evidence",
            ],
            "valid_subclasses": self.list_subclasses(normalized_class),
            "valid_relation_types": self.relation_types,
        }


@lru_cache(maxsize=1)
def get_schema_registry() -> SchemaRegistry:
    """Get global singleton SchemaRegistry (lazy loading, cached within process)."""
    return SchemaRegistry()
