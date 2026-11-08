#!/usr/bin/env python3
"""Lightweight relationship rules loader - loads YAML config and provides validation"""

from functools import lru_cache
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

_RELATION_RULES_PATH = Path(__file__).parent.parent / "config" / "ontology" / "relationship_rules.yaml"


class RelationshipRulesLoader:
    def __init__(self, rules_path: Optional[Path] = None):
        self.rules_path = Path(rules_path) if rules_path else _RELATION_RULES_PATH
        self._rules = {}
        self._family_map = {}
        self._layer_rules = {}
        self._evidence_rules = {}
        self._load_rules()

    def _load_rules(self):
        if not self.rules_path.exists():
            return

        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            self._rules = config.get("relation_rules", {})
            self._family_map = config.get("relation_family_map", {})
            self._layer_rules = config.get("layer_pair_rules", {})
            self._evidence_rules = config.get("evidence_requirements", {})

        except Exception:
            pass

    def get_family(self, rel_type: str) -> str:
        rel_type_upper = str(rel_type or "").strip().upper()
        for family, types in self._family_map.items():
            if rel_type_upper in [t.upper() for t in types]:
                return family
        return "GENERAL"

    def get_family_for_relation(self, rel_type: str) -> str:
        return self.get_family(rel_type)

    @property
    def layer_pair_rules(self) -> Dict[str, Any]:
        return self._layer_rules

    def get_rule(self, rel_type: str) -> Optional[Dict[str, Any]]:
        rel_type_upper = str(rel_type or "").strip().upper()
        for key, rule in self._rules.items():
            if key.upper() == rel_type_upper:
                return rule
        return None

    def validate(
        self,
        rel_type: str,
        source_type: str,
        target_type: str,
        rel: Dict[str, Any],
    ) -> Tuple[bool, str, str]:
        rel_type_upper = str(rel_type or "").strip().upper()
        rule = self.get_rule(rel_type_upper)

        if not rule:
            return True, "", "info"

        severity = rule.get("severity", "warn")
        notes = rule.get("notes", "")

        allowed_source_types = rule.get("allowed_source_types", [])
        if allowed_source_types and source_type:
            source_type_upper = source_type.upper()
            allowed_upper = [t.upper() for t in allowed_source_types]
            if source_type_upper not in allowed_upper:
                reason = f"{rel_type} source should be {', '.join(allowed_source_types)}, got {source_type or 'unknown'}"
                if notes:
                    reason += f" ({notes})"
                return False, reason, severity

        allowed_target_types = rule.get("allowed_target_types", [])
        if allowed_target_types and target_type:
            target_type_upper = target_type.upper()
            allowed_upper = [t.upper() for t in allowed_target_types]
            if target_type_upper not in allowed_upper:
                reason = f"{rel_type} target should be {', '.join(allowed_target_types)}, got {target_type or 'unknown'}"
                if notes:
                    reason += f" ({notes})"
                return False, reason, severity

        required_fields = rule.get("required_fields", [])
        if required_fields:
            has_any = any(str(rel.get(field) or "").strip() for field in required_fields)
            if not has_any:
                reason = f"{rel_type} should carry {', '.join(required_fields)}"
                if notes:
                    reason += f" ({notes})"
                return False, reason, severity

        return True, "", "info"

    def check_layer_pair(self, source_layer: str, target_layer: str) -> Tuple[bool, str]:
        pair = f"{source_layer}-{target_layer}"
        if pair in self._layer_rules:
            rule = self._layer_rules[pair]
            if not rule.get("allowed", True):
                return False, f"Layer pair {pair} is not allowed"
            if rule.get("requires_evidence", False):
                return True, f"Layer pair {pair} requires evidence"
        return True, ""

    def get_rules(self) -> Dict[str, Any]:
        return self._rules

    def get_family_map(self) -> Dict[str, Any]:
        return self._family_map


@lru_cache(maxsize=1)
def get_relationship_rules() -> RelationshipRulesLoader:
    return RelationshipRulesLoader()