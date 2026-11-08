"""Entity Classification Rule Engine.

P3-04 Enhancement (2026-07-05):
Loads classification rules from config/ontology/entity_classification_unified.yaml
and executes them to classify entities based on keywords and regex patterns.

Usage:
    from src.infrastructure.services.entity_classification_rule_engine import EntityClassificationRuleEngine

    engine = EntityClassificationRuleEngine()
    result = engine.classify("产品开发能力", "Capability")
    # Returns: {"type": "Capability", "layer": "L0", "sub_type": "Domain", "confidence": 0.95, "matched_rule": "rule_capability_primary"}
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from tests.fixtures.classification.classification_rules import classify_entity as fixture_classify_entity
except Exception:
    fixture_classify_entity = None

from src.infrastructure.config.unified_config import get_unified_config

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of entity classification."""

    entity_type: str
    layer: str
    sub_type: Optional[str] = None
    confidence: float = 0.0
    matched_rule_id: Optional[str] = None
    matched_rule_description: Optional[str] = None
    needs_review: bool = False
    fallback_decision: bool = False


class EntityClassificationRuleEngine:
    """Rule engine for entity classification based on YAML-configured rules."""

    def __init__(self):
        self._rules: List[Dict[str, Any]] = []
        self._conflict_resolution: Dict[str, Any] = {}
        self._validation: Dict[str, Any] = {}
        self._load_rules()

    def _load_rules(self) -> None:
        """Load classification rules from configuration."""
        config = get_unified_config()
        rules_data = config.classification_rules or {}

        self._rules = rules_data.get("rules", [])
        self._conflict_resolution = rules_data.get("conflict_resolution", {})
        self._validation = rules_data.get("validation", {})

        if self._rules:
            self._rules.sort(key=lambda r: r.get("priority", 0), reverse=True)
            logger.info(f"Loaded {len(self._rules)} classification rules (sorted by priority)")
        else:
            logger.warning("No classification rules found in configuration")

    def classify(
        self,
        name: str,
        text: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> ClassificationResult:
        """Classify an entity based on its name and text content.

        Args:
            name: Entity name
            text: Additional text context (optional)
            entity_id: Entity identifier (optional)

        Returns:
            ClassificationResult with type, layer, sub_type, and confidence
        """
        if not name:
            return self._create_fallback_result()

        matched_rules = self._find_matching_rules(name, text, entity_id)

        if not matched_rules:
            return self._create_fallback_result()

        result = self._resolve_conflicts(matched_rules)
        result = self._apply_validation(result)

        return result

    def classify_with_distribution(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Classify an entity and return comparable distribution metadata."""
        name = str(entity.get("meta_name") or entity.get("name") or "")
        text = str(entity.get("description") or entity.get("text") or "")
        entity_id = entity.get("meta_id") or entity.get("entity_id") or entity.get("id")

        result = self.classify(name=name, text=text, entity_id=entity_id)
        output = dict(entity)
        output["meta_class"] = result.entity_type
        output["layer"] = result.layer
        output["entity_type"] = entity.get("entity_type") or result.entity_type
        output["classification_confidence"] = result.confidence
        output["needs_review"] = result.needs_review
        output["fallback_decision"] = result.fallback_decision
        output["classification_source"] = "rule_engine"
        return output

    def compare_with_fixture_provider(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Compare rule-engine output with fixture fallback/provider behavior."""
        rule_result = self.classify_with_distribution(entity)
        fixture_result = self._classify_fixture(entity)
        return {
            "rule_engine": rule_result,
            "fixture_provider": fixture_result,
            "schema_consistent": self._is_schema_consistent(rule_result),
            "fallback_consistent": self._is_schema_consistent(fixture_result),
            "same_class": rule_result.get("meta_class") == fixture_result.get("meta_class"),
            "same_layer": rule_result.get("layer") == fixture_result.get("layer"),
        }

    def _find_matching_rules(
        self,
        name: str,
        text: Optional[str],
        entity_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Find all rules that match the entity."""
        matched = []

        for rule in self._rules:
            if self._rule_matches(rule, name, text, entity_id):
                matched.append(rule)

        return matched

    def _rule_matches(
        self,
        rule: Dict[str, Any],
        name: str,
        text: Optional[str],
        entity_id: Optional[str],
    ) -> bool:
        """Check if a rule matches the entity."""
        match_conditions = rule.get("match", {}).get("any", [])

        if not match_conditions:
            return False

        name_lower = name.lower() if name else ""
        text_lower = text.lower() if text else ""
        entity_id_str = str(entity_id).lower() if entity_id else ""

        for condition in match_conditions:
            field_name = condition.get("field")
            if not field_name:
                continue

            field_value = ""
            if field_name == "text":
                field_value = text_lower
            elif field_name == "name":
                field_value = name_lower
            elif field_name == "entity_id":
                field_value = entity_id_str

            contains_any = condition.get("contains_any", [])
            regex_pattern = condition.get("regex")
            exists = condition.get("exists")

            if contains_any:
                if any(keyword.lower() in field_value for keyword in contains_any):
                    return True

            if regex_pattern:
                try:
                    if re.search(regex_pattern, field_value, re.IGNORECASE):
                        return True
                except re.error:
                    logger.warning(f"Invalid regex pattern in rule {rule.get('id')}: {regex_pattern}")

            if exists is not None:
                if exists and field_value:
                    return True
                elif not exists and not field_value:
                    return True

        return False

    def _resolve_conflicts(self, matched_rules: List[Dict[str, Any]]) -> ClassificationResult:
        """Resolve conflicts between multiple matched rules."""
        strategy = self._conflict_resolution.get("strategy", "highest_priority")

        if strategy == "highest_priority":
            best_rule = max(matched_rules, key=lambda r: r.get("priority", 0))
            return self._rule_to_result(best_rule)

        elif strategy == "highest_confidence":
            best_rule = max(matched_rules, key=lambda r: r.get("confidence", 0))
            return self._rule_to_result(best_rule)

        elif strategy == "llm_vote":
            best_rule = max(matched_rules, key=lambda r: r.get("priority", 0))
            result = self._rule_to_result(best_rule)
            if len(matched_rules) > 1:
                result.needs_review = True
            return result

        else:
            best_rule = max(matched_rules, key=lambda r: r.get("priority", 0))
            return self._rule_to_result(best_rule)

    def _rule_to_result(self, rule: Dict[str, Any]) -> ClassificationResult:
        """Convert a rule to ClassificationResult."""
        result = rule.get("result", {})
        return ClassificationResult(
            entity_type=result.get("type", "Concept"),
            layer=result.get("layer", "L0"),
            sub_type=result.get("sub_type"),
            confidence=result.get("confidence", 0.5),
            matched_rule_id=rule.get("id"),
            matched_rule_description=rule.get("description"),
        )

    def _apply_validation(self, result: ClassificationResult) -> ClassificationResult:
        """Apply validation rules to the result."""
        min_confidence = self._conflict_resolution.get("min_confidence_to_auto_accept", 0.85)

        if result.confidence < min_confidence:
            result.needs_review = True

        auto_reject = self._validation.get("auto_reject_if", [])
        for condition in auto_reject:
            if isinstance(condition, str):
                if condition == "missing_name":
                    result.needs_review = True
                elif condition == "missing_evidence":
                    result.needs_review = True
            elif isinstance(condition, dict):
                if "confidence_below" in condition:
                    if result.confidence < condition["confidence_below"]:
                        result.needs_review = True

        manual_review = self._validation.get("manual_review_if", [])
        for condition in manual_review:
            if isinstance(condition, dict):
                if "confidence_between" in condition:
                    low, high = condition["confidence_between"]
                    if low <= result.confidence <= high:
                        result.needs_review = True

        return result

    def _create_fallback_result(self) -> ClassificationResult:
        """Create a fallback result when no rules match."""
        return ClassificationResult(
            entity_type="Concept",
            layer="L0",
            sub_type="Domain",
            confidence=0.5,
            needs_review=True,
            fallback_decision=True,
        )

    def _classify_fixture(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        if fixture_classify_entity is None:
            return {
                **entity,
                "meta_class": entity.get("meta_class") or "UNKNOWN",
                "layer": entity.get("layer") or "L0",
                "entity_type": entity.get("entity_type") or entity.get("meta_class") or "Concept",
                "classification_confidence": entity.get("classification_confidence", 0.5),
                "needs_review": True,
                "fallback_decision": True,
                "classification_source": "fixture_unavailable",
            }
        return fixture_classify_entity(entity)

    def _is_schema_consistent(self, entity: Dict[str, Any]) -> bool:
        allowed_layers = {"L0", "L1", "L2", "L2.5", "L3"}
        allowed_classes = {rule.get("result", {}).get("type") for rule in self._rules if rule.get("result")}
        allowed_classes.update({"Capability", "System", "Actor", "Asset", "Data", "Process", "Concept", "UNKNOWN"})
        return entity.get("layer") in allowed_layers and entity.get("meta_class") in allowed_classes

    def get_rules_for_layer(self, layer: str) -> List[Dict[str, Any]]:
        """Get all rules that apply to a specific layer."""
        return [rule for rule in self._rules if rule.get("result", {}).get("layer") == layer]

    def get_rules_for_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Get all rules that classify entities as a specific type."""
        return [rule for rule in self._rules if rule.get("result", {}).get("type") == entity_type]

    def reload_rules(self) -> None:
        """Reload rules from configuration."""
        self._load_rules()

    def to_dict(self) -> Dict[str, Any]:
        """Convert engine state to dictionary."""
        return {
            "rules_count": len(self._rules),
            "conflict_resolution": self._conflict_resolution,
            "validation": self._validation,
        }