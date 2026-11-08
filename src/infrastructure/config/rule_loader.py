"""Unified Rule Loader - 统一规则加载器

根据 PROJECT_ARCHITECTURE_AUDIT_REPORT_20260708.md 第3.2节和 CONTRACT_GOVERNANCE_SPEC.md 的要求：
- 将质量、关系、分类、门禁规则逐步收束到统一治理框架
- 明确每类规则归属
- 消除重复定义
- 提供单一加载入口

规则分类：
1. BUSINESS - 业务规则（实体分类、关系定义）
2. EXECUTION - 执行规则（阈值、策略）
3. GOVERNANCE - 治理规则（质量门禁、冲突检测）

统一规则目录结构：
config/rules/
├── business/
│   ├── entity_classification.yaml
│   ├── relationship_rules.yaml
│   └── skill_layer_mapping.yaml
├── execution/
│   ├── rigor_thresholds.yaml
│   ├── extraction_policies.yaml
│   └── pipeline_policy.yaml
└── governance/
    ├── quality_config.yaml
    ├── conflict_rules.yaml
    └── quality_policy.yaml
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.pipeline.stages.common import PROJECT_ROOT
from .contract_registry import ContractCategory

logger = logging.getLogger(__name__)


class RuleLoader:
    """统一规则加载器"""

    _RULE_CATEGORIES = {
        ContractCategory.BUSINESS: "business",
        ContractCategory.EXECUTION: "execution",
        ContractCategory.GOVERNANCE: "governance",
    }

    _RULE_MAPPING = {
        "entity_classification": {"category": ContractCategory.BUSINESS, "file": "entity_classification_unified.yaml"},
        "relationship_rules": {"category": ContractCategory.BUSINESS, "file": "ontology/relationship_rules.yaml"},
        "skill_layer_mapping": {"category": ContractCategory.BUSINESS, "file": "ontology/skill_layer_mapping.yaml"},
        "rigor_thresholds": {"category": ContractCategory.EXECUTION, "file": "quality/rigor_thresholds.yaml"},
        "extraction_policies": {"category": ContractCategory.EXECUTION, "file": "extraction/extraction_policies.yaml"},
        "pipeline_policy": {"category": ContractCategory.EXECUTION, "file": "pipeline/pipeline_policy.json"},
        "quality_config": {"category": ContractCategory.GOVERNANCE, "file": "quality/quality_config.yaml"},
        "conflict_rules": {"category": ContractCategory.GOVERNANCE, "file": "quality/conflict_rules.yaml"},
        "quality_policy": {"category": ContractCategory.GOVERNANCE, "file": "quality_policy.yaml"},
    }

    def __init__(self, config_dir: Optional[Path] = None):
        self._config_dir = config_dir or (PROJECT_ROOT / "config")
        self._rules_cache: Dict[str, Dict[str, Any]] = {}
        self._loaded_rules: Dict[str, bool] = {}

    def load_all_rules(self) -> Dict[str, Dict[str, Any]]:
        """加载所有规则"""
        logger.info("Loading all rules from unified rule loader...")
        all_rules = {}

        for rule_id, mapping in self._RULE_MAPPING.items():
            rule_data = self.load_rule(rule_id)
            if rule_data:
                all_rules[rule_id] = rule_data

        logger.info(f"Loaded {len(all_rules)} rules")
        return all_rules

    def load_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """加载指定规则"""
        if rule_id in self._rules_cache:
            return self._rules_cache[rule_id]

        mapping = self._RULE_MAPPING.get(rule_id)
        if not mapping:
            logger.warning(f"Rule '{rule_id}' not found in mapping")
            return None

        category = mapping["category"]
        relative_path = mapping["file"]
        rule_path = self._config_dir / relative_path

        if not rule_path.exists():
            logger.warning(f"Rule file not found: {rule_path}")
            return None

        try:
            with open(rule_path, "r", encoding="utf-8") as f:
                if rule_path.suffix == ".json":
                    rule_data = json.load(f)
                else:
                    rule_data = yaml.safe_load(f) or {}

            self._rules_cache[rule_id] = rule_data
            self._loaded_rules[rule_id] = True

            logger.debug(f"Loaded rule: {rule_id} from {relative_path}")
            return rule_data
        except Exception as e:
            logger.error(f"Failed to load rule {rule_id}: {e}")
            return None

    def load_by_category(self, category: ContractCategory) -> Dict[str, Dict[str, Any]]:
        """按分类加载规则"""
        rules = {}
        for rule_id, mapping in self._RULE_MAPPING.items():
            if mapping["category"] == category:
                rule_data = self.load_rule(rule_id)
                if rule_data:
                    rules[rule_id] = rule_data
        return rules

    def get_rule_category(self, rule_id: str) -> Optional[ContractCategory]:
        """获取规则分类"""
        mapping = self._RULE_MAPPING.get(rule_id)
        return mapping["category"] if mapping else None

    def get_all_rule_ids(self) -> List[str]:
        """获取所有规则 ID"""
        return list(self._RULE_MAPPING.keys())

    def get_rule_ids_by_category(self, category: ContractCategory) -> List[str]:
        """按分类获取规则 ID"""
        return [rule_id for rule_id, mapping in self._RULE_MAPPING.items() if mapping["category"] == category]

    def is_loaded(self, rule_id: str) -> bool:
        """检查规则是否已加载"""
        return self._loaded_rules.get(rule_id, False)

    def invalidate_cache(self, rule_id: Optional[str] = None) -> None:
        """失效缓存"""
        if rule_id:
            self._rules_cache.pop(rule_id, None)
            self._loaded_rules.pop(rule_id, None)
        else:
            self._rules_cache.clear()
            self._loaded_rules.clear()

    def validate_rules(self) -> Dict[str, List[str]]:
        """验证所有规则"""
        errors = {}

        for rule_id in self.get_all_rule_ids():
            rule_errors = self._validate_rule(rule_id)
            if rule_errors:
                errors[rule_id] = rule_errors

        return errors

    def _validate_rule(self, rule_id: str) -> List[str]:
        """验证单个规则"""
        errors = []
        rule_data = self.load_rule(rule_id)

        if not rule_data:
            errors.append("Rule data is empty or failed to load")
            return errors

        if rule_id == "entity_classification":
            if "classification_rules" not in rule_data:
                errors.append("Missing 'classification_rules' field")
            else:
                rules = rule_data["classification_rules"]
                rule_ids = [rule["id"] for rule in rules]
                if len(rule_ids) != len(set(rule_ids)):
                    errors.append("Duplicate rule IDs found")

        elif rule_id == "relationship_rules":
            if "relation_family_map" not in rule_data:
                errors.append("Missing 'relation_family_map' field")

        elif rule_id == "quality_config":
            if "quality_grades" not in rule_data:
                errors.append("Missing 'quality_grades' field")

        return errors

    def get_stats(self) -> Dict[str, Any]:
        """获取规则统计信息"""
        return {
            "total_rules": len(self._RULE_MAPPING),
            "by_category": {
                cat.value: len(self.get_rule_ids_by_category(cat))
                for cat in ContractCategory
            },
            "loaded_rules": sum(self._loaded_rules.values()),
        }


_rule_loader: Optional[RuleLoader] = None


def get_rule_loader(config_dir: Optional[Path] = None) -> RuleLoader:
    """获取规则加载器实例（自动加载所有规则）"""
    global _rule_loader

    if _rule_loader is None:
        _rule_loader = RuleLoader(config_dir)
        _rule_loader.load_all_rules()

    return _rule_loader


def reload_rules(config_dir: Optional[Path] = None) -> RuleLoader:
    """重新加载规则"""
    global _rule_loader
    _rule_loader = RuleLoader(config_dir)
    return _rule_loader