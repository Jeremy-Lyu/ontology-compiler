"""关系合法性验证服务 - 运行时门禁

P3-07: 建立关系合法性 gate，支持运行时非法关系阻断

架构目标：
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum


class ViolationLevel(str, Enum):
    """违规级别"""
    BLOCKING = "blocking"
    WARNING = "warning"
    INFO = "info"


class ViolationType(str, Enum):
    """违规类型"""
    UNKNOWN_RELATION_TYPE = "unknown_relationship_type"
    SOURCE_TYPE_VIOLATION = "source_type_violation"
    TARGET_TYPE_VIOLATION = "target_type_violation"
    LAYER_VIOLATION = "layer_violation"
    EVIDENCE_MISSING = "evidence_missing"
    REQUIRED_FIELD_MISSING = "required_field_missing"


@dataclass
class Violation:
    """违规记录"""
    violation_type: ViolationType
    violation_level: ViolationLevel
    relationship_type: str
    source_id: str = ""
    source_type: str = ""
    source_layer: str = ""
    target_id: str = ""
    target_type: str = ""
    target_layer: str = ""
    allowed_types: List[str] = field(default_factory=list)
    allowed_layers: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelationshipValidationResult:
    """验证结果"""
    valid: bool
    total_relationships: int = 0
    valid_count: int = 0
    blocked_count: int = 0
    warnings_count: int = 0
    violations: List[Violation] = field(default_factory=list)
    allowed_relationships: List[Dict[str, Any]] = field(default_factory=list)
    blocked_relationships: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["violations"] = [v.to_dict() for v in result["violations"]]
        return result


@dataclass
class RelationshipRule:
    """关系规则"""
    relation_type: str
    description: str
    allowed_source_types: List[str] = field(default_factory=list)
    allowed_target_types: List[str] = field(default_factory=list)
    allowed_source_layers: List[str] = field(default_factory=list)
    allowed_target_layers: List[str] = field(default_factory=list)
    required_fields: List[str] = field(default_factory=list)
    evidence_required: bool = False
    violation_level: ViolationLevel = ViolationLevel.BLOCKING


class RelationshipValidityGate:
    """关系合法性验证服务"""

    DEFAULT_REQUIRED_FIELDS = ["source", "target", "type"]

    def __init__(self, rules_path: Optional[str] = None):
        self.rules: Dict[str, RelationshipRule] = {}
        self.layer_whitelist: Dict[str, set] = {}
        self._load_rules(rules_path)

    def _load_rules(self, rules_path: Optional[str]) -> None:
        """加载关系规则配置"""
        self._load_rules_from_config()

        if rules_path and Path(rules_path).exists():
            try:
                with open(rules_path, "r", encoding="utf-8") as f:
                    rules_data = json.load(f)
                for rule_data in rules_data.get("rules", []):
                    self.rules[rule_data["relation_type"]] = RelationshipRule(
                        relation_type=rule_data["relation_type"],
                        description=rule_data.get("description", ""),
                        allowed_source_types=rule_data.get("allowed_source_types", []),
                        allowed_target_types=rule_data.get("allowed_target_types", []),
                        allowed_source_layers=rule_data.get("allowed_source_layers", []),
                        allowed_target_layers=rule_data.get("allowed_target_layers", []),
                        required_fields=rule_data.get("required_fields", []),
                        evidence_required=rule_data.get("evidence_required", False),
                        violation_level=ViolationLevel(rule_data.get("violation_level", "blocking")),
                    )
            except Exception:
                pass

        self._load_default_rules()

    def _load_rules_from_config(self) -> None:
        """从统一配置加载关系规则"""
        try:
            from src.infrastructure.config.unified_config import get_unified_config
            config = get_unified_config()

            entity_classification = config.entity_classification or {}
            relation_family_defaults = entity_classification.get("relation_family_defaults", {})

            entity_type_registry = config.entity_type_registry or {}
            relation_defaults = entity_type_registry.get("relation_defaults", {})

            all_relation_types = set()
            for family in relation_family_defaults.values():
                all_relation_types.update(family)
            for family in relation_defaults.values():
                all_relation_types.update(family)

            types_info = entity_type_registry.get("types", {})
            all_entity_types = list(types_info.keys())

            for rel_type in all_relation_types:
                if rel_type not in self.rules:
                    self.rules[rel_type] = RelationshipRule(
                        relation_type=rel_type,
                        description=f"{rel_type} 关系",
                        allowed_source_types=all_entity_types,
                        allowed_target_types=all_entity_types,
                    )

            layer_rules = entity_classification.get("layer_rules", {})
            for layer, rules in layer_rules.items():
                self.layer_whitelist[layer] = set(rules.get("allowed_layers", []))

            if not self.layer_whitelist:
                self.layer_whitelist = {
                    "L0": {"L0", "L1"},
                    "L1": {"L0", "L1", "L2"},
                    "L2": {"L1", "L2", "L2.5"},
                    "L2.5": {"L2", "L2.5", "L3"},
                    "L3": {"L2.5", "L3"},
                }

        except Exception:
            self.layer_whitelist = {
                "L0": {"L0", "L1"},
                "L1": {"L0", "L1", "L2"},
                "L2": {"L1", "L2", "L2.5"},
                "L2.5": {"L2", "L2.5", "L3"},
                "L3": {"L2.5", "L3"},
            }

    def _load_default_rules(self) -> None:
        """加载默认关系规则"""
        default_rules = [
            RelationshipRule(
                relation_type="SUBSUMES",
                description="泛化关系：父类包含子类",
                allowed_source_types=["Capability", "System", "Actor", "Asset", "Data", "Process", "Concept"],
                allowed_target_types=["Capability", "System", "Actor", "Asset", "Data", "Process", "Concept"],
                required_fields=["source", "target", "type"],
            ),
            RelationshipRule(
                relation_type="DECOMPOSES_TO",
                description="分解关系：整体分解为部分",
                allowed_source_types=["Capability", "System", "Process"],
                allowed_target_types=["Capability", "System", "Process"],
            ),
            RelationshipRule(
                relation_type="CONTAINS",
                description="包含关系",
                allowed_source_types=["System", "Asset", "Process"],
                allowed_target_types=["Capability", "System", "Asset", "Data"],
            ),
            RelationshipRule(
                relation_type="HAS_PART",
                description="组成关系",
                allowed_source_types=["System", "Asset", "Process"],
                allowed_target_types=["System", "Asset", "Capability"],
            ),
            RelationshipRule(
                relation_type="PART_OF",
                description="属于关系（反向）",
                allowed_source_types=["Capability", "System", "Asset"],
                allowed_target_types=["System", "Asset", "Process"],
            ),
            RelationshipRule(
                relation_type="INSTANCE_OF",
                description="实例化关系",
                allowed_source_types=["Unique_Instance", "Workflow_Instance"],
                allowed_target_types=["Capability", "System", "Actor", "Asset", "Data", "Process"],
            ),
            RelationshipRule(
                relation_type="HAS_VARIANT",
                description="变体关系",
                allowed_source_types=["Asset", "Product_Definition"],
                allowed_target_types=["Asset", "Product_Definition"],
            ),
            RelationshipRule(
                relation_type="HAS_COMPONENT",
                description="组件关系",
                allowed_source_types=["System", "Asset"],
                allowed_target_types=["System", "Asset", "Capability"],
            ),
            RelationshipRule(
                relation_type="MANUFACTURED_BY",
                description="制造关系",
                allowed_source_types=["Asset", "Product_Definition"],
                allowed_target_types=["Actor", "Business_Function"],
            ),
            RelationshipRule(
                relation_type="USED_IN",
                description="使用关系",
                allowed_source_types=["Capability", "Asset", "Data"],
                allowed_target_types=["Process", "Capability", "System"],
            ),
            RelationshipRule(
                relation_type="HAS_OPERATION",
                description="操作关系",
                allowed_source_types=["System", "Capability"],
                allowed_target_types=["Process", "Business_Function"],
            ),
            RelationshipRule(
                relation_type="ROUTES_TO",
                description="路由关系",
                allowed_source_types=["Process", "Asset"],
                allowed_target_types=["Process", "Asset"],
            ),
            RelationshipRule(
                relation_type="HAS_MOVEMENT",
                description="移动关系",
                allowed_source_types=["Asset"],
                allowed_target_types=["Process", "Asset"],
            ),
            RelationshipRule(
                relation_type="STORED_AT",
                description="存储关系",
                allowed_source_types=["Asset", "Data"],
                allowed_target_types=["Asset"],
            ),
            RelationshipRule(
                relation_type="TRANSFERS_TO",
                description="转移关系",
                allowed_source_types=["Asset", "Data"],
                allowed_target_types=["Asset", "Actor"],
            ),
            RelationshipRule(
                relation_type="REPLENISHES",
                description="补充关系",
                allowed_source_types=["Asset"],
                allowed_target_types=["Asset"],
            ),
            RelationshipRule(
                relation_type="DERIVED_FROM",
                description="派生关系",
                allowed_source_types=["Data", "Asset"],
                allowed_target_types=["Data", "Asset"],
                violation_level=ViolationLevel.WARNING,
            ),
            RelationshipRule(
                relation_type="TRANSFORMS_TO",
                description="转换关系",
                allowed_source_types=["Data", "Asset"],
                allowed_target_types=["Data", "Asset"],
                violation_level=ViolationLevel.WARNING,
            ),
            RelationshipRule(
                relation_type="REALIZES",
                description="实现关系",
                allowed_source_types=["System", "Process"],
                allowed_target_types=["Capability"],
            ),
            RelationshipRule(
                relation_type="SUPPORTS",
                description="支持关系",
                allowed_source_types=["System", "Capability"],
                allowed_target_types=["Capability", "Process"],
            ),
            RelationshipRule(
                relation_type="APPLICABLE_TO",
                description="适用关系",
                allowed_source_types=["BusinessRule", "Capability"],
                allowed_target_types=["Process", "Asset"],
            ),
            RelationshipRule(
                relation_type="SATISFIES",
                description="满足关系",
                allowed_source_types=["Process", "System"],
                allowed_target_types=["BusinessRule", "Capability"],
            ),
            RelationshipRule(
                relation_type="ENABLES",
                description="使能关系",
                allowed_source_types=["Capability", "System"],
                allowed_target_types=["Process", "Capability"],
            ),
            RelationshipRule(
                relation_type="RELATED_TO",
                description="关联关系",
                allowed_source_types=["Capability", "System", "Actor", "Asset", "Data", "Process", "Concept"],
                allowed_target_types=["Capability", "System", "Actor", "Asset", "Data", "Process", "Concept"],
                violation_level=ViolationLevel.INFO,
            ),
            RelationshipRule(
                relation_type="EXTENDS",
                description="扩展关系",
                allowed_source_types=["Capability", "System"],
                allowed_target_types=["Capability", "System"],
            ),
            RelationshipRule(
                relation_type="GENERATES",
                description="生成关系",
                allowed_source_types=["Process"],
                allowed_target_types=["Data", "Asset"],
            ),
            RelationshipRule(
                relation_type="TRIGGER",
                description="触发关系",
                allowed_source_types=["Process", "Event"],
                allowed_target_types=["Process", "Action"],
            ),
            RelationshipRule(
                relation_type="VALIDATES",
                description="验证关系",
                allowed_source_types=["BusinessRule", "ValidationRule"],
                allowed_target_types=["Data", "Process"],
            ),
            RelationshipRule(
                relation_type="GOVERNS_DATA",
                description="治理关系",
                allowed_source_types=["BusinessRule", "Actor"],
                allowed_target_types=["Data", "Process"],
            ),
            RelationshipRule(
                relation_type="REPLACES",
                description="替换关系",
                allowed_source_types=["Asset", "System"],
                allowed_target_types=["Asset", "System"],
            ),
            RelationshipRule(
                relation_type="SUPERSEDES",
                description="替代关系",
                allowed_source_types=["Asset", "System", "BusinessRule"],
                allowed_target_types=["Asset", "System", "BusinessRule"],
            ),
            RelationshipRule(
                relation_type="SAME_AS",
                description="等价关系",
                allowed_source_types=["Capability", "System", "Actor", "Asset", "Data", "Process", "Concept"],
                allowed_target_types=["Capability", "System", "Actor", "Asset", "Data", "Process", "Concept"],
            ),
            RelationshipRule(
                relation_type="EFFECTIVE_FOR",
                description="时效关系",
                allowed_source_types=["BusinessRule", "Asset"],
                allowed_target_types=["Asset", "Process"],
            ),
            RelationshipRule(
                relation_type="MODIFIES",
                description="修改关系",
                allowed_source_types=["Process", "Action"],
                allowed_target_types=["Asset", "Data"],
            ),
            RelationshipRule(
                relation_type="RESULTS_IN",
                description="导致关系",
                allowed_source_types=["Process", "Action"],
                allowed_target_types=["Asset", "Data"],
            ),
            RelationshipRule(
                relation_type="REFERENCES",
                description="引用关系",
                allowed_source_types=["Document", "Data"],
                allowed_target_types=["Document", "Data", "Asset"],
            ),
            RelationshipRule(
                relation_type="TRACES_TO",
                description="追溯关系",
                allowed_source_types=["Asset", "Process"],
                allowed_target_types=["Asset", "Process"],
            ),
            RelationshipRule(
                relation_type="EXECUTES",
                description="执行关系",
                allowed_source_types=["Actor", "System"],
                allowed_target_types=["Process", "Action"],
            ),
            RelationshipRule(
                relation_type="TRIGGERS",
                description="触发关系",
                allowed_source_types=["Event", "Process"],
                allowed_target_types=["Process", "Action"],
            ),
            RelationshipRule(
                relation_type="PARTICIPATES_IN",
                description="参与关系",
                allowed_source_types=["Actor", "System"],
                allowed_target_types=["Process"],
            ),
            RelationshipRule(
                relation_type="ASSIGNED_TO",
                description="分配关系",
                allowed_source_types=["Process", "Action"],
                allowed_target_types=["Actor", "System"],
            ),
            RelationshipRule(
                relation_type="RESPONSIBLE_FOR",
                description="负责关系",
                allowed_source_types=["Actor", "System"],
                allowed_target_types=["Process", "Asset"],
            ),
            RelationshipRule(
                relation_type="OWNS",
                description="拥有关系",
                allowed_source_types=["Actor", "System"],
                allowed_target_types=["Asset", "Data"],
            ),
            RelationshipRule(
                relation_type="OPERATES",
                description="操作关系",
                allowed_source_types=["Actor", "System"],
                allowed_target_types=["Asset", "Process"],
            ),
            RelationshipRule(
                relation_type="MAINTAINS",
                description="维护关系",
                allowed_source_types=["Actor", "System"],
                allowed_target_types=["Asset", "System"],
            ),
            RelationshipRule(
                relation_type="USES",
                description="使用关系",
                allowed_source_types=["Capability", "Asset", "Process"],
                allowed_target_types=["Capability", "System", "Asset", "Data"],
            ),
            RelationshipRule(
                relation_type="DEPLOYS_TO",
                description="部署关系",
                allowed_source_types=["System", "Asset"],
                allowed_target_types=["System", "Asset"],
            ),
            RelationshipRule(
                relation_type="OPTION_FOR",
                description="选项关系",
                allowed_source_types=["Asset"],
                allowed_target_types=["Asset", "Capability"],
            ),
            RelationshipRule(
                relation_type="INTEGRATES_WITH",
                description="集成关系",
                allowed_source_types=["System"],
                allowed_target_types=["System"],
            ),
            RelationshipRule(
                relation_type="CONNECTED_TO",
                description="连接关系",
                allowed_source_types=["System", "Asset"],
                allowed_target_types=["System", "Asset"],
            ),
            RelationshipRule(
                relation_type="UPLOADS_TO",
                description="上传关系",
                allowed_source_types=["Data", "Asset"],
                allowed_target_types=["System", "Asset"],
            ),
            RelationshipRule(
                relation_type="CONNECTS_TO",
                description="连接到关系",
                allowed_source_types=["System", "Asset"],
                allowed_target_types=["System", "Asset"],
            ),
            RelationshipRule(
                relation_type="ASSOCIATED_WITH",
                description="关联关系",
                allowed_source_types=["Capability", "System", "Actor", "Asset", "Data", "Process", "Concept"],
                allowed_target_types=["Capability", "System", "Actor", "Asset", "Data", "Process", "Concept"],
                violation_level=ViolationLevel.INFO,
            ),
        ]

        for rule in default_rules:
            if rule.relation_type not in self.rules:
                self.rules[rule.relation_type] = rule

    def validate(
        self,
        relationships: List[Dict[str, Any]],
        enforce_layer_rules: bool = True,
        enforce_evidence_rules: bool = True,
        min_confidence: float = 0.7,
    ) -> RelationshipValidationResult:
        """验证关系合法性"""
        violations: List[Violation] = []
        allowed_relationships: List[Dict[str, Any]] = []
        blocked_relationships: List[Dict[str, Any]] = []

        for rel in relationships:
            rel_violations = self._validate_single(
                rel,
                enforce_layer_rules=enforce_layer_rules,
                enforce_evidence_rules=enforce_evidence_rules,
                min_confidence=min_confidence,
            )

            has_blocking = any(v.violation_level == ViolationLevel.BLOCKING for v in rel_violations)

            if has_blocking:
                blocked_relationships.append(rel)
            else:
                allowed_relationships.append(rel)

            violations.extend(rel_violations)

        blocked_count = sum(1 for v in violations if v.violation_level == ViolationLevel.BLOCKING)
        warnings_count = sum(1 for v in violations if v.violation_level == ViolationLevel.WARNING)

        return RelationshipValidationResult(
            valid=blocked_count == 0,
            total_relationships=len(relationships),
            valid_count=len(allowed_relationships),
            blocked_count=blocked_count,
            warnings_count=warnings_count,
            violations=violations,
            allowed_relationships=allowed_relationships,
            blocked_relationships=blocked_relationships,
        )

    def _validate_single(
        self,
        rel: Dict[str, Any],
        enforce_layer_rules: bool = True,
        enforce_evidence_rules: bool = True,
        min_confidence: float = 0.7,
    ) -> List[Violation]:
        """验证单个关系"""
        violations: List[Violation] = []

        rel_type = rel.get("type", "").upper()
        source_id = rel.get("source", "")
        target_id = rel.get("target", "")
        source_type = rel.get("source_type", "")
        target_type = rel.get("target_type", "")
        source_layer = rel.get("source_layer", "")
        target_layer = rel.get("target_layer", "")
        confidence = rel.get("confidence")

        if not source_id or not target_id:
            violations.append(Violation(
                violation_type=ViolationType.REQUIRED_FIELD_MISSING,
                violation_level=ViolationLevel.BLOCKING,
                relationship_type=rel_type,
                source_id=source_id,
                target_id=target_id,
                missing_fields=[field for field in ["source", "target"] if not rel.get(field)],
                reason="关系缺少 source 或 target",
            ))

        if confidence is not None:
            try:
                if float(confidence) < float(min_confidence):
                    violations.append(Violation(
                        violation_type=ViolationType.REQUIRED_FIELD_MISSING,
                        violation_level=ViolationLevel.BLOCKING,
                        relationship_type=rel_type,
                        source_id=source_id,
                        target_id=target_id,
                        missing_fields=["confidence"],
                        reason=f"关系置信度过低: {confidence} < {min_confidence}",
                    ))
            except (TypeError, ValueError):
                violations.append(Violation(
                    violation_type=ViolationType.REQUIRED_FIELD_MISSING,
                    violation_level=ViolationLevel.BLOCKING,
                    relationship_type=rel_type,
                    source_id=source_id,
                    target_id=target_id,
                    missing_fields=["confidence"],
                    reason="关系置信度不可解析",
                ))

        for field in self.DEFAULT_REQUIRED_FIELDS:
            if not rel.get(field):
                violations.append(Violation(
                    violation_type=ViolationType.REQUIRED_FIELD_MISSING,
                    violation_level=ViolationLevel.BLOCKING,
                    relationship_type=rel_type,
                    source_id=source_id,
                    target_id=target_id,
                    missing_fields=[field],
                    reason=f"缺少必需字段: {field}",
                ))

        rule = self.rules.get(rel_type)
        if not rule:
            violations.append(Violation(
                violation_type=ViolationType.UNKNOWN_RELATION_TYPE,
                violation_level=ViolationLevel.BLOCKING,
                relationship_type=rel_type,
                source_id=source_id,
                target_id=target_id,
                reason=f"未知关系类型: {rel_type}",
            ))
            return violations

        if source_type and rule.allowed_source_types:
            if source_type not in rule.allowed_source_types:
                violations.append(Violation(
                    violation_type=ViolationType.SOURCE_TYPE_VIOLATION,
                    violation_level=rule.violation_level,
                    relationship_type=rel_type,
                    source_id=source_id,
                    source_type=source_type,
                    target_id=target_id,
                    allowed_types=rule.allowed_source_types,
                    reason=f"源实体类型 {source_type} 不允许作为 {rel_type} 的源",
                ))

        if target_type and rule.allowed_target_types:
            if target_type not in rule.allowed_target_types:
                violations.append(Violation(
                    violation_type=ViolationType.TARGET_TYPE_VIOLATION,
                    violation_level=rule.violation_level,
                    relationship_type=rel_type,
                    source_id=source_id,
                    target_id=target_id,
                    target_type=target_type,
                    allowed_types=rule.allowed_target_types,
                    reason=f"目标实体类型 {target_type} 不允许作为 {rel_type} 的目标",
                ))

        if enforce_layer_rules and source_layer and target_layer:
            allowed_target_layers = self.layer_whitelist.get(source_layer, set())
            if target_layer not in allowed_target_layers:
                violations.append(Violation(
                    violation_type=ViolationType.LAYER_VIOLATION,
                    violation_level=ViolationLevel.BLOCKING,
                    relationship_type=rel_type,
                    source_id=source_id,
                    source_layer=source_layer,
                    target_id=target_id,
                    target_layer=target_layer,
                    allowed_layers=list(allowed_target_layers),
                    reason=f"L{source_layer} 层实体不能关联到 L{target_layer} 层实体",
                ))

        if enforce_evidence_rules:
            evidence_fields = ["evidence_ref", "citation", "rule_no"]
            missing_fields = [f for f in evidence_fields if not rel.get(f)]
            if missing_fields:
                violations.append(Violation(
                    violation_type=ViolationType.EVIDENCE_MISSING,
                    violation_level=ViolationLevel.BLOCKING,
                    relationship_type=rel_type,
                    source_id=source_id,
                    target_id=target_id,
                    missing_fields=missing_fields,
                    reason=f"缺少证据字段: {', '.join(missing_fields)}",
                ))

        return violations

    def validate_and_block(
        self,
        relationships: List[Dict[str, Any]],
        enforce_layer_rules: bool = True,
        enforce_evidence_rules: bool = True,
    ) -> Tuple[List[Dict[str, Any]], RelationshipValidationResult]:
        """验证并阻断非法关系"""
        result = self.validate(
            relationships,
            enforce_layer_rules=enforce_layer_rules,
            enforce_evidence_rules=enforce_evidence_rules,
        )
        return result.allowed_relationships, result

    def check_relation_allowed(
        self,
        relation_type: str,
        source_type: str,
        target_type: str,
        source_layer: str = "",
        target_layer: str = "",
    ) -> bool:
        """检查关系是否允许"""
        rule = self.rules.get(relation_type.upper())
        if not rule:
            return False

        if source_type and rule.allowed_source_types and source_type not in rule.allowed_source_types:
            return False

        if target_type and rule.allowed_target_types and target_type not in rule.allowed_target_types:
            return False

        if source_layer and target_layer:
            allowed_target_layers = self.layer_whitelist.get(source_layer, set())
            if target_layer not in allowed_target_layers:
                return False

        return True

    def get_rule(self, relation_type: str) -> Optional[RelationshipRule]:
        """获取关系规则"""
        return self.rules.get(relation_type.upper())

    def add_rule(self, rule: RelationshipRule) -> None:
        """添加关系规则"""
        self.rules[rule.relation_type] = rule

    def remove_rule(self, relation_type: str) -> None:
        """移除关系规则"""
        self.rules.pop(relation_type.upper(), None)

    def list_rules(self) -> List[RelationshipRule]:
        """列出所有规则"""
        return list(self.rules.values())

    def export_rules(self, output_path: str) -> None:
        """导出规则到文件"""
        rules_data = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "rules": [asdict(rule) for rule in self.rules.values()],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rules_data, f, ensure_ascii=False, indent=2)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_rules": len(self.rules),
            "layer_whitelist_layers": list(self.layer_whitelist.keys()),
        }
