"""PROJECT_ARCHITECTURE_AUDIT_REPORT_20260708.md 的要求：
- 问题 P9: 质量门禁与治理规则未统一
- 将 audit、guard、quality_policy、relationship governance 收束到一个治理层

设计原则：
- 判定口径一致：所有质量判定使用同一套规则
- 统一入口：所有治理检查通过单一接口
- 可配置：阈值和规则可通过配置文件调整
- 可追溯：每次检查记录详细信息，便于问题追踪
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.infrastructure.config.unified_config import get_unified_config

logger = logging.getLogger(__name__)


class QualityGrade(Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class GateResult(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    BLOCK = "block"


class IssueSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueCategory(Enum):
    ENTITY_COMPLETENESS = "entity_completeness"
    RELATIONSHIP_VALIDITY = "relationship_validity"
    METADATA_COMPLETENESS = "metadata_completeness"
    DOMAIN_CONSISTENCY = "domain_consistency"
    GOVERNANCE = "governance"
    SCHEMA = "schema"


@dataclass
class QualityIssue:
    id: str
    severity: IssueSeverity
    category: IssueCategory
    message: str
    entity_id: Optional[str] = None
    field: Optional[str] = None
    expected: Optional[str] = None
    actual: Optional[str] = None
    evidence: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "entity_id": self.entity_id,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "evidence": self.evidence,
        }


@dataclass
class QualityScore:
    overall: float = 0.0
    entity_completeness: float = 0.0
    relationship_validity: float = 0.0
    metadata_completeness: float = 0.0
    domain_consistency: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall,
            "entity_completeness": self.entity_completeness,
            "relationship_validity": self.relationship_validity,
            "metadata_completeness": self.metadata_completeness,
            "domain_consistency": self.domain_consistency,
        }


@dataclass
class QualityGateReport:
    gate_id: str
    result: GateResult
    grade: QualityGrade
    score: QualityScore
    issues: List[QualityIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())
    rejection_reason: str = ""

    @property
    def critical_issue_count(self) -> int:
        return len([i for i in self.issues if i.severity == IssueSeverity.CRITICAL])

    @property
    def high_issue_count(self) -> int:
        return len([i for i in self.issues if i.severity == IssueSeverity.HIGH])

    @property
    def medium_issue_count(self) -> int:
        return len([i for i in self.issues if i.severity == IssueSeverity.MEDIUM])

    @property
    def low_issue_count(self) -> int:
        return len([i for i in self.issues if i.severity == IssueSeverity.LOW])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "result": self.result.value,
            "grade": self.grade.value,
            "score": self.score.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
            "critical_issue_count": self.critical_issue_count,
            "high_issue_count": self.high_issue_count,
            "medium_issue_count": self.medium_issue_count,
            "low_issue_count": self.low_issue_count,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "rejection_reason": self.rejection_reason,
        }

    def to_quality_report(self) -> Dict[str, Any]:
        """Convert to canonical QualityReport contract payload.

        Aligns with schemas/quality_report.schema.yaml v1.0.0.
        Bridge method for F8: quality gate → quality_report contract.
        """
        grade_map = {
            QualityGrade.GOLD: "A",
            QualityGrade.SILVER: "B",
            QualityGrade.BRONZE: "C",
        }
        quality_level_map = {
            QualityGrade.GOLD: "GOLD",
            QualityGrade.SILVER: "SILVER",
            QualityGrade.BRONZE: "BRONZE",
        }
        score_int = int(self.score.overall * 100)
        grade = grade_map.get(self.grade, "UNKNOWN")
        quality_level = quality_level_map.get(self.grade, "UNKNOWN")

        rule_scores = []
        for issue in self.issues:
            rule_scores.append({
                "rule_id": issue.id,
                "score": 0.0 if issue.severity in (IssueSeverity.CRITICAL, IssueSeverity.HIGH) else 0.5,
                "weight": 1.0,
                "passed": False,
                "issues": [issue.message],
            })

        issue_messages = [i.message for i in self.issues]
        passed = self.result in (GateResult.PASS, GateResult.WARN)

        return {
            "grade": grade,
            "score": score_int,
            "quality_level": quality_level,
            "passed": passed,
            "issues": issue_messages,
            "warnings": [i.message for i in self.issues if i.severity == IssueSeverity.LOW],
            "l1_score": self.score.entity_completeness,
            "l2_score": self.score.relationship_validity,
            "scores": {
                "entity_completeness": self.score.entity_completeness,
                "relationship_validity": self.score.relationship_validity,
                "metadata_completeness": self.score.metadata_completeness,
                "domain_consistency": self.score.domain_consistency,
            },
            "rule_scores": rule_scores,
            "failed_rules": [i.id for i in self.issues if not i.severity == IssueSeverity.LOW],
            "passed_rules": [],
            "execution_time_ms": self.metadata.get("execution_time_ms", 0.0),
            "generated_at": self.timestamp,
            "metadata": {
                "gate_id": self.gate_id,
                "gate_result": self.result.value,
                "critical_issue_count": self.critical_issue_count,
                "high_issue_count": self.high_issue_count,
                "medium_issue_count": self.medium_issue_count,
                "low_issue_count": self.low_issue_count,
            },
            "contract_id": "quality_report",
            "contract_version": "1.0.0",
        }


class QualityGate:
    """统一质量门禁 - 合并 audit、guard、quality_policy 逻辑"""

    def __init__(self):
        self._config = get_unified_config()
        self._quality_config = self._config.quality_config
        self._entity_classification = self._config.entity_classification
        self._relationship_rules = self._load_relationship_rules()

    def _load_relationship_rules(self) -> Dict[str, Any]:
        """加载关系规则"""
        try:
            rule_path = Path("config") / "ontology" / "relationship_rules.yaml"
            if rule_path.exists():
                with open(rule_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load relationship rules: {e}")
        return {}

    def run(self, pipeline_output: Dict[str, Any]) -> QualityGateReport:
        """运行质量门禁"""
        logger.info("Running unified quality gate...")
        report = QualityGateReport(
            gate_id="main_quality_gate",
            result=GateResult.PASS,
            grade=QualityGrade.BRONZE,
            score=QualityScore(),
        )

        issues: List[QualityIssue] = []

        issues.extend(self._check_entity_completeness(pipeline_output))
        issues.extend(self._check_relationship_validity(pipeline_output))
        issues.extend(self._check_metadata_completeness(pipeline_output))
        issues.extend(self._check_domain_consistency(pipeline_output))
        issues.extend(self._check_governance_rules(pipeline_output))

        report.issues = issues
        report.score = self._calculate_score(issues)
        report.grade = self._determine_grade(report.score.overall)
        report.result = self._determine_result(report)

        self._log_report(report)
        return report

    def _check_entity_completeness(self, output: Dict[str, Any]) -> List[QualityIssue]:
        """检查实体完整性"""
        issues = []
        entities = output.get("entities", [])
        required_fields = self._entity_classification.get("output_fields", {}).get("node", {}).get("required", [])

        if not required_fields:
            required_fields = ["id", "name", "type", "layer"]

        for i, entity in enumerate(entities):
            for field in required_fields:
                if field not in entity or entity[field] is None or str(entity[field]).strip() == "":
                    issues.append(QualityIssue(
                        id=f"entity_completeness_{i}_{field}",
                        severity=self._get_field_severity(field),
                        category=IssueCategory.ENTITY_COMPLETENESS,
                        message=f"Entity {entity.get('id', i)} missing required field '{field}'",
                        entity_id=entity.get("id"),
                        field=field,
                    ))

        return issues

    def _check_relationship_validity(self, output: Dict[str, Any]) -> List[QualityIssue]:
        """检查关系有效性"""
        issues = []
        relationships = output.get("relationships", [])
        entities = output.get("entities", [])
        entity_ids = {e.get("id") for e in entities if e.get("id")}

        required_fields = self._entity_classification.get("output_fields", {}).get("edge", {}).get("required", [])
        if not required_fields:
            required_fields = ["source_id", "target_id", "relation_type"]

        for i, relation in enumerate(relationships):
            for field in required_fields:
                if field not in relation or relation[field] is None or str(relation[field]).strip() == "":
                    issues.append(QualityIssue(
                        id=f"relationship_validity_{i}_{field}",
                        severity=IssueSeverity.HIGH,
                        category=IssueCategory.RELATIONSHIP_VALIDITY,
                        message=f"Relationship {i} missing required field '{field}'",
                        field=field,
                    ))

            source_id = relation.get("source_id")
            target_id = relation.get("target_id")
            if source_id and source_id not in entity_ids:
                issues.append(QualityIssue(
                    id=f"relationship_validity_{i}_source",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.RELATIONSHIP_VALIDITY,
                    message=f"Relationship source_id '{source_id}' does not exist in entities",
                    field="source_id",
                    actual=source_id,
                ))
            if target_id and target_id not in entity_ids:
                issues.append(QualityIssue(
                    id=f"relationship_validity_{i}_target",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.RELATIONSHIP_VALIDITY,
                    message=f"Relationship target_id '{target_id}' does not exist in entities",
                    field="target_id",
                    actual=target_id,
                ))

            issues.extend(self._check_layer_pair_rules(source_id, target_id, entities, i))

        return issues

    def _check_layer_pair_rules(self, source_id: str, target_id: str, entities: List[Dict[str, Any]], index: int) -> List[QualityIssue]:
        """检查层间关系规则"""
        issues = []
        layer_pair_rules = self._relationship_rules.get("layer_pair_rules", {})

        if not source_id or not target_id:
            return issues

        source_layer = self._get_entity_layer(source_id, entities)
        target_layer = self._get_entity_layer(target_id, entities)

        if not source_layer or not target_layer:
            return issues

        pair_key = f"{source_layer}-{target_layer}"
        reversed_pair_key = f"{target_layer}-{source_layer}"

        for key in [pair_key, reversed_pair_key]:
            if key in layer_pair_rules:
                rule = layer_pair_rules[key]
                if not rule.get("allowed", True):
                    issues.append(QualityIssue(
                        id=f"relationship_validity_{index}_layer_pair",
                        severity=IssueSeverity.CRITICAL,
                        category=IssueCategory.RELATIONSHIP_VALIDITY,
                        message=f"Layer pair '{key}' is not allowed: {rule.get('notes', '')}",
                        field="layer_pair",
                        actual=key,
                    ))
                break

        return issues

    def _get_entity_layer(self, entity_id: str, entities: List[Dict[str, Any]]) -> Optional[str]:
        """获取实体的层级"""
        for entity in entities:
            if entity.get("id") == entity_id:
                return entity.get("layer")
            if entity.get("meta_id") == entity_id:
                return entity.get("layer")
        return None

    def _check_metadata_completeness(self, output: Dict[str, Any]) -> List[QualityIssue]:
        """检查元数据完整性"""
        issues = []
        entities = output.get("entities", [])
        optional_fields = ["description", "evidence_ref", "source_file"]

        incomplete_count = 0
        for entity in entities:
            missing_opt = [f for f in optional_fields if f not in entity or not entity[f]]
            if missing_opt:
                incomplete_count += 1

        incompleteness_ratio = incomplete_count / len(entities) if entities else 0

        if incompleteness_ratio > 0.5:
            issues.append(QualityIssue(
                id="metadata_completeness_high",
                severity=IssueSeverity.MEDIUM,
                category=IssueCategory.METADATA_COMPLETENESS,
                message=f"High metadata incompleteness ratio: {incompleteness_ratio:.2f} (over 50% entities missing optional fields)",
                expected="<= 0.5",
                actual=f"{incompleteness_ratio:.2f}",
            ))

        return issues

    def _check_domain_consistency(self, output: Dict[str, Any]) -> List[QualityIssue]:
        """检查领域一致性"""
        issues = []
        entities = output.get("entities", [])
        valid_types = self._entity_classification.get("valid_meta_classes", [])

        for i, entity in enumerate(entities):
            entity_type = entity.get("type")
            if entity_type and valid_types and entity_type not in valid_types:
                issues.append(QualityIssue(
                    id=f"domain_consistency_{i}",
                    severity=IssueSeverity.LOW,
                    category=IssueCategory.DOMAIN_CONSISTENCY,
                    message=f"Entity type '{entity_type}' not in valid meta classes",
                    entity_id=entity.get("id"),
                    field="type",
                    actual=entity_type,
                    expected=f"One of: {', '.join(valid_types)}",
                ))

        return issues

    def _check_governance_rules(self, output: Dict[str, Any]) -> List[QualityIssue]:
        """检查治理规则"""
        issues = []
        entities = output.get("entities", [])
        relationships = output.get("relationships", [])

        if len(entities) > 0 and len(relationships) == 0:
            issues.append(QualityIssue(
                id="governance_no_relationships",
                severity=IssueSeverity.MEDIUM,
                category=IssueCategory.GOVERNANCE,
                message=f"{len(entities)} entities but 0 relationships - potential relationship extraction issue",
            ))

        if len(entities) > 0 and len(relationships) < len(entities) * 0.1:
            issues.append(QualityIssue(
                id="governance_low_relationship_ratio",
                severity=IssueSeverity.LOW,
                category=IssueCategory.GOVERNANCE,
                message=f"Low relationship ratio: {len(relationships)} relations for {len(entities)} entities (< 10%)",
            ))

        return issues

    def _get_field_severity(self, field: str) -> IssueSeverity:
        """根据字段类型确定严重级别"""
        critical_fields = ["id", "name"]
        high_fields = ["type", "layer", "source_id", "target_id", "relation_type"]

        if field in critical_fields:
            return IssueSeverity.CRITICAL
        elif field in high_fields:
            return IssueSeverity.HIGH
        return IssueSeverity.MEDIUM

    def _calculate_score(self, issues: List[QualityIssue]) -> QualityScore:
        """计算质量分数"""
        weights = self._quality_config.get("validation_weights", {})
        entity_weight = weights.get("node_completeness", 0.4)
        relation_weight = weights.get("relationship_validity", 0.3)
        metadata_weight = weights.get("metadata_completeness", 0.2)
        domain_weight = weights.get("domain_consistency", 0.1)

        entity_issues = [i for i in issues if i.category == IssueCategory.ENTITY_COMPLETENESS]
        relation_issues = [i for i in issues if i.category == IssueCategory.RELATIONSHIP_VALIDITY]
        metadata_issues = [i for i in issues if i.category == IssueCategory.METADATA_COMPLETENESS]
        domain_issues = [i for i in issues if i.category == IssueCategory.DOMAIN_CONSISTENCY]

        entity_score = max(0, 1 - self._score_by_severity(entity_issues) * 0.5)
        relation_score = max(0, 1 - self._score_by_severity(relation_issues) * 0.5)
        metadata_score = max(0, 1 - self._score_by_severity(metadata_issues) * 0.3)
        domain_score = max(0, 1 - self._score_by_severity(domain_issues) * 0.2)

        overall = (
            entity_score * entity_weight
            + relation_score * relation_weight
            + metadata_score * metadata_weight
            + domain_score * domain_weight
        )

        return QualityScore(
            overall=overall,
            entity_completeness=entity_score,
            relationship_validity=relation_score,
            metadata_completeness=metadata_score,
            domain_consistency=domain_score,
        )

    def _score_by_severity(self, issues: List[QualityIssue]) -> float:
        """根据严重级别计算扣分"""
        severity_weights = {
            IssueSeverity.CRITICAL: 1.0,
            IssueSeverity.HIGH: 0.8,
            IssueSeverity.MEDIUM: 0.5,
            IssueSeverity.LOW: 0.2,
        }

        if not issues:
            return 0.0

        total_weight = sum(severity_weights[i.severity] for i in issues)
        return min(total_weight / len(issues), 1.0)

    def _determine_grade(self, score: float) -> QualityGrade:
        """根据分数确定质量等级"""
        grades = self._quality_config.get("quality_grades", {})
        gold_threshold = grades.get("gold", {}).get("min_confidence", 0.95)
        silver_threshold = grades.get("silver", {}).get("min_confidence", 0.8)

        if score >= gold_threshold:
            return QualityGrade.GOLD
        elif score >= silver_threshold:
            return QualityGrade.SILVER
        return QualityGrade.BRONZE

    def _determine_result(self, report: QualityGateReport) -> GateResult:
        """确定门禁结果"""
        online_gate = self._quality_config.get("online_quality_gate", {})

        if online_gate.get("reject_below_bronze", True) and report.grade == QualityGrade.BRONZE:
            if report.score.overall < 0.6:
                report.rejection_reason = "score_below_block_threshold"
                return GateResult.BLOCK

        if online_gate.get("warn_below_silver", True) and report.grade != QualityGrade.GOLD:
            if report.critical_issue_count > 0:
                report.rejection_reason = "critical_issue_present"
                return GateResult.FAIL
            if report.high_issue_count > 5:
                report.rejection_reason = "too_many_high_issues"
                return GateResult.WARN

        report.rejection_reason = ""
        return GateResult.PASS

    def _log_report(self, report: QualityGateReport):
        """记录质量门禁报告"""
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Quality Gate Report: {report.gate_id}")
        logger.info(f"{'='*60}")
        logger.info(f"Result: {report.result.value}")
        logger.info(f"Grade: {report.grade.value}")
        logger.info(f"Score: {report.score.overall:.3f}")
        logger.info(f"Entity Completeness: {report.score.entity_completeness:.3f}")
        logger.info(f"Relationship Validity: {report.score.relationship_validity:.3f}")
        logger.info(f"Metadata Completeness: {report.score.metadata_completeness:.3f}")
        logger.info(f"Domain Consistency: {report.score.domain_consistency:.3f}")

        if report.issues:
            logger.info(f"\n❌ Issues ({len(report.issues)}):")
            for issue in report.issues[:10]:
                logger.info(f"   [{issue.severity.value}] {issue.category.value}: {issue.message}")

        logger.info(f"\n{'='*60}")


class GovernanceRulesEngine:
    """治理规则引擎 - 统一管理所有治理规则"""

    def __init__(self):
        self._quality_gate = QualityGate()
        self._config = get_unified_config()

    def run_audit(self, pipeline_output: Dict[str, Any]) -> QualityGateReport:
        """执行审计"""
        logger.info("Running governance audit...")
        return self._quality_gate.run(pipeline_output)

    def run_guard(self, pipeline_output: Dict[str, Any]) -> Tuple[bool, QualityGateReport]:
        """执行质量门禁"""
        logger.info("Running governance guard...")
        report = self._quality_gate.run(pipeline_output)

        if report.result in [GateResult.FAIL, GateResult.BLOCK]:
            logger.error(f"Quality gate blocked pipeline execution: {report.result.value}")
            return False, report

        return True, report

    def validate_relationships(self, relationships: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> List[QualityIssue]:
        """验证关系"""
        output = {"entities": entities, "relationships": relationships}
        report = self._quality_gate.run(output)
        return report.issues

    def validate_entities(self, entities: List[Dict[str, Any]]) -> List[QualityIssue]:
        """验证实体"""
        output = {"entities": entities, "relationships": []}
        report = self._quality_gate.run(output)
        return report.issues

    def get_quality_thresholds(self) -> Dict[str, Any]:
        """获取质量阈值配置"""
        return self._config.quality_config.get("quality_grades", {})


def get_governance_engine() -> GovernanceRulesEngine:
    """获取治理规则引擎实例"""
    return GovernanceRulesEngine()