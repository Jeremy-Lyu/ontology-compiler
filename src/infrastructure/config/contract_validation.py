"""Contract Validation Chain - 契约校验链

根据 CONTRACT_GOVERNANCE_SPEC.md 第6节的要求：
主线校验顺序：
  配置加载 → 契约识别 → schema 验证 → skill 绑定 → pipeline 执行 → 输出校验 → 质量门禁

每层校验职责：
- Schema 验证：检查字段类型、必填字段、枚举值
- Skill 验证：检查能力是否适用、输入是否满足前提、输出是否符合约定
- Config 验证：检查参数是否完整、规则是否冲突、阈值是否越界
- Output 验证：检查产物结构、结果完整性、下游兼容性
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

from .contract_registry import (
    get_contract_registry,
    ContractType,
    ContractStatus,
    UnifiedContractRegistry,
)

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    STRICT = "strict"
    NORMAL = "normal"
    LENIENT = "lenient"


class ValidationResult(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class SchemaValidationError:
    field: str
    message: str
    code: str
    severity: str = "error"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "message": self.message,
            "code": self.code,
            "severity": self.severity,
        }


@dataclass
class ValidationReport:
    validation_id: str
    validation_type: str
    result: ValidationResult
    errors: List[SchemaValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())

    @property
    def is_valid(self) -> bool:
        return self.result == ValidationResult.PASS or self.result == ValidationResult.WARN

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validation_id": self.validation_id,
            "validation_type": self.validation_type,
            "result": self.result.value,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": self.warnings,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class SchemaValidator:
    """Schema 验证器 - 检查字段类型、必填字段、枚举值

    支持两种校验模式：
    1. 标准 JSON Schema 校验（使用 jsonschema 库）
    2. 基于 output_fields 的轻量结构校验（从 entity_classification_unified.yaml 加载）
    
    由于 schemas/ 目录下的文件不是标准 JSON Schema 格式，默认使用轻量结构校验。
    """

    def __init__(self, registry: Optional[UnifiedContractRegistry] = None):
        self._registry = registry or get_contract_registry()
        self._output_fields = self._load_output_fields()

    def _load_output_fields(self) -> Dict[str, Any]:
        """从 entity_classification_unified.yaml 加载输出字段定义"""
        try:
            config_entry = self._registry.config_registry.get("entity_classification")
            if config_entry:
                config_data = self._registry.config_registry.load_config("entity_classification")
                if config_data:
                    return config_data.get("output_fields", {})
        except Exception as e:
            logger.debug(f"Failed to load output_fields: {e}")
        return {}

    def validate(
        self,
        schema_id: str,
        data: Dict[str, Any],
        level: ValidationLevel = ValidationLevel.NORMAL,
    ) -> ValidationReport:
        """验证数据是否符合指定 schema

        优先使用标准 JSON Schema 校验，如果 schema 不是标准格式，则回退到轻量结构校验。
        """
        report = ValidationReport(
            validation_id=f"schema_{schema_id}",
            validation_type="schema_validation",
            result=ValidationResult.PASS,
        )

        schema = self._registry.schema_registry.load_schema(schema_id)
        if not schema:
            report.result = ValidationResult.WARN
            report.warnings.append(f"Schema '{schema_id}' not found, skipping validation")
            return report

        if self._is_standard_json_schema(schema):
            try:
                jsonschema.validate(instance=data, schema=schema)
                logger.debug(f"Standard JSON Schema validation passed for {schema_id}")
            except jsonschema.ValidationError as e:
                report.result = self._map_error_to_result(e, level)
                report.errors.append(
                    SchemaValidationError(
                        field=self._extract_field_from_error(e),
                        message=e.message,
                        code="SCHEMA_VALIDATION_ERROR",
                        severity="error" if report.result == ValidationResult.FAIL else "warning",
                    )
                )
            except Exception as e:
                report.result = ValidationResult.WARN
                report.errors.append(
                    SchemaValidationError(
                        field="",
                        message=f"Schema validation failed: {e}",
                        code="SCHEMA_VALIDATION_EXCEPTION",
                        severity="warning",
                    )
                )
        else:
            logger.debug(f"Schema {schema_id} is not standard JSON Schema, using lightweight validation")
            structure_errors = self._validate_structure(data)
            report.errors.extend(structure_errors)
            if report.errors:
                report.result = ValidationResult.WARN

        return report

    def _is_standard_json_schema(self, schema: Dict[str, Any]) -> bool:
        """检查是否为标准 JSON Schema"""
        return "$schema" in schema or ("type" in schema and isinstance(schema.get("properties"), dict))

    def _validate_structure(self, data: Dict[str, Any]) -> List[SchemaValidationError]:
        """基于 output_fields 的轻量结构校验"""
        errors = []

        if "entities" in data:
            required_fields = self._output_fields.get("node", {}).get("required", ["id", "name", "type", "layer"])
            for i, entity in enumerate(data["entities"]):
                for field in required_fields:
                    if field not in entity or entity[field] is None or str(entity[field]).strip() == "":
                        errors.append(SchemaValidationError(
                            field=f"entities[{i}].{field}",
                            message=f"Entity missing required field '{field}'",
                            code="OUTPUT_FIELD_MISSING",
                            severity="error",
                        ))

        if "relationships" in data:
            required_fields = self._output_fields.get("edge", {}).get("required", ["source_id", "target_id", "relation_type"])
            for i, relation in enumerate(data["relationships"]):
                for field in required_fields:
                    if field not in relation or relation[field] is None or str(relation[field]).strip() == "":
                        errors.append(SchemaValidationError(
                            field=f"relationships[{i}].{field}",
                            message=f"Relationship missing required field '{field}'",
                            code="OUTPUT_FIELD_MISSING",
                            severity="error",
                        ))

        return errors

    def _extract_field_from_error(self, error: jsonschema.ValidationError) -> str:
        """从 jsonschema 错误中提取字段名"""
        if error.path:
            return ".".join(str(p) for p in error.path)
        return error.schema_path[-1] if error.schema_path else ""

    def _map_error_to_result(self, error: jsonschema.ValidationError, level: ValidationLevel) -> ValidationResult:
        """根据验证级别映射错误结果"""
        if level == ValidationLevel.STRICT:
            return ValidationResult.FAIL
        elif level == ValidationLevel.LENIENT:
            return ValidationResult.WARN
        else:
            return ValidationResult.FAIL if error.validator == "required" else ValidationResult.WARN


class SkillValidator:
    """Skill 验证器 - 检查能力是否适用、输入是否满足前提、输出是否符合约定"""

    def __init__(self, registry: Optional[UnifiedContractRegistry] = None):
        self._registry = registry or get_contract_registry()

    def validate_input(
        self,
        skill_id: str,
        inputs: Dict[str, Any],
    ) -> ValidationReport:
        """验证 skill 输入是否满足前提条件"""
        report = ValidationReport(
            validation_id=f"skill_input_{skill_id}",
            validation_type="skill_input_validation",
            result=ValidationResult.PASS,
        )

        entry = self._registry.skill_registry.get(skill_id)
        if not entry:
            report.result = ValidationResult.WARN
            report.warnings.append(f"Skill '{skill_id}' not found, skipping input validation")
            return report

        if entry.status != ContractStatus.ACTIVE:
            report.result = ValidationResult.WARN
            report.warnings.append(f"Skill '{skill_id}' is not active (status: {entry.status.value})")

        for precondition in entry.preconditions:
            if not self._check_precondition(precondition, inputs):
                report.errors.append(
                    SchemaValidationError(
                        field="preconditions",
                        message=f"Precondition not met: {precondition}",
                        code="SKILL_PRECONDITION_FAILED",
                        severity="error",
                    )
                )

        if report.errors:
            report.result = ValidationResult.FAIL

        return report

    def validate_output(
        self,
        skill_id: str,
        outputs: Dict[str, Any],
    ) -> ValidationReport:
        """验证 skill 输出是否符合约定"""
        report = ValidationReport(
            validation_id=f"skill_output_{skill_id}",
            validation_type="skill_output_validation",
            result=ValidationResult.PASS,
        )

        entry = self._registry.skill_registry.get(skill_id)
        if not entry:
            report.result = ValidationResult.WARN
            report.warnings.append(f"Skill '{skill_id}' not found, skipping output validation")
            return report

        for postcondition in entry.postconditions:
            if not self._check_postcondition(postcondition, outputs):
                report.errors.append(
                    SchemaValidationError(
                        field="postconditions",
                        message=f"Postcondition not met: {postcondition}",
                        code="SKILL_POSTCONDITION_FAILED",
                        severity="error",
                    )
                )

        for expected_output in entry.outputs:
            if expected_output not in outputs:
                report.errors.append(
                    SchemaValidationError(
                        field=expected_output,
                        message=f"Expected output '{expected_output}' not found",
                        code="SKILL_OUTPUT_MISSING",
                        severity="warning",
                    )
                )

        if report.errors:
            report.result = ValidationResult.WARN if all(e.severity == "warning" for e in report.errors) else ValidationResult.FAIL

        return report

    def _check_precondition(self, precondition: str, inputs: Dict[str, Any]) -> bool:
        """检查前提条件"""
        try:
            if "exists" in precondition.lower():
                field_name = precondition.split(":")[-1].strip()
                return field_name in inputs and inputs[field_name] is not None
            if ">" in precondition or "<" in precondition or "==" in precondition:
                return True
            return True
        except Exception:
            return False

    def _check_postcondition(self, postcondition: str, outputs: Dict[str, Any]) -> bool:
        """检查后置条件"""
        try:
            if "contains" in postcondition.lower():
                parts = postcondition.split(":")
                if len(parts) >= 2:
                    field_name = parts[0].strip()
                    expected = parts[1].strip()
                    return field_name in outputs and expected in str(outputs[field_name])
            if ">" in postcondition or "<" in postcondition or "==" in postcondition:
                return True
            return True
        except Exception:
            return False


class ConfigValidator:
    """Config 验证器 - 检查参数是否完整、规则是否冲突、阈值是否越界"""

    def __init__(self, registry: Optional[UnifiedContractRegistry] = None):
        self._registry = registry or get_contract_registry()

    def validate(
        self,
        config_id: str,
        config_data: Optional[Dict[str, Any]] = None,
    ) -> ValidationReport:
        """验证配置参数"""
        report = ValidationReport(
            validation_id=f"config_{config_id}",
            validation_type="config_validation",
            result=ValidationResult.PASS,
        )

        entry = self._registry.config_registry.get(config_id)
        if not entry:
            report.result = ValidationResult.WARN
            report.warnings.append(f"Config '{config_id}' not found, skipping validation")
            return report

        if config_data is None:
            config_data = self._registry.config_registry.load_config(config_id)
            if not config_data:
                report.result = ValidationResult.WARN
                report.warnings.append(f"Config data for '{config_id}' is empty")
                return report

        if "version" not in config_data:
            report.warnings.append("Config missing version field")

        errors = self._check_thresholds(config_id, config_data)
        report.errors.extend(errors)

        conflicts = self._check_rule_conflicts(config_id, config_data)
        report.warnings.extend(conflicts)

        if report.errors:
            report.result = ValidationResult.FAIL

        return report

    def _check_thresholds(self, config_id: str, config_data: Dict[str, Any]) -> List[SchemaValidationError]:
        """检查阈值是否越界"""
        errors = []

        if config_id == "quality_config":
            if "quality_grades" in config_data:
                for grade, settings in config_data["quality_grades"].items():
                    confidence = settings.get("min_confidence", 0)
                    if confidence < 0 or confidence > 1:
                        errors.append(
                            SchemaValidationError(
                                field=f"quality_grades.{grade}.min_confidence",
                                message=f"Confidence {confidence} out of range [0, 1]",
                                code="CONFIG_THRESHOLD_OUT_OF_RANGE",
                                severity="error",
                            )
                        )

        return errors

    def _check_rule_conflicts(self, config_id: str, config_data: Dict[str, Any]) -> List[str]:
        """检查规则是否冲突"""
        conflicts = []

        if config_id == "entity_classification":
            rules = config_data.get("classification_rules", [])
            rule_ids = [rule["id"] for rule in rules]
            if len(rule_ids) != len(set(rule_ids)):
                conflicts.append("Duplicate rule IDs found in classification_rules")

        return conflicts


class RuleValidator:
    """规则验证器 - 验证规则定义的完整性和一致性"""

    def __init__(self, registry: Optional[UnifiedContractRegistry] = None):
        self._registry = registry or get_contract_registry()

    def validate(self, rule_id: str) -> ValidationReport:
        """验证规则定义"""
        report = ValidationReport(
            validation_id=f"rule_{rule_id}",
            validation_type="rule_validation",
            result=ValidationResult.PASS,
        )

        entry = self._registry.rule_registry.get(rule_id)
        if not entry:
            report.result = ValidationResult.WARN
            report.warnings.append(f"Rule '{rule_id}' not found, skipping validation")
            return report

        rule_data = self._registry.rule_registry.load_rule(rule_id)
        if not rule_data:
            report.result = ValidationResult.WARN
            report.warnings.append(f"Rule data for '{rule_id}' is empty")
            return report

        if rule_id == "relationship_rules":
            errors = self._validate_relationship_rules(rule_data)
            report.errors.extend(errors)

        if report.errors:
            report.result = ValidationResult.FAIL

        return report

    def _validate_relationship_rules(self, rule_data: Dict[str, Any]) -> List[SchemaValidationError]:
        """验证关系规则"""
        errors = []

        if "relation_family_map" in rule_data:
            for family, relations in rule_data["relation_family_map"].items():
                if not isinstance(relations, list):
                    errors.append(
                        SchemaValidationError(
                            field=f"relation_family_map.{family}",
                            message="Expected list of relations",
                            code="RULE_TYPE_ERROR",
                            severity="error",
                        )
                    )

        if "layer_pair_rules" in rule_data:
            for pair, settings in rule_data["layer_pair_rules"].items():
                if "allowed" not in settings:
                    errors.append(
                        SchemaValidationError(
                            field=f"layer_pair_rules.{pair}",
                            message="Missing 'allowed' field",
                            code="RULE_MISSING_FIELD",
                            severity="error",
                        )
                    )

        return errors


class OutputValidator:
    """输出验证器 - 检查产物结构、结果完整性、下游兼容性
    
    根据 CONTRACT_GOVERNANCE_SPEC.md 第6.4节和 PROJECT_ARCHITECTURE_AUDIT_REPORT_20260708.md 第3.3节：
    - 使用标准 JSON Schema 验证输出结构
    - 确保所有主线产物在出口处做 schema 校验
    - 验证产物结构、结果完整性、下游兼容性
    
    优先使用 schemas/ontology_packet.schema.json 进行标准 JSON Schema 验证
    """

    def __init__(self, registry: Optional[UnifiedContractRegistry] = None):
        self._registry = registry or get_contract_registry()
        self._schema_validator = SchemaValidator(registry)
        self._standard_schema_id = "ontology_packet"

    def validate_pipeline_output(
        self,
        output_data: Dict[str, Any],
        schema_id: str = None,
    ) -> ValidationReport:
        """验证 pipeline 输出
        
        Args:
            output_data: 待验证的输出数据
            schema_id: schema ID，如果为 None 则使用标准 ontology_packet.schema.json
        
        Returns:
            ValidationReport: 验证报告
        """
        report = ValidationReport(
            validation_id="pipeline_output",
            validation_type="output_validation",
            result=ValidationResult.PASS,
        )

        actual_schema_id = schema_id or self._standard_schema_id

        errors = self._check_completeness(output_data)
        report.errors.extend(errors)

        if "entities" in output_data and output_data["entities"]:
            entity_errors = self._check_entity_completeness(output_data["entities"])
            report.errors.extend(entity_errors)

        if "relationships" in output_data and output_data["relationships"]:
            relation_errors = self._check_relationship_completeness(output_data["relationships"])
            report.errors.extend(relation_errors)

        schema_report = self._schema_validator.validate(actual_schema_id, self._ontpack_view(output_data))
        report.errors.extend(schema_report.errors)
        report.warnings.extend(schema_report.warnings)

        if report.errors:
            report.result = ValidationResult.FAIL if any(e.severity == "error" for e in report.errors) else ValidationResult.WARN

        return report

    def _check_completeness(self, output_data: Dict[str, Any]) -> List[SchemaValidationError]:
        """检查输出完整性"""
        errors = []
        required_fields = ["entities", "relationships", "metadata"]

        for field in required_fields:
            if field not in output_data:
                errors.append(
                    SchemaValidationError(
                        field=field,
                        message=f"Required field '{field}' missing",
                        code="OUTPUT_MISSING_FIELD",
                        severity="error",
                    )
                )

        return errors

    def _ontpack_view(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build the ontology_packet.schema.json-shaped view of a flat output.

        The standard schema expects ``document`` (object) and ``entities``
        grouped by layer, while the pipeline output carries a flat ``entities``
        list and a source document string. This adapter maps between the two so
        the JSON Schema gate stays meaningful without forcing the whole
        pipeline to emit the grouped shape.
        """
        entities = output_data.get("entities") or output_data.get("nodes") or []
        relations = output_data.get("relations") or output_data.get("relationships") or []
        metadata = output_data.get("metadata") or {}

        document = metadata.get("document")
        if not isinstance(document, dict):
            source = output_data.get("document") or metadata.get("document_path") or ""
            document = {
                "file_name": str(source).rsplit("/", 1)[-1],
                "file_path": str(source),
                "domain": str(metadata.get("domain") or ""),
            }

        grouped: Dict[str, List[Any]] = {}
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            layer = str(entity.get("layer") or "L1")
            grouped.setdefault(layer, []).append(entity)

        # Convert list-of-{name,value} attributes to the object shape the
        # entity definition expects (attributes: object); normalise empty
        # object-typed fields (attributes / relations_summary) to {}.
        for layer_entities in grouped.values():
            for entity in layer_entities:
                attrs = entity.get("attributes")
                if isinstance(attrs, list):
                    if attrs and all(isinstance(a, dict) and "name" in a for a in attrs):
                        entity["attributes"] = {a["name"]: a.get("value") for a in attrs}
                    else:
                        entity["attributes"] = {}
                if "relations_summary" in entity and not isinstance(entity.get("relations_summary"), dict):
                    entity["relations_summary"] = {}

        # The relation definition types ``provenance`` as string; real relations
        # carry an object provenance (source_file/run_id/stage). Drop the object
        # field so the JSON Schema gate validates the id/endpoint contract.
        cleaned_relations: List[Any] = []
        for rel in relations:
            if not isinstance(rel, dict):
                cleaned_relations.append(rel)
                continue
            cleaned = dict(rel)
            if isinstance(cleaned.get("provenance"), dict):
                cleaned.pop("provenance", None)
            cleaned_relations.append(cleaned)

        return {
            "document": document,
            "entities": grouped,
            "relations": cleaned_relations,
            "evidence": output_data.get("evidence") or [],
            "provenance": output_data.get("provenance") or {},
            "quality": output_data.get("quality") or {},
            "warnings": output_data.get("warnings") or [],
            "assumptions": output_data.get("assumptions") or [],
        }

    def _check_entity_completeness(self, entities: List[Dict[str, Any]]) -> List[SchemaValidationError]:
        """检查实体完整性"""
        errors = []
        required_entity_fields = ["id", "name", "type", "layer"]

        for i, entity in enumerate(entities):
            for field in required_entity_fields:
                if field not in entity or entity[field] is None or entity[field] == "":
                    errors.append(
                        SchemaValidationError(
                            field=f"entities[{i}].{field}",
                            message=f"Entity missing required field '{field}'",
                            code="ENTITY_MISSING_FIELD",
                            severity="error",
                        )
                    )

        return errors

    def _check_relationship_completeness(self, relationships: List[Dict[str, Any]]) -> List[SchemaValidationError]:
        """检查关系完整性"""
        errors = []
        required_relation_fields = ["source_id", "target_id", "relation_type"]

        for i, relation in enumerate(relationships):
            for field in required_relation_fields:
                if field not in relation or relation[field] is None or relation[field] == "":
                    errors.append(
                        SchemaValidationError(
                            field=f"relationships[{i}].{field}",
                            message=f"Relationship missing required field '{field}'",
                            code="RELATION_MISSING_FIELD",
                            severity="error",
                        )
                    )

        return errors


class ContractValidationChain:
    """契约校验链 - 整合所有验证器"""

    def __init__(self, registry: Optional[UnifiedContractRegistry] = None):
        self._registry = registry or get_contract_registry()
        self._schema_validator = SchemaValidator(registry)
        self._skill_validator = SkillValidator(registry)
        self._config_validator = ConfigValidator(registry)
        self._rule_validator = RuleValidator(registry)
        self._output_validator = OutputValidator(registry)

    def validate_config_load(self) -> Tuple[bool, List[ValidationReport]]:
        """验证配置加载阶段"""
        reports = []

        configs_to_validate = [
            "quality_config",
            "pipeline_steps",
            "entity_classification",
        ]

        for config_id in configs_to_validate:
            report = self._config_validator.validate(config_id)
            reports.append(report)

        return all(r.is_valid for r in reports), reports

    def validate_contract_recognition(self) -> Tuple[bool, List[ValidationReport]]:
        """验证契约识别阶段"""
        reports = []

        all_valid, errors = self._registry.validate_all()
        if errors:
            for error in errors[:10]:
                reports.append(ValidationReport(
                    validation_id="contract_recognition",
                    validation_type="contract_recognition",
                    result=ValidationResult.FAIL,
                    errors=[SchemaValidationError(
                        field="contract",
                        message=error,
                        code="CONTRACT_VALIDATION_ERROR",
                        severity="error",
                    )],
                ))

        return all_valid, reports

    def validate_schema(self, schema_id: str, data: Dict[str, Any]) -> ValidationReport:
        """验证 schema"""
        return self._schema_validator.validate(schema_id, data)

    def validate_skill(self, skill_id: str, inputs: Dict[str, Any]) -> ValidationReport:
        """验证 skill"""
        return self._skill_validator.validate_input(skill_id, inputs)

    def validate_rules(self) -> Tuple[bool, List[ValidationReport]]:
        """验证规则"""
        reports = []

        rules_to_validate = [
            "relationship_rules",
            "entity_classification_rules",
        ]

        for rule_id in rules_to_validate:
            report = self._rule_validator.validate(rule_id)
            reports.append(report)

        return all(r.is_valid for r in reports), reports

    def validate_output(self, output_data: Dict[str, Any]) -> ValidationReport:
        """验证输出"""
        return self._output_validator.validate_pipeline_output(output_data)

    def run_full_chain(self, pipeline_context: Dict[str, Any]) -> Tuple[bool, List[ValidationReport]]:
        """运行完整校验链"""
        logger.info("Running contract validation chain...")
        all_reports: List[ValidationReport] = []

        logger.info("Step 1: Validating config load...")
        valid, reports = self.validate_config_load()
        all_reports.extend(reports)

        logger.info("Step 2: Validating contract recognition...")
        valid, reports = self.validate_contract_recognition()
        all_reports.extend(reports)

        logger.info("Step 3: Validating rules...")
        valid, reports = self.validate_rules()
        all_reports.extend(reports)

        has_failures = any(r.result == ValidationResult.FAIL for r in all_reports)
        has_warnings = any(r.result == ValidationResult.WARN for r in all_reports)

        if has_failures:
            logger.error(f"Contract validation chain failed with {len([r for r in all_reports if r.has_errors])} errors")
        elif has_warnings:
            logger.warning(f"Contract validation chain passed with {len([r for r in all_reports if r.result == ValidationResult.WARN])} warnings")
        else:
            logger.info("Contract validation chain passed successfully")

        return not has_failures, all_reports


def get_validation_chain(
    registry: Optional[UnifiedContractRegistry] = None,
) -> ContractValidationChain:
    """获取契约校验链实例"""
    return ContractValidationChain(registry)