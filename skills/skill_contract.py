"""Skill Contract - Unified skill contract model.

This module consolidates skill_contract.py and unified_skill_contract.py into a single
canonical model. It defines:

Core Contracts:
1. ClassificationContract - 分类契约：适用文档、优先主类、忽略噪声模式
2. EntityContract - 实体契约：必须提取的实体类型、必须字段、层级优先级
3. RelationContract - 关系契约：允许/禁止的关系类型、证据要求
4. SkillContract - 完整契约组合（legacy）
5. SkillContractValidator - 契约验证器

Unified Model:
6. UnifiedSkillContract - 统一技能契约
7. UnifiedSkillContractValidator - 统一契约验证器
8. TriggerCondition - 触发条件
9. FieldDefinition / InputSchema / OutputSchema - 输入输出定义
10. SkillStats / SkillQuality / SkillTrace - 质量和追踪指标

契约化的核心目标：
- 明确技能适用范围
- 约束抽取结果质量
- 为审查提供可量化标准
- 支持自动化验证
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from enum import Enum
from core.schema_registry import get_schema_registry
from ontology_compiler.services.quality_assurance.relation_type_registry import RelationTypeRegistry

_relation_registry = RelationTypeRegistry()


class ContractSeverity(Enum):
    """契约约束的严格程度"""
    OPTIONAL = "optional"
    RECOMMENDED = "recommended"
    REQUIRED = "required"


class MatchType(Enum):
    """匹配类型"""
    CONTAINS = "contains"
    EQUALS = "equals"
    REGEX = "regex"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"


@dataclass
class ClassificationContract:
    """分类契约 - 定义技能适用范围和分类规则"""
    
    name: str = ""
    description: str = ""
    
    applicable_domains: List[str] = field(default_factory=list)
    applicable_document_types: List[str] = field(default_factory=list)
    file_patterns: List[Dict[str, str]] = field(default_factory=list)
    
    priority_main_classes: List[str] = field(default_factory=list)
    ignored_classes: List[str] = field(default_factory=list)
    
    noise_patterns: List[str] = field(default_factory=list)
    required_keywords: List[str] = field(default_factory=list)
    
    def is_applicable(self, domain: str, document_type: str, file_name: str) -> bool:
        """判断技能是否适用于给定文档"""
        if self.applicable_domains and domain not in self.applicable_domains:
            return False
        
        if self.applicable_document_types and document_type not in self.applicable_document_types:
            return False
        
        if self.file_patterns:
            matched = False
            for pattern in self.file_patterns:
                match_type = MatchType(pattern.get("type", "contains"))
                pattern_value = pattern.get("value", "")
                
                if match_type == MatchType.CONTAINS and pattern_value in file_name:
                    matched = True
                    break
                elif match_type == MatchType.EQUALS and file_name == pattern_value:
                    matched = True
                    break
                elif match_type == MatchType.REGEX:
                    import re
                    if re.search(pattern_value, file_name):
                        matched = True
                        break
                elif match_type == MatchType.STARTS_WITH and file_name.startswith(pattern_value):
                    matched = True
                    break
                elif match_type == MatchType.ENDS_WITH and file_name.endswith(pattern_value):
                    matched = True
                    break
            
            if not matched:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "applicable_domains": self.applicable_domains,
            "applicable_document_types": self.applicable_document_types,
            "file_patterns": self.file_patterns,
            "priority_main_classes": self.priority_main_classes,
            "ignored_classes": self.ignored_classes,
            "noise_patterns": self.noise_patterns,
            "required_keywords": self.required_keywords,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassificationContract":
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            applicable_domains=data.get("applicable_domains", []),
            applicable_document_types=data.get("applicable_document_types", []),
            file_patterns=data.get("file_patterns", []),
            priority_main_classes=data.get("priority_main_classes", []),
            ignored_classes=data.get("ignored_classes", []),
            noise_patterns=data.get("noise_patterns", []),
            required_keywords=data.get("required_keywords", []),
        )


@dataclass
class EntityFieldRequirement:
    """实体字段要求"""
    field_name: str
    severity: ContractSeverity = ContractSeverity.OPTIONAL
    valid_values: List[str] = field(default_factory=list)
    min_length: int = 0
    max_length: int = 0
    
    def is_satisfied(self, value: Any) -> bool:
        """检查字段是否满足要求"""
        if self.severity == ContractSeverity.OPTIONAL:
            return True
        
        if self.severity == ContractSeverity.REQUIRED and value is None:
            return False
        
        if value is None:
            return True
        
        if isinstance(value, str):
            if self.min_length > 0 and len(value) < self.min_length:
                return False
            if self.max_length > 0 and len(value) > self.max_length:
                return False
            if self.valid_values and value not in self.valid_values:
                return False
        
        if isinstance(value, list) and self.min_length > 0 and len(value) < self.min_length:
            return False
        
        return True


@dataclass
class EntityContract:
    """实体契约 - 定义必须提取的实体类型、必须字段、层级优先级"""
    
    name: str = ""
    description: str = ""
    entity_type: str = ""
    
    required_entity_types: List[Dict[str, Any]] = field(default_factory=list)
    required_fields: List[EntityFieldRequirement] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    
    priority_layers: List[str] = field(default_factory=lambda: ["L1", "L2"])
    min_entities_per_layer: Dict[str, int] = field(default_factory=dict)
    max_entities_per_layer: Dict[str, int] = field(default_factory=dict)
    
    confidence_threshold: float = 0.7
    evidence_required: bool = False
    
    priority: int = 0
    examples: List[str] = field(default_factory=list)
    
    def get_field_requirement(self, field_name: str) -> Optional[EntityFieldRequirement]:
        """获取字段要求"""
        for req in self.required_fields:
            if req.field_name == field_name:
                return req
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "entity_type": self.entity_type,
            "required_entity_types": self.required_entity_types,
            "required_fields": [
                {
                    "field_name": req.field_name,
                    "severity": req.severity.value,
                    "valid_values": req.valid_values,
                    "min_length": req.min_length,
                    "max_length": req.max_length,
                }
                for req in self.required_fields
            ],
            "optional_fields": self.optional_fields,
            "priority_layers": self.priority_layers,
            "min_entities_per_layer": self.min_entities_per_layer,
            "max_entities_per_layer": self.max_entities_per_layer,
            "confidence_threshold": self.confidence_threshold,
            "evidence_required": self.evidence_required,
            "priority": self.priority,
            "examples": self.examples,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityContract":
        required_fields = []
        for field_data in data.get("required_fields", []):
            required_fields.append(EntityFieldRequirement(
                field_name=field_data["field_name"],
                severity=ContractSeverity(field_data.get("severity", "optional")),
                valid_values=field_data.get("valid_values", []),
                min_length=field_data.get("min_length", 0),
                max_length=field_data.get("max_length", 0),
            ))
        
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            entity_type=data.get("entity_type", ""),
            required_entity_types=data.get("required_entity_types", []),
            required_fields=required_fields,
            optional_fields=data.get("optional_fields", []),
            priority_layers=data.get("priority_layers", ["L1", "L2"]),
            min_entities_per_layer=data.get("min_entities_per_layer", {}),
            max_entities_per_layer=data.get("max_entities_per_layer", {}),
            confidence_threshold=data.get("confidence_threshold", 0.7),
            evidence_required=data.get("evidence_required", False),
            priority=data.get("priority", 0),
            examples=data.get("examples", []),
        )


@dataclass
class RelationContract:
    """关系契约 - 定义允许/禁止的关系类型、证据要求"""

    name: str = ""
    description: str = ""
    relation_type: str = ""
    allowed: bool = True

    allowed_relation_types: List[str] = field(default_factory=list)
    forbidden_relation_types: List[str] = field(default_factory=list)

    priority_families: List[str] = field(default_factory=lambda: ["structural", "semantic"])

    min_relations: int = 0
    max_relations: int = 100

    confidence_threshold: float = 0.7
    evidence_required: bool = False

    requires_existing_entities: bool = True
    
    def is_relation_type_allowed(self, relation_type: str) -> bool:
        """检查关系类型是否允许"""
        normalized_type = _relation_registry.normalize_type(relation_type)
        
        if self.forbidden_relation_types and normalized_type in self.forbidden_relation_types:
            return False
        
        if self.allowed_relation_types and normalized_type not in self.allowed_relation_types:
            return False
        
        return _relation_registry.is_valid_type(normalized_type)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "allowed_relation_types": self.allowed_relation_types,
            "forbidden_relation_types": self.forbidden_relation_types,
            "priority_families": self.priority_families,
            "min_relations": self.min_relations,
            "max_relations": self.max_relations,
            "confidence_threshold": self.confidence_threshold,
            "evidence_required": self.evidence_required,
            "requires_existing_entities": self.requires_existing_entities,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationContract":
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            allowed_relation_types=data.get("allowed_relation_types", []),
            forbidden_relation_types=data.get("forbidden_relation_types", []),
            priority_families=data.get("priority_families", ["structural", "semantic"]),
            min_relations=data.get("min_relations", 0),
            max_relations=data.get("max_relations", 100),
            confidence_threshold=data.get("confidence_threshold", 0.7),
            evidence_required=data.get("evidence_required", False),
            requires_existing_entities=data.get("requires_existing_entities", True),
        )


@dataclass
class SkillContract:
    """完整的 Skill 契约"""
    
    classification: ClassificationContract = field(default_factory=ClassificationContract)
    entity: EntityContract = field(default_factory=EntityContract)
    relation: RelationContract = field(default_factory=RelationContract)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification.to_dict(),
            "entity": self.entity.to_dict(),
            "relation": self.relation.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillContract":
        return cls(
            classification=ClassificationContract.from_dict(data.get("classification", {})),
            entity=EntityContract.from_dict(data.get("entity", {})),
            relation=RelationContract.from_dict(data.get("relation", {})),
        )


@dataclass
class ContractViolation:
    """契约违规记录"""
    contract_type: str
    field: str
    severity: ContractSeverity
    message: str
    entity_id: Optional[str] = None
    relation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_type": self.contract_type,
            "field": self.field,
            "severity": self.severity.value,
            "message": self.message,
            "entity_id": self.entity_id,
            "relation_id": self.relation_id,
        }


class SkillContractValidator:
    """Skill 契约验证器"""
    
    def __init__(self, contract: SkillContract) -> None:
        self._contract = contract
        self._violations: List[ContractViolation] = []
    
    def validate_entities(self, entities: Dict[str, List[Dict]]) -> List[ContractViolation]:
        """验证实体是否符合契约"""
        self._violations = []
        
        for layer, layer_entities in entities.items():
            if isinstance(layer_entities, list):
                self._validate_layer_entities(layer, layer_entities)
        
        return self._violations.copy()
    
    def _validate_layer_entities(self, layer: str, layer_entities: List[Dict]) -> None:
        """验证单层实体"""
        min_required = self._contract.entity.min_entities_per_layer.get(layer, 0)
        max_allowed = self._contract.entity.max_entities_per_layer.get(layer, 0)
        
        if min_required > 0 and len(layer_entities) < min_required:
            self._violations.append(ContractViolation(
                contract_type="entity",
                field=f"layer_{layer}_count",
                severity=ContractSeverity.REQUIRED,
                message=f"Layer {layer} requires at least {min_required} entities, got {len(layer_entities)}",
            ))
        
        if max_allowed > 0 and len(layer_entities) > max_allowed:
            self._violations.append(ContractViolation(
                contract_type="entity",
                field=f"layer_{layer}_count",
                severity=ContractSeverity.RECOMMENDED,
                message=f"Layer {layer} exceeds max {max_allowed} entities, got {len(layer_entities)}",
            ))
        
        for ent in layer_entities:
            self._validate_single_entity(ent)
    
    def _validate_single_entity(self, entity: Dict) -> None:
        """验证单个实体"""
        entity_id = entity.get("id", "unknown")
        
        for field_req in self._contract.entity.required_fields:
            field_value = entity.get(field_req.field_name)
            if not field_req.is_satisfied(field_value):
                self._violations.append(ContractViolation(
                    contract_type="entity",
                    field=field_req.field_name,
                    severity=field_req.severity,
                    message=f"Field '{field_req.field_name}' violates requirement: {field_value}",
                    entity_id=entity_id,
                ))
        
        confidence = entity.get("confidence", 0.0)
        if confidence < self._contract.entity.confidence_threshold:
            self._violations.append(ContractViolation(
                contract_type="entity",
                field="confidence",
                severity=ContractSeverity.RECOMMENDED,
                message=f"Entity confidence {confidence} below threshold {self._contract.entity.confidence_threshold}",
                entity_id=entity_id,
            ))
        
        if self._contract.entity.evidence_required and not entity.get("evidence_ref"):
            self._violations.append(ContractViolation(
                contract_type="entity",
                field="evidence_ref",
                severity=ContractSeverity.REQUIRED,
                message="Entity requires evidence reference",
                entity_id=entity_id,
            ))
    
    def validate_relations(self, relations: List[Dict], entity_ids: Set[str]) -> List[ContractViolation]:
        """验证关系是否符合契约"""
        violations = []
        
        if self._contract.relation.min_relations > 0 and len(relations) < self._contract.relation.min_relations:
            violations.append(ContractViolation(
                contract_type="relation",
                field="count",
                severity=ContractSeverity.RECOMMENDED,
                message=f"Requires at least {self._contract.relation.min_relations} relations, got {len(relations)}",
            ))
        
        if self._contract.relation.max_relations > 0 and len(relations) > self._contract.relation.max_relations:
            violations.append(ContractViolation(
                contract_type="relation",
                field="count",
                severity=ContractSeverity.RECOMMENDED,
                message=f"Exceeds max {self._contract.relation.max_relations} relations, got {len(relations)}",
            ))
        
        for rel in relations:
            rel_type = rel.get("relation_type", "")
            if not self._contract.relation.is_relation_type_allowed(rel_type):
                violations.append(ContractViolation(
                    contract_type="relation",
                    field="relation_type",
                    severity=ContractSeverity.REQUIRED,
                    message=f"Relation type '{rel_type}' is not allowed",
                ))
            
            confidence = rel.get("confidence", 0.0)
            if confidence < self._contract.relation.confidence_threshold:
                violations.append(ContractViolation(
                    contract_type="relation",
                    field="confidence",
                    severity=ContractSeverity.RECOMMENDED,
                    message=f"Relation confidence {confidence} below threshold {self._contract.relation.confidence_threshold}",
                ))
            
            if self._contract.relation.evidence_required and not rel.get("evidence_ref"):
                violations.append(ContractViolation(
                    contract_type="relation",
                    field="evidence_ref",
                    severity=ContractSeverity.REQUIRED,
                    message="Relation requires evidence reference",
                ))
            
            if self._contract.relation.requires_existing_entities:
                source_id = rel.get("source_id", "")
                target_id = rel.get("target_id", "")
                
                if source_id and source_id not in entity_ids:
                    violations.append(ContractViolation(
                        contract_type="relation",
                        field="source_id",
                        severity=ContractSeverity.REQUIRED,
                        message=f"Source entity '{source_id}' does not exist",
                    ))
                
                if target_id and target_id not in entity_ids:
                    violations.append(ContractViolation(
                        contract_type="relation",
                        field="target_id",
                        severity=ContractSeverity.REQUIRED,
                        message=f"Target entity '{target_id}' does not exist",
                    ))
        
        return violations
    
    def validate(self, entities: Dict[str, List[Dict]], relations: List[Dict]) -> List[ContractViolation]:
        """完整验证"""
        entity_violations = self.validate_entities(entities)
        
        entity_ids = set()
        for layer_entities in entities.values():
            if isinstance(layer_entities, list):
                for ent in layer_entities:
                    if isinstance(ent, dict):
                        entity_ids.add(ent.get("id", ""))
        
        relation_violations = self.validate_relations(relations, entity_ids)
        
        return entity_violations + relation_violations
    
    def get_violation_summary(self, violations: List[ContractViolation]) -> Dict[str, Any]:
        """获取违规摘要"""
        summary = {
            "total": len(violations),
            "required": 0,
            "recommended": 0,
            "optional": 0,
            "by_type": {},
        }
        
        for v in violations:
            summary[v.severity.value] += 1
            summary["by_type"].setdefault(v.contract_type, 0)
            summary["by_type"][v.contract_type] += 1
        
        return summary


class SkillStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"


@dataclass
class TriggerCondition:
    keywords: List[str] = field(default_factory=list)
    file_patterns: List[str] = field(default_factory=list)
    min_confidence: float = 0.7
    supported_layers: List[str] = field(default_factory=list)
    domain_tags: List[str] = field(default_factory=list)
    regex_patterns: List[str] = field(default_factory=list)


@dataclass
class FieldDefinition:
    name: str
    type: str = "string"
    required: bool = False
    description: str = ""
    examples: List[str] = field(default_factory=list)


@dataclass
class InputSchema:
    type: str = "object"
    description: str = ""
    fields: List[FieldDefinition] = field(default_factory=list)


@dataclass
class OutputSchema:
    type: str = "object"
    description: str = ""
    fields: List[FieldDefinition] = field(default_factory=list)


@dataclass
class SkillStats:
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time: float = 0.0
    avg_confidence: float = 0.0
    last_used_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None


@dataclass
class SkillQuality:
    confidence: float = 0.0
    completeness: float = 0.0
    evidence_coverage: float = 0.0
    duplicate_rate: float = 0.0
    conflict_count: int = 0
    review_required: bool = False


@dataclass
class SkillTrace:
    model_name: Optional[str] = None
    prompt_version: Optional[str] = None
    skill_version: Optional[str] = None
    execution_time: float = 0.0
    retries: int = 0
    upstream_skills: List[str] = field(default_factory=list)
    run_id: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class UnifiedSkillContract:
    skill_id: str
    name: str
    description: str
    version: str = "1.0.0"
    category: str = ""
    tags: List[str] = field(default_factory=list)
    
    trigger: TriggerCondition = field(default_factory=TriggerCondition)
    
    input_schema: InputSchema = field(default_factory=InputSchema)
    output_schema: OutputSchema = field(default_factory=OutputSchema)
    
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    
    supported_layers: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    business_domain: List[str] = field(default_factory=list)
    
    status: SkillStatus = SkillStatus.ACTIVE
    
    classification_contract: ClassificationContract = field(default_factory=ClassificationContract)
    entity_contracts: List[EntityContract] = field(default_factory=list)
    relation_contracts: List[RelationContract] = field(default_factory=list)
    
    stats: SkillStats = field(default_factory=SkillStats)
    
    quality: SkillQuality = field(default_factory=SkillQuality)
    trace: SkillTrace = field(default_factory=SkillTrace)
    
    llm_prompt_template: Optional[str] = None
    
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    author: Optional[str] = None
    
    def is_applicable(self, domain: str, document_type: str, file_name: str) -> bool:
        if self.classification_contract.applicable_domains and domain not in self.classification_contract.applicable_domains:
            return False
        
        if self.classification_contract.applicable_document_types and document_type not in self.classification_contract.applicable_document_types:
            return False
        
        if self.classification_contract.file_patterns:
            matched = False
            for pattern in self.classification_contract.file_patterns:
                match_type = MatchType(pattern.get("type", "contains"))
                pattern_value = pattern.get("value", "")
                
                if match_type == MatchType.CONTAINS and pattern_value in file_name:
                    matched = True
                    break
                elif match_type == MatchType.EQUALS and file_name == pattern_value:
                    matched = True
                    break
                elif match_type == MatchType.REGEX:
                    import re
                    if re.search(pattern_value, file_name):
                        matched = True
                        break
                elif match_type == MatchType.STARTS_WITH and file_name.startswith(pattern_value):
                    matched = True
                    break
                elif match_type == MatchType.ENDS_WITH and file_name.endswith(pattern_value):
                    matched = True
                    break
            
            if not matched:
                return False
        
        return True
    
    def should_activate(self, text: str) -> bool:
        if not self.trigger.keywords:
            return True
        
        text_lower = text.lower()
        for keyword in self.trigger.keywords:
            if keyword.lower() in text_lower:
                return True
        
        for pattern in self.trigger.regex_patterns:
            import re
            try:
                if re.search(pattern, text):
                    return True
            except re.error:
                pass
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category,
            "tags": self.tags,
            "trigger": {
                "keywords": self.trigger.keywords,
                "file_patterns": self.trigger.file_patterns,
                "min_confidence": self.trigger.min_confidence,
                "supported_layers": self.trigger.supported_layers,
                "domain_tags": self.trigger.domain_tags,
                "regex_patterns": self.trigger.regex_patterns,
            },
            "input_schema": {
                "type": self.input_schema.type,
                "description": self.input_schema.description,
                "fields": [
                    {"name": f.name, "type": f.type, "required": f.required, "description": f.description, "examples": f.examples}
                    for f in self.input_schema.fields
                ],
            },
            "output_schema": {
                "type": self.output_schema.type,
                "description": self.output_schema.description,
                "fields": [
                    {"name": f.name, "type": f.type, "required": f.required, "description": f.description, "examples": f.examples}
                    for f in self.output_schema.fields
                ],
            },
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "supported_layers": self.supported_layers,
            "dependencies": self.dependencies,
            "business_domain": self.business_domain,
            "status": self.status.value,
            "classification_contract": {
                "name": self.classification_contract.name,
                "description": self.classification_contract.description,
                "applicable_domains": self.classification_contract.applicable_domains,
                "applicable_document_types": self.classification_contract.applicable_document_types,
                "file_patterns": self.classification_contract.file_patterns,
                "priority_main_classes": self.classification_contract.priority_main_classes,
                "ignored_classes": self.classification_contract.ignored_classes,
                "noise_patterns": self.classification_contract.noise_patterns,
                "required_keywords": self.classification_contract.required_keywords,
            },
            "entity_contracts": [
                {
                    "entity_type": ec.entity_type,
                    "required_fields": [{"field_name": rf.field_name, "severity": rf.severity.value, "description": rf.description} for rf in ec.required_fields],
                    "optional_fields": ec.optional_fields,
                    "priority": ec.priority,
                    "description": ec.description,
                    "examples": ec.examples,
                }
                for ec in self.entity_contracts
            ],
            "relation_contracts": [
                {
                    "relation_type": rc.relation_type,
                    "allowed": rc.allowed,
                    "description": rc.description,
                    "evidence_required": rc.evidence_required,
                    "confidence_threshold": rc.confidence_threshold,
                }
                for rc in self.relation_contracts
            ],
            "quality": {
                "confidence": self.quality.confidence,
                "completeness": self.quality.completeness,
                "evidence_coverage": self.quality.evidence_coverage,
                "duplicate_rate": self.quality.duplicate_rate,
                "conflict_count": self.quality.conflict_count,
                "review_required": self.quality.review_required,
            },
            "trace": {
                "model_name": self.trace.model_name,
                "prompt_version": self.trace.prompt_version,
                "skill_version": self.trace.skill_version,
                "execution_time": self.trace.execution_time,
                "retries": self.trace.retries,
                "upstream_skills": self.trace.upstream_skills,
                "run_id": self.trace.run_id,
                "timestamp": self.trace.timestamp.isoformat() if self.trace.timestamp else None,
            },
            "llm_prompt_template": self.llm_prompt_template,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "author": self.author,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedSkillContract":
        trigger_data = data.get("trigger", {})
        trigger = TriggerCondition(
            keywords=trigger_data.get("keywords", []),
            file_patterns=trigger_data.get("file_patterns", []),
            min_confidence=trigger_data.get("min_confidence", 0.7),
            supported_layers=trigger_data.get("supported_layers", []),
            domain_tags=trigger_data.get("domain_tags", []),
            regex_patterns=trigger_data.get("regex_patterns", []),
        )
        
        input_schema_data = data.get("input_schema", {})
        input_schema = InputSchema(
            type=input_schema_data.get("type", "object"),
            description=input_schema_data.get("description", ""),
            fields=[
                FieldDefinition(
                    name=f.get("name"),
                    type=f.get("type", "string"),
                    required=f.get("required", False),
                    description=f.get("description", ""),
                    examples=f.get("examples", []),
                )
                for f in input_schema_data.get("fields", [])
            ],
        )
        
        output_schema_data = data.get("output_schema", {})
        output_schema = OutputSchema(
            type=output_schema_data.get("type", "object"),
            description=output_schema_data.get("description", ""),
            fields=[
                FieldDefinition(
                    name=f.get("name"),
                    type=f.get("type", "string"),
                    required=f.get("required", False),
                    description=f.get("description", ""),
                    examples=f.get("examples", []),
                )
                for f in output_schema_data.get("fields", [])
            ],
        )
        
        classification_data = data.get("classification_contract", {})
        classification_contract = ClassificationContract(
            name=classification_data.get("name", ""),
            description=classification_data.get("description", ""),
            applicable_domains=classification_data.get("applicable_domains", []),
            applicable_document_types=classification_data.get("applicable_document_types", []),
            file_patterns=classification_data.get("file_patterns", []),
            priority_main_classes=classification_data.get("priority_main_classes", []),
            ignored_classes=classification_data.get("ignored_classes", []),
            noise_patterns=classification_data.get("noise_patterns", []),
            required_keywords=classification_data.get("required_keywords", []),
        )
        
        entity_contracts = []
        for ec_data in data.get("entity_contracts", []):
            required_fields = [
                EntityFieldRequirement(
                    field_name=rf.get("field_name"),
                    severity=ContractSeverity(rf.get("severity", "required")),
                    description=rf.get("description", ""),
                )
                for rf in ec_data.get("required_fields", [])
            ]
            entity_contracts.append(EntityContract(
                entity_type=ec_data.get("entity_type"),
                required_fields=required_fields,
                optional_fields=ec_data.get("optional_fields", []),
                priority=ec_data.get("priority", 0),
                description=ec_data.get("description", ""),
                examples=ec_data.get("examples", []),
            ))
        
        relation_contracts = [
            RelationContract(
                relation_type=rc.get("relation_type"),
                allowed=rc.get("allowed", True),
                description=rc.get("description", ""),
                evidence_required=rc.get("evidence_required", False),
                confidence_threshold=rc.get("confidence_threshold", 0.5),
            )
            for rc in data.get("relation_contracts", [])
        ]
        
        status = SkillStatus(data.get("status", "active"))
        
        quality_data = data.get("quality", {})
        quality = SkillQuality(
            confidence=quality_data.get("confidence", 0.0),
            completeness=quality_data.get("completeness", 0.0),
            evidence_coverage=quality_data.get("evidence_coverage", 0.0),
            duplicate_rate=quality_data.get("duplicate_rate", 0.0),
            conflict_count=quality_data.get("conflict_count", 0),
            review_required=quality_data.get("review_required", False),
        )
        
        trace_data = data.get("trace", {})
        trace_timestamp_str = trace_data.get("timestamp")
        trace_timestamp = datetime.fromisoformat(trace_timestamp_str) if trace_timestamp_str else None
        trace = SkillTrace(
            model_name=trace_data.get("model_name"),
            prompt_version=trace_data.get("prompt_version"),
            skill_version=trace_data.get("skill_version"),
            execution_time=trace_data.get("execution_time", 0.0),
            retries=trace_data.get("retries", 0),
            upstream_skills=trace_data.get("upstream_skills", []),
            run_id=trace_data.get("run_id"),
            timestamp=trace_timestamp,
        )
        
        return cls(
            skill_id=data["skill_id"],
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0.0"),
            category=data.get("category", ""),
            tags=data.get("tags", []),
            trigger=trigger,
            input_schema=input_schema,
            output_schema=output_schema,
            preconditions=data.get("preconditions", []),
            postconditions=data.get("postconditions", []),
            supported_layers=data.get("supported_layers", []),
            dependencies=data.get("dependencies", []),
            business_domain=data.get("business_domain", []),
            status=status,
            classification_contract=classification_contract,
            entity_contracts=entity_contracts,
            relation_contracts=relation_contracts,
            quality=quality,
            trace=trace,
            llm_prompt_template=data.get("llm_prompt_template"),
            author=data.get("author"),
        )


class UnifiedSkillContractValidator:
    def __init__(self, contract: Optional[UnifiedSkillContract] = None) -> None:
        self._contract = contract
        self._violations: List[Dict[str, Any]] = []
    
    def validate(self, contract: UnifiedSkillContract) -> Dict[str, Any]:
        errors = []
        warnings = []
        
        if not contract.skill_id:
            errors.append("skill_id is required")
        if not contract.name:
            errors.append("name is required")
        if not contract.description:
            errors.append("description is required")
        
        import re
        if not re.match(r"^\d+\.\d+\.\d+$", contract.version):
            warnings.append(f"version '{contract.version}' does not match semantic versioning pattern")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
    
    def validate_entities(self, entities: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        self._violations = []
        
        if not self._contract:
            return self._violations
        
        for layer, layer_entities in entities.items():
            if isinstance(layer_entities, list):
                self._validate_layer_entities(layer, layer_entities)
        
        return self._violations.copy()
    
    def _validate_layer_entities(self, layer: str, layer_entities: List[Dict]) -> None:
        for ent in layer_entities:
            self._validate_single_entity(ent)
    
    def _validate_single_entity(self, entity: Dict) -> None:
        if not self._contract:
            return
        
        entity_id = entity.get("id", "unknown")
        
        for entity_contract in self._contract.entity_contracts:
            entity_type = entity.get("type", "")
            if entity_type != entity_contract.entity_type:
                continue
            
            for field_req in entity_contract.required_fields:
                field_value = entity.get(field_req.field_name)
                if field_req.severity == ContractSeverity.REQUIRED and (field_value is None or str(field_value).strip() == ""):
                    self._violations.append({
                        "contract_type": "entity",
                        "field": field_req.field_name,
                        "severity": field_req.severity.value,
                        "message": f"Field '{field_req.field_name}' violates requirement: {field_value}",
                        "entity_id": entity_id,
                    })
        
        confidence = entity.get("confidence", 0.0)
        if confidence < 0.7:
            self._violations.append({
                "contract_type": "entity",
                "field": "confidence",
                "severity": ContractSeverity.RECOMMENDED.value,
                "message": f"Entity confidence {confidence} below threshold 0.7",
                "entity_id": entity_id,
            })
        
        if not entity.get("evidence") and not entity.get("evidence_ref"):
            self._violations.append({
                "contract_type": "entity",
                "field": "evidence",
                "severity": ContractSeverity.RECOMMENDED.value,
                "message": "Entity missing evidence reference",
                "entity_id": entity_id,
            })
    
    def validate_relations(self, relations: List[Dict]) -> List[Dict[str, Any]]:
        self._violations = []
        
        if not self._contract:
            return self._violations
        
        for rel in relations:
            self._validate_single_relation(rel)
        
        return self._violations.copy()
    
    def _validate_single_relation(self, relation: Dict) -> None:
        if not self._contract:
            return
        
        relation_id = relation.get("id", "unknown")
        relation_type = relation.get("relation_type", "")
        
        allowed_types = [rc.relation_type for rc in self._contract.relation_contracts if rc.allowed]
        forbidden_types = [rc.relation_type for rc in self._contract.relation_contracts if not rc.allowed]
        
        if forbidden_types and relation_type in forbidden_types:
            self._violations.append({
                "contract_type": "relation",
                "field": "relation_type",
                "severity": ContractSeverity.REQUIRED.value,
                "message": f"Relation type '{relation_type}' is forbidden",
                "relation_id": relation_id,
            })
        
        if allowed_types and relation_type not in allowed_types:
            self._violations.append({
                "contract_type": "relation",
                "field": "relation_type",
                "severity": ContractSeverity.RECOMMENDED.value,
                "message": f"Relation type '{relation_type}' not in allowed list",
                "relation_id": relation_id,
            })
        
        confidence = relation.get("confidence", 0.0)
        if confidence < 0.7:
            self._violations.append({
                "contract_type": "relation",
                "field": "confidence",
                "severity": ContractSeverity.RECOMMENDED.value,
                "message": f"Relation confidence {confidence} below threshold 0.7",
                "relation_id": relation_id,
            })
        
        if not relation.get("evidence_ref"):
            self._violations.append({
                "contract_type": "relation",
                "field": "evidence_ref",
                "severity": ContractSeverity.RECOMMENDED.value,
                "message": "Relation missing evidence reference",
                "relation_id": relation_id,
            })
    
    def validate_content(self, entities: Dict[str, List[Dict]], relations: List[Dict]) -> List[Dict[str, Any]]:
        self._violations = []
        
        entity_violations = self.validate_entities(entities)
        relation_violations = self.validate_relations(relations)
        
        self._violations.extend(entity_violations)
        self._violations.extend(relation_violations)
        
        return self._violations.copy()
    
    def get_violation_summary(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary = {
            "total": len(violations),
            "required": 0,
            "recommended": 0,
            "optional": 0,
        }
        
        for v in violations:
            severity = v.get("severity", "optional")
            if severity in summary:
                summary[severity] += 1
        
        return summary