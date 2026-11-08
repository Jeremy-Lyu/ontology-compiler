"""
OntologyRelationship - 本体关系模型

定义 Ontology 中实体之间的关系，包括依赖、包含、实现、使用等。
替代原有的 dict[str, Any] 传递方式，提供类型安全和 IDE 支持。

示例:
    rel = OntologyRelationship(
        source_id="SYS_001",
        target_id="CAP_001",
        relation_type="implements",
        description="InventorySystem implements InventoryManagement capability",
        evidence=Evidence(source_file="system.md", line_range=(100, 120))
    )
"""
from dataclasses import dataclass, field
from typing import Any, Optional

from .evidence import Evidence


@dataclass
class OntologyRelationship:
    """
    本体关系 - 表示两个 Ontology 实体之间的关系。

    Attributes:
        source_id: 源实体 ID（meta_id）
        target_id: 目标实体 ID（meta_id）
        relation_type: 关系类型（implements, depends_on, part_of, uses, manages 等）
        description: 关系描述
        properties: 额外属性字典
        evidence: 证据信息
        confidence: 置信度评分（0.0 - 1.0）
        version: 版本号
        bidirectional: 是否为双向关系
    """
    source_id: str
    target_id: str
    relation_type: str
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    evidence: Optional[Evidence] = None
    confidence: float = 1.0
    version: str = "1.0"
    bidirectional: bool = False

    # Contract metadata (see schemas/ontology_relationship.schema.yaml v1.0.0)
    contract_id: str = "ontology_relationship"
    contract_version: str = "1.0.0"
    rewrite_reason: str = ""

    def __post_init__(self):
        """验证必需字段"""
        if not self.source_id:
            raise ValueError("source_id is required")
        if not self.target_id:
            raise ValueError("target_id is required")
        if not self.relation_type:
            raise ValueError("relation_type is required")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OntologyRelationship":
        """从字典创建关系（兼容旧数据格式）"""
        evidence_data = data.get("evidence")
        evidence = Evidence.from_dict(evidence_data) if evidence_data else None

        return cls(
            source_id=data.get("source_id", ""),
            target_id=data.get("target_id", ""),
            relation_type=data.get("relation_type", ""),
            description=data.get("description", ""),
            properties=data.get("properties", {}),
            evidence=evidence,
            confidence=data.get("confidence", 1.0),
            version=data.get("version", "1.0"),
            bidirectional=data.get("bidirectional", False),
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（兼容旧接口）"""
        return {
            k: v for k, v in {
                "source_id": self.source_id,
                "target_id": self.target_id,
                "relation_type": self.relation_type,
                "description": self.description,
                "properties": self.properties,
                "evidence": self.evidence.to_dict() if self.evidence else None,
                "confidence": self.confidence,
                "version": self.version,
                "bidirectional": self.bidirectional,
                "contract_id": self.contract_id,
                "contract_version": self.contract_version,
                "rewrite_reason": self.rewrite_reason,
            }.items() if v is not None and v != {}
        }

    def get_property(self, key: str, default: Any = None) -> Any:
        """获取属性值"""
        return self.properties.get(key, default)

    def set_property(self, key: str, value: Any) -> None:
        """设置属性值"""
        self.properties[key] = value

    def get_inverse(self) -> "OntologyRelationship":
        """
        获取反向关系（用于双向关系场景）。
        例如：part_of 的反向是 contains
        """
        inverse_map = {
            "part_of": "contains",
            "contains": "part_of",
            "child_of": "parent_of",
            "parent_of": "child_of",
            "depends_on": "depended_by",
            "depended_by": "depends_on",
            "uses": "used_by",
            "used_by": "uses",
            "implements": "implemented_by",
            "implemented_by": "implements",
            "manages": "managed_by",
            "managed_by": "manages",
        }
        inverse_type = inverse_map.get(self.relation_type, self.relation_type)

        return OntologyRelationship(
            source_id=self.target_id,
            target_id=self.source_id,
            relation_type=inverse_type,
            description=f"Inverse of: {self.description}",
            properties=self.properties.copy(),
            evidence=self.evidence,
            confidence=self.confidence,
            version=self.version,
            bidirectional=self.bidirectional,
        )

    def __str__(self) -> str:
        return f"OntologyRelationship({self.source_id} --{self.relation_type}--> {self.target_id})"

    def __hash__(self) -> int:
        """基于 source_id, target_id, relation_type 的哈希"""
        return hash((self.source_id, self.target_id, self.relation_type))

    def __eq__(self, other: object) -> bool:
        """基于 source_id, target_id, relation_type 的相等比较"""
        if not isinstance(other, OntologyRelationship):
            return False
        return (
            self.source_id == other.source_id
            and self.target_id == other.target_id
            and self.relation_type == other.relation_type
        )
