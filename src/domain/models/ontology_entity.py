"""
OntologyEntity - 核心本体实体模型

定义 Ontology 中的核心概念实体，包括系统、能力、参与者、资产、数据、流程等。
替代原有的 dict[str, Any] 传递方式，提供类型安全和 IDE 支持。

[W1 / 2026-06-20] meta_class / meta_subclass 与 schemas/ontology_schema.yaml
（schema_registry）对齐，加载时自动归一化 legacy 别名（如 L1_Domain → Domain）。

示例:
    entity = OntologyEntity(
        meta_id="SYS_001",
        meta_name="InventorySystem",
        meta_class="System",
        meta_subclass="Enterprise_Platform",
        description="库存管理系统",
        domain="SUPPLY_CHAIN",
        evidence=Evidence(source_file="inventory.md", line_range=(10, 50))
    )
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .evidence import Evidence


class AnalysisStatus(str, Enum):
    """分析状态枚举"""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class OntologyEntity:
    """
    本体实体 - 表示 Ontology 中的一个概念节点。

    Attributes:
        meta_id: 全局唯一标识符（如 SYS_001, CAP_002）
        meta_name: 实体名称（如 InventorySystem）
        meta_class: 实体主类（System, Capability, Actor, Asset, Data, Process）
        meta_subclass: 实体子类（如 Enterprise_Platform, Domain）。legacy 别名
            （L1_Domain / L2_SubDomain / L3_Actionable / Equipment_Infrastructure）
            会在 from_dict / __post_init__ 中被自动归一化为标准名。
        description: 实体描述
        domain: 所属业务域（SUPPLY_CHAIN, MANUFACTURING, ENTERPRISE 等）
        properties: 额外属性字典
        evidence: 证据信息（来源文件、位置等）
        confidence: 置信度评分（0.0 - 1.0）
        version: 版本号
    """
    meta_id: str
    meta_name: str
    meta_class: str
    meta_subclass: str = ""
    description: str = ""
    domain: str = "ENTERPRISE"
    properties: dict[str, Any] = field(default_factory=dict)
    evidence: Optional[Evidence] = None
    confidence: float = 1.0
    version: str = "1.0"

    # Contract metadata (see schemas/ontology_entity.schema.yaml v1.0.0)
    contract_id: str = "ontology_entity"
    contract_version: str = "1.0.0"

    def __post_init__(self):
        """验证必需字段并归一化 legacy 子类别名。"""
        if not self.meta_id:
            raise ValueError("meta_id is required")
        if not self.meta_name:
            raise ValueError("meta_name is required")
        if not self.meta_class:
            raise ValueError("meta_class is required")
        # legacy 别名归一化（不强制校验是否合法，避免破坏历史数据加载）
        if self.meta_subclass:
            self.meta_subclass = _normalize_subclass(self.meta_subclass)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OntologyEntity":
        """从字典创建实体（兼容旧数据格式）"""
        evidence_data = data.get("evidence")
        evidence = Evidence.from_dict(evidence_data) if evidence_data else None

        return cls(
            meta_id=data.get("meta_id", ""),
            meta_name=data.get("meta_name", ""),
            meta_class=data.get("meta_class", ""),
            meta_subclass=data.get("meta_subclass", ""),
            description=data.get("description", ""),
            domain=data.get("domain", "ENTERPRISE"),
            properties=data.get("properties", {}),
            evidence=evidence,
            confidence=data.get("confidence", 1.0),
            version=data.get("version", "1.0"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta_id": self.meta_id,
            "meta_name": self.meta_name,
            "meta_class": self.meta_class,
            "meta_subclass": self.meta_subclass,
            "description": self.description,
            "domain": self.domain,
            "properties": self.properties,
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "confidence": self.confidence,
            "version": self.version,
            "contract_id": self.contract_id,
            "contract_version": self.contract_version,
        }

    def get_property(self, key: str, default: Any = None) -> Any:
        """获取属性值"""
        return self.properties.get(key, default)

    def set_property(self, key: str, value: Any) -> None:
        """设置属性值"""
        self.properties[key] = value

    def has_property(self, key: str) -> bool:
        """检查是否存在属性"""
        return key in self.properties

    def __str__(self) -> str:
        return f"OntologyEntity({self.meta_class}:{self.meta_name})"

    def __hash__(self) -> int:
        """基于 meta_id 的哈希，支持在集合中使用"""
        return hash(self.meta_id)

    def __eq__(self, other: object) -> bool:
        """基于 meta_id 的相等比较"""
        if not isinstance(other, OntologyEntity):
            return False
        return self.meta_id == other.meta_id


def _normalize_subclass(subclass: str) -> str:
    """归一化 legacy 子类别名。

    优先使用 schema_registry；schema_registry 加载失败时退回内置最小映射，
    保证 OntologyEntity 不会因 schema 文件缺失而崩溃。
    """
    try:
        from core.schema_registry import get_schema_registry
        return get_schema_registry().normalize_subclass(subclass)
    except Exception:
        _FALLBACK = {
            "L1_Domain": "Domain",
            "L2_SubDomain": "SubDomain",
            "L3_Actionable": "Actionable",
            "Equipment_Infrastructure": "Equipment_Infra",
        }
        return _FALLBACK.get(subclass, subclass)
