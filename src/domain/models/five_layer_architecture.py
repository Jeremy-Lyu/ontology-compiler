"""5层架构模型定义

定义5层架构（L0-L3）的完整数据模型和类型系统。

层级说明：
- L0: 原始证据层 - 存储原始文档和LLM响应
- L1: 候选实体层 - 提取的候选实体和关系
- L2: 规范实体层 - 规范化、去重后的实体和关系
- L2.5: 业务语义层 - 业务关系、语义聚类
- L3: 跨文件关系层 - 跨文件证据、文件夹关系

架构设计原则：
1. 每层独立但可追溯
2. 支持增量更新
3. 保留完整的证据链
4. 支持质量评估和置信度计算
"""

from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


from .canonical import LayerType
from .evidence import Evidence

FiveLayerType = LayerType
RawEvidence = Evidence


class LayerMetadata(BaseModel):
    """层元数据"""

    layer_type: FiveLayerType
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"
    source_file: str = ""
    run_id: str = ""
    source_run_id: str = ""
    parent_layer_ids: List[str] = Field(default_factory=list)
    transformation_count: int = 0
    confidence: float = 0.0
    quality_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# L0: 原始证据层
# ============================================================================

class L0LayerArtifact(BaseModel):
    """L0层产物 - 原始证据层"""

    run_id: str
    source_file: str
    evidences: List[RawEvidence] = Field(default_factory=list)

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def layer(self) -> FiveLayerType:
        return FiveLayerType.L0_RAW_EVIDENCE

    @property
    def evidence_count(self) -> int:
        return len(self.evidences)

    @property
    def total_evidences(self) -> int:
        return self.evidence_count

    def add_evidence(self, evidence: RawEvidence) -> None:
        """添加原始证据"""
        self.evidences.append(evidence)


# ============================================================================
# L1: 候选实体层
# ============================================================================

class CandidateEntity(BaseModel):
    """候选实体"""

    entity_id: str
    name: str
    entity_type: str = "Concept"
    description: str = ""

    status: str = "raw"
    confidence: float = 0.0
    quality_score: float = 0.0

    source_evidence_ids: List[str] = Field(default_factory=list)
    source_file: str = ""

    conflicts: List[str] = Field(default_factory=list)

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CandidateRelation(BaseModel):
    """候选关系"""

    relation_id: str
    source_entity: str
    target_entity: str
    relation_type: str = "related_to"
    description: str = ""

    status: str = "raw"
    confidence: float = 0.0
    weight: float = 0.0

    source_evidence_ids: List[str] = Field(default_factory=list)
    source_file: str = ""

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class L1LayerArtifact(BaseModel):
    """L1层产物 - 候选实体层"""

    run_id: str
    source_file: str
    entities: List[CandidateEntity] = Field(default_factory=list)
    relations: List[CandidateRelation] = Field(default_factory=list)

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)
    transformation_count: int = 0

    @property
    def layer(self) -> FiveLayerType:
        return FiveLayerType.L1_CANDIDATE

    @property
    def layer_metadata(self) -> LayerMetadata:
        return LayerMetadata(
            layer_type=FiveLayerType.L1_CANDIDATE,
            transformation_count=self.transformation_count,
        )

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def relation_count(self) -> int:
        return len(self.relations)

    @property
    def total_entities(self) -> int:
        return self.entity_count

    @property
    def total_relations(self) -> int:
        return self.relation_count

    def add_entity(self, entity: CandidateEntity) -> None:
        """添加候选实体"""
        self.entities.append(entity)

    def add_relation(self, relation: CandidateRelation) -> None:
        """添加候选关系"""
        self.relations.append(relation)


# ============================================================================
# L2: 规范实体层
# ============================================================================

class CanonicalEntityLayer(BaseModel):
    """规范实体层"""

    entity_id: str
    name: str
    entity_type: str = "Concept"
    layer: str = "L2"
    description: str = ""

    status: str = "canonical"
    confidence: float = 0.0
    quality_score: float = 0.0

    candidate_ids: List[str] = Field(default_factory=list)
    evidence_count: int = 0
    source_files: Set[str] = Field(default_factory=set)

    properties: Dict[str, Any] = Field(default_factory=dict)
    aliases: List[str] = Field(default_factory=list)

    version: str = "1.0"
    lineage: List[str] = Field(default_factory=list)

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CanonicalRelationLayer(BaseModel):
    """规范关系层"""

    relation_id: str
    source_id: str
    target_id: str
    relation_type: str = "related_to"
    description: str = ""

    status: str = "canonical"
    confidence: float = 0.0
    weight: float = 0.0
    layer: str = "L2"

    candidate_ids: List[str] = Field(default_factory=list)
    evidence_count: int = 0

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class L2LayerArtifact(BaseModel):
    """L2层产物 - 规范实体层"""

    run_id: str
    source_file: str
    entities: List[CanonicalEntityLayer] = Field(default_factory=list)
    relations: List[CanonicalRelationLayer] = Field(default_factory=list)

    deduplication_count: int = 0
    conflict_resolution_count: int = 0

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def layer(self) -> FiveLayerType:
        return FiveLayerType.L2_CANONICAL

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def relation_count(self) -> int:
        return len(self.relations)

    @property
    def total_entities(self) -> int:
        return self.entity_count

    @property
    def total_relations(self) -> int:
        return self.relation_count

    @property
    def layer_metadata(self) -> LayerMetadata:
        return LayerMetadata(
            layer_type=FiveLayerType.L2_CANONICAL,
            transformation_count=self.deduplication_count + self.conflict_resolution_count,
        )


# ============================================================================
# L2.5: 业务语义层
# ============================================================================

class BusinessRelation(BaseModel):
    """业务关系"""

    relation_id: str
    source_entity: str
    target_entity: str
    relation_type: str = "business_related"
    business_meaning: str = ""
    business_importance: float = 0.0
    impact_level: str = "medium"
    inference_confidence: float = 0.0
    reasoning_path: List[str] = Field(default_factory=list)

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SemanticCluster(BaseModel):
    """语义聚类"""

    cluster_id: str
    cluster_name: str
    entity_ids: List[str] = Field(default_factory=list)
    cohesion_score: float = 0.0
    separation_score: float = 0.0
    cluster_confidence: float = 0.0
    main_topics: List[str] = Field(default_factory=list)
    topic_weights: Dict[str, float] = Field(default_factory=dict)

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class L2_5LayerArtifact(BaseModel):
    """L2.5层产物 - 业务语义层"""

    run_id: str
    source_file: str
    business_relations: List[BusinessRelation] = Field(default_factory=list)
    semantic_clusters: List[SemanticCluster] = Field(default_factory=list)

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def layer(self) -> FiveLayerType:
        return FiveLayerType.L2_5_SEMANTIC

    @property
    def relation_count(self) -> int:
        return len(self.business_relations)

    @property
    def cluster_count(self) -> int:
        return len(self.semantic_clusters)

    @property
    def total_business_relations(self) -> int:
        return self.relation_count

    @property
    def total_semantic_clusters(self) -> int:
        return self.cluster_count


# ============================================================================
# L3: 最终实例层
# ============================================================================

class EntityInstance(BaseModel):
    """实体实例 - L3层的最终实例对象"""

    instance_id: str
    name: str
    entity_type: str = "Instance"
    parent_entity_id: str = ""
    parent_entity_type: str = ""
    domain: str = "SUPPLY_CHAIN"
    source_file: str = ""
    description: str = ""
    status: str = "active"
    version: str = "1.0"
    identifiers: Dict[str, str] = Field(default_factory=dict)
    properties: Dict[str, Any] = Field(default_factory=dict)
    values: Dict[str, Any] = Field(default_factory=dict)
    evidence_ref: str = ""
    citation: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CrossFileEvidence(BaseModel):
    """跨文件证据"""

    evidence_id: str
    entity_id: str
    source_files: List[str] = Field(default_factory=list)
    association_strength: float = 0.0
    cross_file_similarity: float = 0.0
    relation_type: str = "same_entity"

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FolderRelation(BaseModel):
    """文件夹关系"""

    relation_id: str
    source_folder: str
    target_folder: str
    relation_type: str = "related_to"
    relation_strength: float = 0.0
    connection_count: int = 0
    is_bridge: bool = False
    bridge_entities: List[str] = Field(default_factory=list)

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class L3LayerArtifact(BaseModel):
    """L3层产物 - 最终实例层"""

    run_id: str
    source_folder: str
    instances: List[EntityInstance] = Field(default_factory=list)
    cross_file_evidences: List[CrossFileEvidence] = Field(default_factory=list)
    folder_relations: List[FolderRelation] = Field(default_factory=list)

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def layer(self) -> FiveLayerType:
        return FiveLayerType.L3_CROSS_FILE

    @property
    def instance_count(self) -> int:
        return len(self.instances)

    @property
    def evidence_count(self) -> int:
        return len(self.cross_file_evidences)

    @property
    def relation_count(self) -> int:
        return len(self.folder_relations)

    @property
    def total_instances(self) -> int:
        return self.instance_count

    @property
    def total_cross_file_evidences(self) -> int:
        return self.evidence_count

    @property
    def total_folder_relations(self) -> int:
        return self.relation_count


# ============================================================================
# 完整架构
# ============================================================================

class ArtifactMapping(BaseModel):
    """产物映射关系"""

    default_mapping: Dict[str, str] = Field(default_factory=dict)
    transformation_rules: Dict[str, str] = Field(default_factory=dict)
    existing_products: Dict[str, str] = Field(default_factory=dict)

    def __init__(self, **data):
        """Initialize with default transformation rules."""
        super().__init__(**data)
        # Set default transformation rules if empty
        if not self.transformation_rules:
            self.transformation_rules = {
                "L0 -> L1": "extract_entities,extract_relations,extract_evidence",
                "L1 -> L2": "normalize_entities,merge_duplicates,resolve_conflicts",
                "L2 -> L2.5": "infer_business_relations,cluster_semantics",
                "L2.5 -> L3": "cross_file_links,build_folder_graph",
            }
        # Set default existing products if empty
        if not self.existing_products:
            self.existing_products = {
                "CanonicalEntity(CANONICAL)": "L2_CANONICAL",
                "CanonicalEntity(RAW_CANDIDATE)": "L1_CANDIDATE",
                "CanonicalRelation(CANONICAL)": "L2_CANONICAL",
                "CanonicalRelation(RAW_CANDIDATE)": "L1_CANDIDATE",
                "RawResponseArtifact": "L0_RAW_EVIDENCE",
                "LLMRawResponse": "L0_RAW_EVIDENCE",
            }

    def map_existing_product(self, product_name: str) -> FiveLayerType:
        """根据产物名称映射到对应的层级类型"""
        if "Raw" in product_name or "Evidence" in product_name:
            return FiveLayerType.L0_RAW_EVIDENCE
        elif "Candidate" in product_name or "L1" in product_name:
            return FiveLayerType.L1_CANDIDATE
        elif "Canonical" in product_name and "RAW_CANDIDATE" not in product_name:
            return FiveLayerType.L2_CANONICAL
        elif "Semantic" in product_name or "L2.5" in product_name:
            return FiveLayerType.L2_5_SEMANTIC
        elif "Cross" in product_name or "L3" in product_name:
            return FiveLayerType.L3_CROSS_FILE
        return FiveLayerType.L1_CANDIDATE

    def get_transformation_rules(self, source_layer: str, target_layer: str) -> List[str]:
        """获取从源层到目标层的转换规则"""
        # Try both formats: "L0_L1" and "L0 -> L1"
        key = f"{source_layer} -> {target_layer}"
        if key not in self.transformation_rules:
            key = f"{source_layer}_{target_layer}"
        if key in self.transformation_rules:
            return self.transformation_rules[key].split(",") if isinstance(self.transformation_rules[key], str) else self.transformation_rules[key]
        return []


class FiveLayerArchitecture(BaseModel):
    """完整的5层架构"""

    run_id: str
    source_path: str

    l0_artifact: Optional[L0LayerArtifact] = None
    l1_artifact: Optional[L1LayerArtifact] = None
    l2_artifact: Optional[L2LayerArtifact] = None
    l2_5_artifact: Optional[L2_5LayerArtifact] = None
    l3_artifact: Optional[L3LayerArtifact] = None

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def update_layer(self, layer_type: FiveLayerType, artifact: Any) -> None:
        """更新指定层的产物"""
        if layer_type == FiveLayerType.L0_RAW_EVIDENCE:
            self.l0_artifact = artifact
        elif layer_type == FiveLayerType.L1_CANDIDATE:
            self.l1_artifact = artifact
        elif layer_type == FiveLayerType.L2_CANONICAL:
            self.l2_artifact = artifact
        elif layer_type == FiveLayerType.L2_5_SEMANTIC:
            self.l2_5_artifact = artifact
        elif layer_type == FiveLayerType.L3_CROSS_FILE:
            self.l3_artifact = artifact
        self.updated_at = datetime.now().isoformat()

    def get_layer(self, layer_type: FiveLayerType) -> Optional[Any]:
        """获取指定层的产物"""
        if layer_type == FiveLayerType.L0_RAW_EVIDENCE:
            return self.l0_artifact
        elif layer_type == FiveLayerType.L1_CANDIDATE:
            return self.l1_artifact
        elif layer_type == FiveLayerType.L2_CANONICAL:
            return self.l2_artifact
        elif layer_type == FiveLayerType.L2_5_SEMANTIC:
            return self.l2_5_artifact
        elif layer_type == FiveLayerType.L3_CROSS_FILE:
            return self.l3_artifact
        return None

    def get_progress(self) -> float:
        """获取架构完成进度"""
        layers = [self.l0_artifact, self.l1_artifact, self.l2_artifact,
                  self.l2_5_artifact, self.l3_artifact]
        completed = sum(1 for layer in layers if layer is not None)
        return completed / len(layers) if layers else 0.0

    @property
    def completed_layers(self) -> List[str]:
        """获取已完成的层列表"""
        completed = []
        if self.l0_artifact is not None:
            completed.append("L0")
        if self.l1_artifact is not None:
            completed.append("L1")
        if self.l2_artifact is not None:
            completed.append("L2")
        if self.l2_5_artifact is not None:
            completed.append("L2.5")
        if self.l3_artifact is not None:
            completed.append("L3")
        return completed

    def get_layer_artifact(self, layer_type: FiveLayerType) -> Optional[Any]:
        """获取指定层的产物"""
        return self.get_layer(layer_type)

    def get_layer_status(self) -> Dict[str, bool]:
        """获取各层的完成状态"""
        return {
            "L0": self.l0_artifact is not None,
            "L1": self.l1_artifact is not None,
            "L2": self.l2_artifact is not None,
            "L2.5": self.l2_5_artifact is not None,
            "L3": self.l3_artifact is not None,
        }

    def get_entity_count(self) -> Dict[str, int]:
        """获取各层实体数量"""
        return {
            "L0": self.l0_artifact.evidence_count if self.l0_artifact else 0,
            "L1": self.l1_artifact.entity_count if self.l1_artifact else 0,
            "L2": self.l2_artifact.entity_count if self.l2_artifact else 0,
            "L2.5": self.l2_5_artifact.cluster_count if self.l2_5_artifact else 0,
            "L3": self.l3_artifact.instance_count if self.l3_artifact else 0,
        }

    def get_relation_count(self) -> Dict[str, int]:
        """获取各层关系数量"""
        return {
            "L1": self.l1_artifact.relation_count if self.l1_artifact else 0,
            "L2": self.l2_artifact.relation_count if self.l2_artifact else 0,
            "L2.5": self.l2_5_artifact.relation_count if self.l2_5_artifact else 0,
            "L3": self.l3_artifact.relation_count if self.l3_artifact else 0,
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "run_id": self.run_id,
            "source_path": self.source_path,
            "progress": self.get_progress(),
            "entity_count": self.get_entity_count(),
            "relation_count": self.get_relation_count(),
            "layers": {
                "L0": self.l0_artifact.model_dump() if self.l0_artifact else None,
                "L1": self.l1_artifact.model_dump() if self.l1_artifact else None,
                "L2": self.l2_artifact.model_dump() if self.l2_artifact else None,
                "L2.5": self.l2_5_artifact.model_dump() if self.l2_5_artifact else None,
                "L3": self.l3_artifact.model_dump() if self.l3_artifact else None,
            },
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }
