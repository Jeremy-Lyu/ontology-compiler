"""结构化索引 - 基于 Canonical Entity/Relation/Evidence 的索引

本模块提供基于 Canonical 模型的结构化索引能力，支持：
1. 实体索引（按类型、层级、属性）
2. 关系索引（按类型、源实体、目标实体）
3. 证据索引（按来源、类型、置信度）
4. 属性索引（按属性名、值）
5. 多维度查询与过滤

架构目标：
- G4: 统一知识消费层
- G5: 统一质量闭环
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from pydantic import BaseModel, Field, model_validator

from src.domain.models.canonical import (
    CanonicalEntity,
    CanonicalRelation,
    CanonicalEvidence,
    CanonicalAttribute,
    LayerType,
    EntityType,
    RelationType,
)


class IndexStats(BaseModel):
    """索引统计"""
    total_entities: int = Field(0, description="实体总数")
    total_relations: int = Field(0, description="关系总数")
    total_evidences: int = Field(0, description="证据总数")
    entities_by_type: Dict[str, int] = Field(default_factory=dict, description="按类型统计实体")
    entities_by_layer: Dict[str, int] = Field(default_factory=dict, description="按层级统计实体")
    relations_by_type: Dict[str, int] = Field(default_factory=dict, description="按类型统计关系")
    evidences_by_source: Dict[str, int] = Field(default_factory=dict, description="按来源统计证据")
    last_updated: str = Field("", description="最后更新时间")


class EntityIndex(BaseModel):
    """实体索引"""
    entity_id: str = Field(..., description="实体 ID")
    name: str = Field(..., description="实体名称")
    entity_type: str = Field("unknown", description="实体类型")
    layer: str = Field("L1", description="层级")
    description: str = Field("", description="描述")
    aliases: List[str] = Field(default_factory=list, description="别名")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="属性")
    confidence: float = Field(1.0, description="置信度")
    evidence_count: int = Field(0, description="证据数量")
    
    def matches_query(self, query: str) -> bool:
        """匹配查询"""
        query_lower = query.lower()
        return (
            query_lower in self.name.lower() or
            (self.description and query_lower in self.description.lower()) or
            any(query_lower in alias.lower() for alias in self.aliases) or
            any(query_lower in str(v).lower() for v in self.attributes.values())
        )


class RelationIndex(BaseModel):
    """关系索引"""
    relation_id: str = Field(..., description="关系 ID")
    from_entity: str = Field(..., description="源实体 ID")
    to_entity: str = Field(..., description="目标实体 ID")
    relation_type: str = Field(..., description="关系类型")
    properties: Dict[str, Any] = Field(default_factory=dict, description="属性")
    confidence: float = Field(1.0, description="置信度")
    evidence_count: int = Field(0, description="证据数量")
    
    def matches_query(self, query: str, from_name: str = "", to_name: str = "") -> bool:
        """匹配查询"""
        query_lower = query.lower()
        return (
            query_lower in self.relation_type.lower() or
            (from_name and query_lower in from_name.lower()) or
            (to_name and query_lower in to_name.lower())
        )


class EvidenceIndex(BaseModel):
    """证据索引"""
    evidence_id: str = Field(..., description="证据 ID")
    entity_id: str = Field(..., description="关联实体 ID")
    source: str = Field(..., description="来源")
    source_type: str = Field("", description="来源类型")
    content: str = Field("", description="内容")
    confidence: float = Field(1.0, description="置信度")
    created_at: str = Field("", description="创建时间")
    
    def matches_query(self, query: str) -> bool:
        """匹配查询"""
        query_lower = query.lower()
        return (
            query_lower in self.content.lower() or
            query_lower in self.source.lower()
        )


class StructuredIndex:
    """结构化索引
    
    提供基于 Canonical 模型的结构化索引和查询能力。
    """
    
    def __init__(self) -> None:
        self._entities: Dict[str, EntityIndex] = {}
        self._relations: Dict[str, RelationIndex] = {}
        self._evidences: Dict[str, EvidenceIndex] = {}
        self._entity_name_index: Dict[str, str] = {}  # name -> entity_id
        self._entity_type_index: Dict[str, Set[str]] = defaultdict(set)  # type -> entity_ids
        self._entity_layer_index: Dict[str, Set[str]] = defaultdict(set)  # layer -> entity_ids
        self._relation_from_index: Dict[str, Set[str]] = defaultdict(set)  # from_entity -> relation_ids
        self._relation_to_index: Dict[str, Set[str]] = defaultdict(set)  # to_entity -> relation_ids
        self._relation_type_index: Dict[str, Set[str]] = defaultdict(set)  # type -> relation_ids
        self._evidence_entity_index: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> evidence_ids
        self._evidence_source_index: Dict[str, Set[str]] = defaultdict(set)  # source -> evidence_ids
        self._stats: IndexStats = IndexStats()
    
    def index_entity(self, entity: CanonicalEntity) -> None:
        """索引实体"""
        entity_idx = EntityIndex(
            entity_id=entity.entity_id,
            name=entity.name,
            entity_type=entity.entity_type,
            layer=entity.layer,
            description=entity.description,
            aliases=entity.aliases,
            attributes=entity.properties,
            confidence=entity.confidence,
            evidence_count=len(entity.evidence),
        )
        
        self._entities[entity.entity_id] = entity_idx
        self._entity_name_index[entity.name] = entity.entity_id
        self._entity_type_index[entity.entity_type].add(entity.entity_id)
        self._entity_layer_index[entity.layer].add(entity.entity_id)
        
        # 索引别名
        for alias in entity.aliases:
            self._entity_name_index[alias] = entity.entity_id
        
        self._update_stats()
    
    def index_relation(self, relation: CanonicalRelation, from_name: str = "", to_name: str = "") -> None:
        """索引关系"""
        # 兼容 CanonicalRelation 的字段名
        from_entity = relation.source_id
        to_entity = relation.target_id
        
        relation_idx = RelationIndex(
            relation_id=relation.relation_id,
            from_entity=from_entity,
            to_entity=to_entity,
            relation_type=relation.relation_type,
            properties=relation.properties,
            confidence=relation.confidence,
            evidence_count=len(relation.evidence),
        )
        
        self._relations[relation.relation_id] = relation_idx
        self._relation_from_index[from_entity].add(relation.relation_id)
        self._relation_to_index[to_entity].add(relation.relation_id)
        self._relation_type_index[relation.relation_type].add(relation.relation_id)
        
        self._update_stats()
    
    def index_evidence(self, evidence: CanonicalEvidence) -> None:
        """索引证据"""
        # 兼容 CanonicalEvidence 的字段名
        entity_id = evidence.metadata.get("entity_id", "")
        source = evidence.source_file or evidence.source_url or ""
        source_type = evidence.metadata.get("source_type", "")
        
        evidence_idx = EvidenceIndex(
            evidence_id=evidence.evidence_id,
            entity_id=entity_id,
            source=source,
            source_type=source_type,
            content=evidence.content,
            confidence=evidence.confidence,
            created_at=evidence.created_at,
        )
        
        self._evidences[evidence.evidence_id] = evidence_idx
        if entity_id:
            self._evidence_entity_index[entity_id].add(evidence.evidence_id)
        if source:
            self._evidence_source_index[source].add(evidence.evidence_id)
        
        self._update_stats()
    
    def search_entities(
        self,
        query: str = "",
        entity_type: Optional[str] = None,
        layer: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> List[EntityIndex]:
        """搜索实体"""
        candidates = set(self._entities.keys())
        
        # 按类型过滤
        if entity_type:
            candidates &= self._entity_type_index.get(entity_type, set())
        
        # 按层级过滤
        if layer:
            candidates &= self._entity_layer_index.get(layer, set())
        
        # 按置信度过滤
        results = []
        for entity_id in candidates:
            entity = self._entities[entity_id]
            if entity.confidence >= min_confidence:
                if not query or entity.matches_query(query):
                    results.append(entity)
        
        # 按置信度排序
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:limit]
    
    def search_relations(
        self,
        query: str = "",
        from_entity: Optional[str] = None,
        to_entity: Optional[str] = None,
        relation_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> List[Tuple[RelationIndex, str, str]]:
        """搜索关系
        
        Returns:
            List of (relation, from_name, to_name)
        """
        candidates = set(self._relations.keys())
        
        # 按源实体过滤
        if from_entity:
            candidates &= self._relation_from_index.get(from_entity, set())
        
        # 按目标实体过滤
        if to_entity:
            candidates &= self._relation_to_index.get(to_entity, set())
        
        # 按关系类型过滤
        if relation_type:
            candidates &= self._relation_type_index.get(relation_type, set())
        
        # 按置信度过滤
        results = []
        for relation_id in candidates:
            relation = self._relations[relation_id]
            if relation.confidence >= min_confidence:
                from_entity = self._entities.get(relation.from_entity)
                to_entity = self._entities.get(relation.to_entity)
                from_name = from_entity.name if from_entity else ""
                to_name = to_entity.name if to_entity else ""
                if not query or relation.matches_query(query, from_name, to_name):
                    results.append((relation, from_name, to_name))
        
        # 按置信度排序
        results.sort(key=lambda x: x[0].confidence, reverse=True)
        return results[:limit]
    
    def search_evidences(
        self,
        query: str = "",
        entity_id: Optional[str] = None,
        source: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> List[EvidenceIndex]:
        """搜索证据"""
        candidates = set(self._evidences.keys())
        
        # 按实体过滤
        if entity_id:
            candidates &= self._evidence_entity_index.get(entity_id, set())
        
        # 按来源过滤
        if source:
            candidates &= self._evidence_source_index.get(source, set())
        
        # 按置信度过滤
        results = []
        for evidence_id in candidates:
            evidence = self._evidences[evidence_id]
            if evidence.confidence >= min_confidence:
                if not query or evidence.matches_query(query):
                    results.append(evidence)
        
        # 按置信度排序
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:limit]
    
    def get_entity_relations(
        self,
        entity_id: str,
        direction: str = "both",
        limit: int = 10,
    ) -> List[Tuple[RelationIndex, str, str]]:
        """获取实体关系
        
        Args:
            entity_id: 实体 ID
            direction: "out" (出边), "in" (入边), "both" (双向)
            limit: 最大数量
        
        Returns:
            List of (relation, from_name, to_name)
        """
        results = []
        
        if direction in ["out", "both"]:
            for relation_id in self._relation_from_index.get(entity_id, set()):
                relation = self._relations[relation_id]
                to_name = self._entities.get(relation.to_entity, EntityIndex(entity_id="", name="")).name
                results.append((relation, self._entities[entity_id].name, to_name))
        
        if direction in ["in", "both"]:
            for relation_id in self._relation_to_index.get(entity_id, set()):
                relation = self._relations[relation_id]
                from_name = self._entities.get(relation.from_entity, EntityIndex(entity_id="", name="")).name
                results.append((relation, from_name, self._entities[entity_id].name))
        
        # 按置信度排序
        results.sort(key=lambda x: x[0].confidence, reverse=True)
        return results[:limit]
    
    def get_entity_evidences(self, entity_id: str, limit: int = 10) -> List[EvidenceIndex]:
        """获取实体证据"""
        evidence_ids = self._evidence_entity_index.get(entity_id, set())
        evidences = [self._evidences[eid] for eid in evidence_ids]
        evidences.sort(key=lambda x: x.confidence, reverse=True)
        return evidences[:limit]
    
    def resolve_entity_name(self, name: str) -> Optional[str]:
        """解析实体名称到 ID"""
        return self._entity_name_index.get(name)
    
    def get_entity(self, entity_id: str) -> Optional[EntityIndex]:
        """获取实体"""
        return self._entities.get(entity_id)
    
    def get_relation(self, relation_id: str) -> Optional[RelationIndex]:
        """获取关系"""
        return self._relations.get(relation_id)
    
    def get_stats(self) -> IndexStats:
        """获取索引统计"""
        return self._stats
    
    def _update_stats(self) -> None:
        """更新统计"""
        self._stats.total_entities = len(self._entities)
        self._stats.total_relations = len(self._relations)
        self._stats.total_evidences = len(self._evidences)
        
        self._stats.entities_by_type = {
            t: len(ids) for t, ids in self._entity_type_index.items()
        }
        self._stats.entities_by_layer = {
            l: len(ids) for l, ids in self._entity_layer_index.items()
        }
        self._stats.relations_by_type = {
            t: len(ids) for t, ids in self._relation_type_index.items()
        }
        self._stats.evidences_by_source = {
            s: len(ids) for s, ids in self._evidence_source_index.items()
        }
        self._stats.last_updated = datetime.now().isoformat()
    
    def clear(self) -> None:
        """清空索引"""
        self._entities.clear()
        self._relations.clear()
        self._evidences.clear()
        self._entity_name_index.clear()
        self._entity_type_index.clear()
        self._entity_layer_index.clear()
        self._relation_from_index.clear()
        self._relation_to_index.clear()
        self._relation_type_index.clear()
        self._evidence_entity_index.clear()
        self._evidence_source_index.clear()
        self._stats = IndexStats()