"""5层架构适配器 - 将现有产物映射到5层架构

本模块提供从现有产物（CanonicalEntity, CanonicalRelation, LayerArtifact等）
到5层架构（L0-L3）的转换和映射功能。

映射关系：
- L0: RawResponseArtifact -> RawEvidence
- L1: CanonicalEntity(RAW_CANDIDATE) -> CandidateEntity
- L2: CanonicalEntity(CANONICAL) -> CanonicalEntityLayer
- L2.5: BusinessRelation, SemanticCluster
- L3: CrossFileEvidence, FolderRelation

Note: 依赖 ontology_compiler 的 LayerArtifact 类型，保留在此以避免循环依赖。
"""

from typing import Any, Dict, List, Optional, Set
from datetime import datetime
import hashlib

from .five_layer_architecture import (
    FiveLayerType,
    LayerMetadata,
    RawEvidence,
    CandidateEntity,
    CandidateRelation,
    CanonicalEntityLayer,
    CanonicalRelationLayer,
    BusinessRelation,
    SemanticCluster,
    EntityInstance,
    CrossFileEvidence,
    FolderRelation,
    L0LayerArtifact,
    L1LayerArtifact,
    L2LayerArtifact,
    L2_5LayerArtifact,
    L3LayerArtifact,
    FiveLayerArchitecture,
    ArtifactMapping,
)

from .canonical import (
    CanonicalEntity,
    CanonicalRelation,
    CanonicalEvidence,
    CanonicalStatus,
)


class FiveLayerAdapter:
    """5层架构适配器

    负责将现有产物映射和转换为5层架构格式。
    """

    def __init__(self):
        self.mapping = ArtifactMapping()

    def raw_response_to_evidence(
        self,
        raw_response: Any,
        run_id: str,
        source_file: str,
    ) -> RawEvidence:
        """将原始响应转换为 RawEvidence

        Args:
            raw_response: 原始响应对象，需包含 response_text, prompt_tokens,
                         completion_tokens, fallback_used, model, timestamp,
                         strategy, fallback_reason 属性
            run_id: 运行ID
            source_file: 源文件路径

        Returns:
            RawEvidence 对象
        """
        content = getattr(raw_response, "response_text", "")
        content_hash = self._calculate_hash(content)

        return RawEvidence(
            evidence_id=f"ev_{run_id}_{source_file}",
            source_file=source_file,
            file_hash="",
            content=content,
            content_hash=content_hash,
            line_numbers=[],
            llm_response={"response_text": content},
            prompt_tokens=getattr(raw_response, "prompt_tokens", 0),
            completion_tokens=getattr(raw_response, "completion_tokens", 0),
            latency_ms=0,
            fallback_used=getattr(raw_response, "fallback_used", False),
            model_name=getattr(raw_response, "model", ""),
            extraction_confidence=0.0,
            quality_issues=[],
            created_at=(
                raw_response.timestamp.isoformat()
                if hasattr(raw_response, "timestamp")
                else datetime.now().isoformat()
            ),
            metadata={
                "strategy": getattr(raw_response, "strategy", ""),
                "fallback_reason": getattr(raw_response, "fallback_reason", ""),
            },
        )

    def create_l0_artifact(
        self,
        run_id: str,
        source_file: str,
        raw_responses: List[Any],
    ) -> L0LayerArtifact:
        """创建 L0 层产物"""
        evidences = [self.raw_response_to_evidence(r, run_id, source_file) for r in raw_responses]

        artifact = L0LayerArtifact(
            run_id=run_id,
            source_file=source_file,
            evidences=evidences,
        )

        return artifact

    def canonical_entity_to_candidate(self, entity: CanonicalEntity) -> CandidateEntity:
        """将 CanonicalEntity 转换为 CandidateEntity"""
        return CandidateEntity(
            entity_id=entity.entity_id,
            name=entity.name,
            entity_type=entity.entity_type,
            description=entity.description,
            status=entity.status,
            confidence=entity.confidence,
            quality_score=entity.metadata.get("quality_score", 0.0),
            source_evidence_ids=[e.evidence_id for e in entity.evidence],
            source_file=entity.source_trace.get("source_file", ""),
            conflicts=entity.metadata.get("conflicts", []),
            created_at=entity.created_at,
            metadata=entity.metadata,
        )

    def canonical_relation_to_candidate(self, relation: CanonicalRelation) -> CandidateRelation:
        """将 CanonicalRelation 转换为 CandidateRelation"""
        return CandidateRelation(
            relation_id=relation.relation_id,
            source_entity=relation.source_id,
            target_entity=relation.target_id,
            relation_type=relation.relation_type,
            description=relation.description,
            status=relation.status,
            confidence=relation.confidence,
            weight=relation.weight,
            source_evidence_ids=[e.evidence_id for e in relation.evidence],
            source_file=relation.metadata.get("source_file", ""),
            created_at=relation.created_at,
            metadata=relation.metadata,
        )

    def create_l1_artifact(
        self,
        run_id: str,
        source_file: str,
        entities: List[CanonicalEntity],
        relations: List[CanonicalRelation],
    ) -> L1LayerArtifact:
        """创建 L1 层产物"""
        candidate_entities = [
            self.canonical_entity_to_candidate(e)
            for e in entities
            if e.status == CanonicalStatus.RAW_CANDIDATE.value
        ]

        candidate_relations = [
            self.canonical_relation_to_candidate(r)
            for r in relations
            if r.status == CanonicalStatus.RAW_CANDIDATE.value
        ]

        artifact = L1LayerArtifact(
            run_id=run_id,
            source_file=source_file,
            entities=candidate_entities,
            relations=candidate_relations,
            transformation_count=1,
        )

        return artifact

    def canonical_entity_to_canonical_layer(self, entity: CanonicalEntity) -> Optional[CanonicalEntityLayer]:
        """将 CanonicalEntity 转换为 CanonicalEntityLayer"""
        if entity.status != CanonicalStatus.GOVERNED_CANONICAL.value:
            return None

        return CanonicalEntityLayer(
            entity_id=entity.entity_id,
            name=entity.name,
            entity_type=entity.entity_type,
            layer=entity.layer,
            description=entity.description,
            status=entity.status,
            confidence=entity.confidence,
            quality_score=entity.metadata.get("quality_score", 0.0),
            candidate_ids=entity.metadata.get("candidate_ids", []),
            evidence_count=len(entity.evidence),
            source_files=set(entity.source_trace.get("source_files", [])),
            properties=entity.properties,
            aliases=entity.aliases,
            version=entity.version,
            lineage=entity.lineage.derived_from if hasattr(entity.lineage, 'derived_from') else [],
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            metadata=entity.metadata,
        )

    def canonical_relation_to_canonical_layer(self, relation: CanonicalRelation) -> Optional[CanonicalRelationLayer]:
        """将 CanonicalRelation 转换为 CanonicalRelationLayer"""
        if relation.status != CanonicalStatus.GOVERNED_CANONICAL.value:
            return None

        return CanonicalRelationLayer(
            relation_id=relation.relation_id,
            source_id=relation.source_id,
            target_id=relation.target_id,
            relation_type=relation.relation_type,
            description=relation.description,
            status=relation.status,
            confidence=relation.confidence,
            weight=relation.weight,
            layer=relation.layer,
            candidate_ids=relation.metadata.get("candidate_ids", []),
            evidence_count=len(relation.evidence),
            created_at=relation.created_at,
            updated_at=relation.created_at,
            metadata=relation.metadata,
        )

    def create_l2_artifact(
        self,
        run_id: str,
        source_file: str,
        entities: List[CanonicalEntity],
        relations: List[CanonicalRelation],
        deduplication_count: int = 0,
        conflict_resolution_count: int = 0,
    ) -> L2LayerArtifact:
        """创建 L2 层产物"""
        canonical_entities = [
            self.canonical_entity_to_canonical_layer(e)
            for e in entities
            if e.status == CanonicalStatus.GOVERNED_CANONICAL.value
        ]

        canonical_relations = [
            self.canonical_relation_to_canonical_layer(r)
            for r in relations
            if r.status == CanonicalStatus.GOVERNED_CANONICAL.value
        ]

        canonical_entities = [e for e in canonical_entities if e is not None]
        canonical_relations = [r for r in canonical_relations if r is not None]

        artifact = L2LayerArtifact(
            run_id=run_id,
            source_file=source_file,
            entities=canonical_entities,
            relations=canonical_relations,
            deduplication_count=deduplication_count,
            conflict_resolution_count=conflict_resolution_count,
        )

        return artifact

    def create_business_relation(
        self,
        relation_id: str,
        source_entity: str,
        target_entity: str,
        relation_type: str,
        business_meaning: str = "",
        business_importance: float = 0.0,
        impact_level: str = "medium",
        inference_confidence: float = 0.0,
        reasoning_path: Optional[List[str]] = None,
    ) -> BusinessRelation:
        """创建业务关系"""
        return BusinessRelation(
            relation_id=relation_id,
            source_entity=source_entity,
            target_entity=target_entity,
            relation_type=relation_type,
            business_meaning=business_meaning,
            business_importance=business_importance,
            impact_level=impact_level,
            inference_confidence=inference_confidence,
            reasoning_path=reasoning_path or [],
        )

    def create_semantic_cluster(
        self,
        cluster_id: str,
        cluster_name: str,
        entity_ids: List[str],
        cohesion_score: float = 0.0,
        separation_score: float = 0.0,
        cluster_confidence: float = 0.0,
        main_topics: Optional[List[str]] = None,
        topic_weights: Optional[Dict[str, float]] = None,
    ) -> SemanticCluster:
        """创建语义聚类"""
        return SemanticCluster(
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            entity_ids=entity_ids,
            cohesion_score=cohesion_score,
            separation_score=separation_score,
            cluster_confidence=cluster_confidence,
            main_topics=main_topics or [],
            topic_weights=topic_weights or {},
        )

    def create_l2_5_artifact(
        self,
        run_id: str,
        source_file: str,
        business_relations: List[BusinessRelation] = None,
        semantic_clusters: List[SemanticCluster] = None,
    ) -> L2_5LayerArtifact:
        """创建 L2.5 层产物"""
        return L2_5LayerArtifact(
            run_id=run_id,
            source_file=source_file,
            business_relations=business_relations or [],
            semantic_clusters=semantic_clusters or [],
        )

    def create_cross_file_evidence(
        self,
        evidence_id: str,
        entity_id: str,
        source_files: List[str],
        association_strength: float = 0.0,
        cross_file_similarity: float = 0.0,
        relation_type: str = "same_entity",
    ) -> CrossFileEvidence:
        """创建跨文件证据"""
        return CrossFileEvidence(
            evidence_id=evidence_id,
            entity_id=entity_id,
            source_files=source_files,
            association_strength=association_strength,
            cross_file_similarity=cross_file_similarity,
            relation_type=relation_type,
        )

    def create_folder_relation(
        self,
        relation_id: str,
        source_folder: str,
        target_folder: str,
        relation_type: str,
        relation_strength: float = 0.0,
        connection_count: int = 0,
        is_bridge: bool = False,
        bridge_entities: Optional[List[str]] = None,
    ) -> FolderRelation:
        """创建文件夹关系"""
        return FolderRelation(
            relation_id=relation_id,
            source_folder=source_folder,
            target_folder=target_folder,
            relation_type=relation_type,
            relation_strength=relation_strength,
            connection_count=connection_count,
            is_bridge=is_bridge,
            bridge_entities=bridge_entities or [],
        )

    def fragment_instance_to_entity_instance(self, instance: Any) -> EntityInstance:
        """将 FragmentInstance 转换为 EntityInstance"""
        return EntityInstance(
            instance_id=getattr(instance, "meta_id", ""),
            name=getattr(instance, "meta_name", ""),
            entity_type=getattr(instance, "meta_class", "Instance"),
            domain=getattr(instance, "domain", "SUPPLY_CHAIN"),
            source_file=getattr(instance, "source_file", ""),
            values=getattr(instance, "values", {}),
            properties=getattr(instance, "properties", {}),
            evidence_ref=getattr(instance, "evidence_ref", ""),
            metadata=getattr(instance, "metadata", {}),
        )

    def dict_to_entity_instance(self, instance_data: Dict[str, Any]) -> EntityInstance:
        """将字典数据转换为 EntityInstance"""
        return EntityInstance(
            instance_id=instance_data.get("meta_id") or instance_data.get("instance_id") or instance_data.get("id") or "",
            name=instance_data.get("meta_name") or instance_data.get("name") or "",
            entity_type=instance_data.get("meta_class") or instance_data.get("entity_type") or "Instance",
            parent_entity_id=instance_data.get("parent_entity_id") or "",
            parent_entity_type=instance_data.get("parent_entity_type") or "",
            domain=instance_data.get("domain", "SUPPLY_CHAIN"),
            source_file=instance_data.get("source_file", ""),
            description=instance_data.get("description", ""),
            status=instance_data.get("status", "active"),
            version=instance_data.get("version", "1.0"),
            identifiers=instance_data.get("identifiers", {}),
            properties=instance_data.get("properties", {}),
            values=instance_data.get("values", {}),
            evidence_ref=instance_data.get("evidence_ref") or instance_data.get("evidence", ""),
            citation=instance_data.get("citation", ""),
            created_at=instance_data.get("created_at", datetime.now().isoformat()),
            updated_at=instance_data.get("updated_at", datetime.now().isoformat()),
            metadata=instance_data.get("metadata", {}),
        )

    def create_l3_artifact(
        self,
        run_id: str,
        source_folder: str,
        instances: List[Any] = None,
        cross_file_evidences: List[CrossFileEvidence] = None,
        folder_relations: List[FolderRelation] = None,
    ) -> L3LayerArtifact:
        """创建 L3 层产物"""
        entity_instances = []
        if instances:
            for inst_data in instances:
                if isinstance(inst_data, dict):
                    entity_instances.append(self.dict_to_entity_instance(inst_data))
                elif isinstance(inst_data, EntityInstance):
                    entity_instances.append(inst_data)
        
        return L3LayerArtifact(
            run_id=run_id,
            source_folder=source_folder,
            instances=entity_instances,
            cross_file_evidences=cross_file_evidences or [],
            folder_relations=folder_relations or [],
        )

    def convert_layer_artifacts_to_five_layer(
        self,
        run_id: str,
        source_path: str,
        layer_artifacts: Dict[str, Any],
        entities: List[CanonicalEntity],
        relations: List[CanonicalRelation],
    ) -> FiveLayerArchitecture:
        """将现有产物转换为完整的5层架构"""
        architecture = FiveLayerArchitecture(
            run_id=run_id,
            source_path=source_path,
        )

        raw_responses = layer_artifacts.get("L0", [])
        if raw_responses:
            l0_artifact = self.create_l0_artifact(
                run_id=run_id,
                source_file=source_path,
                raw_responses=raw_responses,
            )
            architecture.update_layer(FiveLayerType.L0_RAW_EVIDENCE, l0_artifact)

        if entities and relations:
            l1_artifact = self.create_l1_artifact(
                run_id=run_id,
                source_file=source_path,
                entities=entities,
                relations=relations,
            )
            architecture.update_layer(FiveLayerType.L1_CANDIDATE, l1_artifact)

        if entities and relations:
            dedup_count = len([e for e in entities if e.metadata.get("deduplicated")])
            conflict_count = len([e for e in entities if e.metadata.get("conflict_resolved")])

            l2_artifact = self.create_l2_artifact(
                run_id=run_id,
                source_file=source_path,
                entities=entities,
                relations=relations,
                deduplication_count=dedup_count,
                conflict_resolution_count=conflict_count,
            )
            architecture.update_layer(FiveLayerType.L2_CANONICAL, l2_artifact)

        return architecture

    def _calculate_hash(self, content: str) -> str:
        """计算内容哈希"""
        return hashlib.sha256(content.encode()).hexdigest()

    def get_layer_mapping(self) -> ArtifactMapping:
        """获取产物映射关系"""
        return self.mapping

    def validate_layer_consistency(self, architecture: FiveLayerArchitecture) -> Dict[str, bool]:
        """验证各层之间的一致性"""
        validation_results = {
            "L0_to_L1": False,
            "L1_to_L2": False,
            "L2_to_L2_5": False,
            "L2_5_to_L3": False,
        }

        if architecture.l0_artifact and architecture.l1_artifact:
            validation_results["L0_to_L1"] = self._validate_l0_l1_consistency(
                architecture.l0_artifact, architecture.l1_artifact
            )

        if architecture.l1_artifact and architecture.l2_artifact:
            validation_results["L1_to_L2"] = self._validate_l1_l2_consistency(
                architecture.l1_artifact, architecture.l2_artifact
            )

        if architecture.l2_artifact and architecture.l2_5_artifact:
            validation_results["L2_to_L2_5"] = self._validate_l2_l2_5_consistency(
                architecture.l2_artifact, architecture.l2_5_artifact
            )

        if architecture.l2_5_artifact and architecture.l3_artifact:
            validation_results["L2_5_to_L3"] = self._validate_l2_5_l3_consistency(
                architecture.l2_5_artifact, architecture.l3_artifact
            )

        return validation_results

    def _validate_l0_l1_consistency(
        self,
        l0_artifact: L0LayerArtifact,
        l1_artifact: L1LayerArtifact,
    ) -> bool:
        """验证 L0 到 L1 的一致性"""
        l0_evidence_ids = {e.evidence_id for e in l0_artifact.evidences}
        l1_referenced_evidence_ids = set()

        for entity in l1_artifact.entities:
            l1_referenced_evidence_ids.update(entity.source_evidence_ids)

        for relation in l1_artifact.relations:
            l1_referenced_evidence_ids.update(entity.source_evidence_ids)

        return l1_referenced_evidence_ids.issubset(l0_evidence_ids)

    def _validate_l1_l2_consistency(
        self,
        l1_artifact: L1LayerArtifact,
        l2_artifact: L2LayerArtifact,
    ) -> bool:
        """验证 L1 到 L2 的一致性"""
        l1_candidate_ids = {e.entity_id for e in l1_artifact.entities}
        l2_referenced_candidate_ids = set()

        for entity in l2_artifact.entities:
            l2_referenced_candidate_ids.update(entity.candidate_ids)

        return l2_referenced_candidate_ids.issubset(l1_candidate_ids)

    def _validate_l2_l2_5_consistency(
        self,
        l2_artifact: L2LayerArtifact,
        l2_5_artifact: L2_5LayerArtifact,
    ) -> bool:
        """验证 L2 到 L2.5 的一致性"""
        l2_entity_ids = {e.entity_id for e in l2_artifact.entities}
        l2_5_referenced_entity_ids = set()

        for relation in l2_5_artifact.business_relations:
            l2_5_referenced_entity_ids.add(relation.source_entity)
            l2_5_referenced_entity_ids.add(relation.target_entity)

        for cluster in l2_5_artifact.semantic_clusters:
            l2_5_referenced_entity_ids.update(cluster.entity_ids)

        return l2_5_referenced_entity_ids.issubset(l2_entity_ids)

    def _validate_l2_5_l3_consistency(
        self,
        l2_5_artifact: L2_5LayerArtifact,
        l3_artifact: L3LayerArtifact,
    ) -> bool:
        """验证 L2.5 到 L3 的一致性"""
        return len(l3_artifact.cross_file_evidences) > 0 or len(l3_artifact.folder_relations) > 0


def create_five_layer_from_existing_data(
    run_id: str,
    source_path: str,
    raw_responses: List[Any],
    entities: List[CanonicalEntity],
    relations: List[CanonicalRelation],
) -> FiveLayerArchitecture:
    """从现有数据创建5层架构的便捷函数"""
    adapter = FiveLayerAdapter()

    l0_artifact = adapter.create_l0_artifact(run_id, source_path, raw_responses)
    l1_artifact = adapter.create_l1_artifact(run_id, source_path, entities, relations)
    l2_artifact = adapter.create_l2_artifact(run_id, source_path, entities, relations)

    architecture = FiveLayerArchitecture(
        run_id=run_id,
        source_path=source_path,
    )

    architecture.update_layer(FiveLayerType.L0_RAW_EVIDENCE, l0_artifact)
    architecture.update_layer(FiveLayerType.L1_CANDIDATE, l1_artifact)
    architecture.update_layer(FiveLayerType.L2_CANONICAL, l2_artifact)

    return architecture


def get_layer_mapping_summary() -> Dict[str, Dict[str, str]]:
    """获取产物映射摘要"""
    adapter = FiveLayerAdapter()
    mapping = adapter.get_layer_mapping()

    return {
        "existing_products": mapping.existing_products,
        "transformation_rules": mapping.transformation_rules,
    }
