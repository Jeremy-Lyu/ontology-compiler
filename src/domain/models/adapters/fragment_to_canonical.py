"""Fragment to Canonical Adapter - Stable adapter from compilation layer to consumption layer.

Provides conversion from Fragment objects to CanonicalResult objects.
"""

from typing import Any, Dict, List, Optional, Union

from ..canonical import (
    CanonicalEntity,
    CanonicalRelation,
    CanonicalEvidence,
    CanonicalResult,
    EntityType,
    RelationType,
)
from ..fragment import Fragment, FragmentNode, FragmentInstance, FragmentRelationship


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""


class FragmentToCanonicalAdapter:
    """Adapter for converting Fragment objects to Canonical objects."""

    @staticmethod
    def _convert_node_to_entity(
        node: Union[FragmentNode, FragmentInstance],
        fragment: Fragment,
    ) -> tuple[CanonicalEntity, CanonicalEvidence]:
        """Convert a FragmentNode or FragmentInstance to CanonicalEntity and CanonicalEvidence."""
        properties = {}
        if hasattr(node, "properties") and node.properties:
            properties = dict(node.properties)

        layer = getattr(node, "layer", "")
        if not layer:
            layer = "L1"

        entity_type = EntityType.SYSTEM
        if hasattr(node, "type") and node.type:
            type_str = str(node.type).lower()
            for et in EntityType:
                if et.value.lower() == type_str:
                    entity_type = et
                    break

        evidence = CanonicalEvidence(
            evidence_id=f"ev_{fragment.metadata.get('run_id', '')}_{node.meta_id}",
            source_file=fragment.metadata.get("source_file", ""),
            raw_text=getattr(node, "description", "") or getattr(node, "content", ""),
            extraction_rule="fragment_to_canonical",
            confidence=getattr(node, "confidence", 1.0),
            created_at=fragment.metadata.get("generated_at", ""),
        )

        entity = CanonicalEntity(
            entity_id=node.meta_id,
            name=getattr(node, "name", node.meta_id),
            main_class=entity_type.value,
            layer=layer,
            description=getattr(node, "description", ""),
            properties=properties,
            evidence=[evidence],
        )
        return entity, evidence

    @staticmethod
    def _convert_relation_type(rel_type: str) -> RelationType:
        """Convert fragment relation type to Canonical RelationType."""
        rel_type_lower = str(rel_type).lower()
        if rel_type_lower in {"part_of", "partof", "is_part_of"}:
            return RelationType.CONTAINS
        elif rel_type_lower in {"uses", "use", "consumes"}:
            return RelationType.USES
        elif rel_type_lower in {"produces", "generates", "creates"}:
            return RelationType.PRODUCES
        elif rel_type_lower in {"defines", "def", "describes"}:
            return RelationType.CONTAINS
        elif rel_type_lower in {"executes", "performs", "runs"}:
            return RelationType.CALLS
        elif rel_type_lower in {"depends_on", "depends", "requires"}:
            return RelationType.DEPENDS_ON
        elif rel_type_lower in {"implements", "realizes"}:
            return RelationType.IMPLEMENTS
        elif rel_type_lower in {"triggers", "fires", "activates"}:
            return RelationType.TRIGGERS
        else:
            return RelationType.ASSOCIATED_WITH

    @staticmethod
    def fragment_to_result(
        fragment: Fragment,
        run_id: Optional[str] = None,
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> CanonicalResult:
        """Convert a Fragment to CanonicalResult."""
        entities: List[CanonicalEntity] = []
        relations: List[CanonicalRelation] = []
        evidence_list: List[CanonicalEvidence] = []

        entity_id_map: Dict[str, str] = {}

        for node in fragment.nodes:
            entity, evidence = FragmentToCanonicalAdapter._convert_node_to_entity(node, fragment)
            entities.append(entity)
            evidence_list.append(evidence)
            entity_id_map[node.meta_id] = node.meta_id

        for instance in fragment.instances:
            entity, evidence = FragmentToCanonicalAdapter._convert_node_to_entity(instance, fragment)
            entities.append(entity)
            evidence_list.append(evidence)
            entity_id_map[instance.meta_id] = instance.meta_id

        for rel in fragment.relationships:
            source_id = entity_id_map.get(rel.source, rel.source)
            target_id = entity_id_map.get(rel.target, rel.target)

            relation_type = FragmentToCanonicalAdapter._convert_relation_type(rel.type)

            properties = {}
            if hasattr(rel, "properties") and rel.properties:
                properties = dict(rel.properties)

            relations.append(CanonicalRelation(
                relation_id=f"rel_{rel.source}_{rel.target}",
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type.value,
                confidence=getattr(rel, "confidence", 0.75),
                properties=properties,
            ))

        return CanonicalResult(
            run_id=run_id or fragment.metadata.get("run_id", ""),
            entities=entities,
            relations=relations,
            evidence=evidence_list,
            metadata={
                "source_file": fragment.metadata.get("source_file", ""),
                "generated_at": fragment.metadata.get("generated_at", ""),
                "prompt_context": prompt_context or {},
            },
        )

    @staticmethod
    def node_to_entity(node: Union[FragmentNode, FragmentInstance]) -> CanonicalEntity:
        """Convert a single node to entity (simplified version for validation)."""
        fragment = Fragment(nodes=[], instances=[], relationships=[], metadata={})
        entity, _ = FragmentToCanonicalAdapter._convert_node_to_entity(node, fragment)
        return entity


def fragment_to_canonical(fragment: Fragment) -> CanonicalResult:
    """Convert a Fragment to CanonicalResult."""
    return FragmentToCanonicalAdapter.fragment_to_result(fragment)


def batch_fragments_to_canonical(fragments: List[Fragment]) -> CanonicalResult:
    """Convert multiple fragments to a single CanonicalResult."""
    all_entities: List[CanonicalEntity] = []
    all_relations: List[CanonicalRelation] = []
    all_evidence: List[CanonicalEvidence] = []

    for fragment in fragments:
        result = FragmentToCanonicalAdapter.fragment_to_result(fragment)
        all_entities.extend(result.entities)
        all_relations.extend(result.relations)
        all_evidence.extend(result.evidence)

    return CanonicalResult(
        run_id=fragments[0].metadata.get("run_id", "") if fragments else "",
        entities=all_entities,
        relations=all_relations,
        evidence=all_evidence,
        metadata={
            "fragment_count": len(fragments),
        },
    )
