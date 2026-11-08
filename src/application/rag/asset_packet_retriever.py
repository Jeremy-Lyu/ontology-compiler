"""Asset Packet Retriever - RAG consumption of governed asset packages.

[DATA CONTINUITY PLAN 2026-07-09]: Implements RAG consumption of governed assets:
- Read entities from governed asset packages
- Read relations from governed asset packages
- Read evidence chunks from governed asset packages
- Support for hybrid search combining entities and evidence

This module enables:
- RAG directly consumes governed pipeline outputs
- Entity-based semantic search
- Relation-aware retrieval
- Complete provenance tracking in search results
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.application.ports.rag_ports import RAGSearchPort, RAGSearchResult
from src.domain.models.core import RetrievalAssetPacket

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class AssetSearchResult:
    """Search result from asset package."""
    chunk_id: str
    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    entity_refs: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_rag_result(self) -> RAGSearchResult:
        """Convert to standard RAG search result."""
        return RAGSearchResult(
            chunk_id=self.chunk_id,
            content=self.content,
            source=self.source,
            score=self.score,
            metadata={**self.metadata, "entity_refs": self.entity_refs, "provenance": self.provenance},
        )


class AssetPacketRetriever(RAGSearchPort):
    """RAG retriever that consumes governed asset packages directly."""

    def __init__(self, packet_path: Optional[str] = None, packet: Optional[RetrievalAssetPacket | Dict[str, Any]] = None):
        self._asset_packet_service = None
        self._asset_index_service = None
        self._initialized = False
        self._packet_path = Path(packet_path) if packet_path else None
        self._packet = RetrievalAssetPacket.model_validate(packet) if isinstance(packet, dict) else packet

    def load_packet(self, packet_path: Optional[str] = None) -> Optional[RetrievalAssetPacket]:
        """Load the canonical packet emitted by MainPipeline."""
        path = Path(packet_path) if packet_path else self._packet_path
        if self._packet is not None:
            return self._packet
        if not path or not path.exists():
            return None
        try:
            self._packet = RetrievalAssetPacket.model_validate_json(path.read_text(encoding="utf-8"))
            return self._packet
        except Exception as exc:
            logger.warning("Failed to load RetrievalAssetPacket from %s: %s", path, exc)
            return None

    def set_packet(self, packet: RetrievalAssetPacket | Dict[str, Any]) -> None:
        self._packet = RetrievalAssetPacket.model_validate(packet) if isinstance(packet, dict) else packet

    async def _init_services(self):
        """Initialize asset services lazily."""
        if self._initialized:
            return

        try:
            from ontology_compiler.services.asset_packet_service import get_asset_packet_service
            from ontology_compiler.services.asset_index_service import get_asset_index_service

            self._asset_packet_service = get_asset_packet_service()
            self._asset_index_service = get_asset_index_service()
            self._initialized = True
            logger.info("✅ Asset packet retriever initialized")
        except ImportError as e:
            logger.error(f"Failed to initialize asset services: {e}")
            raise

    async def search(self, query: str, top_k: int = 10, **kwargs: Any) -> list[RAGSearchResult]:
        """Search governed asset packages.

        Args:
            query: Search query
            top_k: Number of results to return
            kwargs: Additional parameters:
                - governed_only: Only search governed packages (default: True)
                - include_entities: Include entity matches (default: True)
                - include_evidence: Include evidence matches (default: True)
                - include_relations: Include relation matches (default: True)
                - layer_filter: Filter by entity layer (e.g., "L2")
                - min_confidence: Minimum confidence threshold (default: 0.0)

        Returns:
            List of RAG search results
        """
        packet = self.load_packet(kwargs.get("packet_path"))
        if packet is not None:
            return self._search_packet(packet, query, top_k=top_k, **kwargs)

        await self._init_services()

        governed_only = kwargs.get("governed_only", True)
        include_entities = kwargs.get("include_entities", True)
        include_evidence = kwargs.get("include_evidence", True)
        include_relations = kwargs.get("include_relations", True)
        layer_filter = kwargs.get("layer_filter")
        min_confidence = kwargs.get("min_confidence", 0.0)

        results: List[AssetSearchResult] = []

        if include_entities:
            entity_results = self._search_entities(query, governed_only, layer_filter, min_confidence)
            results.extend(entity_results)

        if include_evidence:
            evidence_results = self._search_evidence(query, governed_only, min_confidence)
            results.extend(evidence_results)

        if include_relations:
            relation_results = self._search_relations(query, governed_only, min_confidence)
            results.extend(relation_results)

        results.sort(key=lambda r: r.score, reverse=True)

        return [r.to_rag_result() for r in results[:top_k]]

    def _search_packet(self, packet: RetrievalAssetPacket, query: str, top_k: int = 10, **kwargs: Any) -> list[RAGSearchResult]:
        """Search a canonical packet without rebuilding data from package files."""
        query_lower = query.lower()
        min_confidence = float(kwargs.get("min_confidence", 0.0) or 0.0)
        layer_filter = kwargs.get("layer_filter")
        results: list[AssetSearchResult] = []

        for entity in packet.entity_refs:
            haystack = " ".join([entity.canonical_name, entity.entity_type, *entity.aliases]).lower()
            if query_lower not in haystack or entity.confidence < min_confidence:
                continue
            if layer_filter and entity.layer != layer_filter:
                continue
            exact = 1.0 if query_lower == entity.canonical_name.lower() else 0.7
            results.append(AssetSearchResult(
                chunk_id=f"entity_{entity.entity_id}",
                content=f"{entity.canonical_name} ({entity.entity_type})",
                source="retrieval_asset_packet.entity",
                score=0.7 * exact + 0.3 * entity.confidence,
                metadata={"entity_id": entity.entity_id, "entity_type": entity.entity_type, "layer": entity.layer, "run_id": packet.run_id},
                entity_refs=[entity.entity_id],
                provenance=packet.provenance,
            ))

        for relation in packet.relation_refs:
            haystack = f"{relation.source_id} {relation.target_id} {relation.relation_type}".lower()
            if query_lower not in haystack or relation.confidence < min_confidence:
                continue
            results.append(AssetSearchResult(
                chunk_id=f"relation_{relation.relation_id}",
                content=f"{relation.source_id} -{relation.relation_type}-> {relation.target_id}",
                source="retrieval_asset_packet.relation",
                score=relation.confidence,
                metadata={"relation_id": relation.relation_id, "relation_type": relation.relation_type, "run_id": packet.run_id},
                entity_refs=[relation.source_id, relation.target_id],
                provenance=packet.provenance,
            ))

        for evidence in packet.evidence_refs:
            content = evidence.span or ""
            if query_lower not in content.lower() or evidence.confidence < min_confidence:
                continue
            results.append(AssetSearchResult(
                chunk_id=f"evidence_{evidence.evidence_id}",
                content=content,
                source="retrieval_asset_packet.evidence",
                score=evidence.confidence,
                metadata={"evidence_id": evidence.evidence_id, "source_document_id": evidence.source_document_id, "run_id": packet.run_id},
                provenance=packet.provenance,
            ))

        results.sort(key=lambda item: item.score, reverse=True)
        return [item.to_rag_result() for item in results[:top_k]]

    async def batch_search(self, queries: list[str], top_k: int = 10, **kwargs: Any) -> list[list[RAGSearchResult]]:
        """Batch search governed asset packages."""
        return [await self.search(q, top_k, **kwargs) for q in queries]

    def _search_entities(self, query: str, governed_only: bool, layer_filter: Optional[str],
                         min_confidence: float) -> List[AssetSearchResult]:
        """Search entities in asset packages."""
        if not self._asset_index_service:
            return []

        results = self._asset_index_service.search_entities(
            query=query,
            governed_only=governed_only,
            layer=layer_filter,
            min_confidence=min_confidence,
        )

        entity_results: List[AssetSearchResult] = []
        query_lower = query.lower()

        for result in results:
            run_id = result["run_id"]
            package_path = Path(result["package_path"])
            entities_file = package_path / "entities.json"

            if not entities_file.exists():
                continue

            try:
                with open(entities_file, "r", encoding="utf-8") as f:
                    entities_data = json.load(f)
                    for entity in entities_data.get("entities", []):
                        if entity.get("id") != result["id"]:
                            continue

                        name_match = query_lower in entity.get("name", entity.get("meta_name", "")).lower()
                        code_match = query_lower in entity.get("code", entity.get("meta_id", "")).lower()
                        type_match = query_lower in entity.get("type", entity.get("meta_class", "")).lower()

                        if name_match or code_match or type_match:
                            confidence = entity.get("confidence", entity.get("evidence", {}).get("confidence", 0.0))
                            match_score = (0.4 if name_match else 0) + \
                                          (0.3 if code_match else 0) + \
                                          (0.3 if type_match else 0)
                            score = 0.7 * match_score + 0.3 * confidence

                            entity_results.append(AssetSearchResult(
                                chunk_id=f"entity_{entity.get('id')}",
                                content=f"{entity.get('name', entity.get('meta_name', ''))} ({entity.get('type', entity.get('meta_class', ''))}): {entity.get('description', '')}",
                                source="governed_asset_entity",
                                score=score,
                                metadata={
                                    "entity_id": entity.get("id"),
                                    "entity_name": entity.get("name", entity.get("meta_name", "")),
                                    "entity_code": entity.get("code", entity.get("meta_id", "")),
                                    "entity_type": entity.get("type", entity.get("meta_class", "")),
                                    "layer": entity.get("layer"),
                                    "confidence": confidence,
                                    "run_id": run_id,
                                },
                                entity_refs=[entity.get("id")],
                                provenance=entity.get("provenance", {}),
                            ))
            except Exception as e:
                logger.warning(f"Failed to search entities in {package_path}: {e}")

        return entity_results

    def _search_evidence(self, query: str, governed_only: bool, min_confidence: float) -> List[AssetSearchResult]:
        """Search evidence chunks in asset packages."""
        if not self._asset_index_service:
            return []

        packages = self._asset_index_service.find_all(governed_only=governed_only)
        query_lower = query.lower()

        evidence_results: List[AssetSearchResult] = []

        for pkg_entry in packages:
            package_path = Path(pkg_entry.package_path)
            evidence_file = package_path / "evidence.json"

            if not evidence_file.exists():
                continue

            try:
                with open(evidence_file, "r", encoding="utf-8") as f:
                    evidence_data = json.load(f)
                    for evidence in evidence_data.get("evidence", []):
                        content = evidence.get("content", "").lower()
                        if query_lower in content:
                            confidence = evidence.get("confidence", 0.0)
                            if confidence >= min_confidence:
                                score = confidence * min(1.0, content.count(query_lower) * 0.2 + 0.5)

                                evidence_results.append(AssetSearchResult(
                                    chunk_id=f"evidence_{evidence.get('id', '')}",
                                    content=evidence.get("content", ""),
                                    source="governed_asset_evidence",
                                    score=score,
                                    metadata={
                                        "evidence_id": evidence.get("id"),
                                        "source_file": evidence.get("source_file"),
                                        "source_line": evidence.get("source_line"),
                                        "confidence": confidence,
                                        "run_id": pkg_entry.run_id,
                                    },
                                    entity_refs=evidence.get("entity_refs", []),
                                    provenance={},
                                ))
            except Exception as e:
                logger.warning(f"Failed to search evidence in {package_path}: {e}")

        return evidence_results

    def _search_relations(self, query: str, governed_only: bool, min_confidence: float) -> List[AssetSearchResult]:
        """Search relations in asset packages."""
        if not self._asset_index_service:
            return []

        packages = self._asset_index_service.find_all(governed_only=governed_only)
        query_lower = query.lower()

        relation_results: List[AssetSearchResult] = []

        for pkg_entry in packages:
            package_path = Path(pkg_entry.package_path)
            relations_file = package_path / "relations.json"

            if not relations_file.exists():
                continue

            try:
                with open(relations_file, "r", encoding="utf-8") as f:
                    relations_data = json.load(f)
                    for relation in relations_data.get("relations", []):
                        relation_type = relation.get("relation_type", "").lower()
                        if query_lower in relation_type:
                            confidence = relation.get("confidence", 0.0)
                            if confidence >= min_confidence:
                                score = confidence

                                relation_results.append(AssetSearchResult(
                                    chunk_id=f"relation_{relation.get('id', '')}",
                                    content=f"{relation.get('source_id')} -{relation.get('relation_type')}-> {relation.get('target_id')}",
                                    source="governed_asset_relation",
                                    score=score,
                                    metadata={
                                        "relation_id": relation.get("id"),
                                        "source_id": relation.get("source_id"),
                                        "target_id": relation.get("target_id"),
                                        "relation_type": relation.get("relation_type"),
                                        "confidence": confidence,
                                        "run_id": pkg_entry.run_id,
                                    },
                                    entity_refs=[relation.get("source_id"), relation.get("target_id")],
                                    provenance=relation.get("provenance", {}),
                                ))
            except Exception as e:
                logger.warning(f"Failed to search relations in {package_path}: {e}")

        return relation_results

    async def get_entity_by_id(self, entity_id: str, governed_only: bool = True) -> Optional[Dict[str, Any]]:
        """Get entity by ID from governed packages."""
        await self._init_services()

        locations = self._asset_index_service.get_entity_locations(entity_id)
        for run_id, package_path in locations:
            pkg_data = self._asset_packet_service.load_asset_packet(run_id, governed=governed_only)
            if pkg_data:
                entities = pkg_data.get("entities", {}).get("entities", [])
                for entity in entities:
                    if entity.get("id") == entity_id:
                        return entity
        return None

    async def get_relations_for_entity(self, entity_id: str, governed_only: bool = True) -> List[Dict[str, Any]]:
        """Get all relations for an entity."""
        await self._init_services()

        relations: List[Dict[str, Any]] = []
        locations = self._asset_index_service.get_entity_locations(entity_id)

        for run_id, package_path in locations:
            pkg_data = self._asset_packet_service.load_asset_packet(run_id, governed=governed_only)
            if pkg_data:
                pkg_relations = pkg_data.get("relations", {}).get("relations", [])
                for relation in pkg_relations:
                    if relation.get("source_id") == entity_id or relation.get("target_id") == entity_id:
                        relations.append(relation)

        return relations

    async def get_governed_stats(self) -> Dict[str, Any]:
        """Get statistics about governed assets."""
        await self._init_services()

        if not self._asset_index_service:
            return {}

        return self._asset_index_service.get_statistics()

    def get_retriever_type(self) -> str:
        """Get retriever type identifier."""
        return "asset_packet"


def get_asset_packet_retriever() -> AssetPacketRetriever:
    """Get or create asset packet retriever instance."""
    return AssetPacketRetriever()


__all__ = [
    "AssetSearchResult",
    "AssetPacketRetriever",
    "get_asset_packet_retriever",
]