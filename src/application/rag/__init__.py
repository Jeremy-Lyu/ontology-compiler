"""RAG components for the ontology system.

边界说明（Phase 3）：
- `rag.core`（core/ 包）是 RAG 编排原语（服务/工厂层）的 canonical owner。
- `rag.core`（core.py 模块）是面向编排的 stable re-export 聚合入口。
- 本包 `src.application.rag` 保留 top-level 兼容 re-export（lazy），
  兼容层不应作为新代码的依赖面。
- RAG 属于知识消费层，与 pipeline 编排（src/pipeline）边界分离：
  pipeline 只消费资产包/检索产物，不直接调用 RAG 编排。
"""

from importlib import import_module
from typing import Any

from .core import *

__all__ = [
    "HybridRetriever",
    "HybridSearchResult",
    "RetrievalConfig",
    "RetrievalResult",
    "hybrid_search",
    "build_hybrid_index",
    "HybridIndexManager",
    "HybridIndexArtifacts",
    "QueryClassifier",
    "QueryType",
    "QueryIntent",
    "QueryComplexity",
    "QueryClassification",
    "UnifiedQueryProcessor",
    "parse",
    "ParsedQuery",
    "QueryRewriter",
    "StructuredIndex",
    "EntityIndex",
    "RelationIndex",
    "EvidenceIndex",
    "IndexStats",
    "Neo4jRetriever",
    "Neo4jSyncService",
    "SyncConfig",
    "SyncState",
    "register_neo4j_sync_hook",
    "UnifiedRAGService",
    "RetrieverFactory",
    "LocalRetrieverAdapter",
    "Neo4jRetrieverAdapter",
    "DifyRetrieverAdapter",
    "get_rag_service",
    "ChunkRecord",
    "build_chunks",
    "build_chunks_from_domain_ontology",
    "build_chunks_with_evidence",
    "infer_capability_level",
    "write_chunks_jsonl",
    "load_chunks_jsonl",
    "load_inverted_index",
    "retrieve_keyword",
    "retrieve_from_rag_dir",
    "EvidenceBinder",
    "EvidenceBinding",
    "EvidenceSource",
    "AssetPacketRetriever",
    "get_asset_packet_retriever",
    "derive_rag_objects",
    "rag_chunks_from_packet",
    "write_packet_chunks",
    "build_hybrid_index_from_packet",
    "canonical_models_from_packet",
    "index_packet",
    "entity_drift_report",
]

_MODULE_MAP = {
    "HybridRetriever": ".hybrid_rag",
    "RAGSearchResult": ".hybrid_rag",
    "RetrievalConfig": ".hybrid_rag",
    "hybrid_search": ".hybrid_rag",
    "build_hybrid_index": ".hybrid_rag",
    "RetrievalResult": ".hybrid_rag",
    "HybridSearchResult": ".hybrid_rag",
    "HybridIndexArtifacts": ".hybrid_rag",
    "HybridIndexManager": ".hybrid_rag",
    "QueryClassifier": ".domain.query_core",
    "QueryType": ".domain.query_core",
    "QueryIntent": ".domain.query_core",
    "QueryComplexity": ".domain.query_core",
    "QueryClassification": ".domain.query_core",
    "UnifiedQueryProcessor": ".domain.query_core",
    "parse": ".domain.query_core",
    "ParsedQuery": ".domain.query_core",
    "QueryRewriter": ".domain.query_core",
    "DifyConfig": ".infrastructure.dify_client",
    "StructuredIndex": ".structured_index",
    "EntityIndex": ".structured_index",
    "RelationIndex": ".structured_index",
    "EvidenceIndex": ".structured_index",
    "IndexStats": ".structured_index",
    "VectorIndexManager": ".infrastructure.vector_index_manager",
    "LocalEmbedder": ".infrastructure.embedding",
    "OpenAIEmbedder": ".infrastructure.embedding",
    "EmbeddingConfig": ".infrastructure.embedding",
    "EmbedderBase": ".infrastructure.embedding",
    "ExtendedEmbedderConfig": ".infrastructure.embedding",
    "CohereEmbedder": ".infrastructure.embedding",
    "SentenceTransformersEmbedder": ".infrastructure.embedding",
    "create_embedder": ".infrastructure.embedding",
    "get_embedder": ".infrastructure.embedding",
    "Neo4jRetriever": ".infrastructure.neo4j_retriever",
    "Neo4jSyncService": ".infrastructure.neo4j_sync_service",
    "SyncConfig": ".infrastructure.neo4j_sync_service",
    "SyncState": ".infrastructure.neo4j_sync_service",
    "register_neo4j_sync_hook": ".infrastructure.neo4j_sync_service",
    "UnifiedRAGService": ".core",
    "RetrieverFactory": ".core",
    "LocalRetrieverAdapter": ".core",
    "Neo4jRetrieverAdapter": ".core",
    "DifyRetrieverAdapter": ".core",
    "get_rag_service": ".core",
    "EDGE_TYPES_SYSTEM_TO_CAP": ".domain.models",
    "ChunkRecord": ".chunk_indexing",
    "build_chunks": ".chunk_indexing",
    "build_chunks_from_domain_ontology": ".chunk_indexing",
    "build_chunks_with_evidence": ".chunk_indexing",
    "infer_capability_level": ".chunk_indexing",
    "write_chunks_jsonl": ".chunk_indexing",
    "load_chunks_jsonl": ".chunk_indexing",
    "load_inverted_index": ".chunk_indexing",
    "retrieve_keyword": ".retriever",
    "retrieve_from_rag_dir": ".retriever",
    "EvidenceBinder": ".evidence_binder",
    "AssetPacketRetriever": ".asset_packet_retriever",
    "get_asset_packet_retriever": ".asset_packet_retriever",
    "derive_rag_objects": ".packet_derivation",
    "rag_chunks_from_packet": ".packet_index_bridge",
    "write_packet_chunks": ".packet_index_bridge",
    "build_hybrid_index_from_packet": ".packet_index_bridge",
    "canonical_models_from_packet": ".structured_packet_bridge",
    "index_packet": ".structured_packet_bridge",
    "entity_drift_report": ".entity_drift",
    "EvidenceBinding": ".evidence_binder",
    "EvidenceSource": ".evidence_binder",
}


def __getattr__(name: str) -> Any:
    if name not in _MODULE_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_MODULE_MAP[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
