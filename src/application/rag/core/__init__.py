"""Core RAG orchestration primitives.

This package hosts the canonical service/factory layer for unified RAG access.
The package-level `rag/core.py` module re-exports the stable orchestration-facing
symbols; `src.application.rag` remains a lazy compatibility shell.
"""

from .unified_rag_service import (
    AssetPacketServiceRetrieverAdapter,
    DifyRetrieverAdapter,
    LocalRetrieverAdapter,
    Neo4jRetrieverAdapter,
    RetrieverFactory,
    UnifiedRAGService,
    get_rag_service,
)

__all__ = [
    "AssetPacketServiceRetrieverAdapter",
    "DifyRetrieverAdapter",
    "LocalRetrieverAdapter",
    "Neo4jRetrieverAdapter",
    "RetrieverFactory",
    "UnifiedRAGService",
    "get_rag_service",
]
