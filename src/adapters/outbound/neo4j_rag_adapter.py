"""Neo4j RAG Adapter - Graph database retrieval implementation.

G26-Unify-04: Adapter that wraps the Neo4j retrieval components
to implement the RAGSearchPort interface.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.application.ports.rag_ports import (
    RAGSearchPort,
    RAGSearchResult,
    RAGSearchError,          # P0 BUGFIX 2026-08-04: explicit imports (was NameError at runtime)
    RAGBackendUnavailableError,
)
from src.application.rag.models import RetrievalResult, Neo4jSearchResult


class Neo4jRAGAdapter:
    """Adapter for Neo4j graph-based retrieval.
    
    Wraps the existing Neo4jRetriever class to implement RAGSearchPort.
    
    Args:
        uri: Neo4j database URI.
        user: Database username.
        password: Database password.
        config: Optional configuration dictionary.
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "",
        config: dict[str, Any] | None = None,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.config = config or {}
        self._retriever = None
        self._available: bool | None = None
    
    def _get_retriever(self):
        """Lazy initialization of Neo4j retriever."""
        if self._retriever is None:
            try:
                from src.application.rag.infrastructure.neo4j_retriever import Neo4jRetriever
                self._retriever = Neo4jRetriever(
                    uri=self.uri,
                    user=self.user,
                    password=self.password,
                )
            except ImportError as e:
                raise RAGBackendUnavailableError(f"Neo4j retriever not available: {e}") from e
        return self._retriever
    
    async def search(self, query: str, topk: int = 5) -> RetrievalResult:
        """Perform a search using Neo4j graph database.

        PHASE 1-5 H18 FIX (2026-08-04):
        The Neo4j retriever runs synchronous driver queries (Bolt protocol
        round-trips + full-text index lookups).  To avoid blocking the async
        event loop we wrap the synchronous payload in ``asyncio.to_thread``
        (see ``LocalRAGAdapter.search`` docstring for the concurrency impact).
        """
        start_time = time.time()

        if not self.is_available():
            raise RAGBackendUnavailableError("Neo4j backend is not available")

        try:
            def _run_sync() -> tuple:
                t0 = time.time()
                retriever = self._get_retriever()
                neo4j_results = retriever.search(query, limit=topk)
                chunks = neo4j_results if isinstance(neo4j_results, list) else []
                return chunks, (time.time() - t0) * 1000

            chunks, execution_time = await asyncio.to_thread(_run_sync)
            return RetrievalResult(
                chunks=chunks,
                source_type="neo4j",
                query=query,
                topk=topk,
                execution_time_ms=execution_time,
            )
        except RAGBackendUnavailableError:
            raise
        except Exception as e:
            raise RAGSearchError(f"Neo4j RAG search failed: {e}") from e
    
    def is_available(self) -> bool:
        """Check if Neo4j is available and connected."""
        if self._available is None:
            try:
                # Try to connect to Neo4j
                retriever = self._get_retriever()
                self._available = retriever.is_connected() if hasattr(retriever, 'is_connected') else True
            except Exception:
                self._available = False
        return self._available
    
    def get_source_type(self) -> str:
        """Return the source type identifier."""
        return "neo4j"


# Register as implementing RAGSearchPort
RAGSearchPort.register(Neo4jRAGAdapter)
