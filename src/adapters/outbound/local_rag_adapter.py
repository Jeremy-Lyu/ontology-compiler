"""Local RAG Adapter - BM25 + Vector search implementation.

G26-Unify-04: Adapter that wraps the local RAG retrieval components
(indexer, retriever) to implement the RAGSearchPort interface.

BATCH 3 Consolidation (2026-06-29):
- Migrated from legacy Retriever to canonical HybridIndexManager
- Uses hybrid_search for combined BM25 + vector retrieval
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from src.application.ports.rag_ports import (
    RAGSearchPort,
    RAGSearchResult,
    RAGSearchError,  # P0 BUGFIX 2026-08-04: use canonical port exception (removed duplicate shadow class)
)
from src.application.rag.models import RetrievalResult, RetrievedChunk
from src.application.rag.hybrid_rag import HybridIndexManager, HybridSearchResult


class LocalRAGAdapter:
    """Adapter for local BM25 + vector-based retrieval.
    
    Wraps HybridIndexManager to implement RAGSearchPort.
    
    Args:
        rag_dir: Path to the RAG index directory.
        config: Optional configuration dictionary.
    """
    
    def __init__(self, rag_dir: str | Path | None = None, config: dict[str, Any] | None = None):
        from src.application.rag.governed_packets import resolve_governed_rag_dir

        self.rag_dir = resolve_governed_rag_dir(Path(rag_dir) if rag_dir else None)
        self.config = config or {}
        self._index_manager: HybridIndexManager | None = None
    
    def _get_index_manager(self) -> HybridIndexManager:
        """Lazy initialization of index manager."""
        if self._index_manager is None:
            self._index_manager = HybridIndexManager(self.rag_dir)
        return self._index_manager
    
    def _convert_results(self, hybrid_results: list[HybridSearchResult]) -> list[RetrievedChunk]:
        """Convert HybridSearchResult to RetrievedChunk."""
        return [
            RetrievedChunk(chunk=r.chunk, score=r.hybrid_score)
            for r in hybrid_results
        ]
    
    async def search(self, query: str, topk: int = 5) -> RetrievalResult:
        """Perform a search using local BM25 + vector index.

        PHASE 1-5 H18 FIX (2026-08-04):
        - HybridIndexManager.hybrid_search runs synchronous BM25/NumPy code.
          Directly calling it inside an ``async def`` body blocks the entire
          event loop for O(10-100 ms) per query, which starves other concurrent
          requests in a FastAPI server.
        - Wrap the synchronous payload (get manager + hybrid_search + convert)
          in ``asyncio.to_thread`` so it runs on a worker thread and the loop
          is free to process other I/O concurrently.
        """
        start_time = time.time()

        try:
            def _run_sync() -> tuple:
                index_manager = self._get_index_manager()
                results = index_manager.hybrid_search(query, topk=topk)
                chunks = self._convert_results(results)
                return chunks, (time.time() - start_time) * 1000

            chunks, execution_time = await asyncio.to_thread(_run_sync)
            return RetrievalResult(
                chunks=chunks,
                source_type="local",
                query=query,
                topk=topk,
                execution_time_ms=execution_time,
            )
        except Exception as e:
            raise RAGSearchError(f"Local RAG search failed: {e}") from e
    
    def is_available(self) -> bool:
        """Check if the governed packet files exist.

        NOTE: Real contract integrity (checksums, required files) is enforced by
        ``governed_packets.require_governed_packet``; this method only tells
        callers (service wiring) whether the asset path can be used, without
        incurring the full manifest verify cost on every health probe.
        """
        required_paths = (
            self.rag_dir / "chunks.jsonl",
            self.rag_dir / "inverted_index.json",
            self.rag_dir / "vector_index" / "metadata.json",
            self.rag_dir / "manifest.json",
            self.rag_dir / "packet.json",
        )
        return all(p.exists() for p in required_paths)
    
    def get_source_type(self) -> str:
        """Return the source type identifier."""
        return "local"


# Register as implementing RAGSearchPort
RAGSearchPort.register(LocalRAGAdapter)