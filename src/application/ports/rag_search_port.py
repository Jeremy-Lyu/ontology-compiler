"""RAG Search Port - Abstract interface for retrieval-augmented generation.

G26-Unify-04: Defines the unified port (interface) that all RAG retrieval
backends must implement. This enables dependency injection and allows
the answerer to work with any retrieval source (local BM25, Neo4j, hybrid).

This follows the hexagonal architecture pattern where:
- Ports (this module) define the interface
- Adapters (in src/adapters/) provide concrete implementations
- The application core depends only on ports, not adapters
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.application.rag.models import RetrievalResult


@runtime_checkable
class RAGSearchPort(Protocol):
    """Protocol defining the interface for RAG search backends.
    
    All retrieval implementations (local, Neo4j, hybrid, etc.) must implement
    this interface to be usable by the answerer pipeline.
    
    Methods:
        search: Perform a search query and return retrieval results.
        is_available: Check if the backend is currently available.
        get_source_type: Return a string identifier for this backend type.
    
    Example:
        class LocalRAGAdapter:
            async def search(self, query: str, topk: int = 5) -> RetrievalResult:
                # ... implementation ...
                return RetrievalResult(chunks=..., source_type="local", ...)
            
            def is_available(self) -> bool:
                return True
            
            def get_source_type(self) -> str:
                return "local"
    """
    
    async def search(self, query: str, topk: int = 5) -> RetrievalResult:
        """Perform a search query and return retrieval results.
        
        Args:
            query: The search query string.
            topk: Maximum number of results to return (default: 5).
        
        Returns:
            RetrievalResult containing the retrieved chunks and metadata.
        
        Raises:
            RAGSearchError: If the search operation fails.
        """
        ...
    
    def is_available(self) -> bool:
        """Check if the backend is currently available.
        
        Returns:
            True if the backend can process queries, False otherwise.
        """
        ...
    
    def get_source_type(self) -> str:
        """Return a string identifier for this backend type.
        
        Returns:
            A string like "local", "neo4j", "hybrid", "graph", etc.
        """
        ...


class RAGSearchError(Exception):
    """Base exception for RAG search operations."""
    pass


class RAGBackendUnavailableError(RAGSearchError):
    """Raised when a RAG backend is not available."""
    pass


class RAGSearchTimeoutError(RAGSearchError):
    """Raised when a RAG search operation times out."""
    pass
