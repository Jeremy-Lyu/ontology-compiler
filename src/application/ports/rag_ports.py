"""RAG bounded-context port definitions.

Ports, DTOs and exceptions for RAG search and feedback. Split from
`all_ports.py` during P2 port classification (§9.8) so the RAG bounded
context owns its retrieval contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RAGSearchResult:
    """Unified result format for RAG search operations."""
    chunk_id: str
    content: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGSearchPort(ABC):
    """Unified port interface for RAG search operations.

    Provides a common interface for both internal (local) and external
    (remote/service) retrievers, enabling seamless switching between
    different retrieval strategies.

    M04 - Unified RAG Access Layer (2026-06-10)
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any
    ) -> list[RAGSearchResult]:
        """Search for relevant chunks based on query.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            **kwargs: Additional search parameters (context, filters, etc.)

        Returns:
            List of RAGSearchResult sorted by relevance (descending).
        """
        raise NotImplementedError

    @abstractmethod
    async def batch_search(
        self,
        queries: list[str],
        top_k: int = 10,
        **kwargs: Any
    ) -> list[list[RAGSearchResult]]:
        """Batch search for multiple queries.

        Args:
            queries: List of search query strings.
            top_k: Maximum number of results per query.
            **kwargs: Additional search parameters.

        Returns:
            List of result lists, one for each query.
        """
        raise NotImplementedError

    @abstractmethod
    def get_retriever_type(self) -> str:
        """Return the type of retriever (e.g., 'local', 'neo4j', 'dify')."""
        raise NotImplementedError


class RAGSearchError(Exception):
    """Base exception for RAG search operations."""
    pass


class RAGBackendUnavailableError(RAGSearchError):
    """Raised when a RAG backend is not available."""
    pass


class RAGSearchTimeoutError(RAGSearchError):
    """Raised when a RAG search operation times out."""
    pass


class RAGFeedbackPort(ABC):
    """Port for collecting RAG feedback records."""

    @abstractmethod
    def collect_rag_feedback(
        self,
        *,
        query: str,
        answer: str,
        rating: str,
        user_comment: str = "",
        session_id: str | None = None,
        run_id: str | None = None,
        retrieved_sources: list[dict[str, Any]] | None = None,
        trace_ref: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Collect and normalize RAG feedback."""
        pass


__all__ = [
    "RAGSearchResult",
    "RAGSearchPort",
    "RAGSearchError",
    "RAGBackendUnavailableError",
    "RAGSearchTimeoutError",
    "RAGFeedbackPort",
]
