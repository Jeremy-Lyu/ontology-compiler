"""Unified RAG Data Models

This module contains the unified data models for the RAG system,
replacing scattered definitions across multiple files.

G26-Unify-01: Consolidate all RAG-related data classes into a single module
for consistency and maintainability.

G26-Unify-02: Introduce BaseRetrievalItem protocol for unified retrieval result handling
across different retrieval sources (local BM25, Neo4j graph, etc.).
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Sequence, Protocol

from src.domain.models.evidence import Evidence


@dataclass(frozen=True)
class ChunkRecord:
    """A retrieval unit derived from ontology nodes/edges.

    G26-Evidence-01: Enhanced with evidence chain for full traceability
    from RAG answer back to original document lines.
    """

    chunk_id: str
    chunk_type: str
    content: str
    node_ids: List[str]
    linked_node_ids: List[str]
    meta: Dict[str, Any]
    # G26-Evidence-01: Evidence chain fields - using unified Evidence model
    evidence: Evidence | None = None  # Evidence information for traceability

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        result = asdict(self)
        if self.evidence is not None:
            result["evidence"] = self.evidence.model_dump()
        else:
            result.pop("evidence", None)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkRecord":
        """Deserialize from a JSON-compatible dict."""
        # Handle evidence field - default to None if not present or None
        evidence_data = data.get("evidence")
        evidence = Evidence.from_dict(evidence_data) if evidence_data else None
        return cls(
            chunk_id=data["chunk_id"],
            chunk_type=data["chunk_type"],
            content=data["content"],
            node_ids=data["node_ids"],
            linked_node_ids=data["linked_node_ids"],
            meta=data["meta"],
            evidence=evidence
        )

    def get_evidence_summary(self) -> str:
        """Get human-readable evidence summary."""
        if not self.evidence:
            return "No evidence recorded"

        parts = []
        if self.evidence.source_file:
            parts.append(f"Source: {self.evidence.source_file}")
        if self.evidence.line_range:
            parts.append(f"Lines: {self.evidence.line_range}")
        if self.evidence.confidence:
            parts.append(f"Confidence: {self.evidence.confidence}")

        return " | ".join(parts) if parts else "Evidence recorded (incomplete)"

    def can_trace_to_source(self) -> bool:
        """Check if this chunk has complete evidence for source tracing."""
        if not self.evidence:
            return False
        return bool(
            self.evidence.get("source_file")
            and self.evidence.get("line_range")
        )


class BaseRetrievalItem(Protocol):
    """Protocol defining the common interface for all retrieval result items.
    
    G26-Unify-02: All retrieval sources (local, Neo4j, etc.) must implement this
    interface to ensure consistent handling in the answerer pipeline.
    """
    
    @property
    def content(self) -> str:
        """Get the textual content of the retrieval item."""
        ...
    
    @property
    def source(self) -> str:
        """Get the source identifier (e.g., file path, database ID)."""
        ...
    
    @property
    def score(self) -> float:
        """Get the relevance score (0.0-1.0)."""
        ...
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get additional metadata associated with the item."""
        ...


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk with a simple score."""

    chunk: ChunkRecord
    score: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {"chunk": self.chunk.to_dict(), "score": self.score}
    
    @property
    def content(self) -> str:
        """Get the textual content of the retrieval item."""
        return self.chunk.content
    
    @property
    def source(self) -> str:
        """Get the source identifier."""
        return self.chunk.meta.get("source_file", "") or str(self.chunk.chunk_id)
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get additional metadata associated with the item."""
        return {
            "chunk_type": self.chunk.chunk_type,
            "node_ids": self.chunk.node_ids,
            "linked_node_ids": self.chunk.linked_node_ids,
            **self.chunk.meta,
        }


@dataclass(frozen=True)
class ChunkWithVector:
    """A chunk with vector and keyword scores for hybrid retrieval."""

    chunk: RetrievedChunk
    vector_score: float = 0.0
    keyword_score: float = 0.0
    hybrid_score: float = 0.0


@dataclass(frozen=True)
class LLMAnswerResult:
    """LLM 生成的答案结果"""
    answer: str
    evidence_chunks: List[Dict[str, Any]]  # 使用的证据 chunks
    confidence: float
    model: str
    tokens_used: Optional[int] = None


@dataclass(frozen=True)
class RetrievalResult:
    """Unified retrieval result combining multiple retrieval sources."""
    
    chunks: List["BaseRetrievalItem"]  # Union of RetrievedChunk, Neo4jSearchResult, etc.
    source_type: str  # "local", "neo4j", "hybrid", "graph"
    query: str
    topk: int
    execution_time_ms: float


@dataclass(frozen=True)
class Neo4jSearchResult:
    """Search result from Neo4j graph database."""
    
    entity_name: str
    entity_type: str
    description: str
    related_entities: List[Dict[str, str]]
    scenarios: List[str]
    score: float = 0.0
    
    @property
    def content(self) -> str:
        """Get the textual content of the retrieval item."""
        return f"{self.entity_name}: {self.description}"
    
    @property
    def source(self) -> str:
        """Get the source identifier."""
        return f"neo4j:{self.entity_type}:{self.entity_name}"
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get additional metadata associated with the item."""
        return {
            "entity_type": self.entity_type,
            "related_entities": self.related_entities,
            "scenarios": self.scenarios,
        }


@dataclass(frozen=True)
class RewrittenQuery:
    """重写后的查询结果。"""
    original: str
    rewritten: str
    method: str  # "term_mapping" | "llm_rewrite" | "none"
    confidence: float  # 0-1，表示重写可信度


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for RAG system."""
    
    rag_dir: str = "outputs/ontology/rag"
    model: str = "deepseek-chat"
    max_context_chunks: int = 5
    max_chunk_length: int = 1000
    use_neo4j: bool = False
    neo4j_weight: float = 0.3
    hybrid_alpha: float = 0.5  # Weight between vector and keyword


# Type aliases for consistency
RetrievalResults = List[RetrievedChunk]
ChunkList = Sequence[ChunkRecord]


# =============================================================================
# Shared constants (merged from _constants.py - ST06-05 2026-06-14)
# =============================================================================

# -- Edge types (system → capability) --
EDGE_TYPES_SYSTEM_TO_CAP: frozenset[str] = frozenset({"SUPPORTS", "REALIZES", "ENABLES"})

# -- Chunk ID prefixes --
CHUNK_PREFIX_CAP: str = "cap_"
CHUNK_PREFIX_PATH: str = "path_"
CHUNK_PREFIX_SYSMAP: str = "sysmap_"

# -- Chunk type constants --
CHUNK_TYPE_CAPABILITY_CARD: str = "capability_card"
CHUNK_TYPE_PATH_CARD: str = "path_card"
CHUNK_TYPE_SYSTEM_MAPPING_CARD: str = "system_mapping_card"
CHUNK_TYPE_PROCESS_CARD: str = "process_card"

# -- Chunk ID prefixes (additional) --
CHUNK_PREFIX_PROC: str = "proc_"
CHUNK_PREFIX_DATA: str = "data_"
CHUNK_PREFIX_ACTOR: str = "actor_"
CHUNK_PREFIX_ASSET: str = "asset_"

# -- Chunk type constants (additional meta classes) --
CHUNK_TYPE_DATA_CARD: str = "data_card"
CHUNK_TYPE_ACTOR_CARD: str = "actor_card"
CHUNK_TYPE_ASSET_CARD: str = "asset_card"

# -- Scenario card constants --
CHUNK_TYPE_SCENARIO_CARD: str = "scenario_card"
CHUNK_PREFIX_SCENARIO: str = "scenario_"

# -- Dify response modes --
RESPONSE_MODE_BLOCKING: str = "blocking"
RESPONSE_MODE_STREAMING: str = "streaming"