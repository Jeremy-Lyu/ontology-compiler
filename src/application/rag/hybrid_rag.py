"""混合 RAG 模块 - 统一索引构建与检索

本模块合并了原 hybrid_indexer.py 和 hybrid_retriever.py 的功能，提供：
1. 统一的数据模型（RAGSearchResult）
2. 索引构建（HybridIndexManager）
3. 混合检索（HybridRetriever）
4. 查询改写与规则检索增强

架构目标：
- G4: 统一知识消费层
- G5: 统一质量闭环

[Phase 3 - RAG 2026-06-26]: Merged from hybrid_indexer.py and hybrid_retriever.py
"""

from __future__ import annotations

import json
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime

from src.domain.errors import DataContractError
from src.application.rag.chunk_indexing import (
    ChunkRecord,
    tokenize,
    build_inverted_index,
    compute_idf,
    write_inverted_index,
    write_idf,
    load_chunks_jsonl,
    incremental_update_inverted_index,
)

# =====================================================================
# [RAG-OPT3] 中文 + 英文通用停用词表（纯 BM25 打分时过滤，降低噪音）
# 说明：标准 NLTK/sklearn 停用词是英文的；中文我们不用 jieba（避免额外依赖），
# 而是直接在 token 级别过滤中文虚词 + 常见无信息词汇。若之后引入 jieba
# 分词，可把表换成哈工大停用词表。
# =====================================================================
_BM25_STOPWORDS: set[str] = {
    # ---- 英文常用停用词 ----
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for",
    "to", "of", "in", "on", "at", "with", "by", "from", "as", "is", "are",
    "be", "been", "being", "was", "were", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "must", "can",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "we", "you", "i", "me", "him", "her", "them", "us", "my", "your",
    "his", "their", "our", "what", "which", "who", "whom", "when", "where",
    "why", "how", "not", "no", "nor", "so", "than", "too", "very", "just",
    "about", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "over", "again", "further", "also",
    "once", "here", "there", "both", "each", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "any", "all",
    # ---- 中文常见无信息虚词/助词（按单字拆分，因为无分词依赖的 tokenize 是按标点/空格切） ----
    "的", "了", "和", "是", "就", "都", "而", "及", "與", "並", "或", "但",
    "在", "於", "从", "自", "到", "至", "对", "对於", "对于", "关于", "有关",
    "将", "被", "把", "让", "使", "给", "向", "往", "由", "以", "为",
    "之", "等", "等等", "所", "其", "此", "那", "这", "些", "哪",
    "个", "一个", "每", "各", "某", "该", "本", "另", "其他", "其它",
    "可以", "可能", "应该", "需要", "必须", "能够", "已", "已经", "曾", "曾经",
    "会", "要", "能", "可", "得", "着", "过", "也", "又", "再", "还",
    "很", "最", "更", "较", "比", "非常", "十分", "相当",
    "请", "谢谢", "您好", "你好", "我们", "你们", "他们", "大家",
    # ---- 常见低信息量业务词汇（在每个 chunk 里几乎都出现 -> IDF 本来就低，再过滤一次更干净） ----
    "system", "process", "data", "entity", "object", "item", "value", "type",
    "use", "used", "using", "include", "includes", "including", "contain",
    "see", "also", "note", "notes", "ref", "refer", "reference",
}
from src.application.rag.infrastructure.vector_index_manager import (
    VectorIndexManager,
    VectorIndexMetadata,
    IndexVersionInfo,
    LayeredSimilarityResult,
)
from src.application.rag.structured_index import (
    StructuredIndex,
    EntityIndex,
    RelationIndex,
    EvidenceIndex,
)
from src.application.rag.unified_query import QueryRewriter as BaseQueryRewriter, RewrittenQuery


@dataclass(frozen=True)
class RAGSearchResult:
    """统一的 RAG 检索结果"""

    chunk: Optional[ChunkRecord] = None
    result_id: str = ""
    content: str = ""
    source: str = ""
    source_type: str = ""
    keyword_score: float = 0.0
    vector_score: float = 0.0
    rule_score: float = 0.0
    hybrid_score: float = 0.0
    confidence: float = 0.0
    evidence: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.evidence is None:
            object.__setattr__(self, "evidence", [])
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "result_id": self.result_id,
            "content": self.content,
            "source": self.source,
            "source_type": self.source_type,
            "keyword_score": self.keyword_score,
            "vector_score": self.vector_score,
            "rule_score": self.rule_score,
            "hybrid_score": self.hybrid_score,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }
        if self.chunk:
            result["chunk"] = self.chunk.to_dict()
        return result

    @classmethod
    def from_hybrid_search_result(cls, result: "HybridSearchResult") -> "RAGSearchResult":
        """从旧的 HybridSearchResult 转换"""
        return cls(
            chunk=result.chunk,
            result_id=f"chunk-{result.chunk.chunk_id}" if result.chunk else "",
            content=result.chunk.content if result.chunk else "",
            source="hybrid_index",
            source_type="chunk",
            keyword_score=result.keyword_score,
            vector_score=result.vector_score,
            hybrid_score=result.hybrid_score,
            confidence=result.hybrid_score,
        )


@dataclass
class HybridSearchResult:
    """混合检索结果（向后兼容）"""

    chunk: ChunkRecord
    keyword_score: float
    vector_score: float
    structured_score: float = 0.0
    graph_score: float = 0.0
    hybrid_score: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "chunk": self.chunk.to_dict(),
            "keyword_score": self.keyword_score,
            "vector_score": self.vector_score,
            "hybrid_score": self.hybrid_score,
        }


@dataclass
class HybridIndexArtifacts:
    """混合索引构建产物"""

    chunk_count: int
    token_count: int
    vector_dimension: int
    inverted_index_path: str
    idf_path: str
    vector_index_path: str
    metadata_path: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RetrievalConfig:
    """检索配置

    ``fusion_mode`` (G6, 2026-08-17): "weighted" keeps the legacy weighted-sum
    fusion (backward compatible default); "rrf" uses Reciprocal Rank Fusion,
    which is robust to score-scale differences across retrieval routes.
    """

    vector_weight: float = 0.5
    structured_weight: float = 0.3
    graph_weight: float = 0.4
    rule_weight: float = 0.2
    keyword_weight: float = 0.5
    min_confidence: float = 0.0
    top_k: int = 10
    enable_structured: bool = True
    enable_graph: bool = True
    enable_vector: bool = True
    enable_rules: bool = True
    enable_rewriting: bool = True
    fusion_mode: str = "rrf"  # P2-1 (2026-09-09): RRF-first — robust to
    # BM25-vs-cosine scale mismatch; "weighted" kept as opt-out fallback.
    rrf_k: int = 60  # RRF constant (standard default ~60)


@dataclass
class RetrievalResult:
    """检索结果（向后兼容）"""

    result_id: str
    content: str
    source: str
    source_type: str
    structured_score: float = 0.0
    vector_score: float = 0.0
    rule_score: float = 0.0
    graph_score: float = 0.0
    hybrid_score: float = 0.0
    confidence: float = 0.0
    evidence: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_id": self.result_id,
            "content": self.content,
            "source": self.source,
            "source_type": self.source_type,
            "structured_score": self.structured_score,
            "vector_score": self.vector_score,
            "rule_score": self.rule_score,
            "hybrid_score": self.hybrid_score,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


class HybridIndexManager:
    """混合索引管理器 - 统一管理倒排索引和向量索引

    Phase B hard constraint (2026-08-29): by default the manager points at the
    governed packet directory (``asset_packets/governed/<latest>/``).  Passing
    an explicit path pointing into the legacy ``outputs/runs/ontology/rag``
    directory is *disallowed* unless ``RAG_ALLOW_LEGACY_DIR=1`` is set (never
    in CI/CD).  ``RAG_GOVERNED_ONLY=0`` relaxes the path check entirely.
    """

    def __init__(
        self,
        rag_dir: Optional[Path | str] = None,
        embedder=None,
        vector_dimension: int = 384,
        structured_index=None,
        graph_index=None,
    ):
        from .governed_packets import resolve_governed_rag_dir

        if rag_dir is None:
            resolved = resolve_governed_rag_dir()
        else:
            resolved = resolve_governed_rag_dir(Path(rag_dir))
        self.rag_dir = resolved
        self.rag_dir.mkdir(parents=True, exist_ok=True)

        self.vector_manager = VectorIndexManager(
            rag_dir=self.rag_dir,
            embedder=embedder,
            dimension=vector_dimension,
        )

        self.inverted_index_path = self.rag_dir / "inverted_index.json"
        self.idf_path = self.rag_dir / "idf.json"
        self.chunks_path = self.rag_dir / "chunks.jsonl"

        self._inverted_index: Optional[Dict[str, List[str]]] = None
        self._idf: Optional[Dict[str, float]] = None
        self._chunks: Optional[List[ChunkRecord]] = None
        self._chunk_by_id: Optional[Dict[str, ChunkRecord]] = None
        self._structured_index = structured_index
        self._graph_index = graph_index

    def _load_inverted_index(self) -> Optional[Dict[str, List[str]]]:
        if not self.inverted_index_path.exists():
            return None
        try:
            data = json.loads(self.inverted_index_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, list)}
        except Exception:
            pass
        return None

    def _load_idf(self) -> Optional[Dict[str, float]]:
        if not self.idf_path.exists():
            return None
        try:
            data = json.loads(self.idf_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
        except Exception:
            pass
        return None

    def _load_chunks(self) -> Optional[List[ChunkRecord]]:
        if not self.chunks_path.exists():
            return None
        try:
            return load_chunks_jsonl(self.chunks_path)
        except Exception:
            return None

    def _save_chunks(self, chunks: Sequence[ChunkRecord]):
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    def build_hybrid_index(
        self,
        chunks: Sequence[ChunkRecord],
        force_rebuild: bool = False,
    ) -> HybridIndexArtifacts:
        if not force_rebuild and self._index_exists():
            return self._incremental_from_existing(chunks)
        return self._full_rebuild(chunks)

    def _index_exists(self) -> bool:
        return (
            self.inverted_index_path.exists()
            and self.vector_manager.metadata_path.exists()
        )

    def _full_rebuild(self, chunks: Sequence[ChunkRecord]) -> HybridIndexArtifacts:
        self._save_chunks(chunks)

        inverted_index = build_inverted_index(chunks)
        write_inverted_index(inverted_index, self.inverted_index_path)

        idf = compute_idf(inverted_index, len(chunks))
        write_idf(idf, self.idf_path)

        vectors, chunk_ids = self.vector_manager.build_index(chunks, force_rebuild=True)

        self._inverted_index = inverted_index
        self._idf = idf
        self._chunks = list(chunks)
        self._chunk_by_id = {c.chunk_id: c for c in chunks if c.chunk_id}

        return HybridIndexArtifacts(
            chunk_count=len(chunks),
            token_count=len(inverted_index),
            vector_dimension=vectors.shape[1] if len(vectors) > 0 else 0,
            inverted_index_path=str(self.inverted_index_path),
            idf_path=str(self.idf_path),
            vector_index_path=str(self.vector_manager.vectors_path),
            metadata_path=str(self.vector_manager.metadata_path),
        )

    def _incremental_from_existing(
        self,
        chunks: Sequence[ChunkRecord],
    ) -> HybridIndexArtifacts:
        existing_chunks = self._chunks or self._load_chunks()
        if not existing_chunks:
            return self._full_rebuild(chunks)

        existing_chunk_ids = {c.chunk_id for c in existing_chunks}
        new_chunk_ids = {c.chunk_id for c in chunks if c.chunk_id}

        deleted_ids = list(existing_chunk_ids - new_chunk_ids)
        added_chunks = [c for c in chunks if c.chunk_id and c.chunk_id not in existing_chunk_ids]

        existing_by_id = {c.chunk_id: c for c in existing_chunks if c.chunk_id}
        modified_chunks = []
        for c in chunks:
            if c.chunk_id and c.chunk_id in existing_by_id:
                if c.content != existing_by_id[c.chunk_id].content:
                    modified_chunks.append(c)

        return self.incremental_update(
            new_chunks=added_chunks,
            modified_chunks=modified_chunks,
            deleted_chunk_ids=deleted_ids,
        )

    def incremental_update(
        self,
        new_chunks: List[ChunkRecord],
        modified_chunks: List[ChunkRecord],
        deleted_chunk_ids: List[str],
    ) -> HybridIndexArtifacts:
        existing_chunks = self._load_chunks() or []
        existing_by_id = {c.chunk_id: c for c in existing_chunks if c.chunk_id}

        final_chunks: Dict[str, ChunkRecord] = {}
        for c in existing_chunks:
            if c.chunk_id and c.chunk_id not in deleted_chunk_ids:
                final_chunks[c.chunk_id] = c

        for c in modified_chunks:
            if c.chunk_id:
                final_chunks[c.chunk_id] = c

        for c in new_chunks:
            if c.chunk_id:
                final_chunks[c.chunk_id] = c

        chunk_list = list(final_chunks.values())

        vectors, chunk_ids = self.vector_manager.incremental_update(
            new_chunks=new_chunks,
            modified_chunks=modified_chunks,
            deleted_chunk_ids=deleted_chunk_ids,
        )

        existing_index = self._load_inverted_index() or {}
        inverted_index = incremental_update_inverted_index(
            existing_index=existing_index,
            new_chunks=new_chunks,
            modified_chunks=modified_chunks,
            deleted_chunk_ids=deleted_chunk_ids,
        )
        write_inverted_index(inverted_index, self.inverted_index_path)

        idf = compute_idf(inverted_index, len(chunk_list))
        write_idf(idf, self.idf_path)

        self._save_chunks(chunk_list)

        self._inverted_index = inverted_index
        self._idf = idf
        self._chunks = chunk_list
        self._chunk_by_id = final_chunks

        return HybridIndexArtifacts(
            chunk_count=len(chunk_list),
            token_count=len(inverted_index),
            vector_dimension=vectors.shape[1] if len(vectors) > 0 else 0,
            inverted_index_path=str(self.inverted_index_path),
            idf_path=str(self.idf_path),
            vector_index_path=str(self.vector_manager.vectors_path),
            metadata_path=str(self.vector_manager.metadata_path),
        )

    def _build_chunk_tf(self, chunks: Sequence[ChunkRecord]) -> Dict[str, Dict[str, float]]:
        chunk_tf: Dict[str, Dict[str, float]] = {}
        for c in chunks:
            # [RAG-OPT3] tokenize 后过一遍停用词表，消除"的/了/the/of"等噪音 token。
            raw_tokens = tokenize(c.content)
            tokens = [t for t in raw_tokens if t.lower() not in _BM25_STOPWORDS]
            n = max(len(tokens), 1)
            tf: Dict[str, float] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0.0) + 1.0
            chunk_tf[c.chunk_id] = {t: cnt / n for t, cnt in tf.items()}
        return chunk_tf

    def _compute_bm25_scores(
        self,
        query: str,
        chunks: Sequence[ChunkRecord],
        inverted_index: Dict[str, List[str]],
        idf: Dict[str, float],
    ) -> Dict[str, float]:
        chunk_tf = self._build_chunk_tf(chunks)

        scores: Dict[str, float] = {}

        # [RAG-OPT3] 查询侧同样做停用词过滤，避免用户问"什么是MBOM的流程"时"的"拉低打分。
        raw_query_tokens = tokenize(query)
        query_tokens = [t for t in raw_query_tokens if t.lower() not in _BM25_STOPWORDS]

        for tok in query_tokens:
            tok_idf = idf.get(tok, 1.0) if idf else 1.0
            for cid in inverted_index.get(tok, []):
                tf_norm = chunk_tf.get(cid, {}).get(tok, 0.0)
                scores[cid] = scores.get(cid, 0.0) + tok_idf * tf_norm

        # ---------------------------------------------------------------------
        # [RAG-OPT3] parent_id 反向链路增强（chunk 级 parent_id 反向链路）：
        # 如果某个 chunk 命中了查询，并且它的 meta.parent_id 指向另一个 chunk，
        # 则给 parent chunk 也加一点折扣后的分数（类似 PageRank 的反向链接加成）。
        # 这样当用户查询一个 L2 业务实体时，其所属 L1/L0 父类 chunk 也能获得
        # 一点曝光，帮助 LLM 形成完整的层级理解。
        # ---------------------------------------------------------------------
        if scores and chunks:
            # 构造 chunk_id -> ChunkRecord 的快速索引
            by_id: Dict[str, ChunkRecord] = {c.chunk_id: c for c in chunks if c.chunk_id}
            # 构造 parent_id -> [children_chunk_ids...] 的反向索引
            children_of: Dict[str, List[str]] = {}
            for c in chunks:
                meta = getattr(c, "meta", None) or {}
                pid = (
                    meta.get("parent_id")
                    or meta.get("parent")
                    or meta.get("parent_meta_id")
                    or ""
                )
                if isinstance(pid, str) and pid and pid in by_id and pid != c.chunk_id:
                    children_of.setdefault(pid, []).append(c.chunk_id)

            parent_boost: Dict[str, float] = {}
            for child_id, child_score in scores.items():
                if child_score <= 0:
                    continue
                child_chunk = by_id.get(child_id)
                if child_chunk is None:
                    continue
                # 上溯最多 2 层（L3→L2→L1，或 L2→L1→L0）
                current_id = child_id
                depth_bonus = 1.0
                for _ in range(2):
                    meta = getattr(by_id.get(current_id, object()), "meta", None) or {}
                    pid = (
                        meta.get("parent_id")
                        or meta.get("parent")
                        or meta.get("parent_meta_id")
                        or ""
                    )
                    if not pid or pid not in by_id or pid == current_id:
                        break
                    # 父节点本身得分已经高时不必再加太多（抑制长尾爆炸）
                    existing = scores.get(pid, 0.0)
                    if existing >= 2.0:
                        break
                    depth_bonus *= 0.6  # 每上溯一层打 0.6 折
                    bonus = child_score * 0.25 * depth_bonus
                    parent_boost[pid] = parent_boost.get(pid, 0.0) + bonus
                    current_id = pid

            # 子节点众数加成：若某 parent 有 ≥2 个孩子 chunk 命中，则额外 +5%*N
            for pid, children in children_of.items():
                hit_children = [cid for cid in children if scores.get(cid, 0.0) > 0]
                if len(hit_children) >= 2:
                    parent_boost[pid] = parent_boost.get(pid, 0.0) + 0.05 * len(hit_children)

            for pid, bonus in parent_boost.items():
                scores[pid] = scores.get(pid, 0.0) + bonus

        return scores

    def _compute_vector_scores(
        self,
        query: str,
        chunks: Sequence[ChunkRecord],
        top_k: int = 50,
    ) -> Dict[str, float]:
        vectors = self.vector_manager.get_vectors()
        vector_chunk_ids = self.vector_manager.get_chunk_ids()

        if vectors is None or len(vectors) == 0:
            return {}

        query_vector = self.vector_manager.embedder.embed([query])[0]
        similarities = self.vector_manager.embedder.cosine_similarity(query_vector, vectors)
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return {vector_chunk_ids[idx]: float(similarities[idx]) for idx in top_indices}

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        max_score = max(scores.values()) if scores else 1.0
        if max_score == 0:
            return {k: 0.0 for k in scores}
        return {k: v / max_score for k, v in scores.items()}

    def _rank_hybrid_scores(
        self,
        norm_keyword: Dict[str, float],
        norm_vector: Dict[str, float],
        keyword_weight: float,
        vector_weight: float,
    ) -> List[Tuple[str, float]]:
        all_chunk_ids = set(norm_keyword.keys()) | set(norm_vector.keys())
        hybrid_scores: Dict[str, float] = {}

        for cid in all_chunk_ids:
            k_score = norm_keyword.get(cid, 0.0)
            v_score = norm_vector.get(cid, 0.0)
            hybrid_scores[cid] = keyword_weight * k_score + vector_weight * v_score

        return sorted(hybrid_scores.items(), key=lambda x: (-x[1], x[0]))

    # P2-1 (2026-09-09): channel-routed RRF-first hybrid search.
    #  1. route_channels decides which channels deserve budget for THIS query
    #     (lexical always on; vector off for pure lookups; graph/structured
    #     per intent) — replaces the old "always fuse both, weighted-sum".
    #  2. per-channel raw scores are max-normalized, then fused with RRF over
    #     channel ranks (scale-robust; weights kept as opt-out fallback).
    #  3. fixed tail: rerank (hybrid_score desc, chunk_id tiebreak) + citation
    #     payload via the shared policy builder (build_citation contract).
    def hybrid_search(
        self,
        query: str,
        topk: int = 10,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        vector_top_k: int = 50,
        *, 
        fusion_mode: Optional[str] = None,
        rrf_k: int = 60,
        route: Optional[FrozenSet[str]] = None,
        structured_weight: float = 0.3,
        graph_weight: float = 0.4,
        enable_structured: bool = True,
        enable_graph: bool = True,
    ) -> List[HybridSearchResult]:
        if self._chunks is None:
            self._chunks = self._load_chunks() or []
        if self._inverted_index is None:
            self._inverted_index = self._load_inverted_index() or {}
        if self._idf is None:
            self._idf = self._load_idf() or {}
        if self._chunk_by_id is None:
            self._chunk_by_id = {c.chunk_id: c for c in self._chunks if c.chunk_id}

        if not self._chunks:
            return []

        channels = route or self._route_query(query)
        # 2026-10-16: 路由通道名是 "lexical"（query_channel_router），但本函数
        # 内部一律用 "keyword" 作 BM25 通道键。此前 "keyword" 永不 ∈ channels，
        # 关键字通道被静默丢弃（纯 lookup 查询直接零命中）。归一化一次。
        channels = frozenset("keyword" if c == "lexical" else c for c in channels)
        keyword_scores = self._compute_bm25_scores(
            query, self._chunks, self._inverted_index, self._idf
        )
        vector_scores: Dict[str, float] = {}
        structured_scores: Dict[str, float] = {}
        graph_scores: Dict[str, float] = {}
        
        if "vector" in channels:
            vector_scores = self._compute_vector_scores(query, self._chunks, vector_top_k)
        
        # Handle structured retrieval if enabled and channel is available
        structured_results = []
        if enable_structured and "structured" in channels and hasattr(self, '_structured_index'):
            try:
                from .structured_index import StructuredIndex
                if isinstance(self._structured_index, StructuredIndex):
                    structured_results = self._structured_search(query, topk=max(5, topk // 2))
                    # Convert structured results to chunk-like scores for fusion
                    for result in structured_results:
                        if hasattr(result, 'chunk_id'):
                            structured_scores[result.chunk_id] = result.score * structured_weight
            except Exception as exc:
                logger.warning("Structured retrieval failed (ignored): %s", exc)
        
        # Handle graph retrieval if enabled and channel is available
        graph_results = []
        if enable_graph and "graph" in channels and hasattr(self, '_graph_index'):
            try:
                # Implement graph search if graph index is available
                if hasattr(self._graph_index, 'search'):
                    graph_results = self._graph_index.search(query, topk=max(5, topk // 2))
                    for result in graph_results:
                        if hasattr(result, 'chunk_id'):
                            graph_scores[result.chunk_id] = result.score * graph_weight
            except Exception as exc:
                logger.warning("Graph retrieval failed (ignored): %s", exc)

        mode = fusion_mode or "rrf"
        if mode == "rrf":
            # Collect all relevant scores for RRF fusion
            all_scores = {
                "keyword": keyword_scores,
                "vector": vector_scores,
                "structured": structured_scores,
                "graph": graph_scores
            }
            
            # Filter out empty channels
            active_channels = {k: v for k, v in all_scores.items() if v and k in channels}
            
            ranked = self._rank_rrf_multiple_channels(active_channels, rrf_k=rrf_k)
            
            # Normalize individual channel scores
            norm_keyword = self._normalize_scores(keyword_scores)
            norm_vector = self._normalize_scores(vector_scores)
            norm_structured = self._normalize_scores(structured_scores)
            norm_graph = self._normalize_scores(graph_scores)
        else:  # opt-out: legacy weighted sum over max-normalized scores
            norm_keyword = self._normalize_scores(keyword_scores)
            norm_vector = self._normalize_scores(vector_scores)
            norm_structured = self._normalize_scores(structured_scores)
            norm_graph = self._normalize_scores(graph_scores)
            
            # Calculate hybrid scores
            hybrid_scores: Dict[str, float] = {}
            all_chunk_ids = set(norm_keyword.keys()) | set(norm_vector.keys()) | set(norm_structured.keys()) | set(norm_graph.keys())
            
            for cid in all_chunk_ids:
                score = 0.0
                if "keyword" in channels and cid in norm_keyword:
                    score += keyword_weight * norm_keyword[cid]
                if "vector" in channels and cid in norm_vector:
                    score += vector_weight * norm_vector[cid]
                if "structured" in channels and cid in norm_structured:
                    score += structured_weight * norm_structured[cid]
                if "graph" in channels and cid in norm_graph:
                    score += graph_weight * norm_graph[cid]
                hybrid_scores[cid] = score
            
            ranked = sorted(hybrid_scores.items(), key=lambda x: (-x[1], x[0]))

        results: List[HybridSearchResult] = []
        for cid, hybrid_score in ranked[:topk]:
            chunk = self._chunk_by_id.get(cid)
            if chunk:
                results.append(
                    HybridSearchResult(
                        chunk=chunk,
                        keyword_score=norm_keyword.get(cid, 0.0),
                        vector_score=norm_vector.get(cid, 0.0),
                        structured_score=norm_structured.get(cid, 0.0),
                        graph_score=norm_graph.get(cid, 0.0),
                        hybrid_score=hybrid_score,
                    )
                )

        return self._rerank_with_citations(results)

    @staticmethod
    def _route_query(query: str) -> FrozenSet[str]:
        """Best-effort channel routing without a query_core dependency.

        hybrid_rag must not import domain.query_core at module load
        (import-boundary direction); the router module itself is
        dependency-free, so a local import here is safe and lazy.
        Falls back to both-channels on any parse failure.
        """
        try:
            from .query_channel_router import ALL_CHANNELS, route_channels
            from .domain.query_core import QueryClassifier

            cls = QueryClassifier().classify(query)
            qtype = getattr(cls.query_type, "value", str(cls.query_type))
            intent = getattr(cls.intent, "value", str(cls.intent))
            decision = route_channels(
                query_type=qtype,
                intent=intent,
                is_multi_entity=False,
                entity_count=len(cls.entities or []),
            )
            # lexical+vector are the only channels hybrid_search fuses;
            # graph/structured decisions are returned for the caller/eval log.
            fused = {c for c in decision.channels if c in ("lexical", "vector")}
            return frozenset(fused) or frozenset({"lexical", "vector"})
        except Exception:
            return frozenset({"lexical", "vector"})

    @staticmethod
    def _rank_rrf(
        keyword_scores: Dict[str, float],
        vector_scores: Dict[str, float],
        *,
        rrf_k: int = 60,
    ) -> List[Tuple[str, float]]:
        """RRF over per-channel ranks (same contract as _fuse_results_rrf)."""
        k = max(1, int(rrf_k or 60))
        rank_kw = {cid: i for i, (cid, _) in enumerate(
            sorted(keyword_scores.items(), key=lambda x: (-x[1], x[0])))} 
        rank_v = {cid: i for i, (cid, _) in enumerate(
            sorted(vector_scores.items(), key=lambda x: (-x[1], x[0])))} 
        fused: Dict[str, float] = {}
        for cid in set(rank_kw) | set(rank_v):
            score = 0.0
            if cid in rank_kw:
                score += 1.0 / (k + rank_kw[cid])
            if cid in rank_v:
                score += 1.0 / (k + rank_v[cid])
            fused[cid] = score
        return sorted(fused.items(), key=lambda x: (-x[1], x[0]))
    
    @staticmethod
    def _rank_rrf_multiple_channels(
        channel_scores: Dict[str, Dict[str, float]],
        *,
        rrf_k: int = 60,
    ) -> List[Tuple[str, float]]:
        """RRF over multiple channels' scores."""
        k = max(1, int(rrf_k or 60))
        ranks: Dict[str, Dict[str, int]] = {}
        all_cids: set[str] = set()
        
        # Calculate ranks for each channel
        for channel_name, scores in channel_scores.items():
            if not scores:
                continue
            # Sort by score descending and assign ranks
            sorted_items = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
            ranks[channel_name] = {cid: i for i, (cid, _) in enumerate(sorted_items)}
            all_cids.update(scores.keys())
        
        # Calculate RRF scores
        fused: Dict[str, float] = {}
        for cid in all_cids:
            score = 0.0
            for channel_name, channel_ranks in ranks.items():
                if cid in channel_ranks:
                    score += 1.0 / (k + channel_ranks[cid])
            fused[cid] = score
        
        return sorted(fused.items(), key=lambda x: (-x[1], x[0]))

    @staticmethod
    def _rerank_with_citations(
        results: List[HybridSearchResult],
    ) -> List[HybridSearchResult]:
        """Fixed tail: deterministic rerank + citation metadata attach.

        Rerank is hybrid_score desc with chunk_id tiebreak (stable across
        runs). Each result gains ``citation`` metadata via the shared
        build_citation policy shape — chunk payloads keep flowing to the
        answerer unchanged; the citation dict rides in metadata.
        """
        from .policy.fallback import build_citation

        ordered = sorted(
            results,
            key=lambda r: (-r.hybrid_score, r.chunk.chunk_id if r.chunk else ""),
        )
        for r in ordered:
            chunk = r.chunk
            if chunk is None:
                continue
            meta = getattr(chunk, "meta", None) or {}
            citation = build_citation(
                chunk_id=getattr(chunk, "chunk_id", "unknown"),
                chunk_type=getattr(chunk, "chunk_type", "unknown"),
                source=meta.get("source", "unknown"),
                score=float(r.hybrid_score),
                content_preview=(chunk.content or "")[:200],
            )
            try:
                meta = dict(meta)
                meta["citation"] = citation
                object.__setattr__(chunk, "meta", meta)
            except Exception:
                pass
        return ordered

    def keyword_search(
        self,
        query: str,
        topk: int = 10,
    ) -> List[HybridSearchResult]:
        if self._chunks is None:
            self._chunks = self._load_chunks() or []
        if self._inverted_index is None:
            self._inverted_index = self._load_inverted_index() or {}
        if self._idf is None:
            self._idf = self._load_idf() or {}
        if self._chunk_by_id is None:
            self._chunk_by_id = {c.chunk_id: c for c in self._chunks if c.chunk_id}

        scores = self._compute_bm25_scores(
            query, self._chunks, self._inverted_index, self._idf
        )

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

        results: List[HybridSearchResult] = []
        for cid, score in ranked[:topk]:
            chunk = self._chunk_by_id.get(cid)
            if chunk:
                results.append(
                    HybridSearchResult(
                        chunk=chunk,
                        keyword_score=score,
                        vector_score=0.0,
                        hybrid_score=score,
                    )
                )

        return results

    def vector_search(
        self,
        query: str,
        topk: int = 10,
    ) -> List[HybridSearchResult]:
        if self._chunks is None:
            self._chunks = self._load_chunks() or []
        if self._chunk_by_id is None:
            self._chunk_by_id = {c.chunk_id: c for c in self._chunks if c.chunk_id}

        scores = self._compute_vector_scores(query, self._chunks, topk)

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

        results: List[HybridSearchResult] = []
        for cid, score in ranked[:topk]:
            chunk = self._chunk_by_id.get(cid)
            if chunk:
                results.append(
                    HybridSearchResult(
                        chunk=chunk,
                        keyword_score=0.0,
                        vector_score=score,
                        structured_score=0.0,
                        graph_score=0.0,
                        hybrid_score=score,
                    )
                )

        return results
    
    def _structured_search(
        self,
        query: str,
        topk: int = 10,
    ) -> List[HybridSearchResult]:
        """Search structured entities/relations/evidences"""
        if not self._structured_index:
            return []
            
        try:
            from .structured_index import StructuredIndex
            if not isinstance(self._structured_index, StructuredIndex):
                return []
                
            entities = self._structured_index.search_entities(
                query=query,
                limit=topk
            )
            
            results = []
            for entity in entities:
                if hasattr(entity, 'entity_id') and hasattr(entity, 'name'):
                    content = f"{entity.name}: {getattr(entity, 'description', '')}"
                    chunk_id = getattr(entity, 'entity_id', '')
                    score = getattr(entity, 'confidence', 0.5)

                    # Create a dummy chunk record for structured results
                    dummy_chunk = ChunkRecord(
                        chunk_id=chunk_id,
                        content=content,
                        chunk_type="structured-entity",
                        source_file="structured-index",
                    meta={
                            "entity_type": getattr(entity, 'entity_type', ''),
                            "layer": getattr(entity, 'layer', ''),
                            "confidence": score
                    }
                    )

                    results.append(HybridSearchResult(
                        chunk=dummy_chunk,
                        keyword_score=0.0,
                        vector_score=0.0,
                        structured_score=score,
                        graph_score=0.0,
                        hybrid_score=score,
                    ))

            return results
        except Exception as exc:
            logger.warning("Structured search failed: %s", exc)
            return []

    def list_versions(self) -> List[IndexVersionInfo]:
        return self.vector_manager.list_versions()

    def create_version(self, description: str = "") -> str:
        return self.vector_manager.create_version(description)

    def switch_version(self, version_id: str) -> bool:
        return self.vector_manager.switch_version(version_id)

    def rollback_to_version(self, version_id: str) -> bool:
        return self.vector_manager.rollback_to_version(version_id)


class HybridRetriever:
    """混合检索器 - 融合结构化检索、向量检索和规则检索"""

    def __init__(
        self,
        structured_index: StructuredIndex,
        graph_index=None,
        query_rewriter: Optional[BaseQueryRewriter] = None,
        vector_search_fn: Optional[callable] = None,
        rule_search_fn: Optional[callable] = None,
    ) -> None:
        self._structured_index = structured_index
        self._graph_index = graph_index
        self._query_rewriter = query_rewriter
        self._vector_search_fn = vector_search_fn
        self._rule_search_fn = rule_search_fn

    def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        config = config or RetrievalConfig()

        rewritten_query = query
        if config.enable_rewriting and self._query_rewriter:
            rewritten = self._query_rewriter.rewrite(query)
            rewritten_query = rewritten.rewritten if hasattr(rewritten, "rewritten") else rewritten

        structured_results = []
        vector_results = []
        rule_results = []
        graph_results = []

        if config.enable_structured:
            structured_results = self._structured_search(rewritten_query, config)

        if config.enable_vector and self._vector_search_fn:
            vector_results = self._vector_search(rewritten_query, config)
            
        if config.enable_graph and hasattr(self, '_graph_index') and self._graph_index:
            try:
                graph_results = self._graph_search(rewritten_query, config)
            except Exception as exc:
                logger.warning("Graph search failed: %s", exc)

        if config.enable_rules and self._rule_search_fn:
            rule_results = self._rule_search(rewritten_query, config)

        fused_results = self._fuse_results(
            structured_results,
            vector_results,
            rule_results,
            graph_results,
            config,
        )

        ranked_results = self._rerank_results(fused_results, config)

        filtered_results = [
            r for r in ranked_results if r.confidence >= config.min_confidence
        ]

        return filtered_results[: config.top_k]

    def _structured_search(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> List[RetrievalResult]:
        results = []

        entities = self._structured_index.search_entities(
            query=query,
            min_confidence=config.min_confidence,
            limit=config.top_k,
        )

        for entity in entities:
            result = RetrievalResult(
                result_id=f"entity-{entity.entity_id}",
                content=f"{entity.name}: {entity.description}" if hasattr(entity, "description") else entity.name,
                source="structured_index",
                source_type="entity",
                structured_score=entity.confidence,
                confidence=entity.confidence,
                metadata={
                    "entity_id": entity.entity_id,
                    "entity_type": entity.entity_type,
                    "layer": entity.layer,
                    "attributes": entity.attributes,
                },
            )
            results.append(result)

        relations = self._structured_index.search_relations(
            query=query,
            min_confidence=config.min_confidence,
            limit=config.top_k,
        )

        for relation, from_name, to_name in relations:
            result = RetrievalResult(
                result_id=f"relation-{relation.relation_id}",
                content=f"{from_name} -> {relation.relation_type} -> {to_name}",
                source="structured_index",
                source_type="relation",
                structured_score=relation.confidence,
                confidence=relation.confidence,
                metadata={
                    "relation_id": relation.relation_id,
                    "from_entity": relation.from_entity,
                    "to_entity": relation.to_entity,
                    "relation_type": relation.relation_type,
                },
            )
            results.append(result)

        evidences = self._structured_index.search_evidences(
            query=query,
            min_confidence=config.min_confidence,
            limit=config.top_k,
        )

        for evidence in evidences:
            result = RetrievalResult(
                result_id=f"evidence-{evidence.evidence_id}",
                content=evidence.content,
                source=evidence.source,
                source_type="evidence",
                structured_score=evidence.confidence,
                confidence=evidence.confidence,
                evidence=[evidence.content],
                metadata={
                    "evidence_id": evidence.evidence_id,
                    "entity_id": evidence.entity_id,
                    "source": evidence.source,
                },
            )
            results.append(result)

        return results

    def _vector_search(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> List[RetrievalResult]:
        if not self._vector_search_fn:
            return []

        try:
            vector_results = self._vector_search_fn(query, top_k=config.top_k)

            results = []
            for i, item in enumerate(vector_results):
                if isinstance(item, dict):
                    result = RetrievalResult(
                        result_id=f"vector-{i}",
                        content=item.get("content", ""),
                        source=item.get("source", "vector_index"),
                        source_type="vector",
                        vector_score=item.get("score", 0.0),
                        confidence=item.get("score", 0.0),
                        metadata=item.get("metadata", {}),
                    )
                    results.append(result)

            return results
        except Exception:
            return []

    def _rule_search(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> List[RetrievalResult]:
        if not self._rule_search_fn:
            return []

        try:
            rule_results = self._rule_search_fn(query, top_k=config.top_k)

            results = []
            for i, item in enumerate(rule_results):
                if isinstance(item, dict):
                    result = RetrievalResult(
                        result_id=f"rule-{i}",
                        content=item.get("content", ""),
                        source=item.get("source", "rule_engine"),
                        source_type="rule",
                        rule_score=item.get("score", 0.0),
                        confidence=item.get("score", 0.0),
                        metadata=item.get("metadata", {}),
                    )
                    results.append(result)

            return results
        except Exception:
            return []

    def _graph_search(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> List[RetrievalResult]:
        """Search graph entities/relations/evidences"""
        if not self._graph_index:
            return []
            
        try:
            results = []
            
            # Assume graph search returns entities, relations, and evidences
            if hasattr(self._graph_index, 'search_entities'):
                entities = self._graph_index.search_entities(
                    query=query,
                    min_confidence=config.min_confidence,
                    limit=config.top_k,
                )
                
                for entity in entities:
                    result = RetrievalResult(
                        result_id=f"graph-entity-{getattr(entity, 'entity_id', '')}",
                        content=f"{getattr(entity, 'name', '')}: {getattr(entity, 'description', '')}",
                        source="graph_index",
                        source_type="graph-entity",
                        graph_score=getattr(entity, 'confidence', 0.5),
                        confidence=getattr(entity, 'confidence', 0.5),
                        metadata={
                            "entity_id": getattr(entity, 'entity_id', ''),
                            "entity_type": getattr(entity, 'entity_type', ''),
                            "layer": getattr(entity, 'layer', ''),
                            "attributes": getattr(entity, 'attributes', {}),
                        },
                    )
                    results.append(result)
            
            if hasattr(self._graph_index, 'search_relations'):
                relations = self._graph_index.search_relations(
                    query=query,
                    min_confidence=config.min_confidence,
                    limit=config.top_k,
                )
                
                for relation, from_name, to_name in relations:
                    result = RetrievalResult(
                        result_id=f"graph-relation-{getattr(relation, 'relation_id', '')}",
                        content=f"{from_name} -> {getattr(relation, 'relation_type', '')} -> {to_name}",
                        source="graph_index",
                        source_type="graph-relation",
                        graph_score=getattr(relation, 'confidence', 0.5),
                        confidence=getattr(relation, 'confidence', 0.5),
                        metadata={
                            "relation_id": getattr(relation, 'relation_id', ''),
                            "from_entity": getattr(relation, 'from_entity', ''),
                            "to_entity": getattr(relation, 'to_entity', ''),
                            "relation_type": getattr(relation, 'relation_type', ''),
                            "properties": getattr(relation, 'properties', {}),
                        },
                    )
                    results.append(result)
            
            if hasattr(self._graph_index, 'search_evidences'):
                evidences = self._graph_index.search_evidences(
                    query=query,
                    min_confidence=config.min_confidence,
                    limit=config.top_k,
                )
                
                for evidence in evidences:
                    result = RetrievalResult(
                        result_id=f"graph-evidence-{getattr(evidence, 'evidence_id', '')}",
                        content=getattr(evidence, 'content', ''),
                        source=getattr(evidence, 'source', "graph_index"),
                        source_type="graph-evidence",
                        graph_score=getattr(evidence, 'confidence', 0.5),
                        confidence=getattr(evidence, 'confidence', 0.5),
                        evidence=[getattr(evidence, 'content', '')] if getattr(evidence, 'content', '') else [],
                        metadata={
                            "evidence_id": getattr(evidence, 'evidence_id', ''),
                            "entity_id": getattr(evidence, 'entity_id', ''),
                            "source": getattr(evidence, 'source', ""),
                        },
                    )
                    results.append(result)
            
            return results
        except Exception as exc:
            logger.warning("Graph search failed: %s", exc)
            return []

    def _fuse_results(
        self,
        structured_results: List[RetrievalResult],
        vector_results: List[RetrievalResult],
        rule_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        config: RetrievalConfig,
    ) -> List[RetrievalResult]:
        # G6 (2026-08-17): RRF mode is robust to score-scale differences across
        # routes (BM25 vs cosine); weighted mode preserved as default fallback.
        # 2026-10-16: 调用方传 5 路（含 graph_results），weighted 分支此前丢弃 graph。
        # graph 无权重字段，按 structured 通道权重折入（graph_score 已在 RetrievalResult 上）。
        if getattr(config, "fusion_mode", "weighted") == "rrf":
            return self._fuse_results_rrf(
                structured_results, vector_results, rule_results, graph_results, config
            )
        result_map: Dict[str, RetrievalResult] = {}

        for result in structured_results:
            result_map[result.result_id] = result

        for result in vector_results:
            if result.result_id in result_map:
                existing = result_map[result.result_id]
                existing.vector_score = result.vector_score
                existing.hybrid_score = (
                    existing.structured_score * config.structured_weight
                    + existing.vector_score * config.vector_weight
                )
            else:
                result.hybrid_score = result.vector_score * config.vector_weight
                result_map[result.result_id] = result

        for result in rule_results:
            if result.result_id in result_map:
                existing = result_map[result.result_id]
                existing.rule_score = result.rule_score
                existing.hybrid_score = (
                    existing.structured_score * config.structured_weight
                    + existing.vector_score * config.vector_weight
                    + existing.rule_score * config.rule_weight
                    + existing.graph_score * config.graph_weight
                )
            else:
                result.hybrid_score = result.rule_score * config.rule_weight
                result_map[result.result_id] = result

        for result in graph_results:
            if result.result_id in result_map:
                existing = result_map[result.result_id]
                existing.graph_score = result.graph_score
                existing.hybrid_score += existing.graph_score * config.graph_weight
            else:
                result.hybrid_score = result.graph_score * config.graph_weight
                result_map[result.result_id] = result

        for result in result_map.values():
            if result.hybrid_score == 0.0:
                result.hybrid_score = (
                    result.structured_score * config.structured_weight
                    + result.vector_score * config.vector_weight
                    + result.rule_score * config.rule_weight
                    + result.graph_score * config.graph_weight
                )
            result.confidence = result.hybrid_score

        return list(result_map.values())

    def _fuse_results_rrf(
        self,
        structured_results: List[RetrievalResult],
        vector_results: List[RetrievalResult],
        rule_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        config: RetrievalConfig,
    ) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion (RRF).

        Each route contributes 1 / (rrf_k + rank) to every result it returns;
        documents returned by multiple routes accumulate score and rise above
        single-route high scorers — the standard fix for cross-route score
        scale mismatch (BM25 vs cosine vs rule).
        """
        k = max(1, int(getattr(config, "rrf_k", 60) or 60))
        scores: Dict[str, float] = {}
        holders: Dict[str, RetrievalResult] = {}

        def merge_route(route: List[RetrievalResult]) -> None:
            for rank, result in enumerate(route):
                rid = result.result_id
                scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank)
                if rid not in holders:
                    holders[rid] = result

        merge_route(structured_results)
        merge_route(vector_results)
        merge_route(rule_results)
        merge_route(graph_results)

        merged = []
        for rid, result in holders.items():
            result.hybrid_score = scores[rid]
            result.confidence = scores[rid]
            merged.append(result)
        return merged

    def _rerank_results(
        self,
        results: List[RetrievalResult],
        config: RetrievalConfig,
    ) -> List[RetrievalResult]:
        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        return results

    def bind_evidence(
        self,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        for result in results:
            entity_id = result.metadata.get("entity_id")
            if entity_id:
                evidences = self._structured_index.get_entity_evidences(entity_id)
                result.evidence = [e.content for e in evidences]

        return results

    # ------------------------------------------------------------------
    # C-5: RAG evidence gate integration
    # ------------------------------------------------------------------

    def retrieve_with_gate(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
        *,
        query_terms: Optional[List[str]] = None,
        has_conflict: bool = False,
        max_retrieval_rounds: int = 3,
        evidence_gate=None,
    ) -> "GateRetrievalResult":
        """Retrieve with evidence-gate decision loop (C-5).

        Runs :meth:`retrieve` up to ``max_retrieval_rounds`` times. After each
        round, :class:`EvidenceGate` evaluates the accumulated evidence and
        decides whether to ANSWER, RETRIEVE_MORE, or ESCALATE.

        - ``ANSWER``: return the accumulated results immediately.
        - ``RETRIEVE_MORE``: bump ``top_k`` by 50% and retrieve again; merge
          new results (dedup by ``result_id``).
        - ``ESCALATE``: stop retrieving; caller should escalate to a stronger
          model or human review.

        Args:
            query: the user query string.
            config: optional base retrieval config (``top_k`` is expanded per
                round when the gate requests more evidence).
            query_terms: key terms extracted from the query, used for coverage
                scoring. When omitted, coverage defaults to 1.0 (no artificial
                demand for more retrieval).
            has_conflict: explicit conflict flag from upstream (e.g. entity
                resolution disagreement or contradiction detector).
            max_retrieval_rounds: cap on retrieval iterations.
            evidence_gate: optional injected :class:`EvidenceGate` (defaults to
                the process-wide singleton).
        """
        from src.application.rag.evidence_gate import (
            EvidenceGate,
            EvidenceGateAction,
            EvidenceItem,
            get_evidence_gate,
        )

        gate = evidence_gate or get_evidence_gate()
        config = config or RetrievalConfig()
        accumulated: Dict[str, RetrievalResult] = {}
        decision = None
        current_config = config

        for round_num in range(1, max_retrieval_rounds + 1):
            results = self.retrieve(query, current_config)
            for r in results:
                if r.result_id not in accumulated:
                    accumulated[r.result_id] = r

            evidence_items = [
                EvidenceItem(
                    text=r.content,
                    score=max(0.0, min(1.0, r.confidence)),
                    source_id=r.source or r.source_type or r.result_id,
                )
                for r in accumulated.values()
            ]
            decision = gate.evaluate(
                evidence_items,
                query_terms=query_terms or [],
                has_conflict=has_conflict,
                retrieval_round=round_num,
                max_retrieval_rounds=max_retrieval_rounds,
            )

            if decision.action == EvidenceGateAction.ANSWER:
                break
            if decision.action == EvidenceGateAction.ESCALATE:
                break
            # RETRIEVE_MORE: expand top_k by 50% for the next round.
            expanded_top_k = max(current_config.top_k, int(current_config.top_k * 1.5) + 1)
            current_config = RetrievalConfig(
                vector_weight=config.vector_weight,
                structured_weight=config.structured_weight,
                rule_weight=config.rule_weight,
                keyword_weight=config.keyword_weight,
                min_confidence=config.min_confidence,
                top_k=expanded_top_k,
                enable_structured=config.enable_structured,
                enable_vector=config.enable_vector,
                enable_rules=config.enable_rules,
                enable_rewriting=config.enable_rewriting,
                fusion_mode=config.fusion_mode,
                rrf_k=config.rrf_k,
            )

        return GateRetrievalResult(
            results=list(accumulated.values()),
            decision=decision,
            rounds_used=round_num,
        )


@dataclass
class GateRetrievalResult:
    """Result of :meth:`HybridRetriever.retrieve_with_gate` (C-5).

    Attributes:
        results: accumulated retrieval results (deduplicated by result_id).
        decision: the final :class:`EvidenceGateDecision` (ANSWER / RETRIEVE_MORE
            / ESCALATE) with computed signals for telemetry.
        rounds_used: number of retrieval rounds actually executed.
    """

    results: List[RetrievalResult]
    decision: Any  # EvidenceGateDecision (typed loosely to avoid import cycle)
    rounds_used: int


@dataclass
class QueryAnalysis:
    """查询分析结果"""

    original_query: str
    rewritten_query: str
    query_type: str
    entities: List[str]
    keywords: List[str]
    expansion_terms: List[str]


@dataclass
class RetrievalEvidenceBinding:
    """检索结果的证据绑定信息（内部使用）"""

    chunk_id: str
    source_file: str
    confidence: float
    relevance_score: float
    matched_entities: List[str]
    extraction_layer: str = ""
    timestamp: str = ""


class RuleBasedRetrieval:
    """规则检索辅助器 - 基于业务规则过滤和增强结果"""

    def __init__(self):
        self.rules = []

    def add_rule(self, name: str, condition_func, boost_factor: float = 1.0):
        self.rules.append({"name": name, "condition": condition_func, "boost": boost_factor})

    def apply_rules(
        self, results: List[HybridSearchResult], query_analysis: QueryAnalysis
    ) -> List[HybridSearchResult]:
        enhanced = []
        for result in results:
            score_boost = 1.0
            for rule in self.rules:
                if rule["condition"](result, query_analysis):
                    score_boost *= rule["boost"]

            enhanced.append(
                HybridSearchResult(
                    chunk=result.chunk,
                    keyword_score=result.keyword_score,
                    vector_score=result.vector_score,
                    hybrid_score=result.hybrid_score * score_boost,
                )
            )

        return sorted(enhanced, key=lambda x: x.hybrid_score, reverse=True)

    def build_default_rules(self):
        def entity_match_rule(result, analysis):
            chunk_text = result.chunk.content.lower()
            return any(entity.lower() in chunk_text for entity in analysis.entities)

        def definition_rule(result, analysis):
            if analysis.query_type == "fact":
                chunk_text = result.chunk.content.lower()
                return any(
                    term in chunk_text
                    for term in ["define", "definition", "means", "is a", "refers to"]
                )
            return False

        def procedure_rule(result, analysis):
            if analysis.query_type == "procedure":
                chunk_text = result.chunk.content.lower()
                return any(
                    term in chunk_text
                    for term in ["step", "process", "first", "second", "then", "finally"]
                )
            return False

        self.add_rule("entity_match", entity_match_rule, boost_factor=1.3)
        self.add_rule("definition_priority", definition_rule, boost_factor=1.2)
        self.add_rule("procedure_priority", procedure_rule, boost_factor=1.2)


class EnhancedHybridIndexManager(HybridIndexManager):
    """增强版混合索引管理器 - 集成查询改写和规则检索"""

    def __init__(
        self,
        rag_dir: Path | str,
        embedder=None,
        vector_dimension: int = 384,
    ):
        super().__init__(rag_dir=rag_dir, embedder=embedder, vector_dimension=vector_dimension)
        self.query_rewriter = BaseQueryRewriter()
        self.rule_engine = RuleBasedRetrieval()
        self.rule_engine.build_default_rules()

    def analyze_and_search(
        self,
        query: str,
        topk: int = 10,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        use_rules: bool = True,
        use_rewrite: bool = True,
    ) -> Dict[str, Any]:
        analysis = self._analyze_query(query)

        search_query = analysis.rewritten_query if use_rewrite else query

        results = self.hybrid_search(
            query=search_query,
            topk=topk,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
        )

        if use_rules:
            results = self.rule_engine.apply_rules(results, analysis)

        evidence_bindings = self._build_evidence_bindings(results, analysis)

        return {
            "original_query": query,
            "rewritten_query": search_query,
            "query_analysis": {
                "query_type": analysis.query_type,
                "entities": analysis.entities,
                "keywords": analysis.keywords,
                "expansion_terms": analysis.expansion_terms,
            },
            "results": [r.to_dict() for r in results],
            "evidence": [asdict(e) for e in evidence_bindings],
        }

    def _analyze_query(self, query: str) -> QueryAnalysis:
        """分析查询并返回结构化结果"""
        query_lower = query.lower()

        query_type = "fact"
        QUERY_TYPE_PATTERNS = {
            "fact": ["what is", "what are", "define", "definition of", "describe"],
            "procedure": ["how to", "steps to", "process for", "method for"],
            "comparison": ["compare", "difference between", "vs", "versus"],
            "reasoning": ["why", "because", "explain why", "causes of"],
            "numeric": ["how many", "how much", "percentage of", "value of"],
        }
        for qtype, patterns in QUERY_TYPE_PATTERNS.items():
            if any(pattern in query_lower for pattern in patterns):
                query_type = qtype
                break

        stop_words = {
            "the", "and", "or", "is", "are", "be", "to", "of", "for", "in", "on", "at", "with",
            "by", "from",
        }
        keywords = [t for t in tokenize(query) if t.lower() not in stop_words]

        KEYWORD_EXPANSION_RULES = {
            "inventory": ["stock", "warehouse", "storage", "stockkeeping"],
            "supply": ["supplier", "procurement", "chain", "logistics"],
            "demand": ["forecast", "order", "requirement", "need"],
            "performance": ["efficiency", "effectiveness", "productivity", "throughput"],
            "cost": ["expense", "price", "budget", "expenditure"],
            "risk": ["threat", "hazard", "uncertainty", "vulnerability"],
        }

        entities = []
        expansion_terms = []
        for kw in keywords:
            expanded = KEYWORD_EXPANSION_RULES.get(kw.lower(), [])
            expansion_terms.extend(expanded)
            if kw.lower() in KEYWORD_EXPANSION_RULES:
                entities.append(kw)

        expanded_terms = set(keywords + expansion_terms)
        rewritten_query = f"{query} {' '.join(expanded_terms)}" if expanded_terms else query

        return QueryAnalysis(
            original_query=query,
            rewritten_query=rewritten_query,
            query_type=query_type,
            entities=entities,
            keywords=keywords,
            expansion_terms=expansion_terms,
        )

    def _build_evidence_bindings(
        self, results: List[HybridSearchResult], analysis: QueryAnalysis
    ) -> List[RetrievalEvidenceBinding]:
        bindings = []
        for result in results:
            chunk = result.chunk
            matched_entities = [
                entity
                for entity in analysis.entities
                if entity.lower() in chunk.content.lower()
            ]

            binding = RetrievalEvidenceBinding(
                chunk_id=chunk.chunk_id,
                source_file=getattr(chunk, "source_file", ""),
                confidence=result.hybrid_score,
                relevance_score=result.hybrid_score,
                matched_entities=matched_entities,
                extraction_layer=getattr(chunk, "layer", ""),
                timestamp=datetime.now().isoformat(),
            )
            bindings.append(binding)

        return bindings

    def analyze_and_search_with_layers(
        self,
        query: str,
        topk: int = 10,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        use_rules: bool = True,
        use_rewrite: bool = True,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        analysis = self._analyze_query(query)
        search_query = analysis.rewritten_query if use_rewrite else query

        results = self.hybrid_search(
            query=search_query,
            topk=topk * 3,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
        )

        if use_rules:
            results = self.rule_engine.apply_rules(results, analysis)

        if layers:
            results = [r for r in results if getattr(r.chunk, "layer", "") in layers]

        results = results[:topk]

        evidence_bindings = self._build_evidence_bindings(results, analysis)

        layer_stats = self.vector_manager.get_layer_statistics()

        return {
            "original_query": query,
            "rewritten_query": search_query,
            "query_analysis": {
                "query_type": analysis.query_type,
                "entities": analysis.entities,
                "keywords": analysis.keywords,
                "expansion_terms": analysis.expansion_terms,
            },
            "results": [r.to_dict() for r in results],
            "evidence": [asdict(e) for e in evidence_bindings],
            "layer_statistics": layer_stats,
            "requested_layers": layers or ["all"],
        }


def build_hybrid_index(
    chunks: Sequence[ChunkRecord],
    rag_dir: Path | str,
    embedder=None,
) -> HybridIndexArtifacts:
    """便捷函数：构建混合索引"""
    manager = HybridIndexManager(rag_dir=rag_dir, embedder=embedder)
    return manager.build_hybrid_index(chunks)


def hybrid_search(
    query: str,
    rag_dir: Path | str,
    topk: int = 10,
    vector_weight: float = 0.5,
    keyword_weight: float = 0.5,
) -> List[HybridSearchResult]:
    """便捷函数：混合检索"""
    manager = HybridIndexManager(rag_dir=rag_dir)
    return manager.hybrid_search(
        query=query,
        topk=topk,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )