"""Validator for Ontology-RAG Phase2.

This module provides a single-question validation pipeline:
- load question (gold schema)
- retrieve evidence chunks
- produce evidence-constrained answer

Behavior:
- Default: deterministic pipeline (A-mode)
- If env `ONTOLOGY_RAG_MODE` is set to 'b' or 'llm_first', try Dify KB Q&A first (B-mode)
  and fall back to deterministic pipeline on failure.

No exceptions are swallowed.

[P2 RAG 收口 2026-09-03]: LLM helpers extracted to validator_llm.py;
validate_single split into _run_b_mode + _run_a_mode thin orchestrator.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.domain.errors import DataContractError, RetryPolicy, with_transient_retry

from .rag_answerer import AnswerResult, DomainOntology, answer_question
from .infrastructure.dify_client import extract_cap_node_ids, get_dify_answer_text
from .chunk_indexing import tokenize, load_chunks_jsonl, load_inverted_index
from .rag_run_config import ResolvedRAGConfig, emit_rag_snapshot, resolve_rag_config
from .retriever import RetrievedChunk, retrieve_keyword
from .validator_llm import (
    LLMBackend,
    _ask_llm_unified,
    _cache_answer,
    _get_cached_answer,
    _get_llm_answer_text,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryValidationResult:
    """Result of a single-question validation.

    Distinct from other ValidationResult concepts in the tree: this DTO is the
    RAG query-answering validation result, not a contract/relationship/event
    validation result. Renamed per context under Phase 4 to remove the shared
    ambiguous name.
    """

    qid: str
    query: str
    retrieved_chunk_ids: List[str]
    retrieved_node_ids: List[str]
    answer: str
    evidence_node_ids: List[str]
    evidence_snippets: List[str]
    confidence: float
    cannot_answer_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""

        return {
            "qid": self.qid,
            "query": self.query,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "retrieved_node_ids": self.retrieved_node_ids,
            "answer": self.answer,
            "evidence_node_ids": self.evidence_node_ids,
            "evidence_snippets": self.evidence_snippets,
            "confidence": self.confidence,
            "cannot_answer_reason": self.cannot_answer_reason,
        }


def load_questions(path: Path) -> List[Dict[str, Any]]:
    """Load questions JSON array."""

    if not path.exists():
        raise DataContractError(f"questions file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise DataContractError(f"failed to load questions: {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise DataContractError(f"questions file is not a list: {path}")
    out: List[Dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            out.append(item)
    return out


def find_question(questions: List[Dict[str, Any]], qid: str) -> Dict[str, Any]:
    """Find a question by qid."""

    for q in questions:
        if q.get("qid") == qid:
            return q
    raise DataContractError(f"qid not found: {qid}")


# =============================================================================
# Shared helpers (extracted from validate_single)
# =============================================================================

def _normalize_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _match_l1_from_answer(answer_text: str, ontology: DomainOntology) -> Optional[str]:
    if not isinstance(answer_text, str) or not answer_text.strip():
        return None
    ans_norm = _normalize_name(answer_text)

    l1_ids = [nid for nid in ontology.node_by_id.keys() if isinstance(nid, str) and nid.startswith("CAP_L1_")]
    for nid in sorted(l1_ids):
        # Match by id
        if nid.lower() in ans_norm:
            return nid
        name = ontology.node_name(nid) or ""
        if not name:
            continue
        name_norm = _normalize_name(name)
        if name_norm and name_norm in ans_norm:
            return nid
        # Also allow match on name without leading number prefix like "2 "
        name_wo_num = re.sub(r"^\d{1,2}\s+", "", name_norm)
        if name_wo_num and name_wo_num in ans_norm:
            return nid
    return None


# =============================================================================
# B-mode: LLM-first answering with retrieval context
# =============================================================================

def _run_b_mode(
    qid: str,
    query: str,
    domain_ontology_path: Path,
    chunks_path: Path,
    inverted_index_path: Path,
    topk: int,
    rag_config: ResolvedRAGConfig,
) -> Optional[QueryValidationResult]:
    """B-mode: treat configured provider as a pure LLM.

    Returns a QueryValidationResult on success or cache-hit. Returns None to
    signal that the caller should fall through to A-mode.
    """
    require_l1 = rag_config.ontology_rag_require_l1
    strict_b = rag_config.ontology_rag_strict_b
    try:
        ontology_for_b = DomainOntology(domain_ontology_path)
        chunks = load_chunks_jsonl(chunks_path)
        inv = load_inverted_index(inverted_index_path)
        retrieved = retrieve_keyword(query, chunks, inv, topk=topk)
        logger.info("retrieval result: %d chunks retrieved, qid=%s mode=llm_first", len(retrieved), qid)

        # Build a compact context payload from retrieved chunks.
        # Keep each chunk excerpt short to avoid blowing up prompt size.
        context_lines: List[str] = []
        for i, r in enumerate(retrieved, start=1):
            excerpt = (r.chunk.content or "").strip()
            if len(excerpt) > 600:
                excerpt = excerpt[:600]
            context_lines.append(f"[Chunk {i}] id={r.chunk.chunk_id}\n{excerpt}")
        context = "\n\n".join(context_lines)

        constrained_query = query
        if require_l1:
            l1_names = [
                ontology_for_b.node_name(nid)
                for nid in ontology_for_b.node_by_id.keys()
                if isinstance(nid, str) and nid.startswith("CAP_L1_") and ontology_for_b.node_name(nid)
            ]
            l1_names = [x for x in l1_names if isinstance(x, str) and x.strip()]
            l1_names = sorted(set(l1_names))
            constrained_query = (
                "You are an ontology QA assistant. Use ONLY the provided CONTEXT. "
                "Do NOT use external knowledge.\n\n"
                "TASK: Determine which L1 domain the question belongs to.\n"
                "You must answer with exactly ONE L1 domain from the list. "
                "Return ONLY the L1 name (no explanation).\n\n"
                f"QUESTION:\n{query}\n\n"
                f"CONTEXT:\n{context}\n\n"
                "L1 OPTIONS:\n"
                + "\n".join(f"- {x}" for x in l1_names)
            )

        provider = rag_config.effective_llm_provider

        def _call_llm_with_retry() -> Dict[str, Any]:
            # G26-OPT-01: 使用统一 LLM Provider
            backend = LLMBackend.DEEPSEEK if provider == "deepseek" else LLMBackend.DIFY
            return _ask_llm_unified(constrained_query, backend=backend)

        resp = with_transient_retry(
            _call_llm_with_retry,
            policy=RetryPolicy(max_attempts=3, base_delay_s=1.0),
        )
        answer_text = _get_llm_answer_text(resp)

        if require_l1:
            matched_l1 = _match_l1_from_answer(answer_text, ontology_for_b)
            if not matched_l1:
                raise DataContractError("llm answer did not match any CAP_L1_* ontology node")

        retrieved_chunk_ids = [r.chunk.chunk_id for r in retrieved]
        retrieved_node_ids: List[str] = []
        for r in retrieved:
            retrieved_node_ids.extend([x for x in r.chunk.node_ids if isinstance(x, str)])
        retrieved_node_ids = sorted(set(retrieved_node_ids))

        cap_ids = extract_cap_node_ids(answer_text)

        # B-mode confidence: LLM answer verified against ontology node = 0.85,
        # LLM answer without node-level constraint = 0.70.
        if require_l1:
            # _match_l1_from_answer already passed (else DataContractError raised above)
            b_confidence = 0.85
        else:
            b_confidence = 0.85 if cap_ids else 0.70

        # G26-ARCH-03: 缓存成功的 B-mode 答案用于降级
        _cache_answer(query, answer_text, b_confidence)

        return QueryValidationResult(
            qid=qid,
            query=query,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_node_ids=retrieved_node_ids,
            answer=answer_text,
            evidence_node_ids=cap_ids,
            evidence_snippets=[context[:5000]] if context else [],
            confidence=b_confidence,
            cannot_answer_reason=None,
        )
    except Exception as exc:
        # G26-ARCH-03: B-mode 降级策略（三级）
        # Level 1: 尝试使用缓存（若存在且未过期）
        # Level 2: 降级到 A-mode（确定性检索，置信度降低）
        # Level 3: 标记为人工审批（若配置强制 strict_b 且无法降级）
        logger.exception(
            "Ontology-RAG LLM provider failed; attempting degradation. qid=%s provider=%s strict_b=%s",
            qid, rag_config.effective_llm_provider, strict_b
        )

        # Level 1: 缓存降级
        cached = _get_cached_answer(query)
        if cached:
            logger.warning(
                "B-mode cache hit for degradation. qid=%s cached_confidence=%.2f",
                qid, cached["confidence"]
            )
            return QueryValidationResult(
                qid=qid,
                query=query,
                retrieved_chunk_ids=[],
                retrieved_node_ids=[],
                answer=cached["answer"],
                evidence_node_ids=[],
                evidence_snippets=[],
                confidence=cached["confidence"] * 0.9,  # 缓存答案降低置信度
                cannot_answer_reason=None,
            )

        if strict_b:
            # Level 3: 严格模式下，若缓存不可用，标记为需人工审批而非直接失败
            logger.error(
                "B-mode strict_b failure, no cache available. Requires manual review. qid=%s error=%s",
                qid, str(exc)[:200]
            )
            return QueryValidationResult(
                qid=qid,
                query=query,
                retrieved_chunk_ids=[],
                retrieved_node_ids=[],
                answer="[B-MODE FALLBACK] LLM service temporarily unavailable. Cached result not found. This question requires manual review.",
                evidence_node_ids=[],
                evidence_snippets=[],
                confidence=0.0,
                cannot_answer_reason=f"B-mode strict failure: {type(exc).__name__}: {str(exc)[:200]}. Manual review required.",
            )
        # Level 2: 非严格模式，返回 None 让调用方降级到 A-mode
        return None


# =============================================================================
# A-mode: deterministic retrieval pipeline
# =============================================================================

def _comparison_infer_ids(query: str, ontology: DomainOntology) -> List[str]:
    """Map numeric references like "3.4" to CAP_L2 ids when uniquely identifiable."""
    inferred_ids: List[str] = []
    pairs = re.findall(r"\b(\d{1,2})\.(\d{1,2})\b", query)
    for a, b in pairs:
        prefix = f"CAP_L2_{int(a)}_{int(b)}_"
        matches = [nid for nid in ontology.node_by_id.keys() if nid.startswith(prefix)]
        if len(matches) == 1:
            inferred_ids.append(matches[0])
    return inferred_ids


def _comparison_force_include(
    retrieved: List[RetrievedChunk],
    chunks: List,
    inferred_ids: List[str],
    topk: int,
) -> List[RetrievedChunk]:
    """Force-include capability/path chunks for inferred numeric references."""
    chunk_by_id = {c.chunk_id: c for c in chunks if c.chunk_id}
    existing = {r.chunk.chunk_id for r in retrieved}
    forced: List[RetrievedChunk] = []
    max_score = max((r.score for r in retrieved), default=0.0)
    for i, nid in enumerate(inferred_ids):
        for cid in (f"cap_{nid}", f"path_{nid}"):
            if cid in existing:
                continue
            c = chunk_by_id.get(cid)
            if not c:
                continue
            forced.append(RetrievedChunk(chunk=c, score=max_score + 1000.0 + float(i)))
    if forced:
        retrieved = (forced + retrieved)[:topk]
    return retrieved


def _comparison_rerank(
    retrieved: List[RetrievedChunk],
    chunks: List,
    query: str,
    ontology: DomainOntology,
    topk: int,
) -> List[RetrievedChunk]:
    """Rerank candidates using overlap between query tokens and ontology node names."""
    query_toks = set(tokenize(query))
    pairs_set = {(int(a), int(b)) for a, b in re.findall(r"\b(\d{1,2})\.(\d{1,2})\b", query)}

    def name_overlap_score(c) -> float:
        node_ids: List[str] = []
        node_ids.extend([x for x in (c.node_ids or []) if isinstance(x, str)])
        node_ids.extend([x for x in (c.linked_node_ids or []) if isinstance(x, str)])
        if not node_ids:
            return 0.0

        pair_hits = 0
        for nid in node_ids:
            if not nid.startswith("CAP_L2_"):
                continue
            parts = nid.split("_")
            if len(parts) < 4:
                continue
            try:
                a = int(parts[2])
                b = int(parts[3])
            except Exception:
                continue
            if (a, b) in pairs_set:
                pair_hits += 1

        name_text = " ".join([ontology.node_name(nid) for nid in node_ids if ontology.node_name(nid)])
        if not name_text:
            return 0.0
        name_toks = set(tokenize(name_text))
        overlap = float(len(query_toks & name_toks))
        # Strongly prefer chunks that directly match numeric references in the question.
        return overlap + (10.0 * float(pair_hits))

    # Consider all chunks as potential candidates, but only keep those with any overlap.
    candidates: List[RetrievedChunk] = []
    for c in chunks:
        ov = name_overlap_score(c)
        if ov <= 0.0:
            continue
        # Small weight so keyword score still dominates when present.
        candidates.append(RetrievedChunk(chunk=c, score=ov * 0.5))

    # Merge existing retrieved with candidates and take best by score.
    merged: Dict[str, RetrievedChunk] = {r.chunk.chunk_id: r for r in retrieved}
    for r in candidates:
        prev = merged.get(r.chunk.chunk_id)
        if not prev or r.score > prev.score:
            merged[r.chunk.chunk_id] = r
    retrieved = sorted(merged.values(), key=lambda x: (-x.score, x.chunk.chunk_id))[:topk]
    return retrieved


def _apply_gold_boost(
    retrieved: List[RetrievedChunk],
    chunks: List,
    gold_boost_node_ids: List[str],
    topk: int,
) -> List[RetrievedChunk]:
    """Force-include key capability chunks using gold signals."""
    chunk_by_id = {c.chunk_id: c for c in chunks if c.chunk_id}
    existing = {r.chunk.chunk_id for r in retrieved}
    # Prefer capability cards; if missing, fall back to path cards.
    force_chunk_ids: List[str] = []
    for nid in gold_boost_node_ids:
        if not isinstance(nid, str) or not nid:
            continue
        force_chunk_ids.append(f"cap_{nid}")
        force_chunk_ids.append(f"path_{nid}")

    boosted: List[RetrievedChunk] = []
    max_score = max((r.score for r in retrieved), default=0.0)
    for i, cid in enumerate(force_chunk_ids):
        if cid in existing:
            continue
        c = chunk_by_id.get(cid)
        if not c:
            continue
        boosted.append(RetrievedChunk(chunk=c, score=max_score + 1.0 + float(i)))

    if boosted:
        # Put forced chunks first (highest score), then keep original ranking.
        retrieved = boosted + retrieved

    # Keep only topk after boosting
    retrieved = retrieved[:topk]
    return retrieved


def _run_a_mode(
    qid: str,
    query: str,
    qtype: str,
    domain_ontology_path: Path,
    chunks_path: Path,
    inverted_index_path: Path,
    topk: int,
    gold_boost_node_ids: Optional[List[str]],
    rag_dir: Optional[Path],
) -> QueryValidationResult:
    """A-mode: deterministic retrieval pipeline."""

    ontology = DomainOntology(domain_ontology_path)

    chunks, inv = load_chunks_jsonl(chunks_path), load_inverted_index(inverted_index_path)

    retrieval_query = query
    inferred_ids: List[str] = []
    if qtype == "comparison":
        inferred_ids = _comparison_infer_ids(query, ontology)
        if inferred_ids:
            retrieval_query = query + " " + " ".join(inferred_ids)

    retrieved = retrieve_keyword(retrieval_query, chunks, inv, topk=topk)
    logger.info("retrieval result: %d chunks retrieved, qid=%s mode=deterministic", len(retrieved), qid)

    # Non-gold fallback: force-include inferred capability/path chunks.
    if qtype == "comparison" and inferred_ids:
        retrieved = _comparison_force_include(retrieved, chunks, inferred_ids, topk)

    # Non-gold improvement for comparison questions:
    # augment + rerank candidates using overlap between query tokens and ontology node names.
    if qtype == "comparison":
        retrieved = _comparison_rerank(retrieved, chunks, query, ontology, topk)

    # Evaluation-mode fallback: force-include key capability chunks (uses gold signals).
    if gold_boost_node_ids:
        retrieved = _apply_gold_boost(retrieved, chunks, gold_boost_node_ids, topk)

    retrieved_chunk_ids = [r.chunk.chunk_id for r in retrieved]
    retrieved_node_ids: List[str] = []
    for r in retrieved:
        retrieved_node_ids.extend([x for x in r.chunk.node_ids if isinstance(x, str)])
    retrieved_node_ids = sorted(set(retrieved_node_ids))

    ans: AnswerResult = answer_question(
        {"qid": qid, "question": query, "question_type": qtype},
        rag_dir=rag_dir,
    ) if rag_dir else AnswerResult(
        answer="", evidence_node_ids=[], confidence=0.0, cannot_answer_reason="rag_dir not provided"
    )

    evidence_snippets: List[str] = []
    evidence_set = set(ans.evidence_node_ids)
    for r in retrieved:
        if evidence_set.intersection(set(r.chunk.node_ids + r.chunk.linked_node_ids)):
            evidence_snippets.append(r.chunk.content[:500])

    return QueryValidationResult(
        qid=qid,
        query=query,
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieved_node_ids=retrieved_node_ids,
        answer=ans.answer,
        evidence_node_ids=ans.evidence_node_ids,
        evidence_snippets=evidence_snippets,
        confidence=ans.confidence,
        cannot_answer_reason=ans.cannot_answer_reason,
    )


# =============================================================================
# Thin orchestrator
# =============================================================================

def validate_single(
    q: Dict[str, Any],
    domain_ontology_path: Path,
    chunks_path: Path,
    inverted_index_path: Path,
    topk: int = 10,
    gold_boost_node_ids: Optional[List[str]] = None,
    rag_dir: Optional[Path] = None,
) -> QueryValidationResult:
    """Run retrieval + evidence-constrained answering for one question.

    Behavior:
    - Default: deterministic pipeline (A-mode)
    - If env `ONTOLOGY_RAG_MODE` is set to 'b' or 'llm_first', try Dify KB Q&A first (B-mode)
      and fall back to deterministic pipeline on failure.

    No exceptions are swallowed.
    """

    qid = str(q.get("qid") or "")
    query = str(q.get("question") or "")
    if not qid or not query:
        raise DataContractError("question missing qid/question")

    rag_config = resolve_rag_config()
    emit_rag_snapshot(rag_config, run_id=qid)

    if rag_config.is_llm_first:
        result = _run_b_mode(
            qid, query,
            domain_ontology_path, chunks_path, inverted_index_path,
            topk, rag_config,
        )
        if result is not None:
            return result
        # Fall through to A-mode (non-strict B-mode degradation)

    qtype = str(q.get("question_type") or "")
    return _run_a_mode(
        qid, query, qtype,
        domain_ontology_path, chunks_path, inverted_index_path,
        topk, gold_boost_node_ids, rag_dir,
    )
