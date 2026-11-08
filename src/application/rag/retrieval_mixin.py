"""检索能力 Mixin - BM25+Vector 融合检索、实体扩展与 gate 循环。

RAG-07 (§9.7, 2026-09-03): 从 rag_answerer.py 拆出检索能力，使
rag_answerer.py 聚焦于统一路由/编排。本模块提供 ``RetrievalMixin``：

1. 混合检索（BM25+Vector RRF 融合，纯 BM25 兜底）
2. 实体扩展漏斗与 L2 场景合成摘要
3. Evidence-gate 驱动的多轮重检索（legacy + packet 路径）

RAGAnswerer 通过 Mixin 继承获得这些方法；实例级 ``mock.patch.object``
对继承方法同样生效，behavior 由 characterization 测试锁定。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .evidence_gate import EvidenceGateAction, EvidenceItem
from .hybrid_rag import HybridIndexManager, HybridSearchResult
from .models import RetrievedChunk
from .retriever import retrieve_keyword


class RetrievalMixin:
    """检索能力（混合检索 / 实体扩展 / gate 循环）。"""

    def _format_retrieved_results(self, question: str, results: List[Any]) -> str:
        lines = [
            f"(未启用 LLM，仅显示检索到的 {len(results)} 条相关信息)",
            "",
            "您的问题可以基于以下信息回答:",
            "=" * 70,
            ""
        ]
        for i, r in enumerate(results[:self.max_context_chunks], 1):
            chunk = r.chunk
            source = chunk.meta.get("source", "unknown")
            chunk_type = getattr(chunk, "chunk_type", "unknown")
            lines.append(f"[{i}] 来源: {source} (类型: {chunk_type}, 相关度: {r.score:.3f})")
            lines.append("-" * 70)
            content = chunk.content[:self.max_chunk_length]
            lines.append(content)
            lines.append("")
        lines.append("=" * 70)
        lines.append("提示: 设置 OPENAI_API_KEY 可启用 LLM 智能解答")
        return "\n".join(lines)

    def _is_entity_master_chunk(self, chunk: Any, entity: str) -> bool:
        chunk_type = getattr(chunk, "chunk_type", "")
        if chunk_type not in self._MASTER_NODE_TYPES:
            return False
        entity_upper = entity.upper()
        chunk_id = getattr(chunk, "chunk_id", "").upper()
        content_head = getattr(chunk, "content", "")[:80].upper()
        return entity_upper in chunk_id or entity_upper in content_head

    def _synthesize_entity_summary(self, entity: str, l2_results: List[Any]) -> Optional[Any]:
        from .chunk_indexing import ChunkRecord

        entity_upper = entity.upper()
        direct_l2 = [
            r for r in l2_results
            if entity_upper in getattr(r.chunk, "chunk_id", "").upper()
            and getattr(r.chunk, "chunk_type", "") in self._L2_SCENARIO_TYPES
        ]

        lines = [f"[合成摘要] {entity} — 从 L2 场景推断的实体概述:"]
        seen_desc: Set[str] = set()
        has_l2_data = False
        meta_class = "Concept"
        domain = "Unknown"

        if direct_l2:
            has_l2_data = True
            for r in direct_l2[:5]:
                content = getattr(r.chunk, "content", "")
                for line in content.splitlines():
                    if line.startswith("Description:") and line not in seen_desc:
                        seen_desc.add(line)
                        lines.append(f"  • {line}")
                        break
                if len(seen_desc) >= 3:
                    break

        if len(seen_desc) == 0:
            meta_class = self._extract_entity_meta_class(entity, l2_results)
            domain = self._extract_entity_domain(entity, l2_results)
            default_desc = self._generate_default_description(entity, meta_class, domain)
            lines.append(f"  • Description: {default_desc}")

        synthesized_content = "\n".join(lines)
        synth_chunk = ChunkRecord(
            chunk_id=f"SYNTH_{entity_upper}_SUMMARY",
            chunk_type="synthesized_summary",
            content=synthesized_content,
            node_ids=[],
            linked_node_ids=[],
            meta={
                "source": "synthesized_from_l2" if has_l2_data else "template_generated",
                "entity": entity,
                "meta_class": meta_class,
            },
        )

        if direct_l2:
            first = direct_l2[0]
            ResultType = type(first)
            try:
                return ResultType(chunk=synth_chunk, score=first.score * 0.9)
            except Exception:
                return None
        elif l2_results:
            try:
                for r in l2_results[:1]:
                    ResultType = type(r)
                    return ResultType(chunk=synth_chunk, score=0.5)
            except Exception:
                return None
        else:
            try:
                synth_chunk.score = 0.3
                return synth_chunk
            except Exception:
                return None

    def _extract_entity_meta_class(self, entity: str, l2_results: List[Any]) -> str:
        entity_upper = entity.upper()
        for r in l2_results:
            chunk_id = getattr(r.chunk, "chunk_id", "").upper()
            if entity_upper in chunk_id:
                content = getattr(r.chunk, "content", "")
                for line in content.splitlines():
                    if line.startswith("Class:") or line.startswith("meta_class:"):
                        return line.split(":", 1)[1].strip()
        return "Concept"

    def _extract_entity_domain(self, entity: str, l2_results: List[Any]) -> str:
        entity_upper = entity.upper()
        for r in l2_results:
            chunk_id = getattr(r.chunk, "chunk_id", "").upper()
            if entity_upper in chunk_id:
                content = getattr(r.chunk, "content", "")
                for line in content.splitlines():
                    if line.startswith("Domain:") or line.startswith("domain:"):
                        return line.split(":", 1)[1].strip()
        return "Unknown"

    def _generate_default_description(self, entity: str, meta_class: str, domain: str) -> str:
        if domain and domain != "Unknown":
            return f"A {meta_class} entity representing {entity} in the {domain} domain."
        return f"A {meta_class} entity representing {entity}."

    def _append_unique_results(self, merged: List[Any], seen_ids: Set[str], items: List[Any]) -> None:
        for r in items:
            cid = r.chunk.chunk_id
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(r)

    # ------------------------------------------------------------------
    # RAG-P0-1 (2026-08-04): True BM25+Vector fusion retrieval entry point
    # ------------------------------------------------------------------
    def _get_hybrid_manager(self) -> Optional[HybridIndexManager]:
        """Lazy accessor for the in-process HybridIndexManager.

        Returns None when the vector-index / inverted-index artifacts aren't
        readable (e.g. developer ran with a partial rag_dir). In that case the
        caller is expected to fall back to pure-BM25 ``retrieve_keyword``.
        """
        if self._hybrid_manager is not None:
            return self._hybrid_manager
        if self._hybrid_manager_tried:
            return None  # previous attempt failed; don't retry
        self._hybrid_manager_tried = True
        try:
            # Rely on the same rag_dir that was used to load chunks/index/idf above.
            mgr = HybridIndexManager(self.rag_dir)
            # Lightweight validation: try to list known chunks.  If the manager
            # raises for any reason (missing vector index, corrupt metadata…) we
            # don't want to propagate; the pure-BM25 fallback preserves behaviour.
            try:
                chunks = mgr._load_chunks()  # pylint: disable=protected-access
                if chunks is None or len(chunks) == 0:
                    return None
            except Exception:
                return None
            self._hybrid_manager = mgr
            return self._hybrid_manager
        except Exception:
            return None

    def _retrieve_hybrid_fused(self, query: str, topk: int) -> List[Any]:
        """BM25+Vector fused retrieval with graceful BM25-only fallback.

        Returns a list of items that all expose ``.chunk.chunk_id`` / ``.chunk``
        (either ``HybridSearchResult`` from hybrid manager or ``RetrievedChunk``
        from the legacy keyword path).  Callers in ``_retrieve_with_entity_expansion`` /
        ``_retrieve_single_entity_scenarios`` only read ``.chunk`` and
        ``.chunk.chunk_id``, so both return shapes are compatible.
        """
        mgr = self._get_hybrid_manager()
        if mgr is not None:
            try:
                results: List[HybridSearchResult] = mgr.hybrid_search(query, topk=topk)
                if results:
                    # Convert to RetrievedChunk to keep downstream filtering code
                    # (which uses ``.score`` in some display paths) happy.
                    return [
                        RetrievedChunk(chunk=r.chunk, score=float(r.hybrid_score))
                        for r in results
                    ]
            except Exception:
                # Any error in the hybrid path → fall through to pure BM25.
                pass
        # Pure-BM25 fallback (original legacy behaviour).
        return retrieve_keyword(query, self.chunks, self.index, topk=topk, idf=self.idf)

    def _retrieve_with_entity_expansion(self, pq: Any, topk: int) -> List[Any]:
        """Query-driven retrieval with ontology entity expansion.

        RAG-P0-1 (2026-08-04):
        - Entry-point retrieval is now ``_retrieve_hybrid_fused`` which goes
          through HybridIndexManager (BM25+Vector RRF fusion) when the vector
          index is available on disk, falling back to the original pure BM25
          retrieve_keyword otherwise.  This eliminates the "old chain" that
          completely ignored vectors.
        - Per-entity L2 pool retrieval (used for multi-entity scenarios and
          single-entity scenario expansion) uses the same fused helper.
        - Behavioural guarantee: the legacy BM25-only path remains reachable
          whenever HybridIndexManager can't be instantiated, so no pipeline /
          CI breakage from this change.
        """
        seen_ids: Set[str] = set()
        merged: List[Any] = []

        self._append_unique_results(
            merged,
            seen_ids,
            self._retrieve_hybrid_fused(pq.normalized, topk=topk),
        )

        if not pq.is_multi_entity:
            if pq.entities:
                self._append_unique_results(
                    merged,
                    seen_ids,
                    self._retrieve_single_entity_scenarios(pq.entities[0]),
                )
            return merged[:pq.recommended_topk]

        entity_l2_pool: dict[str, List[Any]] = {}
        for entity in pq.entities:
            entity_results = self._retrieve_hybrid_fused(entity, topk=8)
            entity_l2_pool[entity] = entity_results
            self._append_unique_results(merged, seen_ids, entity_results)
            self._append_unique_results(
                merged,
                seen_ids,
                self._filter_entity_scenarios(entity, entity_results),
            )

        boosted, rest, entities_with_master = self._split_boosted_results(merged, pq.entities)
        self._add_synthesized_summaries(boosted, seen_ids, pq.entities, entity_l2_pool, entities_with_master)
        merged = boosted + rest
        return merged[:pq.recommended_topk]

    def _retrieve_with_gate(self, pq: Any, topk: int, max_retrieval_rounds: int = 3) -> List[Any]:
        """P1-1 (2026-09-01): Evidence-gate driven retrieval loop (legacy path).

        Round 1 runs the existing entity-expansion funnel unchanged, so the base
        retrieval behaviour is preserved.  When :class:`EvidenceGate` decides
        RETRIEVE_MORE, subsequent rounds re-run the raw fused funnel (which
        honours ``top_k`` — the entity-expansion funnel clips to
        ``pq.recommended_topk``) with a 50%-expanded top_k and merge the new
        chunks (dedup by ``chunk.chunk_id``).  Stops on ANSWER / ESCALATE.

        Returns the merged result list.  The final gate decision and the number
        of retrieval rounds are recorded on ``self._last_gate_decision`` /
        ``self._last_gate_rounds_used`` (None when the gate is disabled).
        """
        gate = getattr(self, "evidence_gate", None)
        if gate is None:
            return self._retrieve_with_entity_expansion(pq, topk=topk)

        accumulated: Dict[str, Any] = {}
        for r in self._retrieve_with_entity_expansion(pq, topk=topk):
            accumulated[r.chunk.chunk_id] = r

        query_terms = list(getattr(pq, "entities", None) or [])
        current_topk = topk
        round_num = 1
        decision = None
        while round_num <= max_retrieval_rounds:
            evidence_items = [
                EvidenceItem(
                    text=r.chunk.content,
                    score=max(0.0, min(1.0, float(r.score))),
                    source_id=r.chunk.chunk_id,
                )
                for r in accumulated.values()
            ]
            decision = gate.evaluate(
                evidence_items,
                query_terms=query_terms,
                has_conflict=False,
                retrieval_round=round_num,
                max_retrieval_rounds=max_retrieval_rounds,
            )
            if decision.action in (EvidenceGateAction.ANSWER, EvidenceGateAction.ESCALATE):
                break
            current_topk = max(current_topk, int(current_topk * 1.5) + 1)
            for r in self._retrieve_hybrid_fused(pq.normalized, topk=current_topk):
                cid = r.chunk.chunk_id
                if cid not in accumulated:
                    accumulated[cid] = r
            round_num += 1

        self._last_gate_decision = decision
        self._last_gate_rounds_used = round_num
        return list(accumulated.values())

    def _retrieve_single_entity_scenarios(self, entity: str) -> List[Any]:
        extra_scenarios = self._retrieve_hybrid_fused(entity, topk=8)
        return self._filter_scenarios(extra_scenarios)

    def _filter_scenarios(self, results: List[Any]) -> List[Any]:
        return [r for r in results if getattr(r.chunk, "chunk_type", "") in self._L2_SCENARIO_TYPES]

    def _filter_entity_scenarios(self, entity: str, results: List[Any]) -> List[Any]:
        entity_upper = entity.upper()
        return [
            r for r in self._filter_scenarios(results)
            if entity_upper in getattr(r.chunk, "chunk_id", "").upper()
        ]

    def _split_boosted_results(self, merged: List[Any], entities: List[str]) -> tuple[List[Any], List[Any], Set[str]]:
        boosted: List[Any] = []
        rest: List[Any] = []
        entities_with_master: Set[str] = set()

        for r in merged:
            matched = [e for e in entities if self._is_entity_master_chunk(r.chunk, e)]
            if matched:
                boosted.append(r)
                entities_with_master.update(matched)
            else:
                rest.append(r)

        return boosted, rest, entities_with_master

    def _add_synthesized_summaries(
        self,
        boosted: List[Any],
        seen_ids: Set[str],
        entities: List[str],
        entity_l2_pool: dict[str, List[Any]],
        entities_with_master: Set[str],
    ) -> None:
        for entity in entities:
            if entity not in entities_with_master:
                synth = self._synthesize_entity_summary(entity, entity_l2_pool.get(entity, []))
                if synth and synth.chunk.chunk_id not in seen_ids:
                    seen_ids.add(synth.chunk.chunk_id)
                    boosted.insert(0, synth)

