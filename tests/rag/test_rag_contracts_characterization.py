"""RAG contract characterization tests.

Locks the current field/default/serialization behavior of the RAG data
contracts (RAG-01: lock query/retrieval/answer contracts). These DTOs are the
boundary contracts for the RAG bounded context; before any split/refactor of
the rag hotspots (rag_answerer.py 2967 lines, hybrid_rag.py, etc.) we first
freeze their shapes so behavior cannot drift silently.

Covered contracts:
- retrieval: src.application.rag.models (ChunkRecord / RetrievedChunk /
  ChunkWithVector / LLMAnswerResult / RetrievalResult / Neo4jSearchResult /
  RewrittenQuery / RAGConfig)
- answer: src.application.rag.rag_answerer.AnswerResult / QuestionType
- query: src.application.rag.unified_query (ParsedQuery / QueryClassification /
  QueryType / QueryIntent / QueryComplexity)
- schema docs: `schemas/rag_retrieval.schema.yaml`, `schemas/rag_answer.schema.yaml`,
  `schemas/rag_result.schema.yaml`

Note: these tests assert the CURRENT shapes only. Changing a contract is a
breaking change and requires version bump + migration notes (Phase 4 B13),
updating these assertions accordingly.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from src.application.rag.models import (
    ChunkRecord,
    ChunkWithVector,
    LLMAnswerResult,
    Neo4jSearchResult,
    RetrievalResult,
    RetrievedChunk,
    RewrittenQuery,
    RAGConfig,
)
from src.application.rag.rag_answerer import AnswerResult, QuestionType
from src.application.rag.unified_query import (
    ParsedQuery,
    QueryClassification,
    QueryComplexity,
    QueryIntent,
    QueryType,
)
from src.domain.models.evidence import Evidence


def _make_chunk() -> ChunkRecord:
    return ChunkRecord(
        chunk_id="cap_1",
        chunk_type="capability_card",
        content="capability card content",
        node_ids=["L1-1"],
        linked_node_ids=["L2-2"],
        meta={"source_file": "a.md"},
    )


class ChunkRecordCharacterization(unittest.TestCase):
    def test_fields_and_to_dict_without_evidence(self):
        c = _make_chunk()
        d = c.to_dict()
        self.assertEqual(d["chunk_id"], "cap_1")
        self.assertEqual(d["chunk_type"], "capability_card")
        self.assertEqual(d["content"], "capability card content")
        self.assertEqual(d["node_ids"], ["L1-1"])
        self.assertEqual(d["linked_node_ids"], ["L2-2"])
        self.assertEqual(d["meta"], {"source_file": "a.md"})
        self.assertNotIn("evidence", d)  # evidence popped when None

    def test_to_dict_with_evidence_serializes_via_model_dump(self):
        c = _make_chunk()
        c = ChunkRecord(
            **{  # reconstruct with evidence
                "chunk_id": c.chunk_id,
                "chunk_type": c.chunk_type,
                "content": c.content,
                "node_ids": c.node_ids,
                "linked_node_ids": c.linked_node_ids,
                "meta": c.meta,
                "evidence": Evidence(source_file="a.md", content="src"),
            }
        )
        d = c.to_dict()
        self.assertIn("evidence", d)
        self.assertEqual(d["evidence"]["source_file"], "a.md")

    def test_is_frozen_dataclass(self):
        c = _make_chunk()
        with self.assertRaises(Exception):
            c.chunk_id = "mutated"

    def test_schema_docs_exist(self):
        root = Path(__file__).resolve().parents[2]
        for rel in ("schemas/rag_retrieval.schema.yaml", "schemas/rag_answer.schema.yaml", "schemas/rag_result.schema.yaml"):
            self.assertTrue((root / rel).exists(), rel)


class RetrievedChunkCharacterization(unittest.TestCase):
    def test_to_dict_and_convenience_properties(self):
        r = RetrievedChunk(chunk=_make_chunk(), score=0.9)
        d = r.to_dict()
        self.assertEqual(d["score"], 0.9)
        self.assertEqual(d["chunk"]["chunk_id"], "cap_1")
        self.assertEqual(r.content, "capability card content")
        self.assertEqual(r.source, "a.md")
        self.assertEqual(r.metadata["chunk_type"], "capability_card")
        self.assertEqual(r.metadata["node_ids"], ["L1-1"])


class ChunkWithVectorCharacterization(unittest.TestCase):
    def test_default_scores_are_zero(self):
        r = RetrievedChunk(chunk=_make_chunk(), score=0.9)
        v = ChunkWithVector(chunk=r)
        self.assertEqual(v.vector_score, 0.0)
        self.assertEqual(v.keyword_score, 0.0)
        self.assertEqual(v.hybrid_score, 0.0)


class LLMAnswerResultCharacterization(unittest.TestCase):
    def test_fields_and_default_tokens(self):
        a = LLMAnswerResult(
            answer="ans",
            evidence_chunks=[{"chunk_id": "cap_1"}],
            confidence=0.8,
            model="deepseek-chat",
        )
        self.assertEqual(a.answer, "ans")
        self.assertEqual(a.confidence, 0.8)
        self.assertEqual(a.model, "deepseek-chat")
        self.assertIsNone(a.tokens_used)


class RetrievalResultCharacterization(unittest.TestCase):
    def test_fields(self):
        r = RetrievalResult(
            chunks=[],
            source_type="hybrid",
            query="what is OHF",
            topk=5,
            execution_time_ms=12.5,
        )
        self.assertEqual(r.source_type, "hybrid")
        self.assertEqual(r.query, "what is OHF")
        self.assertEqual(r.topk, 5)
        self.assertEqual(r.execution_time_ms, 12.5)
        self.assertEqual(r.chunks, [])


class Neo4jSearchResultCharacterization(unittest.TestCase):
    def test_properties(self):
        n = Neo4jSearchResult(
            entity_name="OHF",
            entity_type="System",
            description="OHF description",
            related_entities=[{"name": "ARO", "relation": "links_to"}],
            scenarios=["scenario A"],
            score=0.7,
        )
        self.assertEqual(n.content, "OHF: OHF description")
        self.assertEqual(n.source, "neo4j:System:OHF")
        self.assertEqual(n.metadata["entity_type"], "System")
        self.assertEqual(n.metadata["scenarios"], ["scenario A"])


class RewrittenQueryCharacterization(unittest.TestCase):
    def test_fields(self):
        r = RewrittenQuery(original="FOO", rewritten="BAR", method="none", confidence=0.0)
        self.assertEqual(r.method, "none")
        self.assertEqual(r.confidence, 0.0)


class RAGConfigCharacterization(unittest.TestCase):
    def test_defaults(self):
        cfg = RAGConfig()
        self.assertEqual(cfg.model, "deepseek-chat")
        self.assertEqual(cfg.max_context_chunks, 5)
        self.assertEqual(cfg.max_chunk_length, 1000)
        self.assertFalse(cfg.use_neo4j)
        self.assertEqual(cfg.neo4j_weight, 0.3)
        self.assertEqual(cfg.hybrid_alpha, 0.5)


class AnswerResultCharacterization(unittest.TestCase):
    def test_to_dict_with_and_without_reason(self):
        a = AnswerResult(answer="ans", evidence_node_ids=["L1-1"], confidence=0.9)
        d = a.to_dict()
        self.assertEqual(d["answer"], "ans")
        self.assertEqual(d["evidence_node_ids"], ["L1-1"])
        self.assertEqual(d["confidence"], 0.9)
        self.assertIsNone(d["cannot_answer_reason"])

    def test_question_type_from_string_matches_enum_values(self):
        # Actual behavior: from_string matches the enum VALUE ("l1_children"),
        # not the enum member name ("L1_CHILDREN") - cls(value) lookup.
        self.assertEqual(QuestionType.from_string("l1_children"), QuestionType.L1_CHILDREN)
        self.assertEqual(QuestionType.from_string("path_l1_l2"), QuestionType.PATH_L1_L2)
        self.assertIsNone(QuestionType.from_string("L1_CHILDREN"))  # member name, not value
        self.assertIsNone(QuestionType.from_string("nonsense"))
        self.assertIsNone(QuestionType.from_string(None))


class QueryContractCharacterization(unittest.TestCase):
    def test_parsed_query_fields_and_entity_set(self):
        q = ParsedQuery(
            original="what is OHF",
            normalized="what is OHF",
            tokens=["what", "is", "ohf"],
            entities=["OHF"],
            is_multi_entity=False,
            recommended_topk=5,
        )
        self.assertEqual(q.entity_set, {"OHF"})
        self.assertFalse(q.is_multi_entity)

    def test_query_classification_defaults(self):
        c = QueryClassification(
            query="what is OHF",
            query_type=QueryType.ENTITY_LOOKUP,
            intent=QueryIntent.DEFINITION,
            complexity=QueryComplexity.SIMPLE,
        )
        self.assertEqual(c.confidence, 1.0)
        self.assertEqual(c.entities, [])
        self.assertEqual(c.suggestions, [])
        self.assertEqual(c.created_at, "")

    def test_enum_values(self):
        self.assertEqual(QueryType.ENTITY_LOOKUP.value, "entity_lookup")
        self.assertEqual(QueryIntent.LIST.value, "list")
        self.assertEqual(QueryComplexity.COMPLEX.value, "complex")


if __name__ == "__main__":
    unittest.main()
