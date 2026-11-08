"""RAG answerer formatting characterization tests (RAG-04 precondition).

Locks the presentation/formatting behaviors of `rag_answerer.RAGAnswerer`
before any split of the 2967-line hotspot. These pure formatting methods are
the first candidates to move into a dedicated formatter/presentation module;
these tests freeze their output shapes so the split cannot drift silently.

Covered:
- _format_neo4j_result
- _format_document_result (incl. RAG-OPT4 structured-header injection)

The bare instance (`object.__new__` + minimal attrs) avoids the heavy
constructor side-effects (env / prompts / domain ontology loading).
"""
from __future__ import annotations

import json
import unittest

from src.application.rag.formatting import (
    build_section_a,
    build_section_b,
    build_section_c,
    build_section_d,
    build_section_e,
    build_section_f,
    categorize_relation,
    collect_related_entities,
    compute_difference_hint,
    extract_business_context,
    format_dify_structured_answer,
    format_document_result,
    format_neo4j_result,
    generate_ah_framework_report,
    generate_natural_language_report,
    identify_target_entity,
    synthesize_natural_answer,
    try_parse_dify_json,
)
from src.application.rag.models import ChunkRecord, RetrievedChunk
from src.application.rag.rag_answerer import RAGAnswerer


def _bare_answerer(max_chunk_length: int = 1000) -> RAGAnswerer:
    ans = object.__new__(RAGAnswerer)
    ans.max_chunk_length = max_chunk_length
    return ans


def _retrieved(content: str, *, chunk_type: str = "capability_card", meta=None, chunk_id: str = "cap_1", score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=ChunkRecord(
            chunk_id=chunk_id,
            chunk_type=chunk_type,
            content=content,
            node_ids=[],
            linked_node_ids=[],
            meta=meta or {},
        ),
        score=score,
    )


class Neo4jResultFormattingCharacterization(unittest.TestCase):
    def test_full_fields(self):
        a = _bare_answerer()
        out = a._format_neo4j_result(1, {
            "node_type": "System",
            "name": "OHF",
            "description": "desc",
            "domain": "MANUFACTURING",
        })
        self.assertIn("[1] OHF (System, 领域: MANUFACTURING)", out)
        self.assertIn("描述: desc", out)

    def test_no_domain_omits_domain_part(self):
        a = _bare_answerer()
        out = a._format_neo4j_result(0, {"node_type": "System", "name": "OHF"})
        self.assertIn("[0] OHF (System)", out)
        self.assertNotIn("领域", out)

    def test_description_truncated_at_200(self):
        a = _bare_answerer()
        long_desc = "x" * 250
        out = a._format_neo4j_result(0, {"name": "OHF", "description": long_desc})
        self.assertIn("x" * 200 + "...", out)
        # The description portion is exactly the first 200 chars (no 201st x).
        self.assertNotIn("x" * 201, out.split("描述: ")[1].split("...")[0])


class DocumentResultFormattingCharacterization(unittest.TestCase):
    def test_structured_header_injected(self):
        a = _bare_answerer()
        r = _retrieved(
            "capability text",
            meta={
                "meta_class": "Capability",
                "domain": "SUPPLY_CHAIN",
                "aliases": ["C1", "C2"],
            },
            chunk_id="cap_1",
        )
        out = a._format_document_result(2, r)
        self.assertIn("[2] 来源: unknown (类型: capability_card)", out)
        self.assertIn("ID: cap_1", out)
        self.assertIn("类型=Capability", out)
        self.assertIn("域=SUPPLY_CHAIN", out)
        self.assertIn("别名=C1,C2", out)
        self.assertIn("相关度=0.900", out)
        self.assertIn("capability text", out)

    def test_scenario_type_truncates_at_300(self):
        a = _bare_answerer(max_chunk_length=50)  # small default
        r = _retrieved(
            "s" * 400,
            chunk_type="l2_business_scenario",
            meta={"meta_class": "Process"},
        )
        out = a._format_document_result(0, r)
        self.assertIn("s" * 300, out)
        self.assertNotIn("s" * 301, out)

    def test_no_metadata_header_and_heuristic_from_content(self):
        a = _bare_answerer()
        r = _retrieved("meta_class: Capability\nDomain: AUTO\nplain content")
        out = a._format_document_result(0, r)
        # Heuristic parse from first content lines plugs in class + domain.
        self.assertIn("类型=Capability", out)
        self.assertIn("域=AUTO", out)

    def test_string_aliases_split_on_separators(self):
        a = _bare_answerer()
        # Actual separator regex: ,;，；、/| — spaces are NOT separators,
        # so "D1，D2 D3" becomes ["D1", "D2 D3"].
        r = _retrieved("t", meta={"entity_type": "Data", "aliases": "D1，D2 D3"})
        out = a._format_document_result(0, r)
        self.assertIn("别名=D1,D2 D3", out)

    def test_no_header_line_when_no_structured_fields(self):
        a = _bare_answerer()
        # score must also be absent, otherwise [相关度=…] still injects a header.
        r = _retrieved("plain text only", chunk_id="", score=None)
        out = a._format_document_result(0, r)
        self.assertNotIn("相关度", out)
        self.assertIn("plain text only", out)


class DirectFormatterCharacterization(unittest.TestCase):
    """The extracted pure functions must be independently testable (RAG-04)."""

    def test_format_neo4j_result_is_pure_module_function(self):
        out = format_neo4j_result(1, {"node_type": "System", "name": "OHF", "domain": "AUTO"})
        self.assertIn("[1] OHF (System, 领域: AUTO)", out)

    def test_format_document_result_accepts_explicit_params(self):
        r = _retrieved("text", meta={"meta_class": "Data"})
        out = format_document_result(
            0,
            r,
            max_chunk_length=1000,
            l2_scenario_types={"l2_business_scenario"},
        )
        self.assertIn("类型=Data", out)
        self.assertIn("text", out)

    def test_delegation_equals_direct_call(self):
        a = _bare_answerer()
        r = _retrieved("same text", meta={"entity_type": "System", "aliases": ["X"]})
        via_method = a._format_document_result(0, r)
        via_module = format_document_result(
            0,
            r,
            max_chunk_length=a.max_chunk_length,
            l2_scenario_types=a._L2_SCENARIO_TYPES,
        )
        self.assertEqual(via_method, via_module)


class DifyStructuredAnswerCharacterization(unittest.TestCase):
    def test_renders_entities_and_relationships(self):
        payload = {
            "L2_entities": [
                {
                    "meta_id": "E1",
                    "meta_class": "业务实体",
                    "name": "物料",
                    "description": "物料描述",
                    "domain": "MFG",
                    "status": "active",
                    "evidence": "ev1",
                }
            ],
            "relationships": [
                {"from": "E1", "to": "E2", "type": "REALIZES", "confidence": 0.9}
            ],
        }
        out = format_dify_structured_answer(json.dumps(payload, ensure_ascii=False), "什么是物料？")
        self.assertIn("### 针对问题「什么是物料？」的本体实体与关系分析", out)
        self.assertIn("#### L2 业务实体（共 1 个）", out)
        self.assertIn("**物料**", out)
        self.assertIn("ID: `E1`", out)
        self.assertIn("证据: ev1", out)
        self.assertIn("**REALIZES**", out)
        self.assertIn("实体总计: 1 ｜ 关系总计: 1", out)

    def test_empty_layers_rendered_explicitly(self):
        payload = {"L2_entities": [], "relationships": []}
        out = format_dify_structured_answer(json.dumps(payload), "q")
        self.assertIn("（共 0 个）", out)
        self.assertIn("_本层未返回实体（Dify Workflow 本次输出为空列表）_", out)
        self.assertIn("本次未返回关联关系", out)

    def test_non_structured_text_falls_back(self):
        raw = "这是一个普通的中文回答。"
        self.assertEqual(format_dify_structured_answer(raw, "q"), raw)

    def test_status_banner_scrubbed_before_parse(self):
        raw = '{"L2_entities": [], "relationships": []}\nL2_entities[] ❌ 空数组 ❌ 空数组 100%'
        out = format_dify_structured_answer(raw, "q")
        self.assertNotIn("❌ 空数组", out)


class SectionReportCharacterization(unittest.TestCase):
    """A-H framework report section builders (pure functions)."""

    def test_section_b_classifies_processes_and_systems(self):
        entity_map = {
            "P1": {"name": "生产流程", "meta_class": "Process", "description": "生产流程描述"},
            "S1": {"name": "SAP", "meta_class": "System", "description": "SAP 系统"},
        }
        relationships = [
            {"from": "T", "to": "P1", "type": "CONTAINS"},
            {"from": "T", "to": "S1", "type": "USES"},
        ]
        out = build_section_b(
            {"name": "目标"},
            {"contains": [], "uses": []},
            relationships,
            entity_map,
        )
        self.assertIn("### B. 涉及流程与系统", out)
        self.assertIn("**关键流程步骤**", out)
        self.assertIn("生产流程: 生产流程描述", out)
        self.assertIn("**涉及系统**", out)
        self.assertIn("- **SAP**: SAP 系统", out)

    def test_section_b_fallback_when_no_relations(self):
        out = build_section_b(
            {"name": "目标"},
            {"supports": [{"entity": {"name": "X"}, "relationship": {"type": "SUPPORTS"}}]},
            [],
            {},
        )
        self.assertIn("根据当前检索到的信息，该实体主要涉及以下关联", out)
        self.assertIn("通过 **SUPPORTS** 关系关联到 **X**", out)

    def test_section_c_identifies_actors_and_decision_rows(self):
        entity_map = {
            "A1": {"name": "项目经理", "meta_class": "Actor", "description": "负责项目"},
        }
        out = build_section_c({"name": "物料"}, {"uses": []}, entity_map)
        self.assertIn("### C. 角色与决策", out)
        self.assertIn("**相关角色**", out)
        self.assertIn("- **项目经理**: 负责项目", out)
        self.assertIn("基于 物料 数据驱动生产执行", out)
        self.assertIn("基于 物料 结构优化库存水平", out)

    def test_natural_language_report_uses_answer_and_context(self):
        long_answer = "x" * 200
        out = generate_natural_language_report("Q", long_answer, "ctx")
        self.assertIn("## 深度业务分析报告", out)
        self.assertIn("**分析主题**: Q", out)
        self.assertIn(long_answer, out)

        out2 = generate_natural_language_report("Q", "short", "context-body")
        self.assertIn("基于检索到的本体知识，以下是相关分析", out2)
        self.assertIn("context-body", out2)

    def test_section_d_rules_and_default_standards(self):
        out = build_section_d(
            {"name": "目标"},
            {},
            {"R1": {"name": "命名规则", "meta_class": "Rule", "description": "规则描述"}},
        )
        self.assertIn("### D. 常见问题与数据标准", out)
        self.assertIn("**相关规则和标准**", out)
        self.assertIn("- **命名规则**: 规则描述", out)

        out2 = build_section_d({"name": "目标"}, {}, {})
        self.assertIn("**数据标准建议**", out2)
        self.assertIn("- **更新频率**: 必须与发布快照同步", out2)

    def test_section_e_flows_and_scenarios(self):
        entity_map = {"A": {"name": "上游"}, "B": {"name": "下游"}}
        relationships = [{"from": "A", "to": "B", "type": "LINKS"}]
        out = build_section_e({"name": "物料"}, {}, relationships, entity_map)
        self.assertIn("### E. 典型业务场景", out)
        self.assertIn("上游 → [LINKS] → 下游", out)
        self.assertIn("1. 数据接收: 从上游系统接收 物料 结构数据", out)
        self.assertIn("4. 数据反馈: 将执行结果反馈给上游进行优化", out)

    def test_section_f_authority(self):
        out = build_section_f(
            {"meta_id": "E1", "name": "物料", "status": "active", "domain": "MFG"},
            {},
            {},
        )
        self.assertIn("### F. 数据来源与权威源", out)
        self.assertIn("- **数据标识**: E1", out)
        self.assertIn("- **数据名称**: 物料", out)
        self.assertIn("- **数据状态**: active", out)
        self.assertIn("- **数据领域**: MFG", out)
        self.assertIn("**权威源说明**", out)

    def test_section_a_definition_and_internal_names(self):
        out = build_section_a(
            {"meta_id": "E1", "name": "制造BOM", "description": "制造清单", "meta_class": "Data", "domain": "MFG"},
            {},
            {},
        )
        self.assertIn("### A. 数据对象定义", out)
        self.assertIn("**制造BOM** (ID: E1)", out)
        self.assertIn("> 制造清单", out)
        self.assertIn("- **内部叫法**: ", out)
        self.assertIn("MBOM", out)
        self.assertIn("- **数据分类**: Data", out)
        self.assertIn("- **所属领域**: MFG", out)

    def test_section_a_no_target_lists_entities(self):
        out = build_section_a(None, {}, {"E1": {"name": "物料", "description": "物料描述"}})
        self.assertIn("根据检索到的本体知识，以下是相关实体的分析", out)
        self.assertIn("- **物料**: 物料描述", out)

    def test_difference_hint_directional(self):
        self.assertIn("反映制造执行视角", compute_difference_hint("MBOM", "EBOM"))
        self.assertIn("反映设计视角", compute_difference_hint("EBOM", "MBOM"))
        self.assertEqual(compute_difference_hint("A", "B"), "")


class AhFrameworkPipelineCharacterization(unittest.TestCase):
    """Full A-H framework report pipeline (parse → identify → collect → render)."""

    def test_generate_ah_framework_report_end_to_end(self):
        payload = {
            "L0_entities": [
                {"meta_id": "L0-1", "name": "物料清单", "meta_class": "概念", "description": "物料清单概念"},
            ],
            "L2_entities": [
                {"meta_id": "L2-1", "name": "制造BOM", "meta_class": "数据", "description": "制造BOM数据", "domain": "MFG"},
                {"meta_id": "L2-2", "name": "工程BOM", "meta_class": "数据", "description": "工程BOM数据", "domain": "ENG"},
            ],
            "relationships": [
                {"from": "L0-1", "to": "L2-1", "type": "USES"},
            ],
        }
        out = generate_ah_framework_report(
            "什么是物料清单？",
            json.dumps(payload, ensure_ascii=False),
            "CMP 系统 流程 角色 上下文内容 " + "业务信息" * 200,
        )
        self.assertIn("### A. 数据对象定义", out)
        self.assertIn("### B. 涉及流程与系统", out)
        self.assertIn("### F. 数据来源与权威源", out)
        self.assertIn("补充业务上下文", out)

    def test_parse_fallback_to_natural_report_when_no_structured(self):
        out = generate_ah_framework_report("Q", "x" * 150, "ctx-body")
        self.assertIn("## 深度业务分析报告", out)

    def test_try_parse_dify_json_multiple_formats(self):
        # raw JSON
        self.assertEqual(try_parse_dify_json('{"L0_entities": [1]}'), {"L0_entities": [1]})
        # fenced json block
        out = try_parse_dify_json('```json\n{"L0_entities": []}\n```')
        self.assertEqual(out, {"L0_entities": []})
        # regex extraction
        out2 = try_parse_dify_json('text {"L0_entities": []} trailing')
        self.assertEqual(out2, {"L0_entities": []})
        # invalid
        self.assertEqual(try_parse_dify_json("not json"), {})
        self.assertEqual(try_parse_dify_json(""), {})

    def test_categorize_relation_keywords(self):
        self.assertEqual(categorize_relation("USES"), "uses")
        self.assertEqual(categorize_relation("Produces"), "produces")
        self.assertEqual(categorize_relation("contains"), "contains")
        self.assertEqual(categorize_relation("GOVERNED_BY"), "governs")
        self.assertEqual(categorize_relation("unrelated"), "other")

    def test_identify_target_entity_scoring(self):
        entities = [
            {"meta_id": "L0-1", "name": "物料清单", "description": "xx"},
            {"meta_id": "L0-2", "name": "BOM", "description": "yy"},
        ]
        self.assertEqual(
            identify_target_entity("什么是物料清单？", entities)["meta_id"], "L0-1"
        )
        self.assertIsNone(identify_target_entity("q", []))

    def test_collect_related_entities_direction_and_produces_flip(self):
        entity_map = {"E1": {"name": "A"}, "E2": {"name": "B"}}
        rels = [
            {"from": "T", "to": "E1", "type": "USES"},
            {"from": "E2", "to": "T", "type": "PRODUCES"},  # reversed -> uses
            {"from": "T", "to": "E2", "type": "CONTAINS"},
        ]
        related = collect_related_entities({"meta_id": "T"}, rels, entity_map)
        self.assertEqual([r["entity"]["name"] for r in related["uses"]], ["A", "B"])
        self.assertEqual([r["entity"]["name"] for r in related["contains"]], ["B"])

    def test_synthesize_natural_answer(self):
        payload = {
            "L2_entities": [
                {"meta_id": "L2-1", "name": "制造BOM", "meta_class": "数据", "description": "制造BOM数据", "domain": "MFG"},
            ],
            "relationships": [
                {"from": "L2-1", "to": "L2-2", "type": "USES", "description": "用于生产"},
            ],
        }
        out = synthesize_natural_answer("Q", json.dumps(payload, ensure_ascii=False), "ctx")
        self.assertIn("## 基于本体知识的回答", out)
        self.assertIn("**L2 层关键实体：**", out)
        self.assertIn("**制造BOM**（数据·MFG） — 制造BOM数据", out)
        self.assertIn("**实体间关系：**", out)
        self.assertIn("`L2-1` **USES** `L2-2`", out)
        self.assertIn("**综合分析：**", out)

    def test_synthesize_natural_answer_empty_and_invalid(self):
        self.assertEqual(synthesize_natural_answer("Q", "not json", "ctx"), "")
        self.assertEqual(synthesize_natural_answer("Q", "", "ctx"), "")
        self.assertEqual(synthesize_natural_answer("Q", '{"L2_entities": []}', "ctx"), "")


if __name__ == "__main__":
    unittest.main()