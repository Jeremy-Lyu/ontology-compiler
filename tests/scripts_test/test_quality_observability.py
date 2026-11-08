"""Regression checks for P1-5 quality observability (warn-only, Gate D).

口径锁定（2026-09-09 实证）：
1. dead_relation 按"来路不明"（无 provenance 又无 evidence）统计——source_cq
   在生产链路无生产者（仅 adapter 透传；adapter 注释明示 master 无 source_cq），
   按无 source_cq 统计恒 ~100% 无信号；source_cq 覆盖度单列追踪 CQ 接入。
2. entity_drift 按 name+aliases 匹配；ghost 节点（"auto created ghost node"）
   单列 ghost_missing——merge 清洗 ghost 是治理成功，语义丢失以
   layered_golden merge recall / retention 为准。
"""
from __future__ import annotations

import unittest

from scripts.quality_observability import (
    aggregate_quality_observability,
    build_quality_observability,
    dead_relation_stats,
    entity_drift_extract_to_master,
    evidence_coverage_master,
    related_to_share_master,
)


class DeadRelationCaliberTest(unittest.TestCase):
    """dead_ratio 必须反映"来路不明"，而不是"无 source_cq"。"""

    def test_traced_relations_are_not_dead_even_without_source_cq(self):
        rels = [
            {"relation_type": "DERIVED_FROM", "provenance": "rule"},
            {"relation_type": "PART_OF", "evidence_ref": "ev-1"},
            {"relation_type": "USES", "provenance": "llm", "evidence": {"content": "x"}},
        ]
        out = dead_relation_stats(rels)
        self.assertEqual(out["dead_relations"], 0)
        self.assertEqual(out["dead_ratio"], 0.0)
        # source_cq 无人生产时覆盖度应诚实为 0，而不是污染 dead_ratio
        self.assertEqual(out["with_source_cq"], 0)
        self.assertEqual(out["source_cq_coverage"], 0.0)

    def test_only_untraceable_relations_count_as_dead(self):
        rels = [
            {"relation_type": "DERIVED_FROM", "provenance": "rule"},
            {"relation_type": "RELATED_TO"},  # 无 provenance、无 evidence → dead
            {"relation_type": "USES", "provenance": "", "evidence": None},
        ]
        out = dead_relation_stats(rels)
        self.assertEqual(out["dead_relations"], 2)
        self.assertEqual(out["dead_ratio"], round(2 / 3, 4))
        self.assertEqual(out["dead_by_type"], {"RELATED_TO": 1, "USES": 1})

    def test_empty_master_is_zero_not_error(self):
        out = dead_relation_stats([])
        self.assertEqual(out["dead_ratio"], 0.0)


class EntityDriftCaliberTest(unittest.TestCase):
    """drift 必须按 name+aliases 匹配，ghost 单列。"""

    def test_alias_rename_is_match_not_missing(self):
        extract = [{"name": "MBOM Art Fg Pos Pa Structure", "aliases": []}]
        master = [{"name": "MBOM", "aliases": ["MBOM Art Fg Pos Pa Structure"]}]
        out = entity_drift_extract_to_master(extract, master)
        self.assertEqual(out["missing_count"], 0)
        self.assertEqual(out["matched"], 1)

    def test_ghost_cleanup_is_not_real_missing(self):
        extract = [
            {"name": "engine", "description": "auto created ghost node (was referenced in relationship"},
            {"name": "MBOM", "description": "real entity"},
        ]
        master = [{"name": "MBOM"}]
        out = entity_drift_extract_to_master(extract, master)
        self.assertEqual(out["missing_count"], 1)
        self.assertEqual(out["ghost_missing_count"], 1)
        self.assertEqual(out["real_missing_count"], 0)
        self.assertEqual(out["real_missing_names"], [])
        self.assertEqual(out["ghost_missing_names"], ["engine"])

    def test_case_insensitive(self):
        out = entity_drift_extract_to_master([{"name": "EBOM"}], [{"name": "ebom"}])
        self.assertEqual(out["missing_count"], 0)


class EvidenceAndRelatedToTest(unittest.TestCase):
    def test_evidence_coverage(self):
        ents = [{"name": "A", "evidence": {"content": "x"}}, {"name": "B"}]
        out = evidence_coverage_master(ents)
        self.assertEqual(out, {"total": 2, "with_evidence": 1, "coverage": 0.5})

    def test_related_to_share(self):
        rels = [{"relation_type": "RELATED_TO"}, {"relation_type": "DERIVED_FROM"}]
        out = related_to_share_master(rels)
        self.assertEqual(out, {"total": 2, "related_to": 1, "share": 0.5})


class BuildAndAggregateTest(unittest.TestCase):
    def test_build_block_shape_and_warn_only_semantics(self):
        block = build_quality_observability(
            [{"name": "A"}],
            [],
            [{"name": "A", "evidence": {"content": "x"}}],
            [{"relation_type": "USES", "provenance": "llm"}],
        )
        self.assertIn("entity_drift", block)
        self.assertIn("evidence", block)
        self.assertIn("dead_relations", block)
        self.assertIn("related_to", block)
        self.assertIn("口径", block)
        self.assertEqual(block["dead_relations"]["dead_ratio"], 0.0)
        self.assertEqual(block["evidence"]["coverage"], 1.0)

    def test_aggregate_means_and_recurring(self):
        b1 = build_quality_observability(
            [{"name": "A"}, {"name": "B", "description": "auto created ghost node x"}],
            [],
            [{"name": "A"}],
            [{"relation_type": "RELATED_TO"}],
        )
        b2 = build_quality_observability(
            [{"name": "A"}],
            [],
            [{"name": "A"}],
            [{"relation_type": "USES", "provenance": "rule"}],
        )
        agg = aggregate_quality_observability([b1, b2])
        self.assertEqual(agg["metrics"]["evidence_coverage"]["n"], 2)
        self.assertIn("recurring_missing_in_master", agg)
        self.assertIn("recurring_real_missing_in_master", agg)
        self.assertIn("real_missing_ratio", agg["metrics"])
        self.assertIn("source_cq_coverage", agg["metrics"])
        # ghost（B）只进 ghost 口径，不进 real_missing
        self.assertNotIn("b", agg["recurring_real_missing_in_master"])


if __name__ == "__main__":
    unittest.main()
