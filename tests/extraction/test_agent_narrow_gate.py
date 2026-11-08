"""P2-3 (2026-09-09 Gate E): lock the narrow-relation -> reviewer -> DAG precedence guarantee.

P2-3's policy (enforced elsewhere):
- Relation narrow must be verifiable independently of entity narrow
- Reviewer has precedence (reviewer > relation > entity) with a deterministic verdict
- Entity narrow does not mask relation quality improvements
- DAG parallelism is only meaningful after serial stability

These tests codify the guardrails, not the final tune.
"""
from __future__ import annotations

import unittest


def _valid_lazy_packet(**overrides):
    """Minimal ontology packet shaped dict with all required fields (L1 + relation)."""
    packet = {
        "document": {
            "file_name": "test_doc.pdf",
            "file_path": "test_doc.pdf",
            "domain": "ENTERPRISE",
        },
        "layer_scope": "spec",
        "entities": {
            "L1": [{"id": "E1", "name": "System X", "layer": "L1", "type": "System"}],
            "L0": [],
            "L2": [],
            "L2.5": [],
            "L3": [],
        },
        "relations": [
            {
                "source_id": "E1",
                "target_id": "E1",
                "relation_type": "USED_IN",
                "source_cq": "CQ-1",
            }
        ],
    }
    for k, v in overrides.items():
        packet[k] = v
    return packet


class ReviewerRecommendationContractTest(unittest.TestCase):
    def test_pure_helper_has_precedence_field(self):
        from ontology_compiler.services.extraction_agents import reviewer_recommendation
        rec = reviewer_recommendation([], fail_on_error=False)
        self.assertEqual(rec["precedence"], "reviewer > relation > entity")
        self.assertEqual(rec["verdict"], "pass")
        self.assertTrue(rec["answerable"])

    def test_error_in_warn_mode_is_answerable_but_degraded(self):
        from ontology_compiler.services.extraction_agents import reviewer_recommendation
        rec = reviewer_recommendation([{"severity": "error"}], fail_on_error=False)
        self.assertTrue(rec["answerable"])
        self.assertEqual(rec["verdict"], "warn")

    def test_error_in_fail_mode_blocks_release(self):
        from ontology_compiler.services.extraction_agents import reviewer_recommendation
        rec = reviewer_recommendation([{"severity": "error"}], fail_on_error=True)
        self.assertFalse(rec["answerable"])
        self.assertEqual(rec["verdict"], "degrade")
        self.assertEqual(rec["fallback"], "merged_stage_packet")

    def test_no_issues_always_pass(self):
        from ontology_compiler.services.extraction_agents import reviewer_recommendation
        self.assertTrue(reviewer_recommendation([], fail_on_error=False)["answerable"])
        self.assertTrue(reviewer_recommendation([], fail_on_error=True)["answerable"])


class NarrowRelationEvidenceBindingTest(unittest.TestCase):
    def test_relation_narrow_provider_binds_evidence_deterministically(self):
        from ontology_compiler.services.extraction_agents import run_serial_extraction
        source_text = "System X depends on Supplier Ledger.\nSecond line."
        rels = [
            {"source_id": "E1", "target_id": "E2", "relation_type": "DEPENDS_ON"},
        ]

        def narrow_provider(legacy):
            return rels

        report = run_serial_extraction(
            legacy_extraction=_valid_lazy_packet(relations=rels),
            relation_narrow_provider=narrow_provider,
            document_text=source_text,
        )
        # Narrow provider output flows through evidence binding — relation should cite a line
        out_rels = (report["packet"] or {}).get("relations") or []
        tagged = [r for r in out_rels if r.get("evidence")]
        self.assertEqual(len(tagged), len(out_rels))

    def test_relation_narrow_never_breaks_chain(self):
        from ontology_compiler.services.extraction_agents import run_serial_extraction
        valid_packet_kwargs = dict(document_name="x.pptx", domain="test")
        def boom(legacy):
            raise RuntimeError("relation narrow blew up")
        report = run_serial_extraction(
            legacy_extraction=_valid_lazy_packet(),
            relation_narrow_provider=boom,
            **valid_packet_kwargs,
        )
        # chain must still release — degraded fallback on narrow failure
        self.assertIsNotNone(report["packet"])
        self.assertTrue(report["release_ok"])


class DagParallelGateTest(unittest.TestCase):
    def test_dag_disallowed_until_serial_stable(self):
        from ontology_compiler.services.extraction_agents import dag_parallel_feasibility
        r = dag_parallel_feasibility(
            narrow_relation_enabled=True,
            narrow_entity_enabled=True,
            dag_enabled=True,
            p0_cq_stable=False,
            evidence_complete=False,
        )
        self.assertFalse(r["dag_allowed"])

    def test_dag_disallowed_without_narrow_ready(self):
        from ontology_compiler.services.extraction_agents import dag_parallel_feasibility
        r = dag_parallel_feasibility(
            narrow_relation_enabled=True,
            narrow_entity_enabled=False,
            dag_enabled=True,
            p0_cq_stable=True,
            evidence_complete=True,
        )
        self.assertFalse(r["dag_allowed"])

    def test_dag_requires_explicit_toggle(self):
        from ontology_compiler.services.extraction_agents import dag_parallel_feasibility
        r = dag_parallel_feasibility(
            narrow_relation_enabled=True,
            narrow_entity_enabled=True,
            dag_enabled=False,
            p0_cq_stable=True,
            evidence_complete=True,
        )
        self.assertFalse(r["dag_allowed"])

    def test_dag_allowed_when_all_conditions_met(self):
        from ontology_compiler.services.extraction_agents import dag_parallel_feasibility
        r = dag_parallel_feasibility(
            narrow_relation_enabled=True,
            narrow_entity_enabled=True,
            dag_enabled=True,
            p0_cq_stable=True,
            evidence_complete=True,
        )
        self.assertTrue(r["dag_allowed"])
        self.assertEqual(r["reasons"], [])
        self.assertTrue(r["serial_stable"])
        self.assertTrue(r["narrow_ready"])

    def test_run_serial_exposes_dag_allowed_field(self):
        from ontology_compiler.services.extraction_agents import run_serial_extraction
        report = run_serial_extraction(_valid_lazy_packet())
        self.assertIn("dag_allowed", report)
        self.assertIsInstance(report["dag_allowed"], bool)

    def test_stable_packet_auto_enables_entity_and_dag(self):
        import os
        from ontology_compiler.services.extraction_agents import (
            assess_cq_evidence_stability,
            run_serial_extraction,
        )

        packet = _valid_lazy_packet()
        packet["relations"] = [
            {
                "source_id": "E1",
                "target_id": "E1",
                "relation_type": "USED_IN",
                "source_cq": "CQ-1",
                "evidence_ref": "doc:slide-1",
            }
        ]
        stab = assess_cq_evidence_stability(packet)
        self.assertTrue(stab["p0_cq_stable"])
        self.assertTrue(stab["evidence_complete"])

        os.environ.pop("EXTRACTION_NARROW_ENTITY", None)
        os.environ.pop("EXTRACTION_DAG_PARALLEL", None)
        report = run_serial_extraction(
            legacy_extraction=packet,
            document_text="System X depends on Supplier Ledger.",
        )
        self.assertIn("stability", report)
        self.assertTrue(report["dag_allowed"])

if __name__ == "__main__":
    unittest.main()
