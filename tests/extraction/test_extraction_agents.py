"""Characterization tests for the extraction multi-agent skeleton (T3.1).

Locks: role registry, Reviewer SPEC checks (required fields / enums /
endpoint resolution / CQ-dead summary), and the serial orchestrator's
never-raise contract with injectable providers.
"""
from __future__ import annotations

import unittest

from ontology_compiler.services.extraction_agents import (
    ALL_ROLES,
    ROLE_CLASSIFIER,
    ROLE_ENTITY,
    ROLE_RELATION,
    ROLE_REVIEWER,
    review_ontology_packet,
    run_serial_extraction,
)
from tests.extraction.test_packet_e2e_mbom import _mbom_ecn_legacy_fixture


def _valid_min_packet() -> dict:
    return {
        "document": {"file_name": "d.pptx", "file_path": "d.pptx", "domain": "ENTERPRISE"},
        "layer_scope": "spec",
        "entities": {
            "L1": [{
                "id": "E1", "name": "Part A", "layer": "L1", "type": "Asset",
                "layer_scope": "spec",
            }],
            "L0": [], "L2": [], "L2.5": [], "L3": [],
        },
        "relations": [
            {"source_id": "E1", "target_id": "E1", "relation_type": "USED_IN",
             "layer_scope": "spec", "evidence_ref": "slide-1", "confidence": 1.0,
             "source_cq": "CQ-X-001"}
        ],
    }


class RoleRegistryTest(unittest.TestCase):
    def test_six_roles_registered_in_order(self):
        self.assertEqual(
            ALL_ROLES,
            ("classifier", "entity", "rule_l25", "relation", "evidence", "reviewer"),
        )


class ReviewerSpecChecksTest(unittest.TestCase):
    def test_valid_packet_has_no_issues(self):
        self.assertEqual(review_ontology_packet(_valid_min_packet()), [])

    def test_missing_required_entity_fields_flagged_as_errors(self):
        packet = {
            "entities": {"L1": [{"id": "", "name": "", "layer": "", "type": ""}]},
            "relations": [],
        }
        codes = [i["code"] for i in review_ontology_packet(packet)]
        for expected in (
            "ENTITY_MISSING_ID", "ENTITY_MISSING_NAME",
            "ENTITY_MISSING_LAYER", "ENTITY_MISSING_TYPE",
        ):
            self.assertIn(expected, codes)

    def test_enum_violations_flagged(self):
        packet = {
            "entities": {
                "L9": [{
                    "id": "E1", "name": "n", "type": "Asset",
                    "layer": "L9", "status": "banana", "confidence": 7.5,
                }]
            },
            "relations": [],
        }
        # layer bucket key is cosmetic; entity.layer drives the check
        issues = {i["code"]: i for i in review_ontology_packet(packet)}
        self.assertIn("LAYER_INVALID", issues)
        self.assertIn("STATUS_INVALID", issues)
        self.assertIn("CONFIDENCE_OUT_OF_RANGE", issues)
        self.assertEqual(issues["LAYER_INVALID"]["severity"], "error")

    def test_unresolved_endpoints_are_warnings_and_dead_cq_is_info(self):
        packet = {
            "entities": {
                "L1": [{"id": "E1", "name": "A", "layer": "L1", "type": "Asset"}]
            },
            "relations": [
                {"source_id": "E1", "target_id": "Ghost", "relation_type": "DEPENDS_ON"},
                {"source_id": "E1", "target_id": "Ghost", "relation_type": "RELATED_TO"},
            ],
        }
        issues = review_ontology_packet(packet)
        by_code = [i["code"] for i in issues]
        self.assertEqual(by_code.count("ENDPOINT_UNRESOLVED"), 2)
        dead = [i for i in issues if i["code"] == "CQ_DEAD_RELATIONS"]
        self.assertEqual(len(dead), 1)
        self.assertEqual(dead[0]["severity"], "info")
        self.assertEqual(dead[0]["dead_relations"], 2)

    def test_issues_sorted_error_first(self):
        packet = {
            "entities": {"L1": [{"id": "E1", "name": "A", "layer": "L1", "type": "Asset"}]},
            "relations": [
                {"source_id": "E1", "target_id": "Ghost", "relation_type": "X"},
            ],
        }
        severities = [i["severity"] for i in review_ontology_packet(packet)]
        self.assertEqual(severities, sorted(severities, key={"error": 0, "warning": 1, "info": 2}.get))


class SerialOrchestratorTest(unittest.TestCase):
    def test_precomputed_legacy_runs_all_roles(self):
        report = run_serial_extraction(
            _mbom_ecn_legacy_fixture(),
            document_name="ecn_case.pptx",
            domain="MANUFACTURING",
        )
        roles_run = [r["role"] for r in report["role_runs"]]
        self.assertEqual(list(ALL_ROLES), roles_run)
        self.assertIsNotNone(report["packet"])
        # reviewer ran on a healthy fixture: only the seeded noise edge speaks
        codes = [i["code"] for i in report["issues"]]
        self.assertIn("CQ_DEAD_RELATIONS", codes)
        self.assertNotIn("ENTITY_MISSING_ID", codes)

    def test_extractor_callable_invoked_and_adapted(self):
        calls = []

        def fake_extractor(text: str) -> dict:
            calls.append(text)
            return {
                "L1_entities": [{"meta_id": "X1", "name": "Node X"}],
                "relationships": [],
            }

        report = run_serial_extraction(extractor=fake_extractor, document_text="hello")
        self.assertEqual(calls, ["hello"])
        entities = list(report["packet"]["entities"].values())
        flat = [e for bucket in entities for e in bucket]
        self.assertEqual([e["id"] for e in flat], ["X1"])

    def test_no_inputs_degrades_without_raising(self):
        report = run_serial_extraction()
        skipped = {r["role"] for r in report["role_runs"] if r.get("skipped")}
        self.assertIn(ROLE_ENTITY, skipped)
        self.assertIsNone(report["packet"])
        self.assertEqual(report["issues"], [])

    def test_exploding_providers_never_raise(self):
        def boom(*a, **k):
            raise RuntimeError("provider down")

        report = run_serial_extraction(
            _mbom_ecn_legacy_fixture(),
            document_classifier=boom,
            evidence_provider=boom,
            document_text="t",
        )
        runs = {r["role"]: r for r in report["role_runs"]}
        self.assertFalse(runs["classifier"]["ok"])
        self.assertFalse(runs["evidence"]["ok"])
        # precomputed legacy still flows through despite provider failures
        self.assertIsNotNone(report["packet"])
        self.assertTrue(any(r["role"] == ROLE_REVIEWER and r["ok"] for r in report["role_runs"]))

    def test_exploding_extractor_fails_entity_role_only(self):
        def boom(text):
            raise RuntimeError("llm down")

        report = run_serial_extraction(extractor=boom, document_text="t")
        runs = {r["role"]: r for r in report["role_runs"]}
        self.assertFalse(runs["entity"]["ok"])
        self.assertIn("llm down", runs["entity"]["error"])
        self.assertIsNone(report["packet"])
        skipped = {r["role"] for r in report["role_runs"] if r.get("skipped")}
        self.assertIn(ROLE_RELATION, skipped)
        self.assertIn(ROLE_REVIEWER, skipped)


if __name__ == "__main__":
    unittest.main()
