"""Characterization tests for T3.2 skill-contract enforcement (ROADMAP).

Locks: three-layer contract validation (classification/entity/relation) as a
pure function; the mbom skill YAML carries a well-formed contract block; the
extraction_agents orchestrator enforces contracts in the rule_l25 role and
surfaces violations without ever raising.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from ontology_compiler.services.extraction_agents import (
    ROLE_RULE_L25,
    run_serial_extraction,
)
from ontology_compiler.services.skill_contract import (
    CONTRACT_CLASSIFICATION,
    CONTRACT_ENTITY,
    CONTRACT_RELATION,
    validate_extraction_against_contract,
)


def _legacy_ok() -> dict:
    return {
        "L1_entities": [
            {"meta_id": "P1", "meta_class": "Asset", "name": "Part A",
             "evidence_ref": "doc:§1"},
            {"meta_id": "E1", "meta_class": "Process", "name": "ECN-1",
             "evidence_ref": "doc:§2"},
        ],
        "L2_entities": [
            {"meta_id": "S1", "meta_class": "BusinessScenario", "name": "变更场景",
             "evidence_ref": "doc:§3"},
        ],
        "relationships": [
            {"from": "Part A", "to": "变更场景", "type": "USED_IN",
             "evidence_ref": "doc:§3", "source_cq": "CQ-1"},
        ],
    }


def _contract() -> dict:
    return {
        "skill_id": "test_skill",
        CONTRACT_CLASSIFICATION: {
            "required_layers": ["L1", "L2"],
            "noise_patterns": ["^TODO"],
        },
        CONTRACT_ENTITY: {
            "rag_required_fields": ["evidence_ref"],
            "required_entity_types": {
                "Asset": {"min_count": 1, "required_fields": ["meta_id"]},
                "Process": {"min_count": 0, "required_fields": ["name"]},
            },
        },
        CONTRACT_RELATION: {
            "forbidden_relation_types": ["DERIVED_FROM"],
            "evidence_required_relation_types": ["USED_IN"],
        },
    }


class ContractValidatorTest(unittest.TestCase):
    def test_clean_extraction_passes(self):
        verdict = validate_extraction_against_contract(_legacy_ok(), _contract())
        self.assertTrue(verdict["ok"], msg=str(verdict["violations"]))
        self.assertEqual(verdict["violations"], [])

    def test_required_layer_empty_flagged(self):
        legacy = _legacy_ok()
        legacy["L2_entities"] = []
        verdict = validate_extraction_against_contract(legacy, _contract())
        codes = {v["code"] for v in verdict["violations"]}
        self.assertFalse(verdict["ok"])
        self.assertIn("REQUIRED_LAYER_EMPTY", codes)

    def test_noise_pattern_and_rag_field_violations(self):
        legacy = _legacy_ok()
        legacy["L1_entities"].append(
            {"meta_id": "T9", "meta_class": "Asset", "name": "TODO fixme"}
        )
        legacy["L1_entities"][0]["evidence_ref"] = ""  # RAG field gone
        verdict = validate_extraction_against_contract(legacy, _contract())
        codes = [v["code"] for v in verdict["violations"]]
        self.assertIn("NOISE_PATTERN_HIT", codes)
        self.assertIn("RAG_REQUIRED_FIELD_MISSING", codes)

    def test_relation_layers_enforced(self):
        legacy = _legacy_ok()
        legacy["relationships"] = [
            {"from": "Part A", "to": "变更场景", "type": "DERIVED_FROM"},  # forbidden + no evidence
        ]
        verdict = validate_extraction_against_contract(legacy, _contract())
        codes = {v["code"] for v in verdict["violations"]}
        self.assertIn("FORBIDDEN_RELATION_USED", codes)

    def test_evidence_required_missing(self):
        legacy = _legacy_ok()
        legacy["relationships"] = [
            {"from": "Part A", "to": "变更场景", "type": "USED_IN"},  # no evidence_ref
        ]
        verdict = validate_extraction_against_contract(legacy, _contract())
        codes = {v["code"] for v in verdict["violations"]}
        self.assertIn("EVIDENCE_REQUIRED_MISSING", codes)

    def test_invalid_regex_reported_not_raised(self):
        contract = _contract()
        contract[CONTRACT_CLASSIFICATION]["noise_patterns"] = ["([bad"]
        verdict = validate_extraction_against_contract(_legacy_ok(), contract)
        codes = {v["code"] for v in verdict["violations"]}
        self.assertIn("NOISE_PATTERN_INVALID", codes)

    def test_pure_function_never_mutates_input(self):
        legacy = _legacy_ok()
        snapshot = repr(legacy)
        validate_extraction_against_contract(legacy, _contract())
        self.assertEqual(repr(legacy), snapshot)


class MbomSkillContractBlockTest(unittest.TestCase):
    def test_mbom_yaml_declares_valid_three_layer_contract(self):
        path = Path("skills/l25/mbom_extraction.yaml")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        contract = data.get("contract") or {}
        for layer in (CONTRACT_CLASSIFICATION, CONTRACT_ENTITY, CONTRACT_RELATION):
            self.assertIn(layer, contract, f"mbom skill contract missing {layer}")
        self.assertTrue(contract[CONTRACT_ENTITY].get("rag_required_fields"))
        self.assertTrue(contract[CONTRACT_RELATION].get("forbidden_relation_types"))


class OrchestratorContractIntegrationTest(unittest.TestCase):
    def test_rule_l25_role_runs_contracts_and_surfaces_violations(self):
        bad = _legacy_ok()
        bad["relationships"].append(
            {"from": "Part A", "to": "Part A", "type": "DERIVED_FROM"}  # forbidden
        )
        report = run_serial_extraction(bad, skill_contracts=[_contract()])
        viol = report["contract_violations"]
        self.assertTrue(any(v["code"] == "FORBIDDEN_RELATION_USED" for v in viol))
        self.assertTrue(all(v.get("skill_id") == "test_skill" for v in viol))
        rule_runs = [r for r in report["role_runs"]
                     if r["role"] == ROLE_RULE_L25 and r.get("output", {}).get("kind") == "contract_check"]
        self.assertTrue(rule_runs)

    def test_no_contracts_means_no_violations_key_entries(self):
        report = run_serial_extraction(_legacy_ok())
        self.assertEqual(report["contract_violations"], [])


if __name__ == "__main__":
    unittest.main()
