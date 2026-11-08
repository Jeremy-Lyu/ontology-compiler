from __future__ import annotations

"""Characterization tests for build_chunks_from_domain_ontology (plan Phase 1 #4).

Locks the current chunk-building behavior for a domain ontology payload: which
meta_class branches produce cards, how capability levels/numbers/parents are
derived, how linked node ids are collected/sorted/deduped, and the data-contract
errors for missing keys. Pure function — no I/O, no LLM. This is the regression
baseline for a Phase 3 split of the 389-line builder.
"""

import unittest

from src.application.rag.chunk_indexing import (
    CHUNK_PREFIX_ACTOR,
    CHUNK_PREFIX_ASSET,
    CHUNK_PREFIX_CAP,
    CHUNK_PREFIX_DATA,
    CHUNK_PREFIX_PATH,
    CHUNK_PREFIX_PROC,
    CHUNK_PREFIX_SCENARIO,
    CHUNK_PREFIX_SYSMAP,
    _make_chunk_id,
    build_chunks_from_domain_ontology,
)
from src.domain.errors import DataContractError


def _node(nid: str, meta_class: str, **extra) -> dict:
    node = {
        "meta_id": nid,
        "meta_class": meta_class,
        "meta_name": f"{meta_class}-{nid}",
        "meta_description": f"description of {nid}",
        "sandbox_domain": "SUPPLY_CHAIN",
        "meta_owner": "tester",
    }
    node.update(extra)
    return node


def _rel(source: str, target: str, rtype: str) -> dict:
    return {"source": source, "target": target, "type": rtype}


class BuildDomainChunksCharacterizationTest(unittest.TestCase):
    """Characterize build_chunks_from_domain_ontology chunk semantics."""

    def test_full_coverage_card_types_and_structure(self):
        payload = {
            "nodes": [
                _node("CAP_L1_2_PRODUCT_DEVELOPMENT", "Capability"),
                _node("CAP_L2_2_4_PRODUCT_VERIFICATION", "Capability"),
                _node("CAP_L3_2_4_1_PART_STRUCTURE", "Capability"),
                _node("SYS-001", "System"),
                _node("D-001", "Data"),
                _node("A-001", "Actor"),
                _node("AS-001", "Asset"),
                _node("SC-001", "BusinessScenario"),
                _node("UC-001", "UseCase"),
                _node("C-001", "Concept"),  # unhandled meta_class -> no card
            ],
            "relationships": [
                _rel("SYS-001", "CAP_L1_2_PRODUCT_DEVELOPMENT", "SUPPORTS"),
                _rel("SYS-001", "CAP_L2_2_4_PRODUCT_VERIFICATION", "REALIZES"),
                _rel("D-001", "CAP_L2_2_4_PRODUCT_VERIFICATION", "supports"),
                _rel("CAP_L2_2_4_PRODUCT_VERIFICATION", "A-001", "uses"),
                _rel("AS-001", "D-001", "depends_on"),
                _rel("SC-001", "CAP_L1_2_PRODUCT_DEVELOPMENT", "contains"),
                _rel("UC-001", "SC-001", "extends"),
            ],
        }

        chunks = build_chunks_from_domain_ontology(payload, "master.json")

        by_prefix = {}
        for c in chunks:
            by_prefix.setdefault(c.chunk_id.split("_")[0], []).append(c)

        # Behavior lock: per-type card counts.
        self.assertEqual(len(by_prefix["cap"]), 3)          # 3 capabilities
        self.assertEqual(len(by_prefix["path"]), 2)         # L2 (1 parent) + L3 (2 parents)
        self.assertEqual(len(by_prefix["sysmap"]), 1)       # 1 system
        self.assertEqual(len(by_prefix["data"]), 1)
        self.assertEqual(len(by_prefix["actor"]), 1)
        self.assertEqual(len(by_prefix["asset"]), 1)
        self.assertEqual(len(by_prefix["scenario"]), 2)     # BusinessScenario + UseCase
        self.assertEqual(len(chunks), 11)
        # unhandled Concept produced no card
        self.assertNotIn("C-001", [nid for c in chunks for nid in c.node_ids])

        cap = {c.chunk_id: c for c in by_prefix["cap"]}
        l2 = cap["cap_CAP_L2_2_4_PRODUCT_VERIFICATION"]
        # level/number inference from id; layer field wins when present.
        self.assertEqual(l2.meta["level"], "L2")
        self.assertIn("Number: 2.4", l2.content)
        self.assertEqual(l2.linked_node_ids, ["CAP_L1_2_PRODUCT_DEVELOPMENT"])

        l3 = cap["cap_CAP_L3_2_4_1_PART_STRUCTURE"]
        self.assertEqual(l3.meta["level"], "L3")
        self.assertIn("Number: 2.4.1", l3.content)
        # both parents resolved (unique prefixes) -> 2 linked
        self.assertEqual(
            l3.linked_node_ids,
            ["CAP_L1_2_PRODUCT_DEVELOPMENT", "CAP_L2_2_4_PRODUCT_VERIFICATION"],
        )

        # path cards: only when at least one parent resolved
        path_ids = {c.chunk_id: c for c in by_prefix["path"]}
        p3 = path_ids["path_CAP_L3_2_4_1_PART_STRUCTURE"]
        self.assertIn(
            "Capability-CAP_L1_2_PRODUCT_DEVELOPMENT -> Capability-CAP_L2_2_4_PRODUCT_VERIFICATION -> Capability-CAP_L3_2_4_1_PART_STRUCTURE",
            p3.content,
        )

        # system mapping card collects all sync->cap edges
        sysmap = by_prefix["sysmap"][0]
        self.assertEqual(sysmap.chunk_id, "sysmap_SYS-001")
        self.assertEqual(
            sysmap.linked_node_ids,
            ["CAP_L1_2_PRODUCT_DEVELOPMENT", "CAP_L2_2_4_PRODUCT_VERIFICATION"],
        )
        self.assertEqual(sysmap.meta["edge_types"], ["REALIZES", "SUPPORTS"])
        self.assertIn("Mappings (2):", sysmap.content)

        # generic linked cards: sorted + deduped node ids and relationship types
        data = by_prefix["data"][0]
        self.assertEqual(data.chunk_id, "data_D-001")
        # D-001 is both a source (-> capability) and a target (from asset).
        self.assertEqual(data.linked_node_ids, ["AS-001", "CAP_L2_2_4_PRODUCT_VERIFICATION"])
        self.assertEqual(data.meta["relationship_types"], ["depends_on", "supports"])

        actor = by_prefix["actor"][0]
        self.assertEqual(actor.linked_node_ids, ["CAP_L2_2_4_PRODUCT_VERIFICATION"])
        self.assertEqual(actor.meta["relationship_types"], ["uses"])

        asset = by_prefix["asset"][0]
        self.assertEqual(asset.linked_node_ids, ["D-001"])
        self.assertEqual(asset.meta["relationship_types"], ["depends_on"])

        sc = {c.chunk_id: c for c in by_prefix["scenario"]}
        # SC-001 is both source (-> capability) and target (from UC-001).
        self.assertEqual(sc["scenario_SC-001"].meta["layer"], "L2")
        self.assertEqual(sc["scenario_SC-001"].linked_node_ids, ["CAP_L1_2_PRODUCT_DEVELOPMENT", "UC-001"])
        self.assertEqual(sc["scenario_UC-001"].linked_node_ids, ["SC-001"])

    def test_ambiguous_parent_prefix_suppresses_path_card(self):
        # Two nodes share the CAP_L1_2_ prefix -> parent resolution is ambiguous
        # (2 matches) so no path card may be emitted for the child.
        payload = {
            "nodes": [
                _node("CAP_L1_2_PRODUCT_DEVELOPMENT", "Capability"),
                _node("CAP_L1_2_PRODUCT_VALIDATION", "Capability"),
                _node("CAP_L2_2_4_PRODUCT_VERIFICATION", "Capability"),
            ],
            "relationships": [],
        }
        chunks = build_chunks_from_domain_ontology(payload, "master.json")
        cap = {c.chunk_id: c for c in chunks}
        l2 = cap["cap_CAP_L2_2_4_PRODUCT_VERIFICATION"]
        # Ambiguous -> no parent resolved, but capability card itself still produced.
        self.assertEqual(l2.linked_node_ids, [])
        self.assertNotIn("path_CAP_L2_2_4_PRODUCT_VERIFICATION", cap)
        self.assertEqual(len(chunks), 3)  # 3 capability cards, no path cards

    def test_node_layer_field_overrides_id_level_inference(self):
        payload = {
            "nodes": [
                _node("CAP_L1_2_PRODUCT_DEVELOPMENT", "Capability", layer="L2"),
            ],
            "relationships": [],
        }
        chunks = build_chunks_from_domain_ontology(payload, "master.json")
        self.assertEqual(chunks[0].meta["level"], "L2")  # layer wins over id pattern

    def test_contract_error_and_non_string_rows_are_skipped(self):
        # Missing keys -> DataContractError.
        with self.assertRaises(DataContractError):
            build_chunks_from_domain_ontology({}, "master.json")

        # Non-string relationship fields are silently skipped (no crash).
        payload = {
            "nodes": [_node("D-001", "Data")],
            "relationships": [
                {"source": 123, "target": "D-001", "type": "x"},  # non-str source
                {"source": "D-001", "target": ["list"], "type": "y"},  # non-str target
            ],
        }
        chunks = build_chunks_from_domain_ontology(payload, "master.json")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].linked_node_ids, [])

        # Empty nodes -> no chunks.
        self.assertEqual(build_chunks_from_domain_ontology({"nodes": [], "relationships": []}, "m.json"), [])


class StableChunkIdPolicyTest(unittest.TestCase):
    """RAG-03 stable ID policy: chunk_id is always prefix + node_id, unique per type."""

    def test_make_chunk_id_concatenates_prefix_and_node_id(self):
        self.assertEqual(_make_chunk_id("cap_", "CAP_L1_1_X"), "cap_CAP_L1_1_X")
        self.assertEqual(_make_chunk_id("data_", "D-001"), "data_D-001")

    def test_distinct_prefixes_never_collide_for_same_node(self):
        nid = "NODE-9"
        ids = {
            _make_chunk_id(prefix, nid)
            for prefix in (
                CHUNK_PREFIX_CAP,
                CHUNK_PREFIX_PATH,
                CHUNK_PREFIX_PROC,
                CHUNK_PREFIX_SYSMAP,
                CHUNK_PREFIX_DATA,
                CHUNK_PREFIX_ACTOR,
                CHUNK_PREFIX_ASSET,
                CHUNK_PREFIX_SCENARIO,
            )
        }
        self.assertEqual(len(ids), 8)  # all distinct even for the same node id

    def test_builders_use_stable_id_policy(self):
        # Capability/sysmap/data/scenario cards must all produce ids through the
        # same prefix+node_id rule (no ad-hoc id shapes). NOTE: path cards are
        # NOT generated for every capability (additional trigger conditions),
        # so they are not asserted here.
        payload = {
            "nodes": [
                _node("CAP_L1_2_PRODUCT_DEVELOPMENT", "Capability"),
                _node("SYS-001", "System"),
                _node("D-001", "Data"),
                _node("SCN-1", "Scenario"),
            ],
            "relationships": [
                {"source": "SYS-001", "target": "CAP_L1_2_PRODUCT_DEVELOPMENT", "type": "SUPPORTS"},
            ],
        }
        chunks = build_chunks_from_domain_ontology(payload, "master.json")
        ids = [c.chunk_id for c in chunks]
        expected = {
            _make_chunk_id(CHUNK_PREFIX_CAP, "CAP_L1_2_PRODUCT_DEVELOPMENT"),
            _make_chunk_id(CHUNK_PREFIX_SYSMAP, "SYS-001"),
            _make_chunk_id(CHUNK_PREFIX_DATA, "D-001"),
            _make_chunk_id(CHUNK_PREFIX_SCENARIO, "SCN-1"),
        }
        self.assertTrue(expected.issubset(set(ids)))


if __name__ == "__main__":
    unittest.main()