"""Behavioral tests for llm_response_parser and entity_normalizer.

These tests pin down the exact behavior of the pure functions extracted from
``true_unified_extractor`` so future refactors can rely on a regression net.
"""

import asyncio
import json
import unittest
from unittest import mock

from ontology_compiler.services.llm_response_parser import (
    clean_llm_response,
    parse_unified_response,
    repair_json,
    extract_json_from_text,
)
from ontology_compiler.services.entity_normalizer import normalize_entities
from ontology_compiler.services.extraction_postprocessing import (
    is_noise_entity,
    normalize_evidence,
    normalize_entity_type,
    infer_l1_subclass,
    normalize_relationship,
    correct_meta_class,
    resolve_entity_id,
    enforce_l3_parents,
    L1_ALLOWED_ENUMERATION,
)
from ontology_compiler.services.extraction_steps import (
    reclassify_l1_to_l2,
    reclassify_l2_to_l1,
    apply_layer_quality_filter,
)


def _infer_class(name):
    return "System"


def _assign_ids(entities, prefix):
    for i, entity in enumerate(entities):
        if not entity.get("meta_id"):
            entity["meta_id"] = f"{prefix}-{i + 1:03d}"
    return entities


def _parse(response):
    return parse_unified_response(
        response,
        infer_entity_class=_infer_class,
        assign_meta_ids=_assign_ids,
    )


class TestCleanLlmResponse(unittest.TestCase):
    def test_strips_think_and_text_prefix(self):
        raw = 'text=```json\n{"k": 1}\n```'
        cleaned = clean_llm_response(raw)
        self.assertEqual(cleaned, '{"k": 1}')
        raw2 = ' response thinking\n{"a": 1}\n response'
        # think-tag stripping only matches `` response`` markers, so this stays intact
        self.assertIn('{"a": 1}', clean_llm_response(raw2))


class TestParseUnifiedResponse(unittest.TestCase):
    def test_parses_all_layers_and_relationships(self):
        payload = {
            "L0_entities": [{"meta_id": "L0-001", "name": "System", "meta_class": "Capability"}],
            "L1_entities": [{"meta_id": "L1-001", "name": "teamcenter", "meta_class": "System", "evidence": {"c": 1}}],
            "L2_entities": [{"meta_id": "L2-001", "name": "Warehouse", "meta_class": "Actor"}],
            "L2_5_entities": [{"meta_id": "L25-001", "name": "Variant Rule", "meta_class": "Data"}],
            "L3_entities": [{"meta_id": "L3-001", "name": "Part-1", "meta_class": "Asset"}],
            "relationships": [
                {"from": "L1-001", "to": "L2-001", "type": "CONTAINS"},
            ],
        }
        l0, l1, l2, l25, l3, rels = _parse(json.dumps(payload))
        self.assertEqual(len(l0), 1)
        self.assertEqual(len(l1), 1)
        self.assertEqual(len(l2), 1)
        self.assertEqual(len(l25), 1)
        self.assertEqual(len(l3), 1)
        self.assertEqual(len(rels), 1)
        self.assertEqual(l1[0]["parent_class"], "Capability")
        self.assertEqual(l2[0]["scenario_name"], "Warehouse")
        self.assertEqual(rels[0]["confidence"], 0.7)

    def test_handles_entities_wrapper_and_lowercase_keys(self):
        payload = {
            "entities": {
                "L1": [{"name": "SAP", "meta_class": "System"}],
                "L2": [{"name": "Plant", "meta_class": "Asset"}],
            },
            "relationships": [],
        }
        l0, l1, l2, l25, l3, rels = _parse(json.dumps(payload))
        self.assertEqual(len(l1), 1)
        self.assertEqual(len(l2), 1)
        self.assertEqual(l1[0]["meta_id"], "L1-001")

    def test_infers_missing_fields(self):
        payload = {
            "L2_entities": [{"name": "Order Handling", "meta_class": "Process"}],
            "relationships": [{"from": "", "to": "L2-001", "type": "CONTAINS"}],
        }
        l0, l1, l2, l25, l3, rels = _parse(json.dumps(payload))
        self.assertEqual(len(l2), 1)
        entity = l2[0]
        self.assertIn("evidence", entity)
        self.assertEqual(rels[0]["confidence"], 0.7)

    def test_non_json_response_degrades_gracefully(self):
        l0, l1, l2, l25, l3, rels = _parse("this is not json at all")
        self.assertEqual(l0, [])
        self.assertEqual(l1, [])
        self.assertEqual(l2, [])
        self.assertEqual(l25, [])
        self.assertEqual(l3, [])
        self.assertEqual(rels, [])


class TestRepairJson(unittest.TestCase):
    def test_single_quotes_and_missing_commas(self):
        raw = "{'L1_entities': [{'meta_id': 'L1-001' 'name': 'X'}]}"
        result = repair_json(raw)
        self.assertIn("L1_entities", result)

    def test_truncated_array_recovers_elements(self):
        raw = '{"L2_entities": [{"meta_id": "L2-001", "name": "A"}, {"meta_id": "L2-002", "name": "B"}'
        result = repair_json(raw)
        self.assertIsInstance(result.get("L2_entities"), list)
        self.assertTrue(len(result.get("L2_entities", [])) >= 1)

    def test_json_fence_and_think_tags(self):
        raw = " response thinking\n```json\n{\"L1_entities\": [], \"relationships\": []}\n````"
        result = repair_json(raw)
        self.assertEqual(result.get("L1_entities"), [])

    def test_returns_empty_structure_on_unrecoverable(self):
        result = repair_json("### totally garbled ###")
        self.assertEqual(result["L1_entities"], [])
        self.assertEqual(result["relationships"], [])


class TestExtractionPostprocessing(unittest.TestCase):
    def test_noise_entity_filter(self):
        self.assertTrue(is_noise_entity({"name": "Slide 26"}))
        self.assertTrue(is_noise_entity({"name": "20241711"}))
        self.assertTrue(is_noise_entity({"name": "overview"}))
        self.assertFalse(is_noise_entity({"name": "Teamcenter"}))

    def test_entity_type_override(self):
        entity = {"name": "MBOM", "meta_class": "Process"}
        normalized = normalize_entity_type(entity, "L1")
        self.assertEqual(normalized["meta_class"], "Data")
        self.assertEqual(normalized["meta_subclass"], "Document_Artifact")

    def test_l1_subclass_inference(self):
        entity = {"name": "Teamcenter", "parent_class": "System"}
        self.assertEqual(infer_l1_subclass(entity), "Enterprise_Platform")
        entity2 = {"name": "Portal X", "parent_class": "System"}
        self.assertEqual(infer_l1_subclass(entity2), "External_Portal")

    def test_normalize_evidence_shapes(self):
        self.assertEqual(normalize_evidence({"a": 1}), {"a": 1})
        self.assertEqual(normalize_evidence("src text"), {"source_text": "src text", "confidence": 0.7})
        self.assertEqual(normalize_evidence(None), {"confidence": 0.0})

    def test_relationship_normalization(self):
        rel = {"source_id": "A", "target_id": "B", "relation_type": "contains"}
        normalized = normalize_relationship(rel)
        self.assertEqual(normalized["from"], "A")
        self.assertEqual(normalized["to"], "B")
        self.assertEqual(normalized["type"], "CONTAINS")
        self.assertEqual(normalized["confidence"], 0.7)

    def test_unknown_relation_falls_back_to_related_to(self):
        rel = {"from": "A", "to": "B", "type": "unknown"}
        normalized = normalize_relationship(rel)
        self.assertEqual(normalized["type"], "RELATED_TO")

    def test_correct_meta_class(self):
        self.assertEqual(correct_meta_class("Wheel Unit", "Data"), "Asset")
        self.assertEqual(correct_meta_class("SAP System", "Process"), "System")
        self.assertEqual(correct_meta_class("Order Handling Process", "Capability"), "Process")
        self.assertEqual(correct_meta_class("Quality Department", "System"), "Actor")
        self.assertEqual(correct_meta_class("Cycle Time Calculator", "Capability"), "System")
        self.assertEqual(correct_meta_class("Variant Rule", "Capability"), "BusinessRule")
        self.assertEqual(correct_meta_class("Generic Name", "Capability"), "Capability")

    def test_resolve_entity_id(self):
        valid = {"L1-001", "L2-002", "L3-001", "L2-010"}
        id_map = {"SC-ENT-L1-001": "L1-001", "SC_L2_10": "L2-010"}
        name_map = {("L3", "wheel"): "L3-001", "wheel": "L3-001"}
        insts = [{"name": "Wheel", "meta_id": "L3-001"}]
        resolve = lambda rid, rn="": resolve_entity_id(rid, rn, valid, id_map, name_map, insts)
        self.assertEqual(resolve("L1-001"), "L1-001")
        self.assertEqual(resolve("SC-ENT-L1-001"), "L1-001")
        self.assertEqual(resolve("SC_L2_10"), "L2-010")
        self.assertEqual(resolve("L3-xyz", "Wheel"), "L3-001")
        self.assertEqual(resolve("unknown-id"), "unknown-id")

    def test_enforce_l3_parents(self):
        nodes = [
            {"name": "Warehouse", "layer": "L2", "meta_id": "L2-001", "meta_class": "Actor"},
            {"name": "Rack A", "layer": "L3", "meta_id": "L3-001", "meta_class": "Asset"},
        ]
        enforce_l3_parents(nodes, [{"from": "L2-001", "to": "L3-001"}])
        self.assertEqual(nodes[1]["parent_id"], "L2-001")

    def test_quality_filter_enumeration_present(self):
        self.assertIn("costmanagement", L1_ALLOWED_ENUMERATION)
        self.assertIn("part_asset", L1_ALLOWED_ENUMERATION)


class TestExtractionSteps(unittest.TestCase):
    def _state(self):
        return {
            "nodes": [
                {"meta_id": "L1-001", "name": "MBOM", "meta_class": "Data", "layer": "L1", "parent_class": "Data"},
                {"meta_id": "L1-002", "name": "SAP", "meta_class": "System", "layer": "L1", "parent_class": "System"},
                {"meta_id": "L1-003", "name": "Process", "meta_class": "Process", "layer": "L1"},
                {"meta_id": "L2-001", "name": "ChangeManagement", "meta_class": "Process", "layer": "L2"},
                {"meta_id": "L2-002", "name": "TARR", "meta_class": "System", "layer": "L2"},
                {"meta_id": "L2-003", "name": "TARR", "meta_class": "Process", "layer": "L2"},
                {"meta_id": "L2-004", "name": "Order Handling Table", "meta_class": "Data", "layer": "L2"},
            ],
            "seen_ids": {},
            "counter": {"L0": 1, "L1": 10, "L2": 10, "L2.5": 1, "L3": 1},
            "id_mapping": {},
            "l0_class_to_id": {"system": "L0-001", "data": "L0-002"},
            "valid": {"Capability", "System", "Actor", "Asset", "Data", "Process", "Concept"},
        }

    def test_reclassify_l1_to_l2(self):
        st = self._state()
        reclassify_l1_to_l2(st["nodes"], st["seen_ids"], st["counter"], st["id_mapping"],
                            st["l0_class_to_id"], st["valid"])
        by_name = {n["name"]: n for n in st["nodes"]}
        self.assertEqual(by_name["MBOM"]["layer"], "L1")     # retention
        self.assertEqual(by_name["Process"]["layer"], "L1")  # standalone stays
        sap = by_name["SAP"]
        self.assertEqual(sap["layer"], "L2")                 # moved to L2
        self.assertIn("L1-002", st["id_mapping"])            # mapping recorded
        intermediates = [n for n in st["nodes"] if n.get("name") == "System" and n.get("layer") == "L1"]
        self.assertTrue(intermediates)
        self.assertEqual(sap["parent_id"], intermediates[0]["meta_id"])

    def test_reclassify_l2_to_l1(self):
        st = self._state()
        reclassify_l2_to_l1(st["nodes"], st["counter"], st["id_mapping"])
        by_name = {n["name"]: n for n in st["nodes"]}
        self.assertEqual(by_name["ChangeManagement"]["layer"], "L1")  # abstract -> L1
        self.assertEqual(by_name["TARR"]["layer"], "L2")              # concrete stays
        self.assertIn("L2-001", st["id_mapping"])

    def test_apply_layer_quality_filter(self):
        st = self._state()
        rels = [{"from": "L2-002", "to": "L2-003", "type": "CONTAINS"},
                {"from": "L1-003", "to": "L2-004", "type": "CONTAINS"}]
        nodes = apply_layer_quality_filter(st["nodes"], st["counter"], st["id_mapping"], rels)
        ids = {n["meta_id"] for n in nodes}
        self.assertNotIn("L2-004", ids)                    # table downgraded to L3
        l3 = [n for n in nodes if n.get("layer") == "L3"]
        self.assertTrue(l3)
        # TARR disambiguation
        tarrs = sorted(n.get("name") for n in nodes if n.get("name", "").startswith("TARR"))
        self.assertEqual(tarrs, ["TARR_Process", "TARR_System"])


class TestExtractJsonFromText(unittest.TestCase):
    def test_valid_markdown_json(self):
        result = extract_json_from_text("```json\n{\"L1_entities\": [{\"meta_id\": \"L1-001\"}]}\n```")
        self.assertEqual(len(result["L1_entities"]), 1)


class TestNormalizeEntities(unittest.TestCase):
    def test_status_and_layer_defaults(self):
        l0 = [{"meta_id": "L0-001", "name": "System"}]
        normalized = normalize_entities(l0, [], [], [], [])
        self.assertEqual(normalized[0][0]["status"], "active")
        self.assertEqual(normalized[0][0]["layer"], "L0")

    def test_name_camelization_and_acronym_preservation(self):
        l1 = [{"meta_id": "L1-001", "name": "SAP s4hana", "meta_class": "System", "evidence": {}}]
        normalized = normalize_entities([], l1, [], [], [])
        self.assertEqual(normalized[1][0]["name"], "SAP S4hana")

    def test_attribute_shape_normalization(self):
        l2 = [{
            "meta_id": "L2-001",
            "name": "Warehouse",
            "meta_class": "Actor",
            "evidence": {},
            "attributes": [{"name": "a", "value": "1"}, "b=2", "plain"],
        }]
        normalized = normalize_entities([], [], l2, [], [])
        attrs = normalized[2][0]["attributes"]
        self.assertEqual(len(attrs), 3)
        self.assertEqual(attrs[1], {"name": "b", "value": "2"})

    def test_invalid_meta_id_gets_layer_prefix(self):
        l1 = [{"meta_id": "SC-ENT-001", "name": "X", "meta_class": "System", "evidence": {}}]
        normalized = normalize_entities([], l1, [], [], [])
        self.assertEqual(normalized[1][0]["meta_id"], "L1-SC-ENT-001")

    def test_nested_definitions_stripped(self):
        l1 = [{
            "meta_id": "L1-001",
            "name": "Teamcenter",
            "meta_class": "System",
            "evidence": {},
            "attributes": dict,
        }]
        # non-list attribute values must be collapsed and must not throw
        normalized = normalize_entities([], l1, [], [], [])
        self.assertEqual(normalized[1][0]["attributes"], [])


class TestExtractAsyncContentSlowPath(unittest.IsolatedAsyncioTestCase):
    """Drives extract_async_content through its full post-processing pipeline
    with a canned LLM result, guarding against NameError regressions from the
    split (e.g. enhance_entity, normalized_relationships)."""

    def _canned(self):
        from ontology_compiler.services.true_unified_extractor import UnifiedExtractionResult
        return UnifiedExtractionResult(
            l0_entities=[
                {"meta_id": "L0-001", "name": "System", "meta_class": "Capability", "evidence": {"content": "c"}},
            ],
            l1_entities=[
                {"meta_id": "L1-001", "name": "MBOM", "meta_class": "Data", "evidence": {"content": "c"}},
                {"meta_id": "L1-002", "name": "SAP", "meta_class": "System", "evidence": {"content": "c"}},
            ],
            l2_entities=[
                {"meta_id": "L2-001", "name": "ChangeManagement", "meta_class": "Process", "evidence": {"content": "c"}},
                {"meta_id": "L2-002", "name": "Warehouse", "meta_class": "Actor", "evidence": {"content": "c"}},
            ],
            l25_entities=[
                {"meta_id": "L25-001", "name": "Variant Rule", "meta_class": "Data", "evidence": {"content": "c"}},
            ],
            l3_entities=[
                {"meta_id": "L3-001", "name": "XC60Wheel", "meta_class": "Asset", "evidence": {"content": "c"}},
            ],
            relationships=[
                {"from": "L1-002", "to": "L2-002", "relation_type": "contains"},
            ],
            raw_response="{}",
            processing_time_ms=1.0,
            token_usage={"prompt": 1, "completion": 1},
        )

    async def test_full_postprocessing_pipeline(self):
        from ontology_compiler.services.true_unified_extractor import TrueUnifiedExtractor

        class _Fake(TrueUnifiedExtractor):
            def __init__(self, result):
                super().__init__()
                self._canned = result

            async def extract_unified(self, **kwargs):
                return self._canned

        ex = _Fake(self._canned())
        result = await ex.extract_async_content(
            content="dummy", domain="SUPPLY_CHAIN", source_file="test.md"
        )
        nodes = result.get("nodes") or []
        insts = result.get("instances") or []
        rels = result.get("relationships") or []
        self.assertGreaterEqual(len(nodes), 1)
        ids = [n.get("meta_id") for n in nodes + insts]
        self.assertEqual(len(ids), len(set(ids)), "duplicate meta_ids in output")
        self.assertTrue(all(n.get("meta_id") for n in nodes + insts))
        self.assertGreaterEqual(len(rels), 1)


class TestRelationshipAndHardFixStages(unittest.TestCase):
    """Unit coverage for the pipeline stages extracted in the 4th split:
    relationship filtering, L1->L2 decoupling, and hard fixes (ID collision /
    ghost nodes / type correction / ECN injection / time-dimension)."""

    def _ctx(self):
        from ontology_compiler.services.extraction_steps import (
            normalize_and_filter_relationships,
            decouple_l1_to_l2_relationships,
            apply_hard_fixes,
        )
        nodes = [
            {"meta_id": "L1-001", "name": "SupplyChain_System", "meta_class": "System", "layer": "L1"},
            {"meta_id": "L2-001", "name": "SAP", "meta_class": "System", "layer": "L2", "parent_id": "L1-001"},
            {"meta_id": "L2-002", "name": "MES", "meta_class": "System", "layer": "L2", "parent_id": "L1-001"},
            {"meta_id": "L2-003", "name": "KDP", "meta_class": "System", "layer": "L2", "parent_id": "L1-001"},
            {"meta_id": "L3-001", "name": "XC60Wheel", "meta_class": "Asset", "layer": "L3", "parent_id": "L2-001"},
        ]
        return {
            "nodes": nodes,
            "instances": [{"meta_id": "L3-099", "name": "Seat", "meta_class": "Asset", "layer": "L3", "parent_id": "L2-001"}],
            "valid": {n["meta_id"] for n in nodes} | {"L3-099"},
            "id_mapping": {},
            "name_map": {"sap": "L2-001", "mes": "L2-002"},
            "entity_map": {n["meta_id"]: n for n in nodes},
            "layer_map": {n["meta_id"]: n.get("layer", "") for n in nodes},
            "rels": [
                {"from": "L1-001", "to": "L2-001", "relation_type": "contains"},
                {"from": "L1-001", "to": "L2-002", "relation_type": "contains"},
                {"from": "L1-001", "to": "L2-001", "relation_type": "uses"},   # L1->L2 ratio 3/3 > 25%
                {"from": "GHOST-1", "to": "L2-001", "relation_type": "contains"},  # invalid from -> filtered
                {"from": "L1-001", "to": "L1-001", "relation_type": "contains"},   # self-reference -> filtered
            ],
        }

    def test_normalize_and_filter_relationships(self):
        from ontology_compiler.services.extraction_steps import normalize_and_filter_relationships

        class _R:
            relationships = None

        r = _R()
        c = self._ctx()
        r.relationships = c["rels"]
        out = normalize_and_filter_relationships(
            r, c["nodes"], c["instances"], c["valid"],
            c["id_mapping"], c["name_map"], c["entity_map"], c["layer_map"],
        )
        self.assertIsInstance(out, list)
        ids = {rel.get("from") for rel in out}
        self.assertNotIn("GHOST-1", ids)          # invalid endpoint dropped
        self.assertFalse(any(rel["from"] == rel["to"] for rel in out))  # self-refs dropped
        # 3 L1->L2 rels survive (they are not dropped by the filter)
        l1_l2 = [rel for rel in out if rel.get("from") == "L1-001"]
        self.assertEqual(len(l1_l2), 3)

    def test_decouple_l1_to_l2_relationships(self):
        from ontology_compiler.services.extraction_steps import decouple_l1_to_l2_relationships
        c = self._ctx()
        rels = [
            {"from": "L1-001", "to": "L2-001", "relation_type": "contains"},
            {"from": "L1-001", "to": "L2-002", "relation_type": "contains"},
            {"from": "L1-001", "to": "L2-001", "relation_type": "uses"},
        ]
        decouple_l1_to_l2_relationships(rels, c["nodes"], c["layer_map"], c["valid"], c["entity_map"])
        # excess L1->L2 rels should have been rerouted to L2 (proxy or policy)
        remaining_l1_l2 = [rel for rel in rels if rel.get("from") == "L1-001"]
        self.assertLessEqual(len(remaining_l1_l2), 1)
        # all rerouted targets must be valid L2 ids
        for rel in rels:
            self.assertIn(rel["from"], c["valid"], rel)

    def test_apply_hard_fixes_injects_ecn_and_fixes_collisions(self):
        from ontology_compiler.services.extraction_steps import apply_hard_fixes
        c = self._ctx()
        # introduce an ID collision
        c["nodes"].append(dict(c["nodes"][0]))
        c["valid"].add(c["nodes"][-1]["meta_id"])
        rels = list(c["rels"][:2])
        apply_hard_fixes(c["nodes"], c["instances"], rels, c["entity_map"],
                         c["layer_map"], c["valid"], c["id_mapping"])
        ids = [n["meta_id"] for n in c["nodes"]]
        self.assertEqual(len(ids), len(set(ids)), "ID collision not resolved")
        names_lower = {n["name"].lower() for n in c["nodes"]}
        self.assertTrue(any("bom" in name for name in names_lower), "ECN backbone not injected")


class TestEntityAssemblyStages(unittest.TestCase):
    """Unit coverage for the entity-assembly stages extracted in the 5th
    split: enhance_entity, L0 validation, dedup, index builders, and the
    parent_id auto-generation / repair passes."""

    def test_enhance_entity_tags_layer_and_source_file(self):
        from ontology_compiler.services.extraction_steps import enhance_entity
        out = enhance_entity({"name": "X"}, "L1", "f.md")
        self.assertEqual(out["layer"], "L1")
        self.assertEqual(out["source_file"], "f.md")
        # L1 without parent_class defaults to meta_class
        self.assertEqual(out["parent_class"], "System")
        # evidence normalization is applied
        out2 = enhance_entity({"name": "Y", "evidence": "raw"}, "L2", "f.md")
        self.assertIsInstance(out2["evidence"], dict)

    def test_validate_l0_entities_demotes_non_classes(self):
        from ontology_compiler.services.extraction_steps import validate_l0_entities
        valid = {"Capability", "System", "Actor", "Asset", "Data", "Process", "Concept"}
        entities = [
            {"meta_id": "L0-001", "name": "system", "meta_class": "Capability"},
            {"meta_id": "L0-002", "name": "KDP", "meta_class": "System"},       # domain concept -> L2
            {"meta_id": "L0-003", "name": "NothingHere", "meta_class": "Meta"},  # invalid -> L2
        ]
        out = validate_l0_entities(entities, valid, "f.md")
        self.assertEqual(out[0]["layer"], "L0")
        self.assertEqual(out[0]["name"], "System")
        self.assertEqual(out[1]["layer"], "L2")
        self.assertTrue(out[1]["meta_id"].startswith("L2-"))
        self.assertEqual(out[2]["layer"], "L2")

    def test_deduplicate_nodes_filters_noise_and_merges(self):
        from ontology_compiler.services.extraction_steps import deduplicate_nodes
        counter = {"L0": 1, "L1": 1, "L2": 1, "L2.5": 1, "L3": 1}
        id_mapping, seen_ids = {}, {}
        entities = [
            {"meta_id": "L1-001", "name": "MBOM", "meta_class": "Data", "layer": "L1"},
            {"meta_id": "L1-002", "name": "MBOM", "meta_class": "Data", "layer": "L1",
             "attributes": [{"name": "a", "value": "1"}], "description": "dup"},
            {"meta_id": "L1-003", "name": "Introduction", "meta_class": "Data", "layer": "L1"},  # noise
        ]
        out = deduplicate_nodes(entities, counter, id_mapping, seen_ids)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["meta_id"], "L1-001")
        self.assertEqual(out[0]["attributes"], [{"name": "a", "value": "1"}])
        self.assertEqual(out[0]["description"], "dup")
        self.assertEqual(id_mapping.get("L1-002"), "L1-001")
        self.assertEqual(id_mapping.get("L1-003"), "")  # noise recorded as empty

    def test_parent_autogenerate_and_fix(self):
        from ontology_compiler.services.extraction_steps import (
            autogenerate_parent_classes,
            autogenerate_parent_ids,
            build_entity_maps,
            build_name_to_id_map,
            apply_parent_id_fixes,
        )
        nodes = [
            {"meta_id": "L0-001", "name": "System", "layer": "L0"},
            {"meta_id": "L1-001", "name": "SAP", "layer": "L1", "parent_class": "system"},
            {"meta_id": "L2-001", "name": "MES", "layer": "L2", "parent_id": "L1-999"},
        ]
        _, _, l0_class_to_id = build_entity_maps(nodes, [])
        autogenerate_parent_ids(nodes, l0_class_to_id)
        self.assertEqual(nodes[1]["parent_id"], "L0-001")   # L1 references L0
        autogenerate_parent_classes(nodes)
        self.assertEqual(nodes[2]["parent_class"], "BusinessScenario")   # default when no meta_class
        name_map = build_name_to_id_map(nodes, [])
        self.assertEqual(name_map[("L2", "mes")], "L2-001")
        apply_parent_id_fixes(nodes, [], {"L0-001", "L1-001", "L2-001"}, {}, name_map, [])
        self.assertNotIn("parent_id", nodes[2])             # unresolvable parent dropped


class TestLlmCallService(unittest.TestCase):
    """Unit coverage for the LLM call machinery extracted in the 6th split
    (ontology_compiler.services.llm_call)."""

    def _valid(self):
        import json
        return json.dumps({
            "L0_entities": [{"name": "System"}], "L1_entities": [], "L2_entities": [],
            "L2_5_entities": [], "L3_entities": [], "relationships": [],
        }, ensure_ascii=False)

    def test_validate_json_response_accepts_full_schema(self):
        from ontology_compiler.services.llm_call import validate_json_response
        r = validate_json_response(self._valid())
        self.assertTrue(r["valid"])
        r2 = validate_json_response("not json")
        self.assertFalse(r2["valid"])
        r3 = validate_json_response('{"L0_entities": "x"}')
        self.assertFalse(r3["valid"])

    def test_build_retry_prompt_appends_corrections(self):
        from ontology_compiler.services.llm_call import build_retry_prompt
        out = build_retry_prompt("original", "bad response", "JSON parse error")
        self.assertIn("original", out)
        self.assertIn("CORRECTION", out)
        self.assertIn("bad response", out)

    async def test_call_llm_happy_path(self):
        from ontology_compiler.services.llm_call import call_llm

        class FakeLLM:
            model = "fake"
            async def call_with_retry(self, prompt, **kw):
                return self._valid

        llm = FakeLLM()
        llm._valid = self._valid()
        r = await call_llm(llm, 100, 0.2, 5, "", None, lambda: type("M", (), {"name": "s"})(), "p")
        self.assertTrue(r["parse_success"])
        self.assertEqual(r["retry_count"], 0)
        self.assertNotIn("offline_fallback", r)

    async def test_call_llm_offline_fallback(self):
        import os
        from ontology_compiler.services.llm_call import call_llm
        os.environ["LLM_OFFLINE_MODE"] = "1"
        try:
            r = await call_llm(None, 100, 0.2, 5, "", None, lambda: None, "p")
            self.assertTrue(r["offline_fallback"])
            self.assertTrue(r["parse_success"])
        finally:
            os.environ.pop("LLM_OFFLINE_MODE", None)


class TestExtractFileTimeoutConfig(unittest.TestCase):
    """Regression (2026-08-15): VSM/ATACQ large-document extraction died at
    the hardcoded 600s file ceiling when smart-chunking + JSON retries took
    longer. The ceiling is now configurable via EXTRACT_FILE_TIMEOUT."""

    def _patch_extract_timeout(self, value):
        import os
        saved = os.environ.get("EXTRACT_FILE_TIMEOUT")
        if value is None:
            os.environ.pop("EXTRACT_FILE_TIMEOUT", None)
        else:
            os.environ["EXTRACT_FILE_TIMEOUT"] = value
        return saved

    def test_default_ceiling_is_300(self):
        import os
        from src.pipeline.stages.extract import _file_timeout_seconds
        saved = self._patch_extract_timeout(None)
        try:
            self.assertEqual(_file_timeout_seconds(), 300)
        finally:
            self._patch_extract_timeout(saved)

    def test_env_override_applied(self):
        from src.pipeline.stages.extract import _file_timeout_seconds
        saved = self._patch_extract_timeout("1200")
        try:
            self.assertEqual(_file_timeout_seconds(), 1200)
        finally:
            self._patch_extract_timeout(saved)

    def test_floor_prevents_too_short_ceiling(self):
        from src.pipeline.stages.extract import _file_timeout_seconds
        saved = self._patch_extract_timeout("10")
        try:
            self.assertEqual(_file_timeout_seconds(), 300)
        finally:
            self._patch_extract_timeout(saved)

    def test_invalid_value_falls_back_to_300(self):
        from src.pipeline.stages.extract import _file_timeout_seconds
        saved = self._patch_extract_timeout("abc")
        try:
            self.assertEqual(_file_timeout_seconds(), 300)
        finally:
            self._patch_extract_timeout(saved)


class TestTestExtractChunkRetryBudget(unittest.TestCase):
    """Regression (2026-08-15): process_chunk hardcoded ``max_retries = 3``,
    which combined with llm_call's internal JSON retries could spend 12+ API
    round-trips on a single bad chunk and blow past the whole-file timeout.
    The chunk-level retry budget is now bounded by EXTRACT_CHUNK_MAX_RETRIES."""

    @staticmethod
    def _max_chunk_retries(value):
        import os
        saved = os.environ.get("EXTRACT_CHUNK_MAX_RETRIES")
        if value is None:
            os.environ.pop("EXTRACT_CHUNK_MAX_RETRIES", None)
        else:
            os.environ["EXTRACT_CHUNK_MAX_RETRIES"] = value
        return saved

    def test_default_chunk_retries_is_1(self):
        import asyncio
        from ontology_compiler.services.chunking_service import extract_with_chunking
        from ontology_compiler.services.extraction_result import UnifiedExtractionResult

        class _Chunker:
            def __init__(self):
                self.config = {}

            def smart_chunk(self, text, source_file="", strategy="auto"):
                return [{"content": "boom", "section": "S1"}]

        calls = {"n": 0}

        async def boom_call_llm(prompt, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("transient")
            return {"response": '{"L1_entities":[{"meta_id":"L1-001","name":"SAP","meta_class":"System"}],"relationships":[]}'}

        def parse_response(text):
            return [], [{"meta_id": "L1-001", "name": "SAP", "meta_class": "System"}], [], [], [], []

        async def extract_relationships(text, entities, domain):
            return []

        saved = self._max_chunk_retries(None)
        try:
            result = asyncio.run(extract_with_chunking(
                text="boom", file_path=None, domain="SUPPLY_CHAIN", use_context=True,
                enable_validation=False, max_retries=1, document_metadata=None,
                chunker=_Chunker(), build_prompt=lambda *a, **k: "p",
                call_llm=boom_call_llm, parse_response=parse_response,
                extract_relationships=extract_relationships,
                result_cls=UnifiedExtractionResult,
                cache_enabled=False,
            ))
        finally:
            self._max_chunk_retries(saved)

        self.assertEqual(calls["n"], 2)   # default budget 1 -> at most 2 attempts

    def test_env_lower_budget_aborts_sooner(self):
        import asyncio
        from ontology_compiler.services.chunking_service import extract_with_chunking
        from ontology_compiler.services.extraction_result import UnifiedExtractionResult

        class _Chunker:
            def __init__(self):
                self.config = {}

            def smart_chunk(self, text, source_file="", strategy="auto"):
                return [{"content": "boom", "section": "S1"}]

        calls = {"n": 0}

        async def boom_call_llm(prompt, **kwargs):
            calls["n"] += 1
            raise RuntimeError("persistent")

        saved = self._max_chunk_retries("0")   # no retries: single attempt only
        try:
            asyncio.run(extract_with_chunking(
                text="boom", file_path=None, domain="SUPPLY_CHAIN", use_context=True,
                enable_validation=False, max_retries=1, document_metadata=None,
                chunker=_Chunker(), build_prompt=lambda *a, **k: "p",
                call_llm=boom_call_llm,
                parse_response=lambda t: ([], [], [], [], [], []),
                extract_relationships=lambda *a, **k: [],
                result_cls=UnifiedExtractionResult,
                cache_enabled=False,
            ))
        finally:
            self._max_chunk_retries(saved)

        self.assertEqual(calls["n"], 1)


class TestChunkOutputBudget(unittest.TestCase):
    """Regression (2026-08-15): chunked extraction must not hand the LLM the
    full 32K output budget — a 6834-char chunk once produced a ~29K-token
    non-JSON narrative. Budget now scales with chunk size."""

    def test_budget_scales_with_chunk_size(self):
        from ontology_compiler.services.chunking_service import (
            _chunk_output_budget,
            _CHUNK_MAX_OUTPUT_TOKENS_ENV,
        )
        self.assertEqual(_chunk_output_budget(0), 4096)
        self.assertEqual(_chunk_output_budget(100), 4096)
        self.assertEqual(_chunk_output_budget(6000), 6000)
        # Scales with input, capped by the env-configured ceiling
        # (CHUNK_MAX_OUTPUT_TOKENS, e.g. 14000 when config/env.env is active).
        ceiling = _CHUNK_MAX_OUTPUT_TOKENS_ENV or 32000
        self.assertEqual(_chunk_output_budget(20000), min(20000, ceiling))
        self.assertEqual(_chunk_output_budget(40000), min(32000, ceiling))   # capped

    def test_max_tokens_override_flows_to_call_llm(self):
        """process_chunk must pass a scaled max_tokens to the LLM callback;
        an extractor stub that records it proves the override is forwarded."""
        import asyncio
        from ontology_compiler.services.chunking_service import extract_with_chunking
        from ontology_compiler.services.extraction_result import UnifiedExtractionResult

        seen = {}

        class _Chunker:
            def __init__(self):
                self.config = {}

            def smart_chunk(self, text, source_file="", strategy="auto"):
                return [{"content": "x" * 8000, "section": "S1"}]

        async def recording_call_llm(prompt, **kwargs):
            seen["max_tokens"] = kwargs.get("max_tokens")
            return {"response": '{"L1_entities":[{"meta_id":"L1-001","name":"SAP","meta_class":"System"}],"relationships":[]}'}

        def parse_response(text):
            return [], [{"meta_id": "L1-001", "name": "SAP", "meta_class": "System"}], [], [], [], []

        asyncio.run(extract_with_chunking(
            text="x" * 8000, file_path=None, domain="SUPPLY_CHAIN", use_context=True,
            enable_validation=False, max_retries=1, document_metadata=None,
            chunker=_Chunker(), build_prompt=lambda *a, **k: "p",
            call_llm=recording_call_llm, parse_response=parse_response,
            extract_relationships=lambda *a, **k: [],
            result_cls=UnifiedExtractionResult, cache_enabled=False,
        ))
        self.assertLess(seen.get("max_tokens"), 32000)


class TestLlmCacheVersioning(unittest.TestCase):
    """D-2 regression: LLM cache keys must include a version bundle so a
    prompt/schema/model change does not silently hit a stale cached response."""

    @staticmethod
    def _cache():
        from ontology_compiler.rigor.llm_backend import LLMCache
        return LLMCache()

    def test_different_bundles_produce_different_keys(self):
        from ontology_compiler.rigor.llm_backend import LLMCache
        cache_a = LLMCache(version_bundle="v1")
        cache_b = LLMCache(version_bundle="v2")
        prompt = "extract entities from this text"
        self.assertNotEqual(cache_a._get_key(prompt, temperature=0.2), cache_b._get_key(prompt, temperature=0.2))

    def test_same_bundle_hits_cache(self):
        from ontology_compiler.rigor.llm_backend import LLMCache
        cache = LLMCache(version_bundle="v1")
        prompt = "some prompt"
        cache.set(prompt, "response", temperature=0.2, max_tokens=100)
        self.assertEqual(cache.get(prompt, temperature=0.2, max_tokens=100), "response")

    def test_no_bundle_keeps_legacy_behavior(self):
        from ontology_compiler.rigor.llm_backend import LLMCache
        cache = LLMCache()
        prompt = "some prompt"
        cache.set(prompt, "response", temperature=0.2, max_tokens=100)
        self.assertEqual(cache.get(prompt, temperature=0.2, max_tokens=100), "response")


class TestDeepSeekClientCache(unittest.TestCase):
    """D-2 regression (2026-08-15): DeepSeekClient was the primary provider but
    had no LLM cache at all — llm_call passes ``use_cache=True`` and the client
    silently ignored it, so every run re-paid the same prompts. It now uses the
    same version-bundled LLMCache as DifyClient."""

    def _client(self):
        import types
        from ontology_compiler.rigor.llm_backend import DeepSeekClient
        config = types.SimpleNamespace(
            deepseek_api_key="sk-test",
            deepseek_model="deepseek-test",
            deepseek_timeout=30.0,
        )
        return DeepSeekClient(config=config)

    def test_second_identical_call_hits_cache(self):
        import asyncio
        from unittest import mock

        client = self._client()
        calls = {"n": 0}

        async def fake_stream(prompt, temperature=0.7, max_tokens=2000, **kw):
            calls["n"] += 1
            return "streamed answer"

        with mock.patch.object(client, "_call_streaming", new=fake_stream):
            r1 = asyncio.run(client.call("same prompt", temperature=0.2, max_tokens=100))
            r2 = asyncio.run(client.call("same prompt", temperature=0.2, max_tokens=100))
        self.assertEqual(r1, "streamed answer")
        self.assertEqual(r2, "streamed answer")
        self.assertEqual(calls["n"], 1)   # second call served from cache

    def test_different_temperature_misses_cache(self):
        import asyncio
        from unittest import mock

        client = self._client()
        calls = {"n": 0}

        async def fake_stream(prompt, temperature=0.7, max_tokens=2000, **kw):
            calls["n"] += 1
            return "answer"

        with mock.patch.object(client, "_call_streaming", new=fake_stream):
            asyncio.run(client.call("prompt", temperature=0.2, max_tokens=100))
            asyncio.run(client.call("prompt", temperature=0.7, max_tokens=100))
        self.assertEqual(calls["n"], 2)

    def test_use_cache_false_bypasses_cache(self):
        import asyncio
        from unittest import mock

        client = self._client()
        calls = {"n": 0}

        async def fake_stream(prompt, temperature=0.7, max_tokens=2000, **kw):
            calls["n"] += 1
            return "answer"

        with mock.patch.object(client, "_call_streaming", new=fake_stream):
            asyncio.run(client.call("prompt", temperature=0.2, max_tokens=100, use_cache=False))
            asyncio.run(client.call("prompt", temperature=0.2, max_tokens=100, use_cache=False))
        self.assertEqual(calls["n"], 2)


class TestPromptBudgetEnvAliasing(unittest.TestCase):
    """D-3 regression: prompt/input budget hardcodes are now env-tunable with
    the same defaults, so the defaults remain stable for existing runs."""

    def test_call_llm_honors_env_input_budget(self):
        import asyncio
        import os
        from ontology_compiler.services.llm_call import call_llm
        os.environ["LLM_MAX_INPUT_CHARS"] = "200"
        try:
            seen = {}
            class FakeLLM:
                model = "fake"
                async def call_with_retry(self, prompt, **kw):
                    seen["prompt_len"] = len(prompt)
                    return '{"L1_entities": [], "relationships": []}'
            asyncio.run(call_llm(FakeLLM(), 100, 0.2, 5, "", None, lambda: type("M", (), {"name": "s"})(), "x" * 5000))
            self.assertLessEqual(seen["prompt_len"], 200)
        finally:
            del os.environ["LLM_MAX_INPUT_CHARS"]

    def test_call_llm_default_budget_is_80000(self):
        import os
        from ontology_compiler.services.llm_call import call_llm
        saved = os.environ.pop("LLM_MAX_INPUT_CHARS", None)
        try:
            import inspect
            import asyncio
            seen = {}
            class FakeLLM:
                model = "fake"
                async def call_with_retry(self, prompt, **kw):
                    seen["prompt_len"] = len(prompt)
                    return '{"L1_entities": [], "relationships": []}'
            # 10k chars < 80k default => no truncation
            asyncio.run(call_llm(FakeLLM(), 100, 0.2, 5, "", None, lambda: type("M", (), {"name": "s"})(), "x" * 10000))
            self.assertEqual(seen["prompt_len"], 10000)
        finally:
            if saved is not None:
                os.environ["LLM_MAX_INPUT_CHARS"] = saved


class TestPromptTruncationMetrics(unittest.TestCase):
    """D-5 regression: build_unified_prompt reports truncation/compression
    outcomes via the optional metrics dict — no change to the str contract."""

    @staticmethod
    def _service():
        from unittest import mock
        from ontology_compiler.services.prompt_service import PromptService
        ps = PromptService(
            document_analyzer=mock.MagicMock(),
            context_loader=None,
            prompt_strategy=mock.MagicMock(),
            prompt_loader=None,
            skill_loader=mock.MagicMock(),
        )
        ps.document_analyzer.select_content_intelligently = lambda text, limit: text[:limit]
        ps.prompt_strategy.build_base_fragment = lambda d: "BASE-"
        ps.prompt_strategy.build_domain_fragment = lambda d, meta, ctx: f"DOMAIN({len(meta)})"
        ps.prompt_strategy.build_task_fragment = lambda schema: "TASK-"
        ps.build_schema_hint_section = lambda: "SCHEMA"
        return ps

    def test_truncation_reports_dropped_chars(self):
        ps = self._service()
        metrics = {}
        text = "A" * 6000  # 6K > dynamic limit 4K => smart selection truncation
        prompt = ps.build_unified_prompt(text, domain="AUTO", use_context=False, metrics=metrics)
        self.assertIsInstance(prompt, str)
        self.assertEqual(metrics["truncation_reason"], "smart_content_selection")
        self.assertGreater(metrics["dropped_chars"], 0)
        self.assertLess(metrics["coverage_ratio"], 1.0)

    def test_full_retention_reports_no_truncation(self):
        ps = self._service()
        metrics = {}
        text = "B" * 500  # < 2K threshold => full retention
        ps.build_unified_prompt(text, domain="AUTO", use_context=False, metrics=metrics)
        self.assertEqual(metrics["truncation_reason"], "none")
        self.assertEqual(metrics["dropped_chars"], 0)
        self.assertEqual(metrics["coverage_ratio"], 1.0)

    def test_skip_truncate_path_still_reports(self):
        ps = self._service()
        metrics = {}
        text = "C" * 3000
        ps.build_unified_prompt(text, domain="AUTO", use_context=False, skip_truncate=True, metrics=metrics)
        self.assertEqual(metrics["truncation_reason"], "none")
        self.assertEqual(metrics["coverage_ratio"], 1.0)


class TestValidationService(unittest.TestCase):
    """Unit coverage for the validation cluster extracted in the 7th split
    (ontology_compiler.services.validation_service)."""

    def test_infer_entity_class_keywords(self):
        from ontology_compiler.services.validation_service import infer_entity_class
        self.assertEqual(infer_entity_class("SAP system"), "System")
        self.assertEqual(infer_entity_class("Change Management"), "Capability")
        self.assertEqual(infer_entity_class(""), "Concept")
        self.assertEqual(infer_entity_class("UnknownThing"), "Concept")

    def test_add_validation_feedback(self):
        from ontology_compiler.services.validation_service import add_validation_feedback
        out = add_validation_feedback("prompt", {"errors": ["missing_meta_id"]})
        self.assertIn("prompt", out)
        self.assertIn("PREVIOUS ATTEMPT FEEDBACK", out)
        self.assertEqual(add_validation_feedback("p", {"errors": []}), "p")

    def test_add_governance_feedback(self):
        from ontology_compiler.services.validation_service import add_governance_feedback
        report = {"issues": [{"type": "bad_rel", "detail": "x", "skill_id": "s1"}],
                  "blocked_relations": 2}
        out = add_governance_feedback("p", report)
        self.assertIn("GOVERNANCE VIOLATION FEEDBACK", out)
        self.assertIn("bad_rel", out)
        self.assertEqual(add_governance_feedback("p", {"issues": []}), "p")

    def test_validate_response_semantics_flags_missing_fields(self):
        from ontology_compiler.services.validation_service import validate_response_semantics
        res = validate_response_semantics(
            [{"meta_id": "", "meta_class": ""}], [], [], [],
            [{"from": "X", "to": ""}],
        )
        self.assertFalse(res["valid"])
        types = {e["type"] for e in res["errors"]}
        self.assertIn("missing_meta_id", types)
        self.assertIn("missing_meta_class", types)
        self.assertIn("relationship_missing_endpoint", types)

    def test_validate_with_schema_registry_returns_inputs_without_registry(self):
        import ontology_compiler.services.validation_service as vs
        saved = vs.get_schema_registry
        vs.get_schema_registry = lambda: None
        try:
            out = vs.validate_with_schema_registry([1], [2], [3], [4], "DOMAIN")
            self.assertEqual(out, ([1], [2], [3], [4]))
        finally:
            vs.get_schema_registry = saved

    def test_validate_extraction_accepts_structured_dict_evidence(self):
        """Regression (2026-08-14): the Pydantic entity models required
        ``evidence: str`` while the real pipeline payload carries a structured
        dict (evidence_id/content/confidence...). That mismatch made every
        entity fail validation, forcing DEGRADED extraction on every run even
        for clean LLM output. Evidence must accept both shapes. L2 entities
        also must not require ``domain`` (Optional in the model).
        """
        from ontology_compiler.services.validation_service import validate_extraction

        result = validate_extraction(
            [{"meta_id": "L1-001", "meta_name": "Production Inventory", "meta_class": "Capability",
              "evidence": {"content": "slide 2", "confidence": 1.0}, "layer": "L1"}],
            [{"meta_id": "L2-001", "scenario_name": "Inventory Control", "meta_class": "Capability",
              "evidence": {"content": "slide 5", "confidence": 1.0}, "layer": "L2"}],
            [],
            [{"from": "L1-001", "to": "L2-001", "type": "PARTICIPATES_IN"}],
        )
        self.assertTrue(result["valid"])
        self.assertEqual([e for e in result["errors"] if "evidence" in str(e)], [])
        self.assertEqual([e for e in result["errors"] if "domain" in str(e)], [])

        plain_string = validate_extraction(
            [{"meta_id": "L1-001", "meta_name": "X", "meta_class": "Capability", "evidence": "slide 2", "layer": "L1"}],
            [], [], [],
        )
        self.assertTrue(plain_string["valid"])

    def test_cross_layer_rules_are_consistent_with_quality_gate(self):
        """Regression: global optimizer must not use a cross-layer direction set
        that conflicts with the governed relationship rules. ``_ALLOWED_CROSS_LAYER``
        and ``_cross_layer_direction_allowed`` derive from
        config/ontology/relationship_rules.yaml — the same source used by
        QualityGate — so an optimizer decision can never contradict the gate.
        """
        import yaml
        from pathlib import Path
        from src.pipeline.stages import global_optimizer as go

        rules = yaml.safe_load(Path("config/ontology/relationship_rules.yaml").read_text(encoding="utf-8"))
        lp = rules.get("layer_pair_rules") or {}

        layers = ("L0", "L1", "L2", "L2.5", "L3")
        for src in layers:
            for tgt in layers:
                allowed = go._cross_layer_direction_allowed(src, tgt, lp)
                # Re-implement the exact gate verdict (quality_gate
                # _check_layer_pair_rules): check pair_key first, break on hit.
                gate_flags = False
                for key in (f"{src}-{tgt}", f"{tgt}-{src}"):
                    if key in lp:
                        gate_flags = not lp[key].get("allowed", True)
                        break
                self.assertEqual(
                    allowed,
                    not gate_flags,
                    f"optimizer/gate disagreement on {src}->{tgt}",
                )

        # Sanity anchors: L2.5<->L2 and L2->L1 stay forbidden under current config.
        self.assertNotIn(("L2.5", "L2"), go._ALLOWED_CROSS_LAYER)
        self.assertNotIn(("L2", "L1"), go._ALLOWED_CROSS_LAYER)
        # Forward abstract->concrete directions remain allowed.
        self.assertIn(("L0", "L1"), go._ALLOWED_CROSS_LAYER)
        self.assertIn(("L1", "L2"), go._ALLOWED_CROSS_LAYER)


class TestPromptService(unittest.TestCase):
    """Unit coverage for the prompt cluster extracted in the 8th split
    (ontology_compiler.services.prompt_service)."""

    def _service(self):
        from ontology_compiler.services.prompt_service import PromptService, DocumentAnalyzer, PromptStrategy
        from ontology_compiler.services.prompt_loader import get_prompt_loader
        return PromptService(
            document_analyzer=DocumentAnalyzer(),
            context_loader=None,
            prompt_strategy=PromptStrategy(),
            prompt_loader=get_prompt_loader(),
            skill_loader=None,
        )

    def test_dynamic_text_limit_tiers(self):
        ps = self._service()
        self.assertEqual(ps.get_dynamic_text_limit(1000), 1000)
        self.assertEqual(ps.get_dynamic_text_limit(3000), 3000)
        self.assertEqual(ps.get_dynamic_text_limit(8000), 4000)
        self.assertEqual(ps.get_dynamic_text_limit(30000), 5000)

    def test_build_unified_prompt_composition(self):
        ps = self._service()
        p = ps.build_unified_prompt(
            "Small doc about SAP ERP and warehouse management.", domain="SUPPLY_CHAIN")
        self.assertIn("DOCUMENT:", p)
        self.assertIn("SCHEMA HINTS", p)
        self.assertIn("SUPPLY_CHAIN", p)

    def test_compress_prompt_to_limit(self):
        ps = self._service()
        short = "tiny"
        self.assertEqual(ps.compress_prompt_to_limit(short, 100), short)
        long = ps.build_unified_prompt("x" * 30000)
        compressed = ps.compress_prompt_to_limit(long, 12000)
        self.assertLessEqual(len(compressed), 12000)

    def test_schema_hint_and_experience_sections(self):
        import tempfile
        from pathlib import Path

        from ontology_compiler.services.failure_memory import (
            FailureMemory,
            bind_failure_memory,
            record_json_parse_failure,
            reset_failure_memory,
        )

        ps = self._service()
        self.assertIn("SCHEMA HINTS", ps.build_schema_hint_section())

        # The experience section reads the process-wide store. Other tests
        # (and a previous run's outputs/failure_memory) must not decide which
        # heading this assertion sees.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(reset_failure_memory)
        bind_failure_memory(FailureMemory(storage_path=Path(tmp.name)))

        empty = ps.build_experience_section("SUPPLY_CHAIN")
        self.assertIn("Previous extraction experience", empty)
        self.assertIn("No recurring patterns found", empty)

        record_json_parse_failure("bad", document_key="docs/a.pptx")
        record_json_parse_failure("bad again", document_key="docs/a.pptx")
        learned = ps.build_experience_section("SUPPLY_CHAIN")
        self.assertIn("Previous Extraction Experience", learned)
        self.assertIn("json_parse_error", learned)
        self.assertIn("strict_json", learned)


class TestChunkingService(unittest.TestCase):
    """Unit coverage for the smart-chunking service extracted in the 9th split
    (ontology_compiler.services.chunking_service)."""

    class _FakeChunker:
        def __init__(self):
            self.config = {}

        def smart_chunk(self, text, source_file=""):
            return [{"content": text[:100], "section": "S1"},
                    {"content": text[100:200], "section": "S2"}]

    @staticmethod
    def _build_prompt(chunk_content, domain, use_context, skip_truncate, document_metadata, metrics=None):
        return f"prompt:{chunk_content}"

    @staticmethod
    async def _call_llm(prompt, **kwargs):
        return {"response": '{"L0_entities":[],"L1_entities":[],"L2_entities":[],'
                            '"L2_5_entities":[],"L3_entities":[],"relationships":[]}'}

    @staticmethod
    def _parse_response(text):
        return [], [{"meta_id": "L1-001", "name": "SAP", "meta_class": "System"}], [], [], [], []

    @staticmethod
    async def _extract_relationships(text, entities, domain):
        return [{"from": "L1-001", "to": "L1-001", "type": "uses"}]

    def test_extract_with_chunking_merges_results(self):
        import asyncio
        from ontology_compiler.services.chunking_service import extract_with_chunking
        from ontology_compiler.services.extraction_result import UnifiedExtractionResult

        async def run():
            return await extract_with_chunking(
                text="x" * 300, file_path=None, domain="SUPPLY_CHAIN", use_context=True,
                enable_validation=False, max_retries=1, document_metadata=None,
                chunker=self._FakeChunker(), build_prompt=self._build_prompt,
                call_llm=self._call_llm, parse_response=self._parse_response,
                extract_relationships=self._extract_relationships,
                result_cls=UnifiedExtractionResult,
            )

        r = asyncio.run(run())
        self.assertEqual(len(r.l1_entities), 1)          # identical entity merged across chunks
        self.assertEqual(len(r.relationships), 1)        # two-phase relation extraction
        self.assertIn("CHUNK_SEPARATOR", r.raw_response)
        self.assertTrue(all(e.get("chunk_section") for e in r.l1_entities))
        self.assertEqual(r.l1_entities[0]["source_chunk_ids"], [0, 1])

    def test_extract_with_chunking_skips_relations_when_no_entities(self):
        import asyncio
        from ontology_compiler.services.chunking_service import extract_with_chunking
        from ontology_compiler.services.extraction_result import UnifiedExtractionResult

        async def run():
            return await extract_with_chunking(
                text="x" * 300, file_path=None, domain="SUPPLY_CHAIN", use_context=True,
                enable_validation=False, max_retries=1, document_metadata=None,
                chunker=self._FakeChunker(), build_prompt=self._build_prompt,
                call_llm=self._call_llm, parse_response=lambda t: ([], [], [], [], [], []),
                extract_relationships=self._extract_relationships,
                result_cls=UnifiedExtractionResult,
            )

        r = asyncio.run(run())
        self.assertEqual(len(r.l1_entities), 0)
        self.assertEqual(len(r.relationships), 0)        # guarded by len(all_entities) > 0

    def test_extract_with_chunking_merges_duplicate_entities_and_provenance(self):
        import asyncio
        from ontology_compiler.services.chunking_service import extract_with_chunking
        from ontology_compiler.services.extraction_result import UnifiedExtractionResult

        def parse_response(text):
            return [], [
                {"meta_id": "L1-001", "name": "SAP", "meta_class": "System", "provenance": {"chunk": 1}},
            ], [], [], [], [{"from": "L1-001", "to": "L1-001", "type": "uses", "provenance": {"chunk": 1}}]

        async def run():
            return await extract_with_chunking(
                text="x" * 300, file_path=None, domain="SUPPLY_CHAIN", use_context=True,
                enable_validation=False, max_retries=1, document_metadata={"document_type": "pptx"},
                chunker=self._FakeChunker(), build_prompt=self._build_prompt,
                call_llm=self._call_llm, parse_response=parse_response,
                extract_relationships=self._extract_relationships,
                result_cls=UnifiedExtractionResult,
            )

        r = asyncio.run(run())
        self.assertEqual(len(r.l1_entities), 1)
        self.assertEqual(len(r.relationships), 1)
        entity = r.l1_entities[0]
        rel = r.relationships[0]
        self.assertEqual(entity["source_chunk_ids"], [0, 1])
        self.assertEqual(entity["source_sections"], ["S1", "S2"])
        self.assertEqual(rel["source_chunk_ids"], [0, 1])
        self.assertIn("decision_trace", entity)
        self.assertIn("decision_trace", rel)
        self.assertTrue(r.metadata.get("chunk_metrics"))
        self.assertIn("chunk_quality", r.metadata)


class TestTextChunkingStrategies(unittest.TestCase):
    """Explicit strategy dispatch (slide_based / section_based / hybrid)."""

    def setUp(self):
        from ontology_compiler.services.text_chunking import TextChunker

        self.chunker = TextChunker()

    def test_slide_based_splits_on_slide_markers(self):
        text = ("Slide 1\nIntro content\n"
                "Slide 2\nDetail content A\n"
                "Slide 3\nDetail content B\n")
        chunks = self.chunker.chunk_by_strategy("slide_based", text)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any("Slide" in c.get("section", "") for c in chunks))
        for c in chunks:
            self.assertEqual(c.get("chunk_type"), "slide")

    def test_section_based_uses_headings(self):
        text = "# Alpha\ncontent\n## Beta\nmore\n# Gamma\nrest"
        chunks = self.chunker.chunk_by_strategy("section_based", text)
        self.assertGreaterEqual(len(chunks), 1)
        sections = [c.get("section", "") for c in chunks]
        self.assertTrue(any("Alpha" in s for s in sections))
        self.assertTrue(any("Beta" in s for s in sections))

    def test_hybrid_prefers_slides_then_headings_then_fixed(self):
        slide_text = "Slide 1\nA\nSlide 2\nB"
        slide_chunks = self.chunker.chunk_by_strategy("hybrid", slide_text)
        self.assertTrue(any("Slide" in c.get("section", "") for c in slide_chunks))

        heading_text = "# One\nA\n# Two\nB"
        heading_chunks = self.chunker.chunk_by_strategy("hybrid", heading_text)
        self.assertTrue(any("One" in c.get("section", "") for c in heading_chunks))

        plain_text = "plain " * 200
        fixed_chunks = self.chunker.chunk_by_strategy("hybrid", plain_text)
        self.assertGreaterEqual(len(fixed_chunks), 1)
        self.assertTrue(all(c.get("chunk_type") == "fixed" for c in fixed_chunks))

    def test_smart_chunk_strategy_override(self):
        text = "Slide 1\nIntro\nSlide 2\nDetail\n"
        chunks = self.chunker.smart_chunk(text, strategy="slide_based")
        self.assertTrue(any("Slide" in c.get("section", "") for c in chunks))

    def test_strategy_profile_maps_pptx_to_slide_based(self):
        self.assertEqual(self.chunker.get_chunking_strategy_profile("pptx"), "slide_based")
        self.assertEqual(self.chunker.get_chunking_strategy_profile("md"), "section_based")
        self.assertEqual(self.chunker.get_chunking_strategy_profile("xlsx"), "fixed")


class TestChunkingServiceCache(unittest.TestCase):
    """Chunk-level cache and cache statistics for extract_with_chunking."""

    class _FakeChunker:
        def __init__(self):
            self.config = {}

        def smart_chunk(self, text, source_file="", strategy="auto"):
            return [{"content": text, "section": "S1"}]

    @staticmethod
    def _build_prompt(chunk_content, domain, use_context, skip_truncate, document_metadata, metrics=None):
        return f"prompt:{chunk_content}"

    @staticmethod
    async def _call_llm(prompt, **kwargs):
        return {"response": '{"L1_entities":[{"meta_id":"L1-001","name":"SAP","meta_class":"System"}],"relationships":[]}'}

    @staticmethod
    def _parse_response(text):
        return [], [{"meta_id": "L1-001", "name": "SAP", "meta_class": "System"}], [], [], [], []

    @staticmethod
    async def _extract_relationships(text, entities, domain):
        return []

    def test_duplicate_chunks_hit_cache(self):
        import asyncio
        from ontology_compiler.services.chunking_service import extract_with_chunking
        from ontology_compiler.services.extraction_result import UnifiedExtractionResult

        calls = {"n": 0}
        original_call = self._call_llm

        async def counting_call_llm(prompt, **kwargs):
            calls["n"] += 1
            return await original_call(prompt)

        async def run(make_multiple_chunks: bool):
            class _TwoChunkChunker:
                def __init__(self):
                    self.config = {}

                def smart_chunk(self, text, source_file="", strategy="auto"):
                    # Two identical chunks exercise the in-run cache.
                    return [{"content": text, "section": "S1"},
                            {"content": text, "section": "S2"}]

            return await extract_with_chunking(
                text="dup " * 300,
                file_path=None,
                domain="SUPPLY_CHAIN",
                use_context=True,
                enable_validation=False,
                max_retries=1,
                document_metadata=None,
                chunker=_TwoChunkChunker(),
                build_prompt=self._build_prompt,
                call_llm=counting_call_llm,
                parse_response=self._parse_response,
                extract_relationships=self._extract_relationships,
                result_cls=UnifiedExtractionResult,
                cache_enabled=True,
            )

        r = asyncio.run(run(make_multiple_chunks=True))
        # Both chunks returned the same entity; the chunk-level merge collapses
        # duplicates to a single record, so we assert cache behavior (not count).
        self.assertEqual(len(r.l1_entities), 1)
        self.assertEqual(r.metadata["cache_stats"]["hits"], 1)
        self.assertEqual(r.metadata["cache_stats"]["misses"], 1)
        # Only the first unique chunk touched the LLM.
        self.assertEqual(calls["n"], 1)
        self.assertEqual({m["status"] for m in r.metadata["chunk_metrics"]}, {"ok", "cache_hit"})

    def test_cache_disabled_flag_forces_llm_calls(self):
        import asyncio
        from ontology_compiler.services.chunking_service import extract_with_chunking
        from ontology_compiler.services.extraction_result import UnifiedExtractionResult

        calls = {"n": 0}
        original_call = self._call_llm

        async def counting_call_llm(prompt, **kwargs):
            calls["n"] += 1
            return await original_call(prompt)

        async def run():
            class _ThreeChunkChunker:
                def __init__(self):
                    self.config = {}

                def smart_chunk(self, text, source_file="", strategy="auto"):
                    return [{"content": "alpha " * 50, "section": "S1"},
                            {"content": "alpha " * 50, "section": "S2"},
                            {"content": "beta " * 50, "section": "S3"}]

            return await extract_with_chunking(
                text="dup " * 300,
                file_path=None,
                domain="SUPPLY_CHAIN",
                use_context=True,
                enable_validation=False,
                max_retries=1,
                document_metadata=None,
                chunker=_ThreeChunkChunker(),
                build_prompt=self._build_prompt,
                call_llm=counting_call_llm,
                parse_response=self._parse_response,
                extract_relationships=self._extract_relationships,
                result_cls=UnifiedExtractionResult,
                cache_enabled=False,
            )

        r = asyncio.run(run())
        self.assertEqual(calls["n"], 3)                  # no cache: every chunk hits LLM
        self.assertEqual(r.metadata["cache_stats"]["hits"], 0)
        self.assertEqual(r.metadata["cache_stats"]["enabled"], False)


class TestLlmExtractionLoopReturnContract(unittest.TestCase):
    """Regression: _run_llm_extraction_loop must always return the full tuple.

    Previously the return statement was nested under the except branch, so the
    degraded-validation path (break with entities) fell through to an implicit
    None return and extract_unified crashed with
    "cannot unpack non-iterable NoneType object" (observed on 2026-08-13).
    """

    def _loop_extractor(self):
        from ontology_compiler.services.true_unified_extractor import TrueUnifiedExtractor

        extractor = object.__new__(TrueUnifiedExtractor)
        extractor._call_llm = mock.AsyncMock(return_value={
            "response": '{"L1_entities": [{"meta_id": "L1-001", "name": "SAP", "meta_class": "System"}]}',
            "truncated": False,
            "parse_success": True,
        })
        extractor._build_unified_prompt = mock.Mock(return_value="prompt")
        extractor._parse_unified_response = mock.Mock(return_value=(
            [],
            [{"meta_id": "L1-001", "name": "SAP", "meta_class": "System"}],
            [],
            [],
            [],
            [],
        ))
        return extractor

    def test_degraded_validation_path_returns_tuple(self):
        import ontology_compiler.services.true_unified_extractor as extractor_module
        from ontology_compiler.services.true_unified_extractor import TrueUnifiedExtractor

        extractor = self._loop_extractor()

        validation_result = {
            "valid": False,
            "errors": ["missing evidence"],
            "warnings": [],
            "stats": {"l1_count": 1, "l2_count": 0, "l25_count": 0},
        }
        with mock.patch.object(extractor_module, "validate_extraction", return_value=validation_result), \
             mock.patch.object(extractor_module, "validate_response_semantics",
                               return_value={"valid": True, "errors": [], "warnings": []}), \
             mock.patch(
                 "ontology_compiler.schemas.extraction_schema.validate_extraction_result",
                 return_value={"valid": True, "errors": [], "warnings": []},
             ):

            result = asyncio.run(extractor._run_llm_extraction_loop(
                prompt="prompt",
                text="text",
                domain="SUPPLY_CHAIN",
                use_context=True,
                max_retries=1,
                enable_validation=True,
                document_metadata=None,
            ))

        l0, l1, l2, l25, l3, rels, response_text, degraded, attempt, text, prompt = result
        self.assertEqual(l1, [{"meta_id": "L1-001", "name": "SAP", "meta_class": "System"}])
        self.assertTrue(degraded)
        self.assertEqual(response_text, '{"L1_entities": [{"meta_id": "L1-001", "name": "SAP", "meta_class": "System"}]}')

    def test_success_validation_path_returns_tuple(self):
        import ontology_compiler.services.true_unified_extractor as extractor_module

        extractor = self._loop_extractor()

        with mock.patch.object(
            extractor_module,
            "validate_extraction",
            return_value={
                "valid": True,
                "errors": [],
                "warnings": [],
                "stats": {"l1_count": 1, "l2_count": 0, "l25_count": 0},
            },
        ), \
             mock.patch.object(extractor_module, "validate_response_semantics",
                               return_value={"valid": True, "errors": [], "warnings": []}), \
             mock.patch(
                 "ontology_compiler.schemas.extraction_schema.validate_extraction_result",
                 return_value={"valid": True, "errors": [], "warnings": []},
             ):

            result = asyncio.run(extractor._run_llm_extraction_loop(
                prompt="prompt",
                text="text",
                domain="SUPPLY_CHAIN",
                use_context=True,
                max_retries=1,
                enable_validation=True,
                document_metadata=None,
            ))

        l0, l1, l2, l25, l3, rels, response_text, degraded, attempt, text, prompt = result
        self.assertEqual(l1, [{"meta_id": "L1-001", "name": "SAP", "meta_class": "System"}])
        self.assertFalse(degraded)
        self.assertEqual(response_text, '{"L1_entities": [{"meta_id": "L1-001", "name": "SAP", "meta_class": "System"}]}')

class TestGovernanceService(unittest.TestCase):
    """Unit coverage for the skill-governance service extracted in the 9th split
    (ontology_compiler.services.governance_service)."""

    class _FakeSkill:
        skill_id = "sk_1"
        domain = []
        governance = {
            "validation_mode": "strict",
            "gating_level": "hard",
            "allowed_layers": ["L1", "L2"],
            "allowed_relation_types": ["uses"],
            "blocked_relation_types": ["kills"],
            "preferred_entity_types": ["System"],
            "required_fields": [{"entity_type": "System", "fields": ["description"]}],
            "entity_type_mapping": [{"from_type": "Data", "suggest_type": "System"}],
        }

    class _FakeLoader:
        def find_matching_skills(self, text):
            return [TestGovernanceService._FakeSkill()]

    def test_apply_skill_governance_reports_violations(self):
        from ontology_compiler.services.governance_service import apply_skill_governance
        r = apply_skill_governance(
            "doc", "SUPPLY_CHAIN",
            [{"meta_class": "Data", "name": "X"}], [], [],
            [{"type": "kills"}], self._FakeLoader(),
        )
        self.assertEqual(r["activated_skills"], ["sk_1"])
        self.assertEqual(r["mode"], "strict")
        self.assertEqual(r["gating_level"], "hard")
        self.assertEqual(r["blocked_relations"], 1)
        self.assertEqual(r["entity_type_violations"], 1)
        # required-fields rule matches only entity_type "System"; "Data" entity has none
        self.assertEqual(r["required_field_violations"], 0)
        types = {i["type"] for i in r["issues"]}
        self.assertIn("BLOCKED_RELATION", types)
        self.assertIn("ENTITY_TYPE_VIOLATION", types)

    def test_apply_skill_governance_requires_matching_fields(self):
        from ontology_compiler.services.governance_service import apply_skill_governance
        r = apply_skill_governance(
            "doc", "SUPPLY_CHAIN",
            [{"meta_class": "System", "name": "S"}], [], [],
            [], self._FakeLoader(),
        )
        self.assertEqual(r["required_field_violations"], 1)   # missing "description"


class TestRelationshipService(unittest.TestCase):
    """Unit coverage for relationship_service extracted in the 10th split."""

    def test_build_entity_context_text_selects_mentions(self):
        from ontology_compiler.services.relationship_service import build_entity_context_text
        text = ("--- Slide 1\nSAP overview\n\n"
                "The SAP ERP system manages supply.\n\n"
                "Unrelated filler paragraph about weather.\n\n"
                "Warehouse operations use SAP.")
        entities = [{"name": "SAP"}, {"name": "Warehouse"}]
        out = build_entity_context_text(text, entities)
        self.assertIn("SAP ERP system", out)
        self.assertIn("Warehouse operations", out)

    def test_extract_relationships_separate_parses_list(self):
        import asyncio
        from ontology_compiler.services.relationship_service import extract_relationships_separate

        class FakeLLM:
            async def call_with_retry(self, *args, **kwargs):
                return '[{"from": "A", "to": "B", "type": "uses"}]'

        class FakePrompt:
            def render(self, **kwargs):
                return "prompt"

        class FakeLoader:
            def get_template(self, name):
                return FakePrompt() if name == "relation_extraction" else None

        async def run():
            return await extract_relationships_separate(
                "short text", [{"meta_id": "L1-001", "name": "SAP", "meta_class": "System"}],
                "SUPPLY_CHAIN",
                llm=FakeLLM(), max_tokens=100, temperature=0.1, timeout=30,
                prompt_loader=FakeLoader(), build_entity_context_text=lambda t, e: t,
            )

        rels = asyncio.run(run())
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0]["confidence"], "0.7")   # confidence injected


class TestRefinementService(unittest.TestCase):
    """Unit coverage for refinement_service extracted in the 10th split."""

    def test_refine_l2_adds_new_scenarios(self):
        import asyncio
        from ontology_compiler.services.refinement_service import refine_l2

        class FakeLLM:
            async def call(self, prompt, **kwargs):
                return '{"business_scenarios": [{"name": "New Scenario", "description": "d"}]}'

        async def run():
            return await refine_l2(
                "content", [{"scenario_name": "Existing"}], "SUPPLY_CHAIN",
                llm=FakeLLM(), refinement_threshold=10, enable_refinement=True,
            )

        out = asyncio.run(run())
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1]["meta_class"], "System")
        self.assertEqual(out[1]["layer"], "L2")

    def test_refine_l2_skips_when_threshold_met(self):
        import asyncio
        from ontology_compiler.services.refinement_service import refine_l2

        async def run():
            return await refine_l2(
                "content", [{"scenario_name": "A"}], "SUPPLY_CHAIN",
                llm=None, refinement_threshold=1, enable_refinement=False,
            )

        self.assertEqual(asyncio.run(run()), [{"scenario_name": "A"}])


class TestPluginAdapter(unittest.TestCase):
    """Unit coverage for plugin_adapter extracted in the 10th split."""

    def test_read_source_content_path(self):
        import tempfile
        from ontology_compiler.services.plugin_adapter import read_source_content
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write("hello")
            path = f.name
        content, err = read_source_content(path, 0, "extractor")
        self.assertIsNone(err)
        self.assertEqual(content, "hello")

    def test_read_source_content_str(self):
        from ontology_compiler.services.plugin_adapter import read_source_content
        content, err = read_source_content("inline text", 0, "extractor")
        self.assertIsNone(err)
        self.assertEqual(content, "inline text")

    def test_build_failure_result(self):
        from ontology_compiler.services.plugin_adapter import build_failure_result
        r = build_failure_result("boom", 12.5, "extractor")
        self.assertFalse(r.success)
        self.assertEqual(r.error_message, "boom")
        self.assertEqual(r.execution_time_ms, 12.5)

    def test_build_sync_extractor_result(self):
        from ontology_compiler.services.plugin_adapter import build_sync_extractor_result
        from ontology_compiler.services.extraction_result import UnifiedExtractionResult
        result = UnifiedExtractionResult(
            l0_entities=[{"meta_id": "L0-001", "name": "System", "meta_class": "Capability"}],
            l1_entities=[{"meta_id": "L1-001", "name": "SAP", "meta_class": "System"}],
            l2_entities=[], l25_entities=[], l3_entities=[],
            relationships=[], raw_response="{}", processing_time_ms=1.0,
            token_usage={}, quality_score=0.5,
        )
        r = build_sync_extractor_result(result, "SUPPLY_CHAIN", "file.md", {}, "extractor")
        self.assertTrue(r.success)
        self.assertEqual(len(r.fragment.nodes), 2)   # L0 + L1 nodes
        self.assertEqual(r.fragment.metadata["l1_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)