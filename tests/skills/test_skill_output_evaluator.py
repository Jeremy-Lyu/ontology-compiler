"""P1-2 (2026-10-16): schema 驱动 skill evaluator 契约锁。

覆盖：
- 扁平字段表形态（skill.json 实际形态）与 JSON-Schema 形态都识别
- type/enum/minimum/maximum/required/properties/items
- 无 schema => 空结果（不失败）；非 dict 输出 => 报错
- registry.evaluable_skill_names / evaluate_output 接线
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ontology_compiler.skills.evaluator import (
    evaluate_output,
    evaluate_skill_document,
    is_evaluable,
)


FLAT_SCHEMA = {
    "gate_id": {"type": "string"},
    "result": {"type": "string", "enum": ["pass", "warn", "fail", "block"]},
    "score": {
        "type": "object",
        "properties": {"overall": {"type": "number", "minimum": 0, "maximum": 1}},
    },
    "issues": {"type": "array", "items": {"type": "object"}},
}

JSONSCHEMA_SHAPE = {
    "type": "object",
    "required": ["name"],
    "properties": {
        "name": {"type": "string"},
        "count": {"type": "integer", "minimum": 1},
    },
}


class FlatShapeTest(unittest.TestCase):
    def test_conforming_output_passes(self):
        self.assertEqual(
            evaluate_output(FLAT_SCHEMA, {"result": "pass", "score": {"overall": 0.9}, "issues": [{}]}),
            [],
        )

    def test_enum_and_range_violations(self):
        v = evaluate_output(FLAT_SCHEMA, {"result": "nope", "score": {"overall": 2.0}})
        self.assertTrue(any("enum" in x for x in v))
        self.assertTrue(any("maximum" in x for x in v))

    def test_type_mismatch_reports_once(self):
        v = evaluate_output(FLAT_SCHEMA, {"gate_id": 123})
        self.assertEqual(len(v), 1)
        self.assertIn("expected string", v[0])

    def test_unknown_key_is_ignored(self):
        self.assertEqual(evaluate_output(FLAT_SCHEMA, {"result": "pass", "extra": 1}), [])


class JsonSchemaShapeTest(unittest.TestCase):
    def test_required_and_nested(self):
        v = evaluate_output(JSONSCHEMA_SHAPE, {"count": 0})
        self.assertTrue(any("missing required key 'name'" in x for x in v))
        self.assertTrue(any("minimum" in x for x in v))

    def test_valid(self):
        self.assertEqual(evaluate_output(JSONSCHEMA_SHAPE, {"name": "x", "count": 3}), [])


class EdgeCaseTest(unittest.TestCase):
    def test_no_schema_is_not_a_failure(self):
        self.assertEqual(evaluate_output(None, {"anything": 1}), [])
        self.assertEqual(evaluate_output({}, {"anything": 1}), [])

    def test_non_mapping_output_for_flat_schema(self):
        v = evaluate_output(FLAT_SCHEMA, "not-a-dict")
        self.assertTrue(any("expected object" in x for x in v))

    def test_bool_is_not_a_number(self):
        v = evaluate_output({"x": {"type": "number"}}, {"x": True})
        self.assertTrue(any("expected number" in x for x in v))

    def test_is_evaluable(self):
        self.assertTrue(is_evaluable({"output_schema": FLAT_SCHEMA}))
        self.assertFalse(is_evaluable({"output_schema": {}}))
        self.assertFalse(is_evaluable({}))

    def test_evaluate_skill_document_wrapper(self):
        self.assertEqual(evaluate_skill_document({"output_schema": FLAT_SCHEMA}, {"result": "pass"}), [])


class RegistryWiringTest(unittest.TestCase):
    def _registry_with(self, skill_json: dict):
        from ontology_compiler.harness.skill_registry import SkillRegistry
        from ontology_compiler.harness.manifest import load_skill_manifest

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        skill_dir = root / "skills" / skill_json["name"] / ".ontology-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "skill.json").write_text(json.dumps(skill_json), encoding="utf-8")
        reg = SkillRegistry()
        reg.discover(root / "skills", project_root=root)
        return reg

    def _skill_json(self, with_schema: bool):
        d = {
            "name": "eval_demo",
            "description": "demo",
            "command": "python -m x",
            "script_path": "x.py",
            "category": "validation",
            "enabled": True,
        }
        if with_schema:
            d["output_schema"] = {"result": {"type": "string", "enum": ["pass", "fail"]}}
        return d

    def test_schema_survives_manifest_and_definition(self):
        reg = self._registry_with(self._skill_json(True))
        self.assertIn("eval_demo", reg.evaluable_skill_names())
        self.assertEqual(reg.evaluate_output("eval_demo", {"result": "pass"}), [])
        self.assertTrue(reg.evaluate_output("eval_demo", {"result": "bad"}))

    def test_skill_without_schema_not_evaluable_but_safe(self):
        reg = self._registry_with(self._skill_json(False))
        self.assertNotIn("eval_demo", reg.evaluable_skill_names())
        self.assertEqual(reg.evaluate_output("eval_demo", {"whatever": 1}), [])

    def test_unknown_skill_is_safe_noop(self):
        reg = self._registry_with(self._skill_json(True))
        self.assertEqual(reg.evaluate_output("does_not_exist", {}), [])


if __name__ == "__main__":
    unittest.main()
