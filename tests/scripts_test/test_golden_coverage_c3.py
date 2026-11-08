"""C-3 tests: golden semantic coverage and baseline cost/token/latency fields.

Validates that:
1. Every sample in config/regression_samples.json that has a
   ``golden_semantic`` block can be loaded by ``load_golden_set``.
2. At least 7 samples carry a ``golden_semantic`` block (minimum coverage).
3. The baseline payload produced by the regression runner includes
   cost / token / latency aggregation fields.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.golden_semantic_eval import (
    GoldenSample,
    load_golden_set,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "config" / "regression_samples.json"


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _all_samples(manifest: dict) -> list[dict]:
    samples = []
    for group in manifest.get("sample_groups", []):
        for sample in group.get("samples", []):
            samples.append(sample)
    return samples


def _golden_samples(manifest: dict) -> list[dict]:
    return [s for s in _all_samples(manifest) if s.get("golden_semantic")]


class GoldenManifestCoverageTest(unittest.TestCase):
    """Verify the regression manifest has enough golden_semantic coverage."""

    def test_manifest_loads(self):
        manifest = _load_manifest()
        self.assertIsInstance(manifest, dict)
        self.assertIn("sample_groups", manifest)

    def test_at_least_seven_golden_samples(self):
        manifest = _load_manifest()
        golden = _golden_samples(manifest)
        self.assertGreaterEqual(
            len(golden),
            7,
            "Expected at least 7 samples with golden_semantic; "
            f"found {len(golden)}",
        )

    def test_all_golden_blocks_parse_via_load_golden_set(self):
        manifest = _load_manifest()
        golden = _golden_samples(manifest)
        for sample in golden:
            golden_data = sample["golden_semantic"]
            with self.subTest(sample_id=sample.get("sample_id")):
                parsed = load_golden_set(golden_data)
                self.assertIsInstance(parsed, GoldenSample)
                self.assertTrue(parsed.sample_id or sample.get("sample_id"))

    def test_golden_blocks_have_at_least_one_entity_or_relation(self):
        manifest = _load_manifest()
        golden = _golden_samples(manifest)
        for sample in golden:
            golden_data = sample["golden_semantic"]
            parsed = load_golden_set(golden_data)
            with self.subTest(sample_id=sample.get("sample_id")):
                total = len(parsed.entities) + len(parsed.relations)
                self.assertGreaterEqual(
                    total, 1,
                    f"golden_semantic for {sample.get('sample_id')} has no "
                    f"entities or relations",
                )

    def test_all_golden_blocks_reject_related_to(self):
        """Every golden block should reject RELATED_TO as a relation type."""
        manifest = _load_manifest()
        golden = _golden_samples(manifest)
        for sample in golden:
            golden_data = sample["golden_semantic"]
            parsed = load_golden_set(golden_data)
            with self.subTest(sample_id=sample.get("sample_id")):
                self.assertIn(
                    "RELATED_TO", parsed.reject_relation_types,
                    f"golden_semantic for {sample.get('sample_id')} does not "
                    f"reject RELATED_TO",
                )

    def test_golden_samples_cover_multiple_groups(self):
        """Golden blocks should span at least 3 different sample groups."""
        manifest = _load_manifest()
        golden = _golden_samples(manifest)
        group_ids = set()
        for sample in golden:
            # Find the group this sample belongs to
            for group in manifest.get("sample_groups", []):
                for s in group.get("samples", []):
                    if s.get("sample_id") == sample.get("sample_id"):
                        group_ids.add(group.get("group_id"))
                        break
        self.assertGreaterEqual(
            len(group_ids), 3,
            f"Golden samples only cover {len(group_ids)} groups; expected >= 3",
        )

    def test_golden_metrics_pass_on_synthetic_full_match(self):
        """A synthetic full-match observation should produce ok=True for each
        golden block."""
        manifest = _load_manifest()
        golden = _golden_samples(manifest)
        from scripts.golden_semantic_eval import golden_metrics
        for sample in golden:
            golden_data = sample["golden_semantic"]
            parsed = load_golden_set(golden_data)
            # Build a synthetic observation that includes all golden entities
            # with meta_ids and relations that reference those meta_ids.
            observed_entities = []
            name_to_id = {}
            for idx, e in enumerate(parsed.entities):
                eid = f"L{idx + 1:04d}"
                observed_entities.append({
                    "meta_id": eid,
                    "name": e.name,
                    "meta_class": e.meta_class,
                    "layer": e.layer,
                })
                name_to_id[e.name.lower()] = eid
            observed_relations = []
            for r in parsed.relations:
                observed_relations.append({
                    "source": name_to_id.get(r.source_name.lower(), r.source_name),
                    "target": name_to_id.get(r.target_name.lower(), r.target_name),
                    "relation_type": r.relation_types[0] if r.relation_types else "DERIVED_FROM",
                })
            metrics = golden_metrics(parsed, observed_entities, observed_relations)
            with self.subTest(sample_id=sample.get("sample_id")):
                self.assertTrue(
                    metrics["ok"],
                    f"Synthetic full-match failed for {sample.get('sample_id')}: "
                    f"entity_recall={metrics['entity_recall']}, "
                    f"relation_recall={metrics['relation_recall']}, "
                    f"unmatched={metrics['unmatched_relations']}",
                )


class BaselineCostSummaryTest(unittest.TestCase):
    """C-3: verify the baseline payload aggregation helpers produce
    cost / token / latency fields."""

    def test_aggregate_batch_cost_with_no_results(self):
        from scripts.run_regression_samples import _aggregate_batch_cost
        result = _aggregate_batch_cost([])
        self.assertEqual(result["total_calls"], 0)
        self.assertEqual(result["total_tokens"], 0)
        self.assertEqual(result["total_cost_estimate"], 0.0)
        self.assertEqual(result["total_duration_ms"], 0)
        self.assertEqual(result["unknown_price_calls"], 0)

    def test_aggregate_batch_cost_with_results(self):
        from scripts.run_regression_samples import _aggregate_batch_cost
        results = [
            {
                "llm_cost": {
                    "total_calls": 10,
                    "total_tokens": 5000,
                    "total_cost_estimate": 0.05,
                    "total_duration_ms": 12000,
                    "unknown_price_calls": 1,
                },
            },
            {
                "llm_cost": {
                    "total_calls": 5,
                    "total_tokens": 3000,
                    "total_cost_estimate": 0.03,
                    "total_duration_ms": 8000,
                    "unknown_price_calls": 0,
                },
            },
        ]
        result = _aggregate_batch_cost(results)
        self.assertEqual(result["total_calls"], 15)
        self.assertEqual(result["total_tokens"], 8000)
        self.assertAlmostEqual(result["total_cost_estimate"], 0.08)
        self.assertEqual(result["total_duration_ms"], 20000)
        self.assertEqual(result["unknown_price_calls"], 1)

    def test_baseline_payload_includes_batch_cost(self):
        from scripts.run_regression_samples import _baseline_payload
        summary = {
            "manifest": "test.json",
            "executed_at": "2026-08-18T00:00:00",
            "count": 2,
            "passed_count": 1,
            "failed_count": 1,
            "overall_status": "FAIL",
            "thresholds": {},
            "results": [
                {"llm_cost": {"total_calls": 5, "total_tokens": 2000, "total_cost_estimate": 0.02, "total_duration_ms": 5000, "unknown_price_calls": 0}},
                {"llm_cost": {"total_calls": 3, "total_tokens": 1000, "total_cost_estimate": 0.01, "total_duration_ms": 3000, "unknown_price_calls": 0}},
            ],
        }
        payload = _baseline_payload(summary)
        self.assertIn("batch_cost", payload)
        self.assertEqual(payload["batch_cost"]["total_calls"], 8)
        self.assertEqual(payload["batch_cost"]["total_tokens"], 3000)
        self.assertAlmostEqual(payload["batch_cost"]["total_cost_estimate"], 0.03)

    def test_collect_llm_cost_summary_returns_dict(self):
        """The helper should return a dict even when the monitor is empty."""
        from scripts.run_regression_samples import _collect_llm_cost_summary
        result = _collect_llm_cost_summary()
        self.assertIsInstance(result, dict)
        # When no LLM calls have been made, the fields should be zero or absent.
        if result:
            self.assertIn("total_calls", result)
            self.assertIn("total_tokens", result)
            self.assertIn("total_cost_estimate", result)
            self.assertIn("total_duration_ms", result)


if __name__ == "__main__":
    unittest.main()
