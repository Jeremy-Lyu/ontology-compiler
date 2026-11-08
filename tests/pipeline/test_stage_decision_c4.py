"""C-4 characterization tests: versioned cache key + stage decision + extract wiring.

Locks the behavior behind ``stage_decision`` and ``versioned_cache`` so the
"cache hit 不会错误复用陈旧结果" guarantee stays testable:

- every version dimension must match for a cache hit to be valid
- a version bump invalidates (never silently reuses) a cached result
- the extract stage passes the resolved version bundle into the cache layer
"""
from __future__ import annotations

import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from src.pipeline.stage_decision import (
    BARRIER_STAGES,
    DETERMINISTIC_STAGES,
    LLM_DEPENDENT_STAGES,
    CacheVersionBundle,
    StageDecision,
    StageDecisionInput,
    compute_stage_decision,
    compute_versioned_cache_key,
    version_bundles_match,
)
from src.pipeline.versioned_cache import resolve_version_bundle


class VersionedCacheKeyTest(unittest.TestCase):
    """compute_versioned_cache_key must be stable and version-sensitive."""

    def setUp(self):
        self.bundle = CacheVersionBundle(
            prompt_version="p1",
            schema_version="s1",
            policy_version="pol1",
            model_family="deepseek",
            extraction_mode="production",
        )

    def test_key_is_deterministic(self):
        key1 = compute_versioned_cache_key("text", "prompt", self.bundle)
        key2 = compute_versioned_cache_key("text", "prompt", self.bundle)
        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 64)

    def test_key_changes_when_any_version_dimension_changes(self):
        base = compute_versioned_cache_key("text", "prompt", self.bundle)
        changed_dims = []
        for attr in ("prompt_version", "schema_version", "policy_version", "model_family", "extraction_mode"):
            other = CacheVersionBundle(
                prompt_version="p2" if attr == "prompt_version" else self.bundle.prompt_version,
                schema_version="s2" if attr == "schema_version" else self.bundle.schema_version,
                policy_version="pol2" if attr == "policy_version" else self.bundle.policy_version,
                model_family="dify" if attr == "model_family" else self.bundle.model_family,
                extraction_mode="degraded" if attr == "extraction_mode" else self.bundle.extraction_mode,
            )
            self.assertNotEqual(
                base,
                compute_versioned_cache_key("text", "prompt", other),
                f"version dimension {attr} must affect the cache key",
            )
            changed_dims.append(attr)
        self.assertEqual(len(changed_dims), 5)

    def test_key_changes_with_source_and_prompt(self):
        base = compute_versioned_cache_key("text", "prompt", self.bundle)
        self.assertNotEqual(base, compute_versioned_cache_key("other", "prompt", self.bundle))
        self.assertNotEqual(base, compute_versioned_cache_key("text", "other", self.bundle))


class VersionBundlesMatchTest(unittest.TestCase):
    def test_missing_cached_bundle_never_matches(self):
        self.assertFalse(version_bundles_match(CacheVersionBundle(), None))

    def test_empty_vs_populated_does_not_match(self):
        current = CacheVersionBundle(prompt_version="p1")
        cached = CacheVersionBundle()
        self.assertFalse(version_bundles_match(current, cached))

    def test_identical_bundles_match(self):
        a = CacheVersionBundle(prompt_version="p1", schema_version="s1", policy_version="p", model_family="m", extraction_mode="production")
        b = CacheVersionBundle(prompt_version="p1", schema_version="s1", policy_version="p", model_family="m", extraction_mode="production")
        self.assertTrue(version_bundles_match(a, b))

    def test_any_single_dimension_mismatch_breaks_match(self):
        base = CacheVersionBundle(prompt_version="p1", schema_version="s1", policy_version="p", model_family="m", extraction_mode="production")
        expected_keys = dict(base.to_dict())
        for key in expected_keys:
            from dataclasses import replace
            broken = replace(base, **{key: f"{expected_keys[key]}-changed"})
            self.assertFalse(version_bundles_match(base, broken), f"dimension {key} must be enforced")


class ComputeStageDecisionTest(unittest.TestCase):
    """Decision precedence and barrier determinism."""

    def _bundle(self):
        return CacheVersionBundle(prompt_version="p1", schema_version="s1", policy_version="p", model_family="m", extraction_mode="production")

    def test_missing_source_hash_blocks(self):
        result = compute_stage_decision(StageDecisionInput(
            stage="merge", execution_mode="production", source_hash="", version_bundle=self._bundle(),
        ))
        self.assertEqual(result.decision, StageDecision.BLOCK)

    def test_barrier_stages_always_run(self):
        for stage in BARRIER_STAGES:
            result = compute_stage_decision(StageDecisionInput(
                stage=stage, execution_mode="production", source_hash="abc", version_bundle=self._bundle(),
                has_cached_artifact=True, cached_success=True, upstream_changed=False,
                cached_source_hash="abc", cached_version_bundle=self._bundle(),
            ))
            self.assertEqual(result.decision, StageDecision.RUN, f"{stage} must be a barrier")

    def test_llm_stage_degrades_in_non_production_mode(self):
        for stage in LLM_DEPENDENT_STAGES:
            result = compute_stage_decision(StageDecisionInput(
                stage=stage, execution_mode="offline", source_hash="abc", version_bundle=self._bundle(),
            ))
            self.assertEqual(result.decision, StageDecision.DEGRADE)

    def test_never_reuses_stale_version_bundle(self):
        stale = CacheVersionBundle(prompt_version="old", schema_version="s1", policy_version="p", model_family="m", extraction_mode="production")
        result = compute_stage_decision(StageDecisionInput(
            stage="merge", execution_mode="production", source_hash="abc", version_bundle=self._bundle(),
            has_cached_artifact=True, cached_success=True, upstream_changed=False,
            cached_source_hash="abc", cached_version_bundle=stale,
        ))
        self.assertEqual(result.decision, StageDecision.RUN)
        self.assertIn("stale", result.reason.lower())

    def test_never_reuses_changed_source(self):
        result = compute_stage_decision(StageDecisionInput(
            stage="merge", execution_mode="production", source_hash="newhash", version_bundle=self._bundle(),
            has_cached_artifact=True, cached_success=True, upstream_changed=False,
            cached_source_hash="oldhash", cached_version_bundle=self._bundle(),
        ))
        self.assertEqual(result.decision, StageDecision.RUN)

    def test_failed_previous_run_reruns_only_failed_chunks(self):
        result = compute_stage_decision(StageDecisionInput(
            stage="merge", execution_mode="production", source_hash="abc", version_bundle=self._bundle(),
            has_cached_artifact=True, cached_success=False, upstream_changed=False,
            cached_source_hash="abc", cached_version_bundle=self._bundle(),
        ))
        self.assertEqual(result.decision, StageDecision.RUN)
        self.assertIn("failed", result.reason.lower())

    def test_deterministic_stage_reuses_artifact_when_everything_matches(self):
        result = compute_stage_decision(StageDecisionInput(
            stage="merge", execution_mode="production", source_hash="abc", version_bundle=self._bundle(),
            has_cached_artifact=True, cached_success=True, upstream_changed=False,
            cached_source_hash="abc", cached_version_bundle=self._bundle(),
        ))
        self.assertEqual(result.decision, StageDecision.REUSE_ARTIFACT)

    def test_non_deterministic_stage_always_runs(self):
        result = compute_stage_decision(StageDecisionInput(
            stage="summarize",
            execution_mode="production", source_hash="abc", version_bundle=self._bundle(),
            has_cached_artifact=True, cached_success=True, upstream_changed=False,
            cached_source_hash="abc", cached_version_bundle=self._bundle(),
        ))
        self.assertEqual(result.decision, StageDecision.RUN)


class ResolveVersionBundleTest(unittest.TestCase):
    def test_prefers_explicit_args_over_env(self):
        with mock.patch.dict(os.environ, {"PROMPT_VERSION": "env-p", "SCHEMA_VERSION": "env-s"}, clear=False):
            bundle = resolve_version_bundle(prompt_version="explicit")
            self.assertEqual(bundle.prompt_version, "explicit")
            self.assertEqual(bundle.schema_version, "env-s")

    def test_default_extraction_mode_is_production(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            bundle = resolve_version_bundle()
            self.assertEqual(bundle.extraction_mode, "production")


def _make_args(root: Path) -> Namespace:
    return Namespace(
        file=str(root / "input.md"),
        folder=None,
        domain="SUPPLY_CHAIN",  # explicit domain: skip AUTO document analysis (LLM)
        output_dir=str(root / "output"),
        run_root=str(root / "output"),
        worker_count=1,
    )


class ExtractVersionBundleWiringTest(unittest.TestCase):
    """run_extract forwards the resolved version bundle into the cache layer."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        source = self.root / "input.md"
        source.write_text("# Small doc\n\nSome content here.\n", encoding="utf-8")
        os.environ["EXTRACT_ENABLE_CANDIDATE_GEN"] = "0"
        os.environ["PIPELINE_EXECUTION_MODE"] = "degraded"

    def tearDown(self):
        os.environ.pop("EXTRACT_ENABLE_CANDIDATE_GEN", None)
        os.environ.pop("PIPELINE_EXECUTION_MODE", None)

    def test_version_bundle_is_forwarded_to_cache(self):
        captured = {}

        class _RecordingCache:
            def get_cached_result(self, path, version_bundle=None):
                captured["read_bundle"] = version_bundle
                return None

            def cache_result(self, *args, **kwargs):
                captured["write_bundle"] = kwargs.get("version_bundle")
                return "hash"

        with mock.patch("src.pipeline.stages.extract.DocumentCache", return_value=_RecordingCache()), \
             mock.patch("src.pipeline.stages.extract.get_l25_skill_loader", return_value=mock.MagicMock(get_stats=lambda: {}, find_matching_skills=lambda text: [])), \
             mock.patch("ontology_compiler.compile_source_file", return_value={
                 "status": "SUCCESS",
                 "data": {"nodes": [], "instances": [], "relationships": [], "quality_score": 0.0},
             }):
            from src.pipeline.stages.extract import run_extract
            args = _make_args(self.root)
            import asyncio
            result = asyncio.run(run_extract(args))

        self.assertIsNotNone(captured.get("read_bundle"))
        self.assertEqual(captured["read_bundle"].extraction_mode, "degraded")


if __name__ == "__main__":
    unittest.main()