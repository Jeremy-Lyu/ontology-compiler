from __future__ import annotations

"""Characterization tests for MainPipeline.run (execution plan Phase 1).

Locks the current orchestration behavior of run(): stage loop, failure policy,
status computation, and publishable flag — without invoking real LLM / network /
filesystem side effects. These tests are the regression baseline for the
Phase 3 refactoring of MainPipeline.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.pipeline.main_pipeline import (
    MainPipeline,
    PipelineContext,
    PipelineExecutionMode,
    PipelineExecutionResult,
    PipelineQualityStatus,
    PipelineStage,
    PipelineStatus,
)
from src.pipeline.stages.common import StageResult

ALL_STAGES = [s.value for s in PipelineStage]


class _FakeBridge:
    def get_stats(self):
        return {"skills_count": 0}


class _FakeRouter:
    def get_skills_for_stage(self, stage, domain):
        return []


class _FakeChain:
    def run_full_chain(self, pipeline_context):
        return True, []


class _FakeContinuity:
    def should_process_file(self, path):
        return True


class _FakeSummary:
    def to_dict(self):
        return {}


class _FakeClassification:
    def summarize(self, entities):
        return _FakeSummary()


def _stage_result(stage: str, success: bool) -> StageResult:
    data = {"nodes": [], "instances": [], "relationships": [], "errors": []}
    if not success:
        data["errors"] = [f"{stage} failed"]
    return StageResult(stage=stage, success=success, message="ok" if success else "boom", data=data)


class MainPipelineRunCharacterizationTest(unittest.TestCase):
    """Characterize run() status/publishable behavior for the four core paths."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        source = root / "source.txt"
        source.write_text("dummy source content", encoding="utf-8")
        self.source_path = str(source)
        self.output_dir = str(root / "output")
        self._patches = [
            mock.patch("src.pipeline.main_pipeline.get_skill_pipeline_bridge", return_value=_FakeBridge()),
            mock.patch("src.pipeline.main_pipeline.get_pipeline_skill_router", return_value=_FakeRouter()),
            mock.patch("src.pipeline.main_pipeline.get_validation_chain", return_value=_FakeChain()),
            mock.patch("src.pipeline.main_pipeline.get_continuity_service", return_value=_FakeContinuity()),
            mock.patch("src.pipeline.main_pipeline.get_asset_packet_service"),
            mock.patch("src.pipeline.main_pipeline.get_asset_index_service"),
            mock.patch("src.pipeline.main_pipeline.get_retrieval_asset_packet_service"),
            mock.patch("src.pipeline.main_pipeline.get_pipeline_result_artifact_service"),
            mock.patch(
                "src.application.services.classification_service.get_classification_service",
                return_value=_FakeClassification(),
            ),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(self._tear_down_patches)

    def _tear_down_patches(self):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def _build_pipeline(self) -> MainPipeline:
        context = PipelineContext(
            source_path=self.source_path,
            domain="SUPPLY_CHAIN",
            run_id="CHAR-001",
            output_dir=self.output_dir,
            metadata={
                "quality_gate": {
                    "validation_passed": True,
                    "quality_score": 0.9,
                    "quality_reason": "quality thresholds met",
                }
            },
        )
        pipeline = MainPipeline(context)
        # Neutralize infrastructure side effects so the characterization focuses
        # on orchestration/status logic.
        pipeline._continuity_service = _FakeContinuity()
        pipeline._begin_run_continuity = mock.Mock()
        pipeline._end_run_continuity = mock.Mock()
        pipeline._generate_asset_packet = mock.Mock()
        pipeline._log_summary = mock.Mock()
        pipeline._run_post_execution_validation = mock.Mock()
        pipeline._apply_stage_skills = mock.AsyncMock()
        pipeline._validate_stage_output = mock.Mock()
        pipeline._register_stage_artifacts = mock.Mock()
        return pipeline

    async def _run_with_handler(self, pipeline: MainPipeline, handler) -> object:
        with mock.patch.object(MainPipeline, "_stage_handler", new=mock.AsyncMock(side_effect=handler)):
            return await pipeline.run()

    def _run(self, pipeline: MainPipeline, handler) -> object:
        return asyncio_runner(self._run_with_handler(pipeline, handler))

    def test_all_stages_succeed_status_success_publishable(self):
        pipeline = self._build_pipeline()
        result = self._run(
            pipeline,
            lambda stage: _stage_result(stage.value, success=True),
        )
        self.assertEqual(result.status, PipelineStatus.SUCCESS)
        self.assertTrue(result.publishable)
        self.assertEqual(result.stats.completed_stages, len(ALL_STAGES))
        self.assertEqual(result.stats.failed_stages, 0)
        self.assertEqual(set(result.stage_results), set(ALL_STAGES))
        self.assertFalse(result.errors)

    def test_success_result_to_dict_snapshot_shape(self):
        pipeline = self._build_pipeline()
        result = self._run(
            pipeline,
            lambda stage: _stage_result(stage.value, success=True),
        )
        snapshot = result.to_dict()
        self.assertEqual(
            set(snapshot.keys()),
            {
                "run_id",
                "status",
                "quality_status",
                "quality_reason",
                "validation_passed",
                "quality_score",
                "context",
                "stats",
                "stage_results",
                "output_files",
                "errors",
                "warnings",
                "publishable",
                "execution_mode",
                "execution_reason",
                "metadata",
            },
        )
        self.assertEqual(snapshot["status"], "success")
        self.assertEqual(snapshot["quality_status"], "success")
        self.assertTrue(snapshot["validation_passed"])
        self.assertGreaterEqual(snapshot["quality_score"], 0.6)
        self.assertEqual(snapshot["context"]["domain"], "SUPPLY_CHAIN")
        self.assertEqual(snapshot["stats"]["completed_stages"], len(ALL_STAGES))
        self.assertEqual(sorted(snapshot["stage_results"].keys()), sorted(ALL_STAGES))
        self.assertIn("preflight", snapshot["stage_results"])
        self.assertEqual(
            set(snapshot["stage_results"]["preflight"].keys()),
            {"stage", "success", "message", "data", "elapsed_seconds", "run_dir", "fragment_path", "artifact_paths"},
        )
        self.assertTrue(snapshot["publishable"])
        self.assertEqual(snapshot["output_files"], [])
        self.assertEqual(snapshot["errors"], [])
        self.assertEqual(snapshot["warnings"], [])

    def test_mid_stage_failure_status_partial(self):
        pipeline = self._build_pipeline()
        result = self._run(
            pipeline,
            lambda stage: _stage_result(stage.value, success=stage != PipelineStage.MERGE),
        )
        self.assertEqual(result.status, PipelineStatus.PARTIAL)
        self.assertFalse(result.publishable)
        self.assertEqual(result.stats.completed_stages, len(ALL_STAGES) - 1)
        self.assertEqual(result.stats.failed_stages, 1)
        self.assertIn("merge failed", result.errors[0])

    def test_preflight_failure_stops_pipeline_status_failed(self):
        pipeline = self._build_pipeline()
        result = self._run(
            pipeline,
            lambda stage: _stage_result(stage.value, success=stage != PipelineStage.PREFLIGHT),
        )
        self.assertEqual(result.status, PipelineStatus.FAILED)
        self.assertFalse(result.publishable)
        self.assertEqual(result.stats.completed_stages, 0)
        self.assertEqual(result.stats.failed_stages, 1)
        # Only preflight ran before the hard stop.
        self.assertEqual(set(result.stage_results), {"preflight"})

    def test_contract_validation_failure_status_failed_no_stages(self):
        pipeline = self._build_pipeline()
        pipeline._run_contract_validation_chain = mock.Mock(return_value=False)
        with mock.patch.object(MainPipeline, "_stage_handler", new=mock.AsyncMock()) as stage_handler:
            result = asyncio_runner(pipeline.run())
        self.assertEqual(result.status, PipelineStatus.FAILED)
        self.assertFalse(result.publishable)
        self.assertEqual(result.stats.completed_stages, 0)
        self.assertEqual(result.stats.failed_stages, 0)
        stage_handler.assert_not_awaited()
        self.assertIn("Contract validation chain failed", result.errors)

    def test_unknown_stage_raises_value_error(self):
        pipeline = self._build_pipeline()
        with self.assertRaises(ValueError) as ctx:
            asyncio_runner(pipeline._stage_handler(mock.Mock(value="NOT_A_STAGE")))
        self.assertIn("Unknown stage", str(ctx.exception))

    def test_guard_degraded_empty_failure_continues_offline(self):
        pipeline = self._build_pipeline()
        pipeline._set_execution_mode = mock.Mock()
        stage_result = _stage_result(PipelineStage.GUARD.value, success=False)
        stage_result.data.update({"quality_level": "DEGRADED", "nodes": [], "instances": [], "relationships": [], "fragment_path": None})
        should_stop = pipeline._handle_failed_stage(PipelineStage.GUARD, stage_result)
        self.assertFalse(should_stop)
        pipeline._set_execution_mode.assert_called_once_with(PipelineExecutionMode.OFFLINE, "guard_returned_degraded_empty_output")

    def test_guard_hard_failure_stops_pipeline(self):
        pipeline = self._build_pipeline()
        pipeline._set_execution_mode = mock.Mock()
        stage_result = _stage_result(PipelineStage.GUARD.value, success=False)
        stage_result.data.update({"quality_level": "FAIL", "nodes": [1], "instances": [], "relationships": []})
        should_stop = pipeline._handle_failed_stage(PipelineStage.GUARD, stage_result)
        self.assertTrue(should_stop)
        pipeline._set_execution_mode.assert_not_called()

    def test_extract_stage_quality_aggregates_per_file_scores(self):
        pipeline = self._build_pipeline()
        stage_result = StageResult(
            stage="extract",
            success=True,
            message="ok",
            data={
                "extraction_quality": {
                    "status": "partial",
                    "validation_passed": False,
                    "files": [
                        {"file": "a.pptx", "quality_score": 0.8, "validation_passed": True},
                        {"file": "b.pptx", "quality_score": 0.4, "validation_passed": False},
                    ],
                }
            },
        )
        pipeline._stage_results["extract"] = stage_result
        quality = pipeline._extract_stage_quality()
        self.assertEqual(quality["validation_passed"], False)
        self.assertEqual(quality["quality_score"], 0.6)
        self.assertEqual(quality["extraction_quality_status"], "partial")

    def test_derive_quality_status_gate_missing_does_not_fabricate_score(self):
        """A-2 (2026-08-15): when the governance quality gate did not run, the
        derive must NOT fall back to extract-stage per-file averages as a score
        source — that produced misleading duplicate-quality conclusions (VSM
        showed '0.87 silver' while the gate never ran). Output present now
        yields PARTIAL_SUCCESS with score 0.0 and an explicit 'gate did not run'
        reason; no fabricated number."""
        pipeline = self._build_pipeline()
        stage_result = StageResult(
            stage="extract",
            success=True,
            message="ok",
            data={
                "extraction_quality": {
                    "status": "passed",
                    "validation_passed": True,
                    "files": [{"file": "a.pptx", "quality_score": 0.85, "validation_passed": True}],
                }
            },
        )
        pipeline._stage_results["extract"] = stage_result
        pipeline._stats.entity_count = 10
        pipeline._stats.relation_count = 3
        pipeline.context.metadata["quality_gate"] = {}
        quality_status, quality_reason, validation_passed, quality_score = pipeline._derive_quality_status(PipelineStatus.SUCCESS)
        self.assertEqual(quality_status, PipelineQualityStatus.PARTIAL_SUCCESS)
        self.assertFalse(validation_passed)
        self.assertEqual(quality_score, 0.0)
        self.assertIn("quality gate did not run", quality_reason)

    def test_derive_quality_status_no_output_and_no_gate_is_failure(self):
        """A-2: gate missing AND no structured output must be FAILURE, not a
        fabricated-success from extract averages."""
        pipeline = self._build_pipeline()
        pipeline.context.metadata["quality_gate"] = {}
        pipeline._stats.entity_count = 0
        pipeline._stats.relation_count = 0
        quality_status, quality_reason, validation_passed, quality_score = pipeline._derive_quality_status(PipelineStatus.SUCCESS)
        self.assertEqual(quality_status, PipelineQualityStatus.FAILURE)
        self.assertFalse(validation_passed)
        self.assertEqual(quality_score, 0.0)
        self.assertIn("no publishable structured output", quality_reason)

    def test_degraded_with_output_is_not_failure(self):
        """Regression: output produced but validation degraded must NOT be
        reported as FAILURE with 'no publishable structured output'.
        A-2: score now comes only from the governance gate; when the gate ran
        with a low validation the output is still DEGRADED_SUCCESS/PARTIAL."""
        pipeline = self._build_pipeline()
        pipeline.context.metadata["quality_gate"] = {
            "validation_passed": False,
            "quality_score": 0.69,
            "quality_reason": "gold",
        }
        stage_result = StageResult(
            stage="extract",
            success=True,
            message="ok",
            data={
                "extraction_quality": {
                    "status": "partial",
                    "validation_passed": False,
                    "files": [{"file": "a.pptx", "quality_score": 0.69, "validation_passed": False}],
                }
            },
        )
        pipeline._stage_results["extract"] = stage_result
        pipeline._stats.entity_count = 50
        pipeline._stats.relation_count = 30
        quality_status, quality_reason, validation_passed, quality_score = pipeline._derive_quality_status(PipelineStatus.SUCCESS)
        self.assertIn(quality_status, (PipelineQualityStatus.DEGRADED_SUCCESS, PipelineQualityStatus.PARTIAL_SUCCESS))
        self.assertNotEqual(quality_reason, "no publishable structured output")
        self.assertFalse(validation_passed)
        self.assertGreaterEqual(quality_score, 0.6)

    def test_quality_gate_gets_live_stage_data_without_manifest(self):
        """Regression: _collect_pipeline_output must not return None when
        run_manifest.json is not persisted yet (quality gate runs inside run())."""
        pipeline = self._build_pipeline()
        extract_stage = _stage_result("extract", success=True)
        extract_stage.data = {
            "nodes": [{"meta_id": "L1-001", "name": "SAP", "meta_class": "System"}],
            "instances": [],
            "relationships": [{"from": "L1-001", "to": "L1-001", "type": "uses"}],
            "metadata": {"classification_metrics": {"l1_count": 1}},
        }
        pipeline._stage_results["extract"] = extract_stage
        pipeline._stats.entity_count = 1
        pipeline._stats.relation_count = 1
        output = pipeline._collect_pipeline_output()
        self.assertIsNotNone(output)
        self.assertTrue(any("SAP" in str(n.get("name")) for n in output.get("nodes", [])))
        self.assertEqual(len(output.get("relationships", [])), 1)

    def test_normalize_artifact_path_keeps_real_fragment(self):
        """Regression: normalize must keep the real on-disk fragment path
        (.../files/0000/fragment_X.json) instead of flattening it into a path
        that does not exist."""
        pipeline = self._build_pipeline()
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            real_fragment = root / "extract" / "files" / "0000" / "fragment_X.json"
            real_fragment.parent.mkdir(parents=True)
            real_fragment.write_text("{}", encoding="utf-8")
            normalized = pipeline._normalize_artifact_path(str(real_fragment))
            self.assertTrue(Path(normalized).is_file())

    def test_manifest_payload_includes_resolved_config_snapshot(self):
        """A-3 regression: every run's manifest must carry the resolved LLM /
        budget / version config so runs can be replayed and compared."""
        import os
        pipeline = self._build_pipeline()
        os.environ["LLM_MAX_JSON_RETRIES"] = "1"
        os.environ["EXTRACT_FILE_TIMEOUT"] = "1200"
        try:
            snapshot = pipeline._build_resolved_config_snapshot()
        finally:
            del os.environ["LLM_MAX_JSON_RETRIES"]
            del os.environ["EXTRACT_FILE_TIMEOUT"]
        self.assertIn("llm", snapshot)
        self.assertIn("budget", snapshot)
        self.assertIn("versions", snapshot)
        self.assertEqual(snapshot["llm"]["max_json_retries"], "1")
        self.assertEqual(snapshot["budget"]["extract_file_timeout"], "1200")
        # Versions reflect the resolved env (may be set by config/env.env);
        # assert the snapshot mirrors env rather than hardcoding a value.
        self.assertEqual(snapshot["versions"]["schema_version"], os.environ.get("SCHEMA_VERSION", ""))
        self.assertEqual(snapshot["versions"]["prompt_version"], os.environ.get("PROMPT_VERSION", ""))

        # The manifest payload embeds the snapshot for persistence.
        from src.pipeline.stages.common import StageResult
        result = PipelineExecutionResult(
            run_id="MANIFEST-001",
            status=PipelineStatus.SUCCESS,
            context=pipeline.context,
            stats=pipeline._stats,
            stage_results={"preflight": StageResult("preflight", True, "ok")},
            output_files=[],
            errors=[],
            warnings=[],
            publishable=True,
        )
        payload = pipeline._build_manifest_payload(result, result.to_dict(), {})
        self.assertIn("resolved_config", payload)
        self.assertIn("llm", payload["resolved_config"])

    def test_ontpack_view_maps_flat_output_to_schema_shape(self):
        """Regression: the output-validation schema view must translate the
        flat pipeline payload (entities list, list attributes, object
        provenance) into the ontology_packet.schema.json shape so the JSON
        Schema gate passes on valid output."""
        from src.infrastructure.config.contract_validation import OutputValidator
        from src.infrastructure.config.contract_registry import get_contract_registry
        import jsonschema

        validator = OutputValidator(get_contract_registry())
        sample = {
            "document": "data/sample.pptx",
            "entities": [
                {
                    "id": "L1-001",
                    "name": "SAP",
                    "type": "System",
                    "layer": "L1",
                    "status": "active",
                    "attributes": [{"name": "vendor", "value": "SAP"}],
                    "provenance": {"source_file": "f.pptx", "run_id": "r", "stage": "extract"},
                }
            ],
            "relationships": [
                {"source_id": "L1-001", "target_id": "L2-001", "relation_type": "uses", "provenance": {"source_file": "f.pptx"}}
            ],
            "metadata": {},
        }
        view = validator._ontpack_view(sample)
        self.assertIsInstance(view["document"], dict)
        self.assertIn("L1", view["entities"])
        self.assertEqual(view["entities"]["L1"][0]["attributes"], {"vendor": "SAP"})
        self.assertNotIn("provenance", view["relations"][0])

        schema = get_contract_registry().schema_registry.load_schema("ontology_packet")
        self.assertIsNotNone(schema, "ontology_packet schema must be registered")
        jsonschema.validate(instance=view, schema=schema)


class CorePipelineResultBridgeCharacterizationTest(unittest.TestCase):
    """A-1 (2026-08-15): lock the local→canonical bridge shape so the
    pipeline_execution_result contract can be frozen first, then the local
    dataclass can be retired transparently. Covers success / partial / failed
    stage mapping, validation envelope, cache_stats and artifact refs."""

    def _build_result(
        self,
        *,
        status,
        stage_results,
        entity_count=0,
        relation_count=0,
        errors=None,
        warnings=None,
        output_files=None,
        llm_calls=0,
        latency_ms=0.0,
    ) -> "object":
        from src.pipeline.main_pipeline import PipelineExecutionResult
        from src.pipeline.main_pipeline import PipelineStats

        stats = PipelineStats(
            total_stages=len(stage_results),
            completed_stages=sum(1 for s in stage_results.values() if s.success),
            failed_stages=sum(1 for s in stage_results.values() if not s.success),
            total_latency_ms=latency_ms,
            llm_calls=llm_calls,
            entity_count=entity_count,
            relation_count=relation_count,
        )
        return PipelineExecutionResult(
            run_id="A1-001",
            status=status,
            context=MockContext(),
            stats=stats,
            stage_results=stage_results,
            output_files=output_files or [],
            errors=errors or [],
            warnings=warnings or [],
            publishable=(not errors and status == PipelineStatus.SUCCESS),
            quality_status=PipelineQualityStatus.SUCCESS if (not errors and status == PipelineStatus.SUCCESS) else PipelineQualityStatus.PARTIAL_SUCCESS,
            quality_reason="quality thresholds met" if (not errors and status == PipelineStatus.SUCCESS) else "partial output",
            validation_passed=(not errors and status == PipelineStatus.SUCCESS),
            quality_score=0.9 if (not errors and status == PipelineStatus.SUCCESS) else 0.0,
        )

    def _bridge(self):
        from src.pipeline.main_pipeline import MainPipeline

        pipeline = MainPipeline.__new__(MainPipeline)
        pipeline.context = MockContext()
        pipeline._run_id = "A1-001"
        pipeline._session_id = "SESSION-A1-001-0001"
        pipeline._execution_mode = PipelineExecutionMode.PRODUCTION
        return pipeline

    def test_success_maps_canonical_envelope(self):
        pipeline = self._bridge()
        result = self._build_result(
            status=PipelineStatus.SUCCESS,
            stage_results={"extract": _stage_result("extract", True), "merge": _stage_result("merge", True)},
            entity_count=10,
            relation_count=3,
            output_files=["a.json", "b.json"],
            llm_calls=4,
            latency_ms=1234.5,
        )
        core = pipeline._build_core_pipeline_result(result)
        self.assertEqual(core.contract_name, "pipeline_execution_result")
        self.assertEqual(core.run_id, "A1-001")
        self.assertEqual(core.pipeline_name, "main_pipeline")
        self.assertEqual(core.overall_status, "SUCCESS")
        self.assertEqual(core.domain, "SUPPLY_CHAIN")
        self.assertEqual(core.entity_count, 10)
        self.assertEqual(core.relation_count, 3)
        self.assertEqual(len(core.stages), 2)
        self.assertEqual([s.stage for s in core.stages], ["extract", "merge"])
        self.assertEqual([s.status for s in core.stages], ["success", "success"])
        self.assertAlmostEqual(core.elapsed_seconds, 1.2345, places=3)
        self.assertEqual(core.artifact_refs, ["a.json", "b.json"])
        self.assertEqual(core.validation["status"], "passed")
        self.assertTrue(core.validation["publishable"])
        self.assertEqual(core.validation["quality_score"], 0.9)
        self.assertEqual(core.cache_stats["llm_calls"], 4)
        self.assertEqual(core.provenance["source_path"], "data/sample.pptx")

    def test_partial_maps_failed_stage_and_validation(self):
        pipeline = self._bridge()
        result = self._build_result(
            status=PipelineStatus.PARTIAL,
            stage_results={"extract": _stage_result("extract", True), "merge": _stage_result("merge", False)},
            entity_count=5,
            relation_count=1,
            errors=["merge failed"],
            output_files=["a.json"],
        )
        core = pipeline._build_core_pipeline_result(result)
        self.assertEqual(core.overall_status, "PARTIAL")
        self.assertEqual([s.status for s in core.stages], ["success", "failed"])
        self.assertEqual(core.errors, ["merge failed"])
        self.assertEqual(core.validation["status"], "failed")
        self.assertFalse(core.validation["publishable"])
        self.assertEqual(core.validation["quality_status"], "partial_success")

    def test_failed_with_no_stages_maps_empty_envelope(self):
        from src.pipeline.main_pipeline import PipelineStatus

        pipeline = self._bridge()
        result = self._build_result(
            status=PipelineStatus.FAILED,
            stage_results={},
            errors=["contract validation failed"],
        )
        core = pipeline._build_core_pipeline_result(result)
        self.assertEqual(core.overall_status, "FAILED")
        self.assertEqual(core.stages, [])
        self.assertEqual(core.validation["status"], "failed")
        self.assertEqual(core.errors, ["contract validation failed"])

    def test_stage_error_type_and_message_are_preserved(self):
        from src.pipeline.main_pipeline import PipelineStatus
        from src.pipeline.stages.common import StageResult

        pipeline = self._bridge()
        merge = _stage_result("merge", False)
        merge.data = {"error_type": "TransientInfraError", "action_hint": "retry"}
        merge.message = "boom"
        result = self._build_result(
            status=PipelineStatus.PARTIAL,
            stage_results={"merge": merge},
            errors=["boom"],
        )
        core = pipeline._build_core_pipeline_result(result)
        self.assertEqual(core.stages[0].status, "failed")
        self.assertEqual(core.stages[0].error_type, "TransientInfraError")
        self.assertEqual(core.stages[0].action_hint, "retry")
        self.assertEqual(core.stages[0].message, "boom")


class MockContext:
    """Plain stand-in for PipelineContext used only inside bridge tests."""
    source_path = "data/sample.pptx"
    domain = "SUPPLY_CHAIN"
    source_hash = "abc"
    output_dir = "/tmp/out"
    schema_version = "1.0.0"
    skill_version = "2.0.0"
    prompt_version = "3.0.0"
    fast_mode = False
    metadata = {"pipeline_version": "2.0.0"}


class ResolvedConfigSnapshotTest(unittest.TestCase):
    """A-3 regression (2026-08-15): every run manifest must carry a resolved
    LLM/budget/version snapshot with a deterministic behavior fingerprint so a
    run can be replayed and cache-vs-config drift detected."""

    def _snapshot(self, **env_overrides):
        import types
        from src.pipeline.main_pipeline import MainPipeline, _config_fingerprint

        old = {}
        for key, value in env_overrides.items():
            old[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        try:
            pipeline = MainPipeline.__new__(MainPipeline)
            pipeline.context = MockContext()
            snapshot = pipeline._build_resolved_config_snapshot()
            return snapshot
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_snapshot_sections_exist(self):
        snap = self._snapshot(LLM_PROVIDER="deepseek", PROMPT_MAX_LENGTH="12000")
        self.assertIn("llm", snap)
        self.assertIn("budget", snap)
        self.assertIn("versions", snap)
        self.assertEqual(snap["llm"]["provider"], "deepseek")
        self.assertIn("config_fingerprint", snap["versions"])
        self.assertNotIn("api_key", json.dumps(snap).lower())

    def test_fingerprint_changes_when_behavior_knob_changes(self):
        from src.pipeline.main_pipeline import _config_fingerprint

        fp_base = self._snapshot(PROMPT_MAX_LENGTH="12000")["versions"]["config_fingerprint"]
        fp_changed = self._snapshot(PROMPT_MAX_LENGTH="9000")["versions"]["config_fingerprint"]
        self.assertNotEqual(fp_base, fp_changed)

    def test_fingerprint_stable_across_secret_change(self):
        from src.pipeline.main_pipeline import _config_fingerprint

        fp_a = self._snapshot(DEEPSEEK_API_KEY="secret1")["versions"]["config_fingerprint"]
        fp_b = self._snapshot(DEEPSEEK_API_KEY="secret2")["versions"]["config_fingerprint"]
        self.assertEqual(fp_a, fp_b)


class PipelineOutputMapperTest(unittest.TestCase):
    """B1 regression (2026-08-15): schema-alias normalization extracted from
    MainPipeline._collect_pipeline_output must preserve the exact alias
    semantics the quality gate / ontpack validator rely on."""

    def test_entity_alias_semantics(self):
        from src.application.transformers.pipeline_output_mapper import normalize_pipeline_entities

        entities = [
            {"meta_id": "L1-001", "canonical_name": "SAP", "entity_type": "System", "layer": "L1"},
            {"entity_id": "L2-009", "meta_name": "Warehouse", "meta_class": "Actor", "layer": "L2"},
            {"name": "Plain", "type": "X"},
        ]
        out = normalize_pipeline_entities(entities)
        self.assertEqual(out[0]["id"], "L1-001")
        self.assertEqual(out[0]["name"], "SAP")
        self.assertEqual(out[0]["type"], "System")
        self.assertEqual(out[1]["id"], "L2-009")
        self.assertEqual(out[1]["name"], "Warehouse")
        self.assertEqual(out[1]["type"], "Actor")
        # non-dict passthrough preserved
        self.assertEqual(len(out), 3)
        self.assertEqual(out[2]["name"], "Plain")

    def test_relationship_endpoint_alias_semantics(self):
        from src.application.transformers.pipeline_output_mapper import normalize_pipeline_relationships

        rels = [
            {"from": "L1-001", "to": "L2-002", "type": "uses"},
            {"source": "A", "target": "B", "type": {"nested": True}},
            {"source_id": "X", "target_id": "Y", "relation_type": "part_of"},
        ]
        out = normalize_pipeline_relationships(rels)
        self.assertEqual(out[0]["source_id"], "L1-001")
        self.assertEqual(out[0]["target_id"], "L2-002")
        self.assertEqual(out[0]["relation_type"], "uses")
        self.assertEqual(out[1]["source_id"], "A")
        self.assertEqual(out[1]["target_id"], "B")
        # type is a dict -> no relation_type alias synthesized
        self.assertNotIn("relation_type", out[1])
        self.assertEqual(out[2]["source_id"], "X")
        self.assertEqual(out[2]["relation_type"], "part_of")

    def test_view_shape_and_stats(self):
        from src.application.transformers.pipeline_output_mapper import assemble_pipeline_output_view

        view = assemble_pipeline_output_view(
            source_path="data/sample.pptx",
            nodes=[{"meta_id": "L1-001", "meta_name": "SAP", "meta_class": "System"}],
            instances=[],
            relationships=[{"from": "L1-001", "to": "L1-001", "type": "self"}],
            metadata={"document_path": "data/sample.pptx"},
            stats={"entity_count": 1, "relation_count": 1},
        )
        self.assertEqual(set(view.keys()), {"document", "entities", "relations", "relationships", "nodes", "instances", "metadata", "stats"})
        self.assertEqual(view["document"], "data/sample.pptx")
        self.assertEqual(view["entities"][0]["id"], "L1-001")
        self.assertEqual(view["relationships"][0]["source_id"], "L1-001")
        self.assertEqual(view["stats"]["entity_count"], 1)

    def test_document_falls_back_to_source_path(self):
        from src.application.transformers.pipeline_output_mapper import assemble_pipeline_output_view

        view = assemble_pipeline_output_view(
            source_path="data/sample.pptx",
            nodes=[],
            instances=[],
            relationships=[],
            metadata={},
            stats={},
        )
        self.assertEqual(view["document"], "data/sample.pptx")
        self.assertEqual(view["entities"], [])
        self.assertEqual(view["stats"]["entity_count"], 0)


class QualityGateUseCaseTest(unittest.TestCase):
    """B1 regression (2026-08-15): governance audit + metadata assembly moved
    into the quality_gate use case. Locks the outcome packaging (fatal/warn
    chips + metadata block) that MainPipeline now wires into state."""

    def test_fatal_outcome(self):
        from src.application.use_cases.quality_gate_use_case import run_quality_gate
        from unittest import mock

        class _Score:
            overall = 0.3

        class _Report:
            result = mock.Mock(value="fail")
            grade = mock.Mock(value="bronze")
            score = _Score()
            critical_issue_count = 2
            high_issue_count = 5

        with mock.patch("src.application.use_cases.quality_gate_use_case.get_governance_engine", return_value=mock.Mock(run_audit=mock.Mock(return_value=_Report()))):
            outcome = run_quality_gate({"metadata": {"classification_metrics": {"l1_count": 1}}})
        self.assertEqual(outcome.fatal_errors, ["Quality gate failed: bronze"])
        self.assertEqual(outcome.warning_messages, [])
        self.assertEqual(outcome.metadata["quality_score"], 0.3)
        self.assertEqual(outcome.metadata["quality_grade"], "bronze")
        self.assertEqual(outcome.metadata["validation_passed"], False)
        self.assertEqual(outcome.metadata["classification_metrics"]["l1_count"], 1)

    def test_warn_outcome_and_classification_carriethrough(self):
        from src.application.use_cases.quality_gate_use_case import run_quality_gate
        from unittest import mock

        class _Score:
            overall = 0.7

        class _Report:
            result = mock.Mock(value="warn")
            grade = mock.Mock(value="silver")
            score = _Score()
            critical_issue_count = 0
            high_issue_count = 1

        with mock.patch("src.application.use_cases.quality_gate_use_case.get_governance_engine", return_value=mock.Mock(run_audit=mock.Mock(return_value=_Report()))):
            outcome = run_quality_gate({"metadata": {"classification_metrics": {"l2_count": 4}}})
        self.assertEqual(outcome.fatal_errors, [])
        self.assertEqual(outcome.warning_messages, ["Quality gate warning: silver"])
        self.assertEqual(outcome.metadata["validation_passed"], True)
        self.assertEqual(outcome.metadata["classification_metrics"]["l2_count"], 4)


class ClassificationMetricsUseCaseTest(unittest.TestCase):
    """B2 regression (2026-08-15): classification entity collection + summarize
    migrated into a use case; locks behavior (nodes/instances only, dict
    filtering, None when empty, to_dict wiring)."""

    def _stage_result(self, data):
        from src.pipeline.stages.common import StageResult

        return StageResult(stage="x", success=True, message="ok", data=data)

    def test_collects_nodes_and_instances_only(self):
        from src.application.use_cases.classification_metrics_use_case import collect_classification_entities

        stage_results = {
            "extract": self._stage_result({"nodes": [{"meta_id": "L1"}, {"meta_id": "L2", "nested": [1]}], "relationships": [{"a": 1}]}),
            "merge": self._stage_result({"instances": [{"meta_id": "I1"}], "nodes": "not-a-list"}),
        }
        entities = collect_classification_entities(stage_results)
        ids = [e["meta_id"] for e in entities]
        self.assertEqual(ids, ["L1", "L2", "I1"])

    def test_summarize_none_when_no_entities(self):
        from src.application.use_cases.classification_metrics_use_case import summarize_classification

        result = summarize_classification({}, get_classification_service=lambda: None)
        self.assertIsNone(result)

    def test_summarize_wires_to_dict(self):
        from src.application.use_cases.classification_metrics_use_case import summarize_classification

        class _Service:
            def summarize(self, entities):
                class _Summary:
                    def to_dict(self):
                        return {"entity_count": len(entities), "l1_count": 1}
                return _Summary()

        stage_results = {"extract": self._stage_result({"nodes": [{"meta_id": "L1"}, {"meta_id": "L2"}]})}
        result = summarize_classification(stage_results, get_classification_service=lambda: _Service())
        self.assertEqual(result, {"entity_count": 2, "l1_count": 1})


class PipelineValidationUseCaseTest(unittest.TestCase):
    """B3 regression (2026-08-15): contract-validation chain and output
    validation migrated to a use case; locks error/warning chip assembly."""

    def test_contract_validation_valid_builds_chips(self):
        from src.application.use_cases.pipeline_validation_use_case import run_contract_validation

        class _Msg:
            def __init__(self, message):
                self.message = message

        class _Report:
            def __init__(self, vid, errors, warnings):
                self.validation_id = vid
                self.errors = errors
                self.warnings = warnings
                self.has_errors = bool(errors)

        class _Chain:
            def run_full_chain(self, ctx):
                self.ctx = ctx
                return True, [
                    _Report("schema", [_Msg("bad field")], []),
                    _Report("harmony", [], ["warning a"]),
                ]

        chain = _Chain()
        outcome = run_contract_validation(chain, source_path="s", domain="d", run_id="r")
        self.assertTrue(outcome.valid)
        self.assertEqual(chain.ctx, {"source_path": "s", "domain": "d", "run_id": "r"})
        self.assertEqual(outcome.error_chips, ["Validation schema: bad field"])
        self.assertEqual(outcome.warning_chips, ["warning a"])

    def test_contract_validation_invalid_flag(self):
        from src.application.use_cases.pipeline_validation_use_case import run_contract_validation

        class _Msg:
            message = "boom"

        class _Report:
            validation_id = "guard"
            errors = [_Msg()]
            warnings = []
            has_errors = True

        class _Chain:
            def run_full_chain(self, ctx):
                return False, [_Report()]

        outcome = run_contract_validation(_Chain(), source_path="s", domain="d", run_id="r")
        self.assertFalse(outcome.valid)
        self.assertEqual(outcome.error_chips, ["Validation guard: boom"])

    def test_output_validation_chips(self):
        from src.application.use_cases.pipeline_validation_use_case import validate_pipeline_output

        class _Error:
            message = "missing id"

        class _Report:
            has_errors = True
            errors = [_Error()]
            warnings = ["w1", "w2"]

        class _Chain:
            def validate_output(self, data):
                self.data = data
                return _Report()

        chain = _Chain()
        outcome = validate_pipeline_output(chain, {"entities": []})
        self.assertFalse(outcome.valid)
        self.assertEqual(chain.data, {"entities": []})
        self.assertEqual(outcome.error_chips, ["Output validation: missing id"])
        self.assertEqual(outcome.warning_chips, ["w1", "w2"])

    def test_output_validation_clean_valid(self):
        from src.application.use_cases.pipeline_validation_use_case import validate_pipeline_output

        class _Report:
            has_errors = False
            errors = []
            warnings = []

        class _Chain:
            def validate_output(self, data):
                return _Report()

        outcome = validate_pipeline_output(_Chain(), {})
        self.assertTrue(outcome.valid)
        self.assertEqual(outcome.error_chips, [])
        self.assertEqual(outcome.warning_chips, [])


class PipelineContinuityUseCaseTest(unittest.TestCase):
    """B4 regression (2026-08-15): run begin/end + input-file traversal +
    artifact-path mapping moved into a use case; locks the duplicate-version
    and slice semantics the run manifest relies on."""

    def test_collect_input_files_file(self):
        from src.application.use_cases.pipeline_continuity_use_case import collect_input_files

        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "a.md"
            source.write_text("x", encoding="utf-8")
            self.assertEqual(collect_input_files(str(source)), [str(source)])

    def test_collect_input_files_folder_ext(self):
        from src.application.use_cases.pipeline_continuity_use_case import collect_input_files

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "one.md").write_text("1", encoding="utf-8")
            sub = root / "sub"
            sub.mkdir()
            (sub / "two.txt").write_text("2", encoding="utf-8")
            (sub / "three.py").write_text("3", encoding="utf-8")
            found = {Path(p).name for p in collect_input_files(str(root))}
            self.assertEqual(found, {"one.md", "two.txt"})

    def test_build_artifact_paths_slices_to_10_and_maps_basename(self):
        from src.application.use_cases.pipeline_continuity_use_case import build_artifact_paths

        files = [f"/tmp/out/f{i}.json" for i in range(12)]
        paths = build_artifact_paths(files)
        self.assertEqual(len(paths), 10)
        self.assertEqual(paths["f0.json"], "/tmp/out/f0.json")
        self.assertNotIn("f11.json", paths)

    def test_begin_end_calls_through_service(self):
        from src.application.use_cases.pipeline_continuity_use_case import begin_run, end_run

        calls = {}

        class _Service:
            def begin_run(self, **kwargs):
                calls["begin_kwargs"] = kwargs

            def end_run(self, **kwargs):
                calls["end_kwargs"] = kwargs

        begin_run(_Service(), run_id="r", session_id="s", input_files=["a.md"])
        self.assertEqual(calls["begin_kwargs"]["pipeline_version"], "1.0")
        self.assertEqual(calls["begin_kwargs"]["input_files"], ["a.md"])

        end_run(
            _Service(),
            run_id="r",
            status="SUCCESS",
            output_path="/tmp/out",
            entity_count=3,
            relation_count=2,
            llm_call_count=4,
            errors=[f"e{i}" for i in range(12)],
            warnings=[],
            artifact_paths={"a.json": "/tmp/a.json"},
            classification_metrics={"l1": 1},
        )
        # errors sliced to 10 in the use case (same as previous MainPipeline)
        self.assertEqual(len(calls["end_kwargs"]["errors"]), 10)
        self.assertEqual(calls["end_kwargs"]["status"], "SUCCESS")
        self.assertEqual(calls["end_kwargs"]["classification_metrics"], {"l1": 1})


class LLMClientFactoryTest(unittest.TestCase):
    """D-1 regression (2026-08-15): all production call sites must construct
    the Rigor LLM client through the shared ``get_llm_client`` factory, which
    caches by provider signature so circuit-breaker / rate-limiter / cache
    state is shared and test resets are explicit."""

    def _with_env(self, **env_overrides):
        old = {}
        for key, value in env_overrides.items():
            old[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        try:
            return True
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_same_signature_shares_instance(self):
        from src.application.bootstrap.llm_bootstrap import get_llm_client, reset_llm_client_cache

        reset_llm_client_cache()
        a = get_llm_client()
        b = get_llm_client()
        self.assertIs(a, b)

    def test_reset_produces_fresh_instance(self):
        from src.application.bootstrap.llm_bootstrap import get_llm_client, reset_llm_client_cache

        reset_llm_client_cache()
        a = get_llm_client()
        reset_llm_client_cache()
        c = get_llm_client()
        self.assertIsNot(a, c)

    def test_no_direct_instantiation_outside_bootstrap(self):
        """Every production instantiation must go through the factory. This
        scans the module source (not runtime) so a future direct
        ``UnifiedLLMClient()`` regressions and pulls it back to the factory."""
        import glob as _glob
        import re as _re

        hits = []
        # Only match actual constructor calls — not comments/docstrings
        # ("UnifiedLLMClient (Pipeline-style)" or "UnifiedLLMClient() — ...").
        call_re = _re.compile(r"(?<![#.\w])UnifiedLLMClient\(\s*(?:provider=|\"|'|\)|$)")
        for path in _glob.glob("ontology_compiler/**/*.py", recursive=True) + _glob.glob("src/**/*.py", recursive=True):
            if "archived" in path or "llm_bootstrap.py" in path or "llm_backend.py" in path:
                continue
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    if call_re.search(line):
                        hits.append(f"{path}:{lineno}")
        self.assertEqual(hits, [], f"direct UnifiedLLMClient() instantiations outside bootstrap: {hits}")


class RelationRewriteSkillTest(unittest.TestCase):
    """C-1 experiment regression (2026-08-15): the relation_rewrite skill must
    be real/executable (not a metadata-only no-op) and deterministically
    rewrite RELATED_TO relations with measurable metrics, with zero LLM cost."""

    def _run(self, relationships):
        from ontology_compiler.skills.builtin.relation_rewrite_skill import RelationRewriteSkill
        from ontology_compiler.skills.base import SkillContext

        skill = RelationRewriteSkill()
        context = SkillContext(config={"relationships": relationships}, domain="SUPPLY_CHAIN")
        return asyncio_runner(skill.execute(context))

    def test_rewrites_related_with_keyword_blob(self):
        result = self._run([
            {"source_id": "A", "target_id": "B", "type": "RELATED_TO", "source": "A", "target": "B", "text": "A depends on B"},
            {"source_id": "C", "target_id": "D", "type": "RELATED_TO", "source": "C", "target": "D", "text": "C supports D"},
            {"source_id": "X", "target_id": "Y", "type": "PART_OF", "source": "X", "target": "Y"},
        ])
        self.assertTrue(result.success)
        self.assertTrue(result.applied)
        output = result.output
        # All three relations rewritten/normalized: 2 RELATED_TO (keyword) +
        # 1 PART_OF (canonical family normalization by governance engine).
        self.assertEqual(output["rewrite_count"], 3)
        self.assertEqual(output["total_count"], 3)
        self.assertEqual(output["related_to_share_before"], round(2 / 3, 4))
        self.assertEqual(output["related_to_share_after"], 0.0)
        rewritten = output["rewritten_relationships"]
        by_id = {r["source_id"]: r for r in rewritten}
        self.assertEqual(by_id["A"]["relation_type"], "DEPENDS_ON")
        self.assertEqual(by_id["C"]["relation_type"], "SUPPORTS")
        self.assertIn("relation_rewrite_skill", str(by_id["A"].get("decision_trace", [])))
        # non-weak relation normalized to engine canonical family relation
        self.assertEqual(by_id["X"]["relation_type"], "SUBSUMES")
        self.assertIn("rewritten_relation_type", by_id["X"])

    def test_skip_when_no_relationships(self):
        result = self._run([])
        self.assertTrue(result.skipped)
        self.assertEqual(result.reason, "no relationships to rewrite")


class PipelineSkillRouterGatingTest(unittest.TestCase):
    """C-1 experiment regression (2026-08-15): the relationship-stage
    relation_rewrite rule must be suppressed by default (production unchanged)
    and only activate when PIPELINE_SKILL_RELATION_REWRITE=1."""

    def _router(self):
        from src.pipeline.pipeline_skill_router import PipelineSkillRouter
        from src.pipeline.pipeline_skill_router import reset_pipeline_skill_router
        reset_pipeline_skill_router()
        return PipelineSkillRouter()

    def test_relationship_rule_off_by_default(self):
        os.environ.pop("PIPELINE_SKILL_RELATION_REWRITE", None)
        router = self._router()
        self.assertEqual(router.get_skills_for_stage("relationship", "SUPPLY_CHAIN"), [])
        self.assertEqual(router.get_skills_for_stage("extract", "SUPPLY_CHAIN"), [])

    def test_relationship_rule_activates_with_flag(self):
        os.environ["PIPELINE_SKILL_RELATION_REWRITE"] = "1"
        try:
            router = self._router()
            self.assertEqual(router.get_skills_for_stage("relationship", "SUPPLY_CHAIN"), ["relation_rewrite"])
            # other stages still empty
            self.assertEqual(router.get_skills_for_stage("audit", "SUPPLY_CHAIN"), [])
        finally:
            os.environ.pop("PIPELINE_SKILL_RELATION_REWRITE", None)


class RelationRewriteSkillBridgeIntegrationTest(unittest.TestCase):
    """C-1 E2E regression (2026-08-15): when the experiment flag is on, the
    pipeline bridge must resolve and actually execute relation_rewrite through
    the canonical registry (async skill awaited), producing real rewrite
    metrics — not the old metadata-only empty-output no-op."""

    def test_bridge_executes_relation_rewrite_when_flag_on(self):
        import asyncio
        os.environ["PIPELINE_SKILL_RELATION_REWRITE"] = "1"
        # Patch asyncio.run inside skill_integration to use the Winsock-safe
        # fallback (skill.execute is fully synchronous despite being async def).
        real_run = asyncio.run

        def _safe_run(coro):
            try:
                return real_run(coro)
            except (OSError, RuntimeError) as exc:
                return _manual_drive_coro(coro, exc)

        try:
            from src.pipeline.skill_integration import get_skill_pipeline_bridge, reset_skill_pipeline_bridge
            reset_skill_pipeline_bridge()
            with mock.patch("asyncio.run", _safe_run):
                bridge = get_skill_pipeline_bridge()
                bridge.load_skills()
                self.assertIn("relation_rewrite", bridge.list_skills())

                context = {
                    "stage": "relationship",
                    "run_id": "C1-E2E",
                    "domain": "SUPPLY_CHAIN",
                    "stage_data": {
                        "relationships": [
                            {"source_id": "A", "target_id": "B", "type": "RELATED_TO", "text": "A depends on B"},
                        ]
                    },
                }
                result = bridge.execute_skill("relation_rewrite", context)
                self.assertTrue(result.success)
                out = result.output.get("output") if isinstance(result.output, dict) else result.output
                self.assertIsInstance(out, dict)
                self.assertEqual(out.get("rewrite_count"), 1)
                self.assertEqual(out.get("related_to_share_before"), 1.0)
                self.assertEqual(out.get("related_to_share_after"), 0.0)
                rels = out.get("rewritten_relationships", [])
                self.assertEqual(rels[0]["relation_type"], "DEPENDS_ON")
        finally:
            from src.pipeline.skill_integration import reset_skill_pipeline_bridge
            reset_skill_pipeline_bridge()
            os.environ.pop("PIPELINE_SKILL_RELATION_REWRITE", None)


class LLMObservabilityUseCaseTest(unittest.TestCase):
    """D-4 regression (2026-08-15): LLM monitor aggregation + per-run reset are
    pure and safe — manifest llm_stats reflects current run only."""

    def test_collect_summary_when_available(self):
        from src.application.use_cases.llm_observability_use_case import collect_llm_stats

        class _Monitor:
            def get_summary(self):
                return {"total_calls": 3, "total_tokens": 1000, "total_cost_estimate": 0.5, "stage_metrics": {"extract": {}}}

        stats = collect_llm_stats(_Monitor())
        self.assertEqual(stats["total_calls"], 3)
        self.assertEqual(stats["total_tokens"], 1000)

    def test_collect_none_returns_empty(self):
        from src.application.use_cases.llm_observability_use_case import collect_llm_stats

        self.assertEqual(collect_llm_stats(None)["total_calls"], 0)
        self.assertEqual(collect_llm_stats(None)["stage_metrics"], {})

    def test_reset_guards_missing_method(self):
        from src.application.use_cases.llm_observability_use_case import reset_llm_monitor

        calls = {"n": 0}

        class _Monitor:
            def reset(self):
                calls["n"] += 1

        reset_llm_monitor(_Monitor())
        reset_llm_monitor(None)
        reset_llm_monitor(object())  # no reset method -> no-op
        self.assertEqual(calls["n"], 1)

    def test_manifest_embeds_llm_stats(self):
        """manifest payload includes llm_stats (may be empty when monitor off)."""
        from src.application.use_cases.llm_observability_use_case import collect_llm_stats

        pipeline = _FakePipelineForManifest()
        payload = pipeline._build_manifest_payload_fake()
        self.assertIn("resolved_config", payload)
        self.assertIn("llm_stats", payload)


class _FakePipelineForManifest:
    """Minimal stand-in exercising _build_manifest_payload without full MainPipeline."""

    def _build_manifest_payload_fake(self):
        from src.pipeline.main_pipeline import MainPipeline
        from src.pipeline.main_pipeline import PipelineExecutionResult, PipelineStatus

        pipeline = MainPipeline.__new__(MainPipeline)
        pipeline.context = _MiniContext()
        pipeline._run_id = "D4-001"
        pipeline._session_id = "SESSION-D4"
        result = PipelineExecutionResult(run_id="D4", status=PipelineStatus.SUCCESS, context=pipeline.context, stats=_MiniStats())
        result_dict = result.to_dict()
        return pipeline._build_manifest_payload(result, result_dict, {"pipeline_execution_result": "x"})


class _MiniContext:
    source_path = "data/sample.pptx"
    domain = "SUPPLY_CHAIN"
    source_hash = ""
    schema_version = ""
    skill_version = ""
    prompt_version = ""
    output_dir = "/tmp/d4"
    fast_mode = False
    enable_rigor_extract = False
    enable_llm_conflict_resolver = True
    regression_mode = False
    execution_mode = "production"
    execution_reason = ""
    metadata = {"artifact_bundles": []}


class _MiniStats:
    """Minimal PipelineStats stand-in (entity/relation counts only)."""
    total_stages = 0
    completed_stages = 0
    failed_stages = 0
    total_latency_ms = 0.0
    llm_calls = 0
    entity_count = 0
    relation_count = 0
    evidence_count = 0


def asyncio_runner(coro):
    """Drive a coroutine to completion, with fallback for Winsock LSP corruption.

    On Windows enterprise environments, Winsock LSP DLL injection can corrupt
    ``socket.socketpair()``, breaking ALL asyncio event loop creation (both
    ProactorEventLoop and SelectorEventLoop) with ``WinError 10038``. This is
    non-deterministic — the same test suite may pass or fail across runs.

    Fallback: manually drive the coroutine via ``.send(None)``. This works for
    characterization tests because every ``await`` point resolves immediately
    (AsyncMock coroutines raise ``StopIteration`` on first ``.send`` without
    yielding). If the coroutine yields a value requiring real scheduling, we
    re-raise the original OSError so the test fails with a clear reason.
    """
    import asyncio
    import sys

    try:
        return asyncio.run(coro)
    except (OSError, RuntimeError) as exc:
        if sys.platform != "win32":
            raise
        # WinError 10038 or similar socket corruption — fall back to manual drive
        return _manual_drive_coro(coro, exc)


def _manual_drive_coro(coro, original_error):
    """Drive a coroutine without an event loop (test-only fallback).

    Works for coroutines whose ``await`` points all resolve immediately (e.g.,
    AsyncMocks). If the coroutine yields a value that requires real I/O
    scheduling, re-raise the original error.
    """
    try:
        while True:
            yielded = coro.send(None)
            # Coroutine yielded a value → needs real event loop scheduling
            coro.close()
            raise original_error from RuntimeError(
                f"Manual drive fallback cannot handle yielded value {yielded!r}; "
                f"coroutine requires real I/O scheduling."
            )
    except StopIteration as stop:
        return stop.value
    except BaseException:
        # Ensure the coroutine is closed on any error path
        try:
            coro.close()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    unittest.main()