from __future__ import annotations

"""Characterization tests for MainPipeline._generate_asset_packet (plan Phase 1).

Locks the current asset-packet aggregation semantics: which stage data keys
are collected (nodes / *_entities / relationships / evidence, plus fragment
and governed-fragment files), how quality_score gates the governed flag, and
that the packet path is appended to the run result. All outbound services are
stubbed; this is the regression baseline for moving packet building out of
MainPipeline (Phase 3).
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.domain.models.core import PipelineExecutionResult
from src.pipeline.main_pipeline import (
    MainPipeline,
    PipelineContext,
    PipelineStats,
    PipelineStatus,
)
from src.pipeline.stages.common import StageResult


class _FakeAssetPacketService:
    def __init__(self):
        self.generate_kwargs = None

    def add_provenance_to_entities(self, entities, **kwargs):
        return entities

    def add_provenance_to_relations(self, relations, **kwargs):
        return relations

    def generate_asset_packet(self, **kwargs):
        self.generate_kwargs = kwargs
        return str(Path("outputs") / "fake" / f"asset_packet_{kwargs['run_id']}.json")

    def load_asset_packet(self, run_id, governed=False):
        return None


class _FakeIndexService:
    def __init__(self):
        self.calls = []

    def index_package(self, run_id, package_data, package_path, governed):
        self.calls.append((run_id, package_path, governed))


class _FakeRetrievalPacketService:
    def __init__(self):
        self.calls = []

    def build_and_save_packet(self, **kwargs):
        self.calls.append(kwargs)
        return ("outputs/fake/retrieval.json", {"ok": True})


def _result(pipeline, stage_results, *, errors=None, warnings=None) -> PipelineExecutionResult:
    return PipelineExecutionResult(
        run_id=pipeline._run_id,
        status=PipelineStatus.SUCCESS,
        context=pipeline.context,
        stats=PipelineStats(),
        stage_results=stage_results,
        output_files=[],
        errors=errors or [],
        warnings=warnings or [],
        publishable=True,
    )


class GenerateAssetPacketCharacterizationTest(unittest.TestCase):
    """Characterize _generate_asset_packet aggregation and gating."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self._patches = [
            mock.patch("src.pipeline.main_pipeline.get_skill_pipeline_bridge"),
            mock.patch("src.pipeline.main_pipeline.get_pipeline_skill_router"),
            mock.patch("src.pipeline.main_pipeline.get_validation_chain"),
            mock.patch("src.pipeline.main_pipeline.get_continuity_service"),
            mock.patch("src.pipeline.main_pipeline.get_asset_packet_service"),
            mock.patch("src.pipeline.main_pipeline.get_asset_index_service"),
            mock.patch("src.pipeline.main_pipeline.get_retrieval_asset_packet_service"),
            mock.patch("src.pipeline.main_pipeline.get_pipeline_result_artifact_service"),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(self._tear_down_patches)

    def _tear_down_patches(self):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def _build_pipeline(self):
        source = self.root / "src.md"
        source.write_text("content", encoding="utf-8")
        context = PipelineContext(
            source_path=str(source), domain="SUPPLY_CHAIN", run_id="ASSET-001", output_dir=str(self.root / "out")
        )
        pipeline = MainPipeline(context)
        self.asset_service = _FakeAssetPacketService()
        self.index_service = _FakeIndexService()
        self.retrieval_service = _FakeRetrievalPacketService()
        pipeline._asset_packet_service = self.asset_service
        pipeline._asset_index_service = self.index_service
        pipeline._retrieval_asset_packet_service = self.retrieval_service
        return pipeline

    def _stage(self, name, *, nodes=None, relationships=None, evidence=None, quality_score=None, fragment_path=None):
        data = {}
        if nodes is not None:
            data["nodes"] = nodes
        if relationships is not None:
            data["relationships"] = relationships
        if evidence is not None:
            data["evidence"] = evidence
        if quality_score is not None:
            data["quality_score"] = quality_score
        return StageResult(
            stage=name, success=True, message="ok", data=data, fragment_path=fragment_path or ""
        )

    def test_aggregates_stage_data_and_appends_package_path(self):
        pipeline = self._build_pipeline()
        node = {"meta_id": "L1-001", "name": "MBOM", "layer": "L1", "meta_class": "Data"}
        rel = {"source": "L1-001", "target": "L2-001", "relation_type": "contains"}
        stages = {
            "extract": self._stage("extract", nodes=[node], relationships=[rel]),
            "audit": self._stage("audit", nodes=[], quality_score=0.85),
        }
        pipeline._stage_results = stages
        result = _result(pipeline, stages)

        pipeline._generate_asset_packet(result)

        # Entities and relations reach the packet service (with provenance glue).
        kwargs = self.asset_service.generate_kwargs
        self.assertIsNotNone(kwargs)
        self.assertEqual(kwargs["entities"], [node])
        self.assertEqual(kwargs["relations"], [rel])
        self.assertEqual(kwargs["metadata"]["quality_score"], 0.85)
        self.assertFalse(kwargs["metadata"]["review_required"])
        # governed = quality_score >= 0.7
        self.assertTrue(kwargs["governed"])
        # Package path appended to the run result.
        self.assertTrue(any(p.endswith("asset_packet_ASSET-001.json") for p in result.output_files))
        self.assertEqual(len(self.retrieval_service.calls), 1)

    def test_low_quality_score_ungoverned_and_review_required(self):
        pipeline = self._build_pipeline()
        stages = {"audit": self._stage("audit", nodes=[], quality_score=0.3)}
        pipeline._stage_results = stages
        result = _result(pipeline, stages)

        pipeline._generate_asset_packet(result)

        kwargs = self.asset_service.generate_kwargs
        self.assertFalse(kwargs["governed"])
        self.assertTrue(kwargs["metadata"]["review_required"])

    def test_fragment_file_is_collected(self):
        pipeline = self._build_pipeline()
        fragment = self.root / "fragment.json"
        fragment.write_text(
            json.dumps(
                {
                    "nodes": [{"meta_id": "L2-009", "name": "SAP", "layer": "L2", "meta_class": "System"}],
                    "relationships": [{"source": "L2-009", "target": "L0-001", "relation_type": "uses"}],
                }
            ),
            encoding="utf-8",
        )
        stages = {"extract": self._stage("extract", fragment_path=str(fragment))}
        pipeline._stage_results = stages
        result = _result(pipeline, stages)

        pipeline._generate_asset_packet(result)

        kwargs = self.asset_service.generate_kwargs
        ids = {e.get("meta_id") for e in kwargs["entities"]}
        self.assertIn("L2-009", ids)
        self.assertEqual(len(kwargs["relations"]), 1)

    def test_invalid_fragment_path_is_skipped_without_error(self):
        pipeline = self._build_pipeline()
        stages = {"extract": self._stage("extract", nodes=[], fragment_path=str(self.root / "missing.json"))}
        pipeline._stage_results = stages
        result = _result(pipeline, stages)

        pipeline._generate_asset_packet(result)  # must not raise

        self.assertIsNotNone(self.asset_service.generate_kwargs)
        self.assertEqual(self.asset_service.generate_kwargs["entities"], [])


if __name__ == "__main__":
    unittest.main()