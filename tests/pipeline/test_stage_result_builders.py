"""P1-3 (2026-10-16): stage_result_builders 纯函数契约锁。

对应 MainPipeline 的 3 个薄委托（_make_skipped_stage_result /
_make_degraded_stage_result / _should_skip_stage）。锁住：
- SKIP 结果的 data 字段与 stage_decision 归一化（StageDecision 枚举 / 裸字符串）
- DEGRADE 结果固定 quality_level=DEGRADED
- warnings sink 写入且日志不抛
- skip_reason_for_mode 仅非 PRODUCTION 模式跳过 EMERGENT_REASONING/GUARD
"""

from __future__ import annotations

import unittest

from src.pipeline.pipeline_models import PipelineExecutionMode, PipelineStage
from src.pipeline.stage_decision import StageDecision
from src.pipeline.stage_result_builders import (
    make_degraded_stage_result,
    make_skipped_stage_result,
    skip_reason_for_mode,
)


class MakeSkippedTest(unittest.TestCase):
    def test_enum_decision_is_normalized_to_value(self):
        warnings: list[str] = []
        r = make_skipped_stage_result(
            PipelineStage.MERGE,
            reason="no upstream change",
            decision=StageDecision.SKIP_UNCHANGED,
            run_id="run-1",
            warnings=warnings,
        )
        self.assertTrue(r.success)
        self.assertEqual(r.stage, PipelineStage.MERGE.value)
        self.assertEqual(r.data["stage_decision"], StageDecision.SKIP_UNCHANGED.value)
        self.assertTrue(r.data["skipped"])
        self.assertEqual(r.data["reason"], "no upstream change")
        self.assertEqual(r.data["source_hash"], "run-1")
        self.assertEqual(r.elapsed_seconds, 0.0)
        self.assertEqual(r.run_dir, "")
        self.assertEqual(r.fragment_path, "")
        self.assertEqual(len(warnings), 1)
        self.assertIn(StageDecision.SKIP_UNCHANGED.value, warnings[0])

    def test_plain_string_decision_is_accepted(self):
        warnings: list[str] = []
        r = make_skipped_stage_result(
            PipelineStage.EXTRACT,
            reason="r",
            decision="SKIP_UNCHANGED",  # duck-typed, not an enum member instance
            run_id="run-2",
            warnings=warnings,
        )
        self.assertEqual(r.data["stage_decision"], "SKIP_UNCHANGED")
        self.assertIn("[SKIP_UNCHANGED] Skipped", r.message)


class MakeDegradedTest(unittest.TestCase):
    def test_degraded_marks_quality_level_and_warns(self):
        warnings: list[str] = []
        r = make_degraded_stage_result(
            PipelineStage.RELATIONSHIP,
            reason="llm unavailable",
            run_id="run-3",
            warnings=warnings,
        )
        self.assertTrue(r.success)
        self.assertEqual(r.data["stage_decision"], "DEGRADE")
        self.assertEqual(r.data["quality_level"], "DEGRADED")
        self.assertEqual(r.data["source_hash"], "run-3")
        self.assertEqual(len(warnings), 1)
        self.assertIn("Degraded stage", warnings[0])
        self.assertIn("[DEGRADE]", r.message)

    def test_message_and_data_do_not_leak_between_calls(self):
        w1: list[str] = []
        w2: list[str] = []
        r1 = make_degraded_stage_result(PipelineStage.AUDIT, reason="a", run_id="r1", warnings=w1)
        r2 = make_degraded_stage_result(PipelineStage.EXTRACT, reason="b", run_id="r2", warnings=w2)
        self.assertEqual(r1.data["source_hash"], "r1")
        self.assertEqual(r2.data["source_hash"], "r2")
        self.assertEqual(len(w1), 1)
        self.assertEqual(len(w2), 1)


class SkipReasonForModeTest(unittest.TestCase):
    def test_production_never_skips(self):
        for stage in (PipelineStage.EMERGENT_REASONING, PipelineStage.GUARD, PipelineStage.EXTRACT):
            self.assertIsNone(skip_reason_for_mode(PipelineExecutionMode.PRODUCTION, stage))

    def test_non_production_skips_only_emergent_and_guard(self):
        mode = PipelineExecutionMode.OFFLINE
        for stage in (PipelineStage.EMERGENT_REASONING, PipelineStage.GUARD):
            reason = skip_reason_for_mode(mode, stage)
            self.assertIsNotNone(reason)
            self.assertIn(stage.value, reason)
            self.assertIn(mode.value, reason)
        for stage in (PipelineStage.EXTRACT, PipelineStage.MERGE, PipelineStage.EXPORT):
            self.assertIsNone(skip_reason_for_mode(mode, stage))


class ParityWithPipelineDelegateTest(unittest.TestCase):
    """薄委托必须与纯函数等价（属性级，不构造完整 pipeline）。"""

    def test_pipeline_delegate_delegates(self):
        from src.pipeline.main_pipeline import MainPipeline
        import inspect

        src = inspect.getsource(MainPipeline._make_skipped_stage_result)
        self.assertIn("stage_result_builders.make_skipped_stage_result", src)
        src = inspect.getsource(MainPipeline._make_degraded_stage_result)
        self.assertIn("stage_result_builders.make_degraded_stage_result", src)
        src = inspect.getsource(MainPipeline._should_skip_stage)
        self.assertIn("stage_result_builders.skip_reason_for_mode", src)


if __name__ == "__main__":
    unittest.main()
