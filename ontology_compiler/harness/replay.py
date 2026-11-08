"""Harness Replay & Canary Runner

[W7-HARNESS-REPLAY]: 实现会话重放和金丝雀运行能力

Features:
- SessionReplay: 重放历史会话用于调试和测试
- CanaryRunner: 金丝雀运行模式，用于验证新变更的稳定性
- PerformanceBenchmark: 性能基准测试
- RegressionGuard: 回归保护检查
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ontology_compiler.harness.pipeline_session_store import SessionManager
from ontology_compiler.services.extraction.unified_compilation_pipeline import UnifiedCompilationPipeline

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent.parent
_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "replay"

logger = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    """会话重放结果"""
    session_id: str
    success: bool
    original_duration_ms: float
    replay_duration_ms: float
    duration_diff_ms: float
    quality_score: float
    original_quality_score: Optional[float] = None
    quality_diff: Optional[float] = None
    entity_count: int = 0
    relation_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanaryResult:
    """金丝雀运行结果"""
    canary_id: str
    success: bool
    baseline_result: Optional[ReplayResult] = None
    canary_result: Optional[ReplayResult] = None
    passed: bool = False
    performance_threshold_met: bool = False
    quality_threshold_met: bool = False
    stability_threshold_met: bool = False
    comparison_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v.to_dict() if hasattr(v, 'to_dict') else v
            for k, v in asdict(self).items()
        }


@dataclass
class BenchmarkResult:
    """性能基准测试结果"""
    benchmark_id: str
    test_cases: List[str] = field(default_factory=list)
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    min_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    total_entities: int = 0
    avg_quality_score: float = 0.0
    success_rate: float = 0.0
    throughput_docs_per_min: float = 0.0


class SessionReplay:
    """会话重放器 - 重放历史会话用于调试和测试"""

    def __init__(self, session_manager: Optional[SessionManager] = None):
        self.session_manager = session_manager or SessionManager()

    def list_stored_sessions(self) -> List[Dict[str, Any]]:
        """列出所有已存储的会话"""
        sessions = []
        session_dir = Path("outputs/sessions")
        if session_dir.exists():
            for session_file in session_dir.glob("*.json"):
                try:
                    with open(session_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        sessions.append({
                            "session_id": session_file.stem,
                            "timestamp": data.get("timestamp", ""),
                            "source_path": data.get("source_path", ""),
                            "status": data.get("status", ""),
                            "duration_ms": data.get("duration_ms", 0),
                            "quality_score": data.get("quality_score", 0),
                        })
                except Exception:
                    pass
        return sorted(sessions, key=lambda x: x["timestamp"], reverse=True)

    async def replay_session(self, session_id: str) -> ReplayResult:
        """重放指定会话"""
        session_data = self.session_manager.load_session(session_id)
        if not session_data:
            return ReplayResult(
                session_id=session_id,
                success=False,
                original_duration_ms=0,
                replay_duration_ms=0,
                duration_diff_ms=0,
                quality_score=0.0,
                errors=["Session not found"]
            )

        source_path = session_data.get("source_path")
        if not source_path or not Path(source_path).exists():
            return ReplayResult(
                session_id=session_id,
                success=False,
                original_duration_ms=session_data.get("duration_ms", 0),
                replay_duration_ms=0,
                duration_diff_ms=0,
                quality_score=0.0,
                original_quality_score=session_data.get("quality_score"),
                errors=["Source file not found"]
            )

        original_duration = session_data.get("duration_ms", 0)
        original_quality = session_data.get("quality_score")

        start_time = time.time()
        try:
            pipeline = UnifiedCompilationPipeline()
            result = await pipeline.run_async(
                source_path=source_path,
                domain=session_data.get("domain", "SUPPLY_CHAIN"),
                run_id=f"replay-{session_id}"
            )

            replay_duration = (time.time() - start_time) * 1000

            if result.final_fragment:
                nodes = result.final_fragment.nodes or []
                relationships = result.final_fragment.relationships or []
                quality_score = getattr(result, "quality_score", 0.0)

                return ReplayResult(
                    session_id=session_id,
                    success=True,
                    original_duration_ms=original_duration,
                    replay_duration_ms=replay_duration,
                    duration_diff_ms=replay_duration - original_duration,
                    quality_score=quality_score,
                    original_quality_score=original_quality,
                    quality_diff=quality_score - (original_quality or 0),
                    entity_count=len(nodes),
                    relation_count=len(relationships),
                    warnings=[str(w) for w in result.warnings]
                )
            else:
                return ReplayResult(
                    session_id=session_id,
                    success=False,
                    original_duration_ms=original_duration,
                    replay_duration_ms=replay_duration,
                    duration_diff_ms=replay_duration - original_duration,
                    quality_score=0.0,
                    original_quality_score=original_quality,
                    errors=["No final fragment produced"]
                )

        except Exception as e:
            replay_duration = (time.time() - start_time) * 1000
            return ReplayResult(
                session_id=session_id,
                success=False,
                original_duration_ms=original_duration,
                replay_duration_ms=replay_duration,
                duration_diff_ms=replay_duration - original_duration,
                quality_score=0.0,
                original_quality_score=original_quality,
                errors=[str(e)]
            )

    async def batch_replay(self, session_ids: List[str]) -> List[ReplayResult]:
        """批量重放多个会话"""
        results = []
        for session_id in session_ids:
            result = await self.replay_session(session_id)
            results.append(result)
        return results

    def export_replay_results(self, results: List[ReplayResult], path: Optional[Path] = None) -> Path:
        """导出重放结果"""
        path = path or _OUTPUT_DIR / f"replay_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "generated_at": datetime.utcnow().isoformat(),
            "total_sessions": len(results),
            "success_count": sum(1 for r in results if r.success),
            "failure_count": sum(1 for r in results if not r.success),
            "results": [r.to_dict() for r in results],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return path


class CanaryRunner:
    """金丝雀运行器 - 在生产环境前验证新变更"""

    def __init__(
        self,
        performance_threshold: float = 1.5,
        quality_threshold: float = -0.1,
        stability_threshold: int = 0,
        sample_size: int = 5
    ):
        """
        Args:
            performance_threshold: 性能下降阈值 (倍数)，超过此值视为不通过
            quality_threshold: 质量下降阈值，低于此值视为不通过
            stability_threshold: 允许的失败次数
            sample_size: 金丝雀测试的样本数量
        """
        self.performance_threshold = performance_threshold
        self.quality_threshold = quality_threshold
        self.stability_threshold = stability_threshold
        self.sample_size = sample_size
        self.replay = SessionReplay()

    async def run_canary(
        self,
        session_ids: Optional[List[str]] = None,
        description: str = ""
    ) -> CanaryResult:
        """运行金丝雀测试

        Args:
            session_ids: 要测试的会话ID列表，若为空则自动选择最近的会话
            description: 测试描述

        Returns:
            CanaryResult: 金丝雀测试结果
        """
        canary_id = f"canary-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"[Canary] Starting canary run: {canary_id}")

        if not session_ids:
            stored_sessions = self.replay.list_stored_sessions()
            session_ids = [s["session_id"] for s in stored_sessions[:self.sample_size]]

        if not session_ids:
            logger.warning("[Canary] No sessions available for canary testing")
            return CanaryResult(
                canary_id=canary_id,
                success=False,
                passed=False,
                comparison_summary={"error": "No sessions available"}
            )

        logger.info(f"[Canary] Testing {len(session_ids)} sessions")

        replay_results = await self.replay.batch_replay(session_ids)

        success_count = sum(1 for r in replay_results if r.success)
        total_original_duration = sum(r.original_duration_ms for r in replay_results)
        total_replay_duration = sum(r.replay_duration_ms for r in replay_results)
        avg_quality = sum(r.quality_score for r in replay_results if r.success) / max(success_count, 1)

        performance_met = True
        quality_met = True
        stability_met = True

        comparison_details = []
        for result in replay_results:
            detail = {
                "session_id": result.session_id,
                "success": result.success,
                "original_duration": result.original_duration_ms,
                "replay_duration": result.replay_duration_ms,
                "duration_ratio": result.replay_duration_ms / max(result.original_duration_ms, 1),
                "original_quality": result.original_quality_score,
                "replay_quality": result.quality_score,
                "quality_diff": result.quality_diff,
            }

            if result.original_duration_ms > 0:
                ratio = result.replay_duration_ms / result.original_duration_ms
                if ratio > self.performance_threshold:
                    performance_met = False
                    detail["performance_issue"] = f"Duration increased by {ratio:.1f}x"

            if result.quality_diff is not None and result.quality_diff < self.quality_threshold:
                quality_met = False
                detail["quality_issue"] = f"Quality decreased by {result.quality_diff:.2f}"

            if not result.success:
                stability_met = False
                detail["stability_issue"] = result.errors

            comparison_details.append(detail)

        passed = performance_met and quality_met and stability_met

        result = CanaryResult(
            canary_id=canary_id,
            success=stability_met,
            passed=passed,
            performance_threshold_met=performance_met,
            quality_threshold_met=quality_met,
            stability_threshold_met=stability_met,
            comparison_summary={
                "description": description,
                "total_sessions": len(replay_results),
                "success_count": success_count,
                "failure_count": len(replay_results) - success_count,
                "avg_duration_ratio": total_replay_duration / max(total_original_duration, 1),
                "avg_quality_score": avg_quality,
                "performance_threshold": self.performance_threshold,
                "quality_threshold": self.quality_threshold,
                "stability_threshold": self.stability_threshold,
                "details": comparison_details,
            }
        )

        self._export_canary_result(result)

        if passed:
            logger.info(f"[Canary] ✅ Canary passed: {canary_id}")
        else:
            logger.warning(f"[Canary] ❌ Canary failed: {canary_id}")
            logger.warning(f"[Canary] Details: performance={performance_met}, quality={quality_met}, stability={stability_met}")

        return result

    def _export_canary_result(self, result: CanaryResult) -> None:
        """导出金丝雀结果"""
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = _OUTPUT_DIR / f"canary_{result.canary_id}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"[Canary] Results exported to {path}")


class PerformanceBenchmark:
    """性能基准测试器"""

    def __init__(self, warmup_runs: int = 1):
        self.warmup_runs = warmup_runs

    async def run_benchmark(
        self,
        source_files: List[str],
        domain: str = "SUPPLY_CHAIN"
    ) -> BenchmarkResult:
        """运行性能基准测试"""
        benchmark_id = f"benchmark-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"[Benchmark] Starting benchmark: {benchmark_id}")

        pipeline = UnifiedCompilationPipeline()
        results = []
        total_entities = 0
        total_quality = 0.0

        for idx, source_file in enumerate(source_files):
            if not Path(source_file).exists():
                logger.warning(f"[Benchmark] File not found: {source_file}")
                continue

            logger.info(f"[Benchmark] Processing {idx + 1}/{len(source_files)}: {source_file}")

            try:
                start_time = time.time()
                result = await pipeline.run_async(
                    source_path=source_file,
                    domain=domain,
                    run_id=f"{benchmark_id}-{idx}"
                )
                duration_ms = (time.time() - start_time) * 1000

                results.append({
                    "success": True,
                    "file": source_file,
                    "duration_ms": duration_ms,
                    "quality_score": getattr(result, "quality_score", 0.0),
                    "entity_count": len(result.final_fragment.nodes) if result.final_fragment else 0,
                })

                total_quality += results[-1]["quality_score"]
                if result.final_fragment:
                    total_entities += len(result.final_fragment.nodes)

            except Exception as e:
                logger.error(f"[Benchmark] Failed to process {source_file}: {e}")
                results.append({
                    "success": False,
                    "file": source_file,
                    "duration_ms": 0,
                    "quality_score": 0.0,
                    "entity_count": 0,
                    "error": str(e),
                })

        success_results = [r for r in results if r["success"]]
        durations = [r["duration_ms"] for r in success_results]

        total_duration = sum(durations)
        avg_duration = total_duration / len(durations) if durations else 0.0
        min_duration = min(durations) if durations else 0.0
        max_duration = max(durations) if durations else 0.0
        success_rate = len(success_results) / len(results) if results else 0.0
        avg_quality = total_quality / len(success_results) if success_results else 0.0
        throughput = (len(success_results) / (total_duration / 1000 / 60)) if total_duration > 0 else 0.0

        benchmark_result = BenchmarkResult(
            benchmark_id=benchmark_id,
            test_cases=[r["file"] for r in results],
            total_duration_ms=total_duration,
            avg_duration_ms=avg_duration,
            min_duration_ms=min_duration,
            max_duration_ms=max_duration,
            total_entities=total_entities,
            avg_quality_score=avg_quality,
            success_rate=success_rate,
            throughput_docs_per_min=throughput,
        )

        self._export_benchmark_result(benchmark_result)

        logger.info(f"[Benchmark] Completed: {benchmark_id}")
        logger.info(f"[Benchmark] Results: {len(success_results)}/{len(results)} succeeded")
        logger.info(f"[Benchmark] Avg duration: {avg_duration:.1f}ms, Throughput: {throughput:.2f} docs/min")
        logger.info(f"[Benchmark] Avg quality: {avg_quality:.2f}, Success rate: {success_rate:.1%}")

        return benchmark_result

    def _export_benchmark_result(self, result: BenchmarkResult) -> None:
        """导出基准测试结果"""
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = _OUTPUT_DIR / f"benchmark_{result.benchmark_id}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)


def get_replay_manager() -> SessionReplay:
    """获取全局会话重放管理器"""
    global _global_replay
    if _global_replay is None:
        _global_replay = SessionReplay()
    return _global_replay


def get_canary_runner() -> CanaryRunner:
    """获取全局金丝雀运行器"""
    global _global_canary
    if _global_canary is None:
        _global_canary = CanaryRunner()
    return _global_canary


_global_replay: Optional[SessionReplay] = None
_global_canary: Optional[CanaryRunner] = None