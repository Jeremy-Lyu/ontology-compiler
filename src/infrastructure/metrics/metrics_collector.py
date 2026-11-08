"""Metrics collector for ASI governance (P3-1 Enhancement).

Collects and exports pipeline execution metrics for monitoring and alerting.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ASIMetrics:
    """ASI governance metrics."""
    asi_score: float
    asi_threshold: float
    low_asi_streak: int
    pause_required: bool
    run_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class TimeSavedMetrics:
    """Time saved metrics from incremental processing."""
    estimated_time_saved_ms: int
    estimated_time_saved_ratio: float
    catalog_duration_ms: int
    changed_count: int
    candidate_count: int
    run_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class PipelineMetrics:
    """Pipeline execution metrics."""
    run_id: str
    status: str
    stage_count: int
    success_count: int
    warning_count: int
    failed_count: int
    duration_ms: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class StageExecutionMetrics:
    """Stage/step execution metrics (R7.4 Enhancement).

    Tracks execution time and status for individual pipeline steps.
    """
    run_id: str
    stage_name: str
    step_name: str
    duration_ms: int
    status: str  # SUCCESS, WARNING, FAILED, SKIPPED
    input_count: int = 0
    output_count: int = 0
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class LLMUsageMetrics:
    """LLM call metrics (R7.4 Enhancement).

    Tracks LLM API usage for cost monitoring and optimization.
    """
    run_id: str
    provider: str  # deepseek, dify, etc.
    model: str  # v4-flash, v4-pro, etc.
    call_count: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    avg_latency_ms: float
    success_count: int
    failure_count: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class DeduplicationMetrics:
    """Deduplication and merge metrics (R7.4 Enhancement).

    Tracks entity deduplication efficiency and merge operations.
    """
    run_id: str
    layer: str  # L1, L2, L2.5, etc.
    input_count: int
    unique_count: int
    removed_count: int
    merged_count: int
    dedup_ratio: float  # (removed + merged) / input
    merge_ratio: float  # merged / input
    avg_similarity: float
    duration_ms: int
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Collector for ASI governance and pipeline metrics.

    P3-1 Enhancement (2026-05-30):
    Collects metrics and exports to structured logs for monitoring.
    Optional Prometheus support if prometheus_client is available.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("outputs/metrics")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._asi_metrics: List[ASIMetrics] = []
        self._time_saved_metrics: List[TimeSavedMetrics] = []
        self._pipeline_metrics: List[PipelineMetrics] = []

        # R7.4: New metrics for observability enhancement
        self._stage_metrics: List[StageExecutionMetrics] = []
        self._llm_metrics: List[LLMUsageMetrics] = []
        self._dedup_metrics: List[DeduplicationMetrics] = []

        # Try to import prometheus client (optional)
        self._prometheus_available = False
        try:
            from prometheus_client import Counter, Gauge, Histogram
            self._prometheus_available = True
            self._setup_prometheus_metrics()
            logger.info("Prometheus metrics enabled")
        except ImportError:
            logger.debug("Prometheus client not available, using log-based metrics only")

    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics if client is available."""
        from prometheus_client import Counter, Gauge, Histogram

        # ASI metrics
        self._prom_asi_score = Gauge(
            "ontology_asi_score",
            "Current ASI score (0-1)",
            ["run_id"]
        )
        self._prom_asi_threshold = Gauge(
            "ontology_asi_threshold",
            "ASI threshold for governance",
            ["run_id"]
        )
        self._prom_low_asi_streak = Counter(
            "ontology_low_asi_streak_total",
            "Number of consecutive low ASI runs",
            ["run_id"]
        )

        # Time saved metrics
        self._prom_time_saved_ms = Counter(
            "ontology_time_saved_ms_total",
            "Total time saved by incremental processing (ms)",
            ["run_id"]
        )
        self._prom_time_saved_ratio = Gauge(
            "ontology_time_saved_ratio",
            "Ratio of time saved (0-1)",
            ["run_id"]
        )

        # Pipeline metrics
        self._prom_pipeline_duration = Histogram(
            "ontology_pipeline_duration_seconds",
            "Pipeline execution duration",
            ["status"]
        )
        self._prom_stage_status = Counter(
            "ontology_stage_status_total",
            "Stage execution status counts",
            ["status"]
        )

        # R7.4: Stage execution metrics
        self._prom_stage_duration = Histogram(
            "ontology_stage_duration_seconds",
            "Individual stage execution duration",
            ["stage_name", "step_name", "status"]
        )

        # R7.4: LLM usage metrics
        self._prom_llm_calls = Counter(
            "ontology_llm_calls_total",
            "Total LLM API calls",
            ["provider", "model"]
        )
        self._prom_llm_tokens = Counter(
            "ontology_llm_tokens_total",
            "Total LLM tokens consumed",
            ["provider", "model", "token_type"]
        )
        self._prom_llm_cost = Counter(
            "ontology_llm_cost_usd_total",
            "Estimated LLM cost in USD",
            ["provider", "model"]
        )

        # R7.4: Deduplication metrics
        self._prom_dedup_ratio = Gauge(
            "ontology_dedup_ratio",
            "Deduplication ratio (entities removed/merged)",
            ["layer"]
        )
        self._prom_merge_ratio = Gauge(
            "ontology_merge_ratio",
            "Merge ratio (entities merged)",
            ["layer"]
        )

    def record_asi_score(
        self,
        asi_score: float,
        asi_threshold: float = 0.75,
        low_asi_streak: int = 0,
        pause_required: bool = False,
        run_id: str = "",
    ) -> None:
        """Record ASI governance metrics."""
        metric = ASIMetrics(
            asi_score=asi_score,
            asi_threshold=asi_threshold,
            low_asi_streak=low_asi_streak,
            pause_required=pause_required,
            run_id=run_id,
        )
        self._asi_metrics.append(metric)

        # Log structured metric
        logger.info(
            "ASI_METRIC",
            extra={
                "metric_type": "asi_score",
                "asi_score": asi_score,
                "asi_threshold": asi_threshold,
                "low_asi_streak": low_asi_streak,
                "pause_required": pause_required,
                "run_id": run_id,
                "timestamp": metric.timestamp,
            }
        )

        # Update Prometheus metrics if available
        if self._prometheus_available:
            self._prom_asi_score.labels(run_id=run_id).set(asi_score)
            self._prom_asi_threshold.labels(run_id=run_id).set(asi_threshold)
            if asi_score < asi_threshold:
                self._prom_low_asi_streak.labels(run_id=run_id).inc()

    def record_time_saved(
        self,
        estimated_time_saved_ms: int,
        estimated_time_saved_ratio: float,
        catalog_duration_ms: int,
        changed_count: int,
        candidate_count: int,
        run_id: str = "",
    ) -> None:
        """Record time saved metrics from incremental processing."""
        metric = TimeSavedMetrics(
            estimated_time_saved_ms=estimated_time_saved_ms,
            estimated_time_saved_ratio=estimated_time_saved_ratio,
            catalog_duration_ms=catalog_duration_ms,
            changed_count=changed_count,
            candidate_count=candidate_count,
            run_id=run_id,
        )
        self._time_saved_metrics.append(metric)

        # Log structured metric
        logger.info(
            "TIME_SAVED_METRIC",
            extra={
                "metric_type": "time_saved",
                "estimated_time_saved_ms": estimated_time_saved_ms,
                "estimated_time_saved_ratio": estimated_time_saved_ratio,
                "catalog_duration_ms": catalog_duration_ms,
                "changed_count": changed_count,
                "candidate_count": candidate_count,
                "run_id": run_id,
                "timestamp": metric.timestamp,
            }
        )

        # Update Prometheus metrics if available
        if self._prometheus_available:
            self._prom_time_saved_ms.labels(run_id=run_id).inc(estimated_time_saved_ms)
            self._prom_time_saved_ratio.labels(run_id=run_id).set(estimated_time_saved_ratio)

    def record_pipeline_execution(
        self,
        run_id: str,
        status: str,
        stage_results: List[Dict[str, Any]],
        duration_ms: int,
    ) -> None:
        """Record pipeline execution metrics."""
        success_count = sum(1 for s in stage_results if str(s.get("status")).upper() == "SUCCESS")
        warning_count = sum(1 for s in stage_results if str(s.get("status")).upper() == "WARNING")
        failed_count = sum(1 for s in stage_results if str(s.get("status")).upper() == "FAILED")

        metric = PipelineMetrics(
            run_id=run_id,
            status=status,
            stage_count=len(stage_results),
            success_count=success_count,
            warning_count=warning_count,
            failed_count=failed_count,
            duration_ms=duration_ms,
        )
        self._pipeline_metrics.append(metric)

        # Log structured metric
        logger.info(
            "PIPELINE_METRIC",
            extra={
                "metric_type": "pipeline_execution",
                "run_id": run_id,
                "status": status,
                "stage_count": len(stage_results),
                "success_count": success_count,
                "warning_count": warning_count,
                "failed_count": failed_count,
                "duration_ms": duration_ms,
                "timestamp": metric.timestamp,
            }
        )

        # Update Prometheus metrics if available
        if self._prometheus_available:
            duration_seconds = duration_ms / 1000.0
            self._prom_pipeline_duration.labels(status=status).observe(duration_seconds)

            for stage in stage_results:
                stage_status = str(stage.get("status")).upper()
                self._prom_stage_status.labels(status=stage_status).inc()

    def record_stage_execution(
        self,
        run_id: str,
        stage_name: str,
        step_name: str,
        duration_ms: int,
        status: str,
        input_count: int = 0,
        output_count: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        """Record individual stage/step execution metrics (R7.4)."""
        metric = StageExecutionMetrics(
            run_id=run_id,
            stage_name=stage_name,
            step_name=step_name,
            duration_ms=duration_ms,
            status=status,
            input_count=input_count,
            output_count=output_count,
            error_message=error_message,
        )
        self._stage_metrics.append(metric)

        logger.info(
            "STAGE_METRIC",
            extra={
                "metric_type": "stage_execution",
                "run_id": run_id,
                "stage_name": stage_name,
                "step_name": step_name,
                "duration_ms": duration_ms,
                "status": status,
                "input_count": input_count,
                "output_count": output_count,
                "error_message": error_message,
                "timestamp": metric.timestamp,
            }
        )

        if self._prometheus_available:
            duration_seconds = duration_ms / 1000.0
            self._prom_stage_duration.labels(
                stage_name=stage_name,
                step_name=step_name,
                status=status
            ).observe(duration_seconds)

    def record_llm_usage(
        self,
        run_id: str,
        provider: str,
        model: str,
        call_count: int,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
        avg_latency_ms: float,
        success_count: int,
        failure_count: int,
    ) -> None:
        """Record LLM API usage metrics (R7.4)."""
        metric = LLMUsageMetrics(
            run_id=run_id,
            provider=provider,
            model=model,
            call_count=call_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost_usd,
            avg_latency_ms=avg_latency_ms,
            success_count=success_count,
            failure_count=failure_count,
        )
        self._llm_metrics.append(metric)

        logger.info(
            "LLM_METRIC",
            extra={
                "metric_type": "llm_usage",
                "run_id": run_id,
                "provider": provider,
                "model": model,
                "call_count": call_count,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost_usd": estimated_cost_usd,
                "avg_latency_ms": avg_latency_ms,
                "success_count": success_count,
                "failure_count": failure_count,
                "timestamp": metric.timestamp,
            }
        )

        if self._prometheus_available:
            self._prom_llm_calls.labels(provider=provider, model=model).inc(call_count)
            self._prom_llm_tokens.labels(
                provider=provider, model=model, token_type="input"
            ).inc(input_tokens)
            self._prom_llm_tokens.labels(
                provider=provider, model=model, token_type="output"
            ).inc(output_tokens)
            self._prom_llm_cost.labels(provider=provider, model=model).inc(estimated_cost_usd)

    def record_deduplication(
        self,
        run_id: str,
        layer: str,
        input_count: int,
        unique_count: int,
        removed_count: int,
        merged_count: int,
        avg_similarity: float,
        duration_ms: int,
    ) -> None:
        """Record deduplication and merge metrics (R7.4)."""
        dedup_ratio = (removed_count + merged_count) / input_count if input_count > 0 else 0.0
        merge_ratio = merged_count / input_count if input_count > 0 else 0.0

        metric = DeduplicationMetrics(
            run_id=run_id,
            layer=layer,
            input_count=input_count,
            unique_count=unique_count,
            removed_count=removed_count,
            merged_count=merged_count,
            dedup_ratio=dedup_ratio,
            merge_ratio=merge_ratio,
            avg_similarity=avg_similarity,
            duration_ms=duration_ms,
        )
        self._dedup_metrics.append(metric)

        logger.info(
            "DEDUP_METRIC",
            extra={
                "metric_type": "deduplication",
                "run_id": run_id,
                "layer": layer,
                "input_count": input_count,
                "unique_count": unique_count,
                "removed_count": removed_count,
                "merged_count": merged_count,
                "dedup_ratio": dedup_ratio,
                "merge_ratio": merge_ratio,
                "avg_similarity": avg_similarity,
                "duration_ms": duration_ms,
                "timestamp": metric.timestamp,
            }
        )

        if self._prometheus_available:
            self._prom_dedup_ratio.labels(layer=layer).set(dedup_ratio)
            self._prom_merge_ratio.labels(layer=layer).set(merge_ratio)

    def export_to_logs(self) -> None:
        """Export all collected metrics to structured logs."""
        summary = {
            "asi_metrics_count": len(self._asi_metrics),
            "time_saved_metrics_count": len(self._time_saved_metrics),
            "pipeline_metrics_count": len(self._pipeline_metrics),
            "stage_metrics_count": len(self._stage_metrics),
            "llm_metrics_count": len(self._llm_metrics),
            "dedup_metrics_count": len(self._dedup_metrics),
            "latest_asi": self._asi_metrics[-1].__dict__ if self._asi_metrics else None,
            "latest_time_saved": self._time_saved_metrics[-1].__dict__ if self._time_saved_metrics else None,
            "latest_pipeline": self._pipeline_metrics[-1].__dict__ if self._pipeline_metrics else None,
            "latest_llm": self._llm_metrics[-1].__dict__ if self._llm_metrics else None,
            "latest_dedup": self._dedup_metrics[-1].__dict__ if self._dedup_metrics else None,
        }

        logger.info(
            "METRICS_SUMMARY",
            extra={
                "metric_type": "summary",
                **summary,
                "timestamp": time.time(),
            }
        )

    def export_to_json(self, filename: Optional[str] = None) -> Path:
        """Export all metrics to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_export_{timestamp}.json"

        export_path = self.output_dir / filename

        export_data = {
            "exported_at": datetime.now().isoformat(),
            "asi_metrics": [m.__dict__ for m in self._asi_metrics],
            "time_saved_metrics": [m.__dict__ for m in self._time_saved_metrics],
            "pipeline_metrics": [m.__dict__ for m in self._pipeline_metrics],
            "stage_metrics": [m.__dict__ for m in self._stage_metrics],
            "llm_metrics": [m.__dict__ for m in self._llm_metrics],
            "dedup_metrics": [m.__dict__ for m in self._dedup_metrics],
        }

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Metrics exported to {export_path}")
        return export_path

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        # Calculate LLM summary
        total_llm_calls = sum(m.call_count for m in self._llm_metrics)
        total_llm_cost = sum(m.estimated_cost_usd for m in self._llm_metrics)
        total_llm_failures = sum(m.failure_count for m in self._llm_metrics)

        # Calculate dedup summary
        total_dedup_input = sum(m.input_count for m in self._dedup_metrics)
        total_dedup_removed = sum(m.removed_count for m in self._dedup_metrics)
        total_dedup_merged = sum(m.merged_count for m in self._dedup_metrics)

        return {
            "asi_metrics_count": len(self._asi_metrics),
            "time_saved_metrics_count": len(self._time_saved_metrics),
            "pipeline_metrics_count": len(self._pipeline_metrics),
            "stage_metrics_count": len(self._stage_metrics),
            "llm_metrics_count": len(self._llm_metrics),
            "dedup_metrics_count": len(self._dedup_metrics),
            "average_asi_score": (
                sum(m.asi_score for m in self._asi_metrics) / len(self._asi_metrics)
                if self._asi_metrics else 0.0
            ),
            "total_time_saved_ms": (
                sum(m.estimated_time_saved_ms for m in self._time_saved_metrics)
                if self._time_saved_metrics else 0
            ),
            "total_runs": len(self._pipeline_metrics),
            "successful_runs": sum(
                1 for m in self._pipeline_metrics if m.status.upper() == "SUCCESS"
            ),
            # R7.4: LLM usage summary
            "total_llm_calls": total_llm_calls,
            "total_llm_cost_usd": total_llm_cost,
            "total_llm_failures": total_llm_failures,
            "llm_success_rate": (
                (total_llm_calls - total_llm_failures) / total_llm_calls
                if total_llm_calls > 0 else 0.0
            ),
            # R7.4: Deduplication summary
            "total_dedup_input": total_dedup_input,
            "total_dedup_removed": total_dedup_removed,
            "total_dedup_merged": total_dedup_merged,
            "overall_dedup_ratio": (
                (total_dedup_removed + total_dedup_merged) / total_dedup_input
                if total_dedup_input > 0 else 0.0
            ),
        }

    def clear(self) -> None:
        """Clear all collected metrics."""
        self._asi_metrics.clear()
        self._time_saved_metrics.clear()
        self._pipeline_metrics.clear()
        self._stage_metrics.clear()
        self._llm_metrics.clear()
        self._dedup_metrics.clear()


# Singleton instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(output_dir: Optional[Path] = None) -> MetricsCollector:
    """Get singleton metrics collector instance.

    Usage:
        collector = get_metrics_collector()
        collector.record_asi_score(0.85, run_id="run-001")
    """
    global _metrics_collector

    if _metrics_collector is None or output_dir is not None:
        _metrics_collector = MetricsCollector(output_dir)

    return _metrics_collector


def reset_metrics_collector() -> None:
    """Reset the singleton metrics collector instance."""
    global _metrics_collector
    _metrics_collector = None
