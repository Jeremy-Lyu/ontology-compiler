"""LLM Monitor - Track LLM usage across pipeline stages

根据 PROJECT_ARCHITECTURE_AUDIT_REPORT_20260708.md 第3.3节和用户要求：
- 监控每个阶段的LLM调用次数
- 跟踪token使用量
- 检测截断率
- 跟踪解析成功率
- 暴露为pipeline指标

设计原则：
- 无侵入式监控：通过装饰器和上下文管理器实现
- 统一指标收集：所有LLM调用汇聚到一个监控器
- 阶段级粒度：区分不同pipeline阶段的LLM使用
- 可配置采样：支持全量或抽样监控
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import wraps

from src.application.services.llm_trace import extract_prompt_text, hash_text

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    PREFLIGHT = "preflight"
    EXTRACT = "extract"
    NORMALIZE = "normalize"
    VALIDATE = "validate"
    DEDUP = "dedup"
    RELATIONSHIP = "relationship"
    MERGE = "merge"
    AUDIT = "audit"
    EXPORT = "export"
    AGENT = "agent"
    OTHER = "other"


class TruncationReason(Enum):
    MAX_TOKENS = "max_tokens"
    CONTENT_LENGTH = "content_length"
    TIMEOUT = "timeout"
    ERROR = "error"
    NONE = "none"


class ErrorType(Enum):
    """Unified failure classification for all trace records.

    See docs/UNIFIED_TRACE_SPECIFICATION.md section 4.
    """
    NONE = "none"
    TIMEOUT = "timeout"
    TRUNCATION = "truncation"
    PARSE_ERROR = "parse_error"
    AUTH_ERROR = "auth_error"
    RATE_LIMIT = "rate_limit"
    NETWORK_ERROR = "network_error"
    SCHEMA_VALIDATION = "schema_validation"
    EMPTY_OUTPUT = "empty_output"
    UNKNOWN = "unknown"


@dataclass
class LLMCallRecord:
    stage: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_estimate: float = 0.0
    duration_ms: int = 0
    success: bool = True
    truncation_reason: TruncationReason = TruncationReason.NONE
    error_type: ErrorType = ErrorType.NONE
    error_message: str = ""
    prompt_hash: str = ""
    call_id: str = ""
    degraded_mode: bool = False
    degraded_reason: str = ""
    price_known: bool = True


@dataclass
class StageLLMMetrics:
    stage: str
    call_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_duration_ms: int = 0
    success_count: int = 0
    failure_count: int = 0
    truncation_count: int = 0
    avg_duration_ms: float = 0.0
    avg_tokens_per_call: float = 0.0
    success_rate: float = 0.0
    truncation_rate: float = 0.0

    def update(self, record: LLMCallRecord):
        self.call_count += 1
        self.total_prompt_tokens += record.prompt_tokens
        self.total_completion_tokens += record.completion_tokens
        self.total_tokens += record.total_tokens
        self.total_cost += record.cost_estimate
        self.total_duration_ms += record.duration_ms

        if record.success:
            self.success_count += 1
        else:
            self.failure_count += 1

        if record.truncation_reason != TruncationReason.NONE:
            self.truncation_count += 1

        if self.call_count > 0:
            self.avg_duration_ms = self.total_duration_ms / self.call_count
            self.avg_tokens_per_call = self.total_tokens / self.call_count
            self.success_rate = self.success_count / self.call_count
            self.truncation_rate = self.truncation_count / self.call_count


class LLMMonitor:
    """LLM监控器 - 统一跟踪所有LLM调用"""

    _instance: "LLMMonitor | None" = None

    def __init__(self):
        self._stage_metrics: Dict[str, StageLLMMetrics] = {}
        self._call_records: List[LLMCallRecord] = []
        self._enabled: bool = True
        self._max_records: int = 1000
        self._current_stage: Optional[str] = None
        self._provider_rates: Dict[str, float] = {}
        # C-1: track explicitly when a call had no registered price, so unknown-
        # priced usage is never silently counted as USD 0.00.
        self._unknown_price_calls: int = 0
        # C-5 后续增强 (2026-09-01): routing decisions recorded by
        # ``LLMGateway.execute_routed`` (task_kind -> complexity tier counts)
        # so the manifest / regression baseline can carry ``route_decisions``.
        self._decision_counts: Dict[str, Dict[str, int]] = {}

    @classmethod
    def get_instance(cls) -> "LLMMonitor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_provider_cost_rate(self, provider: str, dollars_per_1k_tokens: float):
        self._provider_rates[provider] = dollars_per_1k_tokens

    def set_stage(self, stage: str | PipelineStage):
        if isinstance(stage, PipelineStage):
            self._current_stage = stage.value
        else:
            self._current_stage = stage

    def unset_stage(self):
        self._current_stage = None

    def record_call(
        self,
        provider: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        duration_ms: int = 0,
        success: bool = True,
        truncation_reason: TruncationReason = TruncationReason.NONE,
        error_message: str = "",
        call_id: str = "",
        prompt_hash: str = "",
        error_type: ErrorType = ErrorType.NONE,
        degraded_mode: bool = False,
        degraded_reason: str = "",
        price_known: Optional[bool] = None,
    ):
        if not self._enabled:
            return

        # P2-1 (2026-10-16): feed the provider circuit breaker from this funnel.
        # Only connection-class failures count: a bare success=False from
        # JSON-parse retries carries no error_message and must NOT trip the
        # breaker (that is a content problem, not provider flakiness).
        try:
            from src.application.services.provider_circuit_breaker import get_circuit_breaker

            breaker = get_circuit_breaker()
            if success:
                breaker.record_success(provider)
            elif error_message:
                breaker.record_failure(provider, error_message)
        except Exception:
            pass  # observability must never break a call

        stage = self._current_stage or PipelineStage.OTHER.value

        # C-1: pricing is resolved from the model-level registry first; a legacy
        # per-provider rate set via set_provider_cost_rate takes precedence so
        # existing manual calibration keeps working. Unknown => cost 0.0 BUT
        # explicitly flag price_known=False instead of silently counting it free.
        cost_estimate = (total_tokens / 1000) * self._provider_rates.get(provider, 0.0)
        if price_known is None:
            if provider in self._provider_rates:
                price_known = True
            else:
                try:
                    from src.application.services.llm_pricing import estimate_cost

                    estimate = estimate_cost(
                        provider, model, prompt_tokens, completion_tokens
                    )
                    cost_estimate = estimate.usd
                    price_known = estimate.price_known
                except Exception:
                    price_known = False
        if not price_known:
            self._unknown_price_calls += 1

        record = LLMCallRecord(
            stage=stage,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_estimate=cost_estimate,
            duration_ms=duration_ms,
            success=success,
            truncation_reason=truncation_reason,
            error_message=error_message,
            prompt_hash=prompt_hash,
            call_id=call_id,
            error_type=error_type,
            degraded_mode=degraded_mode,
            degraded_reason=degraded_reason,
            price_known=price_known,
        )

        if stage not in self._stage_metrics:
            self._stage_metrics[stage] = StageLLMMetrics(stage=stage)

        self._stage_metrics[stage].update(record)
        self._call_records.append(record)

        if len(self._call_records) > self._max_records:
            self._call_records = self._call_records[-self._max_records :]

        logger.debug(
            f"[LLM Monitor] {stage}: {provider}/{model} -> {total_tokens} tokens, "
            f"${cost_estimate:.4f}, {duration_ms}ms, success={success}"
        )

    def get_stage_metrics(self, stage: str) -> Optional[StageLLMMetrics]:
        return self._stage_metrics.get(stage)

    def get_all_metrics(self) -> Dict[str, StageLLMMetrics]:
        return dict(self._stage_metrics)

    def get_summary(self) -> Dict[str, Any]:
        total_calls = sum(m.call_count for m in self._stage_metrics.values())
        total_tokens = sum(m.total_tokens for m in self._stage_metrics.values())
        total_cost = sum(m.total_cost for m in self._stage_metrics.values())
        total_duration_ms = sum(m.total_duration_ms for m in self._stage_metrics.values())
        success_count = sum(m.success_count for m in self._stage_metrics.values())

        stage_summaries = {}
        for stage, metrics in self._stage_metrics.items():
            stage_summaries[stage] = {
                "call_count": metrics.call_count,
                "total_tokens": metrics.total_tokens,
                "total_cost": metrics.total_cost,
                "success_rate": metrics.success_rate,
                "truncation_rate": metrics.truncation_rate,
                "avg_duration_ms": metrics.avg_duration_ms,
            }

        return {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "total_cost_estimate": total_cost,
            "total_duration_ms": total_duration_ms,
            "overall_success_rate": success_count / total_calls if total_calls > 0 else 0.0,
            "avg_tokens_per_call": total_tokens / total_calls if total_calls > 0 else 0.0,
            "unknown_price_calls": self._unknown_price_calls,
            "stage_metrics": stage_summaries,
            # C-5 后续增强 (2026-09-01): routing telemetry for manifest / baseline.
            "route_decisions": {k: dict(v) for k, v in self._decision_counts.items()},
        }

    def last_record(self) -> Optional[LLMCallRecord]:
        """Return the most recently recorded call (None when empty)."""
        return self._call_records[-1] if self._call_records else None

    def record_route(
        self,
        task_kind: str,
        tier: str,
        reason: str = "",
    ) -> None:
        """Record one routing decision for baseline ``route_decisions``.

        C-5 后续增强 (2026-09-01): called by ``LLMGateway.execute_routed`` so
        the monitor aggregates task_kind -> {tier: count}. Never raises:
        telemetry must not break the call it describes.
        """
        try:
            kind = str(task_kind or "unknown")
            bucket = str(tier or "unknown")
            per_kind = self._decision_counts.setdefault(kind, {})
            per_kind[bucket] = per_kind.get(bucket, 0) + 1
        except Exception:  # pragma: no cover - defensive
            pass

    def reset(self):
        self._stage_metrics.clear()
        self._call_records.clear()
        self._current_stage = None
        self._unknown_price_calls = 0
        self._decision_counts.clear()

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def is_enabled(self) -> bool:
        return self._enabled


class llm_monitor_context:
    """上下文管理器 - 在特定阶段内监控LLM调用"""

    def __init__(self, stage: str | PipelineStage):
        self._stage = stage
        self._monitor = LLMMonitor.get_instance()

    def __enter__(self):
        self._monitor.set_stage(self._stage)
        return self._monitor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._monitor.unset_stage()


def monitor_llm_call(stage: str | PipelineStage):
    """装饰器 - 监控LLM调用函数"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = LLMMonitor.get_instance()
            original_stage = monitor._current_stage
            monitor.set_stage(stage)

            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                duration_ms = int((time.time() - start_time) * 1000)

                if isinstance(result, tuple) and len(result) >= 2:
                    content, usage = result[0], result[1]
                    if isinstance(usage, dict):
                        prompt_text = ""
                        if args:
                            prompt_text = extract_prompt_text(*args, **kwargs)
                        prompt_hash = hash_text(prompt_text) if prompt_text else ""
                        monitor.record_call(
                            provider=usage.get("provider", "unknown"),
                            model=usage.get("model", "unknown"),
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0),
                            total_tokens=usage.get("total_tokens", 0),
                            duration_ms=duration_ms,
                            success=True,
                            truncation_reason=TruncationReason(
                                usage.get("truncation_reason", "none")
                            ),
                            prompt_hash=prompt_hash,
                        )
                return result
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                monitor.record_call(
                    provider="unknown",
                    model="unknown",
                    duration_ms=duration_ms,
                    success=False,
                    truncation_reason=TruncationReason.ERROR,
                    error_message=str(e),
                    prompt_hash=hash_text(extract_prompt_text(*args, **kwargs)),
                )
                raise
            finally:
                monitor.set_stage(original_stage)

        return wrapper

    return decorator


def get_llm_monitor() -> LLMMonitor:
    """获取全局LLM监控器实例"""
    return LLMMonitor.get_instance()


def record_llm_call(provider: str, model: str, **kwargs):
    """Record an LLM call via the global monitor."""
    monitor = get_llm_monitor()
    monitor.record_call(provider=provider, model=model, **kwargs)


__all__ = [
    "LLMMonitor",
    "PipelineStage",
    "TruncationReason",
    "LLMCallRecord",
    "StageLLMMetrics",
    "llm_monitor_context",
    "monitor_llm_call",
    "get_llm_monitor",
    "record_llm_call",
]