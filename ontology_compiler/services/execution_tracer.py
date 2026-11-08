"""Execution Tracer - Unified trace and session/run_id management.

[Phase 3 - TRACE 2026-06-17]: Provides consistent run_id tracking across all pipeline steps,
enabling full traceability of multi-agent orchestration.

[BATCH 4 CONSOLIDATION 2026-06-29]:
- This module provides **telemetry/tracing infrastructure** for pipeline execution
- Tight integration with artifacts.py for trace persistence

Features:
- Generate and manage run_id for pipeline executions
- Track execution context (session_id, parent_run_id)
- Record step-level execution details
- Support nested execution tracking
- Output trace reports
- LLM call recording for observability
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .artifacts import write_json_artifact

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent.parent


@dataclass
class TraceEvent:
    """Single trace event in the execution."""
    timestamp: str
    run_id: str
    step_name: str
    event_type: str  # start, end, error, info
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    input_hash: str = ""
    output_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionContext:
    """Execution context passed through the pipeline."""
    run_id: str
    session_id: str
    parent_run_id: Optional[str] = None
    start_time: str = ""
    end_time: str = ""
    status: str = "running"  # running, completed, failed
    step_timings: Dict[str, float] = field(default_factory=dict)
    trace_events: List[TraceEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        # [SMOKE-FIX 2026-06-21] asdict(self) 已经把 trace_events 内的 TraceEvent
        # dataclass 递归转成了 dict；之前再调 e.to_dict() 必崩
        # （'dict' object has no attribute 'to_dict'）。这里只做防御性兜底：
        # - 若元素本身已是 dict（asdict 转换后的，或外部直接 append 的 dict） → 保留
        # - 若元素仍是 TraceEvent / 其它带 to_dict 的对象 → 调用 to_dict()
        # - 其它兜底 str()，永不抛
        result = asdict(self)
        events = result.get("trace_events") or []
        normalized: List[Dict[str, Any]] = []
        for e in events:
            if isinstance(e, dict):
                normalized.append(e)
            elif hasattr(e, "to_dict"):
                normalized.append(e.to_dict())
            else:
                normalized.append({"value": str(e)})
        result["trace_events"] = normalized
        return result


class ExecutionTracer:
    """Unified execution tracer for pipeline tracking."""

    def __init__(self, storage_path: Path | str | None = None):
        if storage_path is None:
            self.storage_path = _PROJECT_ROOT / "outputs" / "traces"
        else:
            self.storage_path = Path(storage_path)
        self._active_contexts: Dict[str, ExecutionContext] = {}

    def generate_run_id(self) -> str:
        """Generate a unique run ID."""
        return f"run-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"

    def generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"session-{datetime.now().strftime('%Y%m%d%H')}-{uuid.uuid4().hex[:6]}"

    def start_execution(
        self,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExecutionContext:
        """Start a new execution context."""
        run_id = run_id or self.generate_run_id()
        session_id = session_id or self.generate_session_id()

        ctx = ExecutionContext(
            run_id=run_id,
            session_id=session_id,
            parent_run_id=parent_run_id,
            start_time=datetime.now().isoformat(),
            metadata=metadata or {}
        )

        self._active_contexts[run_id] = ctx
        self._record_event(run_id, "init", "start", details={"session_id": session_id})

        return ctx

    def end_execution(self, run_id: str, status: str = "completed") -> ExecutionContext | None:
        """End an execution context."""
        ctx = self._active_contexts.get(run_id)
        if not ctx:
            return None

        ctx.end_time = datetime.now().isoformat()
        ctx.status = status

        duration_ms = self._calculate_duration(ctx.start_time, ctx.end_time)
        self._record_event(run_id, "finalize", "end", duration_ms=duration_ms, details={"status": status})

        self._save_trace(ctx)
        return ctx

    def record_step_start(
        self,
        run_id: str,
        step_name: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record step start event."""
        self._record_event(run_id, step_name, "start", details=details or {})

    def record_step_end(
        self,
        run_id: str,
        step_name: str,
        duration_ms: float,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record step end event."""
        ctx = self._active_contexts.get(run_id)
        if ctx:
            ctx.step_timings[step_name] = duration_ms

        self._record_event(run_id, step_name, "end", duration_ms=duration_ms, details=details or {})

    def record_step_error(
        self,
        run_id: str,
        step_name: str,
        error_message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record step error event."""
        self._record_event(run_id, step_name, "error", error_message=error_message, details=details or {})

    def record_info(
        self,
        run_id: str,
        step_name: str,
        details: Dict[str, Any]
    ) -> None:
        """Record informational event."""
        self._record_event(run_id, step_name, "info", details=details)

    def record_quality_check(
        self,
        run_id: str,
        rule_id: str,
        score: float,
        passed: bool,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record quality gate rule check result.
        
        [Phase 6-OBSV-2]: Track individual quality gate rule executions
        """
        self._record_event(
            run_id=run_id,
            step_name="quality_check",
            event_type="quality_check",
            details={
                "rule_id": rule_id,
                "score": score,
                "passed": passed,
                **(details or {})
            }
        )

    def get_context(self, run_id: str) -> ExecutionContext | None:
        """Get active execution context by run_id."""
        return self._active_contexts.get(run_id)

    def get_trace_report(self, run_id: str) -> Dict[str, Any]:
        """Generate a trace report for a run."""
        ctx = self._active_contexts.get(run_id)
        if not ctx:
            return {"error": f"Run {run_id} not found"}

        return {
            "run_id": ctx.run_id,
            "session_id": ctx.session_id,
            "parent_run_id": ctx.parent_run_id,
            "start_time": ctx.start_time,
            "end_time": ctx.end_time,
            "status": ctx.status,
            "duration_ms": self._calculate_duration(ctx.start_time, ctx.end_time),
            "step_timings": ctx.step_timings,
            "event_count": len(ctx.trace_events),
            "metadata": ctx.metadata,
            "events": [e.to_dict() for e in ctx.trace_events]
        }

    def _record_event(
        self,
        run_id: str,
        step_name: str,
        event_type: str,
        duration_ms: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
        error_message: str = ""
    ) -> None:
        """Internal: Record a trace event."""
        ctx = self._active_contexts.get(run_id)
        if not ctx:
            return

        event = TraceEvent(
            timestamp=datetime.now().isoformat(),
            run_id=run_id,
            step_name=step_name,
            event_type=event_type,
            duration_ms=duration_ms,
            details=details or {},
            error_message=error_message
        )

        ctx.trace_events.append(event)

    def record_llm_call(
        self,
        run_id: str,
        step_name: str,
        prompt: str,
        response: str,
        model: str = "unknown",
        duration_ms: float = 0.0,
        token_usage: Optional[Dict[str, int]] = None,
        entity_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """[Phase 6-OBSV-1] Record LLM call with prompt and response.
        
        Args:
            run_id: The execution run ID
            step_name: Current step name (e.g., "TrueUnifiedExtract")
            prompt: The LLM prompt (will be truncated if too long)
            response: The LLM response (will be truncated if too long)
            model: LLM model name
            duration_ms: Call duration in milliseconds
            token_usage: Token usage statistics
            entity_ids: Extracted entity IDs for debugging hallucinations
            metadata: Additional metadata about the LLM call
        """
        ctx = self._active_contexts.get(run_id)
        if not ctx:
            return
        
        # Sanitize and truncate for storage efficiency
        MAX_LENGTH = 4000
        sanitized_prompt = self._sanitize_text(prompt, MAX_LENGTH)
        sanitized_response = self._sanitize_text(response, MAX_LENGTH)
        
        details = {
            "model": model,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "token_usage": token_usage or {},
            "entity_ids": entity_ids or [],
            "has_truncated": len(prompt) > MAX_LENGTH or len(response) > MAX_LENGTH,
            "metadata": metadata or {}
        }
        
        event = TraceEvent(
            timestamp=datetime.now().isoformat(),
            run_id=run_id,
            step_name=step_name,
            event_type="llm_call",
            duration_ms=duration_ms,
            details=details,
            input_hash=prompt[:200] if prompt else "",  # First 200 chars as identifier
            output_hash=response[:200] if response else ""
        )
        
        ctx.trace_events.append(event)
        
        # Also store the full prompt/response separately for debugging
        llm_record = {
            "timestamp": datetime.now().isoformat(),
            "step_name": step_name,
            "model": model,
            "duration_ms": duration_ms,
            "prompt": sanitized_prompt,
            "response": sanitized_response,
            "token_usage": token_usage or {},
            "entity_ids": entity_ids or []
        }
        
        # Store in metadata for later retrieval
        if "llm_calls" not in ctx.metadata:
            ctx.metadata["llm_calls"] = []
        ctx.metadata["llm_calls"].append(llm_record)
    
    def _sanitize_text(self, text: str, max_length: int) -> str:
        """Sanitize text for storage (truncate if too long)."""
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return text[:max_length] + f"\n... [TRUNCATED {len(text) - max_length} chars]"
    
    def get_llm_calls(self, run_id: str) -> List[Dict[str, Any]]:
        """[Phase 6-OBSV-1] Get all LLM calls for a run."""
        ctx = self._active_contexts.get(run_id)
        if not ctx:
            return []
        return ctx.metadata.get("llm_calls", [])

    def _calculate_duration(self, start_time: str, end_time: str) -> float:
        """Calculate duration between two ISO timestamps."""
        if not start_time or not end_time:
            return 0.0
        try:
            start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            return (end - start).total_seconds() * 1000
        except Exception:
            return 0.0

    def _save_trace(self, ctx: ExecutionContext) -> None:
        """Save trace to file."""
        write_json_artifact(run_dir=self.storage_path, filename=f"{ctx.run_id}.json", payload=ctx.to_dict())

    def list_traces(self) -> List[Dict[str, Any]]:
        """List all saved traces."""
        traces = []
        if not self.storage_path.exists():
            return traces

        for trace_file in sorted(self.storage_path.glob("run-*.json")):
            try:
                with open(trace_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    traces.append(data)
            except Exception:
                continue

        return sorted(traces, key=lambda x: x.get("start_time", ""), reverse=True)


# Global tracer instance
_global_tracer: Optional[ExecutionTracer] = None


def get_tracer() -> ExecutionTracer:
    """Get or create global execution tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = ExecutionTracer()
    return _global_tracer


def start_run(
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
) -> ExecutionContext:
    """Convenience function to start a new run."""
    tracer = get_tracer()
    return tracer.start_execution(metadata=metadata, session_id=session_id)


def end_run(run_id: str, status: str = "completed") -> None:
    """Convenience function to end a run."""
    tracer = get_tracer()
    tracer.end_execution(run_id, status)


def trace_step(
    run_id: str,
    step_name: str,
    func,
    *args,
    **kwargs
) -> Any:
    """Decorator-like function to trace step execution."""
    tracer = get_tracer()
    tracer.record_step_start(run_id, step_name)

    import time
    start = time.time()
    try:
        result = func(*args, **kwargs)
        duration_ms = (time.time() - start) * 1000
        tracer.record_step_end(run_id, step_name, duration_ms)
        return result
    except Exception as e:
        duration_ms = (time.time() - start) * 1000
        tracer.record_step_error(run_id, step_name, str(e))
        raise


# =============================================================================
# [BATCH 4] Artifact Integration
# =============================================================================

try:
    from .artifacts import write_json_artifact, write_text_artifact

    def export_trace_as_artifact(ctx: ExecutionContext, run_dir: Path) -> str:
        """Export trace context as a JSON artifact.
        
        [BATCH 4]: Integrates with artifacts.py for consistent output.
        
        Args:
            ctx: Execution context to export
            run_dir: Target run directory
            
        Returns:
            Path to the exported artifact
        """
        return write_json_artifact(
            run_dir=run_dir,
            filename=f"trace_{ctx.run_id}.json",
            payload=ctx.to_dict()
        )

    __all__ = [
        "ExecutionTracer",
        "ExecutionContext",
        "TraceEvent",
        "get_tracer",
        "start_run",
        "end_run",
        "trace_step",
        "export_trace_as_artifact",
    ]
except ImportError:
    # Graceful fallback
    __all__ = [
        "ExecutionTracer",
        "ExecutionContext",
        "TraceEvent",
        "get_tracer",
        "start_run",
        "end_run",
        "trace_step",
    ]
