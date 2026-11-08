"""Agent Loop Engine (ReAct implementation).

Coordinates the Thought -> Action -> Observation cycle by combining
an LLM client, session memory, and a registry of tools.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from src.agent.engine.llm_client import AgentLLMClient
from src.agent.engine.memory import AgentMemory
from src.agent.engine.parser import ReActParser
from src.agent.engine.trace import AgentTrace, AgentTraceStep, AgentTraceStore
from src.agent.tools.base import AgentTool, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """A single step in the agent reasoning chain."""

    step_number: int
    thought: str
    action_name: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str | None = None
    is_finish: bool = False
    error: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: str | None = None


@dataclass
class AgentLoopResult:
    """The final result of an agent run."""

    success: bool
    final_answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_steps: int = 0
    trace: AgentTrace | None = None  # populated when trace_store is provided


class AgentLoop:
    """Core ReAct engine for the Ontology Agent."""

    def __init__(
        self,
        llm_client: AgentLLMClient,
        tools: list[AgentTool],
        memory: AgentMemory,
        max_steps: int = 10,
        on_step: Callable[[AgentStep], None] | None = None,
        parser: ReActParser | None = None,
        trace_store: AgentTraceStore | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self._llm = llm_client
        self._tools = {tool.name: tool for tool in tools}
        self._memory = memory
        self._max_steps = max_steps
        self._on_step = on_step
        self._parser = parser or ReActParser()
        self._steps: list[AgentStep] = []
        self._trace_store = trace_store
        self._session_id = session_id or "default"
        self._run_id = run_id or _make_run_id()

    async def run(self) -> AgentLoopResult:
        """Run the loop until the goal is met or the step limit is reached."""
        logger.info("Starting AgentLoop for goal=%s", self._memory.goal)

        trace = AgentTrace(
            session_id=self._session_id,
            run_id=self._run_id,
            goal=self._memory.goal,
        )

        for step_number in range(1, self._max_steps + 1):
            step_started_at = _now_iso()
            step = await self._run_step(step_number)
            step.finished_at = _now_iso()
            self._steps.append(step)

            # Record into trace
            trace_step = AgentTraceStep(
                step_number=step.step_number,
                thought=step.thought,
                action_name=step.action_name,
                action_input=step.action_input,
                observation=step.observation,
                is_finish=step.is_finish,
                error=step.error,
                started_at=step_started_at,
                finished_at=step.finished_at,
                duration_ms=_duration_ms(step_started_at, step.finished_at),
            )
            trace.add_step(trace_step)

            if self._on_step is not None:
                self._on_step(step)

            if step.is_finish:
                loop_result = AgentLoopResult(
                    success=step.error is None,
                    final_answer=step.thought,
                    steps=self._steps,
                    total_steps=step_number,
                )
                trace.finish(
                    success=loop_result.success,
                    final_answer=loop_result.final_answer,
                )
                if self._trace_store is not None:
                    self._trace_store.save(trace)
                loop_result.trace = trace
                return loop_result

        loop_result = AgentLoopResult(
            success=False,
            final_answer=f"Reached maximum steps ({self._max_steps}) without finishing.",
            steps=self._steps,
            total_steps=self._max_steps,
        )
        trace.finish(success=False, final_answer=loop_result.final_answer)
        if self._trace_store is not None:
            self._trace_store.save(trace)
        loop_result.trace = trace
        return loop_result

    async def _run_step(self, step_number: int) -> AgentStep:
        messages = self._memory.to_llm_messages()

        try:
            response = await self._llm.chat(messages)
        except Exception as exc:
            logger.exception("LLM call failed at step %d", step_number)
            error_msg = f"LLM error: {exc}"
            # Record error as observation instead of immediately finishing
            observation = f"Error: {error_msg}. Please try again or adjust your approach."
            self._memory.add_observation(observation, tool_name="system")
            return AgentStep(
                step_number=step_number,
                thought=f"Encountered error: {error_msg}",
                is_finish=False,  # Allow agent to retry
                error=error_msg,
                observation=observation,
            )

        parsed = self._parser.parse(response)
        step = AgentStep(
            step_number=step_number,
            thought=parsed.thought,
            action_name=parsed.action_name,
            action_input=parsed.action_input,
            is_finish=parsed.is_finish,
        )

        self._memory.add_thought(response)

        if parsed.is_finish:
            self._memory.add_finish(parsed.thought)
            return step

        if not parsed.action_name:
            step.error = "No action found in LLM response."
            observation = f"Error: {step.error} Please use the required Action JSON format."
            step.observation = observation
            self._memory.add_observation(observation, tool_name="system")
            return step

        tool = self._tools.get(parsed.action_name)
        if tool is None:
            step.error = f"Unknown tool: {parsed.action_name}"
            observation = f"Error: {step.error}. Available tools: {list(self._tools.keys())}"
            step.observation = observation
            self._memory.add_observation(observation, tool_name="system")
            return step

        try:
            logger.info("Step %d: running tool=%s", step_number, parsed.action_name)
            result: ToolResult = await tool.run(**(parsed.action_input or {}))
            observation = result.to_observation()
            step.observation = observation
            if not result.success:
                step.error = result.error or "Tool execution failed."
            self._memory.add_observation(observation, tool_name=parsed.action_name)
        except Exception as exc:
            logger.exception("Tool execution failed: %s", parsed.action_name)
            step.error = f"Tool execution error: {exc}"
            observation = f"Error executing {parsed.action_name}: {exc}"
            step.observation = observation
            self._memory.add_observation(observation, tool_name=parsed.action_name)

        return step


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _make_run_id() -> str:
    """Generate a short unique run identifier."""
    import uuid
    return f"run-{uuid.uuid4().hex[:12]}"


def _duration_ms(start_iso: str, end_iso: str | None) -> int | None:
    """Return wall-clock duration between two ISO-8601 timestamps in ms.

    Optimization: Use fromisoformat first (Python 3.11+) which handles most
    ISO-8601 formats efficiently. Only fall back to strptime patterns if needed.
    """
    if end_iso is None:
        return None
    try:
        from datetime import datetime as _dt

        def _parse(s: str) -> _dt:
            # Try fromisoformat first (Python 3.11+ handles most ISO-8601 formats)
            try:
                return _dt.fromisoformat(s)
            except ValueError:
                pass

            # Fallback: try specific strptime patterns
            for f in (
                "%Y-%m-%dT%H:%M:%S.%f+00:00",
                "%Y-%m-%dT%H:%M:%S+00:00",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
            ):
                try:
                    return _dt.strptime(s, f)
                except ValueError:
                    continue
            raise ValueError(f"Unable to parse timestamp: {s}")

        delta = _parse(end_iso) - _parse(start_iso)
        return max(0, int(delta.total_seconds() * 1000))
    except Exception:
        return None
