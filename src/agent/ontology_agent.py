"""Ontology Agent facade.

Assembles the LLM client, session memory, and tool registry into a single
high-level entry point for autonomous ontology analysis tasks.

U4.5 additions
--------------
- ``session_id``: pass a string to resume an existing conversation from disk.
- ``trace_store``: pass an ``AgentTraceStore`` instance to persist run traces.
- ``enable_session_persistence``: when True the agent automatically saves the
  updated memory after each run so the session can be resumed later.
"""
from __future__ import annotations

import logging
from typing import Any

from src.agent.engine.llm_client import AgentLLMClient
from src.agent.engine.loop import AgentLoop, AgentLoopResult
from src.agent.engine.memory import AgentMemory
from src.agent.engine.session_store import AgentSessionStore
from src.agent.engine.trace import AgentTraceStore
from src.agent.tools.base import AgentTool
from src.agent.tools.ontology_tools import default_ontology_tools
from src.application.ports.llm_ports import LLMPort

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """You are an Ontology Analysis Expert.

You help users analyze document folders, extract structured knowledge, and
summarize ontology-relevant findings.

Use the following protocol:
- Thought: explain what you will do next.
- Action: emit JSON in the form {{"tool": "tool_name", "args": {{...}}}}.
- Finish: provide the final answer when the task is complete.

Available tools:
{tool_descriptions}

Rules:
- Use only the listed tools.
- Prefer concise reasoning.
- If a tool fails, inspect the observation and adjust parameters.
- Stop once the goal is complete.
- MEMORY FIRST: when the user mentions a domain or topic, call query_capsule
  before discover_documents or analyze_folder. If the capsule result is
  sufficient, skip the analysis and summarize from experience.
"""


class OntologyAgent:
    """High-level facade that runs the Agent loop."""

    def __init__(
        self,
        llm_port: LLMPort,
        tools: list[AgentTool] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 10,
        capsule_storage_dir: str = "outputs/capsules",
        # U4.5 — observability & persistence
        session_id: str | None = None,
        trace_store: AgentTraceStore | None = None,
        session_store: AgentSessionStore | None = None,
        enable_session_persistence: bool = False,
    ) -> None:
        self._llm_client = AgentLLMClient(llm_port)
        self._tools = tools or default_ontology_tools(
            capsule_storage_dir=capsule_storage_dir
        )
        self._system_prompt = system_prompt
        self._max_steps = max_steps
        self._session_id = session_id
        self._trace_store = trace_store
        self._session_store = session_store or (
            AgentSessionStore() if enable_session_persistence else None
        )
        self._enable_session_persistence = enable_session_persistence

    async def run(self, user_query: str) -> AgentLoopResult:
        """Run the agent against one user query.

        If ``session_id`` is set and ``session_store`` is available the agent
        will attempt to restore a previous memory state before running, and
        will save the updated memory afterwards.
        """
        logger.info("OntologyAgent received query=%s", user_query)

        # -- Restore or initialise session memory --
        memory: AgentMemory | None = None
        if self._session_id and self._session_store:
            memory = self._session_store.load(self._session_id)
            if memory is not None:
                logger.info(
                    "OntologyAgent: restored session %s (%d messages)",
                    self._session_id, len(memory.messages),
                )
                # Append the new user query to the restored conversation
                memory.add_user(user_query)
                memory.goal = user_query  # update goal to the latest query

        if memory is None:
            memory = AgentMemory(goal=user_query)
            tool_descriptions = "\n".join(
                f"- {tool.name}: {tool.description}\n  parameters: {tool.parameters}"
                for tool in self._tools
            )
            memory.add_system(self._system_prompt.format(tool_descriptions=tool_descriptions))
            memory.add_user(user_query)

        loop = AgentLoop(
            llm_client=self._llm_client,
            tools=self._tools,
            memory=memory,
            max_steps=self._max_steps,
            trace_store=self._trace_store,
            session_id=self._session_id or "default",
        )
        result = await loop.run()

        # -- Persist session memory if requested --
        if self._session_id and self._session_store:
            self._session_store.save(self._session_id, memory)
            logger.info("OntologyAgent: saved session %s", self._session_id)

        return result


async def run_ontology_agent(
    user_query: str,
    llm_port: LLMPort,
    tools: list[AgentTool] | None = None,
    max_steps: int = 10,
    capsule_storage_dir: str = "outputs/capsules",
    session_id: str | None = None,
    trace_store: AgentTraceStore | None = None,
    enable_session_persistence: bool = False,
) -> str:
    """Convenience wrapper that returns the final answer string."""
    agent = OntologyAgent(
        llm_port=llm_port,
        tools=tools,
        max_steps=max_steps,
        capsule_storage_dir=capsule_storage_dir,
        session_id=session_id,
        trace_store=trace_store,
        enable_session_persistence=enable_session_persistence,
    )
    result = await agent.run(user_query)
    return result.final_answer
