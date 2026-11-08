"""Capsule memory tools for the Ontology Agent (U4.4).

Wraps the existing ``CapsuleManager`` so the Agent can query long-term
experience before deciding on an analysis strategy.

Available tools
---------------
QueryCapsuleTool   — full-text search across stored capsules; returns the
                     top-N most relevant capsule summaries as an Observation.

Design notes
------------
- Read-only by default: the Agent queries but does not write capsules.
  Capsule persistence is handled by ``save_capsule_experience`` in the
  Application layer (called at the end of ``analyze_folder_once``).
- The manager is lazily instantiated with a configurable ``storage_dir`` so
  tests can point to a temp directory without touching real outputs.
- ``to_observation()`` is intentionally token-conservative: we only surface
  the ``summary``, ``metadata.tags``, ``metadata.contexts``, and the most
  recent experiment metrics to stay within budget.
"""
from __future__ import annotations

import logging
from typing import Any

from src.application.ports.artifact_ports import CapsuleQueryPort
from src.agent.tools.base import AgentTool, ToolResult  # noqa: F401 re-export
from ontology_compiler.capsules.manager import CapsuleManager, CapsuleQueryAdapter

logger = logging.getLogger(__name__)

# Maximum characters allowed in the observation produced by one capsule entry.
# Prevents context explosion when a capsule has a very long description.
_MAX_CAPSULE_OBS_CHARS: int = 500


# ──────────────────────────────────────────────────────────────────────────────
# Tool: query_capsule
# ──────────────────────────────────────────────────────────────────────────────


class QueryCapsuleTool:
    """Query historical capsule experience before starting analysis.

    Always call this *before* ``analyze_folder`` or ``discover_documents``
    when the user mentions a domain or topic that may have been analyzed
    before.  The returned summaries help the agent select the right strategy
    and avoid duplicating completed work.

    Returns a list of the most relevant capsule records, each containing:
    - ``capsule_id``   : unique identifier (e.g. ``capsule-bom``)
    - ``name``         : human-readable capsule name
    - ``summary``      : short description of what the capsule captured
    - ``tags``         : domain / topic classification tags
    - ``contexts``     : run and domain contexts associated with this capsule
    - ``success_rate`` : fraction of runs that were fully successful
    - ``last_experiment``: metrics from the most recent recorded experiment
    """

    name: str = "query_capsule"
    description: str = (
        "Search historical analysis experience (capsules) by free-text query. "
        "Use before analyze_folder when the user mentions a known domain/topic. "
        "Returns top matching capsule summaries including quality metrics and "
        "sample domain objects from previous runs."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Free-text search query, e.g. 'supply chain BOM inventory'. "
                    "Multi-word queries improve recall."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of capsules to return (default 5).",
                "default": 5,
            },
            "storage_dir": {
                "type": "string",
                "description": (
                    "Path to the capsule storage directory. "
                    "Leave empty to use the default 'outputs/capsules'."
                ),
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        storage_dir: str = "outputs/capsules",
        capsule_query_port: CapsuleQueryPort | None = None,
    ) -> None:
        self._storage_dir = storage_dir
        self._port = capsule_query_port

    # ------------------------------------------------------------------ #
    # AgentTool.run
    # ------------------------------------------------------------------ #

    async def run(
        self,
        query: str,
        limit: int = 5,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a capsule query and return summarized results."""
        effective_limit = max(1, int(limit))
        storage_dir_override = kwargs.get("storage_dir")
        effective_dir = (
            str(storage_dir_override).strip()
            if storage_dir_override is not None and str(storage_dir_override).strip()
            else self._storage_dir
        )

        logger.info(
            "QueryCapsuleTool.run: query=%r limit=%d",
            query, effective_limit,
        )
        try:
            port = self._port
            if storage_dir_override is not None:
                # Explicit storage_dir request should always override injected ports.
                port = CapsuleQueryAdapter(CapsuleManager(storage_dir=effective_dir))
            elif port is None:
                port = CapsuleQueryAdapter(CapsuleManager(storage_dir=effective_dir))

            capsules = await port.query_capsules(query, limit=effective_limit)

            if not capsules:
                return ToolResult(
                    success=True,
                    content={
                        "query": query,
                        "capsule_count": 0,
                        "capsules": [],
                        "hint": (
                            "No relevant capsules found. Proceed with discover_documents "
                            "or analyze_folder to start a fresh analysis."
                        ),
                    },
                    tool_name=self.name,
                    metadata={"capsule_count": 0, "storage_dir": effective_dir},
                )

            capsule_records = []
            for cap in capsules:
                # cap is expected to have metadata and summary as per Capsule class
                # but if adapter returns raw objects, we handle that
                
                # Extract the most recent experiment metrics (if any)
                last_exp: dict[str, Any] = {}
                evolution_chain = getattr(cap, "evolution_chain", [])
                if evolution_chain:
                    raw = evolution_chain[-1]
                    if isinstance(raw, dict):
                        last_exp = {
                            k: raw[k]
                            for k in ("grade", "l1_score", "l2_score", "success", "timestamp")
                            if k in raw
                        }

                # Trim summary to token budget
                cap_summary = getattr(cap, "summary", "")
                summary = cap_summary[:_MAX_CAPSULE_OBS_CHARS]
                if len(cap_summary) > _MAX_CAPSULE_OBS_CHARS:
                    summary += "…"

                cap_metadata = getattr(cap, "metadata", None)
                if cap_metadata:
                    capsule_records.append({
                        "capsule_id": cap_metadata.id,
                        "name": cap_metadata.name,
                        "summary": summary,
                        "tags": cap_metadata.tags,
                        "contexts": cap_metadata.contexts[:5],  # cap context list length
                        "success_rate": round(cap_metadata.success_rate, 4),
                        "avg_improvement": cap_metadata.avg_improvement,
                        "last_experiment": last_exp,
                    })
                else:
                    # Fallback for plain dict or unknown shape
                    capsule_records.append({
                        "summary": summary,
                        "last_experiment": last_exp,
                    })

            content = {
                "query": query,
                "capsule_count": len(capsule_records),
                "capsules": capsule_records,
                "hint": (
                    "Relevant capsules found. Review summaries and last_experiment metrics "
                    "to decide whether to reuse insights or run a fresh analysis."
                ),
            }
            return ToolResult(
                success=True,
                content=content,
                tool_name=self.name,
                metadata={
                    "capsule_count": len(capsule_records),
                    "storage_dir": effective_dir,
                    "query": query,
                },
            )
        except Exception as exc:
            logger.exception("QueryCapsuleTool failed: %s", exc)
            return ToolResult(
                success=False,
                content=None,
                tool_name=self.name,
                error=str(exc),
            )
