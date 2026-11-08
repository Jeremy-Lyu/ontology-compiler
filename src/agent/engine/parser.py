"""Agent response parser (ReAct format).

Parses raw LLM text into structured Thought / Action / Finish components.
Extracted from AgentLoop so it can be unit-tested in isolation and swapped
for alternative parsing strategies (XML-tool-call format, JSON-mode, etc.).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Parsed result dataclass
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ParsedResponse:
    """Structured result of parsing one LLM turn.

    Attributes:
        thought:      The agent's reasoning text.
        action_name:  The tool name to invoke (``None`` when finishing or no action).
        action_input: Keyword arguments for the tool call (``None`` if not applicable).
        is_finish:    ``True`` when the agent has declared the task complete.
        raw_text:     Original LLM output, preserved for debugging.
    """

    thought: str
    action_name: str | None
    action_input: dict[str, Any] | None
    is_finish: bool
    raw_text: str


# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns (compiled once at import time for performance)
# ──────────────────────────────────────────────────────────────────────────────

_FINISH_RE = re.compile(r"Finish:\s*(.*)", re.DOTALL | re.IGNORECASE)
_THOUGHT_RE = re.compile(
    r"Thought:\s*(.*?)(?=Action:\s*\{|$)",
    re.DOTALL | re.IGNORECASE,
)
_ACTION_RE = re.compile(r"Action:\s*(\{.*\})", re.DOTALL | re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────────────────
# Public parser class
# ──────────────────────────────────────────────────────────────────────────────


class ReActParser:
    """Parse LLM text produced in the ReAct (Thought/Action/Finish) format.

    Expected format::

        Thought: I should call analyze_folder first.
        Action: {"tool": "analyze_folder", "args": {"folder_path": "/data"}}

    Or when the task is complete::

        Finish: The analysis is complete. Summary: …

    The parser is intentionally lenient to accommodate minor LLM formatting
    deviations (extra whitespace, lower-case keywords, partial JSON, etc.).
    """

    def parse(self, text: str) -> ParsedResponse:
        """Parse *text* and return a :class:`ParsedResponse`.

        Args:
            text: Raw string returned by the LLM for one agent step.

        Returns:
            A ``ParsedResponse`` with all available fields populated.
        """
        # 1. Finish signal takes priority
        finish_match = _FINISH_RE.search(text)
        if finish_match:
            thought = finish_match.group(1).strip()
            logger.debug("ReActParser: Finish detected, thought=%s…", thought[:60])
            return ParsedResponse(
                thought=thought,
                action_name=None,
                action_input=None,
                is_finish=True,
                raw_text=text,
            )

        # 2. Extract thought
        thought_match = _THOUGHT_RE.search(text)
        thought = thought_match.group(1).strip() if thought_match else text.strip()

        # 3. Extract action JSON
        action_match = _ACTION_RE.search(text)
        if action_match:
            raw_json = action_match.group(1)
            try:
                action_data = json.loads(raw_json)
                action_name = action_data.get("tool")
                action_input = action_data.get("args")
                logger.debug(
                    "ReActParser: Action detected, tool=%s args=%s",
                    action_name,
                    list((action_input or {}).keys()),
                )
                return ParsedResponse(
                    thought=thought,
                    action_name=action_name,
                    action_input=action_input,
                    is_finish=False,
                    raw_text=text,
                )
            except json.JSONDecodeError:
                logger.warning("ReActParser: Failed to parse Action JSON: %s", raw_json[:120])

        # 4. No action found — return thought only (loop will handle the error)
        logger.debug("ReActParser: No Action/Finish found in LLM output.")
        return ParsedResponse(
            thought=thought,
            action_name=None,
            action_input=None,
            is_finish=False,
            raw_text=text,
        )
