"""Curated meaning-first MCP tools (read-only).

Mirrors NVIDIA Auto Ontology's allow-list habit: expose glossary/search/
answerability, never raw master JSON dumps or filesystem walks.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from src.application.rag.publish_gate import read_publish_receipt
from src.application.rag.governed_packets import default_governed_rag_dir, read_latest_pointer
from src.application.rag.semantic_surface import (
    ask_question,
    check_answerable,
    get_entity,
    search_entities,
)

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "search_entities",
        "description": (
            "Search the governed ontology glossary by name, alias, type, or id. "
            "Use first to learn what a business word means before answering."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search text"},
                "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_entity",
        "description": (
            "Fetch one entity and its incident relations by id. "
            "Use after search_entities to confirm meaning and links."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "check_answerable",
        "description": (
            "Cheap preflight: grade whether the governed packet can ground a "
            "natural-language question. Prefer this before ask/query."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "check_readiness",
        "description": (
            "Report whether a governed packet pointer and optional publish "
            "receipt are present. Does not dump packet contents."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "ask_question",
        "description": (
            "Ask a meaning question with a mandatory answerability gate. "
            "If the governed packet cannot ground the question, refuses "
            "(refused=true) and lists gaps — does not invent. Returns a "
            "grounded evidence sketch (entities / relation types), not an LLM essay."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "force": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, return evidence sketch even when not answerable",
                },
            },
            "required": ["question"],
        },
    },
]


def _tool_search_entities(arguments: Dict[str, Any]) -> Dict[str, Any]:
    query = str(arguments.get("query") or "")
    limit = int(arguments.get("limit") or 10)
    hits = search_entities(query, limit=limit)
    return {"hits": [h.to_dict() for h in hits], "count": len(hits)}


def _tool_get_entity(arguments: Dict[str, Any]) -> Dict[str, Any]:
    entity_id = str(arguments.get("entity_id") or "")
    payload = get_entity(entity_id)
    if payload is None:
        return {"found": False, "entity_id": entity_id}
    return {"found": True, **payload}


def _tool_check_answerable(arguments: Dict[str, Any]) -> Dict[str, Any]:
    question = str(arguments.get("question") or "")
    return check_answerable(question).to_dict()


def _tool_check_readiness(_arguments: Dict[str, Any]) -> Dict[str, Any]:
    pointer = read_latest_pointer()
    if not pointer:
        return {
            "ready": False,
            "gaps": ["latest_governed_packet.json missing — run build_governed_packet.py"],
        }
    packet_dir = default_governed_rag_dir()
    receipt = None
    receipt_error = None
    try:
        receipt = read_publish_receipt(packet_dir)
    except Exception as exc:
        receipt_error = str(exc)
    ready = True
    gaps: List[str] = []
    if receipt is None:
        gaps.append(f"publish_receipt missing or invalid: {receipt_error}")
    elif not receipt.get("publishable", True):
        gaps.append("publish_receipt.publishable is false")
        ready = False
    return {
        "ready": ready and receipt is not None,
        "packet_id": pointer.get("packet_id"),
        "packet_dir": str(packet_dir),
        "publishable": None if receipt is None else bool(receipt.get("publishable")),
        "certified": None if receipt is None else bool(receipt.get("certified")),
        "gaps": gaps,
    }


def _tool_ask_question(arguments: Dict[str, Any]) -> Dict[str, Any]:
    question = str(arguments.get("question") or "")
    force = bool(arguments.get("force") or False)
    return ask_question(question, force=force)


_HANDLERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "search_entities": _tool_search_entities,
    "get_entity": _tool_get_entity,
    "check_answerable": _tool_check_answerable,
    "check_readiness": _tool_check_readiness,
    "ask_question": _tool_ask_question,
}


def handle_tool(name: str, arguments: Dict[str, Any] | None = None) -> Dict[str, Any]:
    handler = _HANDLERS.get(name)
    if handler is None:
        raise KeyError(f"unknown tool: {name}")
    return handler(arguments or {})


def tools_as_text() -> str:
    return json.dumps(TOOLS, ensure_ascii=False, indent=2)
