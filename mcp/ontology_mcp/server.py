"""Minimal MCP stdio server for the ontology meaning surface.

Implements a practical subset of the MCP JSON-RPC surface:
  initialize / tools/list / tools/call / ping

No writes. Run::

    python -m mcp.ontology_mcp.server

Or register in an MCP client as a stdio command pointing at this module.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is importable when launched as a stdio process.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcp.ontology_mcp.tools import TOOLS, handle_tool  # noqa: E402

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "ontology-mcp", "version": "0.1.0"}


def _ok(req_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def dispatch(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    method = message.get("method")
    req_id = message.get("id")
    params = message.get("params") or {}

    # Notifications have no id — acknowledge by silence.
    if req_id is None and method and method.startswith("notifications/"):
        return None

    if method == "initialize":
        return _ok(
            req_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        )
    if method == "ping":
        return _ok(req_id, {})
    if method == "tools/list":
        return _ok(req_id, {"tools": TOOLS})
    if method == "tools/call":
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        try:
            payload = handle_tool(name, arguments if isinstance(arguments, dict) else {})
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            return _ok(
                req_id,
                {
                    "content": [{"type": "text", "text": text}],
                    "structuredContent": payload,
                    "isError": False,
                },
            )
        except KeyError as exc:
            return _ok(
                req_id,
                {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            return _ok(
                req_id,
                {
                    "content": [{"type": "text", "text": f"tool error: {exc}"}],
                    "isError": True,
                },
            )
    if method == "shutdown":
        return _ok(req_id, {})
    return _err(req_id, -32601, f"Method not found: {method}")


def serve_stdio() -> int:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.stdout.write(json.dumps(_err(None, -32700, f"parse error: {exc}")) + "\n")
            sys.stdout.flush()
            continue
        response = dispatch(message if isinstance(message, dict) else {})
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        if isinstance(message, dict) and message.get("method") == "shutdown":
            break
    return 0


def main() -> int:
    return serve_stdio()


if __name__ == "__main__":
    raise SystemExit(main())
