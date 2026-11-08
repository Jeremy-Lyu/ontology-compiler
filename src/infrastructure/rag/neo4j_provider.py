"""Neo4j SDK provider (B-4, 2026-08-17).

Owns the only ``neo4j`` SDK import in the codebase. ``application/`` code that
needs the driver goes through this module instead of touching the SDK directly,
satisfying RAG-05 (provider calls via ports / infra) and keeping
``src.application.rag`` free of direct SDK imports.
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from neo4j import GraphDatabase
    _NEO4J_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional dep
    GraphDatabase = None  # type: ignore
    _NEO4J_AVAILABLE = False


def is_neo4j_available() -> bool:
    """Whether the neo4j SDK is importable in this environment."""
    return _NEO4J_AVAILABLE and GraphDatabase is not None


def get_graph_database_class() -> Optional[Any]:
    """Return the ``GraphDatabase`` class (or None when SDK is missing).

    Callers use it to open drivers / sessions. Kept as a single indirection so
    the SDK import never leaks into the application layer.
    """
    return GraphDatabase


def open_driver(
    uri: str,
    auth: Optional[tuple[str, str]] = None,
    **kwargs: Any,
) -> Optional[Any]:
    """Open a Neo4j driver via the SDK (shortcut over get_graph_database_class)."""
    cls = get_graph_database_class()
    if cls is None:
        return None
    try:
        return cls.driver(uri, auth=auth, **kwargs)
    except Exception:
        return None