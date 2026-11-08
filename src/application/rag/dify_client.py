"""Compatibility shim for legacy ``from .dify_client import ...`` imports.

Canonical Dify client lives in
:mod:`src.application.rag.infrastructure.dify_client` (see ``rag/__init__.py``
``_MODULE_MAP``: ``"DifyConfig": ".infrastructure.dify_client"``).

P1-2 (2026-09-01): the module moved to ``infrastructure/`` but the relative
imports in ``rag_answerer.py`` and ``validator.py`` were never updated, so
``rag.dify_client`` became an orphan ``.pyc`` and the Dify branch silently
dead-ended.  This shim restores the old import path.
"""

from .infrastructure.dify_client import (
    DifyConfig,
    ask_dify_kb_qa,
    extract_cap_node_ids,
    get_dify_answer_text,
    load_dify_config_from_env,
)

__all__ = [
    "DifyConfig",
    "ask_dify_kb_qa",
    "extract_cap_node_ids",
    "get_dify_answer_text",
    "load_dify_config_from_env",
]
