"""Compatibility shim: ``src.application.rag.unified_query`` -> domain.query_core (B-5).

Canonical owner moved to ``src.application.rag.domain.query_core``. This module
re-exports everything (parse / get_rewriter / RewrittenQuery / QueryClassifier /
UnifiedQueryProcessor / ParsedQuery) for backward compatibility.
"""

from __future__ import annotations

from .domain.query_core import *  # noqa: F401,F403