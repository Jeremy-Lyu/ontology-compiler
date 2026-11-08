"""Compatibility shim: ``src.application.rag.models`` -> domain.models (B-5).

Canonical owner moved to ``src.application.rag.domain.models``. This module
re-exports everything for backward compatibility (adapters / ports / older rag
modules still import ``.models``).
"""

from __future__ import annotations

from .domain.models import *  # noqa: F401,F403
from .domain.models import RESPONSE_MODE_BLOCKING, RESPONSE_MODE_STREAMING  # noqa: F401