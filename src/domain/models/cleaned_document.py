"""Cleaned document model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class DocumentExtractionError(Exception):
    """Error raised during document extraction."""
    pass


@dataclass
class CleanedDocument:
    """Cleaned document representation."""
    content: str
    source_path: str
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}