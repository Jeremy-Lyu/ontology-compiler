"""Document bounded-context port definitions.

Ports and DTOs for reading documents from various sources. Split from
`all_ports.py` during P2 port classification (§9.8) so the document
bounded context has a dedicated, discoverable port module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class TextFragment:
    """A fragment of extracted text with lightweight provenance.

    Attributes:
        source: Short source label, e.g. "slide:3" or "page:10".
        text: Extracted text.
    """

    source: str
    text: str


@dataclass(frozen=True)
class DocumentContent:
    """Content of a discovered document."""
    path: Path
    text: str
    encoding: str = "utf-8"


class DocumentReaderPort(ABC):
    """Port for reading documents from various sources."""

    @abstractmethod
    def read(self, path: Path) -> DocumentContent:
        """Read a single document."""
        raise NotImplementedError

    @abstractmethod
    def discover(self, folder: Path, patterns: Iterable[str]) -> Iterable[Path]:
        """Discover documents matching patterns."""
        raise NotImplementedError

    @abstractmethod
    def can_read(self, file_path: str) -> bool:
        """Return True if the adapter can read the given file."""
        raise NotImplementedError

    @abstractmethod
    def read_fragments(self, file_path: str) -> List[TextFragment]:
        """Read the file and return extracted text fragments."""
        raise NotImplementedError


__all__ = [
    "TextFragment",
    "DocumentContent",
    "DocumentReaderPort",
]
