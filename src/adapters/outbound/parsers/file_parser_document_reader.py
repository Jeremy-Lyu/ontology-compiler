"""Adapter that reads documents using existing `ontology_compiler.agents.file_parser.FileParser`."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from ontology_compiler.agents.file_parser import FileParser
from src.application.ports.document_ports import DocumentReaderPort, TextFragment, DocumentContent

logger = logging.getLogger(__name__)


class FileParserDocumentReader(DocumentReaderPort):
    """Document reader adapter based on `FileParser`.

    This adapter keeps dependencies in the adapters layer while reusing the existing
    parsing logic and approved third-party libraries.
    """

    _SLIDE_MARKER_RE = re.compile(r"^---\s*Slide\s+(\d+)\s*---\s*$", re.IGNORECASE)

    def can_read(self, file_path: str) -> bool:
        """Return True if the underlying `FileParser` supports the file."""

        return FileParser.can_parse(file_path)

    def read_fragments(self, file_path: str) -> List[TextFragment]:
        """Read text using `FileParser` and attempt to split into provenance fragments."""

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        suffix = path.suffix.lower()

        text = FileParser.extract_text(file_path)
        if suffix == ".pptx":
            return self._split_pptx(text)
        if suffix == ".pdf":
            return self._split_pdf_pages(text)

        return [TextFragment(source=suffix.lstrip("."), text=text)]

    def _split_pptx(self, text: str) -> List[TextFragment]:
        fragments: List[TextFragment] = []
        current_slide: Optional[str] = None
        buf: List[str] = []

        def flush() -> None:
            nonlocal buf
            if not buf:
                return
            label = current_slide or "pptx"
            fragments.append(TextFragment(source=label, text="\n".join(buf).strip()))
            buf = []

        for line in (text or "").splitlines():
            match = self._SLIDE_MARKER_RE.match(line.strip())
            if match:
                flush()
                current_slide = f"slide:{match.group(1)}"
                continue
            if line.strip():
                buf.append(line)

        flush()

        if not fragments and (text or "").strip():
            fragments.append(TextFragment(source="pptx", text=text))

        return fragments

    @staticmethod
    def _split_pdf_pages(text: str) -> List[TextFragment]:
        pages = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
        if len(pages) <= 1:
            return [TextFragment(source="pdf", text=(text or ""))]
        return [TextFragment(source=f"page:{idx}", text=page) for idx, page in enumerate(pages, 1)]

    def read(self, path: Path) -> DocumentContent:
        """Read a single document."""
        text = FileParser.extract_text(str(path))
        return DocumentContent(path=path, text=text)

    def discover(self, folder: Path, patterns: Iterable[str]) -> Iterable[Path]:
        """Discover documents matching patterns."""
        from fnmatch import fnmatch

        for pattern in patterns:
            for path in folder.rglob(pattern):
                if path.is_file() and self.can_read(str(path)):
                    yield path
