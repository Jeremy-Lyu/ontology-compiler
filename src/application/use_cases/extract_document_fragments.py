"""Application use-case for extracting text fragments from one document."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.application.ports.document_ports import DocumentReaderPort, TextFragment
from src.domain.models.analysis_models import DocumentExtractionError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractDocumentFragmentsRequest:
    """Request DTO for extracting document fragments."""

    file_path: str


class ExtractDocumentFragmentsUseCase:
    """Validate one file and extract provenance-aware text fragments."""

    def __init__(self, document_reader: DocumentReaderPort):
        self._document_reader = document_reader

    def execute(self, request: ExtractDocumentFragmentsRequest) -> List[TextFragment]:
        file_path = request.file_path
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        if not self._document_reader.can_read(file_path):
            raise ValueError(f"Unsupported file type: {path.suffix.lower()}")

        try:
            fragments = self._document_reader.read_fragments(file_path)
        except Exception as exc:
            logger.exception("Document extraction failed")
            raise DocumentExtractionError(str(exc)) from exc

        if not isinstance(fragments, list):
            raise DocumentExtractionError("DocumentReaderPort.read_fragments must return a list")
        return fragments

