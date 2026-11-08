"""Adapter plugins package.

P1-8 Refactoring (2026-05-15):
Provides concrete plugin implementations for the ExtractorPlugin system.
"""

from __future__ import annotations

from .file_parser_document_reader_plugin import FileParserDocumentReaderPlugin

__all__ = ["FileParserDocumentReaderPlugin"]
