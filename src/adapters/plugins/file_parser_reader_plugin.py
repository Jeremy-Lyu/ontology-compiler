"""File Parser Document Reader Plugin (P1-2 Enhancement).

Provides a concrete DocumentReaderPlugin implementation that wraps the
FileParserDocumentReader adapter for hot-pluggable document reading.
"""
from __future__ import annotations

from typing import Any, Dict

from src.application.services.adapter_plugin_service import (
    AdapterPluginMetadata,
    DocumentReaderPlugin,
)
from src.adapters.outbound.parsers.file_parser_document_reader import FileParserDocumentReader


class FileParserReaderPlugin(DocumentReaderPlugin):
    """Plugin wrapper for FileParserDocumentReader adapter.

    This plugin bridges the existing FileParserDocumentReader adapter with
    the AdapterPlugin system, enabling hot-pluggable document reading.

    Supported formats: .md, .txt, .pdf, .pptx, .docx, .doc
    Capabilities: text_extraction, provenance_tracking, slide_splitting, page_splitting

    P1-2 Enhancement (2026-05-30):
    - Created to fulfill the DocumentReaderPlugin interface
    - Wraps FileParserDocumentReader for backward compatibility
    - Enables runtime plugin registration and hot-swapping
    """

    PLUGIN_NAME = "file_parser_reader"
    PLUGIN_VERSION = "1.0.0"
    SUPPORTED_FORMATS = [".md", ".txt", ".pdf", ".pptx", ".docx", ".doc"]
    PORT_TYPE = "DocumentReaderPort"

    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize the plugin.

        Args:
            config: Optional configuration dictionary (currently unused)
        """
        super().__init__(config)
        self._reader = FileParserDocumentReader()

    @property
    def metadata(self) -> AdapterPluginMetadata:
        """Return plugin metadata."""
        return AdapterPluginMetadata(
            name=self.PLUGIN_NAME,
            version=self.PLUGIN_VERSION,
            port_type=self.PORT_TYPE,
            description="Document reader using FileParser with provenance tracking",
            author="Ontology Compiler Team",
            supported_formats=list(self.SUPPORTED_FORMATS),
            config_schema={},
            priority=100,
            enabled=True,
        )

    def initialize(self) -> bool:
        """Initialize the plugin.

        Returns:
            True if initialization successful
        """
        self._initialized = True
        return True

    def create_reader(self):
        """Create and return a document reader instance.

        Returns:
            FileParserDocumentReader instance
        """
        return self._reader
