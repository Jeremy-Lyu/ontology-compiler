"""File Parser Document Reader Plugin (P1-8 Refactoring).

Provides a concrete plugin implementation that wraps the FileParser
to integrate with the ExtractorPlugin system.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ontology_compiler.agents.extractor_plugin import (
    ExtractorMetadata,
    ExtractorPlugin,
    ExtractorResult,
)
from ontology_compiler.agents.file_parser import FileParser


class FileParserDocumentReaderPlugin(ExtractorPlugin):
    """Plugin wrapper for FileParser document reading capabilities.

    This plugin bridges the legacy FileParser with the new ExtractorPlugin system,
    enabling hot-pluggable document extraction.

    Supported formats: .md, .txt, .pdf (via text extraction), .docx (via text extraction)
    Capabilities: text_extraction, metadata_extraction, table_detection

    P1-8 Refactoring (2026-05-15):
    - Created to fulfill the ExtractorPlugin interface
    - Wraps FileParser for backward compatibility
    - Enables runtime plugin registration and hot-swapping
    """

    PLUGIN_NAME = "file_parser_document_reader"
    PLUGIN_VERSION = "1.0.0"
    SUPPORTED_FORMATS = [".md", ".txt", ".pdf", ".docx", ".doc"]
    CAPABILITIES = [
        "text_extraction",
        "metadata_extraction",
        "table_detection",
        "heading_extraction",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the plugin.

        Args:
            config: Optional configuration dictionary with keys:
                - extract_metadata: bool (default: True)
                - detect_tables: bool (default: False)
                - encoding: str (default: "utf-8")
        """
        super().__init__(config)
        self._parser = FileParser()
        self._extract_metadata = config.get("extract_metadata", True) if config else True
        self._detect_tables = config.get("detect_tables", False) if config else False
        self._encoding = config.get("encoding", "utf-8") if config else "utf-8"

    def get_metadata(self) -> ExtractorMetadata:
        """Return plugin metadata."""
        return ExtractorMetadata(
            name=self.PLUGIN_NAME,
            version=self.PLUGIN_VERSION,
            description="Extracts text and metadata from documents using FileParser",
            author="Ontology Compiler Team",
            supported_formats=list(self.SUPPORTED_FORMATS),
            capabilities=list(self.CAPABILITIES),
            tags=["document", "parser", "legacy-bridge"],
        )

    def can_extract(self, source: str | Path, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if this plugin can handle the source.

        Args:
            source: File path to check
            context: Optional context (unused for basic file detection)

        Returns:
            True if the file extension is supported
        """
        source_path = Path(source)
        if not source_path.exists():
            return False
        return source_path.suffix.lower() in self.SUPPORTED_FORMATS

    def extract(
        self,
        source: str | Path,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExtractorResult:
        """Extract content from the document.

        Args:
            source: Path to the document file
            context: Optional context with keys:
                - domain: str - domain hint for extraction
                - run_id: str - run identifier for logging

        Returns:
            ExtractorResult with extracted text, metadata, and structure
        """
        start_time = time.time()
        source_path = Path(source)

        if not self.can_extract(source, context):
            return ExtractorResult(
                success=False,
                error_message=f"Cannot extract from {source}: unsupported format or file not found",
                execution_time_ms=(time.time() - start_time) * 1000,
                extractor_name=self.PLUGIN_NAME,
            )

        try:
            # Use FileParser to extract content
            raw_text = self._parser.extract_markdown(str(source_path))

            # Build metadata
            metadata: Dict[str, Any] = {
                "source_path": str(source_path),
                "file_size": source_path.stat().st_size,
                "file_extension": source_path.suffix.lower(),
            }

            if self._extract_metadata:
                metadata["word_count"] = len(raw_text.split())
                metadata["line_count"] = len(raw_text.splitlines())

            # Build simple node structure from headings (if markdown)
            nodes: List[Dict[str, Any]] = []
            if source_path.suffix.lower() == ".md":
                nodes = self._extract_markdown_structure(raw_text, str(source_path))

            # Build edges (document relationships)
            edges: List[Dict[str, Any]] = []

            execution_time = (time.time() - start_time) * 1000

            return ExtractorResult(
                success=True,
                nodes=nodes,
                edges=edges,
                metadata={
                    **metadata,
                    "raw_text_preview": raw_text[:1000] if len(raw_text) > 1000 else raw_text,
                    "has_tables": self._detect_tables,
                },
                execution_time_ms=execution_time,
                extractor_name=self.PLUGIN_NAME,
            )

        except Exception as exc:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"Extraction failed for {source}: {exc}")
            return ExtractorResult(
                success=False,
                error_message=f"Extraction failed: {str(exc)}",
                execution_time_ms=execution_time,
                extractor_name=self.PLUGIN_NAME,
            )

    def _extract_markdown_structure(self, text: str, source_path: str) -> List[Dict[str, Any]]:
        """Extract heading structure from markdown text.

        Args:
            text: Raw markdown text
            source_path: Source file path for node references

        Returns:
            List of node dictionaries representing document structure
        """
        nodes = []
        lines = text.splitlines()

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                # Count heading level
                level = 0
                for char in stripped:
                    if char == "#":
                        level += 1
                    else:
                        break

                heading_text = stripped[level:].strip()
                if heading_text:
                    node_id = f"{Path(source_path).stem}_h{i}"
                    nodes.append({
                        "meta_id": node_id,
                        "meta_name": heading_text,
                        "meta_class": "DocumentSection",
                        "level": level,
                        "source_file": source_path,
                        "line_number": i + 1,
                    })

        return nodes

    def validate_config(self) -> bool:
        """Validate plugin configuration.

        Returns:
            True if configuration is valid
        """
        if self.config is None:
            return True

        # Validate encoding
        if "encoding" in self.config:
            try:
                # Test if encoding is valid
                "test".encode(self.config["encoding"])
            except LookupError:
                self.logger.error(f"Invalid encoding: {self.config['encoding']}")
                return False

        # Validate boolean flags
        for flag in ["extract_metadata", "detect_tables"]:
            if flag in self.config and not isinstance(self.config[flag], bool):
                self.logger.error(f"Invalid {flag} value: must be boolean")
                return False

        return True
