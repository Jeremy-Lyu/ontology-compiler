"""Plugin adapter for PDF document extraction.

Provides hot-pluggable interface for PDF text extraction and ontology generation.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .extractor_plugin import ExtractorMetadata, ExtractorPlugin, ExtractorResult


class PDFExtractorPlugin(ExtractorPlugin):
    """Plugin adapter for PDF document extraction."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def get_metadata(self) -> ExtractorMetadata:
        """Return metadata about this extractor."""
        return ExtractorMetadata(
            name="pdf_document",
            version="1.0.0",
            description="PDF document text extractor with structure preservation",
            author="Ontology Compiler Team",
            supported_formats=["pdf"],
            capabilities=[
                "text_extraction",
                "page_structure",
                "section_detection",
                "metadata_extraction",
            ],
            tags=["pdf", "document", "text", "unstructured"],
        )

    def can_extract(self, source: str | Path, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if this extractor can handle the given source."""
        if isinstance(source, Path):
            suffix = source.suffix.lower()
            return suffix == ".pdf"
        return False

    def extract(
        self,
        source: str | Path,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExtractorResult:
        """Extract text from PDF source.

        P0 BUGFIX (2026-08-04):
        - ExtractorResult contract (P0-2 standardization) only accepts ``fragment``
          as the carrier of entities; passing ``nodes=`` / ``metadata=`` would
          raise TypeError because those fields are not on the dataclass.
        - Now we go through ``_create_fragment`` (canonical Fragment builder on
          the ABC base) and wrap via ``ExtractorResult.from_fragment``. This is
          the path guaranteed to work with downstream Fragment->Canonical adapters.
        """
        context = context or {}
        start_time = time.time()
        source_file = str(source) if isinstance(source, Path) else source
        ctx_for_fragment: Dict[str, Any] = {
            "source_file": source_file,
            "domain": context.get("domain", "SUPPLY_CHAIN"),
        }

        try:
            text_content = self._extract_pdf_text(source_file)

            nodes = [{
                "meta_id": f"PDF-{Path(source_file).stem}",
                "meta_name": Path(source_file).stem,
                "meta_class": "Document",
                "description": f"PDF document: {Path(source_file).name}",
                "source_file": source_file,
                "document_type": "PDF",
                "text_preview": text_content[:500] if text_content else "",
            }]

            fragment = self._create_fragment(
                nodes=nodes,
                instances=[],
                relationships=[],
                context=ctx_for_fragment,
            )
            fragment.metadata["text_length"] = len(text_content) if text_content else 0

            result = ExtractorResult.from_fragment(fragment, success=True)
            # execution_time_ms is defaulted to 0 by the dataclass; fill now
            object.__setattr__(result, "execution_time_ms", (time.time() - start_time) * 1000)
            return result

        except Exception as e:
            self.logger.error(f"PDF extraction failed: {e}")
            result = ExtractorResult(
                success=False,
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
                extractor_name="pdf_document",
            )
            object.__setattr__(result, "run_id", context.get("run_id", ""))
            return result

    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file.
        
        Tries multiple libraries in order of preference.
        If FORCE_PDF_OCR environment variable is set to "true", will force OCR
        extraction (good for corrupted/browser-printed PDFs that have garbled
        text layer but readable content on the page images).
        """
        # Check if forced OCR is requested via env var
        force_ocr = os.environ.get("FORCE_PDF_OCR", "false").lower() == "true"
        
        if not force_ocr:
            # Try pdfplumber first (best for structured text)
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text_parts = []
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                    return "\n".join(text_parts)
            except ImportError:
                pass

            # Try PyPDF2 as fallback
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text_parts = []
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                    return "\n".join(text_parts)
            except ImportError:
                pass
        
        # Force OCR or no libraries available: use OCR via pytesseract
        try:
            from pdf2image import convert_from_path
            import pytesseract
            from PIL import Image
            
            pages = convert_from_path(file_path)
            text_parts = []
            for page_image in pages:
                text = pytesseract.image_to_string(page_image)
                if text:
                    text_parts.append(text)
            return "\n".join(text_parts)
        except ImportError:
            self.logger.warning("OCR libraries not available. Install pdf2image, pytesseract and Pillow.")
            return ""
