"""Plugin adapter for CSV/Excel extraction.

[P1-1 2026-06-15]: 合并 CSVSourceExtractor 到 CSVExtractorPlugin
直接使用 StructuredDataExtractor 核心功能，消除重复包装。

[BATCH 1 2026-06-30]: StructuredDataExtractor now implements ExtractorPlugin interface,
so this wrapper is now a thin facade for backward compatibility with the 'csv_excel' plugin name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .extractor_plugin import ExtractorMetadata, ExtractorPlugin, ExtractorResult

# [BATCH 1] StructuredDataExtractor now implements ExtractorPlugin directly
from ontology_compiler.services.structured_data_extractor import StructuredDataExtractor


class CSVExtractorPlugin(StructuredDataExtractor):
    """Plugin adapter for CSV/Excel structured data extraction.
    
    [P1-1]: 直接集成 StructuredDataExtractor，消除重复包装。
    [BATCH 1]: Now inherits from StructuredDataExtractor for unified interface.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config)

    def get_metadata(self) -> ExtractorMetadata:
        """Return metadata about this extractor."""
        return ExtractorMetadata(
            name="csv_excel",
            version="2.0.0",  # [BATCH 1] 版本升级 - unified interface
            description="Structured data extractor for CSV and Excel files (unified)",
            author="Ontology Compiler Team",
            supported_formats=["csv", "xlsx", "xls"],
            capabilities=[
                "column_extraction",
                "row_to_instance",
                "data_type_inference",
                "schema_detection",
                "bom_analysis",
                "temporal_analysis",
            ],
            tags=["structured", "csv", "excel", "tabular", "unified"],
        )
