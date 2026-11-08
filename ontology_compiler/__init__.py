"""Ontology Compiler - Main Entry Point."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import logging

from .config import get_config, update_config

logger = logging.getLogger(__name__)


async def compile_source_file(source_file: str, domain: str = "SUPPLY_CHAIN", document_metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Pure compilation entry that does not re-enter the main pipeline."""
    from .services.extraction.unified_compilation_pipeline import CompilationRequest, UnifiedCompilationService

    logger.info("Starting pure compilation: %s (domain=%s)", source_file, domain)
    service = UnifiedCompilationService()
    request = CompilationRequest(source_path=source_file, is_folder=False, domain=domain)
    setattr(request, "document_metadata", document_metadata or {})
    result = await service.compile_source(request)

    if result.success:
        data = {
            "run_id": getattr(result, "run_id", ""),
            "compilation_id": getattr(result, "run_id", ""),
            "quality_level": getattr(result, "quality_level", "UNKNOWN"),
            "quality_score": getattr(result, "quality_score", 0.0),
            "nodes": getattr(result, "nodes", []),
            "instances": getattr(result, "instances", []),
            "relationships": getattr(result, "relationships", []),
            "node_count": getattr(result, "node_count", len(getattr(result, "nodes", []))),
            "edge_count": getattr(result, "edge_count", len(getattr(result, "relationships", []))),
            "warnings": getattr(result, "warnings", []),
            "extraction_quality_status": getattr(result, "quality_level", "UNKNOWN"),
            "validation_passed": getattr(result, "is_publishable", False),
            "validation_reason": "",
            "chunk_quality": getattr(result, "quality_level", ""),
            "chunk_metrics": [],
        }
        return {"status": "SUCCESS", "timestamp": datetime.now().isoformat(), "data": data, "error": None}

    return {
        "status": "ERROR",
        "timestamp": datetime.now().isoformat(),
        "data": None,
        "error": str(getattr(result, "errors", result)),
    }


class OntologyCompiler:
    """High-level API for ontology compilation."""

    def __init__(self, config_dict: Optional[Dict] = None):
        if config_dict:
            for key, value in config_dict.items():
                update_config(**{key: value})
        self.config = get_config()

    def _format_result(self, status: str, data: Any = None, error: str = None) -> Dict:
        return {"status": status, "timestamp": datetime.now().isoformat(), "data": data, "error": error}

    async def compile(self, source_file: str) -> Dict[str, Any]:
        return await compile_source_file(source_file)

    async def compile_batch(self, source_files: List[str], semantic_mode: bool = False) -> Dict[str, Any]:
        results = [await compile_source_file(source_file) for source_file in source_files]
        successful = sum(1 for item in results if item.get("status") == "SUCCESS")
        return self._format_result(
            "SUCCESS" if successful else "ERROR",
            data={
                "total": len(source_files),
                "successful": successful,
                "mode": "pure_compilation_entry",
                "warnings": [item.get("error") for item in results if item.get("error")],
            },
            error=None if successful else "batch compilation failed",
        )

    async def compile_tbh_semantic(
        self,
        tbh_dir: str = "data/TBH",
        run_id: str | None = None,
        publish_shared: bool = True,
        skip_rigor: bool = False,
        smoke_mode: bool = False,
        max_files: int | None = None,
        include_patterns: List[str] | None = None,
        exclude_patterns: List[str] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        domain: str = "SUPPLY_CHAIN",
    ) -> Dict[str, Any]:
        root = Path(tbh_dir)
        if not root.exists():
            return self._format_result("ERROR", error=f"TBH directory not found: {tbh_dir}")
        source_files = sorted(str(p) for p in root.rglob("*.md"))
        if smoke_mode and source_files:
            source_files = source_files[:1]
        if max_files is not None:
            source_files = source_files[:max_files]
        return await self.compile_batch(source_files)

    def get_config(self) -> Dict[str, Any]:
        return self.config.to_dict()

    def update_config(self, **kwargs) -> None:
        update_config(**kwargs)


__all__ = ["OntologyCompiler", "compile_source_file"]
