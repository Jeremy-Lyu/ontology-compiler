"""Folder Analysis Result - 文件夹分析结果模型"""
from dataclasses import dataclass
from typing import Any


@dataclass
class FolderAnalysisResult:
    """Result from folder analysis."""
    finalization_bundle: dict[str, Any]
    analysis_context: dict[str, Any]
    success: bool
    error_message: str | None = None