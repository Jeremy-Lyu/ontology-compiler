"""Evidence - Unified evidence model for ontology extraction.

This module is the canonical source for the Evidence model.
Shared between src and ontology_compiler to avoid duplication.

Unified model combining Evidence, CanonicalEvidence, and RawEvidence into one.

Fields from Evidence:
    - source_file, line_range, page_range, char_offset, paragraph_id
    - extraction_rule, extractor_version, raw_text, context
    - confidence, extracted_at, run_id, metadata

Fields from CanonicalEvidence:
    - evidence_id, source_url, offset, content, extractor_id, created_at

Fields from RawEvidence:
    - file_hash, content_hash, line_numbers
    - llm_response, prompt_tokens, completion_tokens, latency_ms, fallback_used, model_name
    - extraction_confidence, quality_issues

Example:
    evidence = Evidence(
        source_file="inventory.md",
        line_range=(10, 50),
        extraction_rule="system_catalog",
        confidence=0.95
    )
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Evidence(BaseModel):
    """
    Evidence - Unified records for source evidence tracking.

    Combines fields from Evidence, CanonicalEvidence, and RawEvidence.
    """
    evidence_id: str = ""
    source_file: str = ""
    source_url: str = ""
    line_range: Optional[tuple[int, int]] = None
    page_range: Optional[tuple[int, int]] = None
    offset: Optional[int] = None
    char_offset: Optional[tuple[int, int]] = None
    paragraph_id: Optional[int] = None
    line_numbers: List[int] = Field(default_factory=list)

    content: str = ""
    raw_text: str = ""
    context: str = ""

    extraction_rule: str = ""
    extractor_id: str = ""
    extractor_version: str = "1.0"

    confidence: float = 1.0
    extraction_confidence: float = 0.0

    extracted_at: str = ""
    created_at: str = ""
    run_id: str = ""

    file_hash: str = ""
    content_hash: str = ""

    llm_response: Dict[str, Any] = Field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    fallback_used: bool = False
    model_name: str = ""

    quality_issues: List[str] = Field(default_factory=list)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('line_range', mode='before')
    @classmethod
    def validate_line_range(cls, v: Any) -> Optional[tuple[int, int]]:
        if not v:
            return None
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return (int(v[0]), int(v[1]))
        return None

    @field_validator('page_range', mode='before')
    @classmethod
    def validate_page_range(cls, v: Any) -> Optional[tuple[int, int]]:
        if not v:
            return None
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return (int(v[0]), int(v[1]))
        return None

    @field_validator('char_offset', mode='before')
    @classmethod
    def validate_char_offset(cls, v: Any) -> Optional[tuple[int, int]]:
        if not v:
            return None
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return (int(v[0]), int(v[1]))
        return None

    @field_validator('confidence', 'extraction_confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @field_validator('created_at', 'extracted_at', mode='before')
    @classmethod
    def set_timestamps(cls, v: str) -> str:
        return v if v else datetime.now().isoformat()

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["Evidence"]:
        if not data:
            return None
        return cls.model_validate(data)

    @classmethod
    def from_canonical_evidence_dict(cls, data: Dict[str, Any]) -> "Evidence":
        return cls(
            evidence_id=data.get("evidence_id", data.get("source_file", "")),
            source_file=data.get("source_file", ""),
            source_url=data.get("source_url", ""),
            line_range=data.get("line_range"),
            page_range=data.get("page_range"),
            offset=data.get("offset"),
            content=data.get("content", data.get("raw_text", "")),
            context=data.get("context", ""),
            extraction_rule=data.get("extraction_rule", ""),
            extractor_id=data.get("extractor_id", ""),
            extractor_version=data.get("extractor_version", "1.0"),
            confidence=data.get("confidence", 1.0),
            created_at=data.get("created_at", data.get("extracted_at", "")),
            metadata=data.get("metadata", {}),
        )
