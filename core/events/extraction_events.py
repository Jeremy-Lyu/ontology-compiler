"""Extraction Domain Events for G26-ARCH-01.

G26-ARCH-01: Typed events for extraction workflow.

Events:
- FragmentExtractedEvent: Published when a fragment is successfully extracted
- ExtractionFailedEvent: Published when extraction fails
- RIGORRefinementCompletedEvent: Published when RIGOR refinement completes
- FragmentValidatedEvent: Published when a fragment is validated against schema
- FragmentPersistedEvent: Published when a fragment is persisted to storage

These events enable loose coupling between extraction components and downstream
processors (RAG indexing, validation, storage, quality assurance).
"""

from typing import Any, List, Dict, Optional

from .base_event import DomainEvent


class FragmentExtractedEvent(DomainEvent):
    """Event emitted when a fragment is successfully extracted from a document.

    This event carries structured extraction results for downstream processing:
    - Entities extracted from the document
    - Relations between entities
    - Concepts, scenarios, and rules (when applicable)
    - Metadata for tracing and quality assessment

    Usage:
        event = FragmentExtractedEvent(
            source_file="document.md",
            domain="SUPPLY_CHAIN",
            entities=entities,
            relations=relations,
            run_id="ONT-20260625-173557-abc123",
            extraction_strategy="technical",
            quality_score=0.85,
        )
    """

    def __init__(
        self,
        source_file: str,
        domain: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        concepts: Optional[List[Dict[str, Any]]] = None,
        scenarios: Optional[List[Dict[str, Any]]] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        run_id: Optional[str] = None,
        extraction_strategy: str = "unified",
        quality_score: Optional[float] = None,
        extractor_name: str = "unknown",
    ) -> None:
        super().__init__()
        self.source_file = source_file
        self.domain = domain
        self.entities = entities or []
        self.relations = relations or []
        self.concepts = concepts or []
        self.scenarios = scenarios or []
        self.rules = rules or []
        self.run_id = run_id
        self.extraction_strategy = extraction_strategy
        self.quality_score = quality_score
        self.extractor_name = extractor_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at,
            "source_file": self.source_file,
            "domain": self.domain,
            "entities": self.entities,
            "relations": self.relations,
            "concepts": self.concepts,
            "scenarios": self.scenarios,
            "rules": self.rules,
            "run_id": self.run_id,
            "extraction_strategy": self.extraction_strategy,
            "quality_score": self.quality_score,
            "extractor_name": self.extractor_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FragmentExtractedEvent":
        return cls(
            source_file=data["source_file"],
            domain=data.get("domain", "SUPPLY_CHAIN"),
            entities=data.get("entities", []),
            relations=data.get("relations", []),
            concepts=data.get("concepts", []),
            scenarios=data.get("scenarios", []),
            rules=data.get("rules", []),
            run_id=data.get("run_id"),
            extraction_strategy=data.get("extraction_strategy", "unified"),
            quality_score=data.get("quality_score"),
            extractor_name=data.get("extractor_name", "unknown"),
        )

    def __repr__(self) -> str:
        return (
            f"FragmentExtractedEvent(id={self.event_id[:8]}, "
            f"source={self.source_file}, "
            f"entities={len(self.entities)}, "
            f"relations={len(self.relations)}, "
            f"domain={self.domain})"
        )


class ExtractionFailedEvent(DomainEvent):
    """Event emitted when extraction fails."""

    def __init__(
        self,
        source_file: str,
        error: str,
        extractor_name: str = "unknown",
        run_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.source_file = source_file
        self.error = error
        self.extractor_name = extractor_name
        self.run_id = run_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at,
            "source_file": self.source_file,
            "error": self.error,
            "extractor_name": self.extractor_name,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractionFailedEvent":
        return cls(
            source_file=data["source_file"],
            error=data["error"],
            extractor_name=data.get("extractor_name", "unknown"),
            run_id=data.get("run_id"),
        )

    def __repr__(self) -> str:
        return (
            f"ExtractionFailedEvent(id={self.event_id[:8]}, "
            f"source={self.source_file}, "
            f"error={self.error[:50]}...)"
        )


class RIGORRefinementCompletedEvent(DomainEvent):
    """Event emitted when RIGOR refinement completes."""

    def __init__(
        self,
        source_file: str,
        original_fragment: dict[str, Any],
        refined_fragment: dict[str, Any],
        iterations: int,
        converged: bool,
        run_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.source_file = source_file
        self.original_fragment = original_fragment
        self.refined_fragment = refined_fragment
        self.iterations = iterations
        self.converged = converged
        self.run_id = run_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at,
            "source_file": self.source_file,
            "original_fragment": self.original_fragment,
            "refined_fragment": self.refined_fragment,
            "iterations": self.iterations,
            "converged": self.converged,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RIGORRefinementCompletedEvent":
        return cls(
            source_file=data["source_file"],
            original_fragment=data["original_fragment"],
            refined_fragment=data["refined_fragment"],
            iterations=data["iterations"],
            converged=data["converged"],
            run_id=data.get("run_id"),
        )

    def __repr__(self) -> str:
        return (
            f"RIGORRefinementCompletedEvent(id={self.event_id[:8]}, "
            f"source={self.source_file}, "
            f"iterations={self.iterations}, "
            f"converged={self.converged})"
        )


class FragmentValidatedEvent(DomainEvent):
    """Event emitted when a fragment is validated against the schema."""

    def __init__(
        self,
        source_file: str,
        run_id: Optional[str] = None,
        valid_entities: List[Dict[str, Any]] = None,
        invalid_entities: List[Dict[str, Any]] = None,
        valid_relations: List[Dict[str, Any]] = None,
        invalid_relations: List[Dict[str, Any]] = None,
        validation_errors: List[str] = None,
        passed: bool = True,
    ) -> None:
        super().__init__()
        self.source_file = source_file
        self.run_id = run_id
        self.valid_entities = valid_entities or []
        self.invalid_entities = invalid_entities or []
        self.valid_relations = valid_relations or []
        self.invalid_relations = invalid_relations or []
        self.validation_errors = validation_errors or []
        self.passed = passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at,
            "source_file": self.source_file,
            "run_id": self.run_id,
            "valid_entities": self.valid_entities,
            "invalid_entities": self.invalid_entities,
            "valid_relations": self.valid_relations,
            "invalid_relations": self.invalid_relations,
            "validation_errors": self.validation_errors,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FragmentValidatedEvent":
        return cls(
            source_file=data["source_file"],
            run_id=data.get("run_id"),
            valid_entities=data.get("valid_entities", []),
            invalid_entities=data.get("invalid_entities", []),
            valid_relations=data.get("valid_relations", []),
            invalid_relations=data.get("invalid_relations", []),
            validation_errors=data.get("validation_errors", []),
            passed=data.get("passed", True),
        )

    def __repr__(self) -> str:
        return (
            f"FragmentValidatedEvent(id={self.event_id[:8]}, "
            f"source={self.source_file}, "
            f"valid={len(self.valid_entities)}, "
            f"invalid={len(self.invalid_entities)}, "
            f"passed={self.passed})"
        )


class FragmentPersistedEvent(DomainEvent):
    """Event emitted when a fragment is persisted to storage."""

    def __init__(
        self,
        source_file: str,
        storage_path: str,
        run_id: Optional[str] = None,
        entity_count: int = 0,
        relation_count: int = 0,
    ) -> None:
        super().__init__()
        self.source_file = source_file
        self.storage_path = storage_path
        self.run_id = run_id
        self.entity_count = entity_count
        self.relation_count = relation_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at,
            "source_file": self.source_file,
            "storage_path": self.storage_path,
            "run_id": self.run_id,
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FragmentPersistedEvent":
        return cls(
            source_file=data["source_file"],
            storage_path=data["storage_path"],
            run_id=data.get("run_id"),
            entity_count=data.get("entity_count", 0),
            relation_count=data.get("relation_count", 0),
        )

    def __repr__(self) -> str:
        return (
            f"FragmentPersistedEvent(id={self.event_id[:8]}, "
            f"source={self.source_file}, "
            f"path={self.storage_path})"
        )
