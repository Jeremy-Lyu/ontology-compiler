"""Core Events Module - Domain events and consumer infrastructure.

This module provides the canonical source for domain events.
Shared between src and ontology_compiler to avoid duplication.

G26-ARCH-01: Type-safe event communication for Agent coordination.
"""

from .base_event import DomainEvent
from .extraction_events import (
    FragmentExtractedEvent,
    ExtractionFailedEvent,
    RIGORRefinementCompletedEvent,
    FragmentValidatedEvent,
    FragmentPersistedEvent,
)
from .event_consumer_validator import (
    EventConsumerRegistry,
    ConsumerRegistration,
    ConsumerResult,
    PublishResult,
    EventConsumerValidationResult,
    EventTrackingRecord,
    ConsumerStats,
    HealthReport,
    validate_extraction_consumer_pipeline,
    register_default_consumers,
)

__all__ = [
    "DomainEvent",
    "FragmentExtractedEvent",
    "ExtractionFailedEvent",
    "RIGORRefinementCompletedEvent",
    "FragmentValidatedEvent",
    "FragmentPersistedEvent",
    "EventConsumerRegistry",
    "ConsumerRegistration",
    "ConsumerResult",
    "PublishResult",
    "EventConsumerValidationResult",
    "EventTrackingRecord",
    "ConsumerStats",
    "HealthReport",
    "validate_extraction_consumer_pipeline",
    "register_default_consumers",
]
