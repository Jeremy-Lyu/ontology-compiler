"""Event Consumer Validator for G26-ARCH-01.

[MONITOR-3] Validates that FragmentExtractedEvent is properly consumed by downstream processors.

Features:
- Registry for event consumers
- Validation of consumer registration
- Event tracking to verify consumption
- Health check for consumer pipeline
- Test utilities for consumer verification
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Type

from .base_event import DomainEvent
from .extraction_events import (
    FragmentExtractedEvent,
    FragmentValidatedEvent,
    FragmentPersistedEvent,
    RIGORRefinementCompletedEvent,
    ExtractionFailedEvent,
)

logger = logging.getLogger(__name__)


class EventConsumerRegistry:
    """
    [MONITOR-3] Registry for event consumers with validation capabilities.

    Tracks:
    - Registered consumers per event type
    - Event dispatch status
    - Consumer health status
    - Consumption metrics
    """

    _instance: Optional["EventConsumerRegistry"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._consumers: Dict[Type[DomainEvent], List[ConsumerRegistration]] = defaultdict(list)
            cls._instance._event_tracking: Dict[str, EventTrackingRecord] = {}
            cls._instance._consumer_stats: Dict[str, ConsumerStats] = {}
        return cls._instance

    def register_consumer(
        self,
        event_type: Type[DomainEvent],
        handler: Callable[[DomainEvent], Any],
        consumer_name: str,
        required: bool = False,
        timeout_seconds: float = 30.0,
    ) -> None:
        """
        Register a consumer for an event type.

        Args:
            event_type: The event class to consume
            handler: The handler function
            consumer_name: Unique name for this consumer
            required: Whether this consumer is required for the pipeline
            timeout_seconds: Maximum time allowed for consumption
        """
        registration = ConsumerRegistration(
            consumer_name=consumer_name,
            handler=handler,
            required=required,
            timeout_seconds=timeout_seconds,
            registered_at=datetime.now(),
        )

        existing = [c for c in self._consumers[event_type] if c.consumer_name == consumer_name]
        if existing:
            self._consumers[event_type].remove(existing[0])
            logger.debug(f"Replaced existing consumer: {consumer_name} for {event_type.__name__}")

        self._consumers[event_type].append(registration)
        self._consumer_stats[consumer_name] = ConsumerStats(consumer_name=consumer_name)

        logger.info(
            f"Registered consumer '{consumer_name}' for {event_type.__name__} "
            f"(required={required}, timeout={timeout_seconds}s)"
        )

    def unregister_consumer(self, event_type: Type[DomainEvent], consumer_name: str) -> None:
        """Unregister a consumer."""
        self._consumers[event_type] = [
            c for c in self._consumers[event_type] if c.consumer_name != consumer_name
        ]
        if consumer_name in self._consumer_stats:
            del self._consumer_stats[consumer_name]
        logger.info(f"Unregistered consumer '{consumer_name}' for {event_type.__name__}")

    def get_consumers(self, event_type: Type[DomainEvent]) -> List[ConsumerRegistration]:
        """Get all registered consumers for an event type."""
        return self._consumers.get(event_type, [])

    def publish(self, event: DomainEvent) -> PublishResult:
        """
        Publish an event to all registered consumers.

        Args:
            event: The event to publish

        Returns:
            PublishResult with consumption status
        """
        event_type = type(event)
        consumers = self._consumers.get(event_type, [])

        if not consumers:
            logger.warning(f"No consumers registered for {event_type.__name__}")
            return PublishResult(
                event_id=event.event_id,
                event_type=event_type.__name__,
                total_consumers=0,
                successful=0,
                failed=0,
                timed_out=0,
                results=[],
            )

        tracking_record = EventTrackingRecord(
            event_id=event.event_id,
            event_type=event_type.__name__,
            published_at=datetime.now(),
        )
        self._event_tracking[event.event_id] = tracking_record

        logger.info(f"Publishing {event_type.__name__} (id={event.event_id[:8]}) to {len(consumers)} consumers")

        results = []
        for consumer in consumers:
            start_time = time.time()
            try:
                result = consumer.handler(event)
                elapsed = time.time() - start_time

                if elapsed > consumer.timeout_seconds:
                    status = "timed_out"
                    logger.warning(
                        f"Consumer '{consumer.consumer_name}' timed out after {elapsed:.2f}s "
                        f"(timeout={consumer.timeout_seconds}s)"
                    )
                else:
                    status = "success"
                    logger.debug(f"Consumer '{consumer.consumer_name}' processed event in {elapsed:.2f}s")

                results.append(ConsumerResult(
                    consumer_name=consumer.consumer_name,
                    status=status,
                    elapsed_seconds=elapsed,
                    result=result,
                ))

                self._consumer_stats[consumer.consumer_name].record(status, elapsed)

            except Exception as e:
                elapsed = time.time() - start_time
                status = "failed"
                logger.error(f"Consumer '{consumer.consumer_name}' failed: {e}")

                results.append(ConsumerResult(
                    consumer_name=consumer.consumer_name,
                    status=status,
                    elapsed_seconds=elapsed,
                    error=str(e),
                ))

                self._consumer_stats[consumer.consumer_name].record(status, elapsed)

        tracking_record.consumed_at = datetime.now()
        tracking_record.results = results

        successful = sum(1 for r in results if r.status == "success")
        failed = sum(1 for r in results if r.status == "failed")
        timed_out = sum(1 for r in results if r.status == "timed_out")

        return PublishResult(
            event_id=event.event_id,
            event_type=event_type.__name__,
            total_consumers=len(consumers),
            successful=successful,
            failed=failed,
            timed_out=timed_out,
            results=results,
        )

    def validate_consumers(self, event_type: Type[DomainEvent]) -> EventConsumerValidationResult:
        """
        Validate that all required consumers are registered and healthy.

        Args:
            event_type: The event type to validate

        Returns:
            EventConsumerValidationResult with health status
        """
        consumers = self._consumers.get(event_type, [])

        if not consumers:
            return EventConsumerValidationResult(
                event_type=event_type.__name__,
                valid=False,
                required_missing=[],
                unhealthy_consumers=[],
                message=f"No consumers registered for {event_type.__name__}",
            )

        required_missing = []
        unhealthy_consumers = []

        for consumer in consumers:
            if consumer.required:
                stats = self._consumer_stats.get(consumer.consumer_name)
                if stats and stats.failure_rate > 0.2:
                    unhealthy_consumers.append(consumer.consumer_name)

        valid = not required_missing and not unhealthy_consumers

        message = f"Validated {len(consumers)} consumers for {event_type.__name__}"
        if required_missing:
            message += f" - Missing required: {', '.join(required_missing)}"
        if unhealthy_consumers:
            message += f" - Unhealthy: {', '.join(unhealthy_consumers)}"

        return EventConsumerValidationResult(
            event_type=event_type.__name__,
            valid=valid,
            required_missing=required_missing,
            unhealthy_consumers=unhealthy_consumers,
            message=message,
        )

    def get_event_status(self, event_id: str) -> Optional[EventTrackingRecord]:
        """Get tracking record for a specific event."""
        return self._event_tracking.get(event_id)

    def get_consumer_stats(self, consumer_name: str) -> Optional[ConsumerStats]:
        """Get statistics for a specific consumer."""
        return self._consumer_stats.get(consumer_name)

    def get_all_stats(self) -> Dict[str, ConsumerStats]:
        """Get statistics for all registered consumers."""
        return dict(self._consumer_stats)

    def get_health_report(self) -> HealthReport:
        """Get comprehensive health report for all consumers."""
        all_validation_results = []
        event_types_with_consumers = set(self._consumers.keys())

        for event_type in event_types_with_consumers:
            validation = self.validate_consumers(event_type)
            all_validation_results.append(validation)

        all_valid = all(v.valid for v in all_validation_results)
        total_consumers = sum(len(self._consumers.get(et, [])) for et in event_types_with_consumers)

        return HealthReport(
            timestamp=datetime.now(),
            all_valid=all_valid,
            total_event_types=len(event_types_with_consumers),
            total_consumers=total_consumers,
            validations=all_validation_results,
            consumer_stats=self._consumer_stats,
        )

    def reset(self):
        """Reset all tracking data (for testing)."""
        self._event_tracking.clear()
        self._consumer_stats.clear()
        logger.info("Event consumer registry reset")


@dataclass
class ConsumerRegistration:
    consumer_name: str
    handler: Callable[[DomainEvent], Any]
    required: bool
    timeout_seconds: float
    registered_at: datetime


@dataclass
class ConsumerResult:
    consumer_name: str
    status: str
    elapsed_seconds: float
    result: Any = None
    error: Optional[str] = None


@dataclass
class PublishResult:
    event_id: str
    event_type: str
    total_consumers: int
    successful: int
    failed: int
    timed_out: int
    results: List[ConsumerResult]


@dataclass
class EventConsumerValidationResult:
    event_type: str
    valid: bool
    required_missing: List[str]
    unhealthy_consumers: List[str]
    message: str


@dataclass
class EventTrackingRecord:
    event_id: str
    event_type: str
    published_at: datetime
    consumed_at: Optional[datetime] = None
    results: List[ConsumerResult] = field(default_factory=list)


@dataclass
class ConsumerStats:
    consumer_name: str
    total_events: int = 0
    successful_events: int = 0
    failed_events: int = 0
    timed_out_events: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0

    @property
    def failure_rate(self) -> float:
        if self.total_events == 0:
            return 0.0
        return (self.failed_events + self.timed_out_events) / self.total_events

    @property
    def avg_time(self) -> float:
        if self.total_events == 0:
            return 0.0
        return self.total_time / self.total_events

    def record(self, status: str, elapsed: float):
        self.total_events += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)

        if status == "success":
            self.successful_events += 1
        elif status == "failed":
            self.failed_events += 1
        elif status == "timed_out":
            self.timed_out_events += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consumer_name": self.consumer_name,
            "total_events": self.total_events,
            "successful_events": self.successful_events,
            "failed_events": self.failed_events,
            "timed_out_events": self.timed_out_events,
            "failure_rate": round(self.failure_rate, 4),
            "avg_time_ms": round(self.avg_time * 1000, 2),
            "min_time_ms": round(self.min_time * 1000, 2),
            "max_time_ms": round(self.max_time * 1000, 2),
        }


@dataclass
class HealthReport:
    timestamp: datetime
    all_valid: bool
    total_event_types: int
    total_consumers: int
    validations: List[EventConsumerValidationResult]
    consumer_stats: Dict[str, ConsumerStats]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "all_valid": self.all_valid,
            "total_event_types": self.total_event_types,
            "total_consumers": self.total_consumers,
            "validations": [
                {
                    "event_type": v.event_type,
                    "valid": v.valid,
                    "message": v.message,
                }
                for v in self.validations
            ],
            "consumer_stats": {
                name: stats.to_dict() for name, stats in self.consumer_stats.items()
            },
        }


def validate_extraction_consumer_pipeline() -> Dict[str, Any]:
    """
    [MONITOR-3] Validate the complete extraction consumer pipeline.

    Checks that FragmentExtractedEvent and related events have proper consumer coverage.

    Returns:
        Validation report with recommendations
    """
    registry = EventConsumerRegistry()

    health_report = registry.get_health_report()

    fragment_extracted_consumers = registry.get_consumers(FragmentExtractedEvent)
    fragment_validated_consumers = registry.get_consumers(FragmentValidatedEvent)
    fragment_persisted_consumers = registry.get_consumers(FragmentPersistedEvent)
    rigor_completed_consumers = registry.get_consumers(RIGORRefinementCompletedEvent)
    extraction_failed_consumers = registry.get_consumers(ExtractionFailedEvent)

    recommendations = []

    if not fragment_extracted_consumers:
        recommendations.append({
            "priority": "high",
            "event_type": "FragmentExtractedEvent",
            "issue": "No consumers registered",
            "suggestion": "Register consumers for RAG indexing, validation, and storage",
        })

    if not fragment_validated_consumers:
        recommendations.append({
            "priority": "medium",
            "event_type": "FragmentValidatedEvent",
            "issue": "No consumers registered",
            "suggestion": "Register quality gate consumer to enforce validation",
        })

    if not fragment_persisted_consumers:
        recommendations.append({
            "priority": "high",
            "event_type": "FragmentPersistedEvent",
            "issue": "No consumers registered",
            "suggestion": "Register cross-file association consumer",
        })

    return {
        "status": "healthy" if health_report.all_valid else "degraded",
        "health_report": health_report.to_dict(),
        "consumer_counts": {
            "FragmentExtractedEvent": len(fragment_extracted_consumers),
            "FragmentValidatedEvent": len(fragment_validated_consumers),
            "FragmentPersistedEvent": len(fragment_persisted_consumers),
            "RIGORRefinementCompletedEvent": len(rigor_completed_consumers),
            "ExtractionFailedEvent": len(extraction_failed_consumers),
        },
        "recommendations": recommendations,
    }


def register_default_consumers() -> None:
    """
    Register default consumers for extraction events.

    These consumers provide basic validation and logging for event flow verification.
    """
    registry = EventConsumerRegistry()

    def log_extracted_event(event: FragmentExtractedEvent):
        logger.info(
            f"[CONSUMER] FragmentExtractedEvent received: "
            f"source={event.source_file}, entities={len(event.entities)}, "
            f"relations={len(event.relations)}, quality={event.quality_score}"
        )

    def log_validated_event(event: FragmentValidatedEvent):
        logger.info(
            f"[CONSUMER] FragmentValidatedEvent received: "
            f"source={event.source_file}, valid={len(event.valid_entities)}, "
            f"invalid={len(event.invalid_entities)}, passed={event.passed}"
        )

    def log_persisted_event(event: FragmentPersistedEvent):
        logger.info(
            f"[CONSUMER] FragmentPersistedEvent received: "
            f"source={event.source_file}, path={event.storage_path}, "
            f"entities={event.entity_count}, relations={event.relation_count}"
        )

    def log_rigor_event(event: RIGORRefinementCompletedEvent):
        logger.info(
            f"[CONSUMER] RIGORRefinementCompletedEvent received: "
            f"source={event.source_file}, iterations={event.iterations}, "
            f"converged={event.converged}"
        )

    def log_failed_event(event: ExtractionFailedEvent):
        logger.error(
            f"[CONSUMER] ExtractionFailedEvent received: "
            f"source={event.source_file}, error={event.error}"
        )

    registry.register_consumer(
        FragmentExtractedEvent,
        log_extracted_event,
        consumer_name="Logger.Extracted",
        required=False,
    )

    registry.register_consumer(
        FragmentValidatedEvent,
        log_validated_event,
        consumer_name="Logger.Validated",
        required=False,
    )

    registry.register_consumer(
        FragmentPersistedEvent,
        log_persisted_event,
        consumer_name="Logger.Persisted",
        required=False,
    )

    registry.register_consumer(
        RIGORRefinementCompletedEvent,
        log_rigor_event,
        consumer_name="Logger.RIGOR",
        required=False,
    )

    registry.register_consumer(
        ExtractionFailedEvent,
        log_failed_event,
        consumer_name="Logger.Failed",
        required=False,
    )

    logger.info("Default event consumers registered")


register_default_consumers()
