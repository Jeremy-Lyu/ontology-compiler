"""Skill telemetry service."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class SkillInvocationRecord:
    """Record of a skill invocation."""
    skill_id: str
    start_time: datetime
    end_time: datetime | None = None
    status: str = "running"
    input_size: int = 0
    output_size: int = 0
    error_message: str | None = None

    @property
    def duration_ms(self) -> float:
        """Calculate duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() * 1000


class SkillTelemetry:
    """Skill telemetry service for tracking skill invocations."""
    
    def __init__(self):
        self._records: list[SkillInvocationRecord] = []
    
    def record_invocation(
        self,
        skill_id: str,
        start_time: datetime | None = None,
    ) -> SkillInvocationRecord:
        """Record a skill invocation start."""
        record = SkillInvocationRecord(
            skill_id=skill_id,
            start_time=start_time or datetime.now(),
        )
        self._records.append(record)
        return record
    
    def complete_invocation(
        self,
        record: SkillInvocationRecord,
        output_size: int = 0,
    ) -> SkillInvocationRecord:
        """Complete a skill invocation."""
        record.end_time = datetime.now()
        record.status = "completed"
        record.output_size = output_size
        return record
    
    def fail_invocation(
        self,
        record: SkillInvocationRecord,
        error_message: str,
    ) -> SkillInvocationRecord:
        """Mark a skill invocation as failed."""
        record.end_time = datetime.now()
        record.status = "failed"
        record.error_message = error_message
        return record
    
    def get_records(self, skill_id: str | None = None) -> list[SkillInvocationRecord]:
        """Get invocation records, optionally filtered by skill ID."""
        if skill_id is None:
            return self._records
        return [r for r in self._records if r.skill_id == skill_id]