"""Domain Event Base Classes for G26-ARCH-01.

G26-ARCH-01: Introduce typed domain events for type-safe Agent communication.
This module provides the base event infrastructure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from uuid import uuid4


class DomainEvent(ABC):
    """Base class for all domain events.

    G26-ARCH-01: Provides type-safe event structure for Agent communication.
    """

    def __init__(self) -> None:
        self.event_id: str = str(uuid4())
        self.occurred_at: str = datetime.now().isoformat()
        self.event_type: str = self.__class__.__name__

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainEvent:
        """Create event from dictionary."""
        pass

    def __repr__(self) -> str:
        return f"{self.event_type}(id={self.event_id[:8]}, at={self.occurred_at})"
