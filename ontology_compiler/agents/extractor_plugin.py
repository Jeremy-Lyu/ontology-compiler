"""Ontology Extractor Plugin System.

Provides a hot-pluggable interface for ontology extractors.
Allows dynamic registration and hot-swapping of extractor implementations.

P0-2 (2026-06-11): Standardized Fragment output - all plugins must return
standard Fragment objects instead of dictionaries.

[Phase 1 - ADAPTER 2026-06-17]: Integrated Fragment -> Canonical adapter
for unified consumption layer.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

# Import standard Fragment model for P0-2 standardization
from src.domain.models import Fragment, FragmentNode, FragmentInstance, FragmentRelationship, CanonicalResult

# Import adapter for unified consumption
from src.domain.models.adapters.fragment_to_canonical import (
    FragmentToCanonicalAdapter,
    fragment_to_canonical,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractorMetadata:
    """Metadata for an extractor plugin."""
    name: str
    version: str
    description: str
    author: str = "Unknown"
    supported_formats: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class ExtractorResult:
    """Standardized result from extractor plugins.

    P0-2 (2026-06-11): Now returns standard Fragment objects instead of dictionaries.
    
    [Phase 1 - ADAPTER 2026-06-17]: Added to_canonical() method for unified consumption.
    
    Attributes:
        success: Whether extraction succeeded
        fragment: Standard Fragment object containing extracted entities
        error_message: Error message if failed
        execution_time_ms: Execution time in milliseconds
        extractor_name: Name of the extractor used
        timestamp: Timestamp of extraction
        run_id: Run identifier for tracing
    """

    success: bool
    fragment: Fragment = field(default_factory=Fragment)
    error_message: str = ""
    execution_time_ms: float = 0.0
    extractor_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    run_id: str = ""

    @classmethod
    def from_fragment(cls, fragment: Fragment, success: bool = True) -> "ExtractorResult":
        """Create ExtractorResult from a standard Fragment object.

        Args:
            fragment: Standard Fragment instance
            success: Success status

        Returns:
            ExtractorResult instance
        """
        return cls(
            success=success,
            fragment=fragment,
            extractor_name=fragment.metadata.get("extractor_name", ""),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], success: bool = True) -> "ExtractorResult":
        """Create ExtractorResult from legacy dictionary format.

        This method is provided for backward compatibility with older plugins
        that return dictionary-based results.

        Args:
            data: Dictionary containing nodes, edges/instances/relationships
            success: Success status

        Returns:
            ExtractorResult with standardized Fragment
        """
        fragment = Fragment.from_dict(data)
        return cls(
            success=success,
            fragment=fragment,
            extractor_name=data.get("metadata", {}).get("extractor_name", ""),
        )

    def to_canonical(self, prompt_context: Optional[Dict[str, Any]] = None) -> CanonicalResult:
        """Convert ExtractorResult to CanonicalResult for unified consumption.

        [Phase 1 - ADAPTER 2026-06-17]: Provides canonical output for RAG, reporting, and QA.
        
        Args:
            prompt_context: Optional prompt context for reuse in subsequent extractions

        Returns:
            CanonicalResult containing entities, relations, evidence, and summaries
        """
        if not self.success:
            return CanonicalResult(
                run_id=self.run_id,
                metadata={
                    "error": self.error_message,
                    "extractor_name": self.extractor_name,
                },
            )
        
        return FragmentToCanonicalAdapter.fragment_to_result(
            self.fragment,
            run_id=self.run_id,
            prompt_context=prompt_context,
        )


class ExtractorPlugin(ABC):
    """Abstract base class for ontology extractor plugins.

    All extractors must implement this interface to be hot-pluggable.

    P0-2 (2026-06-11): All plugins must return standard ExtractorResult
    containing standard Fragment objects.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the extractor plugin.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"ExtractorPlugin.{self.__class__.__name__}")

    @abstractmethod
    def get_metadata(self) -> ExtractorMetadata:
        """Return metadata about this extractor.

        Returns:
            ExtractorMetadata with plugin information
        """
        pass

    @abstractmethod
    def can_extract(self, source: str | Path, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if this extractor can handle the given source.

        Args:
            source: Source file path or content
            context: Optional context information

        Returns:
            True if this extractor can handle the source
        """
        pass

    @abstractmethod
    def extract(
        self,
        source: str | Path,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExtractorResult:
        """Extract ontology from the source.

        P0-2 (2026-06-11): Must return ExtractorResult containing standard Fragment.
        
        Args:
            source: Source file path or content
            context: Optional context information (domain, run_id, source_file, etc.)

        Returns:
            ExtractorResult containing standard Fragment with nodes, instances, relationships
        """
        pass

    def validate_config(self) -> bool:
        """Validate the extractor configuration.

        Returns:
            True if configuration is valid
        """
        return True

    def _create_fragment(self, nodes: List[Dict], instances: List[Dict], 
                         relationships: List[Dict], context: Dict[str, Any]) -> Fragment:
        """Helper method to create a standard Fragment from extracted data.

        P0-2 (2026-06-11): Convenience method for plugins to create standardized Fragment.

        Args:
            nodes: List of node dictionaries
            instances: List of instance dictionaries  
            relationships: List of relationship dictionaries
            context: Context containing source_file, domain, etc.

        Returns:
            Standard Fragment instance
        """
        fragment_nodes = [FragmentNode.from_dict(n) for n in nodes]
        fragment_instances = [FragmentInstance.from_dict(i) for i in instances]
        fragment_relationships = [FragmentRelationship.from_dict(r) for r in relationships]

        return Fragment(
            source_file=context.get("source_file", ""),
            domain=context.get("domain", "SUPPLY_CHAIN"),
            nodes=fragment_nodes,
            instances=fragment_instances,
            relationships=fragment_relationships,
            metadata={
                "extractor_name": self.get_metadata().name,
                "source_file": context.get("source_file", ""),
                "domain": context.get("domain", "SUPPLY_CHAIN"),
            },
        )


class ExtractorPluginRegistry:
    """Registry for managing hot-pluggable extractor plugins."""

    def __init__(self):
        self._plugins: Dict[str, Type[ExtractorPlugin]] = {}
        self._instances: Dict[str, ExtractorPlugin] = {}
        self.logger = logging.getLogger("ExtractorPluginRegistry")

    def register(self, plugin_class: Type[ExtractorPlugin], name: Optional[str] = None) -> None:
        """Register an extractor plugin class.

        Args:
            plugin_class: Plugin class to register
            name: Optional custom name (defaults to metadata.name)
        """
        # Create a temporary instance to get metadata
        temp_instance = plugin_class()
        metadata = temp_instance.get_metadata()
        plugin_name = name or metadata.name

        if plugin_name in self._plugins:
            self.logger.warning(f"Overwriting existing plugin: {plugin_name}")

        self._plugins[plugin_name] = plugin_class
        self.logger.info(f"Registered extractor plugin: {plugin_name} v{metadata.version}")

    def unregister(self, name: str) -> bool:
        """Unregister an extractor plugin.

        Args:
            name: Plugin name to unregister

        Returns:
            True if plugin was unregistered
        """
        if name in self._plugins:
            del self._plugins[name]
            if name in self._instances:
                del self._instances[name]
            self.logger.info(f"Unregistered extractor plugin: {name}")
            return True
        return False

    def get_plugin(self, name: str, config: Optional[Dict[str, Any]] = None) -> Optional[ExtractorPlugin]:
        """Get a plugin instance by name.

        Args:
            name: Plugin name
            config: Optional configuration for the instance

        Returns:
            Plugin instance or None if not found
        """
        if name not in self._plugins:
            self.logger.error(f"Plugin not found: {name}")
            return None

        # Return cached instance if no config provided
        if config is None and name in self._instances:
            return self._instances[name]

        # Create new instance
        plugin_class = self._plugins[name]
        instance = plugin_class(config)

        # Cache if no custom config
        if config is None:
            self._instances[name] = instance

        return instance

    def list_plugins(self) -> List[ExtractorMetadata]:
        """List all registered plugins.

        Returns:
            List of metadata for all registered plugins
        """
        metadata_list = []
        for plugin_class in self._plugins.values():
            temp_instance = plugin_class()
            metadata_list.append(temp_instance.get_metadata())
        return metadata_list

    def find_compatible_plugins(
        self,
        source: str | Path,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Find plugins that can handle the given source.

        Args:
            source: Source file path or content
            context: Optional context information

        Returns:
            List of plugin names that can handle the source
        """
        compatible = []
        for name, plugin_class in self._plugins.items():
            instance = plugin_class()
            if instance.can_extract(source, context):
                compatible.append(name)
        return compatible

    def hot_swap(self, name: str, new_plugin_class: Type[ExtractorPlugin]) -> bool:
        """Hot-swap a plugin implementation at runtime.

        Args:
            name: Plugin name to swap
            new_plugin_class: New plugin class to use

        Returns:
            True if swap was successful
        """
        if name not in self._plugins:
            self.logger.error(f"Cannot swap non-existent plugin: {name}")
            return False

        # Unregister old instance
        if name in self._instances:
            del self._instances[name]

        # Register new plugin
        self.register(new_plugin_class, name)
        self.logger.info(f"Hot-swapped plugin: {name}")
        return True


# Global registry instance
_global_registry: Optional[ExtractorPluginRegistry] = None


def get_global_registry() -> ExtractorPluginRegistry:
    """Get the global plugin registry instance.

    Returns:
        Global ExtractorPluginRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ExtractorPluginRegistry()
    return _global_registry


def register_plugin(plugin_class: Type[ExtractorPlugin], name: Optional[str] = None) -> None:
    """Register a plugin to the global registry.

    Args:
        plugin_class: Plugin class to register
        name: Optional custom name
    """
    get_global_registry().register(plugin_class, name)


def get_plugin(name: str, config: Optional[Dict[str, Any]] = None) -> Optional[ExtractorPlugin]:
    """Get a plugin from the global registry.

    Args:
        name: Plugin name
        config: Optional configuration

    Returns:
        Plugin instance or None
    """
    return get_global_registry().get_plugin(name, config)
