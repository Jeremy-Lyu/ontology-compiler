"""Adapter Plugin Service with SPI (Service Provider Interface) mechanism.

C-1 Implementation 2026-05-09:
Provides hot-pluggable adapter management for all data sources and outputs.
"""
from __future__ import annotations

import importlib
import inspect
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from src.application.ports.artifact_ports import ArtifactStorePort
from src.application.ports.document_ports import DocumentReaderPort
from src.application.ports.llm_ports import LLMPort
from src.application.ports.rag_ports import RAGFeedbackPort

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class AdapterPluginMetadata:
    """Metadata for an adapter plugin."""
    name: str
    version: str
    port_type: str  # "DocumentReaderPort", "ArtifactStorePort", etc.
    description: str = ""
    author: str = ""
    supported_formats: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    priority: int = 100  # Lower = higher priority
    enabled: bool = True


class AdapterPlugin(ABC):
    """Base class for all adapter plugins."""
    
    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self._initialized = False
    
    @property
    @abstractmethod
    def metadata(self) -> AdapterPluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        pass
    
    def shutdown(self) -> None:
        """Clean up plugin resources."""
        self._initialized = False
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized


class DocumentReaderPlugin(AdapterPlugin):
    """Plugin for document reading adapters."""
    
    @abstractmethod
    def create_reader(self) -> DocumentReaderPort:
        """Create and return a document reader instance."""
        pass


class ArtifactStorePlugin(AdapterPlugin):
    """Plugin for artifact storage adapters."""
    
    @abstractmethod
    def create_store(self) -> ArtifactStorePort:
        """Create and return an artifact store instance."""
        pass


class LLMPlugin(AdapterPlugin):
    """Plugin for LLM adapters."""
    
    @abstractmethod
    def create_client(self) -> LLMPort:
        """Create and return an LLM client instance."""
        pass


class RAGFeedbackPlugin(AdapterPlugin):
    """Plugin for RAG feedback adapters."""
    
    @abstractmethod
    def create_feedback_handler(self) -> RAGFeedbackPort:
        """Create and return a feedback handler instance."""
        pass


class AdapterPluginRegistry:
    """Registry for managing adapter plugins with hot-pluggable support.
    
    C-1 Core: Provides SPI (Service Provider Interface) for adapters.
    """
    
    def __init__(self, config_path: Path | None = None):
        self._plugins: Dict[str, AdapterPlugin] = {}
        self._factories: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._config_path = config_path or Path("config/adapter_plugins.json")
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load plugin configuration from file."""
        if self._config_path.exists():
            try:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
                logger.info(f"Loaded adapter plugin config from {self._config_path}")
            except Exception as e:
                logger.warning(f"Failed to load plugin config: {e}")
                self._config = {"plugins": {}}
        else:
            self._config = {"plugins": {}}
    
    def _save_config(self) -> None:
        """Save plugin configuration to file."""
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save plugin config: {e}")
    
    def register(
        self,
        plugin_class: Type[AdapterPlugin],
        config: Dict[str, Any] | None = None,
    ) -> AdapterPlugin:
        """Register a plugin class.
        
        Args:
            plugin_class: The plugin class to register
            config: Optional configuration for the plugin
            
        Returns:
            The initialized plugin instance
        """
        # Create temporary instance to get metadata
        temp_instance = plugin_class(config)
        metadata = temp_instance.metadata
        
        # Check if plugin is enabled in config
        plugin_config = self._config.get("plugins", {}).get(metadata.name, {})
        if not plugin_config.get("enabled", metadata.enabled):
            logger.info(f"Plugin {metadata.name} is disabled in config")
            temp_instance.shutdown()
            return temp_instance
        
        # Merge configs
        merged_config = {**metadata.config_schema, **(config or {}), **plugin_config.get("settings", {})}
        
        # Create final instance
        instance = plugin_class(merged_config)
        
        # Initialize
        if instance.initialize():
            self._plugins[metadata.name] = instance
            logger.info(f"Registered and initialized plugin: {metadata.name} v{metadata.version}")
            
            # Update config
            if metadata.name not in self._config.get("plugins", {}):
                if "plugins" not in self._config:
                    self._config["plugins"] = {}
                self._config["plugins"][metadata.name] = {
                    "enabled": True,
                    "version": metadata.version,
                    "settings": merged_config,
                }
                self._save_config()
            
            return instance
        else:
            logger.error(f"Failed to initialize plugin: {metadata.name}")
            instance.shutdown()
            raise RuntimeError(f"Plugin initialization failed: {metadata.name}")
    
    def unregister(self, name: str) -> bool:
        """Unregister a plugin.
        
        Args:
            name: Plugin name to unregister
            
        Returns:
            True if unregistered successfully
        """
        if name in self._plugins:
            plugin = self._plugins[name]
            plugin.shutdown()
            del self._plugins[name]
            logger.info(f"Unregistered plugin: {name}")
            
            # Update config
            if name in self._config.get("plugins", {}):
                self._config["plugins"][name]["enabled"] = False
                self._save_config()
            
            return True
        return False
    
    def get_plugin(self, name: str) -> AdapterPlugin | None:
        """Get a registered plugin by name."""
        return self._plugins.get(name)
    
    def get_plugins_by_port(self, port_type: str) -> List[AdapterPlugin]:
        """Get all plugins for a specific port type."""
        return [
            plugin for plugin in self._plugins.values()
            if plugin.metadata.port_type == port_type
        ]
    
    def get_best_plugin(self, port_type: str, format_hint: str | None = None) -> AdapterPlugin | None:
        """Get the best plugin for a port type, optionally filtered by format.
        
        Args:
            port_type: The port interface type (e.g., "DocumentReaderPort")
            format_hint: Optional format to match (e.g., "pdf", "markdown")
            
        Returns:
            The best matching plugin, or None if no match
        """
        candidates = self.get_plugins_by_port(port_type)
        
        # Filter by format if specified
        if format_hint:
            candidates = [
                p for p in candidates
                if format_hint.lower() in [f.lower() for f in p.metadata.supported_formats]
            ]
        
        # Sort by priority
        candidates.sort(key=lambda p: p.metadata.priority)
        
        return candidates[0] if candidates else None
    
    def create_adapter(self, port_type: str, plugin_name: str | None = None, **config) -> Any:
        """Create an adapter instance.
        
        Args:
            port_type: The port type (e.g., "DocumentReaderPort")
            plugin_name: Specific plugin to use, or None for best match
            **config: Configuration for the adapter
            
        Returns:
            The adapter instance
        """
        if plugin_name:
            plugin = self.get_plugin(plugin_name)
            if not plugin:
                raise ValueError(f"Plugin not found: {plugin_name}")
        else:
            plugin = self.get_best_plugin(port_type)
            if not plugin:
                raise ValueError(f"No plugin available for port type: {port_type}")
        
        # Create the appropriate adapter based on plugin type
        if isinstance(plugin, DocumentReaderPlugin):
            return plugin.create_reader()
        elif isinstance(plugin, ArtifactStorePlugin):
            return plugin.create_store()
        elif isinstance(plugin, LLMPlugin):
            return plugin.create_client()
        elif isinstance(plugin, RAGFeedbackPlugin):
            return plugin.create_feedback_handler()
        else:
            raise ValueError(f"Unknown plugin type: {type(plugin)}")
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins with metadata."""
        return [
            {
                "name": p.metadata.name,
                "version": p.metadata.version,
                "port_type": p.metadata.port_type,
                "description": p.metadata.description,
                "enabled": p.metadata.enabled and p.is_initialized,
                "priority": p.metadata.priority,
            }
            for p in self._plugins.values()
        ]
    
    def reload_plugin(self, name: str) -> AdapterPlugin:
        """Hot-reload a plugin.
        
        This is the key feature for C-1: hot-pluggable adapters.
        """
        if name not in self._plugins:
            raise ValueError(f"Plugin not registered: {name}")
        
        old_plugin = self._plugins[name]
        plugin_class = type(old_plugin)
        config = old_plugin.config
        
        # Unregister old
        self.unregister(name)
        
        # Re-register (will reload module if changed)
        module_name = plugin_class.__module__
        try:
            importlib.reload(importlib.import_module(module_name))
            new_plugin_class = getattr(importlib.import_module(module_name), plugin_class.__name__)
            return self.register(new_plugin_class, config)
        except Exception as e:
            logger.error(f"Failed to reload plugin {name}: {e}")
            # Restore old plugin
            self._plugins[name] = old_plugin
            raise
    
    def discover_plugins(self, package_path: str = "src.adapters.plugins") -> List[Type[AdapterPlugin]]:
        """Auto-discover plugins in a package.
        
        Scans the package for classes implementing AdapterPlugin.
        """
        discovered = []
        try:
            package = importlib.import_module(package_path)
            if hasattr(package, "__path__"):
                for _, name, _ in importlib.iter_modules(package.__path__):
                    try:
                        module = importlib.import_module(f"{package_path}.{name}")
                        for attr_name in dir(module):
                            obj = getattr(module, attr_name)
                            if (isinstance(obj, type) and 
                                issubclass(obj, AdapterPlugin) and 
                                obj is not AdapterPlugin and
                                not inspect.isabstract(obj)):
                                discovered.append(obj)
                                logger.info(f"Discovered plugin: {obj.__name__}")
                    except Exception as e:
                        logger.warning(f"Failed to load plugin module {name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to discover plugins: {e}")
        
        return discovered


class AdapterPluginService:
    """High-level service for adapter plugin management.
    
    This is the main C-1 API for managing pluggable adapters.
    """
    
    _instance: Optional[AdapterPluginService] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = None
        return cls._instance
    
    def initialize(self, config_path: Path | None = None) -> None:
        """Initialize the service."""
        self._registry = AdapterPluginRegistry(config_path)
        logger.info("AdapterPluginService initialized")
    
    @property
    def registry(self) -> AdapterPluginRegistry:
        """Get the plugin registry."""
        if self._registry is None:
            self.initialize()
        return self._registry
    
    def register_builtin_adapters(self) -> None:
        """Register all built-in adapter plugins.

        P1-2 Enhancement (2026-05-30):
        - Explicitly registers FileParserReaderPlugin and FsOutputsStorePlugin
        - Enables automatic discovery and registration of AdapterPlugins
        - Integrates with ExtractorPlugin system for unified plugin management
        """
        # Explicitly register built-in plugins
        from src.adapters.plugins.file_parser_reader_plugin import FileParserReaderPlugin
        from src.adapters.plugins.fs_outputs_store_plugin import FsOutputsStorePlugin

        builtin_plugins = [
            FileParserReaderPlugin,
            FsOutputsStorePlugin,
        ]

        for plugin_class in builtin_plugins:
            try:
                self.registry.register(plugin_class)
                logger.info(f"Registered built-in plugin: {plugin_class.__name__}")
            except Exception as e:
                logger.warning(f"Failed to register built-in plugin {plugin_class.__name__}: {e}")

        # Discover additional plugins from the plugins package
        discovered = self.registry.discover_plugins("src.adapters.plugins")

        for plugin_class in discovered:
            try:
                self.registry.register(plugin_class)
            except Exception as e:
                logger.warning(f"Failed to register discovered plugin {plugin_class.__name__}: {e}")

        # Log registration summary
        plugins = self.registry.list_plugins()
        if plugins:
            logger.info(f"Registered {len(plugins)} built-in adapter plugins")
        else:
            logger.debug("No adapter plugins discovered or registered")
    
    def get_document_reader(self, format_hint: str | None = None) -> DocumentReaderPort:
        """Get the best document reader adapter."""
        return self.registry.create_adapter("DocumentReaderPort", format_hint=format_hint)
    
    def get_artifact_store(self, store_type: str | None = None) -> ArtifactStorePort:
        """Get the best artifact store adapter."""
        return self.registry.create_adapter("ArtifactStorePort", format_hint=store_type)
