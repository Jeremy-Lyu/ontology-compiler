"""Filesystem Outputs Artifact Store Plugin (P1-2 Enhancement).

Provides a concrete ArtifactStorePlugin implementation that wraps the
OutputsArtifactStore adapter for hot-pluggable artifact storage.
"""
from __future__ import annotations

from typing import Any, Dict

from src.application.services.adapter_plugin_service import (
    AdapterPluginMetadata,
    ArtifactStorePlugin,
)
from src.adapters.outbound.fs_outputs_artifact_store import OutputsArtifactStore


class FsOutputsStorePlugin(ArtifactStorePlugin):
    """Plugin wrapper for OutputsArtifactStore adapter.

    This plugin bridges the existing OutputsArtifactStore adapter with
    the AdapterPlugin system, enabling hot-pluggable artifact storage.

    Supported store types: filesystem
    Capabilities: run_artifact_storage, pipeline_summary_storage, legacy_compatibility

    P1-2 Enhancement (2026-05-30):
    - Created to fulfill the ArtifactStorePlugin interface
    - Wraps OutputsArtifactStore for backward compatibility
    - Enables runtime plugin registration and hot-swapping
    """

    PLUGIN_NAME = "fs_outputs_store"
    PLUGIN_VERSION = "1.0.0"
    SUPPORTED_STORE_TYPES = ["filesystem"]
    PORT_TYPE = "ArtifactStorePort"

    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize the plugin.

        Args:
            config: Optional configuration dictionary with keys:
                - outputs_root: str (default: "outputs/ontology/runs")
                - pipeline_root: str (default: "outputs/ontology/pipeline")
                - legacy_root: str (default: "outputs/ontology")
        """
        super().__init__(config)
        outputs_root = config.get("outputs_root", "outputs/ontology/runs") if config else "outputs/ontology/runs"
        pipeline_root = config.get("pipeline_root", "outputs/ontology/pipeline") if config else "outputs/ontology/pipeline"
        legacy_root = config.get("legacy_root", "outputs/ontology") if config else "outputs/ontology"
        self._store = OutputsArtifactStore(
            outputs_root=outputs_root,
            pipeline_root=pipeline_root,
            legacy_root=legacy_root,
        )

    @property
    def metadata(self) -> AdapterPluginMetadata:
        """Return plugin metadata."""
        return AdapterPluginMetadata(
            name=self.PLUGIN_NAME,
            version=self.PLUGIN_VERSION,
            port_type=self.PORT_TYPE,
            description="Filesystem artifact store with pipeline summary support",
            author="Ontology Compiler Team",
            supported_formats=list(self.SUPPORTED_STORE_TYPES),
            config_schema={
                "outputs_root": {"type": "string", "default": "outputs/ontology/runs"},
                "pipeline_root": {"type": "string", "default": "outputs/ontology/pipeline"},
                "legacy_root": {"type": "string", "default": "outputs/ontology"},
            },
            priority=100,
            enabled=True,
        )

    def initialize(self) -> bool:
        """Initialize the plugin.

        Returns:
            True if initialization successful
        """
        self._initialized = True
        return True

    def create_store(self):
        """Create and return an artifact store instance.

        Returns:
            OutputsArtifactStore instance
        """
        return self._store
