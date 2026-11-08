"""Centralized runtime path configuration.

This module provides a single source of truth for all runtime paths,
replacing hardcoded magic strings throughout the codebase.
"""

from pathlib import Path
from typing import NamedTuple


class RuntimePaths(NamedTuple):
    """Runtime paths configuration."""
    ontology_output_dir: Path
    runs_dir: Path
    sessions_dir: Path
    schemas_dir: Path
    config_dir: Path


def get_runtime_paths(project_root: Path | None = None) -> RuntimePaths:
    """Get runtime paths based on project root.
    
    Args:
        project_root: The project root directory. If None, uses current directory.
        
    Returns:
        RuntimePaths instance with all configured paths.
    """
    if project_root is None:
        project_root = Path.cwd()
    
    return RuntimePaths(
        ontology_output_dir=project_root / "outputs" / "runs" / "ontology",
        runs_dir=project_root / "outputs" / "runs",
        sessions_dir=project_root / "outputs" / "runs" / "sessions",
        schemas_dir=project_root / "schemas",
        config_dir=project_root / "config",
    )


# Global instance for convenience (can be overridden by tests)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_runtime_paths = get_runtime_paths(PROJECT_ROOT)


def get_project_root() -> Path:
    """Project root directory (P0-6-4: single canonical definition; the
    latent duplicate path in contract_registry falls back here)."""
    return PROJECT_ROOT


def get_project_root() -> Path:
    """Project root directory (canonical; P0-6-4 unified the duplicate in
    contract_registry's latent fallback path here)."""
    return PROJECT_ROOT


def ontology_output_dir() -> Path:
    """Get the ontology output directory."""
    return _runtime_paths.ontology_output_dir


def runs_dir() -> Path:
    """Get the runs directory."""
    return _runtime_paths.runs_dir


def sessions_dir() -> Path:
    """Get the sessions directory."""
    return _runtime_paths.sessions_dir


def schemas_dir() -> Path:
    """Get the schemas directory."""
    return _runtime_paths.schemas_dir


def config_dir() -> Path:
    """Get the config directory."""
    return _runtime_paths.config_dir
