"""Incremental analysis service for folder-level ontology extraction.

B-1 Implementation 2026-05-09:
Provides fingerprint-based caching and skip logic for folder analysis.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.application.services.unified_fingerprint import folder_fingerprint

logger = logging.getLogger(__name__)


@dataclass
class IncrementalAnalysisResult:
    """Result of incremental analysis check."""
    should_skip: bool
    previous_fingerprint: str | None
    current_fingerprint: str
    manifest_path: Path | None
    reason: str
    status: str  # "skip", "changed", "stale", "new"


@dataclass
class FingerprintCacheEntry:
    """Cache entry for fingerprint storage."""
    fingerprint: str
    run_id: str
    timestamp: str
    manifest_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


class IncrementalAnalysisService:
    """Service for managing incremental folder analysis.
    
    Provides:
    - Fingerprint caching with configurable TTL
    - Smart skip logic based on fingerprint comparison
    - Detailed skip/changed/stale reporting
    """
    
    def __init__(
        self,
        cache_dir: Path | None = None,
        ttl_hours: float = 24.0,
    ):
        """Initialize the incremental analysis service.
        
        Args:
            cache_dir: Directory for fingerprint cache storage.
                      Defaults to .cache/fingerprints in project root.
            ttl_hours: Time-to-live for cache entries in hours.
        """
        if cache_dir is None:
            # Find project root (where .git or pyproject.toml exists)
            current = Path.cwd()
            while current.parent != current:
                if (current / ".git").exists() or (current / "pyproject.toml").exists():
                    break
                current = current.parent
            cache_dir = current / ".cache" / "fingerprints"
        
        self.cache_dir = cache_dir
        self.ttl_hours = ttl_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, folder_path: Path) -> Path:
        """Get cache file path for a folder.
        
        Uses a stable hash of the folder path (not the fingerprint) so the
        cache path remains consistent even when folder contents change.
        """
        # Use folder path hash as cache key (stable, not content-dependent)
        import hashlib
        path_hash = hashlib.md5(str(folder_path.resolve()).encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{folder_path.name}_{path_hash}.json"
    
    def _load_cache_entry(self, folder_path: Path) -> FingerprintCacheEntry | None:
        """Load cached fingerprint entry for a folder."""
        cache_path = self._get_cache_path(folder_path)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Check TTL
            cached_time = datetime.fromisoformat(data["timestamp"])
            hours_since = (datetime.now() - cached_time).total_seconds() / 3600
            
            if hours_since > self.ttl_hours:
                logger.info(f"Cache entry for {folder_path} expired (TTL: {self.ttl_hours}h)")
                return None
            
            return FingerprintCacheEntry(**data)
        except Exception as e:
            logger.warning(f"Failed to load cache for {folder_path}: {e}")
            return None
    
    def _save_cache_entry(
        self,
        folder_path: Path,
        entry: FingerprintCacheEntry,
    ) -> None:
        """Save fingerprint cache entry for a folder."""
        cache_path = self._get_cache_path(folder_path)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(entry.__dict__, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache for {folder_path}: {e}")
    
    def check_should_skip(
        self,
        folder_path: str | Path,
        force_reanalyze: bool = False,
    ) -> IncrementalAnalysisResult:
        """Check if folder analysis should be skipped based on fingerprint.
        
        Args:
            folder_path: Path to the folder to analyze
            force_reanalyze: If True, always reanalyze regardless of cache
            
        Returns:
            IncrementalAnalysisResult with skip decision and metadata
        """
        folder = Path(folder_path)
        if not folder.exists():
            return IncrementalAnalysisResult(
                should_skip=False,
                previous_fingerprint=None,
                current_fingerprint="",
                manifest_path=None,
                reason="Folder does not exist",
                status="error",
            )
        
        # Compute current fingerprint
        fingerprint_result = folder_fingerprint(folder)
        current_fingerprint = getattr(fingerprint_result, "fingerprint", str(fingerprint_result))
        
        # Check force reanalyze
        if force_reanalyze:
            return IncrementalAnalysisResult(
                should_skip=False,
                previous_fingerprint=None,
                current_fingerprint=current_fingerprint,
                manifest_path=None,
                reason="Force reanalyze enabled",
                status="forced",
            )
        
        # Load cached entry
        cache_entry = self._load_cache_entry(folder)
        
        if cache_entry is None:
            return IncrementalAnalysisResult(
                should_skip=False,
                previous_fingerprint=None,
                current_fingerprint=current_fingerprint,
                manifest_path=None,
                reason="No cache entry found (new analysis)",
                status="new",
            )
        
        # Compare fingerprints
        if cache_entry.fingerprint == current_fingerprint:
            # Check if manifest still exists
            manifest_path = Path(cache_entry.manifest_path)
            if manifest_path.exists():
                return IncrementalAnalysisResult(
                    should_skip=True,
                    previous_fingerprint=cache_entry.fingerprint,
                    current_fingerprint=current_fingerprint,
                    manifest_path=manifest_path,
                    reason="Folder unchanged since last analysis",
                    status="skip",
                )
            else:
                return IncrementalAnalysisResult(
                    should_skip=False,
                    previous_fingerprint=cache_entry.fingerprint,
                    current_fingerprint=current_fingerprint,
                    manifest_path=None,
                    reason="Manifest missing (stale cache)",
                    status="stale",
                )
        else:
            return IncrementalAnalysisResult(
                should_skip=False,
                previous_fingerprint=cache_entry.fingerprint,
                current_fingerprint=current_fingerprint,
                manifest_path=Path(cache_entry.manifest_path) if cache_entry.manifest_path else None,
                reason="Folder contents changed since last analysis",
                status="changed",
            )
    
    def record_analysis_complete(
        self,
        folder_path: str | Path,
        run_id: str,
        manifest_path: Path,
        documents: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record successful analysis completion for future incremental checks.
        
        Args:
            folder_path: Path to the analyzed folder
            run_id: Analysis run ID
            manifest_path: Path to the generated manifest
            documents: List of documents with relative_path and hash
            metadata: Optional additional metadata to cache
        """
        folder = Path(folder_path)
        
        if documents:
            import hashlib
            doc_hashes = sorted(doc["hash"] for doc in documents if isinstance(doc, dict) and "hash" in doc)
            fingerprint = hashlib.md5("|".join(doc_hashes).encode("utf-8")).hexdigest()
        else:
            fingerprint_result = folder_fingerprint(folder)
            fingerprint = getattr(fingerprint_result, "fingerprint", str(fingerprint_result))
        
        entry = FingerprintCacheEntry(
            fingerprint=fingerprint,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            manifest_path=str(manifest_path),
            metadata=metadata or {},
        )
        
        self._save_cache_entry(folder, entry)
        logger.info(f"Cached fingerprint for {folder} (run: {run_id})")
    
    def check_incremental_analysis(
        self,
        folder_path: str | Path,
        documents: list[dict[str, Any]] | None = None,
        run_id: str | None = None,
        force_reanalyze: bool = False,
    ) -> IncrementalAnalysisResult:
        """Check if incremental analysis should be performed.
        
        Args:
            folder_path: Path to the folder to analyze
            documents: List of documents with relative_path and hash
            run_id: Analysis run ID
            force_reanalyze: If True, always reanalyze regardless of cache
            
        Returns:
            IncrementalAnalysisResult with skip decision and metadata
        """
        folder = Path(folder_path)
        
        if documents:
            import hashlib
            doc_hashes = sorted(doc["hash"] for doc in documents if isinstance(doc, dict) and "hash" in doc)
            content_hash = hashlib.md5("|".join(doc_hashes).encode("utf-8")).hexdigest()
        else:
            fingerprint_result = folder_fingerprint(folder)
            content_hash = getattr(fingerprint_result, "fingerprint", str(fingerprint_result))
        
        cache_entry = self._load_cache_entry(folder)
        
        if force_reanalyze:
            if run_id:
                manifest_path = folder / "analysis_manifest.json"
                self.record_analysis_complete(folder, run_id, manifest_path, documents)
            return IncrementalAnalysisResult(
                should_skip=False,
                previous_fingerprint=cache_entry.fingerprint if cache_entry else None,
                current_fingerprint=content_hash,
                manifest_path=None,
                reason="Force reanalyze enabled",
                status="forced",
            )
        
        if cache_entry is None:
            cache_path = self._get_cache_path(folder)
            if cache_path.exists():
                manifest_path = folder / "analysis_manifest.json"
                if run_id:
                    self.record_analysis_complete(folder, run_id, manifest_path, documents)
                return IncrementalAnalysisResult(
                    should_skip=False,
                    previous_fingerprint=None,
                    current_fingerprint=content_hash,
                    manifest_path=manifest_path,
                    reason="Cache entry expired (TTL)",
                    status="stale",
                )
            
            manifest_path = folder / "analysis_manifest.json"
            if run_id:
                self.record_analysis_complete(folder, run_id, manifest_path, documents)
            return IncrementalAnalysisResult(
                should_skip=False,
                previous_fingerprint=None,
                current_fingerprint=content_hash,
                manifest_path=manifest_path,
                reason="No cache entry found (new analysis)",
                status="new",
            )
        
        if cache_entry.fingerprint == content_hash:
            manifest_path = Path(cache_entry.manifest_path) if cache_entry.manifest_path else None
            return IncrementalAnalysisResult(
                should_skip=True,
                previous_fingerprint=cache_entry.fingerprint,
                current_fingerprint=content_hash,
                manifest_path=manifest_path,
                reason="Folder unchanged since last analysis",
                status="skip",
            )
        else:
            if run_id:
                manifest_path = folder / "analysis_manifest.json"
                self.record_analysis_complete(folder, run_id, manifest_path, documents)
            return IncrementalAnalysisResult(
                should_skip=False,
                previous_fingerprint=cache_entry.fingerprint,
                current_fingerprint=content_hash,
                manifest_path=Path(cache_entry.manifest_path) if cache_entry.manifest_path else None,
                reason="Folder contents changed since last analysis",
                status="changed",
            )

    def invalidate_cache(self, folder_path: str | Path) -> bool:
        """Invalidate cache entry for a folder.
        
        Returns:
            True if cache was found and removed, False otherwise
        """
        folder = Path(folder_path)
        cache_path = self._get_cache_path(folder)
        
        if cache_path.exists():
            try:
                cache_path.unlink()
                logger.info(f"Invalidated cache for {folder}")
                return True
            except Exception as e:
                logger.warning(f"Failed to invalidate cache for {folder}: {e}")
        
        return False
    
    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the fingerprint cache."""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_entries = len(cache_files)
            
            valid_entries = 0
            expired_entries = 0
            
            for cache_file in cache_files:
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    cached_time = datetime.fromisoformat(data["timestamp"])
                    hours_since = (datetime.now() - cached_time).total_seconds() / 3600
                    
                    if hours_since <= self.ttl_hours:
                        valid_entries += 1
                    else:
                        expired_entries += 1
                except Exception:
                    pass
            
            return {
                "cache_dir": str(self.cache_dir),
                "ttl_hours": self.ttl_hours,
                "total_entries": total_entries,
                "valid_entries": valid_entries,
                "expired_entries": expired_entries,
            }
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {"error": str(e)}

    def check_file_changes(
        self,
        folder_path: str | Path,
    ) -> dict[str, Any] | None:
        """Perform file-level change detection for a folder.

        P0 Implementation 2026-05-26:
        Provides granular file-level change information in addition to
        folder-level fingerprint comparison.

        Args:
            folder_path: Path to the folder to analyze

        Returns:
            FileChangeReport as dict, or None if no previous snapshot exists.
            Use this to understand exactly what changed within a folder.

        Example:
            result = service.check_file_changes("/path/to/folder")
            if result:
                print(f"Changed: {result['changed_files']}")
                print(f"Added: {result['added_files']}")
        """
        from src.application.services.fingerprint_algorithm import (
            detect_file_changes,
            load_snapshot,
        )

        folder = Path(folder_path)

        # Load previous snapshot from cache
        cache_entry = self._load_cache_entry(folder)
        if cache_entry is None:
            logger.info(f"No previous snapshot for {folder}, computing initial snapshot")
            return None

        # Load previous snapshot data
        previous_run_id = cache_entry.run_id
        snapshot_path = self.cache_dir / f"{folder.name}_{previous_run_id}_files.json"

        if not snapshot_path.exists():
            logger.warning(f"Previous snapshot file not found: {snapshot_path}")
            return None

        try:
            previous_snapshot = load_snapshot(snapshot_path)
        except Exception as e:
            logger.warning(f"Failed to load previous snapshot: {e}")
            return None

        # Detect changes
        report = detect_file_changes(folder, previous_snapshot, detect_renames=True)

        return report.to_dict()

    def record_analysis_complete_with_snapshot(
        self,
        folder_path: str | Path,
        run_id: str,
        manifest_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record successful analysis with detailed file-level snapshot.

        This extends record_analysis_complete() by also saving a detailed
        file-level snapshot for future change detection.

        Args:
            folder_path: Path to the analyzed folder
            run_id: Analysis run ID
            manifest_path: Path to the generated manifest
            metadata: Optional additional metadata to cache
        """
        from src.application.services.fingerprint_algorithm import (
            compute_folder_snapshot,
            save_snapshot,
        )

        folder = Path(folder_path)

        # Save fingerprint cache entry (standard)
        fingerprint = folder_fingerprint(folder)
        entry = FingerprintCacheEntry(
            fingerprint=fingerprint,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            manifest_path=str(manifest_path),
            metadata=metadata or {},
        )
        self._save_cache_entry(folder, entry)

        # Save detailed file snapshot
        snapshot = compute_folder_snapshot(folder)
        snapshot_path = self.cache_dir / f"{folder.name}_{run_id}_files.json"
        save_snapshot(snapshot, snapshot_path)

        logger.info(
            f"Cached fingerprint and snapshot for {folder} (run: {run_id}, "
            f"{len(snapshot)} files)"
        )
