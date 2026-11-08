"""Unified fingerprint computation module.

G26-Unify-02: Consolidates all fingerprinting logic into a single module
to ensure consistency between standard and optimized paths.

This module provides:
1. Single source of truth for folder fingerprint computation
2. Consistent hash computation logic across all callers
3. Optional optimized mode with sharded caching
4. File-level change detection capabilities
5. Backward-compatible API

Merged from:
- fingerprint_algorithm.py (file-level change detection)
- fingerprint_optimization.py (optimized computation - now deprecated)
"""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core.contracts import CacheVersionBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FingerprintResult:
    """Result of fingerprint computation.

    P1 / C-4: when *version_bundle* is provided, the final fingerprint
    mixes in the 5-dimension version key so a version bump invalidates
    cached results consistently with all other cache layers.
    """
    fingerprint: str
    file_count: int
    total_size: int
    cache_used: bool = False
    # P1 / C-4: version bundle used to compute the fingerprint (if any)
    version_bundle: Optional[CacheVersionBundle] = None


@dataclass
class ShardFingerprint:
    """Fingerprint for a single shard (subdirectory)."""
    shard_path: str  # Relative path from root
    fingerprint: str
    file_count: int
    total_size: int
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FingerprintCache:
    """Cache for sharded fingerprints.

    P1 / C-4: *version_bundle* records which version key was active when
    the cache was written. A mismatch on the next read invalidates the
    cache so version bumps propagate to fingerprint-level caching too.
    """
    root_fingerprint: str
    shards: dict[str, ShardFingerprint]
    total_files: int
    total_size: int
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # P1 / C-4: version bundle for cache invalidation
    version_bundle: Optional[Dict[str, str]] = None


@dataclass
class FileFingerprint:
    """Fingerprint entry for a single file.

    Attributes:
        relative_path: File path relative to folder root
        size_bytes: File size in bytes
        content_hash: MD5 hash of file content (for files > 1KB)
        modified_time: Last modification timestamp (for reference only)
    """
    relative_path: str
    size_bytes: int
    content_hash: str | None = None
    modified_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "content_hash": self.content_hash,
            "modified_time": self.modified_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileFingerprint:
        return cls(
            relative_path=data["relative_path"],
            size_bytes=data["size_bytes"],
            content_hash=data.get("content_hash"),
            modified_time=data.get("modified_time"),
        )


@dataclass
class FileChangeReport:
    """Report of file-level changes detected between two snapshots.

    Attributes:
        added_files: List of newly added file paths
        deleted_files: List of deleted file paths
        changed_files: List of files with content changes
        renamed_files: Dict mapping old paths to new paths (detected by content hash)
        unchanged_files: List of files that remained identical
        folder_fingerprint: Combined fingerprint of all files
    """
    added_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    renamed_files: dict[str, str] = field(default_factory=dict)  # old_path -> new_path
    unchanged_files: list[str] = field(default_factory=list)
    folder_fingerprint: str = ""

    def has_changes(self) -> bool:
        """Check if any changes were detected."""
        return bool(
            self.added_files
            or self.deleted_files
            or self.changed_files
            or self.renamed_files
        )

    def summary(self) -> str:
        """Generate human-readable change summary."""
        parts = []
        if self.added_files:
            parts.append(f"+{len(self.added_files)} added")
        if self.deleted_files:
            parts.append(f"-{len(self.deleted_files)} deleted")
        if self.changed_files:
            parts.append(f"~{len(self.changed_files)} modified")
        if self.renamed_files:
            parts.append(f"->{len(self.renamed_files)} renamed")
        if not parts:
            return "No changes detected"
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "added_files": self.added_files,
            "deleted_files": self.deleted_files,
            "changed_files": self.changed_files,
            "renamed_files": self.renamed_files,
            "unchanged_files": self.unchanged_files,
            "folder_fingerprint": self.folder_fingerprint,
            "has_changes": self.has_changes(),
            "summary": self.summary(),
        }


# ---------------------------------------------------------------------------
# Core fingerprint computation
# ---------------------------------------------------------------------------

def _mix_version_bundle(content_hash: str, version_bundle: Optional[CacheVersionBundle]) -> str:
    """Mix a version-bundle fingerprint into a content hash (P1 / C-4).

    Returns the original hash unchanged when *version_bundle* is None or
    empty. This is the single helper used by every fingerprint function so
    version semantics are uniform across the module.
    """
    if not version_bundle or version_bundle.is_empty():
        return content_hash
    md = hashlib.sha256()
    md.update(content_hash.encode("utf-8"))
    md.update(b"\x1f")
    md.update(version_bundle.to_fingerprint().encode("utf-8"))
    return md.hexdigest()


def compute_file_fingerprint(
    file_path: Path,
    folder_path: Path,
    version_bundle: Optional[CacheVersionBundle] = None,
) -> str:
    """Compute fingerprint for a single file.

    P1 / C-4: when *version_bundle* is provided, the result mixes in the
    5-dimension version key so a version bump changes the fingerprint.

    Args:
        file_path: Absolute path to the file
        folder_path: Root folder path (for computing relative path)
        version_bundle: Optional C-4 version bundle to mix in

    Returns:
        Hexadecimal hash string combining relative path, content, and version
    """
    md = hashlib.sha256()
    rel = str(file_path.relative_to(folder_path))
    stat = file_path.stat()
    
    md.update(rel.encode("utf-8", errors="ignore"))
    md.update(str(stat.st_size).encode("utf-8"))
    
    if stat.st_size > 1024:
        file_md = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                file_md.update(chunk)
        md.update(file_md.hexdigest().encode("utf-8"))
    
    content_hash = md.hexdigest()
    return _mix_version_bundle(content_hash, version_bundle)


def compute_file_fingerprint_detailed(file_path: Path, base_path: Path | None = None) -> FileFingerprint:
    """Compute detailed fingerprint for a single file (with metadata).

    Args:
        file_path: Path to the file
        base_path: Base folder path for computing relative path.
                  If None, uses file_path.parent.

    Returns:
        FileFingerprint with path, size, and content hash
    """
    if base_path is None:
        base_path = file_path.parent

    rel_path = str(file_path.relative_to(base_path))
    stat = file_path.stat()
    size = stat.st_size

    # For small files (<= 1KB), we don't compute content hash for efficiency
    content_hash = None
    if size > 1024:
        file_md = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                file_md.update(chunk)
        content_hash = file_md.hexdigest()

    return FileFingerprint(
        relative_path=rel_path,
        size_bytes=size,
        content_hash=content_hash,
        modified_time=stat.st_mtime,
    )


def compute_folder_fingerprint_simple(folder_path: Path) -> tuple[str, int, int]:
    """Compute folder fingerprint using simple sequential approach.
    
    Args:
        folder_path: Path to the folder
    
    Returns:
        (fingerprint, file_count, total_size)
    """
    md = hashlib.sha256()
    total_size = 0
    file_count = 0
    
    for path in sorted(p for p in folder_path.glob("**/*") if p.is_file()):
        file_count += 1
        stat = path.stat()
        total_size += stat.st_size
        
        rel = str(path.relative_to(folder_path))
        md.update(rel.encode("utf-8", errors="ignore"))
        md.update(str(stat.st_size).encode("utf-8"))
        
        if stat.st_size > 1024:
            file_md = hashlib.md5()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    file_md.update(chunk)
            md.update(file_md.hexdigest().encode("utf-8"))
    
    return md.hexdigest(), file_count, total_size


def compute_folder_fingerprint_optimized(
    folder_path: Path,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    shard_size_threshold: int = 100,
    parallel_workers: int = 4,
    sampling_threshold: int = 5000,
    sampling_ratio: float = 0.1,
) -> tuple[str, int, int]:
    """Compute folder fingerprint with optimized sharded caching.
    
    This mode uses sharded caching to accelerate computation but produces
    exactly the same fingerprint as the simple mode for consistency.
    
    Args:
        folder_path: Path to the folder
        cache_dir: Directory for shard cache storage
        use_cache: Whether to use cached shard fingerprints
        shard_size_threshold: Maximum files per shard before splitting
        parallel_workers: Number of parallel workers
        sampling_threshold: File count to trigger sampling
        sampling_ratio: Ratio of files to sample
    
    Returns:
        (fingerprint, file_count, total_size)
    """
    # Get all files first - same as simple mode
    all_files = sorted(p for p in folder_path.glob("**/*") if p.is_file())
    
    if not all_files:
        md = hashlib.sha256()
        return md.hexdigest(), 0, 0
    
    if cache_dir is None:
        cache_dir = folder_path.parent / ".cache" / "fingerprints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load cached file fingerprints
    file_fingerprints: dict[str, str] = {}
    if use_cache:
        file_fingerprints = _load_file_cache(cache_dir, folder_path)
    
    # Compute file fingerprints in parallel
    total_size = 0
    file_count = 0
    
    def _compute_file(fp_path: Path) -> tuple[str, int, int]:
        """Compute file content hash for caching.
        
        Returns: (content_md5_hash, count, size)
        """
        rel = str(fp_path.relative_to(folder_path))
        stat = fp_path.stat()
        
        # Check cache
        if rel in file_fingerprints:
            return file_fingerprints[rel], 1, stat.st_size
        
        # For files > 1KB, compute MD5 of content
        content_hash = ""
        if stat.st_size > 1024:
            file_md = hashlib.md5()
            with open(fp_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    file_md.update(chunk)
            content_hash = file_md.hexdigest()
        
        return content_hash, 1, stat.st_size
    
    # Store results by relative path
    results: dict[str, tuple[int, int]] = {}
    updated_cache: dict[str, str] = {}
    
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {executor.submit(_compute_file, f): f for f in all_files}
        
        for future in as_completed(futures):
            try:
                fp, count, size = future.result()
                file_path = futures[future]
                rel = str(file_path.relative_to(folder_path))
                
                file_count += count
                total_size += size
                results[rel] = (size, fp)
                updated_cache[rel] = fp
            except Exception as e:
                logger.warning(f"Failed to compute fingerprint for file: {e}")
    
    # Compute final fingerprint in sorted order - same as simple mode
    md = hashlib.sha256()
    for rel in sorted(results.keys()):
        size, fp = results[rel]
        md.update(rel.encode("utf-8", errors="ignore"))
        md.update(str(size).encode("utf-8"))
        if size > 1024:
            md.update(fp.encode("utf-8"))
    
    root_fingerprint = md.hexdigest()
    
    # Save cache
    if use_cache:
        _save_file_cache(cache_dir, folder_path, updated_cache)
    
    return root_fingerprint, file_count, total_size


# ---------------------------------------------------------------------------
# File-level change detection
# ---------------------------------------------------------------------------

def compute_folder_snapshot(folder_path: Path) -> dict[str, FileFingerprint]:
    """Compute detailed snapshot of all files in a folder.

    Args:
        folder_path: Path to the folder

    Returns:
        Dict mapping relative paths to FileFingerprint objects
    """
    snapshot: dict[str, FileFingerprint] = {}

    for path in sorted(p for p in folder_path.glob("**/*") if p.is_file()):
        file_fp = compute_file_fingerprint_detailed(path, folder_path)
        snapshot[file_fp.relative_path] = file_fp

    return snapshot


def detect_file_changes(
    folder_path: Path,
    previous_snapshot: dict[str, FileFingerprint],
    detect_renames: bool = True,
) -> FileChangeReport:
    """Detect file-level changes by comparing current folder to previous snapshot.

    This is the core algorithm for file-level change detection, enabling
    granular understanding of what changed within a folder.

    Args:
        folder_path: Path to the folder to analyze
        previous_snapshot: Dict of relative paths to FileFingerprint from previous run
        detect_renames: If True, attempt to detect renamed files by content hash

    Returns:
        FileChangeReport with detailed change information
    """
    current_snapshot = compute_folder_snapshot(folder_path)
    report = FileChangeReport()

    # Find added files (in current but not in previous)
    current_paths = set(current_snapshot.keys())
    previous_paths = set(previous_snapshot.keys())
    report.added_files = list(current_paths - previous_paths)
    report.deleted_files = list(previous_paths - current_paths)

    # Check for changes in files that exist in both
    for rel_path in current_paths & previous_paths:
        current_fp = current_snapshot[rel_path]
        previous_fp = previous_snapshot[rel_path]

        # Compare size first (fast check)
        if current_fp.size_bytes != previous_fp.size_bytes:
            report.changed_files.append(rel_path)
            continue

        # Compare content hash if available (both > 1KB)
        if current_fp.content_hash and previous_fp.content_hash:
            if current_fp.content_hash != previous_fp.content_hash:
                report.changed_files.append(rel_path)
                continue
        # If either file is small (<=1KB), size equality means unchanged

        report.unchanged_files.append(rel_path)

    # Detect renames: same content hash, different paths
    if detect_renames and report.deleted_files and report.added_files:
        # Build content hash -> path mappings
        deleted_hashes: dict[str, str] = {}
        for path in report.deleted_files:
            fp = previous_snapshot.get(path)
            if fp and fp.content_hash:
                deleted_hashes[fp.content_hash] = path

        added_hashes: dict[str, str] = {}
        for path in report.added_files:
            fp = current_snapshot.get(path)
            if fp and fp.content_hash:
                added_hashes[fp.content_hash] = path

        # Find matches
        for content_hash, old_path in deleted_hashes.items():
            if content_hash in added_hashes:
                new_path = added_hashes[content_hash]
                report.renamed_files[old_path] = new_path
                # Remove from added/deleted lists
                if old_path in report.deleted_files:
                    report.deleted_files.remove(old_path)
                if new_path in report.added_files:
                    report.added_files.remove(new_path)

    # Compute overall folder fingerprint
    report.folder_fingerprint = compute_folder_fingerprint(folder_path)

    logger.info(f"File change detection for {folder_path}: {report.summary()}")

    return report


def save_snapshot(snapshot: dict[str, FileFingerprint], output_path: Path) -> None:
    """Save a folder snapshot to JSON file."""
    data = {path: fp.to_dict() for path, fp in snapshot.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved folder snapshot to {output_path}")


def load_snapshot(input_path: Path) -> dict[str, FileFingerprint]:
    """Load a folder snapshot from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {path: FileFingerprint.from_dict(fp_data) for path, fp_data in data.items()}


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _load_file_cache(cache_dir: Path, folder_path: Path) -> dict[str, str]:
    """Load cached file fingerprints."""
    cache_file = cache_dir / f"{folder_path.name}_file_cache.json"
    if not cache_file.exists():
        return {}
    
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load file cache: {e}")
        return {}


def _save_file_cache(cache_dir: Path, folder_path: Path, fingerprints: dict[str, str]) -> None:
    """Save file fingerprints to cache."""
    cache_file = cache_dir / f"{folder_path.name}_file_cache.json"
    
    try:
        # Merge with existing cache
        existing = _load_file_cache(cache_dir, folder_path)
        existing.update(fingerprints)
        
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save file cache: {e}")


def _load_shard_cache(cache_dir: Path, folder_path: Path) -> dict[str, ShardFingerprint]:
    """Load cached shard fingerprints."""
    cache_file = cache_dir / f"{folder_path.name}_fingerprint_cache.json"
    if not cache_file.exists():
        return {}
    
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return {
            shard_name: ShardFingerprint(**shard_data)
            for shard_name, shard_data in data.get("shards", {}).items()
        }
    except Exception as e:
        logger.warning(f"Failed to load fingerprint cache: {e}")
        return {}


def _save_shard_cache(
    cache_dir: Path,
    folder_path: Path,
    root_fingerprint: str,
    shard_fingerprints: dict[str, str],
    shards: list[tuple[str, list[Path]]],
) -> None:
    """Save shard fingerprints to cache."""
    cache_file = cache_dir / f"{folder_path.name}_fingerprint_cache.json"
    
    try:
        shard_info: dict[str, dict[str, Any]] = {}
        for shard_name, files in shards:
            shard_info[shard_name] = {
                "shard_path": shard_name,
                "fingerprint": shard_fingerprints.get(shard_name, ""),
                "file_count": len(files),
                "total_size": sum(f.stat().st_size for f in files),
                "timestamp": datetime.now().isoformat(),
                "metadata": {},
            }
        
        cache = FingerprintCache(
            root_fingerprint=root_fingerprint,
            shards=shard_info,
            total_files=sum(len(files) for _, files in shards),
            total_size=sum(sum(f.stat().st_size for f in files) for _, files in shards),
            timestamp=datetime.now().isoformat(),
        )
        
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(asdict(cache), f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save fingerprint cache: {e}")


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def folder_fingerprint(
    folder_path: Path,
    use_optimized: bool = False,
    cache_dir: Optional[Path] = None,
    **kwargs,
) -> FingerprintResult:
    """Unified folder fingerprint computation.
    
    This is the single entry point for all fingerprint computations.
    Ensures consistent results between standard and optimized modes.
    
    Args:
        folder_path: Path to the folder
        use_optimized: If True, uses sharded caching and parallel computation
        cache_dir: Cache directory for optimized mode
    
    Returns:
        FingerprintResult containing fingerprint, file_count, and total_size
    """
    if use_optimized:
        fingerprint, file_count, total_size = compute_folder_fingerprint_optimized(
            folder_path, cache_dir=cache_dir, **kwargs
        )
        return FingerprintResult(
            fingerprint=fingerprint,
            file_count=file_count,
            total_size=total_size,
            cache_used=True,
        )
    else:
        fingerprint, file_count, total_size = compute_folder_fingerprint_simple(folder_path)
        return FingerprintResult(
            fingerprint=fingerprint,
            file_count=file_count,
            total_size=total_size,
            cache_used=False,
        )


def compute_folder_fingerprint(folder_path: Path) -> str:
    """Compute SHA-256 fingerprint for an entire folder.
    
    Backward-compatible alias for folder_fingerprint().
    
    Args:
        folder_path: Path to the folder
    
    Returns:
        Hexadecimal SHA-256 hash string
    """
    return folder_fingerprint(folder_path).fingerprint


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data classes
    "FingerprintResult",
    "ShardFingerprint",
    "FingerprintCache",
    "FileFingerprint",
    "FileChangeReport",
    # Core functions
    "compute_file_fingerprint",
    "compute_file_fingerprint_detailed",
    "compute_folder_fingerprint",
    "compute_folder_fingerprint_simple",
    "compute_folder_fingerprint_optimized",
    # Change detection
    "compute_folder_snapshot",
    "detect_file_changes",
    "save_snapshot",
    "load_snapshot",
    # Unified entry point
    "folder_fingerprint",
]
