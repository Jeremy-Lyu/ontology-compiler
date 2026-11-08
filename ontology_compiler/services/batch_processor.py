"""Batch Processor - Async concurrent file processing.

P6 Optimization (2026-06-09): Implements true async concurrent processing
with semaphore-controlled concurrency limits.

P7 Enhancement (2026-07-12): Added global semaphore to control overall LLM
call rate and prevent DIFY API overload. The default max_concurrent=3 matches
the DIFY_RATE_LIMIT_MAX_CALLS configuration to prevent circuit breaker trips.

Expected benefit: Concurrent processing of files while respecting LLM API rate limits.
"""

from __future__ import annotations

import asyncio
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_global_semaphore: asyncio.Semaphore | None = None


def get_global_semaphore(max_concurrent: int = 3) -> asyncio.Semaphore:
    """Get global semaphore for limiting concurrent LLM calls across all BatchProcessor instances.
    
    This ensures that the total number of concurrent LLM calls across all files
    does not exceed the DIFY rate limit, preventing circuit breaker trips.
    
    Args:
        max_concurrent: Maximum number of concurrent LLM calls (default matches DIFY_RATE_LIMIT_MAX_CALLS)
    
    Returns:
        Global semaphore instance
    """
    global _global_semaphore
    if _global_semaphore is None:
        limit = int(os.environ.get("DIFY_RATE_LIMIT_MAX_CALLS", max_concurrent))
        _global_semaphore = asyncio.Semaphore(limit)
        logger.info(f"Global semaphore initialized with limit={limit}")
    return _global_semaphore


class BatchProcessor:
    """
    Async batch processor for concurrent file processing.
    
    Supports:
    - Semaphore-controlled concurrency limits (both instance and global)
    - Progress callbacks
    - Error handling and partial results
    - Processing time tracking
    - Global LLM call rate limiting
    """

    def __init__(
        self,
        max_concurrent: int = 3,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        use_global_semaphore: bool = True,
    ):
        """
        Initialize batch processor.
        
        Args:
            max_concurrent: Maximum number of concurrent tasks per instance
            progress_callback: Optional callback(current, total, file_path)
            use_global_semaphore: Use global semaphore to limit overall LLM calls
        """
        self.max_concurrent = max_concurrent
        self.progress_callback = progress_callback
        self.use_global_semaphore = use_global_semaphore
        self._global_sem = get_global_semaphore() if use_global_semaphore else None
        self.logger = logging.getLogger("BatchProcessor")

    async def process_batch(
        self,
        items: List[Any],
        process_func: Callable[[Any], Any],
    ) -> Dict[str, Any]:
        """
        Process multiple items concurrently with rate limiting.
        
        Args:
            items: List of items to process (can be Path, Fragment, or any type)
            process_func: Async function that processes a single item
            
        Returns:
            Dict with results, errors, and statistics
        """
        global_limit = self._global_sem._value if self._global_sem else "N/A"
        self.logger.info(f"Starting batch processing of {len(items)} items (max_concurrent={self.max_concurrent}, global_limit={global_limit})")
        
        start_time = datetime.now()
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        results: Dict[str, Any] = {}
        errors: Dict[str, str] = {}
        completed_count = 0
        total_count = len(items)

        async def process_with_limit(item: Any) -> None:
            nonlocal completed_count
            item_key = str(item)
            
            try:
                if self._global_sem:
                    async with self._global_sem:
                        async with semaphore:
                            self.logger.debug(f"Processing: {getattr(item, 'name', str(item))}")
                            result = await process_func(item)
                            results[item_key] = result
                else:
                    async with semaphore:
                        self.logger.debug(f"Processing: {getattr(item, 'name', str(item))}")
                        result = await process_func(item)
                        results[item_key] = result
                    
            except Exception as e:
                self.logger.error(f"Error processing {getattr(item, 'name', str(item))}: {e}", exc_info=True)
                errors[item_key] = str(e)
                
            finally:
                completed_count += 1
                if self.progress_callback:
                    self.progress_callback(completed_count, total_count, item_key)

        tasks = [process_with_limit(item) for item in items]
        await asyncio.gather(*tasks)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        self.logger.info(
            f"Batch processing completed: {len(results)}/{total_count} successful, "
            f"{len(errors)} errors, elapsed={elapsed:.2f}s"
        )
        
        return {
            "results": results,
            "errors": errors,
            "stats": {
                "total_files": total_count,
                "successful": len(results),
                "failed": len(errors),
                "elapsed_seconds": elapsed,
                "average_time_per_file": elapsed / max(total_count, 1),
                "throughput": total_count / elapsed if elapsed > 0 else 0,
            },
        }

    async def process_with_intelligent_batching(
        self,
        files: List[Path],
        process_func: Callable[[Path], Any],
        small_file_threshold: int = 10 * 1024,
        large_file_threshold: int = 50 * 1024,
    ) -> Dict[str, Any]:
        """
        Process files with intelligent batching strategy.
        
        - Small files (< small_file_threshold): Batch together for efficiency
        - Medium files: Process individually with concurrency
        - Large files (> large_file_threshold): Process sequentially to avoid timeouts
        
        Args:
            files: List of file paths to process
            process_func: Async function that processes a single file
            small_file_threshold: Threshold for small files in bytes (default 10KB)
            large_file_threshold: Threshold for large files in bytes (default 50KB)
            
        Returns:
            Dict with results, errors, and statistics
        """
        self.logger.info("Starting intelligent batch processing")
        
        small_files: List[Path] = []
        medium_files: List[Path] = []
        large_files: List[Path] = []
        
        for file in files:
            try:
                size = file.stat().st_size
                if size < small_file_threshold:
                    small_files.append(file)
                elif size > large_file_threshold:
                    large_files.append(file)
                else:
                    medium_files.append(file)
            except Exception:
                medium_files.append(file)
        
        self.logger.info(
            f"File distribution: small={len(small_files)}, "
            f"medium={len(medium_files)}, large={len(large_files)}"
        )
        
        all_results: Dict[str, Any] = {}
        all_errors: Dict[str, str] = {}
        
        if small_files:
            self.logger.info(f"Processing {len(small_files)} small files sequentially")
            for file in small_files:
                try:
                    result = await process_func(file)
                    all_results[str(file)] = result
                except Exception as e:
                    all_errors[str(file)] = str(e)
        
        if medium_files:
            self.logger.info(f"Processing {len(medium_files)} medium files concurrently")
            medium_result = await self.process_batch(medium_files, process_func)
            all_results.update(medium_result["results"])
            all_errors.update(medium_result["errors"])
        
        if large_files:
            self.logger.info(f"Processing {len(large_files)} large files sequentially")
            for file in large_files:
                try:
                    result = await process_func(file)
                    all_results[str(file)] = result
                except Exception as e:
                    all_errors[str(file)] = str(e)
        
        return {
            "results": all_results,
            "errors": all_errors,
            "stats": {
                "total_files": len(files),
                "successful": len(all_results),
                "failed": len(all_errors),
                "file_distribution": {
                    "small": len(small_files),
                    "medium": len(medium_files),
                    "large": len(large_files),
                },
            },
        }


class ExtractionCache:
    """
    [DEPRECATED] Cache for extraction results to avoid reprocessing unchanged files.

    **Deprecated**: Use `ChangeDetector` from `ontology_compiler.services.change_detector`
    for fingerprint-based change detection. ChangeDetector provides superior entity-level
    tracking and is already integrated into the MergeStep.

    [Phase 1-6 / TODO-REMOVE] Verified ZERO callers across active (non-archived) code
    (grep for ``ExtractionCache(`` returned no matches; only archived docs/reports
    reference the class name in prose / source listings).  Safe to delete entirely
    in the next minor release bump (after confirming no downstream consumers still
    import ``from ontology_compiler.services.batch_processor import ExtractionCache``
    outside this monorepo).

    Target removal: v1.x after migration sign-off (see QA report P1 row 3).
    """

    def __init__(self, cache_dir: str = "outputs/cache"):
        """
        Initialize extraction cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        warnings.warn(
            "ExtractionCache is deprecated. Use ChangeDetector instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._in_memory_cache: Dict[str, Any] = {}
        self.logger = logging.getLogger("ExtractionCache")

    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute MD5 hash of file contents.
        
        Args:
            file_path: Path to file
            
        Returns:
            MD5 hash string
        """
        import hashlib
        
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to compute hash for {file_path}: {e}")
            return ""

    def get_cached_result(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Get cached extraction result for a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Cached result or None if not found
        """
        file_hash = self._compute_file_hash(file_path)
        if not file_hash:
            return None
        
        if file_hash in self._in_memory_cache:
            self.logger.debug(f"Cache hit (memory): {file_path.name}")
            return self._in_memory_cache[file_hash]
        
        cache_file = self.cache_dir / f"{file_hash}.json"
        if cache_file.exists():
            try:
                import json
                with open(cache_file, "r", encoding="utf-8") as f:
                    result = json.load(f)
                    self._in_memory_cache[file_hash] = result
                    self.logger.debug(f"Cache hit (disk): {file_path.name}")
                    return result
            except Exception as e:
                self.logger.warning(f"Failed to load cache for {file_path}: {e}")
        
        return None

    def cache_result(self, file_path: Path, result: Dict[str, Any]) -> None:
        """
        Cache extraction result for a file.
        
        Args:
            file_path: Path to file
            result: Extraction result to cache
        """
        file_hash = self._compute_file_hash(file_path)
        if not file_hash:
            return
        
        self._in_memory_cache[file_hash] = result
        
        cache_file = self.cache_dir / f"{file_hash}.json"
        try:
            import json
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Cached result for: {file_path.name}")
        except Exception as e:
            self.logger.warning(f"Failed to cache result for {file_path}: {e}")

    def invalidate_cache(self, file_path: Path) -> None:
        """
        Invalidate cache for a specific file.
        
        Args:
            file_path: Path to file
        """
        file_hash = self._compute_file_hash(file_path)
        if file_hash in self._in_memory_cache:
            del self._in_memory_cache[file_hash]
        
        cache_file = self.cache_dir / f"{file_hash}.json"
        if cache_file.exists():
            cache_file.unlink()
            self.logger.debug(f"Invalidated cache for: {file_path.name}")

    def clear_all(self) -> None:
        """Clear all cached results."""
        self._in_memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        self.logger.info("All cache cleared")