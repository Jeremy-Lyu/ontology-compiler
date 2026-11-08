"""Document cache service for incremental processing.

Features:
- File content fingerprinting using SHA256 hash
- LLM-cleaned text caching with version management
- Incremental skip logic: skip files whose content hasn't changed
- Cache invalidation and cleanup

Use case: When processing the same document multiple times,
if the content hash matches a cached version, skip expensive LLM extraction.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

# P1 / C-4: unified version bundle from the shared contract layer
from core.contracts import CacheVersionBundle, version_bundles_match


class DocumentCache:
    """Document fingerprint cache for incremental processing."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("outputs/cache/documents")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fingerprint_store = self.cache_dir / "fingerprints.json"
        self.text_store = self.cache_dir / "cleaned_texts"
        self.text_store.mkdir(exist_ok=True)
        self._fingerprints = self._load_fingerprints()

    def _load_fingerprints(self) -> Dict[str, Dict[str, Any]]:
        if self.fingerprint_store.exists():
            try:
                with self.fingerprint_store.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_fingerprints(self) -> None:
        try:
            temp_file = self.fingerprint_store.with_suffix(".tmp")
            with temp_file.open("w", encoding="utf-8") as f:
                json.dump(self._fingerprints, f, ensure_ascii=False, indent=2)
            if self.fingerprint_store.exists():
                self.fingerprint_store.unlink()
            temp_file.rename(self.fingerprint_store)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to save fingerprints: {e}")

    def compute_content_hash(self, content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content."""
        with file_path.open("rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def get_cached_result(
        self,
        file_path: Path,
        version_bundle: Optional[CacheVersionBundle] = None,
    ) -> Optional[Dict[str, Any]]:
        """Check if file has been processed and return cached result.

        C-4: when *version_bundle* is provided, the cached entry is only
        considered valid if every version dimension matches. A mismatch
        (stale result) returns ``None`` so the caller re-processes the file —
        never silently reuse a result generated under a different prompt /
        schema / policy / model / mode.
        """
        file_hash = self.compute_file_hash(file_path)
        file_key = str(file_path.resolve())

        if file_key in self._fingerprints:
            cached_entry = self._fingerprints[file_key]
            if cached_entry.get("hash") == file_hash:
                if version_bundle is not None:
                    cached_vb = CacheVersionBundle.from_dict(
                        cached_entry.get("version_bundle")
                    )
                    if not version_bundles_match(version_bundle, cached_vb):
                        # Stale cache: version dimensions differ → invalidate.
                        import logging
                        logging.getLogger(__name__).info(
                            "DocumentCache version mismatch for %s; invalidating stale entry",
                            file_path.name,
                        )
                        self.invalidate_file(file_path)
                        return None
                cleaned_text_path = self.text_store / f"{cached_entry['text_key']}.md"
                if cleaned_text_path.exists():
                    cleaned_text = cleaned_text_path.read_text(encoding="utf-8")
                    return {
                        "hash": file_hash,
                        "cleaned_text": cleaned_text,
                        "processed_at": cached_entry.get("processed_at"),
                        "entity_count": cached_entry.get("entity_count", 0),
                        "relation_count": cached_entry.get("relation_count", 0),
                        "text_key": cached_entry.get("text_key"),
                        "version_bundle": cached_entry.get("version_bundle") or {},
                    }
        return None

    def cache_result(
        self,
        file_path: Path,
        cleaned_text: str,
        entity_count: int = 0,
        relation_count: int = 0,
        result_data: Optional[Dict[str, Any]] = None,
        version_bundle: Optional[CacheVersionBundle] = None,
    ) -> str:
        """Cache processed result for a file.

        C-4: *version_bundle* is persisted alongside the fingerprint so a later
        run can detect staleness and refuse to reuse an outdated result.
        """
        file_hash = self.compute_file_hash(file_path)
        file_key = str(file_path.resolve())
        text_key = f"{file_hash[:16]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        cleaned_text_path = self.text_store / f"{text_key}.md"
        cleaned_text_path.write_text(cleaned_text, encoding="utf-8")

        if result_data:
            result_path = self.text_store / f"{text_key}.json"
            result_path.write_text(json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8")

        entry: Dict[str, Any] = {
            "hash": file_hash,
            "text_key": text_key,
            "processed_at": datetime.now().isoformat(),
            "entity_count": entity_count,
            "relation_count": relation_count,
            "file_size": file_path.stat().st_size,
            "file_name": file_path.name,
        }
        if version_bundle is not None:
            entry["version_bundle"] = version_bundle.to_dict()

        self._fingerprints[file_key] = entry

        self._save_fingerprints()
        return file_hash

    def invalidate_file(self, file_path: Path) -> bool:
        """Remove cached entry for a file."""
        file_key = str(file_path.resolve())
        if file_key in self._fingerprints:
            text_key = self._fingerprints[file_key].get("text_key")
            if text_key:
                text_path = self.text_store / f"{text_key}.md"
                if text_path.exists():
                    text_path.unlink()
            del self._fingerprints[file_key]
            self._save_fingerprints()
            return True
        return False

    def cleanup_old_entries(self, days_to_keep: int = 30) -> int:
        """Remove cache entries older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted = 0

        to_delete = []
        for file_key, entry in self._fingerprints.items():
            try:
                processed_at = datetime.fromisoformat(entry.get("processed_at", ""))
                if processed_at < cutoff_date:
                    to_delete.append(file_key)
            except Exception:
                continue

        for file_key in to_delete:
            text_key = self._fingerprints[file_key].get("text_key")
            if text_key:
                text_path = self.text_store / f"{text_key}.md"
                if text_path.exists():
                    text_path.unlink()
            del self._fingerprints[file_key]
            deleted += 1

        if deleted > 0:
            self._save_fingerprints()

        return deleted

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self._fingerprints)
        total_size = sum(
            (self.text_store / f"{entry['text_key']}.md").stat().st_size
            for entry in self._fingerprints.values()
            if (self.text_store / f"{entry['text_key']}.md").exists()
        )

        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }

    def is_file_changed(self, file_path: Path) -> bool:
        """Check if file content has changed since last processing."""
        cached = self.get_cached_result(file_path)
        return cached is None