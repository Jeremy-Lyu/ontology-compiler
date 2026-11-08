from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ManifestFileState:
    path: str
    last_modified_at: str
    sha256_hash: str
    watermark_id: int
    last_run_id: str | None = None
    domain: str | None = None
    semantic_hash: str | None = None  # [Phase 6-CACHE-1] Semantic content hash

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_for_semantic_hash(content: str) -> str:
    """Normalize text content for semantic hashing.
    
    Removes formatting variations that don't affect semantic meaning:
    - Extra whitespace
    - Line ending differences
    - Case variations (for non-code content)
    - Common punctuation normalization
    
    [Phase 6-CACHE-1]: Reduces false positives in change detection
    """
    # Normalize line endings
    text = content.replace('\r\n', '\n').replace('\r', '\n')
    # Collapse multiple whitespace to single space
    import re
    text = re.sub(r'\s+', ' ', text)
    # Normalize common punctuation
    text = re.sub(r'[\"\"\"\']', '"', text)
    text = re.sub(r'[—–]', '-', text)
    # Lowercase for semantic comparison
    text = text.lower().strip()
    return text


def _calculate_semantic_hash(content: str) -> str:
    """Calculate semantic hash of text content."""
    normalized = _normalize_for_semantic_hash(content)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


class ManifestTracker:
    """Track incremental source-file state for pipeline runs."""

    def __init__(self, manifest_path: Path | None = None) -> None:
        self.manifest_path = manifest_path or Path("outputs/ontology/manifest.json")
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {
                "schema_version": "v1.0",
                "generated_at": None,
                "watermark": 0,
                "files": {},
                "runs": [],
            }
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"invalid manifest payload: {self.manifest_path}")
        payload.setdefault("schema_version", "v1.0")
        payload.setdefault("generated_at", None)
        payload.setdefault("watermark", 0)
        payload.setdefault("files", {})
        payload.setdefault("runs", [])
        return payload

    def save(self, payload: dict[str, Any]) -> None:
        payload = {
            **payload,
            "generated_at": datetime.now().isoformat(),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def fingerprint_file(self, path: Path) -> ManifestFileState:
        stat = path.stat()
        content_bytes = path.read_bytes()
        sha = hashlib.sha256(content_bytes).hexdigest()
        
        # [Phase 6-CACHE-1] Calculate semantic hash for text files
        semantic_hash = None
        try:
            content_text = content_bytes.decode('utf-8')
            semantic_hash = _calculate_semantic_hash(content_text)
        except (UnicodeDecodeError, Exception):
            # Binary files or decode errors: skip semantic hash
            pass
        
        return ManifestFileState(
            path=str(path),
            last_modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            sha256_hash=sha,
            watermark_id=0,
            semantic_hash=semantic_hash,
        )

    def diff_files(self, files: list[Path]) -> dict[str, Any]:
        manifest = self.load()
        stored_files = manifest.get("files") or {}
        changed: list[str] = []
        unchanged: list[str] = []
        new: list[str] = []
        missing: list[str] = []
        current_states: dict[str, ManifestFileState] = {}

        for path in files:
            if not path.exists() or not path.is_file():
                missing.append(str(path))
                continue
            state = self.fingerprint_file(path)
            current_states[str(path)] = state
            stored = stored_files.get(str(path)) or {}
            if not stored:
                new.append(str(path))
                changed.append(str(path))
                continue
            
            # [Phase 6-CACHE-1] Check both sha256 and semantic hash
            sha_changed = str(stored.get("sha256_hash") or "") != state.sha256_hash
            if not sha_changed:
                unchanged.append(str(path))
                continue
            
            # sha256 changed - check if semantic content is the same
            stored_semantic = stored.get("semantic_hash")
            if stored_semantic and state.semantic_hash and stored_semantic == state.semantic_hash:
                # Content is semantically identical (only formatting changed)
                unchanged.append(str(path))
                continue
            
            changed.append(str(path))

        removed = [
            stored_path
            for stored_path in stored_files.keys()
            if stored_path not in current_states and stored_path not in missing
        ]

        return {
            "changed_files": sorted(changed),
            "unchanged_files": sorted(unchanged),
            "new_files": sorted(new),
            "missing_files": sorted(missing),
            "removed_files": sorted(removed),
            "current_states": current_states,
            "previous_watermark": int(manifest.get("watermark", 0) or 0),
        }

    def update_files(
        self,
        *,
        files: list[Path],
        run_id: str,
        domains_by_file: dict[str, str] | None = None,
        artifacts_by_file: dict[str, dict[str, Any]] | None = None,
        stale_retention_runs: int = 5,
        time_window_days: int = 30,
    ) -> dict[str, Any]:
        manifest = self.load()
        diff = self.diff_files(files)
        watermark = int(manifest.get("watermark", 0) or 0)
        file_payload = dict(manifest.get("files") or {})
        domains_by_file = domains_by_file or {}
        artifacts_by_file = artifacts_by_file or {}
        stale_retention_runs = max(1, int(stale_retention_runs or 1))
        
        # Calculate time window cutoff
        now = datetime.now()
        time_window_cutoff = now.timestamp() - (time_window_days * 24 * 60 * 60)

        for path_str, state in diff.get("current_states", {}).items():
            watermark += 1
            file_payload[path_str] = {
                **state.to_dict(),
                "watermark_id": watermark,
                "last_run_id": run_id,
                "domain": domains_by_file.get(path_str),
                "artifacts": dict(artifacts_by_file.get(path_str) or {}),
                "stale_since_run": None,
            }

        for removed in diff.get("removed_files", []):
            stale_entry = dict(file_payload.get(removed) or {})
            stale_entry.update(
                {
                    "path": removed,
                    "stale_since_run": stale_entry.get("stale_since_run") or run_id,
                    "last_seen_run": run_id,
                }
            )
            file_payload[removed] = stale_entry

        for missing in diff.get("missing_files", []):
            stale_entry = dict(file_payload.get(missing) or {})
            stale_entry.update(
                {
                    "path": missing,
                    "stale_since_run": stale_entry.get("stale_since_run") or run_id,
                    "last_seen_run": run_id,
                }
            )
            file_payload[missing] = stale_entry

        runs = list(manifest.get("runs") or [])
        runs.append(
            {
                "run_id": run_id,
                "updated_at": datetime.now().isoformat(),
                "changed_files": diff.get("changed_files") or [],
                "unchanged_files": diff.get("unchanged_files") or [],
                "removed_files": diff.get("removed_files") or [],
                "missing_files": diff.get("missing_files") or [],
            }
        )

        retained_run_ids = {str(item.get("run_id") or "") for item in runs[-stale_retention_runs:]}
        
        # Also retain runs within time window
        retained_run_ids_in_window = set()
        for item in runs:
            run_time_str = item.get("updated_at") or ""
            if run_time_str:
                try:
                    run_time = datetime.fromisoformat(run_time_str)
                    if run_time.timestamp() >= time_window_cutoff:
                        retained_run_ids_in_window.add(str(item.get("run_id") or ""))
                except Exception:
                    pass
        
        # Combine both retention strategies (union)
        retained_run_ids = retained_run_ids.union(retained_run_ids_in_window)
        
        pruned_file_payload: dict[str, Any] = {}
        for path_str, state in file_payload.items():
            stale_since_run = str((state or {}).get("stale_since_run") or "")
            last_seen_run = str((state or {}).get("last_seen_run") or "")
            # Keep file if it's not stale, or if stale_since_run is within retained run IDs
            # last_seen_run only retains non-stale files (when stale_since_run is absent)
            if not stale_since_run:
                if not last_seen_run or last_seen_run in retained_run_ids:
                    pruned_file_payload[path_str] = state
            elif stale_since_run in retained_run_ids:
                pruned_file_payload[path_str] = state

        updated = {
            **manifest,
            "watermark": watermark,
            "files": pruned_file_payload,
            "runs": runs[-100:],
        }
        self.save(updated)
        return updated


