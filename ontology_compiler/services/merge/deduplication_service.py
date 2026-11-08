"""Deduplication Service - Unified deduplication across file, run, entity, and result levels.

[DATA CONTINUITY PLAN 2026-07-09]: Implements four-level deduplication:
- File-level: Skip unchanged files based on hash comparison
- Run-level: Avoid re-running identical input combinations
- Entity-level: Deduplicate entities by unique identifiers
- Result-level: Deduplicate final outputs by content hash

Architecture:
- FileDeduplicator: Hash-based file comparison
- RunDeduplicator: Input fingerprint comparison
- EntityDeduplicator: Entity ID/code based deduplication
- ResultDeduplicator: Output content hash comparison

This module enables:
- Efficient incremental updates
- Avoid duplicate processing
- Consistent entity merging
- Clean final outputs
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DeduplicationLevel(str, Enum):
    """Levels of deduplication."""
    FILE = "file"
    RUN = "run"
    ENTITY = "entity"
    RESULT = "result"


@dataclass
class DeduplicationResult:
    """Result of a deduplication check."""
    is_duplicate: bool
    level: DeduplicationLevel
    duplicate_id: str = ""
    reason: str = ""
    existing_entry: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EntityKey:
    """Unique key for entity deduplication."""
    entity_type: str
    code: Optional[str] = None
    name: Optional[str] = None
    layer: Optional[str] = None

    def to_hashable(self) -> Tuple[str, ...]:
        """Convert to hashable tuple."""
        return (self.entity_type, self.code or "", self.name or "", self.layer or "")

    def __hash__(self):
        return hash(self.to_hashable())

    def __eq__(self, other):
        if isinstance(other, EntityKey):
            return self.to_hashable() == other.to_hashable()
        return False


class FileDeduplicator:
    """Deduplicator for file-level deduplication."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or _PROJECT_ROOT / "outputs" / "deduplication" / "file"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._file_hashes: Dict[str, str] = {}
        self._load_file_hashes()

    def generate_file_hash(self, file_path: Path) -> str:
        """Generate SHA-256 hash for a file."""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to generate hash for {file_path}: {e}")
            return ""

    def check_file(self, file_path: Path) -> DeduplicationResult:
        """Check if file is a duplicate."""
        file_key = str(file_path.resolve())
        current_hash = self.generate_file_hash(file_path)

        if file_key in self._file_hashes:
            if self._file_hashes[file_key] == current_hash:
                return DeduplicationResult(
                    is_duplicate=True,
                    level=DeduplicationLevel.FILE,
                    duplicate_id=file_key,
                    reason="File content unchanged",
                )

        self._file_hashes[file_key] = current_hash
        self._save_file_hashes()

        return DeduplicationResult(
            is_duplicate=False,
            level=DeduplicationLevel.FILE,
            duplicate_id=file_key,
            reason="New or modified file",
        )

    def mark_processed(self, file_path: Path) -> None:
        """Mark a file as processed."""
        file_key = str(file_path.resolve())
        current_hash = self.generate_file_hash(file_path)
        self._file_hashes[file_key] = current_hash
        self._save_file_hashes()

    def _load_file_hashes(self) -> None:
        """Load file hashes from storage."""
        index_file = self.storage_path / "file_hashes.json"
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    self._file_hashes = json.load(f)
                logger.info(f"Loaded {len(self._file_hashes)} file hashes")
            except Exception as e:
                logger.warning(f"Failed to load file hashes: {e}")

    def _save_file_hashes(self) -> None:
        """Save file hashes to storage."""
        index_file = self.storage_path / "file_hashes.json"
        try:
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(self._file_hashes, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save file hashes: {e}")


class RunDeduplicator:
    """Deduplicator for run-level deduplication."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or _PROJECT_ROOT / "outputs" / "deduplication" / "run"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._run_fingerprints: Dict[str, Dict[str, Any]] = {}
        self._load_run_fingerprints()

    def generate_run_fingerprint(self, input_files: List[str], rule_version: str, model_version: str) -> str:
        """Generate a unique fingerprint for a run's inputs."""
        input_str = json.dumps(sorted(input_files), ensure_ascii=False)
        data = f"{input_str}|{rule_version}|{model_version}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def check_run(self, input_files: List[str], rule_version: str, model_version: str) -> DeduplicationResult:
        """Check if this run combination has been executed before."""
        fingerprint = self.generate_run_fingerprint(input_files, rule_version, model_version)

        if fingerprint in self._run_fingerprints:
            existing = self._run_fingerprints[fingerprint]
            return DeduplicationResult(
                is_duplicate=True,
                level=DeduplicationLevel.RUN,
                duplicate_id=fingerprint,
                reason="Identical input combination already processed",
                existing_entry=existing,
            )

        return DeduplicationResult(
            is_duplicate=False,
            level=DeduplicationLevel.RUN,
            duplicate_id=fingerprint,
            reason="New input combination",
        )

    def record_run(self, run_id: str, input_files: List[str], rule_version: str, model_version: str,
                   output_path: str, status: str) -> None:
        """Record a run's fingerprint."""
        fingerprint = self.generate_run_fingerprint(input_files, rule_version, model_version)
        self._run_fingerprints[fingerprint] = {
            "run_id": run_id,
            "input_files": input_files,
            "rule_version": rule_version,
            "model_version": model_version,
            "output_path": output_path,
            "status": status,
            "recorded_at": datetime.now().isoformat(),
        }
        self._save_run_fingerprints()

    def _load_run_fingerprints(self) -> None:
        """Load run fingerprints from storage."""
        index_file = self.storage_path / "run_fingerprints.json"
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    self._run_fingerprints = json.load(f)
                logger.info(f"Loaded {len(self._run_fingerprints)} run fingerprints")
            except Exception as e:
                logger.warning(f"Failed to load run fingerprints: {e}")

    def _save_run_fingerprints(self) -> None:
        """Save run fingerprints to storage."""
        index_file = self.storage_path / "run_fingerprints.json"
        try:
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(self._run_fingerprints, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save run fingerprints: {e}")


class EntityDeduplicator:
    """Deduplicator for entity-level deduplication."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or _PROJECT_ROOT / "outputs" / "deduplication" / "entity"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._entity_index: Dict[str, Dict[str, Any]] = {}
        self._load_entity_index()

    def generate_entity_key(self, entity: Dict[str, Any]) -> str:
        """Generate a unique key for an entity."""
        entity_type = entity.get("type", "")
        code = entity.get("code", "") or entity.get("id", "")
        name = entity.get("name", "")
        layer = entity.get("layer", "")

        if code:
            key = f"{entity_type}|{code}|{layer}"
        elif name:
            key = f"{entity_type}|{name}|{layer}"
        else:
            key = f"{entity_type}|{entity.get('id', '')}"

        return key

    def check_entity(self, entity: Dict[str, Any]) -> DeduplicationResult:
        """Check if entity is a duplicate."""
        entity_key = self.generate_entity_key(entity)

        if entity_key in self._entity_index:
            existing = self._entity_index[entity_key]
            return DeduplicationResult(
                is_duplicate=True,
                level=DeduplicationLevel.ENTITY,
                duplicate_id=entity_key,
                reason="Entity with same type/code/name/layer already exists",
                existing_entry=existing,
            )

        return DeduplicationResult(
            is_duplicate=False,
            level=DeduplicationLevel.ENTITY,
            duplicate_id=entity_key,
            reason="New entity",
        )

    def record_entity(self, entity: Dict[str, Any], run_id: str, source_file: str) -> None:
        """Record an entity in the index."""
        entity_key = self.generate_entity_key(entity)
        self._entity_index[entity_key] = {
            "entity_id": entity.get("id", ""),
            "type": entity.get("type", ""),
            "code": entity.get("code", "") or entity.get("id", ""),
            "name": entity.get("name", ""),
            "layer": entity.get("layer", ""),
            "run_id": run_id,
            "source_file": source_file,
            "recorded_at": datetime.now().isoformat(),
        }
        self._save_entity_index()

    def deduplicate_entities(self, entities: List[Dict[str, Any]], run_id: str, source_file: str) -> Tuple[List[Dict[str, Any]], int]:
        """Deduplicate a list of entities, returning unique entities and duplicate count."""
        unique_entities = []
        duplicates = 0
        seen_keys = set()

        for entity in entities:
            entity_key = self.generate_entity_key(entity)
            if entity_key in seen_keys:
                duplicates += 1
                continue

            seen_keys.add(entity_key)
            unique_entities.append(entity)
            self.record_entity(entity, run_id, source_file)

        return unique_entities, duplicates

    def _load_entity_index(self) -> None:
        """Load entity index from storage."""
        index_file = self.storage_path / "entity_index.json"
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    self._entity_index = json.load(f)
                logger.info(f"Loaded {len(self._entity_index)} entity entries")
            except Exception as e:
                logger.warning(f"Failed to load entity index: {e}")

    def _save_entity_index(self) -> None:
        """Save entity index to storage."""
        index_file = self.storage_path / "entity_index.json"
        try:
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(self._entity_index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save entity index: {e}")


class ResultDeduplicator:
    """Deduplicator for result-level deduplication."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or _PROJECT_ROOT / "outputs" / "deduplication" / "result"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._result_hashes: Dict[str, Dict[str, Any]] = {}
        self._load_result_hashes()

    def generate_result_hash(self, result: Dict[str, Any]) -> str:
        """Generate a hash for the result content."""
        result_str = json.dumps(result, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(result_str.encode("utf-8")).hexdigest()

    def check_result(self, result: Dict[str, Any]) -> DeduplicationResult:
        """Check if result is a duplicate."""
        result_hash = self.generate_result_hash(result)

        if result_hash in self._result_hashes:
            existing = self._result_hashes[result_hash]
            return DeduplicationResult(
                is_duplicate=True,
                level=DeduplicationLevel.RESULT,
                duplicate_id=result_hash,
                reason="Identical result already exists",
                existing_entry=existing,
            )

        return DeduplicationResult(
            is_duplicate=False,
            level=DeduplicationLevel.RESULT,
            duplicate_id=result_hash,
            reason="New result",
        )

    def record_result(self, result: Dict[str, Any], run_id: str, output_path: str) -> None:
        """Record a result's hash."""
        result_hash = self.generate_result_hash(result)
        self._result_hashes[result_hash] = {
            "run_id": run_id,
            "output_path": output_path,
            "entity_count": len(result.get("entities", [])),
            "relation_count": len(result.get("relationships", [])),
            "recorded_at": datetime.now().isoformat(),
        }
        self._save_result_hashes()

    def _load_result_hashes(self) -> None:
        """Load result hashes from storage."""
        index_file = self.storage_path / "result_hashes.json"
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    self._result_hashes = json.load(f)
                logger.info(f"Loaded {len(self._result_hashes)} result hashes")
            except Exception as e:
                logger.warning(f"Failed to load result hashes: {e}")

    def _save_result_hashes(self) -> None:
        """Save result hashes to storage."""
        index_file = self.storage_path / "result_hashes.json"
        try:
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(self._result_hashes, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save result hashes: {e}")


class DeduplicationService:
    """Unified service for all deduplication operations."""

    def __init__(self):
        self.file_deduplicator = FileDeduplicator()
        self.run_deduplicator = RunDeduplicator()
        self.entity_deduplicator = EntityDeduplicator()
        self.result_deduplicator = ResultDeduplicator()

    def check_file(self, file_path: Path) -> DeduplicationResult:
        """Check if file is a duplicate."""
        return self.file_deduplicator.check_file(file_path)

    def mark_file_processed(self, file_path: Path) -> None:
        """Mark a file as processed."""
        self.file_deduplicator.mark_processed(file_path)

    def check_run(self, input_files: List[str], rule_version: str, model_version: str) -> DeduplicationResult:
        """Check if run combination is a duplicate."""
        return self.run_deduplicator.check_run(input_files, rule_version, model_version)

    def record_run(self, run_id: str, input_files: List[str], rule_version: str, model_version: str,
                   output_path: str, status: str) -> None:
        """Record a run."""
        self.run_deduplicator.record_run(run_id, input_files, rule_version, model_version, output_path, status)

    def check_entity(self, entity: Dict[str, Any]) -> DeduplicationResult:
        """Check if entity is a duplicate."""
        return self.entity_deduplicator.check_entity(entity)

    def record_entity(self, entity: Dict[str, Any], run_id: str, source_file: str) -> None:
        """Record an entity."""
        self.entity_deduplicator.record_entity(entity, run_id, source_file)

    def deduplicate_entities(self, entities: List[Dict[str, Any]], run_id: str, source_file: str) -> Tuple[List[Dict[str, Any]], int]:
        """Deduplicate entities."""
        return self.entity_deduplicator.deduplicate_entities(entities, run_id, source_file)

    def check_result(self, result: Dict[str, Any]) -> DeduplicationResult:
        """Check if result is a duplicate."""
        return self.result_deduplicator.check_result(result)

    def record_result(self, result: Dict[str, Any], run_id: str, output_path: str) -> None:
        """Record a result."""
        self.result_deduplicator.record_result(result, run_id, output_path)

    def get_summary(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        return {
            "file_cache_size": len(self.file_deduplicator._file_hashes),
            "run_cache_size": len(self.run_deduplicator._run_fingerprints),
            "entity_cache_size": len(self.entity_deduplicator._entity_index),
            "result_cache_size": len(self.result_deduplicator._result_hashes),
        }


_global_deduplication_service = None


def get_deduplication_service() -> DeduplicationService:
    """Get or create global deduplication service instance."""
    global _global_deduplication_service
    if _global_deduplication_service is None:
        _global_deduplication_service = DeduplicationService()
    return _global_deduplication_service


__all__ = [
    "DeduplicationLevel",
    "DeduplicationResult",
    "FileDeduplicator",
    "RunDeduplicator",
    "EntityDeduplicator",
    "ResultDeduplicator",
    "DeduplicationService",
    "get_deduplication_service",
]