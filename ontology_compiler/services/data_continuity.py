"""Data Continuity Service - Unified fingerprint, manifest, and processed index management.

[DATA CONTINUITY PLAN 2026-07-09]: Implements the four core components:
- File fingerprint service
- Run manifest
- Processed index
- Artifact index

Architecture:
- FingerprintService: Generate and track file hashes
- RunManifest: Record run details (input files, rules, models, outputs)
- ProcessedIndex: Track which files have been processed
- ArtifactIndex: Track generated artifacts

This module enables:
- Skip unchanged files (file-level deduplication)
- Trace any run back to its inputs and outputs
- Incremental updates instead of full re-runs
- Assetization of pipeline outputs
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from core.contracts import CacheVersionBundle

logger = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent.parent


@dataclass
class ContinuityFileRecord:
    """Data-continuity record for a processed file.

    Records the file fingerprint plus its processing history in the pipeline
    continuity index. Distinct context from the canonical lightweight
    `FileFingerprint` (`src.application.services.unified_fingerprint`): this
    record is a continuity-manifest entry, not a pure fingerprint. Renamed per
    context under Phase 4 (统一事实来源) so the `FileFingerprint` name stays
    unambiguous.
    """
    file_path: str
    file_hash: str
    file_size: int
    modified_time: str
    first_processed_at: str
    last_processed_run_id: str = ""
    last_processed_at: str = ""
    file_type: str = ""
    encoding: str = ""
    mime_type: str = ""
    language: str = ""
    source_domain: str = ""
    process_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContinuityFileRecord":
        return cls(
            file_path=data.get("file_path", ""),
            file_hash=data.get("file_hash", ""),
            file_size=data.get("file_size", 0),
            modified_time=data.get("modified_time", ""),
            first_processed_at=data.get("first_processed_at", ""),
            last_processed_run_id=data.get("last_processed_run_id", ""),
            last_processed_at=data.get("last_processed_at", ""),
            file_type=data.get("file_type", ""),
            encoding=data.get("encoding", ""),
            mime_type=data.get("mime_type", ""),
            language=data.get("language", ""),
            source_domain=data.get("source_domain", ""),
            process_count=data.get("process_count", 0),
        )


@dataclass
class RunManifest:
    """Manifest for a single pipeline run.

    P1 / C-4: ``version_bundle`` records the 5-dimension version key
    (prompt / schema / policy / model_family / extraction_mode) for this
    run, enabling precise cross-run incremental reuse decisions.
    """
    run_id: str
    session_id: str
    start_time: str
    end_time: str = ""
    status: str = "running"
    input_files: List[str] = field(default_factory=list)
    input_file_fingerprints: Dict[str, str] = field(default_factory=dict)
    pipeline_version: str = ""
    rule_version: str = ""
    model_version: str = ""
    skill_versions: Dict[str, str] = field(default_factory=dict)
    prompt_version: str = ""
    # P1 / C-4: unified 5-dimension version bundle for cross-run reuse
    version_bundle: Dict[str, str] = field(default_factory=dict)
    output_path: str = ""
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    entity_count: int = 0
    relation_count: int = 0
    llm_call_count: int = 0
    execution_duration_ms: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    quality_score: float = 0.0
    review_required: bool = False
    classification_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunManifest":
        return cls(
            run_id=data.get("run_id", ""),
            session_id=data.get("session_id", ""),
            start_time=data.get("start_time", ""),
            end_time=data.get("end_time", ""),
            status=data.get("status", "running"),
            input_files=data.get("input_files", []),
            input_file_fingerprints=data.get("input_file_fingerprints", {}),
            pipeline_version=data.get("pipeline_version", ""),
            rule_version=data.get("rule_version", ""),
            model_version=data.get("model_version", ""),
            skill_versions=data.get("skill_versions", {}),
            prompt_version=data.get("prompt_version", ""),
            version_bundle=data.get("version_bundle", {}),
            output_path=data.get("output_path", ""),
            artifact_paths=data.get("artifact_paths", {}),
            entity_count=data.get("entity_count", 0),
            relation_count=data.get("relation_count", 0),
            llm_call_count=data.get("llm_call_count", 0),
            execution_duration_ms=data.get("execution_duration_ms", 0),
            errors=data.get("errors", []),
            warnings=data.get("warnings", []),
            metadata=data.get("metadata", {}),
            stage_results=data.get("stage_results", {}),
            quality_score=data.get("quality_score", 0.0),
            review_required=data.get("review_required", False),
            classification_metrics=data.get("classification_metrics", {}),
        )


@dataclass
class ProcessedFileEntry:
    """Entry in the processed index for a single file."""
    file_path: str
    file_hash: str
    last_run_id: str
    last_status: str
    last_output_path: str
    processed_at: str
    skip_count: int = 0
    processing_duration_ms: int = 0
    stages: List[Dict[str, Any]] = field(default_factory=list)
    entity_count: int = 0
    relation_count: int = 0
    quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedFileEntry":
        return cls(
            file_path=data.get("file_path", ""),
            file_hash=data.get("file_hash", ""),
            last_run_id=data.get("last_run_id", ""),
            last_status=data.get("last_status", ""),
            last_output_path=data.get("last_output_path", ""),
            processed_at=data.get("processed_at", ""),
            skip_count=data.get("skip_count", 0),
            processing_duration_ms=data.get("processing_duration_ms", 0),
            stages=data.get("stages", []),
            entity_count=data.get("entity_count", 0),
            relation_count=data.get("relation_count", 0),
            quality_score=data.get("quality_score", 0.0),
        )


@dataclass
class ArtifactEntry:
    """Entry in the artifact index for a single artifact.

    P1 / C-4: ``version_bundle`` records the 5-dimension version key that
    produced this artifact, enabling precise cross-run reuse decisions
    without needing to load the full artifact payload.
    """
    artifact_id: str
    artifact_type: str
    run_id: str
    source_file: str
    file_path: str
    status: str = "available"
    indexed_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifact_hash: str = ""
    entity_count: int = 0
    relation_count: int = 0
    quality_score: float = 0.0
    governed: bool = False
    provenance: Dict[str, Any] = field(default_factory=dict)
    # P1 / C-4: unified 5-dimension version bundle for cross-run reuse
    version_bundle: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactEntry":
        return cls(
            artifact_id=data.get("artifact_id", ""),
            artifact_type=data.get("artifact_type", ""),
            run_id=data.get("run_id", ""),
            source_file=data.get("source_file", ""),
            file_path=data.get("file_path", ""),
            status=data.get("status", "available"),
            indexed_at=data.get("indexed_at", ""),
            metadata=data.get("metadata", {}),
            artifact_hash=data.get("artifact_hash", ""),
            entity_count=data.get("entity_count", 0),
            relation_count=data.get("relation_count", 0),
            quality_score=data.get("quality_score", 0.0),
            governed=data.get("governed", False),
            provenance=data.get("provenance", {}),
            version_bundle=data.get("version_bundle", {}),
        )


class FingerprintService:
    """Service for generating and managing file fingerprints."""

    def __init__(self, storage_path: Path | str | None = None):
        if storage_path is None:
            self.storage_path = _PROJECT_ROOT / "outputs" / "fingerprints"
        else:
            self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._fingerprints: Dict[str, ContinuityFileRecord] = {}
        self._load_fingerprints()

    def generate_fingerprint(self, file_path: Path) -> str:
        """Generate SHA-256 hash for a file."""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to generate fingerprint for {file_path}: {e}")
            return ""

    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get file metadata including type, encoding, and mime type."""
        try:
            stat = file_path.stat()
            info = {
                "file_size": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "file_type": "",
                "encoding": "",
                "mime_type": "",
            }

            ext = file_path.suffix.lower()
            info["file_type"] = ext[1:] if ext else "unknown"

            mime_map = {
                ".txt": "text/plain",
                ".md": "text/markdown",
                ".json": "application/json",
                ".yaml": "application/yaml",
                ".yml": "application/yaml",
                ".xml": "application/xml",
                ".csv": "text/csv",
                ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                ".pdf": "application/pdf",
            }
            info["mime_type"] = mime_map.get(ext, "application/octet-stream")

            if ext in [".txt", ".md", ".json", ".yaml", ".yml"]:
                info["encoding"] = self._detect_encoding(file_path)

            return info
        except Exception as e:
            logger.warning(f"Failed to get file info for {file_path}: {e}")
            return {"file_size": 0, "modified_time": "", "file_type": "", "encoding": "", "mime_type": ""}

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding."""
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read(1024)
            result = chardet.detect(raw_data)
            return result.get('encoding', 'utf-8') or 'utf-8'
        except ImportError:
            return 'utf-8'
        except Exception:
            return 'utf-8'

    def record_file(self, file_path: Path, run_id: str, language: str = "", source_domain: str = "") -> ContinuityFileRecord:
        """Record a file's fingerprint in the index."""
        file_hash = self.generate_fingerprint(file_path)
        file_info = self.get_file_info(file_path)
        file_key = str(file_path.resolve())

        now = datetime.now().isoformat()

        if file_key in self._fingerprints:
            fingerprint = self._fingerprints[file_key]
            fingerprint.file_hash = file_hash
            fingerprint.file_size = file_info["file_size"]
            fingerprint.modified_time = file_info["modified_time"]
            fingerprint.last_processed_run_id = run_id
            fingerprint.last_processed_at = now
            fingerprint.file_type = file_info.get("file_type", "")
            fingerprint.encoding = file_info.get("encoding", "")
            fingerprint.mime_type = file_info.get("mime_type", "")
            fingerprint.process_count += 1
            if language:
                fingerprint.language = language
            if source_domain:
                fingerprint.source_domain = source_domain
        else:
            fingerprint = ContinuityFileRecord(
                file_path=file_key,
                file_hash=file_hash,
                file_size=file_info["file_size"],
                modified_time=file_info["modified_time"],
                first_processed_at=now,
                last_processed_run_id=run_id,
                last_processed_at=now,
                file_type=file_info.get("file_type", ""),
                encoding=file_info.get("encoding", ""),
                mime_type=file_info.get("mime_type", ""),
                language=language,
                source_domain=source_domain,
                process_count=1,
            )
            self._fingerprints[file_key] = fingerprint

        self._save_fingerprints()
        return fingerprint

    def has_changed(self, file_path: Path) -> bool:
        """Check if a file has changed since last processing."""
        file_key = str(file_path.resolve())
        if file_key not in self._fingerprints:
            return True

        existing = self._fingerprints[file_key]
        current_hash = self.generate_fingerprint(file_path)
        current_info = self.get_file_info(file_path)

        return (
            current_hash != existing.file_hash
            or current_info["file_size"] != existing.file_size
            or current_info["modified_time"] != existing.modified_time
        )

    def get_fingerprint(self, file_path: Path) -> Optional[ContinuityFileRecord]:
        """Get the fingerprint for a file."""
        return self._fingerprints.get(str(file_path.resolve()))

    def list_fingerprints(self) -> List[ContinuityFileRecord]:
        """List all recorded fingerprints."""
        return list(self._fingerprints.values())

    def _load_fingerprints(self) -> None:
        """Load fingerprints from storage."""
        index_file = self.storage_path / "fingerprint_index.json"
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("fingerprints", []):
                        fp = ContinuityFileRecord.from_dict(item)
                        self._fingerprints[fp.file_path] = fp
                logger.info(f"Loaded {len(self._fingerprints)} fingerprints")
            except Exception as e:
                logger.warning(f"Failed to load fingerprints: {e}")

    def _save_fingerprints(self) -> None:
        """Save fingerprints to storage."""
        index_file = self.storage_path / "fingerprint_index.json"
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "fingerprints": [fp.to_dict() for fp in self._fingerprints.values()],
            }
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save fingerprints: {e}")


class RunManifestService:
    """Service for managing run manifests."""

    def __init__(self, storage_path: Path | str | None = None):
        if storage_path is None:
            self.storage_path = _PROJECT_ROOT / "outputs" / "manifests"
        else:
            self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._manifests: Dict[str, RunManifest] = {}
        self._load_manifests()

    def create_manifest(
        self,
        run_id: str,
        session_id: str,
        input_files: List[str],
        pipeline_version: str = "",
        rule_version: str = "",
        model_version: str = "",
        skill_versions: Optional[Dict[str, str]] = None,
        prompt_version: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RunManifest:
        """Create a new run manifest."""
        now = datetime.now().isoformat()

        fingerprint_service = FingerprintService()
        fingerprints: Dict[str, str] = {}
        for file_path in input_files:
            fp = fingerprint_service.get_fingerprint(Path(file_path))
            if fp:
                fingerprints[file_path] = fp.file_hash

        manifest = RunManifest(
            run_id=run_id,
            session_id=session_id,
            start_time=now,
            input_files=input_files,
            input_file_fingerprints=fingerprints,
            pipeline_version=pipeline_version,
            rule_version=rule_version,
            model_version=model_version,
            skill_versions=skill_versions or {},
            prompt_version=prompt_version,
            metadata=metadata or {},
        )

        self._manifests[run_id] = manifest
        self._save_manifest(manifest)
        return manifest

    def update_manifest(
        self,
        run_id: str,
        status: Optional[str] = None,
        end_time: Optional[str] = None,
        output_path: Optional[str] = None,
        artifact_paths: Optional[Dict[str, str]] = None,
        entity_count: Optional[int] = None,
        relation_count: Optional[int] = None,
        llm_call_count: Optional[int] = None,
        execution_duration_ms: Optional[int] = None,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        skill_versions: Optional[Dict[str, str]] = None,
        stage_results: Optional[Dict[str, Dict[str, Any]]] = None,
        quality_score: Optional[float] = None,
        review_required: Optional[bool] = None,
        classification_metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[RunManifest]:
        """Update a run manifest."""
        manifest = self._manifests.get(run_id)
        if not manifest:
            return None

        if status is not None:
            manifest.status = status
        if end_time is not None:
            manifest.end_time = end_time
        if output_path is not None:
            manifest.output_path = output_path
        if artifact_paths is not None:
            manifest.artifact_paths.update(artifact_paths)
        if entity_count is not None:
            manifest.entity_count = entity_count
        if relation_count is not None:
            manifest.relation_count = relation_count
        if llm_call_count is not None:
            manifest.llm_call_count = llm_call_count
        if execution_duration_ms is not None:
            manifest.execution_duration_ms = execution_duration_ms
        if errors is not None:
            manifest.errors.extend(errors)
        if warnings is not None:
            manifest.warnings.extend(warnings)
        if skill_versions is not None:
            manifest.skill_versions.update(skill_versions)
        if stage_results is not None:
            manifest.stage_results.update(stage_results)
        if quality_score is not None:
            manifest.quality_score = quality_score
        if review_required is not None:
            manifest.review_required = review_required
        if classification_metrics is not None:
            manifest.classification_metrics = classification_metrics
            manifest.metadata["classification_metrics"] = classification_metrics

        self._save_manifest(manifest)
        return manifest

    def get_manifest(self, run_id: str) -> Optional[RunManifest]:
        """Get a run manifest by ID."""
        return self._manifests.get(run_id)

    def list_manifests(self) -> List[RunManifest]:
        """List all run manifests, sorted by time."""
        return sorted(
            self._manifests.values(),
            key=lambda m: m.start_time,
            reverse=True,
        )

    def get_run_history(self, file_path: str) -> List[RunManifest]:
        """Get all runs that processed a specific file."""
        results = []
        for manifest in self._manifests.values():
            if file_path in manifest.input_files:
                results.append(manifest)
        return sorted(results, key=lambda m: m.start_time, reverse=True)

    def _load_manifests(self) -> None:
        """Load manifests from storage."""
        for manifest_file in self.storage_path.glob("manifest-*.json"):
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    manifest = RunManifest.from_dict(data)
                    self._manifests[manifest.run_id] = manifest
            except Exception as e:
                logger.warning(f"Failed to load manifest {manifest_file}: {e}")
        logger.info(f"Loaded {len(self._manifests)} manifests")

    def _save_manifest(self, manifest: RunManifest) -> None:
        """Save a manifest to storage."""
        manifest_file = self.storage_path / f"manifest-{manifest.run_id}.json"
        try:
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save manifest {manifest.run_id}: {e}")


class ProcessedIndexService:
    """Service for tracking processed files."""

    def __init__(self, storage_path: Path | str | None = None):
        if storage_path is None:
            self.storage_path = _PROJECT_ROOT / "outputs" / "processed_index"
        else:
            self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, ProcessedFileEntry] = {}
        self._load_entries()

    def record_processed(
        self,
        file_path: Path,
        run_id: str,
        status: str,
        output_path: str,
        processing_duration_ms: int = 0,
        stages: Optional[List[Dict[str, Any]]] = None,
        entity_count: int = 0,
        relation_count: int = 0,
        quality_score: float = 0.0,
    ) -> ProcessedFileEntry:
        """Record that a file has been processed."""
        file_key = str(file_path.resolve())
        fingerprint_service = FingerprintService()
        file_hash = fingerprint_service.generate_fingerprint(file_path)

        now = datetime.now().isoformat()

        if file_key in self._entries:
            entry = self._entries[file_key]
            entry.file_hash = file_hash
            entry.last_run_id = run_id
            entry.last_status = status
            entry.last_output_path = output_path
            entry.processed_at = now
            entry.processing_duration_ms = processing_duration_ms
            if stages:
                entry.stages = stages
            entry.entity_count = entity_count
            entry.relation_count = relation_count
            entry.quality_score = quality_score
        else:
            entry = ProcessedFileEntry(
                file_path=file_key,
                file_hash=file_hash,
                last_run_id=run_id,
                last_status=status,
                last_output_path=output_path,
                processed_at=now,
                processing_duration_ms=processing_duration_ms,
                stages=stages or [],
                entity_count=entity_count,
                relation_count=relation_count,
                quality_score=quality_score,
            )
            self._entries[file_key] = entry

        self._save_entries()
        return entry

    def record_skipped(self, file_path: Path) -> ProcessedFileEntry:
        """Record that a file was skipped."""
        file_key = str(file_path.resolve())

        if file_key in self._entries:
            entry = self._entries[file_key]
            entry.skip_count += 1
        else:
            fingerprint_service = FingerprintService()
            file_hash = fingerprint_service.generate_fingerprint(file_path)
            entry = ProcessedFileEntry(
                file_path=file_key,
                file_hash=file_hash,
                last_run_id="",
                last_status="skipped",
                last_output_path="",
                processed_at=datetime.now().isoformat(),
                skip_count=1,
            )
            self._entries[file_key] = entry

        self._save_entries()
        return entry

    def should_process(self, file_path: Path, force: bool = False) -> bool:
        """Determine if a file should be processed."""
        if force:
            return True

        file_key = str(file_path.resolve())
        if file_key not in self._entries:
            return True

        entry = self._entries[file_key]
        fingerprint_service = FingerprintService()

        if fingerprint_service.has_changed(file_path):
            return True

        return False

    def get_entry(self, file_path: Path) -> Optional[ProcessedFileEntry]:
        """Get the processed entry for a file."""
        return self._entries.get(str(file_path.resolve()))

    def list_entries(self) -> List[ProcessedFileEntry]:
        """List all processed entries."""
        return list(self._entries.values())

    def get_skipped_count(self) -> int:
        """Get total number of skipped files across all runs."""
        return sum(entry.skip_count for entry in self._entries.values())

    def _load_entries(self) -> None:
        """Load entries from storage."""
        index_file = self.storage_path / "processed_index.json"
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("entries", []):
                        entry = ProcessedFileEntry.from_dict(item)
                        self._entries[entry.file_path] = entry
                logger.info(f"Loaded {len(self._entries)} processed entries")
            except Exception as e:
                logger.warning(f"Failed to load processed index: {e}")

    def _save_entries(self) -> None:
        """Save entries to storage."""
        index_file = self.storage_path / "processed_index.json"
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "entries": [entry.to_dict() for entry in self._entries.values()],
                "total_files": len(self._entries),
                "total_skips": self.get_skipped_count(),
            }
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save processed index: {e}")


class ArtifactIndexService:
    """Service for managing generated artifacts."""

    def __init__(self, storage_path: Path | str | None = None):
        if storage_path is None:
            self.storage_path = _PROJECT_ROOT / "outputs" / "artifact_index"
        else:
            self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, ArtifactEntry] = {}
        self._load_entries()

    def register_artifact(
        self,
        artifact_id: str,
        artifact_type: str,
        run_id: str,
        source_file: str,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        artifact_hash: str = "",
        entity_count: int = 0,
        relation_count: int = 0,
        quality_score: float = 0.0,
        governed: bool = False,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> ArtifactEntry:
        """Register a new artifact."""
        now = datetime.now().isoformat()

        if not artifact_hash and file_path:
            fingerprint_service = FingerprintService()
            artifact_hash = fingerprint_service.generate_fingerprint(Path(file_path))

        entry = ArtifactEntry(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            run_id=run_id,
            source_file=source_file,
            file_path=file_path,
            status="available",
            indexed_at=now,
            metadata=metadata or {},
            artifact_hash=artifact_hash,
            entity_count=entity_count,
            relation_count=relation_count,
            quality_score=quality_score,
            governed=governed,
            provenance=provenance or {},
        )

        self._entries[artifact_id] = entry
        self._save_entries()
        return entry

    def update_artifact(
        self,
        artifact_id: str,
        status: Optional[str] = None,
        quality_score: Optional[float] = None,
        governed: Optional[bool] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> Optional[ArtifactEntry]:
        """Update artifact metadata."""
        entry = self._entries.get(artifact_id)
        if not entry:
            return None

        if status is not None:
            entry.status = status
        if quality_score is not None:
            entry.quality_score = quality_score
        if governed is not None:
            entry.governed = governed
        if provenance is not None:
            entry.provenance.update(provenance)

        self._save_entries()
        return entry

    def get_governed_artifacts(self) -> List[ArtifactEntry]:
        """Get all governed artifacts."""
        return [e for e in self._entries.values() if e.governed]

    def get_artifacts_by_run(self, run_id: str) -> List[ArtifactEntry]:
        """Get all artifacts from a specific run."""
        return [e for e in self._entries.values() if e.run_id == run_id]

    def get_artifacts_by_source(self, source_file: str) -> List[ArtifactEntry]:
        """Get all artifacts derived from a specific source file."""
        return [e for e in self._entries.values() if e.source_file == source_file]

    def get_artifacts_by_type(self, artifact_type: str) -> List[ArtifactEntry]:
        """Get all artifacts of a specific type."""
        return [e for e in self._entries.values() if e.artifact_type == artifact_type]

    def list_artifacts(self) -> List[ArtifactEntry]:
        """List all registered artifacts."""
        return list(self._entries.values())

    def _load_entries(self) -> None:
        """Load entries from storage."""
        index_file = self.storage_path / "artifact_index.json"
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("entries", []):
                        entry = ArtifactEntry.from_dict(item)
                        self._entries[entry.artifact_id] = entry
                logger.info(f"Loaded {len(self._entries)} artifact entries")
            except Exception as e:
                logger.warning(f"Failed to load artifact index: {e}")

    def _save_entries(self) -> None:
        """Save entries to storage."""
        index_file = self.storage_path / "artifact_index.json"
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "entries": [entry.to_dict() for entry in self._entries.values()],
                "total_artifacts": len(self._entries),
                "types": {t: len([e for e in self._entries.values() if e.artifact_type == t]) for t in set(e.artifact_type for e in self._entries.values())},
            }
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save artifact index: {e}")


class DataContinuityService:
    """Unified service for all data continuity operations."""

    def __init__(self):
        self.fingerprint_service = FingerprintService()
        self.manifest_service = RunManifestService()
        self.processed_index = ProcessedIndexService()
        self.artifact_index = ArtifactIndexService()

    def begin_run(
        self,
        run_id: str,
        session_id: str,
        input_files: List[str],
        pipeline_version: str = "",
        rule_version: str = "",
        model_version: str = "",
        skill_versions: Optional[Dict[str, str]] = None,
        prompt_version: str = "",
    ) -> RunManifest:
        """Begin a new run and create manifest."""
        for file_path in input_files:
            self.fingerprint_service.record_file(Path(file_path), run_id)

        return self.manifest_service.create_manifest(
            run_id=run_id,
            session_id=session_id,
            input_files=input_files,
            pipeline_version=pipeline_version,
            rule_version=rule_version,
            model_version=model_version,
            skill_versions=skill_versions,
            prompt_version=prompt_version,
        )

    def end_run(
        self,
        run_id: str,
        status: str,
        output_path: str,
        entity_count: int = 0,
        relation_count: int = 0,
        llm_call_count: int = 0,
        execution_duration_ms: int = 0,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        artifact_paths: Optional[Dict[str, str]] = None,
        skill_versions: Optional[Dict[str, str]] = None,
        stage_results: Optional[Dict[str, Dict[str, Any]]] = None,
        quality_score: float = 0.0,
        review_required: bool = False,
        classification_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """End a run and update manifest."""
        end_time = datetime.now().isoformat()

        self.manifest_service.update_manifest(
            run_id=run_id,
            status=status,
            end_time=end_time,
            output_path=output_path,
            entity_count=entity_count,
            relation_count=relation_count,
            llm_call_count=llm_call_count,
            execution_duration_ms=execution_duration_ms,
            errors=errors or [],
            warnings=warnings or [],
            artifact_paths=artifact_paths or {},
            skill_versions=skill_versions,
            stage_results=stage_results,
            quality_score=quality_score,
            review_required=review_required,
            classification_metrics=classification_metrics,
        )

    def should_process_file(self, file_path: Path, force: bool = False) -> bool:
        """Check if a file should be processed."""
        return self.processed_index.should_process(file_path, force)

    def record_file_processed(
        self,
        file_path: Path,
        run_id: str,
        status: str,
        output_path: str,
    ) -> None:
        """Record that a file was processed."""
        self.processed_index.record_processed(file_path, run_id, status, output_path)

    def record_file_skipped(self, file_path: Path) -> None:
        """Record that a file was skipped."""
        self.processed_index.record_skipped(file_path)

    def register_artifact(
        self,
        artifact_id: str,
        artifact_type: str,
        run_id: str,
        source_file: str,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        artifact_hash: str = "",
        entity_count: int = 0,
        relation_count: int = 0,
        quality_score: float = 0.0,
        governed: bool = False,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register an artifact."""
        self.artifact_index.register_artifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            run_id=run_id,
            source_file=source_file,
            file_path=file_path,
            metadata=metadata,
            artifact_hash=artifact_hash,
            entity_count=entity_count,
            relation_count=relation_count,
            quality_score=quality_score,
            governed=governed,
            provenance=provenance,
        )

    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Get a summary of a run."""
        manifest = self.manifest_service.get_manifest(run_id)
        if not manifest:
            return {"error": f"Run {run_id} not found"}

        artifacts = self.artifact_index.get_artifacts_by_run(run_id)

        return {
            "run_id": manifest.run_id,
            "session_id": manifest.session_id,
            "status": manifest.status,
            "start_time": manifest.start_time,
            "end_time": manifest.end_time,
            "input_files": manifest.input_files,
            "entity_count": manifest.entity_count,
            "relation_count": manifest.relation_count,
            "llm_call_count": manifest.llm_call_count,
            "artifact_count": len(artifacts),
            "warnings": manifest.warnings,
            "errors": manifest.errors,
        }

    def get_file_history(self, file_path: str) -> List[Dict[str, Any]]:
        """Get processing history for a file."""
        history = []
        runs = self.manifest_service.get_run_history(file_path)
        for manifest in runs:
            history.append({
                "run_id": manifest.run_id,
                "status": manifest.status,
                "processed_at": manifest.start_time,
                "entity_count": manifest.entity_count,
                "relation_count": manifest.relation_count,
            })
        return history


# Global service instance
_global_continuity_service: Optional[DataContinuityService] = None


def get_continuity_service() -> DataContinuityService:
    """Get or create global data continuity service instance."""
    global _global_continuity_service
    if _global_continuity_service is None:
        _global_continuity_service = DataContinuityService()
    return _global_continuity_service


__all__ = [
    "ContinuityFileRecord",
    "RunManifest",
    "ProcessedFileEntry",
    "ArtifactEntry",
    "FingerprintService",
    "RunManifestService",
    "ProcessedIndexService",
    "ArtifactIndexService",
    "DataContinuityService",
    "get_continuity_service",
]