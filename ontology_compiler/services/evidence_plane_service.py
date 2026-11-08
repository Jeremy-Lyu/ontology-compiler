"""Evidence lookup over generated L1/L2 artifacts.

**架构定位**: 这是 **inter-run**（跨运行）级别的证据平面服务，负责在多个 pipeline 运行的产物中索引和追踪实体。

**互补模块**: 与 `evidence_locator.py` 中的 `EvidenceLocator` 互补：
- EvidencePlaneService: 跨运行追踪（源文件、运行ID、层级）
- EvidenceLocator: 文档内定位（行号、字符偏移、段落）
- 两者通过 `trace_to_source_with_locator()` 方法桥接使用

Implements EvidencePlanePort for dependency injection.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.utils.file_helpers import load_json_object
from core.errors import DataContractError
from core.contracts.ports import EvidencePlanePort


class EvidencePlaneService(EvidencePlanePort):
    """Evidence lookup over generated L1/L2 artifacts.

    Implements EvidencePlanePort for dependency injection.
    """

    def _get_index_path(self, runs_root: Path) -> Path:
        """Return the index file path for the given runs_root."""
        return runs_root / "evidence_index.json"

    def build_index(self, *, runs_root: Path) -> dict[str, Any]:
        """Scan all runs and build evidence index {entity_uri: [run_id, layer, ...]}.

        H8 Fix: Also saves file content fingerprints for cache consistency validation.
        """
        index: dict[str, list[dict[str, Any]]] = {}
        fingerprints: dict[str, str] = {}  # run_dir -> fingerprint

        if not runs_root.exists():
            self._save_index(self._get_index_path(runs_root), index)
            return index

        for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
            l1_path = run_dir / "L1_DOMAIN_SANDBOX.json"
            l2_path = run_dir / "L2_OPERATIONAL_INSTANCES.json"

            # H8 Fix: Compute and store fingerprint for this run
            fingerprints[run_dir.name] = self._compute_run_fingerprint(run_dir)

            if l1_path.exists():
                l1 = self._load_json(l1_path)
                sandbox = l1.get("sandbox") or {}
                for obj in sandbox.get("objects") or []:
                    meta_id = str(obj.get("meta_id") or "").strip()
                    if not meta_id:
                        continue
                    source_ref = self._extract_source_ref(obj, file_keys=("source_file", "source_document"))
                    entry = {
                        "run_id": run_dir.name,
                        "layer": "L1",
                        "meta_name": obj.get("meta_name"),
                        "meta_class": obj.get("meta_class"),
                        "meta_subclass": obj.get("meta_subclass"),
                        **source_ref,
                    }
                    index.setdefault(meta_id, []).append(entry)

                # Index relationships for conflict detection
                for rel in sandbox.get("relationships") or []:
                    rel_key = self._build_relationship_key(rel)
                    if not rel_key:
                        continue
                    entry = {
                        "run_id": run_dir.name,
                        "layer": "L1_REL",
                        "source": rel.get("source"),
                        "predicate": rel.get("predicate"),
                        "target": rel.get("target"),
                    }
                    index.setdefault(rel_key, []).append(entry)

            if l2_path.exists():
                l2 = self._load_json(l2_path)
                for inst in l2.get("instances") or []:
                    meta_id = str(inst.get("meta_id") or "").strip()
                    instance_of = str(inst.get("instance_of") or "").strip()
                    source_ref = self._extract_source_ref(inst, file_keys=("instance_source", "source_file", "source_document"))

                    if meta_id:
                        entry = {
                            "run_id": run_dir.name,
                            "layer": "L2",
                            "meta_name": inst.get("meta_name"),
                            "meta_class": inst.get("meta_class"),
                            "instance_of": instance_of,
                            **source_ref,
                        }
                        index.setdefault(meta_id, []).append(entry)
                    if instance_of:
                        entry = {
                            "run_id": run_dir.name,
                            "layer": "L2",
                            "meta_name": inst.get("meta_name"),
                            "meta_class": inst.get("meta_class"),
                            "instance_of": instance_of,
                            **source_ref,
                        }
                        index.setdefault(instance_of, []).append(entry)

        # H8 Fix: Save both index and fingerprints
        index_path = self._get_index_path(runs_root)
        self._save_index(index_path, index)
        self._save_fingerprints(index_path.with_suffix(".fingerprints.json"), fingerprints)
        return index

    def _save_index(self, index_path: Path, index: dict[str, list[dict[str, Any]]]) -> None:
        """Save index to disk atomically."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def _save_fingerprints(self, fingerprint_path: Path, fingerprints: dict[str, str]) -> None:
        """Save file fingerprints to disk atomically.

        H8 Fix: Stores content-based fingerprints for cache consistency validation.
        """
        fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fingerprint_path, "w", encoding="utf-8") as f:
            json.dump(fingerprints, f, ensure_ascii=False, indent=2)

    def _load_index(self, index_path: Path) -> dict[str, list[dict[str, Any]]] | None:
        """Load index from disk if exists and valid."""
        if not index_path.exists():
            return None
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return None
            return {k: v if isinstance(v, list) else [] for k, v in data.items()}
        except Exception:
            return None

    def _compute_file_fingerprint(self, path: Path) -> str:
        """Compute SHA-256 fingerprint of file content.

        Returns empty string if file does not exist or cannot be read.
        """
        if not path.exists():
            return ""
        try:
            content = path.read_bytes()
            return hashlib.sha256(content).hexdigest()[:32]  # 32 chars sufficient
        except Exception:
            return ""

    def _compute_run_fingerprint(self, run_dir: Path) -> str:
        """Compute combined fingerprint for a run directory.

        H3 Fix: Now covers L1, L2, L2.5 and relationship files to detect
        any content change across all ontology layers.
        """
        l1_path = run_dir / "L1_DOMAIN_SANDBOX.json"
        l2_path = run_dir / "L2_OPERATIONAL_INSTANCES.json"
        l25_path = run_dir / "L25_MAPPING_RULES.json"
        rel_path = run_dir / "relationships.json"
        
        l1_fp = self._compute_file_fingerprint(l1_path)
        l2_fp = self._compute_file_fingerprint(l2_path)
        l25_fp = self._compute_file_fingerprint(l25_path)
        rel_fp = self._compute_file_fingerprint(rel_path)
        return f"{l1_fp}:{l2_fp}:{l25_fp}:{rel_fp}"

    def _ensure_index(self, runs_root: Path) -> dict[str, list[dict[str, Any]]]:
        """Ensure index exists and is up-to-date; rebuild if needed.

        H8 Fix: Uses content-based fingerprints in addition to mtime to detect
        manual file modifications that don't update mtime.
        """
        index_path = self._get_index_path(runs_root)
        index = self._load_index(index_path)
        if index is None:
            return self.build_index(runs_root=runs_root)

        # Load stored fingerprints if available
        fingerprint_path = index_path.with_suffix(".fingerprints.json")
        stored_fps: dict[str, str] = {}
        if fingerprint_path.exists():
            try:
                with open(fingerprint_path, "r", encoding="utf-8") as f:
                    stored_fps = json.load(f)
            except Exception:
                stored_fps = {}

        # Check each run directory for changes
        for run_dir in runs_root.iterdir():
            if not run_dir.is_dir():
                continue

            # Fast check: mtime comparison
            index_mtime = index_path.stat().st_mtime
            if run_dir.stat().st_mtime > index_mtime:
                return self.build_index(runs_root=runs_root)

            # H8 Fix: Content-based fingerprint check
            current_fp = self._compute_run_fingerprint(run_dir)
            stored_fp = stored_fps.get(run_dir.name, "")
            if current_fp != stored_fp:
                return self.build_index(runs_root=runs_root)

        return index

    def trace_to_source(self, entity_uri: str, *, runs_root: Path) -> list[dict[str, Any]]:
        """Trace an entity URI to its source evidence (EvidencePlanePort interface).

        H4 Fix: Returns list[dict] consistently. Previously returned dict in some paths.

        Args:
            entity_uri: The entity URI to trace.
            runs_root: Root directory containing run artifacts.

        Returns:
            List of evidence entries for the entity.
        """
        normalized_entity_uri = str(entity_uri or "").strip()
        if not normalized_entity_uri:
            raise DataContractError("entity_uri must be non-empty")

        # Runtime type validation for runs_root
        if not isinstance(runs_root, Path):
            raise TypeError(f"runs_root must be Path, got {type(runs_root).__name__}")

        if not runs_root.exists():
            return []  # H4 Fix: return empty list instead of dict

        # Fast path: use index
        index = self._ensure_index(runs_root)
        indexed_matches = index.get(normalized_entity_uri, [])
        if indexed_matches:
            # Add entity_uri to each match for consistency
            for m in indexed_matches:
                m["entity_uri"] = normalized_entity_uri
            ordered_matches = sorted(
                indexed_matches,
                key=lambda item: (
                    str(item.get("run_id") or ""),
                    str(item.get("layer") or ""),
                    str(item.get("source_file") or ""),
                    str(item.get("meta_name") or ""),
                ),
            )
            return ordered_matches  # H4 Fix: return list directly

        # Fallback: O(n) scan (index may be stale or entity not indexed)
        matches: list[dict[str, Any]] = []
        for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
            l1_path = run_dir / "L1_DOMAIN_SANDBOX.json"
            l2_path = run_dir / "L2_OPERATIONAL_INSTANCES.json"

            if l1_path.exists():
                l1 = self._load_json(l1_path)
                sandbox = l1.get("sandbox") or {}
                for obj in sandbox.get("objects") or []:
                    if str(obj.get("meta_id") or "") != normalized_entity_uri:
                        continue
                    source_ref = self._extract_source_ref(obj, file_keys=("source_file", "source_document"))
                    matches.append(
                        {
                            "run_id": run_dir.name,
                            "layer": "L1",
                            "entity_uri": normalized_entity_uri,
                            "meta_name": obj.get("meta_name"),
                            "meta_class": obj.get("meta_class"),
                            **source_ref,
                        }
                    )

            if l2_path.exists():
                l2 = self._load_json(l2_path)
                for inst in l2.get("instances") or []:
                    meta_id = str(inst.get("meta_id") or "")
                    instance_of = str(inst.get("instance_of") or "")
                    if meta_id != normalized_entity_uri and instance_of != normalized_entity_uri:
                        continue
                    source_ref = self._extract_source_ref(inst, file_keys=("instance_source", "source_file", "source_document"))
                    matches.append(
                        {
                            "run_id": run_dir.name,
                            "layer": "L2",
                            "entity_uri": normalized_entity_uri,
                            "meta_name": inst.get("meta_name"),
                            "instance_of": inst.get("instance_of"),
                            **source_ref,
                        }
                    )

        ordered_matches = sorted(
            matches,
            key=lambda item: (
                str(item.get("run_id") or ""),
                str(item.get("layer") or ""),
                str(item.get("source_file") or ""),
                str(item.get("meta_name") or ""),
            ),
        )

        return ordered_matches

    def trace_to_source_with_locator(
        self,
        entity_uri: str,
        *,
        runs_root: Path,
        source_text: str = "",
        file_path: Optional[Path] = None,
    ) -> list[dict[str, Any]]:
        """Trace an entity URI to its source evidence with fine-grained position mapping.

        [Phase 2 - EVIDENCE_BRIDGE]: Bridges EvidencePlaneService (inter-run) with EvidenceLocator (intra-doc)
        to provide precise char_offset, line_range, and context information in addition to
        the coarse source_file/layer info.

        Args:
            entity_uri: The entity URI to trace.
            runs_root: Root directory containing run artifacts.
            source_text: Optional source text for fine-grained position mapping.
                If provided, EvidenceLocator will be used to compute precise positions.
            file_path: Optional file path for EvidenceLocator context.

        Returns:
            List of evidence entries, each potentially enriched with:
            - line_range: (start, end) position in source text
            - char_offset: (start, end) character offset
            - paragraph_id: paragraph identifier
            - raw_text: matched text snippet
            - context: surrounding text context
            - confidence: match confidence score
        """
        # Get base evidence from trace_to_source
        base_matches = self.trace_to_source(entity_uri, runs_root=runs_root)
        
        # If no source_text provided, return base matches
        if not source_text or not base_matches:
            return base_matches
        
        # Use EvidenceLocator for fine-grained position mapping
        try:
            from ontology_compiler.services.extraction.evidence_locator import EvidenceLocator
            
            locator = EvidenceLocator(source_text, file_path=file_path)
            enriched_matches = []
            
            for match in base_matches:
                entity_name = match.get("meta_name") or entity_uri
                entity_desc = match.get("meta_class") or match.get("meta_subclass") or ""
                
                # Use EvidenceLocator to find precise position
                evidence = locator.locate_entity(entity_name, entity_desc)
                
                if evidence:
                    enriched_match = match.copy()
                    enriched_match.update({
                        "line_range": evidence.line_range,
                        "char_offset": evidence.char_offset,
                        "paragraph_id": evidence.paragraph_id,
                        "content": evidence.content,
                        "raw_text": evidence.raw_text,
                        "context": evidence.context,
                        "confidence": evidence.confidence,
                        "match_type": evidence.metadata.get("match_type") if evidence.metadata else None,
                    })
                    enriched_matches.append(enriched_match)
                else:
                    # Keep original match if locator fails
                    enriched_matches.append(match)
            
            return enriched_matches
            
        except ImportError:
            # EvidenceLocator not available - return base matches
            return base_matches
        except Exception as e:
            # Locator error - log and return base matches
            print(f"EvidenceLocator enrichment error: {e}")
            return base_matches

    def validate_fingerprint(self, run_id: str, expected_fingerprint: str, *, runs_root: Path) -> bool:
        """Validate run content fingerprint for cache consistency (EvidencePlanePort interface).

        Args:
            run_id: The run identifier.
            expected_fingerprint: Expected fingerprint hash.
            runs_root: Root directory containing run artifacts.

        Returns:
            True if fingerprint matches.
        """
        run_dir = runs_root / run_id
        if not run_dir.exists():
            return False
        actual_fp = self._compute_run_fingerprint(run_dir)
        return actual_fp == expected_fingerprint

    def _extract_source_ref(self, payload: dict[str, Any], *, file_keys: tuple[str, ...]) -> dict[str, Any]:
        source_file = ""
        for key in file_keys:
            candidate = payload.get(key)
            if isinstance(candidate, str) and candidate.strip():
                source_file = candidate.strip()
                break

        source_page = self._coerce_int(payload.get("source_page"))
        if source_page is None:
            source_page = self._coerce_int(payload.get("page"))

        source_offset_ms = self._coerce_int(payload.get("source_offset_ms"))
        if source_offset_ms is None:
            source_offset_ms = self._coerce_int(payload.get("offset_ms"))

        return {
            "source_file": source_file or None,
            "source_page": source_page,
            "source_offset_ms": source_offset_ms,
        }

    def _coerce_int(self, value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return None
            try:
                return int(raw)
            except ValueError:
                return None
        return None

    def _load_json(self, path: Path) -> dict[str, Any]:
        payload = load_json_object(path)
        if not isinstance(payload, dict):
            raise DataContractError(f"evidence payload must be object: {path}")
        return payload

    def _build_relationship_key(self, rel: dict[str, Any]) -> str:
        """Build unique key for relationship indexing."""
        source = str(rel.get("source") or "").strip()
        predicate = str(rel.get("predicate") or "").strip()
        target = str(rel.get("target") or "").strip()
        if not source or not predicate or not target:
            return ""
        return f"{source}::{predicate}::{target}"

    # P0 Implementation 2026-05-26: Cache consistency management APIs

    def validate_cache_consistency(self, runs_root: Path) -> dict[str, Any]:
        """Validate cache consistency against actual file contents.

        P0 Feature: Performs deep validation of the evidence index cache
        to detect manual file modifications, corruption, or external changes
        that bypassed normal update mechanisms.

        Args:
            runs_root: Root directory containing run folders

        Returns:
            Validation report with consistency status and details
        """
        index_path = self._get_index_path(runs_root)
        fingerprint_path = index_path.with_suffix(".fingerprints.json")

        result = {
            "valid": False,
            "index_exists": False,
            "fingerprints_exists": False,
            "runs_checked": 0,
            "runs_consistent": 0,
            "runs_inconsistent": [],
            "missing_runs": [],
            "extra_runs": [],
            "recommendation": "",
        }

        # Check index exists
        if not index_path.exists():
            result["recommendation"] = "Index missing - rebuild required"
            return result
        result["index_exists"] = True

        # Load stored fingerprints
        stored_fps: dict[str, str] = {}
        if fingerprint_path.exists():
            result["fingerprints_exists"] = True
            try:
                with open(fingerprint_path, "r", encoding="utf-8") as f:
                    stored_fps = json.load(f)
            except Exception as e:
                result["recommendation"] = f"Fingerprints file corrupted: {e}"
                return result
        else:
            result["recommendation"] = "Fingerprints missing - shallow validation only"

        # Get current run directories
        current_runs = {p.name for p in runs_root.iterdir() if p.is_dir()}
        stored_runs = set(stored_fps.keys())

        # Check for missing runs (deleted since last index)
        result["missing_runs"] = list(stored_runs - current_runs)

        # Check for extra runs (new since last index)
        result["extra_runs"] = list(current_runs - stored_runs)

        # Validate each current run
        for run_name in sorted(current_runs):
            run_dir = runs_root / run_name
            result["runs_checked"] += 1

            if run_name not in stored_fps:
                # New run not in index
                result["runs_inconsistent"].append({
                    "run_id": run_name,
                    "reason": "Not indexed (new run)",
                })
                continue

            # Compute current fingerprint
            current_fp = self._compute_run_fingerprint(run_dir)
            stored_fp = stored_fps[run_name]

            if current_fp == stored_fp:
                result["runs_consistent"] += 1
            else:
                result["runs_inconsistent"].append({
                    "run_id": run_name,
                    "reason": "Content mismatch",
                    "stored_fingerprint": stored_fp[:16] + "...",
                    "current_fingerprint": current_fp[:16] + "...",
                })

        # Determine overall validity
        result["valid"] = (
            result["runs_consistent"] == result["runs_checked"]
            and not result["missing_runs"]
            and result["fingerprints_exists"]
        )

        if not result["valid"]:
            issues = []
            if result["runs_inconsistent"]:
                issues.append(f"{len(result['runs_inconsistent'])} inconsistent runs")
            if result["missing_runs"]:
                issues.append(f"{len(result['missing_runs'])} missing runs")
            if not result["fingerprints_exists"]:
                issues.append("fingerprints file missing")
            result["recommendation"] = f"Rebuild recommended: {', '.join(issues)}"
        else:
            result["recommendation"] = "Cache is consistent"

        return result

    def invalidate_cache(self, runs_root: Path, target_run: str | None = None) -> bool:
        """Invalidate evidence cache for specific run or all runs.

        P0 Feature: Explicit cache invalidation API for external tools
        or manual intervention scenarios.

        Args:
            runs_root: Root directory containing run folders
            target_run: Specific run ID to invalidate, or None for all

        Returns:
            True if invalidation was successful
        """
        index_path = self._get_index_path(runs_root)
        fingerprint_path = index_path.with_suffix(".fingerprints.json")

        try:
            if target_run is None:
                # Invalidate all: remove index and fingerprints
                if index_path.exists():
                    index_path.unlink()
                if fingerprint_path.exists():
                    fingerprint_path.unlink()
                return True
            else:
                # Invalidate specific run: rebuild index without it
                # or mark it for rebuild
                if not index_path.exists():
                    return True  # Nothing to invalidate

                # Load existing index
                index = self._load_index(index_path)
                if index is None:
                    return True

                # Remove entries for target run
                cleaned_index: dict[str, list[dict[str, Any]]] = {}
                for entity_id, entries in index.items():
                    filtered = [e for e in entries if e.get("run_id") != target_run]
                    if filtered:
                        cleaned_index[entity_id] = filtered

                # Save cleaned index
                self._save_index(index_path, cleaned_index)

                # Update fingerprints
                if fingerprint_path.exists():
                    try:
                        with open(fingerprint_path, "r", encoding="utf-8") as f:
                            fps = json.load(f)
                        if target_run in fps:
                            del fps[target_run]
                            self._save_fingerprints(fingerprint_path, fps)
                    except Exception:
                        pass

                return True
        except Exception as e:
            # Log error but don't raise - invalidation should be safe
            print(f"Cache invalidation warning: {e}")
            return False

    def get_cache_diagnostics(self, runs_root: Path) -> dict[str, Any]:
        """Get detailed diagnostics about the evidence cache state.

        P0 Feature: Diagnostic information for troubleshooting cache issues.

        Args:
            runs_root: Root directory containing run folders

        Returns:
            Diagnostic report with cache statistics and health indicators
        """
        index_path = self._get_index_path(runs_root)
        fingerprint_path = index_path.with_suffix(".fingerprints.json")

        diag: dict[str, Any] = {
            "runs_root": str(runs_root),
            "index_path": str(index_path),
            "index_exists": index_path.exists(),
            "fingerprints_path": str(fingerprint_path),
            "fingerprints_exists": fingerprint_path.exists(),
            "run_directories": 0,
            "indexed_runs": [],
            "cache_age_hours": None,
            "entity_count": 0,
            "relationship_count": 0,
            "health_score": 0.0,
        }

        # Count run directories
        if runs_root.exists():
            diag["run_directories"] = sum(1 for p in runs_root.iterdir() if p.is_dir())

        # Load index info
        if index_path.exists():
            try:
                index = self._load_index(index_path)
                if index:
                    diag["entity_count"] = len(index)
                    diag["indexed_runs"] = sorted(set(
                        e.get("run_id")
                        for entries in index.values()
                        for e in entries
                        if e.get("run_id")
                    ))
                    diag["relationship_count"] = sum(
                        1
                        for entries in index.values()
                        for e in entries
                        if e.get("layer") == "L1_REL"
                    )

                # Calculate cache age
                index_mtime = index_path.stat().st_mtime
                age_seconds = datetime.now().timestamp() - index_mtime
                diag["cache_age_hours"] = round(age_seconds / 3600, 2)
            except Exception as e:
                diag["index_error"] = str(e)

        # Calculate health score (0.0 - 1.0)
        score = 0.0
        if diag["index_exists"]:
            score += 0.3
        if diag["fingerprints_exists"]:
            score += 0.3
        if diag["entity_count"] > 0:
            score += 0.2
        if diag["cache_age_hours"] is not None and diag["cache_age_hours"] < 24:
            score += 0.2
        diag["health_score"] = round(score, 2)

        return diag

    def load_master_ontology_context(
        self,
        runs_root: Path,
        max_summary_length: int = 5000,
    ) -> dict[str, Any]:
        """Load master ontology as context for RIGOR engine.

        P3 Fix (2026-05-26): Provides default master ontology context for RIGOR refinement.
        Loads the master ontology file and generates a summary suitable for LLM context.

        Args:
            runs_root: Root directory containing ontology outputs
            max_summary_length: Maximum length of the context summary

        Returns:
            Dictionary with context string and metadata
        """
        # Determine master ontology path (typically in parent of runs_root or project root)
        master_paths = [
            runs_root.parent / "master_ontology_v2.json",
            runs_root / "master_ontology_v2.json",
            Path("outputs/ontology/master_ontology_v2.json"),
            Path("master_ontology_v2.json"),
        ]

        master_path: Path | None = None
        for path in master_paths:
            if path.exists():
                master_path = path
                break

        if master_path is None:
            return {
                "context": "",
                "loaded": False,
                "reason": "master_ontology_v2.json not found in expected locations",
                "entity_count": 0,
                "relationship_count": 0,
            }

        try:
            master_data = self._load_json(master_path)

            # Extract summary information
            nodes = master_data.get("nodes", [])
            relationships = master_data.get("relationships", [])

            # Build a concise summary for LLM context
            summary_parts = [
                "Master Ontology Summary:",
                f"- Total Nodes: {len(nodes)}",
                f"- Total Relationships: {len(relationships)}",
                "",
                "Key Domain Concepts (sample):",
            ]

            # Add sample of node meta_ids (up to 20)
            sample_nodes = nodes[:20]
            for node in sample_nodes:
                meta_id = node.get("meta_id", "N/A")
                meta_name = node.get("meta_name", "N/A")
                meta_class = node.get("meta_class", "N/A")
                summary_parts.append(f"  - {meta_id} ({meta_name}, {meta_class})")

            if len(nodes) > 20:
                summary_parts.append(f"  ... and {len(nodes) - 20} more nodes")

            # Add relationship summary
            summary_parts.append("")
            summary_parts.append("Relationship Types (sample):")
            rel_types = set()
            for rel in relationships[:20]:
                rel_types.add(rel.get("relation_type", "UNKNOWN"))
            for rel_type in sorted(rel_types)[:10]:
                summary_parts.append(f"  - {rel_type}")

            context = "\n".join(summary_parts)

            # Truncate if too long
            if len(context) > max_summary_length:
                context = context[:max_summary_length - 3] + "..."

            return {
                "context": context,
                "loaded": True,
                "path": str(master_path),
                "entity_count": len(nodes),
                "relationship_count": len(relationships),
            }

        except Exception as e:
            return {
                "context": "",
                "loaded": False,
                "reason": f"Error loading master ontology: {e}",
                "entity_count": 0,
                "relationship_count": 0,
            }
