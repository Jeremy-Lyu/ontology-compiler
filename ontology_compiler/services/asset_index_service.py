"""Asset Index Service - Index and query asset packages for RAG consumption.

[DATA CONTINUITY PLAN 2026-07-09]: Implements asset indexing:
- File-based index: Map source files to generated assets
- Run-based index: Map run IDs to asset packages
- Entity-based index: Map entity IDs to their locations
- Relation-based index: Map relation types to relations

This module enables:
- Fast lookup of assets by various criteria
- RAG-ready asset retrieval
- Support for governed-only queries
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from src.utils.file_lock import atomic_write
except ImportError:
    atomic_write = None


@dataclass
class AssetIndexEntry:
    """Asset index entry for quick lookup."""
    run_id: str
    package_path: str
    governed: bool
    generated_at: str
    entity_count: int = 0
    relation_count: int = 0
    quality_score: float = 0.0
    source_files: List[str] = field(default_factory=list)
    entity_ids: List[str] = field(default_factory=list)
    relation_types: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AssetIndexService:
    """Service for indexing and querying asset packages."""

    def __init__(self, index_path: Optional[Path] = None):
        self._index_path = index_path or _PROJECT_ROOT / "outputs" / "asset_index"
        self._index_path.mkdir(parents=True, exist_ok=True)

        self._file_index: Dict[str, List[str]] = {}
        self._run_index: Dict[str, AssetIndexEntry] = {}
        self._entity_index: Dict[str, List[str]] = {}
        self._relation_type_index: Dict[str, List[str]] = {}
        self._layer_index: Dict[str, List[str]] = {}
        self._domain_index: Dict[str, List[str]] = {}

        self._load_index()

    def index_package(self, run_id: str, package_data: Dict[str, Any], package_path: str, governed: bool):
        """Index an asset package."""
        manifest = package_data.get("manifest", {})
        entities = package_data.get("entities", {}).get("entities", [])
        relations = package_data.get("relations", {}).get("relations", [])

        entity_ids = [e.get("id", "") for e in entities if e.get("id")]
        relation_types = list(set(r.get("relation_type", "") for r in relations if r.get("relation_type")))
        source_files = manifest.get("input_files", [])
        layers = list(set(e.get("layer", "") for e in entities if e.get("layer")))

        entry = AssetIndexEntry(
            run_id=run_id,
            package_path=package_path,
            governed=governed,
            generated_at=manifest.get("generated_at", datetime.now().isoformat()),
            entity_count=len(entities),
            relation_count=len(relations),
            quality_score=manifest.get("statistics", {}).get("quality_score", 0.0),
            source_files=source_files,
            entity_ids=entity_ids,
            relation_types=relation_types,
        )

        self._run_index[run_id] = entry

        for source_file in source_files:
            if source_file not in self._file_index:
                self._file_index[source_file] = []
            if run_id not in self._file_index[source_file]:
                self._file_index[source_file].append(run_id)

        for entity_id in entity_ids:
            if entity_id not in self._entity_index:
                self._entity_index[entity_id] = []
            if run_id not in self._entity_index[entity_id]:
                self._entity_index[entity_id].append(run_id)

        for relation_type in relation_types:
            if relation_type not in self._relation_type_index:
                self._relation_type_index[relation_type] = []
            if run_id not in self._relation_type_index[relation_type]:
                self._relation_type_index[relation_type].append(run_id)

        for layer in layers:
            if layer not in self._layer_index:
                self._layer_index[layer] = []
            if run_id not in self._layer_index[layer]:
                self._layer_index[layer].append(run_id)

        self._save_index()
        logger.info(f"Indexed package {run_id} (governed={governed}, entities={len(entities)})")

    def find_by_file(self, file_path: str, governed_only: bool = False) -> List[AssetIndexEntry]:
        """Find packages by source file."""
        run_ids = self._file_index.get(file_path, [])
        return self._filter_by_governed(run_ids, governed_only)

    def find_by_run_id(self, run_id: str) -> Optional[AssetIndexEntry]:
        """Find package by run ID."""
        return self._run_index.get(run_id)

    def find_by_entity_id(self, entity_id: str, governed_only: bool = False) -> List[AssetIndexEntry]:
        """Find packages containing an entity."""
        run_ids = self._entity_index.get(entity_id, [])
        return self._filter_by_governed(run_ids, governed_only)

    def find_by_relation_type(self, relation_type: str, governed_only: bool = False) -> List[AssetIndexEntry]:
        """Find packages containing relations of a specific type."""
        run_ids = self._relation_type_index.get(relation_type, [])
        return self._filter_by_governed(run_ids, governed_only)

    def find_by_layer(self, layer: str, governed_only: bool = False) -> List[AssetIndexEntry]:
        """Find packages containing entities of a specific layer."""
        run_ids = self._layer_index.get(layer, [])
        return self._filter_by_governed(run_ids, governed_only)

    def find_all(self, governed_only: bool = False, min_quality_score: float = 0.0) -> List[AssetIndexEntry]:
        """Find all packages with optional filters."""
        entries = list(self._run_index.values())
        if governed_only:
            entries = [e for e in entries if e.governed]
        if min_quality_score > 0:
            entries = [e for e in entries if e.quality_score >= min_quality_score]
        return sorted(entries, key=lambda e: e.generated_at, reverse=True)

    def get_entity_locations(self, entity_id: str) -> List[Tuple[str, str]]:
        """Get (run_id, package_path) tuples for an entity."""
        run_ids = self._entity_index.get(entity_id, [])
        return [(run_id, self._run_index[run_id].package_path) for run_id in run_ids]

    def get_related_entities(self, entity_id: str, governed_only: bool = False) -> Set[str]:
        """Get all entities that appear in the same packages as the given entity."""
        run_ids = self._entity_index.get(entity_id, [])
        if governed_only:
            run_ids = [rid for rid in run_ids if self._run_index[rid].governed]

        related_entities: Set[str] = set()
        for run_id in run_ids:
            entry = self._run_index[run_id]
            related_entities.update(entry.entity_ids)

        related_entities.discard(entity_id)
        return related_entities

    def search_entities(
        self,
        query: str,
        governed_only: bool = False,
        layer: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search entities across all packages."""
        results = []
        query_lower = query.lower()

        for run_id, entry in self._run_index.items():
            if governed_only and not entry.governed:
                continue
            if layer and run_id not in self._layer_index.get(layer, []):
                continue

            package_path = Path(entry.package_path)
            entities_file = package_path / "entities.json"
            if not entities_file.exists():
                continue

            try:
                with open(entities_file, "r", encoding="utf-8") as f:
                    entities_data = json.load(f)
                    for entity in entities_data.get("entities", []):
                        name = entity.get("name", "").lower()
                        code = entity.get("code", "").lower()
                        entity_type = entity.get("type", "").lower()

                        if query_lower in name or query_lower in code or query_lower in entity_type:
                            confidence = entity.get("confidence", 1.0)
                            if confidence >= min_confidence:
                                results.append({
                                    "id": entity.get("id"),
                                    "name": entity.get("name"),
                                    "code": entity.get("code"),
                                    "type": entity.get("type"),
                                    "layer": entity.get("layer"),
                                    "confidence": confidence,
                                    "run_id": run_id,
                                    "package_path": str(package_path),
                                })
            except Exception as e:
                logger.warning(f"Failed to search entities in {package_path}: {e}")

        return sorted(results, key=lambda r: r.get("confidence", 0), reverse=True)

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        governed_count = sum(1 for e in self._run_index.values() if e.governed)
        total_entities = sum(e.entity_count for e in self._run_index.values())
        total_relations = sum(e.relation_count for e in self._run_index.values())

        return {
            "total_packages": len(self._run_index),
            "governed_packages": governed_count,
            "raw_packages": len(self._run_index) - governed_count,
            "total_entities": total_entities,
            "total_relations": total_relations,
            "total_source_files": len(self._file_index),
            "total_relation_types": len(self._relation_type_index),
            "total_layers": len(self._layer_index),
        }

    def remove_package(self, run_id: str):
        """Remove a package from the index."""
        if run_id not in self._run_index:
            return

        entry = self._run_index[run_id]

        for source_file in entry.source_files:
            if source_file in self._file_index:
                self._file_index[source_file].remove(run_id)
                if not self._file_index[source_file]:
                    del self._file_index[source_file]

        for entity_id in entry.entity_ids:
            if entity_id in self._entity_index:
                self._entity_index[entity_id].remove(run_id)
                if not self._entity_index[entity_id]:
                    del self._entity_index[entity_id]

        for relation_type in entry.relation_types:
            if relation_type in self._relation_type_index:
                self._relation_type_index[relation_type].remove(run_id)
                if not self._relation_type_index[relation_type]:
                    del self._relation_type_index[relation_type]

        del self._run_index[run_id]
        self._save_index()
        logger.info(f"Removed package {run_id} from index")

    def rebuild_index(self, asset_packet_service):
        """Rebuild index from all asset packages."""
        self._clear_index()

        packages = asset_packet_service.list_packages()
        for pkg_info in packages:
            run_id = pkg_info["run_id"]
            governed = pkg_info["governed"]
            package_data = asset_packet_service.load_asset_packet(run_id, governed=governed)
            if package_data:
                package_path = package_data.get("package_path", "")
                self.index_package(run_id, package_data, package_path, governed)

        logger.info(f"Rebuilt index with {len(self._run_index)} packages")

    def _filter_by_governed(self, run_ids: List[str], governed_only: bool) -> List[AssetIndexEntry]:
        """Filter entries by governed status."""
        entries = [self._run_index[rid] for rid in run_ids if rid in self._run_index]
        if governed_only:
            entries = [e for e in entries if e.governed]
        return entries

    def _clear_index(self):
        """Clear all indexes."""
        self._file_index = {}
        self._run_index = {}
        self._entity_index = {}
        self._relation_type_index = {}
        self._layer_index = {}
        self._domain_index = {}

    def _save_index(self):
        """Save index to disk."""
        index_data = {
            "file_index": self._file_index,
            "run_index": {k: v.to_dict() for k, v in self._run_index.items()},
            "entity_index": self._entity_index,
            "relation_type_index": self._relation_type_index,
            "layer_index": self._layer_index,
            "domain_index": self._domain_index,
            "saved_at": datetime.now().isoformat(),
        }

        index_file = self._index_path / "asset_index.json"
        
        if atomic_write:
            try:
                atomic_write(index_file, index_data, timeout=30.0, check_interval=0.5)
                return
            except Exception as e:
                logger.warning(f"Failed to use atomic_write for index: {e}")
        
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)

    def _load_index(self):
        """Load index from disk."""
        index_file = self._index_path / "asset_index.json"
        if not index_file.exists():
            return

        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)

            self._file_index = index_data.get("file_index", {})
            self._run_index = {
                k: AssetIndexEntry(**v) for k, v in index_data.get("run_index", {}).items()
            }
            self._entity_index = index_data.get("entity_index", {})
            self._relation_type_index = index_data.get("relation_type_index", {})
            self._layer_index = index_data.get("layer_index", {})
            self._domain_index = index_data.get("domain_index", {})

            logger.info(f"Loaded index with {len(self._run_index)} packages")
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")


_global_asset_index_service = None


def get_asset_index_service() -> AssetIndexService:
    """Get or create global asset index service instance."""
    global _global_asset_index_service
    if _global_asset_index_service is None:
        _global_asset_index_service = AssetIndexService()
    return _global_asset_index_service


__all__ = [
    "AssetIndexEntry",
    "AssetIndexService",
    "get_asset_index_service",
]