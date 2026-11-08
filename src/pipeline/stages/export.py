"""Export, Publish, and Rollback stages."""
import shutil
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from .common import StageResult, PROJECT_ROOT, _read_json, _write_json, _entity_layer_value

try:
    from src.pipeline.storage.master_store import MasterOntologyStore, create_master_store
except ImportError:
    MasterOntologyStore = None
    create_master_store = None


def run_export(args: Namespace) -> StageResult:
    start = time.time()
    master_path = Path(args.master_path)
    if not master_path.exists():
        return StageResult("export", False, f"master not found: {master_path}", elapsed_seconds=round(time.time() - start, 3))
    
    ontology_base_dir = PROJECT_ROOT / "outputs" / "runs" / "ontology"
    master_dir = ontology_base_dir / "master"
    reports_dir = ontology_base_dir / "reports"
    layers_dir = master_dir / "layers"
    relationships_dir = master_dir / "relationships"
    instances_dir = master_dir / "instances"
    layers_dir.mkdir(parents=True, exist_ok=True)
    relationships_dir.mkdir(parents=True, exist_ok=True)
    instances_dir.mkdir(parents=True, exist_ok=True)
    
    if MasterOntologyStore and create_master_store:
        store = create_master_store(str(master_path), sync_mode=True)
        master = store.read()
    else:
        master = _read_json(master_path, {})
    
    nodes = master.get("nodes", [])
    instances = master.get("instances", [])
    relationships = master.get("relationships") or master.get("edges") or []
    
    layer_nodes = {"L0": [], "L1": [], "L2": [], "L2.5": [], "L3": []}
    for n in nodes:
        layer = _entity_layer_value(n)
        if layer in layer_nodes:
            layer_nodes[layer].append(n)
    
    layer_nodes["L3"].extend(instances)
    
    exported_layers = []
    for layer, layer_node_list in layer_nodes.items():
        layer_path = layers_dir / f"layer_{layer}.json"
        _write_json(layer_path, layer_node_list)
        exported_layers.append(str(layer_path))
    
    exported_instances = []
    instances_path = instances_dir / "instances.json"
    _write_json(instances_path, instances)
    exported_instances.append(str(instances_path))
    
    rel_types: dict[str, list[dict]] = {}
    for rel in relationships:
        rel_type = rel.get("type", "unknown")
        if rel_type not in rel_types:
            rel_types[rel_type] = []
        rel_types[rel_type].append(rel)
    
    exported_relationships = []
    
    all_rels_path = relationships_dir / "relationships.json"
    _write_json(all_rels_path, relationships)
    exported_relationships.append(str(all_rels_path))
    
    for rel_type, rel_list in rel_types.items():
        rel_path = relationships_dir / f"relationships_{rel_type}.json"
        _write_json(rel_path, rel_list)
        exported_relationships.append(str(rel_path))
    
    export_summary = {
        "total_nodes": len(nodes),
        "total_instances": len(instances),
        "total_relationships": len(relationships),
        "layers": {layer: len(layer_node_list) for layer, layer_node_list in layer_nodes.items()},
        "relationship_types": {rel_type: len(rels) for rel_type, rels in rel_types.items()},
    }
    summary_path = reports_dir / "export_summary.json"
    _write_json(summary_path, export_summary)
    
    return StageResult(
        "export",
        True,
        "export completed",
        {
            "export_summary": export_summary,
            "exported_layers": exported_layers,
            "exported_instances": exported_instances,
            "exported_relationships": exported_relationships,
            "summary_path": str(summary_path),
            "export_base_dir": str(ontology_base_dir),
        },
        round(time.time() - start, 3),
    )


def run_publish(args: Namespace) -> StageResult:
    start = time.time()
    master_path = Path(args.master_path)
    if not master_path.exists():
        return StageResult("publish", False, f"master missing: {master_path}", elapsed_seconds=round(time.time() - start, 3))
    
    publish_dir = PROJECT_ROOT / "outputs" / "runs" / "ontology" / "master" / "published"
    publish_dir.mkdir(parents=True, exist_ok=True)
    target = publish_dir / master_path.name
    shutil.copy2(master_path, target)
    
    return StageResult(
        "publish",
        True,
        "publish completed",
        {"published_path": str(target)},
        round(time.time() - start, 3),
    )


def run_rollback(args: Namespace) -> StageResult:
    start = time.time()
    master_path = Path(args.master_path)
    backups = sorted(
        master_path.parent.glob(f"{master_path.stem}.backup_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not backups:
        return StageResult("rollback", False, "no backup found", elapsed_seconds=round(time.time() - start, 3))
    
    shutil.copy2(backups[0], master_path)
    
    return StageResult(
        "rollback",
        True,
        "rollback completed",
        {"restored_from": str(backups[0])},
        round(time.time() - start, 3),
    )