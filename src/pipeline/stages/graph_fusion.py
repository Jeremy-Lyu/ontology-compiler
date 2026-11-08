#!/usr/bin/env python3
"""Graph Algebra & Topology Fusion (步骤2: 图代数拓扑融合)

Implements graph union operations: G_full = ∪ G_CQ_i
- Entity alignment verification (canonical_id consistency)
- Edge stitching (cross-fragment relationship connectivity)
- Topology connectivity audit (isolated components detection)
- Cross-fragment chain validation (multi-hop path integrity)

This stage runs AFTER dedup + relationship building, BEFORE audit.
"""
import random
import time
from argparse import Namespace
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import StageResult, _read_json, _write_json

try:
    from core.utils.relationship_helpers import relationship_endpoints as _rel_endpoints
except ImportError:
    def _rel_endpoints(rel: Any) -> tuple[str, str]:
        if not isinstance(rel, dict):
            return ("", "")
        source = str(rel.get("source") or rel.get("from") or "").strip()
        target = str(rel.get("target") or rel.get("to") or "").strip()
        return (source, target)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_id(node: dict[str, Any]) -> str:
    """Return the canonical id of a node (meta_id > entity_id > id)."""
    if not isinstance(node, dict):
        return ""
    return str(node.get("meta_id") or node.get("entity_id") or node.get("id") or "")


def _extract_source_file(node: dict[str, Any]) -> str:
    """Extract fragment origin from a node's evidence.source_file (with fallbacks)."""
    if not isinstance(node, dict):
        return ""
    evidence = node.get("evidence")
    if isinstance(evidence, dict):
        sf = evidence.get("source_file") or evidence.get("source_document")
        if sf:
            return str(sf)
    prov = node.get("provenance")
    if isinstance(prov, dict):
        sf = prov.get("source_file") or prov.get("source_document")
        if sf:
            return str(sf)
    for key in ("source_file", "source_structure", "source_document", "source_id"):
        val = node.get(key)
        if val:
            return str(val)
    return ""


def _rel_summary(rel: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": str(rel.get("source") or rel.get("from") or ""),
        "target": str(rel.get("target") or rel.get("to") or ""),
        "type": str(rel.get("type") or rel.get("relation_type") or rel.get("rel_type") or ""),
    }


def _build_adjacency(nodes: list[dict[str, Any]], relationships: list[dict[str, Any]]) -> tuple[dict[str, set[str]], set[str]]:
    """Build an undirected adjacency list restricted to known node ids.

    Returns (adjacency, node_ids). Self-references and edges referencing unknown
    endpoints (dangling) are skipped so topology metrics reflect the induced
    subgraph of canonical nodes.
    """
    node_ids = {_node_id(n) for n in nodes if _node_id(n)}
    adjacency: dict[str, set[str]] = {nid: set() for nid in node_ids}
    for rel in relationships:
        s, t = _rel_endpoints(rel)
        if not s or not t or s == t:
            continue
        if s not in node_ids or t not in node_ids:
            continue
        adjacency[s].add(t)
        adjacency[t].add(s)
    return adjacency, node_ids


# ---------------------------------------------------------------------------
# Core analyses
# ---------------------------------------------------------------------------

def _compute_graph_union(nodes: list[dict[str, Any]], relationships: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the graph union G_full = ∪ G_CQ_i and verify completeness.

    - Checks all canonical_ids are unique
    - Verifies orphan nodes (nodes not referenced by any relationship)
    - Counts connected components
    """
    id_list = [_node_id(n) for n in nodes if _node_id(n)]
    id_counts = Counter(id_list)
    duplicates = {eid: cnt for eid, cnt in id_counts.items() if cnt > 1}
    duplicate_count = sum(cnt - 1 for cnt in duplicates.values())

    referenced: set[str] = set()
    for rel in relationships:
        s, t = _rel_endpoints(rel)
        if s:
            referenced.add(s)
        if t:
            referenced.add(t)

    node_ids = set(id_list)
    orphans = sorted(nid for nid in node_ids if nid not in referenced)

    components, _ = _find_connected_components(nodes, relationships)

    return {
        "canonical_id_count": len(node_ids),
        "duplicate_canonical_ids": duplicates,
        "duplicate_count": duplicate_count,
        "orphan_nodes": orphans[:200],
        "orphan_count": len(orphans),
        "connected_components": len(components),
    }


def _validate_edge_stitching(
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    id_remap: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Verify edge connectivity after entity resolution / dedup.

    - Finds edges that reference merged IDs (id_remap keys = stale ids)
    - Verifies all relationship source/target ids exist in nodes
    - Identifies broken chains (edges that should connect via canonical_ids but don't)
    """
    node_ids = {_node_id(n) for n in nodes if _node_id(n)}
    id_remap = id_remap or {}

    dangling: list[dict[str, Any]] = []
    remapped_refs: list[dict[str, Any]] = []

    for rel in relationships:
        s, t = _rel_endpoints(rel)
        s_missing = bool(s) and s not in node_ids
        t_missing = bool(t) and t not in node_ids

        refs_old: list[dict[str, Any]] = []
        if s and s in id_remap:
            refs_old.append({"endpoint": "source", "stale_id": s, "canonical_id": id_remap[s]})
        if t and t in id_remap:
            refs_old.append({"endpoint": "target", "stale_id": t, "canonical_id": id_remap[t]})
        if refs_old:
            remapped_refs.append({"endpoints": refs_old, "rel": _rel_summary(rel)})

        if s_missing or t_missing:
            dangling.append({
                "source": s,
                "target": t,
                "source_missing": s_missing,
                "target_missing": t_missing,
                "rel": _rel_summary(rel),
            })

    return {
        "total_edges": len(relationships),
        "dangling_edges": len(dangling),
        "edges_referencing_merged_ids": len(remapped_refs),
        "broken_chains_count": len(dangling),
        "broken_chains": dangling[:100],
        "remapped_edge_refs": remapped_refs[:100],
        "id_remap_size": len(id_remap),
    }


def _find_connected_components(
    nodes: list[dict[str, Any]], relationships: list[dict[str, Any]]
) -> tuple[list[set[str]], list[str]]:
    """BFS to find connected components.

    Returns (components, isolated_nodes) where isolated_nodes are nodes with no
    edges (degree 0).
    """
    adjacency, node_ids = _build_adjacency(nodes, relationships)
    visited: set[str] = set()
    components: list[set[str]] = []
    isolated: list[str] = []

    for nid in node_ids:
        if nid in visited:
            continue
        comp: set[str] = set()
        queue: deque[str] = deque([nid])
        visited.add(nid)
        while queue:
            cur = queue.popleft()
            comp.add(cur)
            for nb in adjacency.get(cur, ()):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(comp)
        if len(comp) == 1:
            sole = next(iter(comp))
            if not adjacency.get(sole):
                isolated.append(sole)

    components.sort(key=lambda c: -len(c))
    return components, isolated


def _validate_cross_fragment_chains(
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    provenance_map: dict[str, str],
) -> dict[str, Any]:
    """Check multi-hop chains across fragments.

    For each pair of source files, determines whether some connected component
    bridges them (i.e. a path exists). Pairs without a bridging component are
    reported as missing links.
    """
    components, _ = _find_connected_components(nodes, relationships)

    component_fragments: list[set[str]] = []
    for comp in components:
        frags: set[str] = set()
        for nid in comp:
            sf = provenance_map.get(nid, "")
            if sf and sf != "unknown":
                frags.add(sf)
        component_fragments.append(frags)

    all_fragments = sorted(
        {sf for sf in provenance_map.values() if sf and sf != "unknown"}
    )

    connected_pairs = 0
    missing_links: list[dict[str, Any]] = []
    for i in range(len(all_fragments)):
        for j in range(i + 1, len(all_fragments)):
            fA, fB = all_fragments[i], all_fragments[j]
            bridged = any(fA in frags and fB in frags for frags in component_fragments)
            if bridged:
                connected_pairs += 1
            else:
                missing_links.append({"fragment_a": fA, "fragment_b": fB})

    bridged_components = sum(1 for frags in component_fragments if len(frags) > 1)

    return {
        "total_fragments": len(all_fragments),
        "connected_fragment_pairs": connected_pairs,
        "disconnected_fragment_pairs": len(missing_links),
        "missing_links": missing_links[:200],
        "bridged_components": bridged_components,
    }


def _avg_path_length(component: set[str], adjacency: dict[str, set[str]]) -> float:
    """Average shortest-path length within a component (sampled for large sets)."""
    if not component or len(component) <= 1:
        return 0.0
    nodes_list = list(component)
    if len(nodes_list) > 200:
        rng = random.Random(42)
        sample = rng.sample(nodes_list, 200)
    else:
        sample = nodes_list

    total_dist = 0
    pairs = 0
    for start in sample:
        dist: dict[str, int] = {start: 0}
        queue: deque[str] = deque([start])
        while queue:
            u = queue.popleft()
            for v in adjacency.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for nid, d in dist.items():
            if nid != start:
                total_dist += d
                pairs += 1
    return round(total_dist / pairs, 3) if pairs else 0.0


def _compute_topology_metrics(
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    provenance_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return topology metrics for the fused graph."""
    components, isolated = _find_connected_components(nodes, relationships)
    adjacency, _ = _build_adjacency(nodes, relationships)

    largest_comp = components[0] if components else set()
    largest_size = len(largest_comp)

    cross_fragment_edges = 0
    if provenance_map:
        for rel in relationships:
            s, t = _rel_endpoints(rel)
            sf_s = provenance_map.get(s, "")
            sf_t = provenance_map.get(t, "")
            if sf_s and sf_t and sf_s != sf_t:
                cross_fragment_edges += 1

    avg_pl = _avg_path_length(largest_comp, adjacency)

    return {
        "total_nodes": len(nodes),
        "total_edges": len(relationships),
        "connected_components": len(components),
        "largest_component_size": largest_size,
        "isolated_nodes": len(isolated),
        "cross_fragment_edges": cross_fragment_edges,
        "avg_path_length": avg_pl,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run_graph_fusion(args: Namespace) -> StageResult:
    """Main entry point for the Graph Algebra & Topology Fusion stage."""
    start = time.time()
    master_path = Path(args.master_path)
    if not master_path.exists():
        return StageResult(
            "graph_fusion",
            False,
            f"master not found: {master_path}",
            elapsed_seconds=round(time.time() - start, 3),
        )

    if master_path.is_dir():
        nodes = _read_json(master_path / "nodes.json", [])
        instances = _read_json(master_path / "instances.json", [])
        relationships = _read_json(master_path / "relationships.json", [])
        metadata = _read_json(master_path / "metadata.json", {})
        master = {
            "nodes": nodes,
            "instances": instances,
            "relationships": relationships,
            "metadata": metadata,
        }
    else:
        master = _read_json(master_path, {})
        nodes = master.get("nodes", [])
        instances = master.get("instances", [])
        relationships = master.get("relationships") or master.get("edges") or []
        metadata = master.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

    # All entities (nodes + instances) participate in the fused topology
    all_nodes = [n for n in nodes if isinstance(n, dict)] + [i for i in instances if isinstance(i, dict)]
    relationships = [r for r in relationships if isinstance(r, dict)]

    # id_remap (old_id -> canonical_id) produced by merge / entity_resolution
    id_remap = master.get("id_remap") or metadata.get("id_remap") or {}
    if not isinstance(id_remap, dict):
        id_remap = {}

    # Provenance map: node_id -> source_file (fragment origin)
    provenance_map: dict[str, str] = {}
    for n in all_nodes:
        nid = _node_id(n)
        if nid:
            provenance_map[nid] = _extract_source_file(n)

    union_report = _compute_graph_union(all_nodes, relationships)
    stitching_report = _validate_edge_stitching(all_nodes, relationships, id_remap)
    components, isolated_nodes = _find_connected_components(all_nodes, relationships)
    chain_report = _validate_cross_fragment_chains(all_nodes, relationships, provenance_map)
    metrics = _compute_topology_metrics(all_nodes, relationships, provenance_map)

    # Output directory: stage output dir if provided, otherwise next to master
    output_dir_str = getattr(args, "output_dir", "")
    if output_dir_str:
        output_dir = Path(output_dir_str)
    elif master_path.is_file():
        output_dir = master_path.parent
    else:
        output_dir = master_path
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "fusion_report.json"

    fusion_report = {
        "stage": "graph_fusion",
        "timestamp": datetime.now().isoformat(),
        "master_path": str(master_path),
        "run_id": str(getattr(args, "run_id", "")),
        "graph_union": union_report,
        "edge_stitching": stitching_report,
        "cross_fragment_chains": chain_report,
        "topology_metrics": metrics,
        "isolated_nodes": isolated_nodes[:200],
        "connected_components": [
            {"size": len(c), "node_ids": sorted(c)[:50]} for c in components
        ],
    }
    _write_json(report_path, fusion_report)

    total_issues = (
        union_report.get("duplicate_count", 0)
        + stitching_report.get("dangling_edges", 0)
        + chain_report.get("disconnected_fragment_pairs", 0)
    )

    message = (
        f"graph_fusion completed: {metrics['connected_components']} components, "
        f"{metrics['isolated_nodes']} isolated nodes, "
        f"{stitching_report.get('dangling_edges', 0)} dangling edges, "
        f"{total_issues} total issues"
    )

    return StageResult(
        "graph_fusion",
        True,
        message,
        {
            "total_nodes": metrics["total_nodes"],
            "total_edges": metrics["total_edges"],
            "connected_components": metrics["connected_components"],
            "largest_component_size": metrics["largest_component_size"],
            "isolated_nodes": metrics["isolated_nodes"],
            "cross_fragment_edges": metrics["cross_fragment_edges"],
            "avg_path_length": metrics["avg_path_length"],
            "duplicate_canonical_ids": union_report.get("duplicate_count", 0),
            "orphan_nodes": union_report.get("orphan_count", 0),
            "dangling_edges": stitching_report.get("dangling_edges", 0),
            "edges_referencing_merged_ids": stitching_report.get("edges_referencing_merged_ids", 0),
            "broken_chains": stitching_report.get("broken_chains_count", 0),
            "connected_fragment_pairs": chain_report.get("connected_fragment_pairs", 0),
            "disconnected_fragment_pairs": chain_report.get("disconnected_fragment_pairs", 0),
            "total_issues": total_issues,
            "report_path": str(report_path),
        },
        round(time.time() - start, 3),
        artifact_paths=[str(report_path)],
        run_dir=str(output_dir),
    )
