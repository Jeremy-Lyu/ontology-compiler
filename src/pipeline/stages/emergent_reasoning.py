#!/usr/bin/env python3
"""Emergent Reasoning Engine (步骤4: 涌现推理计算)

Provides multi-hop graph traversal and reasoning capabilities:
1. Event Anchoring - Locate event source nodes in the ontology graph
2. Multi-hop BFS Traversal - Traverse the graph from anchor nodes
3. Function Execution - Execute bound calculation functions on nodes
4. Grounded Conclusion Generation - Generate actionable decisions

Example use case:
  Event: Supplier_A delays delivery by 3 days
  Query: "Which VIN vehicles are affected? What's the line stoppage countdown?"

  Engine traces: Supplier -> Part -> MBOM -> BOP Station -> VIN
  Executes: stock_on_hand / shift_demand = remaining_hours
  Output: "120 VINs affected, line stops in 2.5 hours (14:30 today)"
"""
from __future__ import annotations

import ast
import asyncio
import json
import operator
import time
from argparse import Namespace
from collections import deque
from pathlib import Path
from typing import Any

from .common import (
    StageResult,
    PROJECT_ROOT,
    _read_json,
    _write_json,
    _iter_master_entities,
    _entity_layer_value,
)

try:
    from src.pipeline.storage.master_store import MasterOntologyStore
except ImportError:
    MasterOntologyStore = None


# ---------------------------------------------------------------------------
# Internal normalization helpers
# ---------------------------------------------------------------------------

def _node_id(node: dict[str, Any]) -> str:
    return str(
        node.get("meta_id")
        or node.get("entity_id")
        or node.get("id")
        or ""
    ).strip()


def _node_name(node: dict[str, Any]) -> str:
    return str(
        node.get("name")
        or node.get("meta_name")
        or node.get("label")
        or ""
    ).strip()


def _node_layer(node: dict[str, Any]) -> str:
    return str(_entity_layer_value(node) if isinstance(node, dict) else "unknown")


def _edge_type(edge: dict[str, Any]) -> str:
    return str(
        edge.get("type")
        or edge.get("relation_type")
        or edge.get("rel_type")
        or edge.get("predicate")
        or "RELATED_TO"
    ).upper().strip()


def _edge_endpoints(edge: dict[str, Any]) -> tuple[str, str]:
    """Extract (source_id, target_id) from a relationship dict.

    Handles the multiple field naming conventions used across the codebase:
    source/target, from/to, with provenance-value fallback.
    """
    from_id = str(edge.get("from") or "").strip()
    to_id = str(edge.get("to") or "").strip()
    source_id = str(edge.get("source") or "").strip()
    target_id = str(edge.get("target") or "").strip()

    provenance_values = {
        "same_as_detection",
        "cross_file_association",
        "manual_fallback",
        "run_pipeline_orchestrator",
    }
    if source_id in provenance_values or not source_id:
        source_id = from_id
    if not target_id:
        target_id = to_id
    return source_id, target_id


def _node_attributes(node: dict[str, Any]) -> dict[str, Any]:
    """Flatten a node's attributes list plus top-level scalar fields."""
    attrs: dict[str, Any] = {}
    raw_attrs = node.get("attributes") or []
    if isinstance(raw_attrs, list):
        for attr in raw_attrs:
            if isinstance(attr, dict):
                name = str(attr.get("name") or attr.get("key") or "").strip()
                if name:
                    attrs[name] = attr.get("value")
            elif isinstance(attr, (tuple, list)) and len(attr) == 2:
                attrs[str(attr[0])] = attr[1]
    elif isinstance(raw_attrs, dict):
        attrs.update(raw_attrs)
    # Top-level scalar fields (numeric/str) supplement attribute lookup
    for key, value in node.items():
        if key in {"attributes", "evidence", "provenance", "metadata"}:
            continue
        if isinstance(value, (int, float, str)) and key not in attrs:
            attrs[key] = value
    return attrs


# ---------------------------------------------------------------------------
# Safe arithmetic formula evaluation (no eval/exec)
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST, variables: dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, variables)
    if isinstance(node, ast.Constant):  # Python >= 3.8
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.Num):  # Python < 3.8 fallback
        return float(node.n)
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise KeyError(f"variable '{node.id}' not found along path")
        value = variables[node.id]
        return float(value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _safe_eval(node.left, variables)
        right = _safe_eval(node.right, variables)
        return float(_ALLOWED_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        operand = _safe_eval(node.operand, variables)
        return float(_ALLOWED_UNARYOPS[type(node.op)](operand))
    raise ValueError(f"unsupported expression node: {ast.dump(node)}")


# ---------------------------------------------------------------------------
# OntologyGraph
# ---------------------------------------------------------------------------

class OntologyGraph:
    """Graph data structure for the ontology, backed by an adjacency list.

    Supports bidirectional traversal, edge-type filtering, and node lookup
    by ID, name (case-insensitive), meta_class, and layer.
    """

    def __init__(self, nodes: list[dict[str, Any]], relationships: list[dict[str, Any]]):
        self.nodes: dict[str, dict[str, Any]] = {}
        self._name_index: dict[str, list[str]] = {}
        self._class_index: dict[str, list[str]] = {}
        self._layer_index: dict[str, list[str]] = {}
        # adjacency: node_id -> list[edge dict with source/target/type/raw]
        self._outgoing: dict[str, list[dict[str, Any]]] = {}
        self._incoming: dict[str, list[dict[str, Any]]] = {}

        for node in nodes:
            if not isinstance(node, dict):
                continue
            nid = _node_id(node)
            if not nid:
                continue
            if nid not in self.nodes:
                self.nodes[nid] = node
            self._outgoing.setdefault(nid, [])
            self._incoming.setdefault(nid, [])

            name_key = _node_name(node).lower()
            if name_key:
                self._name_index.setdefault(name_key, []).append(nid)
            class_key = str(node.get("meta_class") or node.get("entity_type") or "").strip().lower()
            if class_key:
                self._class_index.setdefault(class_key, []).append(nid)
            layer_key = _node_layer(node)
            self._layer_index.setdefault(layer_key, []).append(nid)

        for edge in relationships:
            if not isinstance(edge, dict):
                continue
            source_id, target_id = _edge_endpoints(edge)
            if not source_id or not target_id:
                continue
            etype = _edge_type(edge)
            norm_edge = {
                "source": source_id,
                "target": target_id,
                "type": etype,
                "raw": edge,
            }
            self._outgoing.setdefault(source_id, []).append(norm_edge)
            self._incoming.setdefault(target_id, []).append(norm_edge)

    # -- node accessors ---------------------------------------------------

    def get_node(self, node_id: str) -> dict[str, Any]:
        return self.nodes.get(node_id, {})

    def get_neighbors(self, node_id: str, direction: str = "both") -> list[str]:
        neighbors: list[str] = []
        seen: set[str] = set()
        if direction in ("both", "outgoing"):
            for edge in self._outgoing.get(node_id, []):
                tgt = edge["target"]
                if tgt != node_id and tgt not in seen:
                    neighbors.append(tgt)
                    seen.add(tgt)
        if direction in ("both", "incoming"):
            for edge in self._incoming.get(node_id, []):
                src = edge["source"]
                if src != node_id and src not in seen:
                    neighbors.append(src)
                    seen.add(src)
        return neighbors

    def get_edges(self, node_id: str, direction: str = "both") -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        if direction in ("both", "outgoing"):
            edges.extend(self._outgoing.get(node_id, []))
        if direction in ("both", "incoming"):
            edges.extend(self._incoming.get(node_id, []))
        return edges

    def find_nodes_by_name(self, name: str) -> list[dict[str, Any]]:
        needle = (name or "").strip().lower()
        if not needle:
            return []
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for name_key, nids in self._name_index.items():
            if needle in name_key:
                for nid in nids:
                    if nid not in seen:
                        seen.add(nid)
                        results.append(self.nodes[nid])
        return results

    def find_nodes_by_class(self, meta_class: str) -> list[dict[str, Any]]:
        key = (meta_class or "").strip().lower()
        if not key:
            return []
        nids = self._class_index.get(key, [])
        return [self.nodes[nid] for nid in nids if nid in self.nodes]

    def find_nodes_by_layer(self, layer: str) -> list[dict[str, Any]]:
        key = str(layer or "").strip()
        nids = self._layer_index.get(key, [])
        return [self.nodes[nid] for nid in nids if nid in self.nodes]

    def __len__(self) -> int:
        return len(self.nodes)

    def all_node_ids(self) -> list[str]:
        return list(self.nodes.keys())


# ---------------------------------------------------------------------------
# BFS traversal & path formatting
# ---------------------------------------------------------------------------

def _bfs_traverse(
    graph: OntologyGraph,
    start_id: str,
    max_hops: int = 5,
    edge_type_filter: list[str] | None = None,
) -> list[list[str]]:
    """BFS implementation returning all simple paths up to max_hops.

    Each path is a list of node IDs. A node is not repeated within a single
    path (simple path) to prevent cycles. Uses collections.deque for O(1)
    popleft. A cap on total paths guards against combinatorial explosion.
    """
    if start_id not in graph.nodes:
        return []

    allowed_types: set[str] | None = None
    if edge_type_filter:
        allowed_types = {t.upper() for t in edge_type_filter}

    max_paths = 5000
    results: list[list[str]] = []
    queue: deque[list[str]] = deque([[start_id]])

    while queue and len(results) < max_paths:
        path = queue.popleft()
        current = path[-1]
        hop = len(path) - 1
        if hop >= max_hops:
            results.append(path)
            continue

        expanded = False
        visited_in_path = set(path)
        for edge in graph.get_edges(current, direction="both"):
            if allowed_types is not None and edge["type"] not in allowed_types:
                continue
            neighbor = edge["target"] if edge["source"] == current else edge["source"]
            if neighbor in visited_in_path:
                continue
            expanded = True
            queue.append(path + [neighbor])

        if not expanded:
            results.append(path)

    return results


def _format_path(graph: OntologyGraph, path: list[str]) -> str:
    """Format a path as readable text.

    Example: "Supplier_A --[CALL_OFF]--> Part_X --[CONTAINS]--> MBOM_001"
    """
    if not path:
        return ""
    parts: list[str] = []
    for idx, nid in enumerate(path):
        node = graph.get_node(nid)
        label = _node_name(node) or nid
        parts.append(label if idx == 0 else f"{label} ({nid})" if not label else label)
        if idx < len(path) - 1:
            next_id = path[idx + 1]
            edge_type = _edge_type_between(graph, nid, next_id)
            parts.append(f" --[{edge_type}]--> ")
    return "".join(parts)


def _edge_type_between(graph: OntologyGraph, src_id: str, tgt_id: str) -> str:
    for edge in graph.get_edges(src_id, direction="both"):
        if edge["source"] == tgt_id or edge["target"] == tgt_id:
            return edge["type"]
    return "RELATED_TO"


# ---------------------------------------------------------------------------
# EmergentReasoningEngine
# ---------------------------------------------------------------------------

class EmergentReasoningEngine:
    """Multi-hop graph reasoning engine.

    Locates anchor nodes, performs BFS impact tracing, computes chain
    metrics via safe formula evaluation, and generates grounded
    human-readable conclusions.
    """

    # Anchor-discovery heuristics: keyword -> list of meta_class candidates
    _ANCHOR_PATTERNS: dict[str, list[str]] = {
        "supplier": ["Supplier", "SupplierParty", "Vendor"],
        "delay": ["Supplier", "DeliveryUnit", "DeliveryUnitBOM"],
        "delivery": ["DeliveryUnit", "DeliveryUnitBOM", "Delivery"],
        "part": ["Part", "MergedPart", "ConsumedPart", "PartFamily"],
        "bom": ["MBOM", "BOM", "PlantMBOM", "ProductMBOM"],
        "station": ["BOPStation", "Station", "BOP_Process", "WorkCenter"],
        "bop": ["BOP", "BOPStation", "BOP_Process"],
        "vin": ["VIN", "Vehicle", "VehicleInstance"],
        "inventory": ["Inventory", "CycleStock", "SafetyStock"],
        "line": ["AssemblyLine", "ProductionLine", "Line"],
    }

    def __init__(self, graph: OntologyGraph):
        self.graph = graph

    # -- anchor discovery -------------------------------------------------

    def _identify_anchor(self, question: str) -> dict[str, Any] | None:
        text = (question or "").lower()
        if not text:
            return None

        # 1. Direct name match: scan node names for tokens in the question
        best_match: dict[str, Any] | None = None
        best_len = 0
        for nid, node in self.graph.nodes.items():
            name = _node_name(node)
            if name and name.lower() in text and len(name) > best_len:
                best_len = len(name)
                best_match = node
        if best_match:
            return best_match

        # 2. Pattern-based class lookup
        for keyword, classes in self._ANCHOR_PATTERNS.items():
            if keyword in text:
                for cls in classes:
                    candidates = self.graph.find_nodes_by_class(cls)
                    if candidates:
                        return candidates[0]

        # 3. Layer fallback: supplier delay often originates at L2
        for layer in ("L2", "L2.5"):
            layer_nodes = self.graph.find_nodes_by_layer(layer)
            for node in layer_nodes:
                name = _node_name(node).lower()
                if any(kw in name for kw in ("supplier", "delivery", "part")):
                    return node
        return None

    # -- impact tracing ---------------------------------------------------

    async def trace_impact(
        self,
        anchor_node_id: str,
        max_hops: int = 5,
        edge_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """BFS traversal from an anchor node.

        Returns structured result with paths found, affected nodes
        (deduplicated with minimum hop distance), formatted impact chain,
        and summary metrics.
        """
        if anchor_node_id not in self.graph.nodes:
            return {
                "anchor_node_id": anchor_node_id,
                "paths_found": [],
                "affected_nodes": [],
                "impact_chain": [],
                "metrics": {
                    "total_paths": 0,
                    "max_depth_reached": 0,
                    "affected_count": 0,
                },
            }

        # Run BFS in a thread to keep the event loop responsive for large graphs
        paths = await asyncio.get_event_loop().run_in_executor(
            None, _bfs_traverse, self.graph, anchor_node_id, max_hops, edge_types
        )

        # Deduplicate affected nodes, recording minimum hop distance
        min_hops: dict[str, int] = {}
        for path in paths:
            for hop, nid in enumerate(path):
                if nid == anchor_node_id:
                    continue
                if nid not in min_hops or hop < min_hops[nid]:
                    min_hops[nid] = hop

        affected_nodes: list[dict[str, Any]] = []
        for nid, hop in sorted(min_hops.items(), key=lambda kv: (kv[1], kv[0])):
            node = self.graph.get_node(nid)
            affected_nodes.append({
                "id": nid,
                "name": _node_name(node),
                "layer": _node_layer(node),
                "meta_class": str(node.get("meta_class") or node.get("entity_type") or ""),
                "hop": hop,
            })

        anchor_node = self.graph.get_node(anchor_node_id)
        anchor_info = {
            "id": anchor_node_id,
            "name": _node_name(anchor_node),
            "layer": _node_layer(anchor_node),
        }

        # Build formatted impact chain (top N distinct paths by depth)
        ranked_paths = sorted(paths, key=lambda p: -len(p))[:20]
        impact_chain = [_format_path(self.graph, p) for p in ranked_paths]

        max_depth = max((len(p) - 1 for p in paths), default=0)

        return {
            "anchor_node": anchor_info,
            "paths_found": paths[:100],
            "affected_nodes": affected_nodes,
            "impact_chain": impact_chain,
            "metrics": {
                "total_paths": len(paths),
                "max_depth_reached": max_depth,
                "affected_count": len(affected_nodes),
            },
        }

    # -- natural-language query ------------------------------------------

    async def query(self, question: str, anchor_entity: str = None) -> dict[str, Any]:
        """Parse a natural-language question, trace impact, conclude.

        Steps:
          1. Locate anchor entity (explicit or inferred from question).
          2. Execute multi-hop BFS traversal (default max_hops=5).
          3. Generate a grounded, human-readable conclusion.
        """
        anchor_node: dict[str, Any] | None = None

        if anchor_entity:
            candidates = self.graph.find_nodes_by_name(anchor_entity)
            if candidates:
                anchor_node = candidates[0]

        if anchor_node is None:
            anchor_node = self._identify_anchor(question)

        if anchor_node is None:
            return {
                "query": question,
                "anchor_node": None,
                "paths_found": [],
                "affected_nodes": [],
                "conclusion": "Unable to identify an anchor entity for this query.",
                "metrics": {
                    "total_paths": 0,
                    "max_depth_reached": 0,
                    "affected_count": 0,
                },
            }

        anchor_id = _node_id(anchor_node)
        trace = await self.trace_impact(anchor_id, max_hops=5)

        conclusion = self.generate_conclusion(
            trace.get("paths_found", []),
            trace.get("metrics", {}),
            question,
        )

        return {
            "query": question,
            "anchor_node": trace.get("anchor_node", {}),
            "paths_found": trace.get("paths_found", []),
            "affected_nodes": trace.get("affected_nodes", []),
            "impact_chain": trace.get("impact_chain", []),
            "conclusion": conclusion,
            "metrics": trace.get("metrics", {}),
        }

    # -- chain metric computation ----------------------------------------

    def compute_chain_metric(self, path: list[str], formula: str) -> float:
        """Execute a calculation along a path.

        The formula is a safe arithmetic expression whose variable names
        are resolved by scanning node attributes/fields along the path
        (first numeric match wins). Example: "stock_on_hand / shift_demand".
        """
        if not path or not formula:
            return 0.0

        variables: dict[str, float] = {}
        for nid in path:
            node = self.graph.get_node(nid)
            if not node:
                continue
            attrs = _node_attributes(node)
            for key, value in attrs.items():
                if key in variables:
                    continue
                try:
                    variables[key] = float(value)
                except (TypeError, ValueError):
                    continue

        try:
            tree = ast.parse(formula, mode="eval")
            result = _safe_eval(tree, variables)
            return round(float(result), 4)
        except (KeyError, ValueError, SyntaxError, ZeroDivisionError):
            return 0.0

    # -- conclusion generation -------------------------------------------

    def generate_conclusion(self, paths: list[list[str]], metrics: dict[str, Any], question: str) -> str:
        """Generate a human-readable conclusion from traversal results."""
        affected_count = int(metrics.get("affected_count", 0))
        total_paths = int(metrics.get("total_paths", 0))
        max_depth = int(metrics.get("max_depth_reached", 0))

        if affected_count == 0:
            return f"No impacted entities found for query: '{question}'."

        # Aggregate affected nodes by layer/class
        layer_counts: dict[str, int] = {}
        class_counts: dict[str, int] = {}
        seen: set[str] = set()
        for path in paths:
            for nid in path[1:]:
                if nid in seen:
                    continue
                seen.add(nid)
                node = self.graph.get_node(nid)
                layer = _node_layer(node)
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
                cls = str(node.get("meta_class") or node.get("entity_type") or "entity")
                class_counts[cls] = class_counts.get(cls, 0) + 1

        layer_summary = ", ".join(
            f"{count} {layer} entities" for layer, count in sorted(layer_counts.items())
        )

        top_classes = sorted(class_counts.items(), key=lambda kv: -kv[1])[:5]
        class_summary = ", ".join(f"{count} {cls}" for cls, count in top_classes)

        anchor_name = ""
        if paths:
            anchor_node = self.graph.get_node(paths[0][0])
            anchor_name = _node_name(anchor_node) or paths[0][0]

        parts: list[str] = []
        if anchor_name:
            parts.append(f"Impact trace from '{anchor_name}'")
        parts.append(f"reached {affected_count} entities across {total_paths} paths (max depth {max_depth}).")
        if layer_summary:
            parts.append(f"Layer breakdown: {layer_summary}.")
        if class_summary:
            parts.append(f"Top affected types: {class_summary}.")
        parts.append(f"Query: \"{question}\"")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Factory + pipeline entry point
# ---------------------------------------------------------------------------

def _load_master_payload(master_path: Path) -> dict[str, Any]:
    """Load master ontology from a file or directory path."""
    if MasterOntologyStore and master_path.is_file():
        try:
            store = MasterOntologyStore(str(master_path))
            return store.read()
        except Exception:
            pass

    if master_path.is_dir():
        return {
            "nodes": _read_json(master_path / "nodes.json", []),
            "instances": _read_json(master_path / "instances.json", []),
            "relationships": _read_json(master_path / "relationships.json", []),
            "metadata": _read_json(master_path / "metadata.json", {}),
        }
    return _read_json(master_path, {})


def build_graph_from_master(master_path: str | Path) -> OntologyGraph:
    """Factory: build an OntologyGraph from a master ontology file/directory."""
    path = Path(master_path)
    master = _load_master_payload(path)
    nodes = _iter_master_entities(master)
    relationships = master.get("relationships") or master.get("edges") or []
    return OntologyGraph(nodes, relationships)


def _default_anchor_candidates(graph: OntologyGraph) -> list[dict[str, Any]]:
    """Pick a small set of representative anchor nodes for default queries."""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    priority_classes = [
        "Supplier", "SupplierParty", "Vendor",
        "DeliveryUnit", "DeliveryUnitBOM",
        "Part", "MergedPart", "ConsumedPart",
        "MBOM", "PlantMBOM",
    ]
    for cls in priority_classes:
        for node in graph.find_nodes_by_class(cls):
            nid = _node_id(node)
            if nid and nid not in seen:
                seen.add(nid)
                candidates.append(node)
            if len(candidates) >= 5:
                return candidates
    # Fallback: first L2 node
    if not candidates:
        for layer in ("L2", "L2.5", "L1"):
            layer_nodes = graph.find_nodes_by_layer(layer)
            if layer_nodes:
                candidates.append(layer_nodes[0])
                break
    return candidates


async def run_emergent_reasoning(args: Namespace) -> StageResult:
    """Pipeline entry point for Step 4: Emergent Reasoning.

    Loads the master ontology, builds the graph, runs configurable or
    default reasoning queries, and persists the results.
    """
    start = time.time()
    master_path = Path(args.master_path)
    if not master_path.exists():
        return StageResult(
            "emergent_reasoning",
            False,
            f"master not found: {master_path}",
            elapsed_seconds=round(time.time() - start, 3),
        )

    try:
        graph = build_graph_from_master(master_path)
    except Exception as exc:
        return StageResult(
            "emergent_reasoning",
            False,
            f"failed to build graph: {exc}",
            elapsed_seconds=round(time.time() - start, 3),
        )

    engine = EmergentReasoningEngine(graph)
    max_hops = int(getattr(args, "max_hops", 5) or 5)

    # Configurable queries: args.reasoning_queries (list of dicts) or defaults
    raw_queries = getattr(args, "reasoning_queries", None)
    queries: list[dict[str, Any]] = []
    if isinstance(raw_queries, list):
        for q in raw_queries:
            if isinstance(q, dict) and q.get("question"):
                queries.append(q)
            elif isinstance(q, str):
                queries.append({"question": q})

    if not queries:
        anchors = _default_anchor_candidates(graph)
        for anchor in anchors:
            name = _node_name(anchor)
            queries.append({
                "question": f"What is the impact of {name}?",
                "anchor_entity": name,
            })

    results: list[dict[str, Any]] = []
    for q in queries[:20]:
        question = q.get("question", "")
        anchor_entity = q.get("anchor_entity")
        result = await engine.query(question, anchor_entity=anchor_entity)
        if q.get("formula") and result.get("paths_found"):
            for path in result["paths_found"][:5]:
                metric = engine.compute_chain_metric(path, q["formula"])
                result.setdefault("computed_metrics", []).append({
                    "path": _format_path(graph, path),
                    "formula": q["formula"],
                    "value": metric,
                })
        results.append(result)

    summary = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "master_path": str(master_path),
        "node_count": len(graph),
        "query_count": len(results),
        "results": results,
    }

    output_dir = Path(getattr(args, "output_dir", str(PROJECT_ROOT / "outputs" / "runs" / "ontology")))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "emergent_reasoning_results.json"
    _write_json(results_path, summary)

    total_affected = sum(r.get("metrics", {}).get("affected_count", 0) for r in results)

    return StageResult(
        "emergent_reasoning",
        True,
        f"emergent reasoning completed ({len(results)} queries, {total_affected} affected entities)",
        {
            "summary": summary,
            "results_path": str(results_path),
            "node_count": len(graph),
            "query_count": len(results),
            "total_affected": total_affected,
            "max_hops": max_hops,
        },
        round(time.time() - start, 3),
    )


if __name__ == "__main__":
    # Standalone CLI: python -m src.pipeline.stages.emergent_reasoning --master-path <path>
    import argparse

    parser = argparse.ArgumentParser(description="Emergent Reasoning Engine (步骤4)")
    parser.add_argument("--master-path", required=True, help="Path to master ontology file or directory")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "runs" / "ontology"))
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--question", help="Optional single question to query")
    cli_args = parser.parse_args()

    if cli_args.question:
        cli_args.reasoning_queries = [{"question": cli_args.question}]

    result = asyncio.run(run_emergent_reasoning(cli_args))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
