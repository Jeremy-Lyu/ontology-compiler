#!/usr/bin/env python3
"""Analyze L0->L3 hierarchy distribution and relationships."""
import json
from collections import Counter, defaultdict
from pathlib import Path

MASTER = Path("outputs/runs/ontology/master_ontology_unified.json")

with open(MASTER, "r", encoding="utf-8") as f:
    master = json.load(f)

nodes = master.get("nodes", [])
rels = master.get("relationships", [])
instances = master.get("instances", [])

node_map = {n.get("meta_id", ""): n for n in nodes if isinstance(n, dict)}
layer_map = {n.get("meta_id", ""): n.get("layer", "") for n in nodes if isinstance(n, dict)}

# --- Layer distribution ---
print("=" * 70)
print("  L0 -> L3 数量及分布")
print("=" * 70)
print()

class_by_layer = defaultdict(Counter)
for n in nodes:
    if isinstance(n, dict):
        class_by_layer[n.get("layer", "")][n.get("meta_class", "UNKNOWN")] += 1

print(f"  {'Layer':<6} | {'Count':>5} | Meta Class Distribution")
print(f"  {'-'*6} | {'-'*5} | {'-'*40}")
for layer in ["L0", "L1", "L2", "L2.5"]:
    counts = class_by_layer[layer]
    total = sum(counts.values())
    cls_str = ", ".join(f"{c}: {n}" for c, n in counts.most_common())
    print(f"  {layer:<6} | {total:>5} | {cls_str}")
print(f"  {'L3':<6} | {len(instances):>5} | (stored in instances field)")
print()

# --- L0 detail ---
print("=" * 70)
print("  L0 Entities (Root Meta-Model)")
print("=" * 70)
l0_nodes = [n for n in nodes if isinstance(n, dict) and n.get("layer") == "L0"]
for n in sorted(l0_nodes, key=lambda x: x.get("meta_class", "")):
    nid = n.get("meta_id", "")
    name = n.get("meta_name", n.get("name", ""))
    cls = n.get("meta_class", "")
    children = sum(1 for r in rels if isinstance(r, dict) and r.get("target") == nid)
    print(f"  {nid:10} | {cls:12} | {name[:40]:40} | in_edges: {children}")
print()

# --- L1 detail ---
print("=" * 70)
print("  L1 Entities (Abstract Concepts)")
print("=" * 70)
l1_nodes = [n for n in nodes if isinstance(n, dict) and n.get("layer") == "L1"]
for n in sorted(l1_nodes, key=lambda x: (x.get("meta_class", ""), x.get("meta_name", ""))):
    nid = n.get("meta_id", "")
    name = n.get("meta_name", n.get("name", ""))
    cls = n.get("meta_class", "")
    parent = n.get("parent_id", "")
    children_count = sum(1 for r in rels if isinstance(r, dict) and r.get("target") == nid)
    out_rels = sum(1 for r in rels if isinstance(r, dict) and r.get("source") == nid)
    print(f"  {nid:10} | {cls:12} | {name[:35]:35} | parent: {parent:10} | in: {children_count:2} | out: {out_rels:2}")
print(f"  Total L1: {len(l1_nodes)}")
print()

# --- L2 summary ---
print("=" * 70)
print("  L2 Entities Summary (Specific Systems/Products)")
print("=" * 70)
l2_nodes = [n for n in nodes if isinstance(n, dict) and n.get("layer") == "L2"]
l2_by_class = Counter(n.get("meta_class", "") for n in l2_nodes)
print(f"  Total L2: {len(l2_nodes)}")
for cls, count in l2_by_class.most_common():
    samples = [n.get("meta_name", n.get("name", "")) for n in l2_nodes if n.get("meta_class") == cls][:5]
    print(f"  {cls:12} | {count:4} | samples: {' | '.join(s[:25] for s in samples[:3])}")
print()

# --- L2.5 summary ---
print("=" * 70)
print("  L2.5 Entities Summary (Mapping Rules)")
print("=" * 70)
l25_nodes = [n for n in nodes if isinstance(n, dict) and n.get("layer") == "L2.5"]
l25_by_class = Counter(n.get("meta_class", "") for n in l25_nodes)
print(f"  Total L2.5: {len(l25_nodes)}")
for cls, count in l25_by_class.most_common():
    samples = [n.get("meta_name", n.get("name", "")) for n in l25_nodes if n.get("meta_class") == cls][:5]
    print(f"  {cls:12} | {count:4} | samples: {' | '.join(s[:25] for s in samples[:3])}")
print()

# --- L3 instances ---
print("=" * 70)
print("  L3 Instances")
print("=" * 70)
print(f"  Total L3 instances: {len(instances)}")
for inst in instances[:15]:
    iid = inst.get("meta_id", inst.get("instance_id", ""))
    name = inst.get("meta_name", inst.get("name", ""))
    cls = inst.get("meta_class", "")
    parent = inst.get("parent_id", "")
    print(f"  {iid:15} | {cls:12} | {name[:35]:35} | parent: {parent}")
if len(instances) > 15:
    print(f"  ... and {len(instances) - 15} more")
print()

# --- Relationship analysis ---
print("=" * 70)
print("  Relationship Analysis")
print("=" * 70)
rel_types = Counter(r.get("relation_type", "UNKNOWN") for r in rels if isinstance(r, dict))
print(f"  Total relationships: {len(rels)}")
print()
print("  Relation Type Distribution:")
for rtype, count in rel_types.most_common():
    print(f"    {rtype:25} | {count:5} | {count/len(rels)*100:.1f}%")
print()

# --- Cross-layer ---
print("  Cross-Layer Relationship Distribution:")
cross_layer = Counter()
for r in rels:
    if not isinstance(r, dict):
        continue
    src = r.get("source", "")
    tgt = r.get("target", "")
    sl = layer_map.get(src, "")
    tl = layer_map.get(tgt, "")
    if sl and tl:
        cross_layer[(sl, tl)] += 1
for (sl, tl), count in cross_layer.most_common():
    direction = "same" if sl == tl else "cross"
    print(f"    {sl:5} -> {tl:5} | {count:5} | {direction}")
print()

# --- Topology ---
print("=" * 70)
print("  Topology Analysis")
print("=" * 70)

adj = defaultdict(list)
adj_rev = defaultdict(list)
for r in rels:
    if not isinstance(r, dict):
        continue
    src = r.get("source", "")
    tgt = r.get("target", "")
    if src and tgt:
        adj[src].append(tgt)
        adj_rev[tgt].append(src)

# Connected components (undirected)
visited = set()
components = []
for nid in node_map:
    if nid in visited:
        continue
    component = []
    queue = [nid]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        component.append(current)
        for neighbor in adj.get(current, []):
            if neighbor not in visited:
                queue.append(neighbor)
        for neighbor in adj_rev.get(current, []):
            if neighbor not in visited:
                queue.append(neighbor)
    components.append(component)

component_sizes = sorted([len(c) for c in components], reverse=True)
print(f"  Connected components: {len(components)}")
print(f"  Component sizes (top 10): {component_sizes[:10]}")
print(f"  Largest component: {component_sizes[0]} nodes")
print(f"  Single-node components: {sum(1 for s in component_sizes if s == 1)}")
print()

# Find longest chain from L0
def find_longest_chain(start_id, max_depth=5):
    longest = [start_id]
    children = adj.get(start_id, [])
    for child in children:
        if child not in [start_id]:
            sub = find_longest_chain(child, max_depth - 1)
            if len(sub) + 1 > len(longest):
                longest = [start_id] + sub
    return longest if max_depth > 0 else [start_id]

longest = []
for l0 in [n.get("meta_id", "") for n in l0_nodes]:
    chain = find_longest_chain(l0, 5)
    if len(chain) > len(longest):
        longest = chain

print(f"  Longest chain from L0: {len(longest)} hops")
if longest:
    chain_names = []
    for nid in longest:
        n = node_map.get(nid, {})
        layer = n.get("layer", "")
        name = n.get("meta_name", n.get("name", ""))
        chain_names.append(f"{layer}:{name}")
    print(f"  Chain: {' -> '.join(chain_names)}")
print()

# --- Potential issues ---
print("=" * 70)
print("  Potential Issues")
print("=" * 70)

# L0 without children
l0_no_children = [n for n in l0_nodes if not adj_rev.get(n.get("meta_id", ""))]
print(f"  L0 nodes without children: {len(l0_no_children)}")
for n in l0_no_children:
    print(f"    {n.get('meta_id', '')} | {n.get('meta_name', n.get('name', ''))}")

# L1 without parent
l1_no_parent = [n for n in l1_nodes if not n.get("parent_id")]
print(f"  L1 nodes without parent_id: {len(l1_no_parent)}")
for n in l1_no_parent[:5]:
    print(f"    {n.get('meta_id', '')} | {n.get('meta_name', n.get('name', ''))}")
if len(l1_no_parent) > 5:
    print(f"    ... and {len(l1_no_parent) - 5} more")

# L2 without parent
l2_no_parent = [n for n in l2_nodes if not n.get("parent_id")]
print(f"  L2 nodes without parent_id: {len(l2_no_parent)}")
for n in l2_no_parent[:5]:
    print(f"    {n.get('meta_id', '')} | {n.get('meta_name', n.get('name', ''))}")
if len(l2_no_parent) > 5:
    print(f"    ... and {len(l2_no_parent) - 5} more")

# L2.5 without parent
l25_no_parent = [n for n in l25_nodes if not n.get("parent_id")]
print(f"  L2.5 nodes without parent_id: {len(l25_no_parent)}")
for n in l25_no_parent[:5]:
    print(f"    {n.get('meta_id', '')} | {n.get('meta_name', n.get('name', ''))}")
if len(l25_no_parent) > 5:
    print(f"    ... and {len(l25_no_parent) - 5} more")

# Unknown relation types
unknown_rels = [r for r in rels if isinstance(r, dict) and r.get("relation_type", "") in ("UNKNOWN", "")]
print(f"  Unknown relation types: {len(unknown_rels)}")

# Ratio check
print()
print("=" * 70)
print("  Hierarchy Health Check")
print("=" * 70)
total_nodes = len(nodes)
l0_pct = len(l0_nodes) / total_nodes * 100
l1_pct = len(l1_nodes) / total_nodes * 100
l2_pct = len(l2_nodes) / total_nodes * 100
l25_pct = len(l25_nodes) / total_nodes * 100
print(f"  L0:   {len(l0_nodes):4} ({l0_pct:.1f}%)  - Root meta-model")
print(f"  L1:   {len(l1_nodes):4} ({l1_pct:.1f}%)  - Abstract concepts")
print(f"  L2:   {len(l2_nodes):4} ({l2_pct:.1f}%)  - Specific systems")
print(f"  L2.5: {len(l25_nodes):4} ({l25_pct:.1f}%)  - Mapping rules")
print(f"  L3:   {len(instances):4} ({len(instances)/total_nodes*100:.1f}%)  - Instances")
print()
print(f"  Ideal pyramid ratio L0:L1:L2:L2.5:L3 = 1:5:20:10:5")
print(f"  Actual pyramid ratio              = 1:{len(l1_nodes)//max(len(l0_nodes),1)}:{len(l2_nodes)//max(len(l0_nodes),1)}:{len(l25_nodes)//max(len(l0_nodes),1)}:{len(instances)//max(len(l0_nodes),1)}")
