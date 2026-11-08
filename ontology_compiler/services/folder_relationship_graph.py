"""Folder Relationship Graph Cache — Multi-folder Linkage Support.

[DIAG-3.3] Cache folder-level relationships for incremental cross-folder analysis.

This module provides:
- Folder-to-folder relationship graph with entity and topic overlap
- Shared entity tracking across folders
- Upstream/downstream dependency detection
- Incremental graph updates for multi-folder analysis
- Support for collaborative multi-folder analysis reusing cached relationships
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FolderNode:
    """Represents a folder in the relationship graph."""
    folder_path: str
    folder_name: str
    entity_ids: Set[str] = field(default_factory=set)
    topic_keywords: Set[str] = field(default_factory=set)
    file_count: int = 0
    last_modified: float = 0.0
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "folder_path": self.folder_path,
            "folder_name": self.folder_name,
            "entity_ids": sorted(self.entity_ids),
            "topic_keywords": sorted(self.topic_keywords),
            "file_count": self.file_count,
            "last_modified": self.last_modified,
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FolderNode":
        return cls(
            folder_path=data["folder_path"],
            folder_name=data["folder_name"],
            entity_ids=set(data.get("entity_ids", [])),
            topic_keywords=set(data.get("topic_keywords", [])),
            file_count=data.get("file_count", 0),
            last_modified=data.get("last_modified", 0.0),
            version=data.get("version", 1),
        )


@dataclass
class FolderEdge:
    """Represents a relationship between two folders."""
    source_folder: str
    target_folder: str
    edge_type: str  # shared_entity, shared_topic, depends_on, references
    weight: float  # 0.0 - 1.0
    shared_entities: List[str] = field(default_factory=list)
    shared_topics: List[str] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_folder": self.source_folder,
            "target_folder": self.target_folder,
            "edge_type": self.edge_type,
            "weight": round(self.weight, 3),
            "shared_entities": sorted(self.shared_entities),
            "shared_topics": sorted(self.shared_topics),
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FolderEdge":
        return cls(
            source_folder=data["source_folder"],
            target_folder=data["target_folder"],
            edge_type=data["edge_type"],
            weight=data["weight"],
            shared_entities=list(data.get("shared_entities", [])),
            shared_topics=list(data.get("shared_topics", [])),
            confidence=data.get("confidence", 0.0),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class FolderGraphMetrics:
    """Metrics for the folder relationship graph."""
    folder_count: int = 0
    edge_count: int = 0
    avg_connectivity: float = 0.0
    max_cluster_size: int = 0
    isolated_folder_count: int = 0
    bridge_folder_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "folder_count": self.folder_count,
            "edge_count": self.edge_count,
            "avg_connectivity": round(self.avg_connectivity, 3),
            "max_cluster_size": self.max_cluster_size,
            "isolated_folder_count": self.isolated_folder_count,
            "bridge_folder_count": self.bridge_folder_count,
            "timestamp": self.timestamp.isoformat(),
        }


class FolderRelationshipGraph:
    """
    [DIAG-3.3] Folder-level relationship graph for multi-folder analysis.
    
    Features:
    - Folder-to-folder relationship tracking
    - Shared entity/topic detection
    - Upstream/downstream dependency discovery
    - Incremental graph updates
    - Bridge folder detection
    - Cluster analysis for related folders
    """
    
    SHARED_ENTITY_THRESHOLD = 1  # Minimum shared entities to create edge
    SHARED_TOPIC_THRESHOLD = 2   # Minimum shared topics to create edge
    SIMILARITY_THRESHOLD = 0.3   # Minimum similarity for topic matching
    
    def __init__(self):
        self._nodes: Dict[str, FolderNode] = {}  # folder_path -> FolderNode
        self._edges: Dict[Tuple[str, str], FolderEdge] = {}  # (src, tgt) -> FolderEdge
        self._entity_to_folders: Dict[str, Set[str]] = {}  # entity_id -> folder_paths
        self._topic_to_folders: Dict[str, Set[str]] = {}   # topic -> folder_paths
        self._version: int = 1
        self._last_updated: datetime = datetime.now()
    
    def add_folder(self, folder_node: FolderNode) -> None:
        """
        Add or update a folder node.
        
        Args:
            folder_node: FolderNode to add
        """
        folder_path = folder_node.folder_path
        
        # Update version if folder already exists
        if folder_path in self._nodes:
            folder_node.version = self._nodes[folder_path].version + 1
        
        self._nodes[folder_path] = folder_node
        
        # Update entity index
        for entity_id in folder_node.entity_ids:
            if entity_id not in self._entity_to_folders:
                self._entity_to_folders[entity_id] = set()
            self._entity_to_folders[entity_id].add(folder_path)
        
        # Update topic index
        for topic in folder_node.topic_keywords:
            if topic not in self._topic_to_folders:
                self._topic_to_folders[topic] = set()
            self._topic_to_folders[topic].add(folder_path)
        
        self._last_updated = datetime.now()
        logger.info(f"Added/updated folder node: {folder_path} (v{folder_node.version})")
    
    def remove_folder(self, folder_path: str) -> None:
        """
        Remove a folder and its associated edges.
        
        Args:
            folder_path: Path of folder to remove
        """
        if folder_path not in self._nodes:
            logger.warning(f"Folder not found in graph: {folder_path}")
            return
        
        folder_node = self._nodes[folder_path]
        
        # Remove from entity index
        for entity_id in folder_node.entity_ids:
            if entity_id in self._entity_to_folders:
                self._entity_to_folders[entity_id].discard(folder_path)
                if not self._entity_to_folders[entity_id]:
                    del self._entity_to_folders[entity_id]
        
        # Remove from topic index
        for topic in folder_node.topic_keywords:
            if topic in self._topic_to_folders:
                self._topic_to_folders[topic].discard(folder_path)
                if not self._topic_to_folders[topic]:
                    del self._topic_to_folders[topic]
        
        # Remove associated edges
        edges_to_remove = [
            key for key in self._edges
            if folder_path in key
        ]
        for key in edges_to_remove:
            del self._edges[key]
        
        # Remove node
        del self._nodes[folder_path]
        
        self._last_updated = datetime.now()
        logger.info(f"Removed folder node: {folder_path}")
    
    def build_edges(self) -> List[FolderEdge]:
        """
        Build edges between folders based on shared entities and topics.
        
        Returns:
            List of newly created edges
        """
        new_edges = []
        folder_paths = list(self._nodes.keys())
        
        for i, src_path in enumerate(folder_paths):
            src_node = self._nodes[src_path]
            
            for tgt_path in folder_paths[i+1:]:
                tgt_node = self._nodes[tgt_path]
                
                # Skip self-loops
                if src_path == tgt_path:
                    continue
                
                # Check shared entities
                shared_entities = src_node.entity_ids & tgt_node.entity_ids
                if len(shared_entities) >= self.SHARED_ENTITY_THRESHOLD:
                    edge = FolderEdge(
                        source_folder=src_path,
                        target_folder=tgt_path,
                        edge_type="shared_entity",
                        weight=min(len(shared_entities) / 10.0, 1.0),
                        shared_entities=sorted(shared_entities),
                        confidence=min(len(shared_entities) / 5.0, 1.0),
                    )
                    self._edges[(src_path, tgt_path)] = edge
                    new_edges.append(edge)
                
                # Check shared topics
                shared_topics = src_node.topic_keywords & tgt_node.topic_keywords
                if len(shared_topics) >= self.SHARED_TOPIC_THRESHOLD:
                    edge = FolderEdge(
                        source_folder=src_path,
                        target_folder=tgt_path,
                        edge_type="shared_topic",
                        weight=min(len(shared_topics) / 5.0, 1.0),
                        shared_topics=sorted(shared_topics),
                        confidence=min(len(shared_topics) / 3.0, 1.0),
                    )
                    # Store both directions for undirected relationship
                    self._edges[(src_path, tgt_path)] = edge
                    new_edges.append(edge)
                
                # Detect dependency (file references)
                # This would be enhanced with actual file dependency parsing
                if self._has_dependency_hint(src_node, tgt_node):
                    edge = FolderEdge(
                        source_folder=src_path,
                        target_folder=tgt_path,
                        edge_type="depends_on",
                        weight=0.5,
                        confidence=0.6,
                    )
                    self._edges[(src_path, tgt_path)] = edge
                    new_edges.append(edge)
        
        self._last_updated = datetime.now()
        logger.info(f"Built {len(new_edges)} folder edges")
        return new_edges
    
    def _has_dependency_hint(self, src: FolderNode, tgt: FolderNode) -> bool:
        """Check if there are hints of dependency between folders."""
        # Simple heuristic: similar folder names suggest dependency
        src_name = src.folder_name.lower()
        tgt_name = tgt.folder_name.lower()
        
        # Check for common dependency patterns
        dependency_keywords = ["depends", "requires", "uses", "imports"]
        return any(kw in src_name or kw in tgt_name for kw in dependency_keywords)
    
    def get_related_folders(
        self,
        folder_path: str,
        min_weight: float = 0.0,
        edge_types: Optional[List[str]] = None,
    ) -> List[Tuple[str, float, str]]:
        """
        Get folders related to the given folder.
        
        Args:
            folder_path: Source folder path
            min_weight: Minimum edge weight to include
            edge_types: Filter by edge types (optional)
            
        Returns:
            List of (folder_path, weight, edge_type) tuples
        """
        related = []
        
        for (src, tgt), edge in self._edges.items():
            if src == folder_path:
                other = tgt
            elif tgt == folder_path:
                other = src
            else:
                continue
            
            if edge.weight < min_weight:
                continue
            
            if edge_types and edge.edge_type not in edge_types:
                continue
            
            related.append((other, edge.weight, edge.edge_type))
        
        # Sort by weight descending
        related.sort(key=lambda x: x[1], reverse=True)
        return related
    
    def get_shared_entities(self, folder_a: str, folder_b: str) -> List[str]:
        """Get entities shared between two folders."""
        edge = self._edges.get((folder_a, folder_b)) or self._edges.get((folder_b, folder_a))
        if edge:
            return edge.shared_entities
        return []
    
    def get_folder_clusters(self) -> List[Set[str]]:
        """
        Find clusters of related folders using connected components.
        
        Returns:
            List of folder path sets (clusters)
        """
        # Union-Find algorithm
        parent = {path: path for path in self._nodes}
        
        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union connected folders
        for (src, tgt) in self._edges:
            union(src, tgt)
        
        # Group by root
        clusters: Dict[str, Set[str]] = {}
        for path in self._nodes:
            root = find(path)
            if root not in clusters:
                clusters[root] = set()
            clusters[root].add(path)
        
        return list(clusters.values())
    
    def get_bridge_folders(self) -> List[str]:
        """
        Find bridge folders (folders connecting multiple clusters).
        
        Returns:
            List of folder paths that are bridges
        """
        clusters = self.get_folder_clusters()
        if len(clusters) < 2:
            return []
        
        # Find folders that connect to folders in other clusters
        bridge_folders = []
        folder_to_cluster = {}
        for i, cluster in enumerate(clusters):
            for folder in cluster:
                folder_to_cluster[folder] = i
        
        for folder_path in self._nodes:
            connected_clusters = set()
            for (src, tgt), edge in self._edges.items():
                if src == folder_path:
                    connected_clusters.add(folder_to_cluster.get(tgt))
                elif tgt == folder_path:
                    connected_clusters.add(folder_to_cluster.get(src))
            
            # Remove None and own cluster
            connected_clusters = {c for c in connected_clusters if c is not None}
            own_cluster = folder_to_cluster.get(folder_path)
            connected_clusters.discard(own_cluster)
            
            if len(connected_clusters) >= 1:
                bridge_folders.append(folder_path)
        
        return bridge_folders
    
    def compute_metrics(self) -> FolderGraphMetrics:
        """Compute graph metrics."""
        folder_count = len(self._nodes)
        edge_count = len(self._edges)
        
        # Average connectivity
        if folder_count > 0:
            avg_connectivity = edge_count * 2 / folder_count
        else:
            avg_connectivity = 0.0
        
        # Cluster analysis
        clusters = self.get_folder_clusters()
        max_cluster_size = max(len(c) for c in clusters) if clusters else 0
        
        # Isolated folders
        isolated = [c for c in clusters if len(c) == 1]
        isolated_count = len(isolated)
        
        # Bridge folders
        bridge_count = len(self.get_bridge_folders())
        
        return FolderGraphMetrics(
            folder_count=folder_count,
            edge_count=edge_count,
            avg_connectivity=avg_connectivity,
            max_cluster_size=max_cluster_size,
            isolated_folder_count=isolated_count,
            bridge_folder_count=bridge_count,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "version": self._version,
            "last_updated": self._last_updated.isoformat(),
            "nodes": {path: node.to_dict() for path, node in self._nodes.items()},
            "edges": [edge.to_dict() for edge in self._edges.values()],
            "metrics": self.compute_metrics().to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FolderRelationshipGraph":
        """Deserialize graph from dictionary."""
        graph = cls()
        graph._version = data.get("version", 1)
        graph._last_updated = datetime.fromisoformat(data.get("last_updated", datetime.now().isoformat()))
        
        # Restore nodes
        for path, node_data in data.get("nodes", {}).items():
            graph._nodes[path] = FolderNode.from_dict(node_data)
            # Rebuild indexes
            for entity_id in graph._nodes[path].entity_ids:
                if entity_id not in graph._entity_to_folders:
                    graph._entity_to_folders[entity_id] = set()
                graph._entity_to_folders[entity_id].add(path)
            for topic in graph._nodes[path].topic_keywords:
                if topic not in graph._topic_to_folders:
                    graph._topic_to_folders[topic] = set()
                graph._topic_to_folders[topic].add(path)
        
        # Restore edges
        for edge_data in data.get("edges", []):
            edge = FolderEdge.from_dict(edge_data)
            graph._edges[(edge.source_folder, edge.target_folder)] = edge
        
        return graph
    
    def save(self, path: Path) -> None:
        """Save graph to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved folder relationship graph to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "FolderRelationshipGraph":
        """Load graph from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        graph = cls.from_dict(data)
        logger.info(f"Loaded folder relationship graph from {path}")
        return graph
    
    def get_folder_paths(self) -> List[str]:
        """Get all folder paths in graph."""
        return list(self._nodes.keys())
    
    def get_edge_count(self) -> int:
        """Get total edge count."""
        return len(self._edges)
    
    def get_neighbors(self, folder_path: str) -> List[str]:
        """Get all neighboring folder paths."""
        neighbors = set()
        for (src, tgt) in self._edges:
            if src == folder_path:
                neighbors.add(tgt)
            elif tgt == folder_path:
                neighbors.add(src)
        return sorted(neighbors)


class FolderGraphCache:
    """
    Cache manager for folder relationship graphs.
    
    Provides:
    - Persistent storage of folder graphs
    - Incremental updates
    - Multi-run graph merging
    """
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._graphs: Dict[str, FolderRelationshipGraph] = {}
    
    def _get_graph_path(self, graph_id: str) -> Path:
        """Get file path for a graph."""
        return self.cache_dir / f"{graph_id}.json"
    
    def get_or_create(self, graph_id: str) -> FolderRelationshipGraph:
        """
        Get existing graph or create new one.
        
        Args:
            graph_id: Unique graph identifier (e.g., project name)
            
        Returns:
            FolderRelationshipGraph instance
        """
        if graph_id in self._graphs:
            return self._graphs[graph_id]
        
        graph_path = self._get_graph_path(graph_id)
        if graph_path.exists():
            graph = FolderRelationshipGraph.load(graph_path)
        else:
            graph = FolderRelationshipGraph()
        
        self._graphs[graph_id] = graph
        return graph
    
    def save_graph(self, graph_id: str) -> None:
        """Save graph to cache."""
        if graph_id not in self._graphs:
            logger.warning(f"Graph not found: {graph_id}")
            return
        
        graph_path = self._get_graph_path(graph_id)
        self._graphs[graph_id].save(graph_path)
    
    def incremental_update(
        self,
        graph_id: str,
        folder_nodes: List[FolderNode],
    ) -> FolderRelationshipGraph:
        """
        Incrementally update graph with new folder nodes.
        
        Args:
            graph_id: Graph identifier
            folder_nodes: New or updated folder nodes
            
        Returns:
            Updated FolderRelationshipGraph
        """
        graph = self.get_or_create(graph_id)
        
        # Add/update nodes
        for node in folder_nodes:
            graph.add_folder(node)
        
        # Rebuild edges
        graph.build_edges()
        
        # Save updated graph
        self.save_graph(graph_id)
        
        return graph
    
    def get_graph_metrics(self, graph_id: str) -> Optional[FolderGraphMetrics]:
        """Get metrics for a graph."""
        graph = self.get_or_create(graph_id)
        return graph.compute_metrics()
    
    def list_cached_graphs(self) -> List[str]:
        """List all cached graph IDs."""
        return [f.stem for f in self.cache_dir.glob("*.json")]


def demo_folder_graph():
    """Demonstrate folder relationship graph usage."""
    import tempfile
    
    # Create graph
    graph = FolderRelationshipGraph()
    
    # Add folders
    folder_a = FolderNode(
        folder_path="/project/inventory",
        folder_name="inventory",
        entity_ids={"E001", "E002", "E003"},
        topic_keywords={"inventory", "stock", "warehouse"},
        file_count=10,
    )
    
    folder_b = FolderNode(
        folder_path="/project/supply_chain",
        folder_name="supply_chain",
        entity_ids={"E002", "E003", "E004"},
        topic_keywords={"supply", "chain", "logistics"},
        file_count=8,
    )
    
    folder_c = FolderNode(
        folder_path="/project/finance",
        folder_name="finance",
        entity_ids={"E005", "E006"},
        topic_keywords={"budget", "cost", "revenue"},
        file_count=5,
    )
    
    graph.add_folder(folder_a)
    graph.add_folder(folder_b)
    graph.add_folder(folder_c)
    
    # Build edges
    edges = graph.build_edges()
    print(f"Created {len(edges)} edges")
    
    # Get related folders
    related = graph.get_related_folders("/project/inventory")
    print(f"\nFolders related to inventory:")
    for folder, weight, edge_type in related:
        print(f"  - {folder} (weight={weight:.2f}, type={edge_type})")
    
    # Get clusters
    clusters = graph.get_folder_clusters()
    print(f"\nFolder clusters: {len(clusters)}")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: {cluster}")
    
    # Get metrics
    metrics = graph.compute_metrics()
    print(f"\nGraph metrics: {metrics.to_dict()}")
    
    # Save and load
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
        temp_path = Path(f.name)
    
    graph.save(temp_path)
    loaded = FolderRelationshipGraph.load(temp_path)
    print(f"\nLoaded graph with {len(loaded.get_folder_paths())} folders")
    
    # Clean up
    temp_path.unlink()


if __name__ == "__main__":
    demo_folder_graph()