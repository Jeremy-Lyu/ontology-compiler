"""Fragment normalization utilities for standardizing extraction outputs.

This module provides utilities to normalize Fragment dictionaries from different
extraction paths (TBH/RIGOR vs folder analysis) into the unified Fragment model.

Usage:
    from src.application.transformers import normalize_fragments

    # Normalize a list of legacy fragments
    standardized = normalize_fragments(legacy_fragments)

    # Access standardized data
    for fragment in standardized:
        print(fragment.to_dict())  # Unified format with nodes/relationships
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from src.domain.models import Fragment, FragmentNode, FragmentInstance, FragmentRelationship


class FragmentNormalizer:
    """Normalizes fragments from different extraction paths to unified format.

    Handles field name variations:
    - 'classes' (RIGOR) -> 'nodes' (unified)
    - 'edges' (folder analysis) -> 'relationships' (unified)
    - Mixed usage of 'nodes' across paths

    Attributes:
        deduplicate: Whether to deduplicate entities by ID during normalization
        track_stats: Whether to collect normalization statistics
    """

    def __init__(self, deduplicate: bool = True, track_stats: bool = True) -> None:
        self.deduplicate = deduplicate
        self.track_stats = track_stats
        self.stats: Dict[str, Any] = {
            "total_fragments": 0,
            "normalized_fragments": 0,
            "classes_converted": 0,
            "edges_converted": 0,
            "duplicates_removed": 0,
        }

    def normalize(self, fragment_data: Dict[str, Any]) -> Fragment:
        """Normalize a single fragment dictionary to unified Fragment model.

        Args:
            fragment_data: Raw fragment dictionary from any extraction path, or already a Fragment

        Returns:
            Standardized Fragment instance
        """
        if self.track_stats:
            self.stats["total_fragments"] += 1

        # Handle already-normalized Fragment objects
        if isinstance(fragment_data, Fragment):
            fragment = fragment_data
        else:
            # Use the legacy conversion method which handles all field variations
            fragment = Fragment.from_legacy_dict(fragment_data)

        if self.track_stats:
            self.stats["normalized_fragments"] += 1
            # Count conversions
            if "classes" in fragment_data:
                self.stats["classes_converted"] += len(fragment_data.get("classes", []))
            if "edges" in fragment_data and "relationships" not in fragment_data:
                self.stats["edges_converted"] += len(fragment_data.get("edges", []))

        return fragment

    def normalize_many(self, fragments: List[Dict[str, Any]]) -> List[Fragment]:
        """Normalize multiple fragments.

        Args:
            fragments: List of raw fragment dictionaries

        Returns:
            List of standardized Fragment instances
        """
        normalized = []
        for frag_data in fragments:
            normalized.append(self.normalize(frag_data))
        return normalized

    def merge_fragments(self, fragments: List[Fragment]) -> Fragment:
        """Merge multiple fragments into a single fragment with deduplication.

        Args:
            fragments: List of Fragment instances to merge

        Returns:
            Merged Fragment with deduplicated entities
        """
        if not fragments:
            return Fragment(source_file="")

        # Start with first fragment
        merged = fragments[0]

        # Merge remaining fragments
        for fragment in fragments[1:]:
            merged = merged.merge(fragment)

        if self.track_stats and self.deduplicate:
            # Calculate duplicates removed (approximate)
            total_entities = sum(f.get_stats()["total_entities"] for f in fragments)
            merged_entities = merged.get_stats()["total_entities"]
            self.stats["duplicates_removed"] += max(0, total_entities - merged_entities)

        return merged

    def to_supplement_format(self, fragment: Fragment) -> Dict[str, Any]:
        """Convert fragment to supplement format (used by downstream processors).

        The supplement format is the internal format used by batch_processor.py
        and other downstream components.

        Args:
            fragment: Fragment instance

        Returns:
            Dictionary in supplement format with nodes/instances/relationships
        """
        return {
            "schema_version": "v1.0",
            "generated_at": fragment.metadata.get("generated_at", ""),
            "nodes": [n.model_dump() for n in fragment.nodes],
            "instances": [i.model_dump() for i in fragment.instances],
            "relationships": [r.model_dump() for r in fragment.relationships],
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get normalization statistics."""
        return dict(self.stats)

    def reset_stats(self) -> None:
        """Reset normalization statistics."""
        self.stats = {
            "total_fragments": 0,
            "normalized_fragments": 0,
            "classes_converted": 0,
            "edges_converted": 0,
            "duplicates_removed": 0,
        }


def normalize_fragments(
    fragments: List[Dict[str, Any]],
    merge: bool = False,
    deduplicate: bool = True,
) -> List[Fragment] | Fragment:
    """Convenience function to normalize a list of fragment dictionaries.

    Args:
        fragments: List of raw fragment dictionaries from any extraction path
        merge: If True, merge all fragments into a single fragment
        deduplicate: Whether to deduplicate entities during normalization

    Returns:
        List of Fragment instances, or single merged Fragment if merge=True

    Example:
        # Normalize fragments from different sources
        rigor_frags = [{"classes": [...], "relationships": [...]}]
        folder_frags = [{"nodes": [...], "edges": [...]}]

        normalized = normalize_fragments(rigor_frags + folder_frags)

        # Convert to standardized dictionaries
        standardized_dicts = [f.to_dict() for f in normalized]
    """
    normalizer = FragmentNormalizer(deduplicate=deduplicate)
    normalized = normalizer.normalize_many(fragments)

    if merge:
        return normalizer.merge_fragments(normalized)
    return normalized


def convert_legacy_fragment(fragment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a legacy fragment dictionary to standardized format.

    This is a convenience function for one-off conversions.

    Args:
        fragment_data: Raw fragment dictionary

    Returns:
        Standardized fragment dictionary
    """
    fragment = Fragment.from_legacy_dict(fragment_data)
    return fragment.to_dict()


def extract_nodes_from_legacy(fragment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract unified nodes list from legacy fragment (handles classes+nodes).

    Args:
        fragment_data: Raw fragment dictionary

    Returns:
        List of node dictionaries in unified format
    """
    fragment = Fragment.from_legacy_dict(fragment_data)
    return [n.to_dict() for n in fragment.nodes]


def extract_relationships_from_legacy(fragment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract unified relationships list from legacy fragment (handles edges/relationships).

    Args:
        fragment_data: Raw fragment dictionary

    Returns:
        List of relationship dictionaries in unified format
    """
    fragment = Fragment.from_legacy_dict(fragment_data)
    return [r.to_dict() for r in fragment.relationships]
