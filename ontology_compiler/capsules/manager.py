"""
Capsule Manager - Manages capsule lifecycle

Handles capsule CRUD operations, storage, and retrieval.
"""

from __future__ import annotations

import json
import logging
import hashlib
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from .capsule import Capsule, CapsuleMetadata
from .execution_plan import ExecutionPlan
from .glossary import Glossary, get_default_glossary
from .evolution import Evolution, Experiment, ExperimentStatus


logger = logging.getLogger(__name__)


from core.contracts.ports import CapsuleQueryPort


class CapsuleQueryAdapter(CapsuleQueryPort):
    """Adapter wrapping the legacy CapsuleManager."""

    def __init__(self, manager: CapsuleManager):
        self._manager = manager

    async def query_capsules(self, query: str, limit: int = 5) -> list[Any]:
        return await self._manager.query_capsules(query, limit=limit)


class CapsuleManager:
    """
    Manages capsule lifecycle.
    
    Provides:
    - Capsule registration and storage
    - Capsule retrieval and querying
    - Experiment tracking
    - Evolution management
    """
    
    def __init__(self, storage_dir: str = "outputs/capsules"):
        """Initialize capsule manager"""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path = self.storage_dir / "PUBLISHED_CAPSULE_AUDIT_LOG.json"
        
        # In-memory storage
        self.capsules: Dict[str, Capsule] = {}
        self.glossary = get_default_glossary()
        self.evolution = Evolution()
        
        # Inverted index for fast text search: token -> set of capsule IDs
        self._search_index: Dict[str, set[str]] = {}
        
        logger.info(f"CapsuleManager initialized with storage: {self.storage_dir}")
    
    def _extract_tokens(self, capsule: Capsule) -> set[str]:
        """Extract searchable tokens from a capsule."""
        tokens = set()
        
        # Name tokens
        name_lower = capsule.metadata.name.lower()
        tokens.update(name_lower.split())
        
        # Summary tokens
        summary_lower = capsule.summary.lower()
        tokens.update(summary_lower.split())
        
        # Tags tokens
        for tag in capsule.metadata.tags:
            tokens.update(tag.lower().split())
        
        # Contexts tokens
        for ctx in capsule.metadata.contexts:
            tokens.update(ctx.lower().split())
        
        # Category tokens
        category_lower = capsule.metadata.category.lower()
        tokens.update(category_lower.split())
        
        return tokens
    
    def _update_index(self, capsule: Capsule, remove: bool = False) -> None:
        """Update the inverted index with a capsule."""
        capsule_id = capsule.metadata.id
        tokens = self._extract_tokens(capsule)
        
        for token in tokens:
            if token not in self._search_index:
                self._search_index[token] = set()
            
            if remove:
                self._search_index[token].discard(capsule_id)
                if not self._search_index[token]:
                    del self._search_index[token]
            else:
                self._search_index[token].add(capsule_id)
    
    def _rebuild_index(self) -> None:
        """Rebuild the entire search index from all capsules."""
        self._search_index.clear()
        for capsule in self.capsules.values():
            self._update_index(capsule)
        logger.debug(f"Search index rebuilt with {len(self._search_index)} tokens")

    
    async def register_capsule(self, capsule: Capsule) -> str:
        """
        Register a new capsule.
        
        Args:
            capsule: Capsule to register
            
        Returns:
            Capsule ID
        """
        capsule_id = capsule.metadata.id
        
        # Check if capsule already exists
        if capsule_id in self.capsules:
            logger.warning(f"Capsule {capsule_id} already exists, updating")
        
        # Store in memory
        self.capsules[capsule_id] = capsule
        
        # Update search index
        self._update_index(capsule)
        
        # Create evolution chain
        self.evolution.create_chain(capsule_id)
        
        # Save to disk
        await self._save_capsule(capsule)
        
        logger.info(f"Capsule registered: {capsule_id}")
        return capsule_id
    
    async def get_capsule(self, capsule_id: str) -> Optional[Capsule]:
        """Get capsule by ID"""
        # Try memory first
        if capsule_id in self.capsules:
            return self.capsules[capsule_id]
        
        # Try disk
        capsule = await self._load_capsule(capsule_id)
        if capsule:
            self.capsules[capsule_id] = capsule
            return capsule
        
        logger.warning(f"Capsule not found: {capsule_id}")
        return None
    
    async def update_capsule(self, capsule: Capsule) -> None:
        """Update existing capsule"""
        capsule_id = capsule.metadata.id
        capsule.metadata.updated_at = datetime.now().isoformat()
        
        self.capsules[capsule_id] = capsule
        
        # Update search index (remove old tokens, add new tokens)
        self._update_index(capsule, remove=True)
        self._update_index(capsule, remove=False)
        
        await self._save_capsule(capsule)
        
        logger.info(f"Capsule updated: {capsule_id}")
    
    async def delete_capsule(self, capsule_id: str) -> None:
        """Delete capsule"""
        if capsule_id in self.capsules:
            capsule = self.capsules[capsule_id]
            # Remove from search index
            self._update_index(capsule, remove=True)
            del self.capsules[capsule_id]
        
        # Delete from disk
        capsule_file = self.storage_dir / f"{capsule_id}.json"
        if capsule_file.exists():
            capsule_file.unlink()
        
        logger.info(f"Capsule deleted: {capsule_id}")
    
    async def list_capsules(self, filters: Optional[Dict[str, Any]] = None) -> List[Capsule]:
        """
        List capsules with optional filters.
        
        Filters:
        - category: Filter by category
        - tags: Filter by tags (any match)
        - contexts: Filter by contexts (any match)
        """
        capsules = list(self.capsules.values())
        
        if not filters:
            return capsules
        
        # Apply filters
        if 'category' in filters:
            capsules = [c for c in capsules if c.metadata.category == filters['category']]
        
        if 'tags' in filters:
            filter_tags = set(filters['tags'])
            capsules = [c for c in capsules if any(t in filter_tags for t in c.metadata.tags)]
        
        if 'contexts' in filters:
            filter_contexts = set(filters['contexts'])
            capsules = [c for c in capsules if any(ctx in filter_contexts for ctx in c.metadata.contexts)]
        
        return capsules
    
    async def query_capsules(self, query: str, limit: int = 10) -> List[Capsule]:
        """
        Query capsules by text search using inverted index.

        Splits query into tokens and scores each capsule across:
        - Capsule name
        - Summary
        - Tags
        - Contexts
        - Category
        """
        # Split into individual tokens so "postgres performance" matches both words
        tokens = [t for t in query.lower().split() if t]
        if not tokens:
            return []
        
        # Use inverted index to find candidate capsules
        candidate_ids = set()
        for token in tokens:
            if token in self._search_index:
                candidate_ids.update(self._search_index[token])
        
        if not candidate_ids:
            return []
        
        # Score only the candidate capsules instead of all capsules
        results = []
        for capsule_id in candidate_ids:
            capsule = self.capsules.get(capsule_id)
            if not capsule:
                continue
            
            score = 0
            name_lower = capsule.metadata.name.lower()
            summary_lower = capsule.summary.lower()
            tags_lower = [t.lower() for t in capsule.metadata.tags]
            contexts_lower = [c.lower() for c in capsule.metadata.contexts]
            category_lower = capsule.metadata.category.lower()

            for token in tokens:
                if token in name_lower:
                    score += 10
                if token in summary_lower:
                    score += 5
                if any(token in tag for tag in tags_lower):
                    score += 3
                if any(token in ctx for ctx in contexts_lower):
                    score += 2
                if token in category_lower:
                    score += 2

            if score > 0:
                results.append((score, capsule))

        # Sort by score and return top results
        results.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in results[:limit]]
    
    async def record_experiment(
        self,
        capsule_id: str,
        project_name: str,
        result_summary: str,
        metrics: Dict[str, Any],
        success: bool = True,
        improvement_rate: float = 0.0,
        notes: str = ""
    ) -> str:
        """
        Record an experiment for a capsule.
        
        Args:
            capsule_id: Capsule ID
            project_name: Project name
            result_summary: Result summary
            metrics: Metrics dictionary
            success: Whether experiment was successful
            improvement_rate: Improvement rate (0.0-1.0)
            notes: Additional notes
            
        Returns:
            Experiment ID
        """
        # Create experiment
        experiment_id = f"exp-{capsule_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        experiment = Experiment(
            id=experiment_id,
            capsule_id=capsule_id,
            project_name=project_name,
            result_summary=result_summary,
            metrics=metrics,
            success=success,
            improvement_rate=improvement_rate,
            notes=notes,
        )
        experiment.mark_completed(success, improvement_rate)
        
        # Add to evolution chain
        self.evolution.add_experiment(capsule_id, experiment)
        
        # Update capsule
        capsule = await self.get_capsule(capsule_id)
        if capsule:
            capsule.add_experiment(experiment.to_dict())
            await self.update_capsule(capsule)
        
        logger.info(f"Experiment recorded: {experiment_id}")
        return experiment_id
    
    def get_capsule_stats(self, capsule_id: str) -> Dict[str, Any]:
        """Get capsule statistics"""
        return self.evolution.get_capsule_stats(capsule_id)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all capsules"""
        return self.evolution.get_all_stats()

    async def publish_as_capsule(
        self,
        *,
        capsule: Capsule,
        requested_by: str,
        approval_signature: str,
        approval_digest: str,
        publish_reason: str,
    ) -> str:
        """Publish capsule with audit envelope and approval metadata."""

        await self.register_capsule(capsule)

        payload_digest = self._build_capsule_digest(capsule)
        audit_entry = {
            "kind": "capsule_publish",
            "ts": datetime.now().isoformat(),
            "capsule_id": capsule.metadata.id,
            "capsule_version": capsule.metadata.version,
            "requested_by": requested_by,
            "approval_signature": approval_signature,
            "approval_digest": approval_digest,
            "publish_reason": publish_reason,
            "payload_digest": payload_digest,
        }
        self._append_publish_audit_log(audit_entry)
        return capsule.metadata.id

    def _build_capsule_digest(self, capsule: Capsule) -> str:
        serialized = json.dumps(capsule.to_dict(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _append_publish_audit_log(self, entry: Dict[str, Any]) -> None:
        history: list[dict[str, Any]] = []
        if self.audit_log_path.exists():
            try:
                payload = json.loads(self.audit_log_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
                    history = [item for item in payload["entries"] if isinstance(item, dict)]
            except Exception:
                history = []

        history.append(entry)
        out = {
            "generated_at": datetime.now().isoformat(),
            "entries": history[-500:],
        }
        with open(self.audit_log_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    async def _save_capsule(self, capsule: Capsule) -> None:
        """Save capsule to disk"""
        capsule_file = self.storage_dir / f"{capsule.metadata.id}.json"
        
        with open(capsule_file, 'w', encoding='utf-8') as f:
            f.write(capsule.to_json())
        
        logger.debug(f"Capsule saved: {capsule_file}")
    
    async def _load_capsule(self, capsule_id: str) -> Optional[Capsule]:
        """Load capsule from disk"""
        capsule_file = self.storage_dir / f"{capsule_id}.json"
        
        if not capsule_file.exists():
            return None
        
        try:
            with open(capsule_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            capsule = Capsule.from_dict(data)
            logger.debug(f"Capsule loaded: {capsule_id}")
            return capsule
        except Exception as e:
            logger.error(f"Failed to load capsule {capsule_id}: {e}")
            return None

