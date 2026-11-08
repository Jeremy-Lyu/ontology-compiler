"""
Neo4jSyncService — 图结构与索引同步更新 (R8)

职责：
- 监听本体数据变更事件，自动同步到 Neo4j
- 支持全量同步和增量同步
- 维护同步状态，支持断点续传
- 与本地索引协同，确保数据一致性
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# B-4 (2026-08-17): neo4j SDK import isolated to infrastructure provider.
from src.infrastructure.rag.neo4j_provider import get_graph_database_class, is_neo4j_available

GraphDatabase = get_graph_database_class()

logger = logging.getLogger(__name__)


@dataclass
class SyncConfig:
    """同步配置。"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    batch_size: int = 100
    enable_incremental: bool = True
    sync_on_extract: bool = True
    state_file: str = "logs/neo4j_sync_state.json"


@dataclass
class SyncState:
    """同步状态。"""
    last_sync: Optional[str] = None
    last_hash: Optional[str] = None
    synced_entities: int = 0
    synced_relationships: int = 0
    sync_history: List[Dict[str, Any]] = field(default_factory=list)


class Neo4jSyncService:
    """Neo4j 同步服务。"""

    def __init__(self, config: Optional[SyncConfig] = None):
        self.config = config or self._load_config()
        self.state = self._load_state()
        self._driver = None

        if GraphDatabase is None:
            logger.warning("neo4j driver not installed. Sync service disabled.")

    def _load_config(self) -> SyncConfig:
        """从 YAML 配置或环境变量加载配置。"""
        import os
        from pathlib import Path

        # 尝试从 YAML 配置加载
        yaml_paths = [
            Path("config/neo4j_config.yaml"),
            Path("../config/neo4j_config.yaml"),
            Path(__file__).parent.parent.parent.parent / "config" / "neo4j_config.yaml",
        ]

        for yaml_path in yaml_paths:
            if yaml_path.exists():
                try:
                    import yaml
                    with open(yaml_path, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)
                    conn = cfg.get("connection", {})
                    sync = cfg.get("sync", {})
                    return SyncConfig(
                        uri=conn.get("uri", "bolt://localhost:7687"),
                        username=conn.get("username", "neo4j"),
                        password=conn.get("password", ""),
                        database=conn.get("database", "neo4j"),
                        batch_size=sync.get("batch_size", 100),
                        enable_incremental=sync.get("enable_incremental", True),
                        sync_on_extract=sync.get("sync_on_extract", True),
                        state_file=sync.get("state_file", "logs/neo4j_sync_state.json"),
                    )
                except Exception as e:
                    logger.debug(f"Failed to load YAML config from {yaml_path}: {e}")

        # 回退到环境变量
        return SyncConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )

    def _load_state(self) -> SyncState:
        state_path = Path(self.config.state_file)
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return SyncState(**data)
            except Exception as e:
                logger.warning(f"Failed to load sync state: {e}")
        return SyncState()

    def _save_state(self):
        state_path = Path(self.config.state_file)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(self.state.__dict__, f, indent=2, default=str)

    def _get_driver(self):
        if self._driver is None and GraphDatabase is not None:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password)
            )
        return self._driver

    def is_available(self) -> bool:
        if GraphDatabase is None:
            return False
        try:
            driver = self._get_driver()
            if driver is None:
                return False
            with driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception as e:
            logger.debug(f"Neo4j not available: {e}")
            return False

    async def full_sync(self, ontology_data: Dict[str, Any]) -> Dict[str, int]:
        if not self.is_available():
            logger.warning("Neo4j not available, skipping full sync")
            return {"entities": 0, "relationships": 0}

        logger.info("Starting full sync to Neo4j...")
        self._clear_all()

        nodes = ontology_data.get("nodes", [])
        entities_synced = self._sync_entities(nodes)

        relationships = ontology_data.get("relationships", [])
        rels_synced = self._sync_relationships(relationships)

        self.state.last_sync = datetime.now().isoformat()
        self.state.synced_entities = entities_synced
        self.state.synced_relationships = rels_synced
        self.state.sync_history.append({
            "type": "full",
            "timestamp": self.state.last_sync,
            "entities": entities_synced,
            "relationships": rels_synced,
        })
        self._save_state()

        logger.info(f"Full sync complete: {entities_synced} entities, {rels_synced} relationships")
        return {"entities": entities_synced, "relationships": rels_synced}

    async def incremental_sync(self, changes: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
        if not self.is_available():
            logger.warning("Neo4j not available, skipping incremental sync")
            return {"added": 0, "modified": 0, "deleted": 0}

        logger.info("Starting incremental sync to Neo4j...")

        added = self._sync_entities(changes.get("added", []))
        modified = self._sync_entities(changes.get("modified", []))
        deleted = self._delete_entities([e.get("id") or e.get("meta_id") for e in changes.get("deleted", [])])
        rels_added = self._sync_relationships(changes.get("added_relationships", []))

        self.state.last_sync = datetime.now().isoformat()
        self.state.sync_history.append({
            "type": "incremental",
            "timestamp": self.state.last_sync,
            "added": added,
            "modified": modified,
            "deleted": deleted,
            "relationships": rels_added,
        })
        self._save_state()

        logger.info(f"Incremental sync complete: +{added}, ~{modified}, -{deleted} entities, +{rels_added} relationships")
        return {"added": added, "modified": modified, "deleted": deleted, "relationships": rels_added}

    def _sync_entities(self, entities: List[Dict[str, Any]]) -> int:
        if not entities:
            return 0
        driver = self._get_driver()
        if driver is None:
            return 0
        synced = 0
        batch_size = self.config.batch_size
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            with driver.session() as session:
                for entity in batch:
                    try:
                        self._merge_entity(session, entity)
                        synced += 1
                    except Exception as e:
                        entity_id = entity.get("meta_id") or entity.get("id", "unknown")
                        logger.warning(f"Failed to sync entity {entity_id}: {e}")
        return synced

    def _merge_entity(self, session, entity: Dict[str, Any]):
        entity_id = entity.get("meta_id") or entity.get("id")
        if not entity_id:
            return
        meta_class = entity.get("meta_class", "OntologyNode")
        layer = entity.get("layer", "L1")
        labels = [meta_class, f"Layer{layer}"]
        label_str = ":".join(labels)
        props = {
            "id": entity_id,
            "name": entity.get("meta_name") or entity.get("name") or entity.get("scenario_name", ""),
            "description": entity.get("description") or entity.get("meta_description", ""),
            "domain": entity.get("sandbox_domain") or entity.get("domain", ""),
            "layer": layer,
        }
        for key, value in entity.items():
            if key not in props and isinstance(value, (str, int, float, bool)):
                props[key] = value
        cypher = f"""
            MERGE (n:{label_str} {{id: $id}})
            SET n += $props
        """
        session.run(cypher, id=entity_id, props=props)

    def _sync_relationships(self, relationships: List[Dict[str, Any]]) -> int:
        if not relationships:
            return 0
        driver = self._get_driver()
        if driver is None:
            return 0
        synced = 0
        with driver.session() as session:
            for rel in relationships:
                try:
                    source = rel.get("source") or rel.get("from") or rel.get("meta_id")
                    target = rel.get("target") or rel.get("to")
                    rel_type = rel.get("type") or rel.get("predicate") or rel.get("relation", "RELATES_TO")
                    if not source or not target:
                        continue
                    cypher = f"""
                        MATCH (a {{id: $source}})
                        MATCH (b {{id: $target}})
                        MERGE (a)-[r:{rel_type}]->(b)
                        SET r += $props
                    """
                    props = {k: v for k, v in rel.items() if k not in ("source", "target", "from", "to", "type", "predicate")}
                    session.run(cypher, source=source, target=target, props=props)
                    synced += 1
                except Exception as e:
                    logger.warning(f"Failed to sync relationship: {e}")
        return synced

    def _delete_entities(self, entity_ids: List[str]) -> int:
        if not entity_ids:
            return 0
        driver = self._get_driver()
        if driver is None:
            return 0
        deleted = 0
        with driver.session() as session:
            for eid in entity_ids:
                if not eid:
                    continue
                try:
                    session.run("""
                        MATCH (n {id: $id})
                        DETACH DELETE n
                    """, id=eid)
                    deleted += 1
                except Exception as e:
                    logger.warning(f"Failed to delete entity {eid}: {e}")
        return deleted

    def _clear_all(self):
        driver = self._get_driver()
        if driver is None:
            return
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared all Neo4j data")

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None


def register_neo4j_sync_hook(pipeline, sync_service=None):
    """注册 Neo4j 同步钩子到 pipeline。"""
    service = sync_service or Neo4jSyncService()

    async def on_extract_complete(result):
        if not service.config.sync_on_extract:
            return
        try:
            changes = {
                "added": result.get("l1_entities", []) + result.get("l2_entities", []),
                "added_relationships": result.get("relationships", []),
            }
            stats = await service.incremental_sync(changes)
            logger.info(f"Auto-synced to Neo4j: {stats}")
        except Exception as e:
            logger.warning(f"Auto-sync to Neo4j failed: {e}")

    if hasattr(pipeline, "add_hook"):
        pipeline.add_hook("extract_complete", on_extract_complete)
    elif hasattr(pipeline, "on_complete"):
        pipeline.on_complete(on_extract_complete)

    logger.info("Neo4j sync hook registered")
    return service
