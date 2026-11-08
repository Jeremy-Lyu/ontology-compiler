"""Master Ontology Store - 双模式存储抽象层

支持两种存储模式：
1. 单体文件模式：master_ontology_unified.json（单一大文件）
2. 分层文件模式：master_ontology/（多个小文件按层拆分）

双模式同步策略：
- 主存储：单体文件（快速读写）
- 镜像存储：分层文件（按需加载、便于版本控制）
- 写操作：同时更新单体文件和分层文件
- 读操作：优先从单体文件读取（更快），支持按需加载单层

分层文件结构：
- master_ontology/nodes.json
- master_ontology/instances.json
- master_ontology/relationships.json
- master_ontology/layer0_metadata.json
- master_ontology/layer1_concepts.json
- master_ontology/layer2_scenarios.json
- master_ontology/layer25_mapping_rules.json
- master_ontology/metadata.json

文件锁机制：
- 使用基于文件的锁防止并发写入冲突
- 支持超时自动清理僵尸锁
- 原子写入确保数据完整性

增量 governed 包重建钩子（Phase B, 2026-08-29）:
- ``MasterOntologyStore.write`` 在单体+分层文件都刷完后调用
  ``MasterOntologyStore._post_write_rebuild_governed_packet()``
- 轻量指纹 = 按 build script 约定的 nodes/relationships.json sha256
  （和 governed 脚本内增量跳过使用的相同字段）；若与
  ``latest_governed_packet.json`` 中记录一致则零成本跳过；否则才拉起
  子进程跑 ``scripts/build_governed_packet.py``
- 全局 kill-switch：``RAG_GOVERNED_REBUILD=0`` 彻底禁用钩子；CLI 也可
  传 ``--skip-governed-rebuild`` 等价效果（设置环境变量即可）。

P2-4 增量触发策略化（2026-09-09）:
- **同 hash 防抖**：``_last_rebuild_key`` 缓存最近一次成功触发（或确认
  跳过）时的 (nodes_hash, rels_hash)。同一进程内对相同 master 内容的
  连续多次 ``write()`` 只会拉起一次子进程——第 2..N 次直接命中防抖键
  NOOP。跨进程场景仍由 manifest ``master_hashes`` 对比兜底（build 脚本
  自身的增量跳过）。
- **失败计数与告警升级**：rebuild 失败（rc≠0 / 超时 / 异常）计入
  ``_rebuild_failure_count``（进程级）并写入
  ``outputs/runs/ontology/governed_rebuild_status.json``（跨进程可观测：
  consecutive_failures / last_error / last_attempt_at）。连续失败达到
  ``RAG_REBUILD_FAIL_ALERT_THRESHOLD``（默认 3）时日志升级为 ERROR 并在
  状态文件标记 ``alert=True``；成功一次即清零。
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

try:
    from src.utils.file_lock import write_with_lock, atomic_write, cleanup_stale_locks, FileLock, FileLockError
except ImportError:
    write_with_lock = None
    atomic_write = None
    cleanup_stale_locks = None
    FileLock = None
    FileLockError = None


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class MasterOntologyStore:
    _LAYER_FILES = [
        'nodes.json',
        'instances.json',
        'relationships.json',
        'layer0_metadata.json',
        'layer1_concepts.json',
        'layer2_scenarios.json',
        'layer25_mapping_rules.json',
        'metadata.json',
    ]
    
    _FIELD_MAP = {
        'nodes.json': 'nodes',
        'instances.json': 'instances',
        'relationships.json': 'relationships',
        'layer0_metadata.json': 'layer0_metadata',
        'layer1_concepts.json': 'layer1_concepts',
        'layer2_scenarios.json': 'layer2_scenarios',
        'layer25_mapping_rules.json': 'layer25_mapping_rules',
        'metadata.json': 'metadata',
    }
    
    def __init__(self, path: str, sync_mode: bool = True):
        self._path = Path(path)
        self._sync_mode = sync_mode
        # P2-4 (2026-09-09): same-hash debounce + failure tracking.
        self._last_rebuild_key: Optional[tuple] = None
        self._rebuild_failure_count: int = 0
        
        if self._path.is_dir():
            self._store_dir = self._path
            self._unified_file = self._store_dir.parent / 'master_ontology_unified.json'
        else:
            self._unified_file = self._path
            self._store_dir = self._path.parent / 'master_ontology'
    
    @property
    def unified_file(self) -> Path:
        return self._unified_file
    
    @property
    def store_dir(self) -> Path:
        return self._store_dir
    
    @property
    def sync_mode(self) -> bool:
        return self._sync_mode
    
    def exists(self) -> bool:
        return self._unified_file.exists() or (self._store_dir.exists() and any(self._store_dir.iterdir()))
    
    def read(self, layers: Optional[List[str]] = None) -> Dict[str, Any]:
        if layers is None:
            return self._read_full()
        
        return self._read_layers(layers)
    
    def _read_full(self) -> Dict[str, Any]:
        if self._unified_file.exists():
            return self._read_unified_file()
        
        if self._store_dir.exists() and any(self._store_dir.iterdir()):
            return self._read_directory()
        
        return {}
    
    def _read_layers(self, layers: List[str]) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        
        for layer_file in self._LAYER_FILES:
            field = self._FIELD_MAP.get(layer_file)
            if field is None or (layers and field not in layers):
                continue
            
            file_path = self._store_dir / layer_file
            
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data[field] = json.load(f)
                except (json.JSONDecodeError, IOError):
                    data[field] = [] if 'layer' in field or field in ['nodes', 'instances', 'relationships'] else {}
            else:
                if self._unified_file.exists():
                    full_data = self._read_unified_file()
                    data[field] = full_data.get(field, [] if 'layer' in field or field in ['nodes', 'instances', 'relationships'] else {})
                else:
                    data[field] = [] if 'layer' in field or field in ['nodes', 'instances', 'relationships'] else {}
        
        return data
    
    def write(self, data: Dict[str, Any], sync: Optional[bool] = None) -> None:
        sync_mode = sync if sync is not None else self._sync_mode
        
        self._write_unified_file(data)
        
        if sync_mode:
            self._write_directory(data)

        # Phase B (2026-08-29) - governed packet auto-rebuild after a successful
        # master write.  This is the only integration hook so that ANY caller
        # (merge stage _write_master, export stage, manual edits via store.write)
        # that ends up changing master nodes/rels automatically refreshes the
        # only authoritative retrieval asset.
        self._post_write_rebuild_governed_packet()

    @staticmethod
    def _rebuild_status_path() -> Path:
        return Path("outputs/runs/ontology/governed_rebuild_status.json")

    def _record_rebuild_outcome(self, ok: bool, error: str = "") -> None:
        """Update in-process failure count + cross-process status file.

        The status file is best-effort observability (never raises): it lets
        CI / run_batch detect a silently-degrading retrieval asset after the
        fact, since the hook itself must not break the master write path.
        """
        if ok:
            self._rebuild_failure_count = 0
            consecutive, last_error, alert = 0, "", False
        else:
            self._rebuild_failure_count += 1
            consecutive = self._rebuild_failure_count
            last_error = error
            try:
                threshold = int(os.environ.get("RAG_REBUILD_FAIL_ALERT_THRESHOLD", "3"))
            except ValueError:
                threshold = 3
            alert = threshold > 0 and consecutive >= threshold
        try:
            status_dir = self._rebuild_status_path().parent
            status_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "last_outcome": "ok" if ok else "failed",
                "consecutive_failures": consecutive,
                "last_error": last_error[-500:],
                "alert": alert,
            }
            self._rebuild_status_path().write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass  # observability must never break persistence

    def _post_write_rebuild_governed_packet(self) -> None:
        """Rebuild the governed RAG packet if master has actually changed.

        Decision order (fail-safe, no-ops by default when anything is
        unavailable):
            1. If ``RAG_GOVERNED_REBUILD=0`` -> NOOP (kill-switch).
            2. No ``master_ontology/`` directory (e.g. unified-only tests) -> NOOP.
            3. Read ``latest_governed_packet.json`` manifest, compare
               ``master_hashes`` for nodes.json / relationships.json.  Identical
               -> NOOP (zero cost skip, same as build script's incremental path).
            4. Otherwise invoke ``scripts/build_governed_packet.py`` as a
               *synchronous* subprocess so a downstream retrieval-eval call on
               the same run sees the refreshed packet.  Failure (non-zero rc)
               is logged but swallowed so a retrieval infra failure never
               breaks an otherwise-successful master write.
        """
        if os.environ.get("RAG_GOVERNED_REBUILD", "1") in {"0", "false", "False", "no"}:
            return
        master_dir = self._store_dir
        if not master_dir.is_dir():
            return
        nodes_hash = _sha256_file(master_dir / "nodes.json")
        rels_hash = _sha256_file(master_dir / "relationships.json")
        if not nodes_hash and not rels_hash:
            return  # master not yet materialized - nothing to index

        # P2-4 debounce: identical master content within this process was
        # already handed to the build script once; repeated writes (e.g.
        # merge -> export both writing the same master) must not fan out
        # extra subprocess rebuilds. Cross-process freshness is still
        # guaranteed by the manifest master_hashes comparison below.
        rebuild_key = (nodes_hash, rels_hash)
        if rebuild_key == self._last_rebuild_key:
            return  # same-hash debounce - already handled this content

        # Skip by comparing against the *already published* governed manifest
        # master_hashes (exactly the same bytes the build script wrote last
        # time).  Build script itself also performs the exact same check when
        # --force is not used, so a duplicate subprocess call is both cheap
        # and self-healing (e.g. earlier publish succeeded but pointer file
        # wasn't written).
        try:
            from src.application.rag.governed_packets import (  # lazy import - avoid storage<->rag circular dependency at module load
                ASSET_PACKETS_ROOT as _asset_root,
                LATEST_POINTER,
            )
        except Exception:
            # Fallback: master_dir is outputs/runs/ontology/master_ontology,
            # so "outputs/runs/ontology" level is one parent up.
            _asset_root = master_dir.parent / "asset_packets"
            LATEST_POINTER = _asset_root.parent / "latest_governed_packet.json"
        if LATEST_POINTER.exists():
            try:
                pointer = json.loads(LATEST_POINTER.read_text(encoding="utf-8"))
                packet_id = pointer.get("packet_id")
                if packet_id:
                    governed_root = _asset_root / "governed"
                    manifest_path = governed_root / str(packet_id) / "manifest.json"
                    if manifest_path.exists():
                        mf = json.loads(manifest_path.read_text(encoding="utf-8"))
                        prev = mf.get("master_hashes") or {}
                        if (
                            prev.get("nodes.json") == nodes_hash
                            and prev.get("relationships.json") == rels_hash
                        ):
                            # Manifest already matches: debounce key + reset
                            # failure counter so a prior transient failure does
                            # not keep alert=True when content is now fresh.
                            self._last_rebuild_key = rebuild_key
                            self._record_rebuild_outcome(True)
                            return  # NOOP fast path - governed content is fresh
            except Exception:
                pass  # Best-effort; fall through to subprocess rebuild

        project_root = Path(__file__).resolve().parents[3]  # src/pipeline/storage -> root
        build_script = project_root / "scripts" / "build_governed_packet.py"
        if not build_script.exists():
            return
        try:
            proc = subprocess.run(
                [sys.executable, str(build_script), "--master-dir", str(master_dir)],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                check=False,
                timeout=10 * 60,
                env=dict(os.environ),
            )
            if proc.returncode != 0:
                self._record_rebuild_outcome(
                    False, f"rc={proc.returncode} stderr={(proc.stderr or '')[-300:]}"
                )
                print(
                    f"[MasterStore] WARNING: governed packet rebuild failed (rc={proc.returncode}).\n"
                    f"                 STDOUT tail: {(proc.stdout or '')[-600:]}\n"
                    f"                 STDERR tail: {(proc.stderr or '')[-600:]}"
                )
                if self._rebuild_failure_count >= int(os.environ.get("RAG_REBUILD_FAIL_ALERT_THRESHOLD", "3") or 3):
                    print(
                        f"[MasterStore] ERROR: governed packet rebuild has failed "
                        f"{self._rebuild_failure_count} consecutive times; retrieval asset may be stale. "
                        f"See outputs/runs/ontology/governed_rebuild_status.json"
                    )
            else:
                self._record_rebuild_outcome(True)
                self._last_rebuild_key = rebuild_key
        except subprocess.TimeoutExpired:
            self._record_rebuild_outcome(False, "timeout after 10 min")
            print("[MasterStore] WARNING: governed packet rebuild timed out after 10 min")
        except Exception as exc:  # never break the master write path
            self._record_rebuild_outcome(False, f"{type(exc).__name__}: {exc}")
            print(f"[MasterStore] WARNING: governed packet rebuild skipped ({type(exc).__name__}: {exc})")
    
    def _write_unified_file(self, data: Dict[str, Any]) -> None:
        self._unified_file.parent.mkdir(parents=True, exist_ok=True)
        
        if atomic_write:
            try:
                atomic_write(self._unified_file, data, timeout=30.0, check_interval=0.5)
                return
            except FileLockError as e:
                pass
        
        with open(self._unified_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _write_directory(self, data: Dict[str, Any]) -> None:
        self._store_dir.mkdir(parents=True, exist_ok=True)
        
        for layer_file in self._LAYER_FILES:
            field = self._FIELD_MAP.get(layer_file)
            file_path = self._store_dir / layer_file
            content = data.get(field, [] if 'layer' in field or field in ['nodes', 'instances', 'relationships'] else {})
            
            if atomic_write:
                try:
                    atomic_write(file_path, content, timeout=30.0, check_interval=0.5)
                    continue
                except FileLockError as e:
                    pass
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
    
    def _read_unified_file(self) -> Dict[str, Any]:
        if not self._unified_file.exists():
            return {}
        
        with open(self._unified_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _read_directory(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        
        for layer_file in self._LAYER_FILES:
            field = self._FIELD_MAP.get(layer_file)
            file_path = self._store_dir / layer_file
            
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data[field] = json.load(f)
                except (json.JSONDecodeError, IOError):
                    data[field] = [] if 'layer' in field or field in ['nodes', 'instances', 'relationships'] else {}
            else:
                data[field] = [] if 'layer' in field or field in ['nodes', 'instances', 'relationships'] else {}
        
        return data
    
    def sync_to_directory(self) -> None:
        data = self._read_unified_file()
        self._write_directory(data)
    
    def sync_to_unified(self) -> None:
        data = self._read_directory()
        self._write_unified_file(data)
    
    def compact(self, deduplicate_keys=None) -> Dict[str, int]:
        if deduplicate_keys is None:
            deduplicate_keys = ['meta_id', 'name', 'description', 'source_file']
        
        data = self.read()
        removed_counts = {}
        
        for layer_file in self._LAYER_FILES:
            field = self._FIELD_MAP.get(layer_file)
            if field not in data or not isinstance(data[field], list):
                continue
            
            if field == 'relationships':
                continue
            
            items = data[field]
            seen = set()
            deduped = []
            
            for item in items:
                if not isinstance(item, dict):
                    continue
                
                key_parts = []
                for key in deduplicate_keys:
                    val = item.get(key)
                    if val:
                        key_parts.append(str(val))
                
                item_key = '||'.join(key_parts) if key_parts else id(item)
                
                if item_key not in seen:
                    seen.add(item_key)
                    deduped.append(item)
            
            removed = len(items) - len(deduped)
            if removed > 0:
                data[field] = deduped
                removed_counts[field] = removed
        
        if removed_counts:
            self.write(data)
        
        return removed_counts
    
    def get_layer_path(self, layer: str) -> Path:
        layer_map = {
            'L0': 'layer0_metadata.json',
            'L1': 'layer1_concepts.json',
            'L2': 'layer2_scenarios.json',
            'L2.5': 'layer25_mapping_rules.json',
        }
        layer_file = layer_map.get(layer)
        if layer_file:
            return self._store_dir / layer_file
        return self._store_dir / 'nodes.json'
    
    def export_to(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        layers_dir = output_dir / 'layers'
        relationships_dir = output_dir / 'relationships'
        layers_dir.mkdir(parents=True, exist_ok=True)
        relationships_dir.mkdir(parents=True, exist_ok=True)
        
        layer_export_map = {
            'layer0_metadata.json': ('layers', 'l0_metadata.json'),
            'layer1_concepts.json': ('layers', 'l1_concepts.json'),
            'layer2_scenarios.json': ('layers', 'l2_entities.json'),
            'layer25_mapping_rules.json': ('layers', 'l25_rules.json'),
            'instances.json': ('layers', 'l3_instances.json'),
            'relationships.json': ('relationships', 'relationships.json'),
        }
        
        data = self.read()
        
        for src_file, (dst_folder, dst_file) in layer_export_map.items():
            field = self._FIELD_MAP.get(src_file)
            if field is None:
                continue
            
            content = data.get(field, [])
            if isinstance(content, list) and len(content) > 0:
                if dst_folder == 'layers':
                    dst_path = layers_dir / dst_file
                else:
                    dst_path = relationships_dir / dst_file
                
                with open(dst_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, ensure_ascii=False, indent=2)


def create_master_store(path: str, sync_mode: bool = True) -> MasterOntologyStore:
    return MasterOntologyStore(path, sync_mode)