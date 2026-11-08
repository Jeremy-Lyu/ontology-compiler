"""向量索引管理器 - 支持增量更新、多版本管理和分层检索

[Phase 4 - RAG 2026-06-17]: Enhanced with multi-version management, rollback, and layered retrieval.
[Phase 6-RAG-1]: Enhanced with configurable embedder support (LocalEmbedder / OpenAIEmbedder).

核心功能：
1. 增量向量索引构建（仅处理新增/修改的 chunks）
2. 多版本索引管理（支持 A/B 测试和回滚）
3. 索引元数据追踪（记录构建时间、数据源、配置哈希）
4. 检索结果分层返回（支持 L0-L3 层级过滤）
5. 版本对比与差异分析
6. 向后兼容现有索引格式
7. 可配置嵌入器（本地 TF-IDF / OpenAI Embedding API）

Usage:
    from src.application.rag.infrastructure.vector_index_manager import VectorIndexManager, IndexVersionInfo
    
    # 使用本地嵌入器（默认）
    manager = VectorIndexManager(rag_dir="outputs/ontology/rag")
    
    # 使用 OpenAI 嵌入器
    manager = VectorIndexManager(
        rag_dir="outputs/ontology/rag",
        embedder_config={"provider": "openai", "api_key": "sk-..."}
    )
    
    # 增量更新
    manager.incremental_update(
        new_chunks=[chunk1, chunk2],
        modified_chunks=[chunk3],
        deleted_chunk_ids=["chunk_001"]
    )
    
    # 获取向量索引
    vectors = manager.get_vectors()
    
    # 版本管理
    versions = manager.list_versions()
    manager.switch_version("v2")
    manager.rollback_to_version("v1")
    
    # 分层检索
    results = manager.compute_similarity_with_layers("query", topk=10, layers=["L1", "L2"])
"""

from __future__ import annotations

import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from src.application.rag.infrastructure.embedding import LocalEmbedder, OpenAIEmbedder, SentenceTransformersEmbedder, EmbedderConfig


@dataclass
class VectorIndexMetadata:
    """向量索引元数据"""
    version: str  # 索引版本号
    created_at: str  # 创建时间 ISO格式
    updated_at: str  # 更新时间 ISO格式
    chunk_count: int  # chunk 数量
    dimension: int  # 向量维度
    embedder_type: str  # 嵌入器类型
    config_hash: str  # 配置哈希（用于失效检测）
    source_manifest: Dict[str, str]  # 数据源清单 {chunk_id: content_hash}
    ontology_hash: str = ""  # 本体产物文件哈希（用于检测本体变更）
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> VectorIndexMetadata:
        return cls(**data)


@dataclass
class IndexVersionInfo:
    """索引版本信息"""
    version_id: str
    version_name: str
    created_at: str
    chunk_count: int
    status: str  # active, archived, deprecated
    description: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "IndexVersionInfo":
        return cls(**data)


@dataclass
class LayeredSimilarityResult:
    """分层检索结果"""
    chunk_id: str
    similarity_score: float
    layer: str
    chunk_content: str = ""
    source_file: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


class VectorIndexManager:
    """向量索引管理器 - 支持多版本管理和分层检索"""
    
    def __init__(
        self,
        rag_dir: Path | str,
        embedder: Optional[LocalEmbedder] = None,
        dimension: int = 384,
        embedder_config: Optional[Dict[str, Any]] = None
    ):
        self.rag_dir = Path(rag_dir)
        
        # [Phase 6-RAG-1] Support configurable embedder
        if embedder:
            self.embedder = embedder
        elif embedder_config:
            self.embedder = self._create_embedder_from_config(embedder_config)
        else:
            # 默认使用SentenceTransformersEmbedder作为真实语义嵌入
            embedder_config = EmbedderConfig(provider="sentence_transformers", model="paraphrase-multilingual-MiniLM-L12-v2", dimension=dimension)
            self.embedder = SentenceTransformersEmbedder(embedder_config)
        
        self.dimension = dimension
        self._embedder_config = embedder_config or {}
        
        # 索引存储路径
        self.index_dir = self.rag_dir / "vector_index"
        self.index_dir.mkdir(exist_ok=True)
        
        # 版本管理路径
        self.versions_dir = self.index_dir / "versions"
        self.versions_dir.mkdir(exist_ok=True)
        self.versions_meta_path = self.index_dir / "versions.json"
        
        # 当前版本路径（向后兼容）
        self.metadata_path = self.index_dir / "metadata.json"
        self.vectors_path = self.index_dir / "vectors.npz"
        self.chunk_id_path = self.index_dir / "chunk_ids.json"
        self.vocab_path = self.index_dir / "vocab.json"
        
        # 运行时状态
        self._vectors: Optional[np.ndarray] = None
        self._chunk_ids: List[str] = []
        self._metadata: Optional[VectorIndexMetadata] = None
        self._current_version: str = "v1"
        self._chunk_layer_map: Dict[str, str] = {}
        self._chunk_content_map: Dict[str, str] = {}
        self._chunk_source_map: Dict[str, str] = {}
        
    def _create_embedder_from_config(self, config: Dict[str, Any]):
        """[Phase 6-RAG-1] Create embedder from configuration dict."""
        provider = config.get("provider", "local").lower()
        
        if provider == "openai":
            api_key = config.get("api_key") or __import__('os').getenv("OPENAI_API_KEY")
            if not api_key:
                import warnings
                warnings.warn("OPENAI_API_KEY not found, falling back to local embedder")
                return LocalEmbedder(dimension=self.dimension)
            
            embedder_config = EmbedderConfig(
                provider="openai",
                api_key=api_key,
                model=config.get("model", "text-embedding-3-small"),
                timeout=config.get("timeout", 60.0),
                batch_size=config.get("batch_size", 100),
            )
            return OpenAIEmbedder(embedder_config)
        
        # Default: local embedder
        return LocalEmbedder(dimension=config.get("dimension", self.dimension))
    
    def _compute_content_hash(self, text: str) -> str:
        """计算内容哈希（用于变更检测）"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def _compute_config_hash(self) -> str:
        """计算配置哈希（用于失效检测）"""
        # 2026-10-16: embedder 维度接口不一致 —— LocalEmbedder 用 .dimension，
        # SentenceTransformersEmbedder 用 get_dimension()。取通用维度，避免
        # AttributeError 使真实索引构建失败。
        dim = getattr(self.embedder, "dimension", None)
        if dim is None:
            getter = getattr(self.embedder, "get_dimension", None)
            dim = getter() if callable(getter) else "unknown"
        config_str = f"{dim}_{type(self.embedder).__name__}"
        # Include embedder config in hash if available
        if self._embedder_config:
            # 2026-10-16: _embedder_config 可能是 dataclass（EmbedderConfig）而非 dict，
            # 直接 json.dumps 会 TypeError。统一转可序列化形式。
            cfg = self._embedder_config
            if not isinstance(cfg, dict):
                cfg = getattr(cfg, "__dict__", None) or str(cfg)
            try:
                config_str += f"_{json.dumps(cfg, sort_keys=True, default=str)}"
            except TypeError:
                config_str += f"_{cfg}"
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _compute_ontology_hash(self, ontology_dir: Optional[str] = None) -> str:
        """
        计算本体产物文件的内容哈希（用于检测本体变更）
        
        Args:
            ontology_dir: 本体产物目录，默认为 rag_dir 的父目录
            
        Returns:
            本体文件组合哈希
        """
        if ontology_dir is None:
            ontology_dir = self.rag_dir.parent / "ontology"
        
        ontology_path = Path(ontology_dir)
        if not ontology_path.exists():
            return ""
        
        combined_content = ""
        ontology_files = sorted(ontology_path.glob("*.json"))
        
        for file_path in ontology_files:
            if file_path.name in ("layer1_domain_ontology.json", "layer2_instances.json",
                                  "layer3_evidence.json", "ontology.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        combined_content += f.read()
                except Exception:
                    continue
        
        return hashlib.sha256(combined_content.encode()).hexdigest()[:16]
    
    def _load_metadata(self) -> Optional[VectorIndexMetadata]:
        """加载元数据"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return VectorIndexMetadata.from_dict(data)
        return None
    
    def _save_metadata(self, metadata: VectorIndexMetadata):
        """保存元数据"""
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _load_vectors(self) -> Optional[np.ndarray]:
        """加载向量数据"""
        if self.vectors_path.exists():
            data = np.load(self.vectors_path, allow_pickle=True)
            chunk_ids = self._load_chunk_ids()
            if data.files and chunk_ids:
                # 按 chunk_ids 顺序重建向量矩阵
                vectors = []
                for cid in chunk_ids:
                    if cid in data.files:
                        vec = data[cid]
                        # 确保向量是 1D
                        if vec.ndim > 1:
                            vec = vec.flatten()
                        vectors.append(vec)
                if vectors:
                    return np.vstack(vectors)
        return None
    
    def _save_vectors(self, vectors: np.ndarray, chunk_ids: List[str]):
        """保存向量数据"""
        # 保存为 npz，以 chunk_id 为 key，确保每个向量是 1D
        save_dict = {}
        for cid, vec in zip(chunk_ids, vectors):
            if vec.ndim > 1:
                vec = vec.flatten()
            save_dict[cid] = vec
        np.savez(self.vectors_path, **save_dict)

        # 同时保存 chunk_id 顺序
        with open(self.chunk_id_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_ids, f)
    
    def _load_chunk_ids(self) -> List[str]:
        """加载 chunk ID 列表"""
        if self.chunk_id_path.exists():
            with open(self.chunk_id_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def check_index_validity(self, chunks: List) -> Tuple[bool, str]:
        """
        检查索引是否有效
        
        Returns:
            (is_valid, reason)
        """
        metadata = self._load_metadata()
        if metadata is None:
            return False, "元数据不存在"
        
        # 检查配置哈希
        current_config_hash = self._compute_config_hash()
        if metadata.config_hash != current_config_hash:
            return False, f"配置变更: {metadata.config_hash} != {current_config_hash}"
        
        # [7.5] 检查本体产物文件哈希（检测本体变更）
        current_ontology_hash = self._compute_ontology_hash()
        if metadata.ontology_hash != current_ontology_hash:
            return False, f"本体产物变更: {metadata.ontology_hash} != {current_ontology_hash}"
        
        # 检查 chunk 数量
        if metadata.chunk_count != len(chunks):
            return False, f"chunk 数量不匹配: {metadata.chunk_count} != {len(chunks)}"
        
        # 检查每个 chunk 的内容哈希
        current_manifest = {
            c.chunk_id: self._compute_content_hash(c.content)
            for c in chunks
        }
        
        mismatched = []
        for chunk_id, current_hash in current_manifest.items():
            stored_hash = metadata.source_manifest.get(chunk_id)
            if stored_hash != current_hash:
                mismatched.append(chunk_id)
        
        if mismatched:
            return False, f"内容变更的 chunks: {mismatched[:5]}..."
        
        return True, "索引有效"
    
    def build_index(
        self,
        chunks: List,
        force_rebuild: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """
        构建向量索引
        
        Args:
            chunks: chunk 对象列表（需有 chunk_id 和 content 属性）
            force_rebuild: 是否强制重建
            
        Returns:
            (vectors, chunk_ids)
        """
        # 检查是否可以复用现有索引
        if not force_rebuild:
            is_valid, reason = self.check_index_validity(chunks)
            if is_valid:
                print(f"[VectorIndex] 索引有效，跳过重建 ({reason})")
                self._vectors = self._load_vectors()
                self._chunk_ids = self._load_chunk_ids()
                return self._vectors, self._chunk_ids
            else:
                print(f"[VectorIndex] 需要重建: {reason}")
        
        # 构建新索引
        print(f"[VectorIndex] 构建索引: {len(chunks)} 个 chunks...")
        texts = [c.content for c in chunks]
        vectors = self.embedder.embed(texts)
        chunk_ids = [c.chunk_id for c in chunks]
        
        # 创建元数据
        source_manifest = {
            c.chunk_id: self._compute_content_hash(c.content)
            for c in chunks
        }
        
        metadata = VectorIndexMetadata(
            version="1.0",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            chunk_count=len(chunks),
            dimension=vectors.shape[1],
            embedder_type=type(self.embedder).__name__,
            config_hash=self._compute_config_hash(),
            source_manifest=source_manifest,
            ontology_hash=self._compute_ontology_hash()
        )
        
        # 保存
        self._save_vectors(vectors, chunk_ids)
        self._save_metadata(metadata)
        
        # 保存词汇表（如果是 LocalEmbedder）
        if hasattr(self.embedder, 'save'):
            self.embedder.save(self.vocab_path)
        
        self._vectors = vectors
        self._chunk_ids = chunk_ids
        self._metadata = metadata
        
        self._update_chunk_maps(chunks)
        
        print(f"[VectorIndex] 索引构建完成: {len(chunk_ids)} 个向量")
        return vectors, chunk_ids
    
    def incremental_update(
        self,
        new_chunks: List,
        modified_chunks: List,
        deleted_chunk_ids: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        增量更新向量索引
        
        Args:
            new_chunks: 新增的 chunks
            modified_chunks: 修改的 chunks
            deleted_chunk_ids: 删除的 chunk IDs
            
        Returns:
            (updated_vectors, updated_chunk_ids)
        """
        # 加载现有索引
        existing_vectors = self._load_vectors()
        existing_chunk_ids = self._load_chunk_ids()
        metadata = self._load_metadata()
        
        if existing_vectors is None:
            print("[VectorIndex] 现有索引不存在，执行完整重建...")
            all_chunks = new_chunks + modified_chunks
            return self.build_index(all_chunks, force_rebuild=True)
        
        print(f"[VectorIndex] 增量更新: +{len(new_chunks)} new, ~{len(modified_chunks)} mod, -{len(deleted_chunk_ids)} del")
        
        # 构建 ID -> 向量的映射
        id_to_vec = dict(zip(existing_chunk_ids, existing_vectors))

        # 删除
        for cid in deleted_chunk_ids:
            id_to_vec.pop(cid, None)
            if metadata:
                metadata.source_manifest.pop(cid, None)
        
        # 更新（删除旧版本，添加新版本）
        for chunk in modified_chunks:
            # 重新计算向量
            vec = self.embedder.embed([chunk.content])[0]
            id_to_vec[chunk.chunk_id] = vec
            if metadata:
                metadata.source_manifest[chunk.chunk_id] = self._compute_content_hash(chunk.content)
        
        # 新增
        if new_chunks:
            new_texts = [c.content for c in new_chunks]
            new_vectors = self.embedder.embed(new_texts)
            for chunk, vec in zip(new_chunks, new_vectors):
                id_to_vec[chunk.chunk_id] = vec
                if metadata:
                    metadata.source_manifest[chunk.chunk_id] = self._compute_content_hash(chunk.content)
        
        # 重新构建有序列表
        updated_chunk_ids = list(id_to_vec.keys())
        updated_vectors = np.array([id_to_vec[cid] for cid in updated_chunk_ids])
        
        # 更新元数据
        if metadata:
            metadata.updated_at = datetime.now().isoformat()
            metadata.chunk_count = len(updated_chunk_ids)
            self._save_metadata(metadata)
        
        # 保存
        self._save_vectors(updated_vectors, updated_chunk_ids)
        
        self._vectors = updated_vectors
        self._chunk_ids = updated_chunk_ids
        
        all_chunks = new_chunks + modified_chunks
        self._update_chunk_maps(all_chunks)
        
        print(f"[VectorIndex] 增量更新完成: {len(updated_chunk_ids)} 个向量")
        return updated_vectors, updated_chunk_ids
    
    def get_vectors(self) -> Optional[np.ndarray]:
        """获取当前向量矩阵"""
        if self._vectors is None:
            self._vectors = self._load_vectors()
            # 加载词汇表（如果是 LocalEmbedder）
            if self._vectors is not None and hasattr(self.embedder, 'load'):
                self.embedder.load(self.vocab_path)
        return self._vectors
    
    def get_chunk_ids(self) -> List[str]:
        """获取 chunk ID 列表"""
        if not self._chunk_ids:
            self._chunk_ids = self._load_chunk_ids()
        return self._chunk_ids
    
    def get_metadata(self) -> Optional[VectorIndexMetadata]:
        """获取元数据"""
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return self._metadata
    
    def compute_similarity(
        self,
        query_text: str,
        topk: int = 10
    ) -> List[Tuple[str, float]]:
        """
        计算查询文本与所有 chunk 的相似度
        
        Returns:
            [(chunk_id, similarity_score), ...] 按相似度排序
        """
        vectors = self.get_vectors()
        chunk_ids = self.get_chunk_ids()
        
        if vectors is None or len(chunk_ids) == 0:
            return []
        
        # 计算查询向量
        query_vec = self.embedder.embed([query_text])[0]
        
        # 余弦相似度
        similarities = np.dot(vectors, query_vec) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec) + 1e-8
        )
        
        # 排序
        top_indices = np.argsort(similarities)[::-1][:topk]
        return [(chunk_ids[i], float(similarities[i])) for i in top_indices]


    def _load_versions_metadata(self) -> Dict[str, IndexVersionInfo]:
        """加载版本元数据"""
        if self.versions_meta_path.exists():
            with open(self.versions_meta_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {k: IndexVersionInfo.from_dict(v) for k, v in data.items()}
        return {}
    
    def _save_versions_metadata(self, versions: Dict[str, IndexVersionInfo]):
        """保存版本元数据"""
        with open(self.versions_meta_path, 'w', encoding='utf-8') as f:
            json.dump({k: v.to_dict() for k, v in versions.items()}, f, indent=2, ensure_ascii=False)
    
    def _get_next_version_id(self) -> str:
        """生成下一个版本 ID"""
        versions = self._load_versions_metadata()
        max_v = 1
        for v_id in versions.keys():
            if v_id.startswith('v') and v_id[1:].isdigit():
                max_v = max(max_v, int(v_id[1:]) + 1)
        return f"v{max_v}"
    
    def list_versions(self) -> List[IndexVersionInfo]:
        """列出所有版本"""
        versions = self._load_versions_metadata()
        return sorted(versions.values(), key=lambda x: x.created_at, reverse=True)
    
    def create_version(self, description: str = "") -> str:
        """创建新版本快照"""
        version_id = self._get_next_version_id()
        version_dir = self.versions_dir / version_id
        version_dir.mkdir(exist_ok=True)
        
        # 复制当前索引文件到版本目录
        for src_path, dst_name in [
            (self.metadata_path, "metadata.json"),
            (self.vectors_path, "vectors.npz"),
            (self.chunk_id_path, "chunk_ids.json"),
        ]:
            if src_path.exists():
                import shutil
                shutil.copy(src_path, version_dir / dst_name)
        
        # 创建版本信息
        metadata = self.get_metadata()
        version_info = IndexVersionInfo(
            version_id=version_id,
            version_name=f"Vector Index {version_id}",
            created_at=datetime.now().isoformat(),
            chunk_count=metadata.chunk_count if metadata else 0,
            status="active",
            description=description
        )
        
        # 更新版本元数据
        versions = self._load_versions_metadata()
        versions[version_id] = version_info
        self._save_versions_metadata(versions)
        
        print(f"[VectorIndex] 创建版本 {version_id}")
        return version_id
    
    def switch_version(self, version_id: str) -> bool:
        """切换到指定版本"""
        version_dir = self.versions_dir / version_id
        if not version_dir.exists():
            return False
        
        # 复制版本文件到当前目录
        import shutil
        for src_name, dst_path in [
            ("metadata.json", self.metadata_path),
            ("vectors.npz", self.vectors_path),
            ("chunk_ids.json", self.chunk_id_path),
        ]:
            src_path = version_dir / src_name
            if src_path.exists():
                shutil.copy(src_path, dst_path)
        
        # 更新运行时状态
        self._vectors = None
        self._chunk_ids = []
        self._metadata = None
        self._current_version = version_id
        
        print(f"[VectorIndex] 切换到版本 {version_id}")
        return True
    
    def rollback_to_version(self, version_id: str) -> bool:
        """回滚到指定版本"""
        if self.switch_version(version_id):
            # 将被回滚的版本标记为 deprecated
            versions = self._load_versions_metadata()
            for vid, info in versions.items():
                if vid != version_id and info.status == "active":
                    info.status = "deprecated"
            self._save_versions_metadata(versions)
            print(f"[VectorIndex] 回滚到版本 {version_id}")
            return True
        return False
    
    def _update_chunk_maps(self, chunks: List):
        """Update chunk metadata maps (for layered retrieval + chunk resolution).

        Populated during build_index / incremental_update. For process restarts
        these in-memory maps are empty; callers MUST go through
        ``_ensure_chunk_maps`` which performs a lazy JSONL backfill.
        """
        for c in chunks:
            self._chunk_layer_map[c.chunk_id] = getattr(c, 'layer', 'unknown')
            self._chunk_content_map[c.chunk_id] = getattr(c, 'content', '')
            self._chunk_source_map[c.chunk_id] = getattr(c, 'source_file', '')

    def _ensure_chunk_maps(self) -> None:
        """Lazily populate chunk-layer/content/source maps after process restart.

        P0 BUGFIX (2026-08-04):
        - ``_chunk_layer_map`` / ``_chunk_content_map`` / ``_chunk_source_map``
          were only ever populated by ``_update_chunk_maps`` which runs inside
          ``build_index`` / ``incremental_update``. After a process restart the
          caller would e.g. ``VectorIndexManager(rag_dir).get_vectors()`` (which
          correctly reloads vectors/chunk_ids from disk) and then call
          ``compute_similarity_with_layers`` -> every result silently falls
          into ``layer='unknown'`` and layered filter ``layers=["L1","L2"]``
          returns 0 hits despite identical data on disk.
        - Resolution: if any map is empty, read ``<rag_dir>/chunks.jsonl`` via
          the canonical loader and repopulate. This is one-off O(N) cost, paid
          once per manager instance on the first layered call.
        """
        if (
            self._chunk_layer_map
            and self._chunk_content_map
            and self._chunk_source_map
        ):
            return  # maps already populated (build or previous ensure)

        chunks_jsonl = self.rag_dir / "chunks.jsonl"
        if not chunks_jsonl.exists():
            # Nothing we can backfill from; leave maps empty (behavior unchanged)
            return

        try:
            from ..chunk_indexing import load_chunks_jsonl
        except Exception:
            return

        loaded = load_chunks_jsonl(chunks_jsonl)
        if not loaded:
            return

        for c in loaded:
            cid = getattr(c, 'chunk_id', None)
            if not cid:
                continue
            if cid not in self._chunk_layer_map:
                self._chunk_layer_map[cid] = getattr(c, 'layer', 'unknown')
            if cid not in self._chunk_content_map:
                self._chunk_content_map[cid] = getattr(c, 'content', '')
            if cid not in self._chunk_source_map:
                self._chunk_source_map[cid] = getattr(c, 'source_file', '')
    
    def compute_similarity_with_layers(
        self,
        query_text: str,
        topk: int = 10,
        layers: Optional[List[str]] = None
    ) -> List[LayeredSimilarityResult]:
        """
        分层相似度检索

        P0 BUGFIX (2026-08-04): Lazy backfill of chunk-layer maps after restart
        (see ``_ensure_chunk_maps``).

        Args:
            query_text: 查询文本
            topk: 返回数量
            layers: 过滤层列表（如 ["L0", "L1", "L2"]），None 表示所有层

        Returns:
            LayeredSimilarityResult 列表，按相似度排序
        """
        self._ensure_chunk_maps()  # <-- BUGFIX: guarantees maps non-empty after restart
        raw_results = self.compute_similarity(query_text, topk=topk * 3)
        
        filtered = []
        for chunk_id, score in raw_results:
            layer = self._chunk_layer_map.get(chunk_id, 'unknown')
            
            if layers and layer not in layers:
                continue
            
            filtered.append(LayeredSimilarityResult(
                chunk_id=chunk_id,
                similarity_score=score,
                layer=layer,
                chunk_content=self._chunk_content_map.get(chunk_id, ''),
                source_file=self._chunk_source_map.get(chunk_id, '')
            ))
        
        return sorted(filtered, key=lambda x: x.similarity_score, reverse=True)[:topk]
    
    def get_layers(self) -> List[str]:
        """获取所有可用的层"""
        self._ensure_chunk_maps()  # BUGFIX 2026-08-04: lazy backfill after restart
        return sorted(set(self._chunk_layer_map.values()))
    
    def get_layer_statistics(self) -> Dict[str, int]:
        """获取各层的 chunk 数量统计"""
        self._ensure_chunk_maps()  # BUGFIX 2026-08-04: lazy backfill after restart
        stats: Dict[str, int] = {}
        for layer in self._chunk_layer_map.values():
            stats[layer] = stats.get(layer, 0) + 1
        return stats


# 向后兼容的简化接口
def build_vector_index(
    chunks: List,
    rag_dir: Path | str,
    embedder: Optional[LocalEmbedder] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    简化接口：构建向量索引
    
    Args:
        chunks: chunk 对象列表
        rag_dir: RAG 目录路径
        embedder: 可选的 embedder 实例
        
    Returns:
        (vectors, chunk_ids)
    """
    manager = VectorIndexManager(rag_dir, embedder)
    return manager.build_index(chunks)


def incremental_update_index(
    rag_dir: Path | str,
    new_chunks: List = None,
    modified_chunks: List = None,
    deleted_chunk_ids: List[str] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    简化接口：增量更新向量索引
    
    Args:
        rag_dir: RAG 目录路径
        new_chunks: 新增的 chunks
        modified_chunks: 修改的 chunks
        deleted_chunk_ids: 删除的 chunk IDs
        
    Returns:
        (updated_vectors, updated_chunk_ids)
    """
    manager = VectorIndexManager(rag_dir)
    return manager.incremental_update(
        new_chunks or [],
        modified_chunks or [],
        deleted_chunk_ids or []
    )
