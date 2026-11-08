"""
RAG Pipeline 配置加载器

基于 QA 报告优化：硬编码参数迁移到 YAML 配置

Usage:
    from src.infrastructure.config.rag_config import get_rag_config
    
    config = get_rag_config()
    
    # 访问配置
    char_budget = config.relevance_gate.char_budget
    thresholds = config.quality_scorer.thresholds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class CharBudgetConfig:
    """字符预算配置

    P2 Enhancement (2026-05-26): Supports dynamic calculation based on LLM token limits.
    Use calculate_for_model() to get budget adjusted for specific model capabilities.
    """
    ontology: int = 50_000
    data: int = 10_000
    max_blocks: int = 50
    max_data_rows: int = 200

    # P2: Model-specific token limits for dynamic budget calculation
    _MODEL_TOKEN_LIMITS: Dict[str, int] = field(default_factory=lambda: {
        # OpenAI models
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "gpt-3.5-turbo": 16385,
        # Anthropic models
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        # DeepSeek models
        "deepseek-chat": 64000,
        "deepseek-coder": 64000,
        # Default for unknown models
        "default": 8192,
    })

    # P2: Budget allocation ratios (as fraction of available context)
    _ONTOLOGY_RATIO: float = 0.6  # 60% for ontology context
    _DATA_RATIO: float = 0.15     # 15% for data context
    _RESERVE_RATIO: float = 0.25  # 25% reserved for output and overhead

    def get_model_token_limit(self, model_name: str) -> int:
        """Get token limit for a specific model."""
        return self._MODEL_TOKEN_LIMITS.get(model_name, self._MODEL_TOKEN_LIMITS["default"])

    def calculate_for_model(
        self,
        model_name: str,
        avg_chars_per_token: float = 4.0,
        safety_margin: float = 0.9,
    ) -> "CharBudgetConfig":
        """Calculate budget configuration for a specific LLM model.

        P2 Feature: Dynamically calculates character budgets based on the model's
        token limit, ensuring optimal context window utilization.

        Args:
            model_name: Name of the LLM model (e.g., "gpt-4", "claude-3-opus")
            avg_chars_per_token: Average characters per token (default 4.0 for English)
            safety_margin: Safety multiplier (default 0.9 to leave 10% headroom)

        Returns:
            New CharBudgetConfig with adjusted budgets for the model
        """
        token_limit = self.get_model_token_limit(model_name)
        available_tokens = int(token_limit * safety_margin * (1 - self._RESERVE_RATIO))

        # Calculate character budgets
        ontology_chars = int(available_tokens * self._ONTOLOGY_RATIO * avg_chars_per_token)
        data_chars = int(available_tokens * self._DATA_RATIO * avg_chars_per_token)

        # Scale max_blocks and max_data_rows proportionally (capped at reasonable limits)
        scale_factor = min(token_limit / 8192, 3.0)  # Cap at 3x for very large models
        max_blocks = min(int(50 * scale_factor), 150)
        max_data_rows = min(int(200 * scale_factor), 600)

        return CharBudgetConfig(
            ontology=ontology_chars,
            data=data_chars,
            max_blocks=max_blocks,
            max_data_rows=max_data_rows,
        )

    def with_auto_detect(self, model_name: str | None = None) -> "CharBudgetConfig":
        """Auto-detect model and return appropriate budget configuration.

        P2 Feature: If model_name is provided, uses that; otherwise attempts to
        detect from environment or falls back to defaults.

        Args:
            model_name: Optional model name to use; if None, attempts auto-detection

        Returns:
            CharBudgetConfig with appropriate budgets
        """
        if model_name:
            return self.calculate_for_model(model_name)

        # Attempt auto-detection from environment
        import os
        env_model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or os.getenv("ANTHROPIC_MODEL")
        if env_model:
            # Try to match partial names
            for known_model in self._MODEL_TOKEN_LIMITS.keys():
                if known_model in env_model.lower():
                    return self.calculate_for_model(known_model)

        # Fall back to default budgets
        return self


@dataclass(frozen=True)
class ThresholdConfig:
    """阈值配置"""
    full: float = 0.7
    summary: float = 0.3
    metadata: float = 0.1
    min_score: float = 0.05


@dataclass(frozen=True)
class RelevanceGateConfig:
    """RelevanceGate 配置"""
    char_budget: CharBudgetConfig = field(default_factory=CharBudgetConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    type_weights: Dict[str, float] = field(default_factory=lambda: {
        "fragment": 1.4,
        "core_results": 1.2,
        "domain_report": 1.1,
        "l2_instances": 1.0,
        "l1_sandbox": 0.9,
        "manifest": 0.7,
    })


@dataclass(frozen=True)
class QualityWeightsConfig:
    """质量评分权重配置"""
    node_completeness: float = 0.33
    relationship_validity: float = 0.33
    domain_consistency: float = 0.34


@dataclass(frozen=True)
class LLMConfig:
    """LLM 配置"""
    temperature: float = 0.3
    max_tokens: int = 4000
    enable_thinking: bool = True


@dataclass(frozen=True)
class RigorousConfig:
    """RIGOR 引擎配置

    P1-3 Enhancement (2026-05-30):
    - Added rigor_enabled for explicit production control
    - Added rigor_mode for logging and status tracking
    """
    max_iterations: int = 3
    confidence_threshold: float = 0.85
    min_improvement: float = 0.05
    rigor_enabled: bool = True  # P1-3: Explicit enable/disable for production control
    quality_weights: QualityWeightsConfig = field(default_factory=QualityWeightsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


@dataclass(frozen=True)
class ChunkBuilderConfig:
    """Chunk Builder 配置"""
    supported_meta_classes: List[str] = field(default_factory=lambda: [
        "Capability", "System", "Process", "Data", "Actor", "Asset"
    ])
    limits: Dict[str, int] = field(default_factory=lambda: {
        "max_content_length": 2000,
        "max_linked_entities": 50,
        "max_edge_types": 20,
    })
    prefixes: Dict[str, str] = field(default_factory=lambda: {
        "capability": "cap_",
        "path": "path_",
        "system_mapping": "sysmap_",
        "process": "proc_",
        "data": "data_",
        "actor": "actor_",
        "asset": "asset_",
    })


@dataclass(frozen=True)
class VectorIndexConfig:
    """向量索引配置"""
    dimension: int = 384
    similarity_metric: str = "cosine"
    cache: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "file_name": "chunk_embeddings.npz",
        "compression": True,
    })
    incremental: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "content_hash_algorithm": "sha256",
        "detect_deleted": True,
    })


@dataclass(frozen=True)
class HybridIndexConfig:
    """混合索引配置"""
    enabled: bool = True
    search_weights: Dict[str, Any] = field(default_factory=lambda: {
        "vector": 0.6,
        "keyword": 0.4,
        "vector_top_k": 50,
    })
    incremental: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "detect_deleted": True,
        "auto_rebuild_threshold": 0.3,
    })


@dataclass(frozen=True)
class OnlineWeightsConfig:
    """在线评估权重配置"""
    completeness: float = 0.40
    consistency: float = 0.40
    provenance: float = 0.20


@dataclass(frozen=True)
class OfflineWeightsConfig:
    """离线评估权重配置"""
    coverage: float = 0.20
    scale: float = 0.25
    depth: float = 0.25
    diversity: float = 0.15
    l1_score: float = 0.15


@dataclass(frozen=True)
class GradeThresholdsConfig:
    """等级阈值配置"""
    A: float = 0.90
    B: float = 0.75
    C: float = 0.60
    D: float = 0.40
    E: float = 0.00


@dataclass(frozen=True)
class QualityScorerConfig:
    """Quality Scorer 配置"""
    online_weights: OnlineWeightsConfig = field(default_factory=OnlineWeightsConfig)
    offline_weights: OfflineWeightsConfig = field(default_factory=OfflineWeightsConfig)
    grade_thresholds: GradeThresholdsConfig = field(default_factory=GradeThresholdsConfig)


@dataclass(frozen=True)
class DataAnalysisConfig:
    """数据分析配置"""
    max_rounds: int = 5
    convergence_threshold: float = 0.85
    relevance: Dict[str, Any] = field(default_factory=lambda: {
        "char_budget": {"ontology": 50_000, "data": 10_000},
    })
    patch_adapter: Dict[str, Any] = field(default_factory=lambda: {
        "use_llm_enhance": True,
        "min_confidence": 0.4,
        "max_operations": 50,
    })


@dataclass(frozen=True)
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    performance: Dict[str, bool] = field(default_factory=lambda: {
        "log_timing": True,
        "log_memory": False,
        "log_token_usage": True,
    })


@dataclass(frozen=True)
class RAGPipelineConfig:
    """RAG Pipeline 总配置"""
    version: str = "1.0.0"
    last_updated: str = "2026-05-22"
    
    relevance_gate: RelevanceGateConfig = field(default_factory=RelevanceGateConfig)
    rigor: RigorousConfig = field(default_factory=RigorousConfig)
    chunk_builder: ChunkBuilderConfig = field(default_factory=ChunkBuilderConfig)
    vector_index: VectorIndexConfig = field(default_factory=VectorIndexConfig)
    hybrid_index: HybridIndexConfig = field(default_factory=HybridIndexConfig)
    quality_scorer: QualityScorerConfig = field(default_factory=QualityScorerConfig)
    data_analysis: DataAnalysisConfig = field(default_factory=DataAnalysisConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# 全局配置缓存
_config_instance: Optional[RAGPipelineConfig] = None


def load_yaml_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    if path is None:
        # 默认路径：项目根目录下的 config/rag_pipeline.yaml
        path = Path(__file__).resolve().parents[3] / "config" / "rag_pipeline.yaml"
    
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dict_to_dataclass(data: Dict[str, Any]) -> RAGPipelineConfig:
    """将字典转换为 dataclass"""
    
    # Helper to safely get nested dict
    def get_section(key: str, default: Dict) -> Dict:
        return data.get(key, default) if isinstance(data, dict) else default
    
    # RelevanceGate
    rg_data = get_section("relevance_gate", {})
    char_budget = CharBudgetConfig(**rg_data.get("char_budget", {}))
    thresholds = ThresholdConfig(**rg_data.get("thresholds", {}))
    relevance_gate = RelevanceGateConfig(
        char_budget=char_budget,
        thresholds=thresholds,
        type_weights=rg_data.get("type_weights", RelevanceGateConfig().type_weights),
    )
    
    # RIGOR
    rigor_data = get_section("rigor", {})
    quality_weights = QualityWeightsConfig(**rigor_data.get("quality_weights", {}))
    llm = LLMConfig(**rigor_data.get("llm", {}))
    rigor = RigorousConfig(
        max_iterations=rigor_data.get("max_iterations", 3),
        confidence_threshold=rigor_data.get("confidence_threshold", 0.85),
        min_improvement=rigor_data.get("min_improvement", 0.05),
        rigor_enabled=rigor_data.get("rigor_enabled", True),  # P1-3: Load from config
        quality_weights=quality_weights,
        llm=llm,
    )
    
    # Chunk Builder
    cb_data = get_section("chunk_builder", {})
    chunk_builder = ChunkBuilderConfig(
        supported_meta_classes=cb_data.get("supported_meta_classes", ChunkBuilderConfig().supported_meta_classes),
        limits=cb_data.get("limits", ChunkBuilderConfig().limits),
        prefixes=cb_data.get("prefixes", ChunkBuilderConfig().prefixes),
    )
    
    # Vector Index
    vi_data = get_section("vector_index", {})
    vector_index = VectorIndexConfig(
        dimension=vi_data.get("dimension", 384),
        similarity_metric=vi_data.get("similarity_metric", "cosine"),
        cache=vi_data.get("cache", VectorIndexConfig().cache),
        incremental=vi_data.get("incremental", VectorIndexConfig().incremental),
    )

    # Hybrid Index
    hi_data = get_section("hybrid_index", {})
    hybrid_index = HybridIndexConfig(
        enabled=hi_data.get("enabled", True),
        search_weights=hi_data.get("search_weights", HybridIndexConfig().search_weights),
        incremental=hi_data.get("incremental", HybridIndexConfig().incremental),
    )
    
    # Quality Scorer
    qs_data = get_section("quality_scorer", {})
    online_weights = OnlineWeightsConfig(**qs_data.get("online_weights", {}))
    offline_weights = OfflineWeightsConfig(**qs_data.get("offline_weights", {}))
    grade_thresholds = GradeThresholdsConfig(**qs_data.get("grade_thresholds", {}))
    quality_scorer = QualityScorerConfig(
        online_weights=online_weights,
        offline_weights=offline_weights,
        grade_thresholds=grade_thresholds,
    )
    
    # Data Analysis
    da_data = get_section("data_analysis", {})
    data_analysis = DataAnalysisConfig(
        max_rounds=da_data.get("max_rounds", 5),
        convergence_threshold=da_data.get("convergence_threshold", 0.85),
        relevance=da_data.get("relevance", DataAnalysisConfig().relevance),
        patch_adapter=da_data.get("patch_adapter", DataAnalysisConfig().patch_adapter),
    )
    
    # Logging
    log_data = get_section("logging", {})
    logging = LoggingConfig(
        level=log_data.get("level", "INFO"),
        format=log_data.get("format", LoggingConfig().format),
        performance=log_data.get("performance", LoggingConfig().performance),
    )
    
    return RAGPipelineConfig(
        version=data.get("version", "1.0.0"),
        last_updated=data.get("last_updated", "2026-05-22"),
        relevance_gate=relevance_gate,
        rigor=rigor,
        chunk_builder=chunk_builder,
        vector_index=vector_index,
        hybrid_index=hybrid_index,
        quality_scorer=quality_scorer,
        data_analysis=data_analysis,
        logging=logging,
    )


def get_rag_config(path: Optional[Path] = None) -> RAGPipelineConfig:
    """
    获取 RAG Pipeline 配置（单例模式）
    
    Args:
        path: YAML 配置文件路径，默认为 config/rag_pipeline.yaml
        
    Returns:
        RAGPipelineConfig 实例
    """
    global _config_instance
    
    if _config_instance is None:
        yaml_data = load_yaml_config(path)
        _config_instance = dict_to_dataclass(yaml_data)
    
    return _config_instance


def reload_rag_config(path: Optional[Path] = None) -> RAGPipelineConfig:
    """重新加载配置（用于热更新）"""
    global _config_instance
    _config_instance = None
    return get_rag_config(path)


# 便捷访问函数
def get_relevance_gate_config() -> RelevanceGateConfig:
    """获取 RelevanceGate 配置"""
    return get_rag_config().relevance_gate


def get_rigor_config() -> RigorousConfig:
    """获取 RIGOR 配置"""
    return get_rag_config().rigor


def get_chunk_builder_config() -> ChunkBuilderConfig:
    """获取 Chunk Builder 配置"""
    return get_rag_config().chunk_builder


def get_quality_scorer_config() -> QualityScorerConfig:
    """获取 Quality Scorer 配置"""
    return get_rag_config().quality_scorer
