"""Pipeline Stages

所有 pipeline 阶段函数都定义在本模块中。

阶段函数列表:
- preflight: 预检查
- health: 健康检查
- extract: 文档抽取
- merge: 本体合并
- dedup: 去重
- graph_fusion: 图代数拓扑融合（步骤2）
- audit: 审计
- dcs_audit: 维度覆盖度审计（步骤3）
- guard: 门禁检查
- emergent_reasoning: 涌涌推理计算（步骤4）
- export: 导出
- publish: 发布
- rollback: 回滚
- summarize: 汇总
"""
from .common import StageResult, PROJECT_ROOT
from .preflight import run_preflight, run_health, expand_stages
from .extract import run_extract
from .relationship import run_relationship
from .merge import run_merge, run_dedup
from .graph_fusion import run_graph_fusion
from .audit import run_audit, run_guard
from .dcs_audit import run_dcs_audit
from .emergent_reasoning import run_emergent_reasoning
from .export import run_export, run_publish, run_rollback
from .document_report import run_document_report
from .summarize import run_summarize

DEFAULT_PIPELINE_DIR = PROJECT_ROOT / "outputs" / "runs" / "ontology" / "pipeline"
DEFAULT_MASTER_PATH = PROJECT_ROOT / "outputs" / "runs" / "ontology" / "master" / "master_ontology"

__all__ = [
    "StageResult",
    "PROJECT_ROOT",
    "DEFAULT_PIPELINE_DIR",
    "DEFAULT_MASTER_PATH",
    "run_preflight",
    "run_health",
    "run_extract",
    "run_relationship",
    "run_merge",
    "run_dedup",
    "run_graph_fusion",
    "run_audit",
    "run_dcs_audit",
    "run_guard",
    "run_emergent_reasoning",
    "run_export",
    "run_publish",
    "run_rollback",
    "run_document_report",
    "run_summarize",
    "expand_stages",
]