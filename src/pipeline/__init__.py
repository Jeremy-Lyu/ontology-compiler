"""Pipeline Module

架构说明（参考 PROJECT_ARCHITECTURE_AUDIT_REPORT_20260708.md）：
- MainPipeline: 主线生产管道，标准、稳定、默认的生产路径
- PipelineRouter: 统一入口路由
- stages: 阶段实现，所有 pipeline 共享的阶段函数

使用建议：
- 生产环境：使用 pipeline_router.get_production_pipeline() 或 MainPipeline

所有阶段函数都定义在 stages/ 子模块中。

注意：DualPipeline 与 ConfigurablePipeline 已停用；仅保留历史兼容实现，不应作为入口使用。
"""

from .main_pipeline import (
    MainPipeline,
    PipelineContext,
    PipelineStage,
    PipelineStatus,
    PipelineExecutionResult,
    create_main_pipeline,
    run_main_pipeline,
)
from .pipeline_router import (
    PipelineRoute,
    PipelineType,
    PipelineInfo,
    get_pipeline,
    get_pipeline_info,
    list_pipelines,
    get_production_pipeline,
    is_production_route,
    get_pipeline_stats,
    print_pipeline_help,
)
from .stages import *

_pipeline_all = [
    "MainPipeline",
    "PipelineContext",
    "PipelineStage",
    "PipelineStatus",
    "PipelineExecutionResult",
    "create_main_pipeline",
    "run_main_pipeline",
    "PipelineRoute",
    "PipelineType",
    "PipelineInfo",
    "get_pipeline",
    "get_pipeline_info",
    "list_pipelines",
    "get_production_pipeline",
    "is_production_route",
    "get_pipeline_stats",
    "print_pipeline_help",
]

try:
    __all__ = _pipeline_all + __all__
except NameError:
    __all__ = _pipeline_all