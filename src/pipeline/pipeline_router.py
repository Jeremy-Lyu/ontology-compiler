"""Pipeline Router - 统一入口路由

当前只保留 `MAIN` 作为可用生产入口；其余路由仅作为历史兼容登记，
不会作为新的执行入口开放。

使用方式：
    from src.pipeline.pipeline_router import get_production_pipeline
    pipeline = get_production_pipeline()
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class PipelineRoute(Enum):
    """Pipeline 路由类型。

    只有 `MAIN` 是可用路由；`DUAL` 保留为过渡观测登记（2027-01-31）。
    （P1-3 2026-09-09：AGENT/CONFIG 注册行已删——指向模块不存在，零调用。）
    """
    MAIN = "main"
    DUAL = "dual"


class PipelineType(Enum):
    """Pipeline 类型分类"""
    PRODUCTION = "production"
    MIGRATION = "migration"
    EXPERIMENTAL = "experimental"


class PipelineInfo:
    """Pipeline 信息描述"""
    
    def __init__(
        self,
        route: PipelineRoute,
        name: str,
        description: str,
        pipeline_type: PipelineType,
        module_path: str,
        class_name: str,
        enabled: bool = True,
        deprecated: bool = False,
        deprecation_message: str = "",
    ):
        self.route = route
        self.name = name
        self.description = description
        self.pipeline_type = pipeline_type
        self.module_path = module_path
        self.class_name = class_name
        self.enabled = enabled
        self.deprecated = deprecated
        self.deprecation_message = deprecation_message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route.value,
            "name": self.name,
            "description": self.description,
            "type": self.pipeline_type.value,
            "module_path": self.module_path,
            "class_name": self.class_name,
            "enabled": self.enabled,
            "deprecated": self.deprecated,
            "deprecation_message": self.deprecation_message,
        }


# Pipeline 路由注册表
_PIPELINE_REGISTRY = {
    PipelineRoute.MAIN: PipelineInfo(
        route=PipelineRoute.MAIN,
        name="MainPipeline",
        description="主线生产管道 - 标准、稳定、默认的生产路径",
        pipeline_type=PipelineType.PRODUCTION,
        module_path="src.pipeline.main_pipeline",
        class_name="MainPipeline",
        enabled=True,
        deprecated=False,
    ),
    PipelineRoute.DUAL: PipelineInfo(
        route=PipelineRoute.DUAL,
        name="DualPipeline",
        description="支线迁移管道 - 仅保留历史兼容",
        pipeline_type=PipelineType.MIGRATION,
        module_path="src.pipeline.dual_pipeline",
        class_name="DualPipeline",
        enabled=False,
        deprecated=True,
        deprecation_message="已停用：仅保留历史兼容，不作为新入口开放",
    ),
}


def get_pipeline_info(route: PipelineRoute) -> Optional[PipelineInfo]:
    """获取指定路由的 Pipeline 信息"""
    return _PIPELINE_REGISTRY.get(route)


def list_pipelines() -> Dict[str, PipelineInfo]:
    """列出所有可用的 Pipeline"""
    return {route.value: info for route, info in _PIPELINE_REGISTRY.items()}


def _adapt_context(route: PipelineRoute, args, kwargs):
    """适配上下文参数，支持 dict 和 PipelineContext 两种格式
    
    Args:
        route: PipelineRoute 枚举值
        args: 位置参数
        kwargs: 关键字参数
    
    Returns:
        适配后的位置参数和关键字参数
    """
    if route == PipelineRoute.MAIN:
        if len(args) == 1 and isinstance(args[0], dict):
            context_dict = args[0]
            source_path = context_dict.get("source_path", "")
            domain = context_dict.get("domain", "AUTO")
            context_kwargs = {k: v for k, v in context_dict.items() if k not in ["source_path", "domain"]}
            
            from .main_pipeline import create_main_pipeline
            return (), {"pipeline": create_main_pipeline(source_path, domain, **context_kwargs)}
        elif "context" in kwargs and isinstance(kwargs["context"], dict):
            context_dict = kwargs.pop("context")
            source_path = context_dict.get("source_path", "")
            domain = context_dict.get("domain", "AUTO")
            context_kwargs = {k: v for k, v in context_dict.items() if k not in ["source_path", "domain"]}
            
            from .main_pipeline import create_main_pipeline
            return (), {"pipeline": create_main_pipeline(source_path, domain, **context_kwargs)}
    
    return args, kwargs


def get_pipeline(
    route: PipelineRoute,
    *args,
    **kwargs,
) -> Any:
    """获取指定路由的 Pipeline 实例
    
    Args:
        route: PipelineRoute 枚举值
        *args: 传递给 Pipeline 构造函数的位置参数
        **kwargs: 传递给 Pipeline 构造函数的关键字参数
    
    Returns:
        Pipeline 实例
    
    Raises:
        ValueError: 如果路由不存在或已禁用
        ImportError: 如果无法导入 Pipeline 模块
    """
    info = _PIPELINE_REGISTRY.get(route)
    if not info:
        raise ValueError(f"Unknown pipeline route: {route}")
    
    if not info.enabled:
        raise ValueError(f"Pipeline '{route.value}' is disabled")
    
    if info.deprecated:
        logger.warning(f"DEPRECATED: Pipeline '{route.value}' ({info.name}) is deprecated: {info.deprecation_message}")
    
    args, kwargs = _adapt_context(route, args, kwargs)
    
    if "pipeline" in kwargs:
        return kwargs["pipeline"]
    
    try:
        module = __import__(info.module_path, fromlist=[info.class_name])
        pipeline_class = getattr(module, info.class_name)
        return pipeline_class(*args, **kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import pipeline '{info.name}' from '{info.module_path}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Pipeline class '{info.class_name}' not found in '{info.module_path}': {e}")


def get_production_pipeline(*args, **kwargs) -> Any:
    """获取生产主线 Pipeline（快捷函数）"""
    return get_pipeline(PipelineRoute.MAIN, *args, **kwargs)


def is_production_route(route: PipelineRoute) -> bool:
    """检查是否为生产路由"""
    info = _PIPELINE_REGISTRY.get(route)
    return info is not None and info.pipeline_type == PipelineType.PRODUCTION


def get_pipeline_stats() -> Dict[str, Any]:
    """获取 Pipeline 统计信息"""
    stats = {
        "total": len(_PIPELINE_REGISTRY),
        "production": sum(1 for info in _PIPELINE_REGISTRY.values() if info.pipeline_type == PipelineType.PRODUCTION),
        "migration": sum(1 for info in _PIPELINE_REGISTRY.values() if info.pipeline_type == PipelineType.MIGRATION),
        "experimental": sum(1 for info in _PIPELINE_REGISTRY.values() if info.pipeline_type == PipelineType.EXPERIMENTAL),
        "enabled": sum(1 for info in _PIPELINE_REGISTRY.values() if info.enabled),
        "deprecated": sum(1 for info in _PIPELINE_REGISTRY.values() if info.deprecated),
    }
    return stats


def print_pipeline_help():
    """打印 Pipeline 路由帮助信息"""
    print("\n" + "=" * 60)
    print("📦 Pipeline Router - 统一入口路由")
    print("=" * 60)
    
    print("\n可用路由：")
    print("-" * 40)
    
    for route, info in _PIPELINE_REGISTRY.items():
        status_icon = "✅" if info.enabled else "❌"
        type_label = {
            PipelineType.PRODUCTION: "生产主线",
            PipelineType.MIGRATION: "迁移兼容",
            PipelineType.EXPERIMENTAL: "实验/停用",
        }[info.pipeline_type]
        
        deprecated_label = " ⚠️ 停用" if info.deprecated else ""
        
        print(f"\n{status_icon} {route.value.upper():10s} [{type_label}] {deprecated_label}")
        print(f"     名称: {info.name}")
        print(f"     描述: {info.description}")
        
        if info.deprecated:
            print(f"     提示: {info.deprecation_message}")
    
    print("\n" + "=" * 60)
    print("使用建议：")
    print("  - 生产环境：仅使用 MAIN 路由")
    print("  - 其余路由：仅保留历史兼容，不作为新入口开放")
    print("=" * 60)
    print("=" * 60)