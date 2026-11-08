"""Pipeline construction helpers extracted from main_pipeline (P0-3 batch 13).

Domain auto-detection + factory/convenience constructors, moved verbatim from
module-level functions in main_pipeline using the plain module pattern
(they never touch pipeline instance state).

Contents:
- :func:`auto_detect_domain` — from ``_auto_detect_domain``
- :func:`match_folder_keywords` / :func:`match_path_keywords` — helpers
- :func:`create_main_pipeline` — from ``create_main_pipeline``
- :func:`run_main_pipeline` — from ``run_main_pipeline`` (async)

Compatibility: main_pipeline keeps ``_auto_detect_domain``,
``create_main_pipeline`` and ``run_main_pipeline`` as thin re-export aliases
(the router and ``src.pipeline.__init__`` import the latter two by name).
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def auto_detect_domain(source_path: str) -> str:
    """Auto-detect business domain from source path and folder structure.

    Detection rules (in priority order):
    1. Folder name keywords (PPM, SWDL, Basic, CHIMAS, etc.)
    2. Path segment keywords (supply, manufacturing, quality, etc.)
    3. Default: SUPPLY_CHAIN
    """
    path_str = str(source_path).lower().replace("\\", "/")
    parts = path_str.split("/")

    folder_match = match_folder_keywords(parts)
    if folder_match:
        return folder_match
    path_match = match_path_keywords(path_str)
    if path_match:
        return path_match
    return "SUPPLY_CHAIN"


def match_folder_keywords(parts: list[str]) -> Optional[str]:
    """Match folder-name keywords to a business domain."""
    folder_keywords = {
        "ppm": "PRODUCT_PROCESS_MANUFACTURING",
        "swdl": "SOFTWARE_DEVELOPMENT_LIFECYCLE",
        "chimas": "MANUFACTURING_SYSTEM",
        "basic": "SUPPLY_CHAIN",
    }
    for part in parts:
        for keyword, domain in folder_keywords.items():
            if keyword in part:
                return domain
    return None


def match_path_keywords(path_str: str) -> Optional[str]:
    """Match path-segment keywords to a business domain."""
    path_keywords = {
        "supply": "SUPPLY_CHAIN",
        "procurement": "SUPPLY_CHAIN",
        "logistics": "SUPPLY_CHAIN",
        "inventory": "SUPPLY_CHAIN",
        "manufacturing": "MANUFACTURING",
        "production": "MANUFACTURING",
        "quality": "QUALITY_MANAGEMENT",
        "engineering": "PRODUCT_ENGINEERING",
        "design": "PRODUCT_ENGINEERING",
        "bom": "PRODUCT_ENGINEERING",
        "sales": "SALES_ORDER",
        "order": "SALES_ORDER",
        "finance": "FINANCE",
        "it": "IT_INFRASTRUCTURE",
        "system": "IT_INFRASTRUCTURE",
    }
    for keyword, domain in path_keywords.items():
        if keyword in path_str:
            return domain
    return None


def create_main_pipeline(source_path: str, domain: str = "AUTO", **kwargs):
    """Create a MainPipeline instance (factory function).

    Args:
        source_path: source file or folder path
        domain: domain type (AUTO for auto-detection)
        **kwargs: PipelineContext parameters
    """
    from .main_pipeline import MainPipeline, PipelineContext

    # Auto-detect domain if requested
    if domain and domain.upper() == "AUTO":
        domain = auto_detect_domain(source_path)
        logger.info(f"[DOMAIN] Auto-detected: {domain} from path: {source_path}")

    context = PipelineContext(
        source_path=source_path,
        domain=domain,
        **kwargs,
    )
    logger.info(
        "[DEBUG-PATH] PipelineContext created source_path=%s domain=%s output_dir=%s run_id=%s",
        context.source_path,
        context.domain,
        context.output_dir,
        context.run_id,
    )
    return MainPipeline(context)


async def run_main_pipeline(source_path: str, domain: str = "SUPPLY_CHAIN", **kwargs):
    """Run the main pipeline (convenience function)."""
    logger.info(
        "[DEBUG-PATH] run_main_pipeline input source_path=%s domain=%s kwargs=%s",
        source_path,
        domain,
        {k: kwargs.get(k) for k in sorted(kwargs.keys())},
    )
    pipeline = create_main_pipeline(source_path, domain, **kwargs)
    result = await pipeline.run()
    pipeline.save_result(result)
    return result


# Historical private alias kept so any introspection keeps working.
_auto_detect_domain = auto_detect_domain
