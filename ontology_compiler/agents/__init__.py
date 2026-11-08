"""Multi-Agent Framework for Ontology Compilation

[P1-6 2026-06-15]: 插件表扁平化 - 直接使用核心提取器类注册，移除包装器
"""

from .layered_builder import LayeredOntologyBuilder

# Plugin system
from .extractor_plugin import (
    ExtractorPlugin,
    ExtractorMetadata,
    ExtractorResult,
    ExtractorPluginRegistry,
    get_global_registry,
    register_plugin,
    get_plugin,
)

def register_builtin_plugins():
    """Register all built-in extractor plugins."""
    _register_builtin_plugins()


def _register_builtin_plugins():
    """Register all built-in extractor plugins."""
    plugins_registered = []
    
    # 1. TrueUnifiedExtractor (核心LLM提取器) - 直接注册，已实现 ExtractorPlugin 接口
    try:
        from ..services.extraction.true_unified_extractor import TrueUnifiedExtractor
        register_plugin(TrueUnifiedExtractor)
        plugins_registered.append("TrueUnifiedExtractor")
    except ImportError as e:
        logger = __import__('logging').getLogger(__name__)
        logger.debug(f"TrueUnifiedExtractor not available: {e}")
    
    # 2. StructuredDataExtractor (CSV/Excel提取器) - 直接注册，已实现 ExtractorPlugin 接口
    try:
        from ..services.extraction.structured_data_extractor import StructuredDataExtractor
        register_plugin(StructuredDataExtractor)
        plugins_registered.append("StructuredDataExtractor")
    except ImportError as e:
        logger = __import__('logging').getLogger(__name__)
        logger.debug(f"StructuredDataExtractor not available: {e}")
    
    # 3. CSVExtractorPlugin - 兼容层，保留 'csv_excel' 插件名称
    try:
        from .csv_extractor_plugin import CSVExtractorPlugin
        register_plugin(CSVExtractorPlugin)
        plugins_registered.append("CSVExtractorPlugin")
    except ImportError as e:
        logger = __import__('logging').getLogger(__name__)
        logger.debug(f"CSVExtractorPlugin not available: {e}")
    
    # 4. PDF 提取器
    try:
        from .pdf_extractor_plugin import PDFExtractorPlugin
        register_plugin(PDFExtractorPlugin)
        plugins_registered.append("PDFExtractor")
    except ImportError as e:
        logger = __import__('logging').getLogger(__name__)
        logger.debug(f"PDFExtractor not available: {e}")
    
    # 5. 文件解析器 (可选)  [原 #5 OfficeExtractor 已 DEPRECATED 并于 2026-08-06 删除，避免 metadata 参数 TypeError]
    try:
        from src.adapters.plugins import FileParserDocumentReaderPlugin
        register_plugin(FileParserDocumentReaderPlugin)
        plugins_registered.append("FileParser")
    except ImportError:
        pass
    
    if plugins_registered:
        logger = __import__('logging').getLogger(__name__)
        logger.info(f"Registered plugins: {', '.join(plugins_registered)}")


# Auto-register on module import
register_builtin_plugins()


__all__ = [
    # Extractors and builders
    'LayeredOntologyBuilder',
    # Plugin system
    'ExtractorPlugin',
    'ExtractorMetadata',
    'ExtractorResult',
    'ExtractorPluginRegistry',
    'get_global_registry',
    'register_plugin',
    'get_plugin',
    # Registration
    'register_builtin_plugins',
]
