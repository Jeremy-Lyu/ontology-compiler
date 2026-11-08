"""Report template configuration loader (P1-7 Refactoring).

Loads static text from YAML configuration to separate presentation from logic.
Uses string.Template for lightweight templating (standard library only).
"""

from __future__ import annotations

import logging
from pathlib import Path
from string import Template
from typing import Any, Dict, List

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

# Template file path
TEMPLATE_PATH = Path("config/report_templates.yaml")

# In-memory cache for loaded templates
_templates_cache: Dict[str, Any] | None = None


def load_templates(force_reload: bool = False) -> Dict[str, Any]:
    """Load report templates from YAML configuration.

    Args:
        force_reload: If True, ignore cache and reload from file.

    Returns:
        Dictionary containing template configuration.

    Raises:
        FileNotFoundError: If template file doesn't exist.
        RuntimeError: If YAML parsing fails.
    """
    global _templates_cache

    if _templates_cache is not None and not force_reload:
        return _templates_cache

    if yaml is None:
        raise RuntimeError("PyYAML is required for template loading")

    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template file not found: {TEMPLATE_PATH}")

    try:
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            _templates_cache = yaml.safe_load(f)
        logger.debug(f"Loaded templates from {TEMPLATE_PATH}")
        return _templates_cache
    except Exception as exc:
        raise RuntimeError(f"Failed to load templates: {exc}") from exc


def get_template(path: str, default: Any = None) -> Any:
    """Get a template by dot-notation path (e.g., 'core_results.header').

    Args:
        path: Dot-separated path to the template value.
        default: Default value if path not found.

    Returns:
        Template value or default.
    """
    try:
        templates = load_templates()
        keys = path.split(".")
        value = templates
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    except Exception:
        return default


def render_template(template_str: str, variables: Dict[str, Any]) -> str:
    """Render a string template with variables.

    Uses string.Template for safe substitution (standard library).
    Does NOT raise on missing variables - keeps placeholder.

    Args:
        template_str: Template string with $variable placeholders.
        variables: Dictionary of variable values.

    Returns:
        Rendered string.
    """
    try:
        return Template(template_str).safe_substitute(variables)
    except Exception as exc:
        logger.warning(f"Template rendering failed: {exc}")
        return template_str


def render_section(template_key: str, **vars: Any) -> Any:
    """Generic section renderer - P1-2 Parameterization.

    Retrieves template content by key and renders it with optional variables.

    Args:
        template_key: Dot-separated path to template (e.g., 'core_results.header')
        **vars: Variables to substitute in the template

    Returns:
        Rendered content (string) or raw template data (list/dict)

    Examples:
        >>> render_section('core_results.header', run_id='ONT-20260615-001')
        '# CORE RESULTS - Run: ONT-20260615-001'

        >>> render_section('core_results.guard_check.steps')
        [{'title': 'Step 1', 'description': '...'}, ...]
    """
    template = get_template(template_key)
    
    if template is None:
        logger.debug(f"Template not found: {template_key}")
        return None
    
    # If it's a string, render with variables
    if isinstance(template, str):
        return render_template(template, vars)
    
    # If it's a list or dict, return as-is (backward compatibility)
    return template


def render_table_row(cells: List[str]) -> str:
    """Render a markdown table row.

    Args:
        cells: List of cell values.

    Returns:
        Markdown table row string.
    """
    return "| " + " | ".join(cells) + " |"


def format_meta_value(value: Any, format_type: str | None = None) -> str:
    """Format a metadata value for display.

    Args:
        value: The value to format.
        format_type: Optional format hint ('code', 'count', etc.).

    Returns:
        Formatted string.
    """
    if value is None:
        return "—"

    if format_type == "code":
        return f"`{value}`"

    if format_type == "count":
        return str(len(value) if isinstance(value, (list, dict, set)) else value)

    return str(value)


def format_list_item(item: str, prefix: str = "-") -> str:
    """Format a list item.

    Args:
        item: Item text.
        prefix: List prefix (default: '-').

    Returns:
        Formatted list item.
    """
    lines = item.split("\n")
    if len(lines) == 1:
        return f"{prefix} {item}"

    # Multi-line item: first line with prefix, rest indented
    result = [f"{prefix} {lines[0]}"]
    result.extend(f"  {line}" for line in lines[1:])
    return "\n".join(result)


def get_core_results_header(**vars: Any) -> str | None:
    """Get core results header template."""
    return render_section("core_results.header", **vars)


def get_checklist_items() -> List[Dict[str, Any]] | None:
    """Get checklist items template."""
    return render_section("core_results.checklist.items")


def get_guard_check_steps() -> List[Dict[str, Any]] | None:
    """Get guard check steps template."""
    return render_section("core_results.guard_check.steps")


def get_entity_tables() -> Dict[str, Any] | None:
    """Get entity tables configuration."""
    return render_section("core_results.entity_tables")


def get_business_scenarios() -> Dict[str, Any] | None:
    """Get business scenarios configuration."""
    return render_section("core_results.business_scenarios")


def get_data_analysis_header(**vars: Any) -> str | None:
    """Get data analysis header template."""
    return render_section("data_analysis.header", **vars)


def get_data_analysis_sections() -> List[Dict[str, Any]] | None:
    """Get data analysis sections configuration."""
    return render_section("data_analysis.sections")


def get_data_analysis_footer(**vars: Any) -> str | None:
    """Get data analysis footer template."""
    return render_section("data_analysis.footer", **vars)


def get_multi_agent_analysis_header(**vars: Any) -> str | None:
    """Get multi-agent analysis header template."""
    return render_section("multi_agent_analysis.header", **vars)


def get_multi_agent_analysis_footer(**vars: Any) -> str | None:
    """Get multi-agent analysis footer template."""
    return render_section("multi_agent_analysis.footer", **vars)


def get_multi_agent_analysis_sections() -> List[Dict[str, Any]] | None:
    """Get multi-agent analysis sections configuration."""
    return render_section("multi_agent_analysis.sections")


def get_multi_agent_analysis_subtitle(**vars: Any) -> str | None:
    """Get multi-agent analysis subtitle template."""
    return render_section("multi_agent_analysis.subtitle", **vars)


def get_ontology_health_sections() -> List[Dict[str, Any]] | None:
    """Get ontology health sections configuration."""
    return render_section("ontology_health.sections")


def get_publish_steps() -> List[Dict[str, Any]] | None:
    """Get publish steps configuration."""
    return render_section("core_results.publish.steps")


def get_risk_matrix() -> Dict[str, Any] | None:
    """Get risk matrix configuration."""
    return render_section("core_results.risk_matrix")


def get_section_title(key: str, **vars: Any) -> str | None:
    """Get section title by key."""
    return render_section(f"section_titles.{key}", **vars)
