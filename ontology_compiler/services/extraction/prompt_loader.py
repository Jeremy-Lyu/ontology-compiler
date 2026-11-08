"""Prompt Template Loader - Load and manage prompt templates from configuration files.

[PIPELINE-IMPROVEMENT]: Decouple prompt templates from code, making them configurable.

This module provides:
1. PromptTemplateLoader: Load templates from JSON/YAML config files
2. Template caching to avoid repeated I/O
3. Variable interpolation with defaults
4. Fallback to hardcoded templates when config files are missing
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from functools import lru_cache


class PromptTemplateLoader:
    """Load and manage prompt templates from configuration files.
    
    Usage:
        loader = PromptTemplateLoader()
        template = loader.load("technical")
        prompt = template.render(text="...")
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize loader with optional config directory.
        
        Args:
            config_dir: Path to prompt config directory. 
                Defaults to config/prompts/ relative to project root.
        """
        self.config_dir = Path(config_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config", "prompts"
        ))
        self._templates: Dict[str, _PromptTemplate] = {}
        self._fragments: Dict[str, Dict[str, _PromptTemplate]] = {}
        self._load_all_templates()
    
    def _load_all_templates(self) -> None:
        """Load all template files from config directory."""
        if not self.config_dir.exists():
            return
        
        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    templates = data.get("templates", {})
                    for name, template_data in templates.items():
                        self._templates[name] = _PromptTemplate(template_data)
                    fragments = data.get("fragments", {})
                    if fragments:
                        for fragment_name, fragment_group in fragments.items():
                            if isinstance(fragment_group, dict) and "template" in fragment_group:
                                self._templates[fragment_name] = _PromptTemplate(fragment_group)
                            elif isinstance(fragment_group, dict):
                                self._fragments[fragment_name] = {
                                    sub_name: _PromptTemplate(sub_data)
                                    for sub_name, sub_data in fragment_group.items()
                                    if isinstance(sub_data, dict) and "template" in sub_data
                                }
                                for sub_name, sub_data in fragment_group.items():
                                    if isinstance(sub_data, dict) and "template" in sub_data:
                                        self._templates[sub_name] = _PromptTemplate(sub_data)
                    legacy_templates = data.get("legacy_templates", {})
                    for name, template_data in legacy_templates.items():
                        if name not in self._templates:
                            self._templates[name] = _PromptTemplate(template_data)
            except Exception:
                pass
    
    def load(self, name: str) -> _PromptTemplate:
        """Load a template by name.
        
        Args:
            name: Template name (technical, business, contract, general)
            
        Returns:
            _PromptTemplate instance that can be rendered with variables
            
        Note:
            If template is not found in config, returns None.
            Caller should fall back to hardcoded templates.
        """
        return self._templates.get(name)
    
    def available_templates(self) -> list[str]:
        """Return list of available template names."""
        return list(self._templates.keys())
    
    def has_template(self, name: str) -> bool:
        """Check if a template exists."""
        return name in self._templates


class _PromptTemplate:
    """Internal class representing a single prompt template."""
    
    def __init__(self, template_data: Dict[str, Any]):
        self.id = template_data.get("id", "")
        self.version = template_data.get("version", "1.0")
        self.description = template_data.get("description", "")
        self.variables = template_data.get("variables", [])
        self.defaults = template_data.get("defaults", {})
        self.template_text = template_data.get("template", "")
    
    def render(self, **kwargs: Any) -> str:
        """Render the template with given variables.
        
        Args:
            **kwargs: Variables to substitute in the template
            
        Returns:
            Rendered prompt string
        """
        context = {**self.defaults, **kwargs}
        
        result = self.template_text
        for key, value in context.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))
        
        return result


class CachedPromptLoader:
    """Singleton cached prompt loader with fallback support."""
    
    _instance: Optional[CachedPromptLoader] = None
    
    def __new__(cls) -> CachedPromptLoader:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loader = PromptTemplateLoader()
        return cls._instance
    
    def get_template(self, name: str) -> Optional[_PromptTemplate]:
        """Get template from cache or config."""
        return self._loader.load(name)
    
    def render(self, name: str, **kwargs: Any) -> Optional[str]:
        """Render template if available."""
        template = self.get_template(name)
        if template:
            return template.render(**kwargs)
        return None

    def render_composed(
        self,
        base: Optional[str] = None,
        domain: Optional[str] = None,
        task: Optional[str] = None,
        schema_hint: Optional[str] = None,
        document: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Render a composed prompt from base, domain, and task fragments."""
        parts = []
        for fragment in (base, domain, task):
            if fragment:
                if hasattr(self, "_loader") and getattr(self._loader, "_templates", None) and fragment in self._loader._templates:
                    rendered = self.render(fragment, **kwargs)
                    if rendered:
                        parts.append(rendered)
                else:
                    parts.append(str(fragment))
        if schema_hint:
            parts.append(str(schema_hint))
        if document:
            parts.append(str(document))
        if parts:
            return "\n\n".join(parts)
        raise ValueError("No prompt fragments could be rendered")


@lru_cache(maxsize=16)
def get_prompt_loader() -> CachedPromptLoader:
    """Get singleton prompt loader with caching."""
    return CachedPromptLoader()


def load_and_render_prompt(
    name: str,
    fallback_fn: Optional[Callable[..., str]] = None,
    **kwargs: Any
) -> str:
    """Load and render prompt with fallback to hardcoded function.
    
    Args:
        name: Template name
        fallback_fn: Callable that returns hardcoded prompt if config template not found
        **kwargs: Variables for template interpolation
        
    Returns:
        Rendered prompt string (from config or fallback)
    """
    loader = get_prompt_loader()
    rendered = loader.render(name, **kwargs)
    
    if rendered is not None:
        return rendered
    
    if fallback_fn is not None:
        return fallback_fn(**kwargs)
    
    raise ValueError(f"Prompt template '{name}' not found and no fallback provided")