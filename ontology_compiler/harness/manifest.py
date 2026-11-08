from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any


class ManifestValidationError(ValueError):
    pass


_ALLOWED_HOOK_EVENTS = {"PreToolUse", "PostToolUse"}
_ALLOWED_SKILL_CATEGORIES = {"workflow", "ontology", "pipeline_stage", "rag", "ops", "catalog", "maintenance", "validation", "pipeline"}


def _require_bool(payload: dict[str, Any], key: str, *, manifest_label: str, default: bool = True) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ManifestValidationError(f"{manifest_label} field '{key}' must be boolean")
    return value


def _require_string_list(value: Any, *, field_name: str, manifest_label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ManifestValidationError(f"{manifest_label} field '{field_name}' must be list")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ManifestValidationError(f"{manifest_label} has invalid {field_name} entries")
        normalized.append(item.strip())
    return normalized


def _as_schema_dict(value: Any) -> dict[str, Any]:
    """P1-2: output_schema 非 dict 时按空处理（不阻断发现；evaluator 见 skills/evaluator.py）。"""
    return value if isinstance(value, dict) else {}


def _validate_args_schema(args_schema: Any, *, manifest_label: str) -> dict[str, Any]:
    if args_schema is None:
        return {}
    if not isinstance(args_schema, dict):
        raise ManifestValidationError(f"{manifest_label} has invalid args_schema")

    normalized: dict[str, Any] = {}
    for arg_name, rule in args_schema.items():
        if not isinstance(arg_name, str) or not arg_name.strip() or not isinstance(rule, dict):
            raise ManifestValidationError(f"{manifest_label} has invalid args_schema entries")

        arg_label = f"{manifest_label} arg '{arg_name}'"
        normalized_rule = dict(rule)
        if "required" in normalized_rule and not isinstance(normalized_rule["required"], bool):
            raise ManifestValidationError(f"{arg_label} field 'required' must be boolean")
        if "type" in normalized_rule and (
            not isinstance(normalized_rule["type"], str) or not normalized_rule["type"].strip()
        ):
            raise ManifestValidationError(f"{arg_label} field 'type' must be non-empty string")
        if "enum" in normalized_rule:
            enum_values = normalized_rule["enum"]
            if not isinstance(enum_values, list) or any(not isinstance(x, str) or not x.strip() for x in enum_values):
                raise ManifestValidationError(f"{arg_label} field 'enum' must be list of non-empty strings")
            normalized_rule["enum"] = [x.strip() for x in enum_values]
        normalized[arg_name.strip()] = normalized_rule

    return normalized


@dataclass
class PluginManifest:
    name: str
    version: str
    enabled: bool
    hooks: list[str]
    permissions: list[str]
    hook_commands: list[str]
    event_hook_commands: dict[str, list[str]]

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "PluginManifest":
        name = str(payload.get("name") or "").strip()
        version = str(payload.get("version") or "").strip()

        if not name:
            raise ManifestValidationError("plugin manifest missing required field: name")
        if not version:
            raise ManifestValidationError(f"plugin manifest '{name}' missing required field: version")

        manifest_label = f"plugin manifest '{name}'"
        hooks = _require_string_list(payload.get("hooks"), field_name="hooks", manifest_label=manifest_label)
        permissions = _require_string_list(
            payload.get("permissions"), field_name="permissions", manifest_label=manifest_label
        )
        hook_commands = _require_string_list(
            payload.get("hook_commands"), field_name="hook_commands", manifest_label=manifest_label
        )
        enabled = _require_bool(payload, "enabled", manifest_label=manifest_label, default=True)

        unknown_hooks = [hook for hook in hooks if hook not in _ALLOWED_HOOK_EVENTS]
        if unknown_hooks:
            raise ManifestValidationError(
                f"{manifest_label} has unsupported hooks: {', '.join(sorted(unknown_hooks))}"
            )

        raw_event_hook_commands = payload.get("event_hook_commands") or {}
        if not isinstance(raw_event_hook_commands, dict):
            raise ManifestValidationError(f"{manifest_label} field 'event_hook_commands' must be object")

        event_hook_commands: dict[str, list[str]] = {}
        for event, commands in raw_event_hook_commands.items():
            if not isinstance(event, str) or not event.strip():
                raise ManifestValidationError(f"{manifest_label} has invalid event_hook_commands key")
            event_name = event.strip()
            if event_name not in _ALLOWED_HOOK_EVENTS:
                raise ManifestValidationError(f"{manifest_label} has unsupported event_hook_commands key: {event_name}")
            event_hook_commands[event_name] = _require_string_list(
                commands,
                field_name=f"event_hook_commands.{event_name}",
                manifest_label=manifest_label,
                )

        return cls(
            name=name,
            version=version,
            enabled=enabled,
            hooks=hooks,
            permissions=permissions,
            hook_commands=hook_commands,
            event_hook_commands=event_hook_commands,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkillManifest:
    name: str
    description: str
    command: str
    script_path: str
    category: str
    enabled: bool
    args_schema: dict[str, Any]
    stages: list[str]
    version: str = "1.0"
    domains: list[str] = field(default_factory=list)
    layers: list[str] = field(default_factory=list)
    supported_formats: list[str] = field(default_factory=list)
    skill_tags: list[str] = field(default_factory=list)
    sub_skills: list[dict[str, Any]] = field(default_factory=list)
    # P1-2 (2026-10-16): skill 输出契约，供 schema 驱动 evaluator 使用。
    output_schema: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "SkillManifest":
        name = str(payload.get("name") or "").strip()
        description = str(payload.get("description") or "").strip()
        command = str(payload.get("command") or "").strip()
        script_path = str(payload.get("script_path") or "").strip()
        raw_category = payload.get("category")
        category = str(raw_category or "workflow").strip() or "workflow"
        version = str(payload.get("version") or "1.0").strip()

        scripts = payload.get("scripts", {})
        if not command and scripts.get("main"):
            command = f"python {scripts['main']}"
        if not script_path and scripts.get("main"):
            script_path = scripts["main"]

        if not name:
            raise ManifestValidationError("skill manifest missing required field: name")
        if not description:
            raise ManifestValidationError(f"skill manifest '{name}' missing required field: description")
        if not command:
            raise ManifestValidationError(f"skill manifest '{name}' missing required field: command")
        if not script_path:
            raise ManifestValidationError(f"skill manifest '{name}' missing required field: script_path")
        if category not in _ALLOWED_SKILL_CATEGORIES:
            raise ManifestValidationError(f"skill manifest '{name}' has unsupported category: {category}")

        manifest_label = f"skill manifest '{name}'"
        enabled = _require_bool(payload, "enabled", manifest_label=manifest_label, default=True)
        args_schema = _validate_args_schema(payload.get("args_schema"), manifest_label=manifest_label)
        stages = _require_string_list(payload.get("stages"), field_name="stages", manifest_label=manifest_label)
        domains = _require_string_list(payload.get("domains"), field_name="domains", manifest_label=manifest_label)
        layers = _require_string_list(payload.get("layers"), field_name="layers", manifest_label=manifest_label)
        supported_formats = _require_string_list(payload.get("supported_formats"), field_name="supported_formats", manifest_label=manifest_label)
        skill_tags = _require_string_list(payload.get("tags"), field_name="tags", manifest_label=manifest_label)

        if stages and raw_category is None:
            category = "pipeline_stage"
        if stages and category != "pipeline_stage":
            raise ManifestValidationError(
                f"skill manifest '{name}' stages require category 'pipeline_stage'"
            )

        return cls(
            name=name,
            description=description,
            command=command,
            script_path=script_path,
            category=category,
            enabled=enabled,
            args_schema=args_schema,
            stages=stages,
            version=version,
            domains=domains,
            layers=layers,
            supported_formats=supported_formats,
            skill_tags=skill_tags,
            sub_skills=payload.get("sub_skills", []),
            output_schema=_as_schema_dict(payload.get("output_schema")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_manifest_json(path: Path, manifest_type: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ManifestValidationError(f"{manifest_type} manifest must be object: {path}")
    return payload


def load_plugin_manifest(path: Path) -> PluginManifest:
    return PluginManifest.from_json(_load_manifest_json(path, "plugin"))


def load_skill_manifest(path: Path) -> SkillManifest:
    return SkillManifest.from_json(_load_manifest_json(path, "skill"))
