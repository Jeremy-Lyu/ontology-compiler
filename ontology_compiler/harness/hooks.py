from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ontology_compiler.config import load_runtime_settings
from core.contracts.ports import HookRunnerPort, HookResult


# Whitelist of safe command prefixes when shell execution is absolutely required
ALLOWED_SHELL_COMMANDS = {
    "python",
    "python3",
    "node",
    "npm",
}


class HookRunnerAdapter(HookRunnerPort):
    """Adapter wrapping the legacy HookRunner to fulfill HookRunnerPort."""

    def __init__(self, runner: HookRunner | None = None):
        self._runner = runner or HookRunner()

    def run(self, event: str, payload: dict[str, Any]) -> HookResult:
        legacy_result: HookRunResult = self._runner.run(event, payload)
        return HookResult(
            allowed=legacy_result.allowed,
            exit_code=legacy_result.exit_code,
            reason=legacy_result.reason,
            output=legacy_result.output,
        )


@dataclass
class HookRunResult:
    allowed: bool
    exit_code: int
    reason: str
    output: str


class HookRunner:
    """Minimal pre/post hook runner with deny semantics."""

    def __init__(self, settings: dict[str, Any] | None = None):
        self.settings = settings or load_runtime_settings()
        hooks = self.settings.get("hooks") if isinstance(self.settings, dict) else {}
        self.enabled = bool((hooks or {}).get("enabled", False))
        self.deny_exit_code = int((hooks or {}).get("deny_exit_code", 2))
        self.timeout_seconds = int((hooks or {}).get("timeout_seconds", 10))
        self.commands = list((hooks or {}).get("commands") or [])
        self.event_commands = dict((hooks or {}).get("event_commands") or {})
        self.execution_mode = str((hooks or {}).get("execution_mode", "exec"))
        self.allow_shell = bool((hooks or {}).get("allow_shell", False))

    def set_event_commands(self, event: str, commands: list[Any]) -> None:
        merged = list(self.event_commands.get(event) or [])
        merged.extend(commands)
        self.event_commands[event] = merged

    def commands_for_event(self, event: str) -> list[Any]:
        commands = list(self.commands)
        commands.extend(list(self.event_commands.get(event) or []))
        return commands

    def run(self, event: str, payload: dict[str, Any]) -> HookRunResult:
        commands = self.commands_for_event(event)
        if not self.enabled or not commands:
            return HookRunResult(True, 0, "hooks disabled", "")

        merged_payload = {
            "event": event,
            **payload,
        }

        for command in commands:
            result = self._run_one(command, merged_payload)
            if not result.allowed:
                return result

        return HookRunResult(True, 0, "ok", "")

    def _resolve_command(self, command: Any) -> tuple[list[str], bool]:
        """Resolve command to list format and determine if shell is needed.
        
        Security: shell=True is only allowed for whitelisted command prefixes.
        All other commands are forced to use shell=False with list format.
        """
        if isinstance(command, list):
            return [str(x) for x in command], False

        text = str(command)
        # Try to parse as list first
        try:
            parsed = shlex.split(text)
            return parsed, False
        except ValueError:
            # If shlex fails, check if shell execution is allowed and whitelisted
            if not self.allow_shell and self.execution_mode != "shell":
                raise ValueError(
                    f"Command parsing failed and shell execution is not allowed: {text[:100]}"
                )
            
            # Check if command starts with a whitelisted prefix
            cmd_prefix = text.strip().split()[0] if text.strip() else ""
            if cmd_prefix not in ALLOWED_SHELL_COMMANDS:
                raise ValueError(
                    f"Shell execution not allowed for command '{cmd_prefix}'. "
                    f"Allowed prefixes: {sorted(ALLOWED_SHELL_COMMANDS)}"
                )
            return text, True

    def _run_one(self, command: Any, payload: dict[str, Any]) -> HookRunResult:
        env = os.environ.copy()
        env["ONTOLOGY_HOOK_PAYLOAD"] = json.dumps(payload, ensure_ascii=False)

        cmd, use_shell = self._resolve_command(command)
        try:
            proc = subprocess.run(
                cmd,
                shell=use_shell,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                env=env,
            )
        except Exception as exc:
            return HookRunResult(False, 1, "error", str(exc))

        output = ((proc.stdout or "") + (proc.stderr or "")).strip()
        if proc.returncode == self.deny_exit_code:
            return HookRunResult(False, proc.returncode, "deny", output)
        if proc.returncode != 0:
            return HookRunResult(False, proc.returncode, "error", output)
        return HookRunResult(True, 0, "ok", output)


def build_hook_payload(
    *,
    stage: str,
    tool: str,
    event: str,
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
    is_error: bool = False,
) -> dict[str, Any]:
    return {
        "ts": datetime.now().isoformat(),
        "stage": stage,
        "tool": tool,
        "event": event,
        "input": input_data or {},
        "output": output_data or {},
        "is_error": is_error,
    }
