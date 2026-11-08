"""HookRunner Adapter - independent implementation for HookRunnerPort.

This adapter was migrated from ontology_compiler.harness.hooks to maintain
clean architectural boundaries between Application and Legacy layers.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any

from src.application.ports.observability_ports import HookRunnerPort, HookResult


# Whitelist of safe command prefixes when shell execution is absolutely required
ALLOWED_SHELL_COMMANDS = {
    "python",
    "python3",
    "node",
    "npm",
}


@dataclass
class HookRunResult:
    """Result of running a hook command."""
    allowed: bool
    exit_code: int
    reason: str
    output: str


class HookRunner:
    """Minimal pre/post hook runner with deny semantics."""

    def __init__(self, settings: dict[str, Any] | None = None):
        self.settings = settings or self._load_default_settings()
        hooks = self.settings.get("hooks") if isinstance(self.settings, dict) else {}
        self.enabled = bool((hooks or {}).get("enabled", False))
        self.deny_exit_code = int((hooks or {}).get("deny_exit_code", 2))
        self.timeout_seconds = int((hooks or {}).get("timeout_seconds", 10))
        self.commands = list((hooks or {}).get("commands") or [])
        self.event_commands = dict((hooks or {}).get("event_commands") or {})
        self.execution_mode = str((hooks or {}).get("execution_mode", "exec"))
        self.allow_shell = bool((hooks or {}).get("allow_shell", False))

    def _load_default_settings(self) -> dict[str, Any]:
        """Load default runtime settings."""
        return {
            "hooks": {
                "enabled": os.environ.get("HOOKS_ENABLED", "false").lower() == "true",
                "deny_exit_code": 2,
                "timeout_seconds": 10,
                "commands": [],
                "event_commands": {},
                "execution_mode": "exec",
                "allow_shell": False,
            }
        }

    def set_event_commands(self, event: str, commands: list[Any]) -> None:
        """Set commands for a specific event."""
        merged = list(self.event_commands.get(event) or [])
        merged.extend(commands)
        self.event_commands[event] = merged

    def commands_for_event(self, event: str) -> list[Any]:
        """Get all commands for an event (global + event-specific)."""
        commands = list(self.commands)
        commands.extend(list(self.event_commands.get(event) or []))
        return commands

    def run(self, event: str, payload: dict[str, Any]) -> HookRunResult:
        """Run hooks for the given event."""
        commands = self.commands_for_event(event)
        if not self.enabled or not commands:
            return HookRunResult(True, 0, "hooks disabled", "")

        merged_payload = {
            "event": event,
            **payload,
        }

        for cmd in commands:
            if isinstance(cmd, str):
                cmd = {"cmd": cmd}

            cmd_str = cmd.get("cmd", "")
            if not cmd_str:
                continue

            # Build command with payload as environment
            env = os.environ.copy()
            env["HOOK_EVENT"] = event
            env["HOOK_PAYLOAD"] = json.dumps(merged_payload, ensure_ascii=False)

            try:
                if self.execution_mode == "shell" and self.allow_shell:
                    # Security: validate against whitelist
                    cmd_parts = shlex.split(cmd_str)
                    if cmd_parts and cmd_parts[0] not in ALLOWED_SHELL_COMMANDS:
                        return HookRunResult(
                            False,
                            self.deny_exit_code,
                            f"shell command '{cmd_parts[0]}' not in whitelist",
                            "",
                        )
                    result = subprocess.run(
                        cmd_str,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_seconds,
                        env=env,
                    )
                else:
                    # exec mode (safer, no shell)
                    cmd_parts = shlex.split(cmd_str)
                    result = subprocess.run(
                        cmd_parts,
                        shell=False,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_seconds,
                        env=env,
                    )

                if result.returncode != 0:
                    return HookRunResult(
                        False,
                        result.returncode,
                        f"hook denied: {cmd_str}",
                        result.stdout + result.stderr,
                    )

            except subprocess.TimeoutExpired:
                return HookRunResult(
                    False,
                    self.deny_exit_code,
                    f"hook timeout: {cmd_str}",
                    "",
                )
            except Exception as e:
                return HookRunResult(
                    False,
                    self.deny_exit_code,
                    f"hook error: {e}",
                    "",
                )

        return HookRunResult(True, 0, "all hooks passed", "")


class HookRunnerAdapter(HookRunnerPort):
    """Adapter wrapping the HookRunner to fulfill HookRunnerPort.
    
    This is the independent implementation migrated from
    ontology_compiler.harness.hooks.
    """

    def __init__(self, runner: HookRunner | None = None):
        self._runner = runner or HookRunner()

    def run(self, event: str, payload: dict[str, Any]) -> HookResult:
        """Run hook and return standardized result."""
        legacy_result: HookRunResult = self._runner.run(event, payload)
        return HookResult(
            allowed=legacy_result.allowed,
            exit_code=legacy_result.exit_code,
            reason=legacy_result.reason,
            output=legacy_result.output,
        )
