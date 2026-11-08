"""Skill Registry Adapter - Implementation of SkillRegistryPort.

[BATCH 4 CONSOLIDATION 2026-06-29]:
- **Thin Adapter**: This is a pure port implementation, delegating to ontology_compiler.harness
- No business logic here - only interface translation between layers
- Bridge between Application layer port interface and infrastructure implementation

10.2-I: Adapter for the skill registry, bridging ontology_compiler/harness
with the Application layer port interface.
"""

from pathlib import Path
from typing import Any, Optional

from ontology_compiler.harness import build_default_skill_registry
from ontology_compiler.harness.skill_registry import SkillRegistry, SkillDefinition

from src.application.ports.skill_ports import (
    SkillRegistryPort,
    SkillManifest,
    SkillExecutionResult,
)


class SkillRegistryAdapter(SkillRegistryPort):
    """Adapter wrapping ontology_compiler SkillRegistry.
    
    Provides a clean Application layer interface while internally
    delegating to the existing ontology_compiler implementation.
    """

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self._registry: SkillRegistry = build_default_skill_registry(project_root)
        self._project_root = project_root or Path.cwd()

    def _to_manifest(self, definition: SkillDefinition) -> SkillManifest:
        """Convert internal SkillDefinition to public SkillManifest."""
        return SkillManifest(
            name=definition.name,
            description=definition.description,
            category=definition.category,
            enabled=definition.enabled,
            script_path=definition.script_path,
            args_schema=definition.args_schema or {},
        )

    def discover_skills(self, skill_root: str) -> list[SkillManifest]:
        """Discover skills from the given root."""
        definitions = self._registry.discover(
            Path(skill_root),
            project_root=self._project_root
        )
        return [self._to_manifest(d) for d in definitions]

    def get_skill(self, name: str) -> Optional[SkillManifest]:
        """Get a specific skill by name."""
        definition = self._registry.get(name)
        if definition:
            return self._to_manifest(definition)
        return None

    def list_skills(
        self,
        category: Optional[str] = None,
        enabled_only: bool = True
    ) -> list[SkillManifest]:
        """List skills with optional filtering."""
        definitions = self._registry.list_skills(enabled_only=enabled_only)
        
        if category:
            definitions = [d for d in definitions if d.category == category]
        
        return [self._to_manifest(d) for d in definitions]

    def execute_skill(
        self,
        name: str,
        args: dict[str, Any],
        working_dir: Optional[str] = None
    ) -> SkillExecutionResult:
        """Execute a skill with given arguments."""
        import subprocess
        import time
        
        definition = self._registry.get(name)
        if not definition:
            return SkillExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Skill '{name}' not found",
                execution_time_ms=None,
            )
        
        if not definition.enabled:
            return SkillExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Skill '{name}' is disabled",
                execution_time_ms=None,
            )
        
        # Build command
        cmd = ["python", definition.script_path]
        
        # Add arguments if provided
        for key, value in args.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
        
        # Execute with timing
        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=working_dir or str(self._project_root),
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            elapsed_ms = int((time.time() - start) * 1000)
            
            return SkillExecutionResult(
                success=result.returncode == 0,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time_ms=elapsed_ms,
            )
        except subprocess.TimeoutExpired:
            elapsed_ms = int((time.time() - start) * 1000)
            return SkillExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr="Execution timed out (> 5 minutes)",
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            return SkillExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=str(e),
                execution_time_ms=elapsed_ms,
            )

    def validate_skill_config(self, name: str) -> tuple[bool, list[str]]:
        """Validate a skill's configuration."""
        definition = self._registry.get(name)
        if not definition:
            return False, [f"Skill '{name}' not found"]
        
        errors = []
        
        # Check script path exists
        script_path = Path(definition.script_path)
        if not script_path.exists():
            errors.append(f"Script not found: {definition.script_path}")
        
        # Check required fields
        if not definition.name:
            errors.append("Skill name is empty")
        if not definition.description:
            errors.append("Skill description is empty")
        if not definition.command:
            errors.append("Skill command is empty")
        
        return len(errors) == 0, errors

    def get_skill_categories(self) -> list[str]:
        """Get all available skill categories."""
        categories = set()
        for skill in self._registry.list_skills(enabled_only=False):
            categories.add(skill.category)
        return sorted(list(categories))

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        all_skills = self._registry.list_skills(enabled_only=False)
        enabled_skills = self._registry.list_skills(enabled_only=True)
        
        # Count by category
        by_category: dict[str, int] = {}
        for skill in all_skills:
            by_category[skill.category] = by_category.get(skill.category, 0) + 1
        
        return {
            "total": len(all_skills),
            "enabled": len(enabled_skills),
            "disabled": len(all_skills) - len(enabled_skills),
            "by_category": by_category,
            "discovery_errors": len(self._registry.get_discovery_errors()),
        }
