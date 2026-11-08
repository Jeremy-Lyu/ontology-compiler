"""Skill/pipeline integration bridge.

Skill describes capability; pipeline describes orchestration.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class SkillInterface(ABC):
    """Skill 能力接口 - 只定义能力，不包含流程控制

    Skill 应该回答：
    1. 这个能力是什么？
    2. 输入是什么？
    3. 输出是什么？
    4. 适用范围是什么？
    5. 质量期望是什么？

    Skill 不应该包含：
    1. 复杂流程控制逻辑
    2. 运行时分支策略
    3. 重复的 schema 定义
    4. 仅属于 pipeline 的执行顺序
    """

    @abstractmethod
    def get_skill_id(self) -> str:
        """获取 skill ID"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """获取 skill 名称"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """获取 skill 描述"""
        pass

    @abstractmethod
    def get_purpose(self) -> str:
        """获取 skill 用途"""
        pass

    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """获取输入 schema"""
        pass

    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """获取输出 schema"""
        pass

    @abstractmethod
    def get_preconditions(self) -> List[str]:
        """获取前置条件"""
        pass

    @abstractmethod
    def get_postconditions(self) -> List[str]:
        """获取后置条件"""
        pass

    @abstractmethod
    def get_applicable_layers(self) -> List[str]:
        """获取适用层"""
        pass

    @abstractmethod
    def get_quality_expectations(self) -> Dict[str, Any]:
        """获取质量期望"""
        pass

    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行能力 - 纯函数式执行，无副作用"""
        pass


class PipelineStepInterface(ABC):
    """Pipeline 步骤接口 - 只负责流程编排，不定义能力

    Pipeline Step 应该回答：
    1. 步骤名称是什么？
    2. 依赖哪些前置步骤？
    3. 输入是什么？
    4. 输出是什么？
    5. 失败如何处理？

    Pipeline Step 不应该包含：
    1. 业务能力定义
    2. 复杂的领域逻辑
    3. 硬编码的策略判断
    """

    @abstractmethod
    def get_step_id(self) -> str:
        """获取步骤 ID"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """获取步骤名称"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """获取步骤描述"""
        pass

    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """获取依赖的前置步骤 ID"""
        pass

    @abstractmethod
    def get_input_keys(self) -> List[str]:
        """获取需要的输入键"""
        pass

    @abstractmethod
    def get_output_keys(self) -> List[str]:
        """获取产生的输出键"""
        pass

    @abstractmethod
    def can_skip(self) -> bool:
        """是否可跳过"""
        pass

    @abstractmethod
    def on_failure(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """失败处理"""
        pass

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行步骤"""
        pass


@dataclass
class SkillExecutionResult:
    """Skill 执行结果"""
    success: bool
    output: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    skill_id: str = ""


@dataclass
class StepExecutionResult:
    """步骤执行结果"""
    step_id: str
    success: bool
    output: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    skipped: bool = False


class SkillExecutor:
    """Skill 执行适配器 - 连接 skill 和 pipeline

    职责：
    1. 将 pipeline context 转换为 skill input
    2. 调用 skill.execute()
    3. 将 skill output 转换为 pipeline context
    4. 处理 skill 执行错误
    5. 记录执行指标

    这是 skill 和 pipeline 之间的唯一桥梁
    """

    def __init__(self, skill: SkillInterface):
        self._skill = skill
        self._logger = logging.getLogger(f"SkillExecutor.{skill.get_skill_id()}")

    def execute(self, context: Dict[str, Any]) -> SkillExecutionResult:
        """执行 skill 并适配上下文"""
        import time
        start_time = time.time()

        try:
            self._validate_preconditions(context)

            input_data = self._convert_context_to_input(context)

            self._logger.debug(f"Executing skill: {self._skill.get_skill_id()}")
            output_data = self._skill.execute(input_data)

            execution_time = time.time() - start_time

            self._validate_postconditions(output_data)

            return SkillExecutionResult(
                success=True,
                output=output_data,
                execution_time=execution_time,
                skill_id=self._skill.get_skill_id(),
            )
        except Exception as e:
            execution_time = time.time() - start_time
            self._logger.error(f"Skill execution failed: {e}")
            return SkillExecutionResult(
                success=False,
                errors=[str(e)],
                execution_time=execution_time,
                skill_id=self._skill.get_skill_id(),
            )

    def _convert_context_to_input(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """将 pipeline context 转换为 skill input"""
        input_schema = self._skill.get_input_schema()
        input_data = {}

        for key in input_schema.get("properties", {}).keys():
            if key in context:
                input_data[key] = context[key]

        # 2026-08-15 (C-1): when the schema declares no properties (common for
        # metadata-only skill manifests whose args_schema lives under
        # stage_data), pass the whole context through so skills can read what
        # they need (e.g. relation_rewrite reads stage_data.relationships).
        if not input_schema.get("properties") and context:
            input_data = dict(context)

        return input_data

    def _convert_output_to_context(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """将 skill output 转换为 pipeline context"""
        return output_data

    def _validate_preconditions(self, context: Dict[str, Any]) -> None:
        """验证前置条件"""
        for condition in self._skill.get_preconditions():
            if "exists" in condition.lower():
                field = condition.split(":")[-1].strip()
                if field not in context or context[field] is None:
                    raise ValueError(f"Precondition failed: {condition}")

    def _validate_postconditions(self, output: Dict[str, Any]) -> None:
        """验证后置条件"""
        for condition in self._skill.get_postconditions():
            if "contains" in condition.lower():
                parts = condition.split(":")
                if len(parts) >= 2:
                    field = parts[0].strip()
                    expected = parts[1].strip()
                    if field not in output or expected not in str(output[field]):
                        self._logger.warning(f"Postcondition warning: {condition}")


class StepExecutor:
    """步骤执行器 - 执行 pipeline step"""

    def __init__(self, step: PipelineStepInterface):
        self._step = step
        self._logger = logging.getLogger(f"StepExecutor.{step.get_step_id()}")

    def execute(self, context: Dict[str, Any]) -> StepExecutionResult:
        """执行步骤"""
        import time
        start_time = time.time()

        try:
            self._check_dependencies(context)

            if self._step.can_skip() and self._should_skip(context):
                execution_time = time.time() - start_time
                return StepExecutionResult(
                    step_id=self._step.get_step_id(),
                    success=True,
                    skipped=True,
                    execution_time=execution_time,
                )

            self._logger.debug(f"Executing step: {self._step.get_step_id()}")
            output = self._step.execute(context)

            execution_time = time.time() - start_time

            return StepExecutionResult(
                step_id=self._step.get_step_id(),
                success=True,
                output=output,
                execution_time=execution_time,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            self._logger.error(f"Step execution failed: {e}")

            try:
                recovery_output = self._step.on_failure(e, context)
                return StepExecutionResult(
                    step_id=self._step.get_step_id(),
                    success=False,
                    output=recovery_output,
                    errors=[str(e)],
                    execution_time=execution_time,
                )
            except Exception as recovery_error:
                return StepExecutionResult(
                    step_id=self._step.get_step_id(),
                    success=False,
                    errors=[str(e), str(recovery_error)],
                    execution_time=execution_time,
                )

    def _check_dependencies(self, context: Dict[str, Any]) -> None:
        """检查依赖"""
        for dep in self._step.get_dependencies():
            if dep not in context:
                self._logger.warning(f"Missing dependency: {dep}")

    def _should_skip(self, context: Dict[str, Any]) -> bool:
        """判断是否应该跳过"""
        return False


class SkillPipelineBridge:
    """Skill-Pipeline 桥接器 - 统一管理 skill 和 pipeline 的交互

    与现有系统集成：
    - 委托给 canonical harness registry（ontology_compiler.harness）进行技能发现
    - 委托给 skills/skill_router.py 进行技能路由
    - 与 core/schema_registry.py 集成进行结构验证

    不维护独立的技能注册表，而是作为现有系统的适配器
    """

    def __init__(self):
        self._skill_executors: Dict[str, SkillExecutor] = {}
        self._step_executors: Dict[str, StepExecutor] = {}
        self._skill_registry = None
        self._skill_router = None
        self._schema_registry = None
        self._loaded = False

    @property
    def skill_registry(self):
        if self._skill_registry is None:
            try:
                from ontology_compiler.skills.registry import get_skill_registry
                self._skill_registry = get_skill_registry()
                # Load legacy metadata into canonical registry once
                try:
                    from pathlib import Path
                    skills_root = Path(__file__).resolve().parents[2] / "skills"
                    self._skill_registry.load_metadata_from_skill_json(str(skills_root))
                except Exception as e:
                    logger.warning(f"Failed to load skill metadata into canonical registry: {e}")
            except ImportError as e:
                logger.warning(f"Failed to import canonical skill_registry: {e}")
        return self._skill_registry

    @property
    def skill_router(self):
        if self._skill_router is None:
            try:
                from skills.skill_router import SkillsRouter
                # Discovery is harness-only; pass runtime registry for BaseSkill
                # instance lookup when apply_skills needs executable objects.
                self._skill_router = SkillsRouter(skill_registry=self.skill_registry)
            except ImportError as e:
                logger.warning(f"Failed to import skill_router: {e}")
        return self._skill_router

    @property
    def schema_registry(self):
        if self._schema_registry is None:
            try:
                from core.schema_registry import get_schema_registry
                self._schema_registry = get_schema_registry()
            except ImportError as e:
                logger.warning(f"Failed to import schema_registry: {e}")
        return self._schema_registry

    def load_skills(self):
        """加载技能（委托给现有技能注册表）"""
        if self._loaded:
            return

        # Primary: canonical harness discovery drives executor population
        # (P0-6-3 2026-09-09: legacy skills.skill_registry already delegates its
        # load_skills() to build_default_skill_registry; driving discovery here
        # removes the bridge's own dependency on the legacy module while the
        # executors still wrap SkillMetadata-shaped objects via _SkillMetadataWrapper).
        legacy_loaded = False
        try:
            from ontology_compiler.harness import build_default_skill_registry
            from ontology_compiler.harness.skill_registry import (
                SkillDefinition as CanonicalSkillDefinition,
            )
            from skills.skill_metadata import SkillMetadata

            canonical = build_default_skill_registry()
            loaded_count = 0
            for skill in canonical.list_skills(enabled_only=False):
                if not isinstance(skill, CanonicalSkillDefinition):
                    continue
                metadata = SkillMetadata(
                    skill_id=skill.name,
                    name=skill.name,
                    description=skill.description,
                    version=skill.version,
                    category=skill.category,
                    tags=list(skill.tags),
                    business_domain=list(skill.domains) if skill.domains else None,
                    active=skill.enabled,
                    deprecated=not skill.enabled,
                    metadata={
                        "script_path": skill.script_path,
                        "command": skill.command,
                        "stages": skill.stages or [],
                        "args_schema": skill.args_schema or {},
                    },
                )
                wrapper = _SkillMetadataWrapper(metadata)
                self._skill_executors[skill.name] = SkillExecutor(wrapper)
                loaded_count += 1
            logger.info(f"Loaded {loaded_count} skills from canonical harness registry")
            legacy_loaded = True
        except ImportError as e:
            logger.warning(f"Canonical harness skill_registry not available: {e}")

        # Secondary: ensure canonical registry has metadata loaded
        if self.skill_registry:
            try:
                from pathlib import Path
                skills_root = Path(__file__).resolve().parents[2] / "skills"
                self.skill_registry.load_metadata_from_skill_json(str(skills_root))
            except Exception as e:
                logger.warning(f"Failed to sync metadata to canonical registry: {e}")

        # C-1 (2026-08-15): register the executable relation_rewrite skill into
        # the canonical (ontology_compiler.skills) registry so the bridge's
        # agent path can actually run it when routed. Guarded by env flag —
        # production behavior is unchanged unless PIPELINE_SKILL_RELATION_REWRITE=1.
        if os.environ.get("PIPELINE_SKILL_RELATION_REWRITE", "0").lower() in {"1", "true", "yes"}:
            try:
                from ontology_compiler.skills.registry import get_skill_registry as get_oc_registry
                from ontology_compiler.skills.builtin.relation_rewrite_skill import RelationRewriteSkill
                oc_registry = get_oc_registry()
                if oc_registry.get("relation_rewrite") is None:
                    oc_registry.register(RelationRewriteSkill(), category="post_processing")
                    logger.info("C-1: registered executable relation_rewrite skill (experiment enabled)")
            except Exception as e:
                logger.warning(f"C-1: failed to register relation_rewrite skill: {e}")

        self._loaded = True

    def register_skill(self, skill: SkillInterface) -> None:
        """注册 skill"""
        self._skill_executors[skill.get_skill_id()] = SkillExecutor(skill)
        logger.info(f"Registered skill: {skill.get_skill_id()}")

    def register_step(self, step: PipelineStepInterface) -> None:
        """注册 step"""
        self._step_executors[step.get_step_id()] = StepExecutor(step)
        logger.info(f"Registered step: {step.get_step_id()}")

    def execute_skill(self, skill_id: str, context: Dict[str, Any]) -> SkillExecutionResult:
        """执行 skill"""
        self._ensure_loaded()

        executor = self._skill_executors.get(skill_id)
        if executor:
            result = executor.execute(context)
            if self.skill_registry:
                try:
                    self.skill_registry.update_skill_usage(
                        skill_id,
                        success=result.success,
                        execution_time=result.execution_time,
                    )
                except AttributeError:
                    # Some registry implementations lack the usage-tracking
                    # method; execution result is unaffected.
                    logger.debug(f"Skill registry has no update_skill_usage for {skill_id}")
                except Exception as exc:
                    logger.debug(f"Failed to record skill usage for {skill_id}: {exc}")
            return result

        raise ValueError(f"Skill not found: {skill_id}")

    def execute_step(self, step_id: str, context: Dict[str, Any]) -> StepExecutionResult:
        """执行 step"""
        executor = self._step_executors.get(step_id)
        if not executor:
            raise ValueError(f"Step not found: {step_id}")
        return executor.execute(context)

    def route_skills(self, entity: Dict[str, Any]) -> List[str]:
        """路由技能（委托给技能路由器）"""
        self._ensure_loaded()

        if self.skill_router:
            return self.skill_router.route(entity)
        return []

    def route_batch(self, entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """批量路由技能"""
        self._ensure_loaded()

        if self.skill_router:
            return self.skill_router.route_batch(entities)
        return {}

    def recommend_skills(self, **kwargs) -> List[Dict[str, Any]]:
        """推荐技能"""
        self._ensure_loaded()

        if self.skill_registry:
            metadata_list = self.skill_registry.recommend(**kwargs)
            return [meta.to_dict() for meta in metadata_list]
        return []

    def get_skill_info(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """获取 skill 信息"""
        self._ensure_loaded()

        if self.skill_registry:
            metadata = self.skill_registry.get(skill_id)
            if metadata:
                return metadata.to_dict()

        executor = self._skill_executors.get(skill_id)
        if executor:
            skill = executor._skill
            return {
                "skill_id": skill.get_skill_id(),
                "name": skill.get_name(),
                "description": skill.get_description(),
                "purpose": skill.get_purpose(),
                "input_schema": skill.get_input_schema(),
                "output_schema": skill.get_output_schema(),
                "applicable_layers": skill.get_applicable_layers(),
                "quality_expectations": skill.get_quality_expectations(),
            }

        return None

    def get_step_info(self, step_id: str) -> Optional[Dict[str, Any]]:
        """获取 step 信息"""
        executor = self._step_executors.get(step_id)
        if not executor:
            return None

        step = executor._step
        return {
            "step_id": step.get_step_id(),
            "name": step.get_name(),
            "description": step.get_description(),
            "dependencies": step.get_dependencies(),
            "input_keys": step.get_input_keys(),
            "output_keys": step.get_output_keys(),
            "can_skip": step.can_skip(),
        }

    def list_skills(self) -> List[str]:
        """列出所有可执行的 skill。

        以真实 executor 为准（_skill_executors）：metadata-only 的 skill id
        （来自 runtime singleton 的 list_active，如 legacy skill.json 记录）
        不在此列——"存在 metadata" 不等于 "可执行"。executor 为空时回退到
        registry active（兼容早期未加载 executors 的路径）。
        """
        self._ensure_loaded()

        executors = list(self._skill_executors.keys())
        if executors:
            return executors

        if self.skill_registry:
            try:
                active = self.skill_registry.list_active()
                if active:
                    return active
            except Exception:
                pass
        return []

    def list_steps(self) -> List[str]:
        """列出所有 step"""
        return list(self._step_executors.keys())

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        self._ensure_loaded()

        stats = {
            "skills_count": len(self.list_skills()),
            "steps_count": len(self._step_executors),
            "skills": self.list_skills(),
            "steps": self.list_steps(),
        }

        if self.skill_registry:
            try:
                stats["skill_registry_stats"] = self.skill_registry.to_dict()["_stats"]
            except Exception:
                pass
            try:
                stats["canonical_registry_stats"] = self.skill_registry.get_stats()
            except Exception:
                pass

        return stats

    def get_skill_stats(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """获取技能使用统计"""
        if self.skill_registry:
            return self.skill_registry.get_skill_stats(skill_id)
        return None

    def _ensure_loaded(self):
        """确保技能已加载"""
        if not self._loaded:
            self.load_skills()


class _SkillMetadataWrapper(SkillInterface):
    """SkillMetadata 包装器 - 将现有 SkillMetadata 适配为 SkillInterface"""

    def __init__(self, metadata):
        self._metadata = metadata

    def get_skill_id(self) -> str:
        return self._metadata.skill_id

    def get_name(self) -> str:
        return self._metadata.name

    def get_description(self) -> str:
        return self._metadata.description

    def get_purpose(self) -> str:
        return self._metadata.purpose

    def get_input_schema(self) -> Dict[str, Any]:
        schema = self._metadata.input_schema.model_dump() if hasattr(self._metadata.input_schema, 'model_dump') else (self._metadata.input_schema or {"properties": {}})
        # 2026-08-15 (C-1): the discovery layer maps skill.json ``args_schema``
        # into ``metadata.args_schema`` but not always into ``input_schema``.
        # When properties are empty, fall back to args_schema so
        # SkillExecutor._convert_context_to_input does not strip all inputs.
        props = schema.get("properties") or {}
        if not props:
            args_schema = getattr(self._metadata, "args_schema", None) or getattr(getattr(self._metadata, "metadata", None) or {}, "args_schema", None)
            if isinstance(args_schema, dict) and args_schema:
                schema = {"type": "object", "required": [], "properties": args_schema}
        return schema

    def get_output_schema(self) -> Dict[str, Any]:
        return self._metadata.output_schema.model_dump() if hasattr(self._metadata.output_schema, 'model_dump') else (self._metadata.output_schema or {"properties": {}})

    def get_preconditions(self) -> List[str]:
        return self._metadata.trigger.keywords if hasattr(self._metadata, 'trigger') and hasattr(self._metadata.trigger, 'keywords') else []

    def get_postconditions(self) -> List[str]:
        return []

    def get_applicable_layers(self) -> List[str]:
        return self._metadata.trigger.supported_layers if hasattr(self._metadata, 'trigger') and hasattr(self._metadata.trigger, 'supported_layers') else []

    def get_quality_expectations(self) -> Dict[str, Any]:
        return self._metadata.quality_expectations or {}

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行技能 - 优先通过脚本执行，回退到代理调用"""
        logger.info(f"Executing skill via wrapper: {self._metadata.skill_id}")
        
        result = self._execute_via_script(input_data)
        if result is not None:
            return result
        
        result = self._execute_via_agent(input_data)
        if result is not None:
            return result
        
        logger.warning(f"Skill {self._metadata.skill_id} has no executable implementation")
        return {"status": "executed", "skill_id": self._metadata.skill_id, "output": {}}

    def _execute_via_script(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """通过脚本路径执行技能"""
        import os
        import subprocess
        import json
        
        if not hasattr(self._metadata, 'scripts') or not hasattr(self._metadata.scripts, 'main'):
            return None
        
        main_script = self._metadata.scripts.main
        if not main_script:
            return None
        
        skill_path = getattr(self._metadata, 'skill_path', None)
        if skill_path:
            script_full_path = os.path.join(skill_path, main_script)
        else:
            script_full_path = main_script
        
        if not os.path.exists(script_full_path):
            logger.debug(f"Script not found: {script_full_path}")
            return None
        
        try:
            logger.debug(f"Executing script: {script_full_path}")
            
            input_json = json.dumps(input_data, ensure_ascii=False)
            
            result = subprocess.run(
                ['python', script_full_path],
                input=input_json,
                capture_output=True,
                text=True,
                cwd=skill_path if skill_path else None,
                timeout=300
            )
            
            if result.returncode == 0:
                try:
                    output_data = json.loads(result.stdout)
                    return {
                        "status": "executed",
                        "skill_id": self._metadata.skill_id,
                        "output": output_data,
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
                except json.JSONDecodeError:
                    return {
                        "status": "executed",
                        "skill_id": self._metadata.skill_id,
                        "output": result.stdout,
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
            else:
                logger.error(f"Script execution failed: {result.stderr}")
                return {
                    "status": "failed",
                    "skill_id": self._metadata.skill_id,
                    "error": result.stderr,
                    "stdout": result.stdout,
                    "return_code": result.returncode
                }
        except Exception as e:
            logger.error(f"Error executing script: {e}")
            return None

    def _execute_via_agent(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """通过技能实例执行（如果已注册）

        2026-08-15 (C-1): canonical skills follow the async
        ``ontology_compiler.skills.base.Skill.execute`` contract; the previous
        code called it synchronously, producing an unawaited coroutine with
        empty output. Await coroutine results so async canonical skills
        (e.g. RelationRewriteSkill) actually execute.
        """
        import inspect
        # P0-6-3 (2026-09-09): legacy skills.skill_registry already delegates
        # discovery to the canonical harness registry; the executors themselves
        # are populated from harness in load_skills(), so the only remaining
        # source of *executable* instances is the canonical runtime registry.
        try:
            from ontology_compiler.skills.registry import get_skill_registry
            oc_registry = get_skill_registry()
            skill_instance = oc_registry.get(self._metadata.skill_id)
            if skill_instance:
                logger.debug(f"Executing via ontology_compiler.skills: {self._metadata.skill_id}")
                from ontology_compiler.skills.base import SkillContext
                context = SkillContext(config=input_data, extraction_result=input_data.get('extraction_result', input_data))
                result = skill_instance.execute(context)
                if inspect.isawaitable(result):
                    import asyncio
                    result = asyncio.run(result)
                return {
                    "status": "executed",
                    "skill_id": self._metadata.skill_id,
                    "output": result.output if hasattr(result, 'output') else {"result": str(result)}
                }
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Error executing via ontology_compiler.skills: {e}")
        
        return None


_skill_pipeline_bridge: Optional[SkillPipelineBridge] = None


def get_skill_pipeline_bridge() -> SkillPipelineBridge:
    """获取 Skill-Pipeline 桥接器实例"""
    global _skill_pipeline_bridge

    if _skill_pipeline_bridge is None:
        _skill_pipeline_bridge = SkillPipelineBridge()
        _skill_pipeline_bridge.load_skills()

    return _skill_pipeline_bridge


def reset_skill_pipeline_bridge() -> None:
    """Reset the Skill-Pipeline bridge singleton (test isolation hook).

    2026-08-15 (C-1): needed to re-load skills under a different
    ``PIPELINE_SKILL_RELATION_REWRITE`` env value between tests."""
    global _skill_pipeline_bridge
    _skill_pipeline_bridge = None

    return _skill_pipeline_bridge