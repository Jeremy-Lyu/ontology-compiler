"""Pipeline step configuration loader.

Provides centralized configuration for SingleFilePipeline and FolderAnalysisPipeline
steps, allowing dynamic adjustment of execution flow without code changes.

Usage:
    from src.infrastructure.config.pipeline_config import PipelineConfig

    config = PipelineConfig.load()

    # Get default flow for SingleFilePipeline
    steps = config.get_flow("single_file", "default")

    # Get step configuration
    step_config = config.get_step_config("single_file", "rigor_extract")
    max_iterations = step_config.get("config", {}).get("max_iterations", 5)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from src.pipeline.stages.common import PROJECT_ROOT

logger = logging.getLogger(__name__)


class PipelineConfigError(Exception):
    """Error in pipeline configuration."""
    pass

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline" / "pipeline_steps.yaml"


@dataclass
class StepConfig:
    """Configuration for a single pipeline step.

    Attributes:
        id: Unique step identifier
        name: Human-readable name
        description: Step description
        required: Whether this step is required
        class_name: Python class name implementing the step
        timeout_seconds: Execution timeout
        retry_count: Number of retries on failure
        depends_on: List of step IDs that must complete before this step
        condition: Optional condition expression for step execution
        config: Additional step-specific configuration
    """

    id: str
    name: str = ""
    description: str = ""
    required: bool = False
    class_name: str = ""
    timeout_seconds: int = 300
    retry_count: int = 2
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StepConfig:
        """Create StepConfig from dictionary."""
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            description=data.get("description", ""),
            required=data.get("required", False),
            class_name=data.get("class", ""),
            timeout_seconds=data.get("timeout_seconds", 300),
            retry_count=data.get("retry_count", 2),
            depends_on=data.get("depends_on", []),
            condition=data.get("condition"),
            config=data.get("config", {}),
        )


@dataclass
class PipelineFlow:
    """Defines a named flow (sequence of steps).

    Attributes:
        name: Flow name (e.g., "default", "quick", "strict")
        description: Flow description
        steps: Ordered list of step IDs
    """

    name: str
    description: str = ""
    steps: List[str] = field(default_factory=list)


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""

    default_timeout: int = 300
    max_workers: int = 4
    batch_size: int = 10
    queue_size: int = 100
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size_mb: int = 512

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PerformanceConfig:
        parallel = data.get("parallel", {})
        cache = data.get("cache", {})
        return cls(
            default_timeout=data.get("default_timeout", 300),
            max_workers=parallel.get("max_workers", 4),
            batch_size=parallel.get("batch_size", 10),
            queue_size=parallel.get("queue_size", 100),
            cache_enabled=cache.get("enabled", True),
            cache_ttl_seconds=cache.get("ttl_seconds", 3600),
            cache_max_size_mb=cache.get("max_size_mb", 512),
        )


@dataclass
class ObservabilityConfig:
    """Observability and monitoring configuration."""

    step_logging_enabled: bool = True
    log_level: str = "INFO"
    log_input_output: bool = False
    metrics_enabled: bool = True
    collect_step_duration: bool = True
    collect_memory_usage: bool = True
    export_interval_seconds: int = 60
    tracing_enabled: bool = False
    service_name: str = "ontology-pipeline"
    sample_rate: float = 0.1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObservabilityConfig:
        step_logging = data.get("step_logging", {})
        metrics = data.get("metrics", {})
        tracing = data.get("tracing", {})
        return cls(
            step_logging_enabled=step_logging.get("enabled", True),
            log_level=step_logging.get("log_level", "INFO"),
            log_input_output=step_logging.get("log_input_output", False),
            metrics_enabled=metrics.get("enabled", True),
            collect_step_duration=metrics.get("collect_step_duration", True),
            collect_memory_usage=metrics.get("collect_memory_usage", True),
            export_interval_seconds=metrics.get("export_interval_seconds", 60),
            tracing_enabled=tracing.get("enabled", False),
            service_name=tracing.get("service_name", "ontology-pipeline"),
            sample_rate=tracing.get("sample_rate", 0.1),
        )


class PipelineConfig:
    """Centralized pipeline step configuration.

    Loads and provides access to pipeline step definitions, flows,
    performance settings, and observability configuration.
    """

    _instance: Optional["PipelineConfig"] = None
    _config_path: Path = DEFAULT_CONFIG_PATH

    def __init__(self) -> None:
        """Initialize configuration. Use load() for singleton access."""
        self._data: Dict[str, Any] = {}
        self._single_file_steps: Dict[str, StepConfig] = {}
        self._folder_analysis_steps: Dict[str, StepConfig] = {}
        self._single_file_flows: Dict[str, PipelineFlow] = {}
        self._folder_analysis_flows: Dict[str, PipelineFlow] = {}
        self._performance: PerformanceConfig = PerformanceConfig()
        self._observability: ObservabilityConfig = ObservabilityConfig()
        self._loaded: bool = False

    @classmethod
    def load(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        force_reload: bool = False,
    ) -> "PipelineConfig":
        """Load pipeline configuration from YAML file.

        Args:
            config_path: Path to configuration file (default: config/pipeline_steps.yaml)
            force_reload: Force reload even if already loaded

        Returns:
            PipelineConfig instance

        Raises:
            ConfigValidationError: If configuration is invalid
            FileNotFoundError: If configuration file not found
        """
        if cls._instance is not None and not force_reload and cls._instance._loaded:
            return cls._instance

        instance = cls()
        path = Path(config_path) if config_path else cls._config_path

        if not path.exists():
            raise FileNotFoundError(f"Pipeline configuration not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise PipelineConfigError(f"Invalid YAML in pipeline config: {e}")

        instance._data = data or {}
        instance._parse_config()
        instance._loaded = True
        cls._instance = instance

        logger.info(f"Loaded pipeline configuration from {path}")
        return instance

    def _parse_config(self) -> None:
        """Parse raw configuration data into structured objects."""
        # Parse SingleFilePipeline steps
        sfp_config = self._data.get("single_file_pipeline", {})
        for step_data in sfp_config.get("available_steps", []):
            step = StepConfig.from_dict(step_data)
            self._single_file_steps[step.id] = step

        # Parse SingleFilePipeline flows
        for flow_name, steps in sfp_config.items():
            if flow_name.endswith("_flow"):
                flow_name = flow_name.replace("_flow", "")
                self._single_file_flows[flow_name] = PipelineFlow(
                    name=flow_name,
                    steps=steps if isinstance(steps, list) else [],
                )

        # Parse FolderAnalysisPipeline steps
        fap_config = self._data.get("folder_analysis_pipeline", {})
        for step_data in fap_config.get("available_steps", []):
            step = StepConfig.from_dict(step_data)
            self._folder_analysis_steps[step.id] = step

        # Parse FolderAnalysisPipeline flows
        for flow_name, steps in fap_config.items():
            if flow_name.endswith("_flow"):
                flow_name = flow_name.replace("_flow", "")
                self._folder_analysis_flows[flow_name] = PipelineFlow(
                    name=flow_name,
                    steps=steps if isinstance(steps, list) else [],
                )

        # Parse performance config
        perf_data = self._data.get("performance", {})
        self._performance = PerformanceConfig.from_dict(perf_data)

        # Parse observability config
        obs_data = self._data.get("observability", {})
        self._observability = ObservabilityConfig.from_dict(obs_data)

    def get_step_config(self, pipeline: str, step_id: str) -> Optional[StepConfig]:
        """Get configuration for a specific step.

        Args:
            pipeline: "single_file" or "folder_analysis"
            step_id: Step identifier

        Returns:
            StepConfig if found, None otherwise
        """
        if pipeline == "single_file":
            return self._single_file_steps.get(step_id)
        elif pipeline == "folder_analysis":
            return self._folder_analysis_steps.get(step_id)
        return None

    def get_all_steps(self, pipeline: str) -> Dict[str, StepConfig]:
        """Get all step configurations for a pipeline.

        Args:
            pipeline: "single_file" or "folder_analysis"

        Returns:
            Dictionary of step_id -> StepConfig
        """
        if pipeline == "single_file":
            return dict(self._single_file_steps)
        elif pipeline == "folder_analysis":
            return dict(self._folder_analysis_steps)
        return {}

    def get_required_steps(self, pipeline: str) -> List[str]:
        """Get list of required step IDs.

        Args:
            pipeline: "single_file" or "folder_analysis"

        Returns:
            List of required step IDs
        """
        steps = self.get_all_steps(pipeline)
        return [s.id for s in steps.values() if s.required]

    def get_optional_steps(self, pipeline: str) -> List[str]:
        """Get list of optional step IDs.

        Args:
            pipeline: "single_file" or "folder_analysis"

        Returns:
            List of optional step IDs
        """
        steps = self.get_all_steps(pipeline)
        return [s.id for s in steps.values() if not s.required]

    def get_flow(self, pipeline: str, flow_name: str = "default") -> List[str]:
        """Get step execution order for a named flow.

        Args:
            pipeline: "single_file" or "folder_analysis"
            flow_name: Flow name (default, quick, strict, etc.)

        Returns:
            Ordered list of step IDs
        """
        flows = (
            self._single_file_flows
            if pipeline == "single_file"
            else self._folder_analysis_flows
        )

        flow = flows.get(flow_name)
        if flow:
            return list(flow.steps)

        # Fallback to default if requested flow not found
        if flow_name != "default":
            logger.warning(f"Flow '{flow_name}' not found, using 'default'")
            return self.get_flow(pipeline, "default")

        # Last resort: return all required steps in arbitrary order
        return self.get_required_steps(pipeline)

    def get_available_flows(self, pipeline: str) -> List[str]:
        """Get list of available flow names.

        Args:
            pipeline: "single_file" or "folder_analysis"

        Returns:
            List of flow names
        """
        flows = (
            self._single_file_flows
            if pipeline == "single_file"
            else self._folder_analysis_flows
        )
        return list(flows.keys())

    @property
    def performance(self) -> PerformanceConfig:
        """Performance optimization configuration."""
        return self._performance

    @property
    def observability(self) -> ObservabilityConfig:
        """Observability configuration."""
        return self._observability

    def get_step_timeout(self, pipeline: str, step_id: str) -> int:
        """Get timeout for a specific step.

        Args:
            pipeline: "single_file" or "folder_analysis"
            step_id: Step identifier

        Returns:
            Timeout in seconds
        """
        step = self.get_step_config(pipeline, step_id)
        if step:
            return step.timeout_seconds
        return self._performance.default_timeout

    def get_step_retry_count(self, pipeline: str, step_id: str) -> int:
        """Get retry count for a specific step.

        Args:
            pipeline: "single_file" or "folder_analysis"
            step_id: Step identifier

        Returns:
            Number of retries
        """
        step = self.get_step_config(pipeline, step_id)
        if step:
            return step.retry_count
        return 2

    def validate_flow(self, pipeline: str, flow: List[str]) -> List[str]:
        """Validate a custom flow and return any errors.

        Args:
            pipeline: "single_file" or "folder_analysis"
            flow: Ordered list of step IDs

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        steps = self.get_all_steps(pipeline)

        # Check all steps exist
        for step_id in flow:
            if step_id not in steps:
                errors.append(f"Unknown step: {step_id}")

        # Check dependencies
        for i, step_id in enumerate(flow):
            step = steps.get(step_id)
            if step:
                for dep in step.depends_on:
                    if dep not in flow:
                        errors.append(f"Step '{step_id}' depends on '{dep}' which is not in flow")
                    elif flow.index(dep) >= i:
                        errors.append(f"Step '{step_id}' depends on '{dep}' which appears later in flow")

        return errors

    def reload(self) -> None:
        """Reload configuration from file."""
        self._loaded = False
        self._parse_config()
        self._loaded = True
        logger.info("Reloaded pipeline configuration")

    def get_flow_with_env_override(
        self, pipeline: str, flow_name: str
    ) -> List[str]:
        """Get flow with environment variable override support.

        R21 Phase 4: 运行时动态调整支持
        - 检查环境变量 {PIPELINE}_{FLOW}_STEPS 是否定义了自定义步骤列表
        - 格式: comma-separated step IDs
        - 示例: SINGLE_FILE_DEFAULT_STEPS=initialize,extract,validate,ingest

        Args:
            pipeline: "single_file" or "folder_analysis"
            flow_name: Flow name (e.g., "default", "quick", "strict")

        Returns:
            List of step IDs (potentially overridden by environment)
        """
        # Check for environment variable override
        env_var_name = f"{pipeline.upper()}_{flow_name.upper()}_STEPS"
        env_override = os.environ.get(env_var_name)

        if env_override:
            # Parse comma-separated steps
            custom_steps = [s.strip() for s in env_override.split(",") if s.strip()]
            logger.info(
                f"Using environment override for {pipeline}/{flow_name}: {custom_steps}"
            )

            # Validate the custom flow
            errors = self.validate_flow(pipeline, custom_steps)
            if errors:
                logger.warning(f"Custom flow validation errors: {errors}")
                logger.warning("Falling back to configured flow")
            else:
                return custom_steps

        # Return configured flow
        return self.get_flow(pipeline, flow_name)

    def get_step_config_with_env_override(
        self, pipeline: str, step_id: str
    ) -> Optional[StepConfig]:
        """Get step configuration with environment variable override.

        R21 Phase 4: 运行时动态调整支持
        - 检查环境变量 {PIPELINE}_{STEP}_TIMEOUT 和 {PIPELINE}_{STEP}_RETRIES
        - 示例: SINGLE_FILE_RIGOR_EXTRACT_TIMEOUT=600

        Args:
            pipeline: "single_file" or "folder_analysis"
            step_id: Step identifier (e.g., "rigor_extract")

        Returns:
            StepConfig (potentially overridden by environment)
        """
        step = self.get_step_config(pipeline, step_id)
        if not step:
            return None

        # Check for timeout override
        timeout_env = os.environ.get(f"{pipeline.upper()}_{step_id.upper()}_TIMEOUT")
        if timeout_env:
            try:
                override_timeout = int(timeout_env)
                step = StepConfig(
                    id=step.id,
                    name=step.name,
                    description=step.description,
                    required=step.required,
                    class_name=step.class_name,
                    timeout_seconds=override_timeout,
                    retry_count=step.retry_count,
                    depends_on=step.depends_on,
                    condition=step.condition,
                    config=step.config,
                )
                logger.info(
                    f"Environment override: {pipeline}/{step_id} timeout={override_timeout}"
                )
            except ValueError:
                logger.warning(
                    f"Invalid timeout override '{timeout_env}' for {pipeline}/{step_id}"
                )

        # Check for retry count override
        retries_env = os.environ.get(f"{pipeline.upper()}_{step_id.upper()}_RETRIES")
        if retries_env:
            try:
                override_retries = int(retries_env)
                step = StepConfig(
                    id=step.id,
                    name=step.name,
                    description=step.description,
                    required=step.required,
                    class_name=step.class_name,
                    timeout_seconds=step.timeout_seconds,
                    retry_count=override_retries,
                    depends_on=step.depends_on,
                    condition=step.condition,
                    config=step.config,
                )
                logger.info(
                    f"Environment override: {pipeline}/{step_id} retries={override_retries}"
                )
            except ValueError:
                logger.warning(
                    f"Invalid retries override '{retries_env}' for {pipeline}/{step_id}"
                )

        return step

    @staticmethod
    def get_env_override_summary() -> Dict[str, Any]:
        """Get summary of all active environment variable overrides.

        R21 Phase 4: 运行时动态调整支持
        - 列出所有以 SINGLE_FILE_ 或 FOLDER_ANALYSIS_ 开头的环境变量

        Returns:
            Dictionary of active overrides by pipeline
        """
        overrides = {"single_file": {}, "folder_analysis": {}}

        for key, value in os.environ.items():
            if key.startswith("SINGLE_FILE_"):
                overrides["single_file"][key] = value
            elif key.startswith("FOLDER_ANALYSIS_"):
                overrides["folder_analysis"][key] = value

        return overrides
