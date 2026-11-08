"""Core error types - shared between src and ontology_compiler.

This package contains all domain-level error types and retry policies.
Import from here instead of legacy locations to ensure consistency.
"""

from core.errors.errors import (
    # Exit codes
    EXIT_SUCCESS,
    EXIT_ERROR,
    EXIT_PIPELINE_ERROR,
    EXIT_VALIDATION_ERROR,
    EXIT_PERMISSION_ERROR,
    EXIT_NOT_FOUND,
    EXIT_CONFIG_ERROR,
    EXIT_ARTIFACT_ERROR,
    EXIT_SCHEMA_ERROR,
    EXIT_CONTRACT_ERROR,
    EXIT_TRANSIENT_ERROR,
    EXIT_TIMEOUT,
    EXIT_UNKNOWN,
    EXIT_CONFIG_ERROR_LEGACY,
    # Error classes
    PipelineError,
    ConfigError,
    ArtifactPathError,
    SchemaValidationError,
    DataContractError,
    TransientInfraError,
    HookDeniedError,
    ExtractionError,
    LLMError,
    CacheError,
    ValidationError,
    # Data classes
    ErrorMeta,
    RetryPolicy,
    # Functions
    with_transient_retry,
    with_transient_retry_async,
    get_exit_code_for_error,
    classify_error,
)

__all__ = [
    # Exit codes
    "EXIT_SUCCESS",
    "EXIT_ERROR",
    "EXIT_PIPELINE_ERROR",
    "EXIT_VALIDATION_ERROR",
    "EXIT_PERMISSION_ERROR",
    "EXIT_NOT_FOUND",
    "EXIT_CONFIG_ERROR",
    "EXIT_ARTIFACT_ERROR",
    "EXIT_SCHEMA_ERROR",
    "EXIT_CONTRACT_ERROR",
    "EXIT_TRANSIENT_ERROR",
    "EXIT_TIMEOUT",
    "EXIT_UNKNOWN",
    "EXIT_CONFIG_ERROR_LEGACY",
    # Error classes
    "PipelineError",
    "ConfigError",
    "ArtifactPathError",
    "SchemaValidationError",
    "DataContractError",
    "TransientInfraError",
    "HookDeniedError",
    "ExtractionError",
    "LLMError",
    "CacheError",
    "ValidationError",
    # Data classes
    "ErrorMeta",
    "RetryPolicy",
    # Functions
    "with_transient_retry",
    "with_transient_retry_async",
    "get_exit_code_for_error",
    "classify_error",
]
