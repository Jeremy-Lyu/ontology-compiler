"""Core error types shared between src and ontology_compiler.

This module contains all domain-level error types and retry policies
used throughout the application. It is the single source of truth for
error definitions to avoid duplication and maintain consistency.

Usage:
    from core.errors import PipelineError, DataContractError

Exit Codes:
    EXIT_SUCCESS = 0
    EXIT_ERROR = 1
    EXIT_PIPELINE_ERROR = 2
    EXIT_VALIDATION_ERROR = 3
    EXIT_PERMISSION_ERROR = 4
    EXIT_NOT_FOUND = 5
    EXIT_CONFIG_ERROR = 11
    EXIT_ARTIFACT_ERROR = 12
    EXIT_SCHEMA_ERROR = 13
    EXIT_CONTRACT_ERROR = 14
    EXIT_TRANSIENT_ERROR = 15
    EXIT_TIMEOUT = 124
    EXIT_UNKNOWN = 125
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass


# Standard exit codes for all scripts
# NOTE: values match the test-spec contract used across test_pipeline_orchestrator.py
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_PIPELINE_ERROR = 2  # pipeline stage failure / hook denied
EXIT_VALIDATION_ERROR = 3
EXIT_PERMISSION_ERROR = 4
EXIT_NOT_FOUND = 5
EXIT_CONFIG_ERROR = 11  # config / stage-selection errors
EXIT_ARTIFACT_ERROR = 12
EXIT_SCHEMA_ERROR = 13
EXIT_CONTRACT_ERROR = 14
EXIT_TRANSIENT_ERROR = 15
EXIT_TIMEOUT = 124
EXIT_UNKNOWN = 125

# Keep legacy alias so existing callers using EXIT_ERROR still compile
EXIT_CONFIG_ERROR_LEGACY = 2  # do not use in new code


class PipelineError(Exception):
    """Base error for pipeline execution failures."""
    error_type = "PipelineError"
    exit_code = EXIT_PIPELINE_ERROR
    action_hint = "Inspect pipeline logs"


class ConfigError(PipelineError):
    """Error when configuration is invalid or missing."""
    error_type = "ConfigError"
    exit_code = EXIT_CONFIG_ERROR
    action_hint = "Check config/*.json syntax and profile names"


class ArtifactPathError(PipelineError):
    """Error when artifact path operations fail."""
    error_type = "ArtifactPathError"
    exit_code = EXIT_ARTIFACT_ERROR
    action_hint = "Check manifest artifacts paths and file existence"


class SchemaValidationError(PipelineError):
    """Error when schema validation fails."""
    error_type = "SchemaValidationError"
    exit_code = EXIT_SCHEMA_ERROR
    action_hint = "Re-generate artifacts and fix schema violations"


class DataContractError(PipelineError):
    """Error when data contract validation fails."""
    error_type = "DataContractError"
    exit_code = EXIT_CONTRACT_ERROR
    action_hint = "Check required fields and contract alignment"


class TransientInfraError(PipelineError):
    """Error when transient infrastructure issues occur."""
    error_type = "TransientInfraError"
    exit_code = EXIT_TRANSIENT_ERROR
    action_hint = "Retry allowed: network/temporary infrastructure issue"


class HookDeniedError(PipelineError):
    """Error when a hook denies execution."""
    error_type = "HookDeniedError"
    exit_code = EXIT_PIPELINE_ERROR
    action_hint = "Hook denied execution; inspect hook output and policy"


class ExtractionError(PipelineError):
    """Error during ontology extraction."""
    error_type = "ExtractionError"
    exit_code = EXIT_PIPELINE_ERROR
    action_hint = "Check source file format and content; verify LLM connectivity"


class LLMError(PipelineError):
    """Error when LLM API calls fail."""
    error_type = "LLMError"
    exit_code = EXIT_TRANSIENT_ERROR
    action_hint = "Check API keys, network connectivity, and LLM service status"


class CacheError(PipelineError):
    """Error when cache operations fail."""
    error_type = "CacheError"
    exit_code = EXIT_ARTIFACT_ERROR
    action_hint = "Check cache directory permissions and disk space"


class ValidationError(PipelineError):
    """Error when validation fails."""
    error_type = "ValidationError"
    exit_code = EXIT_VALIDATION_ERROR
    action_hint = "Fix validation issues and retry"


@dataclass
class ErrorMeta:
    """Error metadata for classification and reporting."""
    error_type: str
    exit_code: int
    action_hint: str


@dataclass
class RetryPolicy:
    """Retry policy for transient errors."""
    max_attempts: int = 5
    base_delay_s: float = 5.0
    max_delay_s: float = 60.0
    jitter: bool = True
    cooldown_s: float = 1.0


def with_transient_retry(fn, policy: RetryPolicy | None = None):
    """Execute fn(), retrying only on TransientInfraError with exponential backoff.

    Non-transient errors are re-raised immediately without retry.
    Returns the result of fn() on success.
    Raises the last TransientInfraError after all attempts are exhausted.
    """
    p = policy or RetryPolicy()
    last_exc: Exception | None = None

    for attempt in range(1, p.max_attempts + 1):
        try:
            return fn()
        except TransientInfraError as exc:
            last_exc = exc
            if attempt == p.max_attempts:
                break
            delay = min(p.base_delay_s * math.pow(2, attempt - 1), p.max_delay_s)
            if p.jitter:
                delay = delay * (0.5 + random.random() * 0.5)
            time.sleep(delay)
        except Exception:
            raise

    raise last_exc  # type: ignore[misc]


async def with_transient_retry_async(coro_fn, policy: RetryPolicy | None = None):
    """Async version of with_transient_retry.

    Awaits coro_fn(), retrying only on TransientInfraError with exponential backoff.
    coro_fn must be a zero-argument async callable (e.g. a lambda or functools.partial).

    Non-transient errors are re-raised immediately without retry.
    Returns the awaited result of coro_fn() on success.
    Raises the last TransientInfraError after all attempts are exhausted.
    """
    p = policy or RetryPolicy()
    last_exc: Exception | None = None

    for attempt in range(1, p.max_attempts + 1):
        try:
            result = await coro_fn()
            if p.cooldown_s > 0:
                await asyncio.sleep(p.cooldown_s)
            return result
        except TransientInfraError as exc:
            last_exc = exc
            if attempt == p.max_attempts:
                break
            delay = min(p.base_delay_s * math.pow(2, attempt - 1), p.max_delay_s)
            if p.jitter:
                delay = delay * (0.5 + random.random() * 0.5)
            await asyncio.sleep(delay)
        except Exception:
            raise

    raise last_exc  # type: ignore[misc]


def get_exit_code_for_error(error: Exception) -> int:
    """Get the appropriate exit code for a given error type.

    Args:
        error: The exception that occurred

    Returns:
        The appropriate exit code
    """
    # If it's a PipelineError subclass, use its exit_code attribute
    if isinstance(error, PipelineError):
        return error.exit_code

    error_name = error.__class__.__name__

    # Map standard exception types to exit codes
    error_exit_codes = {
        "ConfigError": EXIT_CONFIG_ERROR,
        "SchemaValidationError": EXIT_VALIDATION_ERROR,
        "PermissionError": EXIT_PERMISSION_ERROR,
        "FileNotFoundError": EXIT_NOT_FOUND,
        "TimeoutError": EXIT_TIMEOUT,
    }

    return error_exit_codes.get(error_name, EXIT_ERROR)


def classify_error(exc: Exception) -> ErrorMeta:
    """Classify an exception into ErrorMeta for reporting.

    Args:
        exc: The exception to classify

    Returns:
        ErrorMeta with error type, exit code, and action hint
    """
    if isinstance(exc, PipelineError):
        return ErrorMeta(exc.error_type, exc.exit_code, exc.action_hint)
    return ErrorMeta("UnknownError", 99, "Inspect stacktrace and logs")
