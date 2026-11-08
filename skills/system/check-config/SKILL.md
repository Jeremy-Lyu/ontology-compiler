# check-config

Check runtime/config layering and current LLM configuration.

## Description
This skill validates the configuration layering and checks the current LLM configuration. It helps ensure that environment variables, runtime settings, and compiler configuration are properly layered and that sensitive values are masked appropriately.

## Parameters
- `--show-all`: Show all configuration sources (default: false)
- `--show-sensitive`: Show sensitive values (default: false, masks with ***)
- `--check-llm`: Check LLM provider configuration (default: true)

## Heuristics
- Use this skill to diagnose configuration issues
- Use when setting up new environments
- Use for CI/CD configuration validation
- Use to verify environment variable precedence
