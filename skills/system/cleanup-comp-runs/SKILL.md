# cleanup-comp-runs

Cleanup COMP-* intermediate run directories under outputs/ontology/runs.

## Description
This skill cleans up intermediate COMP-* run directories while preserving TBH-* directories. It supports dry-run mode (default) and apply mode with configurable retention policies.

## Parameters
- `--runs-root`: Root directory to scan (default: outputs/ontology/runs)
- `--retention-days`: Keep directories modified within this many days (default: 14)
- `--keep-latest`: Always keep the N most recent COMP- dirs regardless of age (default: 0 = disabled)
- `--mode`: Operation mode - dry-run or apply (default: dry-run)
- `--yes`: Skip confirmation prompt in apply mode (for CI/automation)

## Heuristics
- Use this skill for regular maintenance to free up disk space
- Use in CI/CD pipelines to clean up old intermediate runs
- Prefer dry-run mode first to preview what would be deleted
- Use apply mode only after reviewing dry-run results
