# generate-ontology-patch

Generate ontology patches with evidence tracking and approval digests.

## Description
This skill generates ontology patches for the ontology system. It supports:
- Resolving existing paths for input files
- Building operation evidence entries with automatic source reference resolution
- Generating patch approval digests for validation
- Integrating with EvidencePlaneService for automatic evidence tracking

## Parameters
- `--delta-dir`: Path to delta directory containing ontology changes
- `--output`: Output path for generated patch file
- `--runs-root`: Path to pipeline runs root for evidence tracking (optional)
- `--dry-run`: Generate patch without applying (default: false)

## Heuristics
- Use this skill when preparing ontology changes for review
- Use for generating patches before publish stage
- Use for CI/CD automated patch generation
- Use when manual patch creation is needed
