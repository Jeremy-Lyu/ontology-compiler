# Ontology Compiler

Ontology compiler and extraction runtime. This is the canonical pipeline surface for turning source material into normalized ontology fragments and compiled artifacts.

## Purpose

- Parse source material into compilation fragments
- Extract ontology entities, relationships, and evidence
- Deduplicate, classify, enrich, verify, and merge outputs
- Produce canonical ontology artifacts and supporting outputs
- Keep the runtime pipeline stable and easy to trace

## What belongs here

- Canonical extraction and compilation services
- Quality assurance and conflict handling
- Merge, deduplication, and cross-file association helpers
- Evidence, relationship, and graph-related compilation support
- Skills, capsules, harness, and rigor support that are part of the compiler runtime
- Schema and contract support for the compiler boundary

## What does not belong here

- General application orchestration that belongs in `src/application/services`
- Workflow entry points that belong in `src/application/use_cases`
- Shared primitives and low-level contracts that belong in `core`
- Duplicate parallel pipelines that do not serve the canonical runtime path

## Key modules

- `services/unified_compilation_pipeline.py` — canonical pipeline orchestration
- `services/true_unified_extractor.py` — primary extraction engine
- `services/entity_dedup_service.py` — entity deduplication
- `services/cross_file_association_service.py` — cross-file association
- `services/quality_assurance/` — conflict detection, quality rules, and review helpers
- `services/merge/` — merge and continuity helpers
- `services/extraction/` — extraction-specific helpers
- `services/observability/` — tracing, batching, and telemetry helpers

## Boundary rules

- Keep the canonical runtime path stable
- Avoid adding parallel entry points unless they replace obsolete ones
- Keep canonicalization and validation behavior centralized
- Prefer compatibility shims only when they protect the stable runtime path
- Trim obsolete compatibility only when the canonical path remains unchanged

## Refactor notes

- Phase 0 of the roadmap is to freeze and document the current runtime boundary
- The compiler boundary has been tightened and should now be stabilized
- Remaining cleanup should focus on implicit dependencies, obsolete compatibility, and contract clarity
- Do not expand the surface area unless a change clearly improves the canonical runtime path

## Ownership / stability

- Canonical runtime boundary
- Stable by default; changes should be deliberate and compatibility-aware
- Treat this folder as the canonical compile-and-extract runtime while the rest of the repo is reorganized
