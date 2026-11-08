# `core`

This folder contains shared primitives and stable contracts used across the repository.

## Purpose

- Provide low-level shared utilities and contracts
- Keep reusable primitives separate from application orchestration
- Support the compiler, application layer, and infrastructure with a narrow stable base
- Avoid turning shared primitives into a second application layer

## What belongs here

- Events and event consumers
- Error types and error helpers
- Schema registry and relationship rule loading
- JSON / file helpers
- Provenance and ontology packet primitives
- Other broadly reusable, low-level building blocks

## What does not belong here

- Application orchestration
- Workflow entry points
- RAG-specific indexing / retrieval / answering logic
- Feature-specific service code
- Infrastructure integration logic that is not a shared primitive

## Key modules

- `events/` — event primitives and consumers
- `errors/` — shared error types
- `relationship_rules_loader.py` — rule loading for relationship constraints
- `schema_registry.py` — schema versioning and shared contract support
- `provenance_tracker.py` — provenance helpers
- `ontology_packet.py` — packet-level shared representation helpers
- `utils/` — shared utilities such as JSON and path helpers
- `pipeline_phase.py` — shared pipeline phase definitions

## Boundary rules

- Keep this folder narrow and stable
- Add items here only when they are genuinely shared and broadly reusable
- If logic needs orchestration, coordination, or workflow decisions, it belongs elsewhere
- If a helper is only useful for one application area, keep it near that area instead of moving it into `core`

## Refactor notes

- Phase 5 of the roadmap is to protect `core/` and keep it stable while the application layer is decomposed
- Do not expand this folder during application refactors unless a new primitive clearly belongs here
- Treat this folder as shared-contract infrastructure, not a feature workspace

## Ownership / stability

- Stable shared-contract layer
- Keep it small, explicit, and broadly reusable
