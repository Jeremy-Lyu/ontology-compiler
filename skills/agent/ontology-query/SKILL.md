---
name: ontology-query
description: >-
  Ask questions against the governed semantic layer with answerability
  preflight and evidence. Not for compilation or publish.
version: "0.1.0"
tags: [ontology, query, rag, answerability]
---

# Ontology query

## Purpose

Answer descriptive questions grounded in the governed packet. Prefer
explicit gaps over invented joins.

## Instructions

1. Confirm readiness (`ontology-setup`) if `latest_governed_packet.json` is missing.
2. **Preflight**: `python scripts/rag_cli.py meaning answerable "<question>"`
   - Exit 3 / `answerable=false` → report `gaps`, `missing_concepts`,
     `missing_relation_types`; stop.
3. Optional grounding: `meaning search` / `meaning get` for the matched ids.
4. Answer: `python scripts/rag_cli.py query "<question>"`
   - Default runs the same answerability gate; use
     `--skip-answerable-check` only when the user explicitly overrides.
5. Validate: cite evidence / entity ids from the response; if prose contradicts
   packet relations, say so.

## Validation states

| State | Meaning |
|-------|---------|
| answerable + evidence | Prefer this |
| answerable + low confidence | Answer with caveat |
| not answerable | Return gaps only |
| forced skip | Mark answer as unchecked against semantic coverage |

## Sources of truth

- `src/application/rag/semantic_surface.py`
- `src/application/rag/rag_answerer.py`
- Governed packet under `outputs/runs/ontology/asset_packets/governed/`
