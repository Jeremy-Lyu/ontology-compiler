---
name: ontology-management
description: >-
  Inspect meaning, publish gates, certification, and receipts for governed
  packets. Not deployment and not free-form Q&A.
version: "0.1.0"
tags: [ontology, governance, publish, certification]
---

# Ontology management

## Purpose

Inspect the semantic layer and gate publication. Treat a successful command as
*storage succeeded*, not *business meaning certified*.

## Instructions

1. Discover meaning via the narrow surface only:
   - `python scripts/rag_cli.py meaning search "<term>"`
   - `python scripts/rag_cli.py meaning get <entity_id>`
2. Resolve the contract before proposing changes: layer (L0–L3), type,
   evidence, relation whitelist (`config/ontology/relationship_rules.yaml`).
3. State the proposed change to the user; do not silently rewrite entities.
4. Publish path: `python scripts/build_governed_packet.py`
   - Review `publish_receipt.json` (`publishable`, `errors`, `stats`)
   - `--require-publishable` refuses to move `latest` on gate failure
   - `--certified` only when a human confirms gate-clean meaning
5. Verify: `python scripts/check_governed_publish.py --require-receipt`
   (add `--require-publishable` in CI publish ranks).

## Fail closed

- Instance-like part codes on L0/L1 → not publishable
- Empty required layers L0/L1/L2 → not publishable
- Missing receipt / checksum drift → fail check
- Never set `certified=true` when `errors` is non-empty

## Sources of truth

- `src/application/rag/publish_gate.py`
- `src/application/rag/semantic_surface.py`
- `docs/ENTITY_CLASSIFICATION_SPEC.md`
