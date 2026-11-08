---
name: ontology-setup
description: >-
  Bring up this ontology compiler: env, compile, governed packet build, and
  readiness checks. Not for modeling edits or answering questions.
version: "0.1.0"
tags: [ontology, setup, governed-packet]
---

# Ontology setup

## Purpose

Install / configure / compile / publish retrieval assets until the semantic
layer is ready to answer. Stop when a step lacks product support — return an
explicit gap, do not invent paths.

## Prerequisites

- Python env with project deps (prefer `.venv`)
- LLM credentials via env (see `src/infrastructure/config/model_roles.py`)

## Instructions

1. Health: `python scripts/ontology_cli.py health`
2. Compile samples via MAIN pipeline / `scripts/ontology_cli.py compile …`
3. Ensure master has `nodes.json` + `relationships.json`
4. Build governed packet: `python scripts/build_governed_packet.py`
5. Optional fail-closed promote: add `--require-publishable`
6. Verify: `python scripts/check_governed_publish.py --require-receipt`
7. Hand off to `ontology-management` for meaning inspection, or
   `ontology-query` for questions.

## Sources of truth

- `docs/DOCS_INDEX.md`, `Makefile` (`make check`, `make governed-publish`)
- `scripts/build_governed_packet.py`, `scripts/check_governed_publish.py`

## Limitations

Does not write glossary terms. Does not answer business questions. Does not
bypass publish gate with fabricated receipts.
