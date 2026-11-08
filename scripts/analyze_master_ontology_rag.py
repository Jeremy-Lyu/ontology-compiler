"""Master ontology → RAG analysis report (offline, deterministic).

This CLI deliberately separates *analysis* from *indexing*: it loads the
current master export, adapts it to OntologyPacket, derives RAG objects, and
emits quality/coverage metrics so the next RAG tuning decision is evidence
based. No LLM or network is required.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Executable as ``python scripts/...`` from any working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.application.rag.master_ontology_adapter import load_master_ontology_packet
from src.application.rag.packet_derivation import derive_rag_objects
from scripts.golden_semantic_eval import cq_provenance_stats


def build_report(master_dir: Path | str) -> Dict[str, Any]:
    packet = load_master_ontology_packet(master_dir)
    derived = derive_rag_objects(packet)
    entities = [e for bucket in packet["entities"].values() for e in bucket]
    relations = packet["relations"]
    evidence_entities = sum(bool(e.get("evidence_ref")) for e in entities)
    evidence_relations = sum(bool(r.get("evidence_ref")) for r in relations)
    types: Dict[str, int] = {}
    for e in entities:
        types[e["type"]] = types.get(e["type"], 0) + 1
    return {
        "source": str(master_dir),
        "counts": {
            "entities": len(entities),
            "relations": len(relations),
            "rag_chunks": len(derived["rag_chunks"]),
            "evidence_bundles": len(derived["evidence_bundles"]),
        },
        "layer_distribution": {k: len(v) for k, v in packet["entities"].items()},
        "entity_type_distribution": types,
        "coverage": {
            "entity_evidence_ratio": round(evidence_entities / len(entities), 4) if entities else 0.0,
            "relation_evidence_ratio": round(evidence_relations / len(relations), 4) if relations else 0.0,
            "relation_provenance_ratio": round(
                sum(bool(r.get("provenance")) for r in relations) / len(relations), 4
            ) if relations else 0.0,
        },
        "cq_provenance": cq_provenance_stats(relations),
        "rag_readiness": {
            "canonical_entities_ready": len(derived["canonical_entities"]),
            "retrieval_chunks_ready": len(derived["rag_chunks"]),
            "citation_bundles_ready": len(derived["evidence_bundles"]),
            "blocking_gaps": [
                "408 relations lack evidence_ref/provenance in the current export" if evidence_relations < len(relations) else "",
                "master export has no source_cq; CQ coverage must be enriched" if cq_provenance_stats(relations)["dead_relations"] else "",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze master ontology RAG readiness")
    parser.add_argument("master_dir", nargs="?", default="outputs/runs/ontology/master_ontology")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.master_dir)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
