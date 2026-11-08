"""Offline (LLM-free) coverage check for the deterministic domain safety net.

Feeds each regression sample's extracted document text into
``ensure_deterministic_domain_entities`` / ``append_deterministic_domain_relations``
(using a neutral synthetic node whose evidence carries the full text, mirroring
how chunk fragments carry content in the real pipeline) and reports which golden
entities/relations are hit — without any LLM call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ontology_compiler.agents.file_parser import FileParser
from scripts.golden_semantic_eval import _name_match
from src.pipeline.stages.domain_safety_net import (
    append_deterministic_domain_relations,
    ensure_deterministic_domain_entities,
)

_BINARY_EXTS = {".pptx", ".ppt", ".docx", ".doc", ".pdf", ".xlsx", ".xls", ".csv"}
_TEXT_EXTS = {".md", ".txt", ".sql", ".json"}


def _extract_text(path: Path) -> str:
    if path.is_dir():
        parts = []
        for ext in _BINARY_EXTS | _TEXT_EXTS:
            for f in sorted(path.rglob(f"*{ext}")):
                try:
                    parts.append(FileParser.extract_text(str(f)))
                except Exception as e:  # noqa: BLE001
                    print(f"    [warn] {f.name}: {e}")
        return "\n".join(p for p in parts if p)
    if path.suffix.lower() in _BINARY_EXTS:
        return FileParser.extract_text(str(path))
    return path.read_text(encoding="utf-8", errors="replace")


def _golden_entity_names(golden: dict | None) -> list[str]:
    if not golden:
        return []
    return [str(e.get("name", "")).strip() for e in golden.get("entities", []) if e.get("name")]


def _golden_relations(golden: dict | None) -> list[tuple[str, str, list[str]]]:
    if not golden:
        return []
    out = []
    for r in golden.get("relations", []) or []:
        out.append((str(r.get("source", "")).strip(), str(r.get("target", "")).strip(), list(r.get("types", []) or [])))
    return out


def _hit(golden_names: list[str], obs_names: list[str]) -> tuple[list[str], list[str]]:
    matched = []
    missing = []
    for g in golden_names:
        if any(_name_match(g, o) for o in obs_names):
            matched.append(g)
        else:
            missing.append(g)
    return matched, missing


def main() -> None:
    manifest_path = REPO_ROOT / "config" / "regression_samples.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = [s for g in manifest.get("sample_groups", []) for s in g.get("samples", [])]

    print(f"{'sample_id':<28} {'golden_hit':<24} {'missing':<28} rel")
    print("-" * 100)
    total_hit = total_golden = 0
    for s in samples:
        sid = s.get("sample_id", "?")
        input_rel = s.get("input_path", "")
        golden = s.get("golden_semantic") or {}
        g_entities = _golden_entity_names(golden)
        g_relations = _golden_relations(golden)

        path = REPO_ROOT / input_rel
        if not path.exists():
            print(f"{sid:<28} INPUT MISSING: {input_rel}")
            continue

        text = _extract_text(path)
        # Neutral synthetic node: triggers must come from document content only.
        nodes = [{"name": "Document", "meta_name": "document", "description": "", "evidence": {"content": text}}]
        nodes = ensure_deterministic_domain_entities(nodes)
        added = [
            str(n.get("name", "")).strip()
            for n in nodes
            if n.get("provenance", {}).get("origin") == "deterministic_domain_entity"
        ]

        matched, missing = _hit(g_entities, [n["name"] for n in nodes])

        rel_matched = rel_missing = 0
        rel_detail = []
        for src, tgt, types in g_relations:
            src_ok = any(_name_match(src, o) for o in [n["name"] for n in nodes])
            tgt_ok = any(_name_match(tgt, o) for o in [n["name"] for n in nodes])
            if src_ok and tgt_ok:
                rels = append_deterministic_domain_relations(nodes, [], str(path), "offline-verify")
                rel_types = {str(r.get("type", "")).upper() for r in rels}
                hit_types = [t for t in types if t.upper() in rel_types]
                if hit_types:
                    rel_matched += 1
                    rel_detail.append(f"{src}->{tgt}:{','.join(hit_types)}")
                else:
                    rel_missing += 1
            else:
                rel_missing += 1

        total_hit += len(matched)
        total_golden += len(g_entities)
        print(f"{sid:<28} {str(matched):<24} {str(missing):<28} {rel_matched}/{rel_missing} {','.join(rel_detail)}")
        if missing:
            print(f"    added canonical: {added}")

    print("-" * 100)
    print(f"OVERALL golden-entity hit: {total_hit}/{total_golden}")
    if total_golden == total_hit:
        print("ALL GOLDEN ENTITIES COVERED BY DETERMINISTIC SAFETY NET (offline, LLM-free)")


if __name__ == "__main__":
    main()
