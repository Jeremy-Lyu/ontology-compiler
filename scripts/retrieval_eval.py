"""RAG retrieval golden evaluation (Phase B, 2026-08-29).

Deterministic retrieval-quality gate. It:

1. Derives a retrievable corpus from the CURRENT master
   (``master_ontology_adapter.load_master_ontology_packet`` +
   ``packet_derivation.derive_rag_objects``), so evaluation always runs on the
   authoritative, up-to-date library instead of the stale legacy chunks.jsonl.
2. Derives a golden retrieval eval set from ``config/regression_samples.json``
   golden entities (canonical name + aliases) — LLM-free and network-free.
3. Measures Hit@K / Recall@K / MRR (K in {1,3,5,10}) for two retrievers:
   - BM25 (``retriever.retrieve_bm25``)
   - Local TF-IDF vector (``VectorIndexManager`` + ``LocalEmbedder``)
4. Emits a JSON report + markdown summary under
   ``outputs/runs/ontology/rag/eval/`` so the metrics can later be wired into
   the regression baseline (RAG Recall@K gate).

Relevance rule: a golden entity is "relevant" to every master-derived chunk
whose title matches the canonical name or any alias (normalized equality or
substring containment). Golden entities not present in the current master are
reported as coverage gaps (not scored).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.application.rag.chunk_indexing import (
    ChunkRecord,
    build_inverted_index,
    compute_idf,
    tokenize,
)
from src.application.rag.infrastructure.embedding import LocalEmbedder
from src.application.rag.infrastructure.vector_index_manager import VectorIndexManager
from src.application.rag.master_ontology_adapter import load_master_ontology_packet
from src.application.rag.packet_derivation import derive_rag_objects
from src.application.rag.retriever import retrieve_bm25
from src.application.rag.structured_index import StructuredIndex, EntityIndex, RelationIndex, EvidenceIndex

# --- defaults ----------------------------------------------------------------
MASTER_DIR = ROOT / "outputs" / "runs" / "ontology" / "master_ontology"
CONFIG_PATH = ROOT / "config" / "regression_samples.json"
EVAL_DIR = ROOT / "outputs" / "runs" / "ontology" / "rag" / "eval"
# VectorIndexManager appends "vector_index" to the given rag_dir base.
VECTOR_INDEX_RAG_DIR = EVAL_DIR
KS = (1, 3, 5, 10)
MAX_ALIAS_QUERIES = 3

# --- formal retrieval gate thresholds (2026-08-29, post 分词+双语对齐) ----------
# BM25: 词面召回底限；VECTOR: 语义补充。均要求 Hit@5>=0.9，MRR 保证首位命中质量。
# Scored queries 底线：防止 golden 评估集收缩时门禁假阴性。
RETRIEVAL_GATE: Dict[str, Dict[str, float]] = {
    "bm25": {"hit@5": 0.90, "mrr": 0.80},
    "vector": {"hit@5": 0.90, "mrr": 0.85},
    "structured": {"hit@5": 0.85, "mrr": 0.75},
    "graph": {"hit@5": 0.80, "mrr": 0.70},
}
RETRIEVAL_GATE_MIN_SCORED_QUERIES = 60


def evaluate_gate(
    retrievers_summary: Dict[str, Dict[str, float]], *, scored_queries: int
) -> Dict[str, Any]:
    """Apply RETRIEVAL_GATE thresholds; return {checks, status, passed}."""
    checks: List[Dict[str, Any]] = []
    overall = True
    for retriever, limits in RETRIEVAL_GATE.items():
        actual = retrievers_summary.get(retriever) or {}
        for metric, min_value in limits.items():
            value = float(actual.get(metric, 0.0))
            passed = value >= min_value
            checks.append({
                "retriever": retriever,
                "metric": metric,
                "min": min_value,
                "value": value,
                "passed": passed,
            })
            if not passed:
                overall = False
    coverage_check = {
        "retriever": "*coverage*",
        "metric": "scored_queries",
        "min": RETRIEVAL_GATE_MIN_SCORED_QUERIES,
        "value": scored_queries,
        "passed": scored_queries >= RETRIEVAL_GATE_MIN_SCORED_QUERIES,
    }
    checks.append(coverage_check)
    if not coverage_check["passed"]:
        overall = False
    return {"checks": checks, "status": "PASS" if overall else "FAIL", "passed": overall}


def _norm(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "", (text or "").lower())


def _title_matches(title: str, terms: Sequence[str]) -> bool:
    nt = _norm(title)
    if not nt:
        return False
    for term in terms:
        n = _norm(term)
        if not n:
            continue
        if nt == n:
            return True
        if len(n) >= 2 and n in nt:
            return True
        if len(nt) >= 2 and nt in n:
            return True
    return False


# --- corpus derivation --------------------------------------------------------
def build_corpus(master_dir: Path) -> Tuple[List[ChunkRecord], List[Dict[str, Any]]]:
    """Derive retrievable ChunkRecords from the current master library."""
    packet = load_master_ontology_packet(master_dir)
    derived = derive_rag_objects(packet)
    chunks: List[ChunkRecord] = []
    for c in derived["rag_chunks"]:
        title = str(c.get("title") or "")
        summary = str(c.get("summary") or "")
        keywords = list(c.get("keywords") or [])
        parent_path = list(c.get("parent_path") or [])
        content = "\n".join(
            [title, summary, *keywords, *parent_path]
        ).strip()
        chunks.append(
            ChunkRecord(
                chunk_id=c.get("chunk_id") or f"ragc:{c.get('entity_id')}",
                chunk_type=str(c.get("type") or "entity"),
                content=content,
                node_ids=[str(c.get("entity_id") or "")],
                linked_node_ids=[],
                meta={
                    "layer": str(c.get("layer") or ""),
                    "type": str(c.get("type") or ""),
                    "title": title,
                    "source_file": str(c.get("source_file") or ""),
                    "parent_path": parent_path,
                },
            )
        )
    return chunks, derived["rag_chunks"]


# --- golden retrieval eval set -------------------------------------------------
def load_golden(config_path: Path) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Return (sample_id, group_id, entity) for every golden entity."""
    data = json.loads(config_path.read_text(encoding="utf-8"))
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    for group in data.get("sample_groups", []):
        gid = group.get("group_id", "")
        for sample in group.get("samples", []):
            sid = sample.get("sample_id", "")
            golden = sample.get("golden_semantic") or {}
            for ent in golden.get("entities", []):
                if isinstance(ent, dict) and ent.get("name"):
                    out.append((sid, gid, ent))
    return out


def map_relevance(
    chunks: Sequence[ChunkRecord],
    golden: Sequence[Tuple[str, str, Dict[str, Any]]],
) -> Tuple[Dict[str, List[str]], Dict[str, str], List[Dict[str, Any]]]:
    """Map golden entity key -> relevant chunk_ids; report coverage gaps.

    Returns (relevant_by_key, title_by_chunk_id, gaps). key = f"{sample}|{name}".
    """
    title_by_chunk: Dict[str, str] = {}
    for c in chunks:
        title_by_chunk[c.chunk_id] = str(c.meta.get("title") or "")

    relevant: Dict[str, List[str]] = {}
    gaps: List[Dict[str, Any]] = []
    for sample, group, ent in golden:
        name = str(ent.get("name") or "")
        terms = [name, *list(ent.get("aliases") or [])]
        matched = [
            c.chunk_id
            for c in chunks
            if _title_matches(str(c.meta.get("title") or ""), terms)
        ]
        key = f"{sample}|{name}"
        if not matched:
            gaps.append({"sample_id": sample, "group_id": group, "name": name})
        else:
            relevant[key] = sorted(matched)
    return relevant, title_by_chunk, gaps


def build_queries(
    golden: Sequence[Tuple[str, str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Expand golden entities into concrete query items (name + top aliases)."""
    queries: List[Dict[str, Any]] = []
    for sample, group, ent in golden:
        name = str(ent.get("name") or "")
        aliases = [a for a in (ent.get("aliases") or []) if str(a).strip()]
        key = f"{sample}|{name}"
        query_texts = [name] + [str(a) for a in aliases[:MAX_ALIAS_QUERIES]]
        for q in query_texts:
            queries.append(
                {
                    "sample_id": sample,
                    "group_id": group,
                    "entity_name": name,
                    "key": key,
                    "query": q,
                    "layer": str(ent.get("layer") or ""),
                    "meta_class": str(ent.get("meta_class") or ""),
                }
            )
    return queries


# --- metrics --------------------------------------------------------------------
def _hit_recall_mrr(
    retrieved: Sequence[Tuple[str, float]], relevant: Set[str], k: int
) -> Tuple[bool, float, float]:
    top_ids = [rid for rid, _ in retrieved[:k]]
    hit = any(rid in relevant for rid in top_ids)
    recall = len([rid for rid in top_ids if rid in relevant]) / len(relevant) if relevant else 0.0
    mrr = 0.0
    for rank, (rid, _) in enumerate(retrieved, start=1):
        if rid in relevant:
            mrr = 1.0 / rank
            break
    return hit, recall, mrr


def aggregate(
    results: Sequence[Dict[str, Any]], prefix: str
) -> Dict[str, Any]:
    """Aggregate per-query metrics into per-K summary for one retriever."""
    n = len(results)
    if n == 0:
        return {"queries": 0}
    out: Dict[str, Any] = {"queries": n}
    for k in KS:
        hits = sum(1 for r in results if r[f"{prefix}_hit@{k}"])
        recalls = [r[f"{prefix}_recall@{k}"] for r in results]
        out[f"hit@{k}"] = round(hits / n, 4)
        out[f"recall@{k}"] = round(sum(recalls) / n, 4)
    out["mrr"] = round(sum(r[f"{prefix}_mrr"] for r in results) / n, 4)
    return out


# --- retrievers -----------------------------------------------------------------
def bm25_retriever(chunks: Sequence[ChunkRecord]) -> Any:
    inverted = build_inverted_index(chunks)
    idf = compute_idf(inverted, len(chunks))

    def run(query: str, topk: int) -> List[Tuple[str, float]]:
        hits = retrieve_bm25(query, list(chunks), inverted, idf, topk=topk)
        return [(h.chunk.chunk_id, h.score) for h in hits]

    return run


def vector_retriever(chunks: Sequence[ChunkRecord], rag_dir: Path) -> Any:
    rag_dir.mkdir(parents=True, exist_ok=True)
    # Vocab must cover the corpus bigram vocabulary: a small dimension (384)
    # truncates rare domain terms (e.g. "库存") out of the top-k vocab and
    # zeroes the query vector, producing an identical hit set for every
    # Chinese query. 4096 covers the full bigram vocab of the ~1.3k corpus.
    dim = 4096
    manager = VectorIndexManager(rag_dir, embedder=LocalEmbedder(dimension=dim), dimension=dim)
    manager.build_index(list(chunks), force_rebuild=True)

    def run(query: str, topk: int) -> List[Tuple[str, float]]:
        return manager.compute_similarity(query, topk=topk)

    return run


def structured_retriever(chunks: Sequence[ChunkRecord]) -> Any:
    """Create a structured retriever for entity/relation/evidence search"""
    # Build structured index from chunks
    structured_index = StructuredIndex()
    
    # Extract entities and relations from chunks (simplified for evaluation)
    # In real usage, this would load from a pre-built structured index
    def run(query: str, topk: int) -> List[Tuple[str, float]]:
        # First search entities
        entities = structured_index.search_entities(query=query, limit=topk)
        results = []
        for entity in entities:
            if hasattr(entity, 'entity_id') and hasattr(entity, 'confidence'):
                results.append((entity.entity_id, entity.confidence))
        
        # Then search relations
        relations = structured_index.search_relations(query=query, limit=topk)
        for relation, from_name, to_name in relations:
            if hasattr(relation, 'relation_id') and hasattr(relation, 'confidence'):
                results.append((relation.relation_id, relation.confidence))
        
        # Then search evidences
        evidences = structured_index.search_evidences(query=query, limit=topk)
        for evidence in evidences:
            if hasattr(evidence, 'evidence_id') and hasattr(evidence, 'confidence'):
                results.append((evidence.evidence_id, evidence.confidence))
        
        # Sort by score descending
        results.sort(key=lambda x: -x[1])
        return results[:topk]
    
    return run


def graph_retriever(chunks: Sequence[ChunkRecord]) -> Any:
    """Create a graph retriever for entity/relation search (placeholder)"""
    # In real usage, this would connect to a graph database like Neo4j
    def run(query: str, topk: int) -> List[Tuple[str, float]]:
        # Return empty results for now (placeholder implementation)
        return []
    
    return run


# --- main -----------------------------------------------------------------------
def run_eval(
    master_dir: Path,
    config_path: Path,
    index_dir: Path,
) -> Dict[str, Any]:
    chunks, _ = build_corpus(master_dir)
    golden = load_golden(config_path)
    relevant, title_by_chunk, gaps = map_relevance(chunks, golden)
    queries = build_queries(golden)

    retrievers = {
        "bm25": bm25_retriever(chunks),
        "vector": vector_retriever(chunks, index_dir),
        "structured": structured_retriever(chunks),
        "graph": graph_retriever(chunks),
    }

    per_query: List[Dict[str, Any]] = []
    per_retriever_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    scored_keys = set(relevant.keys())
    for q in queries:
        if q["key"] not in scored_keys:
            continue  # no relevant chunk in master -> unscoreable, already in gaps
        rel = set(relevant[q["key"]])
        row: Dict[str, Any] = {
            "sample_id": q["sample_id"],
            "entity_name": q["entity_name"],
            "query": q["query"],
            "relevant": sorted(rel),
        }
        for rname, run in retrievers.items():
            retrieved = run(q["query"], topk=max(KS))
            row[f"retrieved_{rname}"] = [
                {"chunk_id": cid, "score": round(score, 4), "title": title_by_chunk.get(cid, "")}
                for cid, score in retrieved[:5]
            ]
            for k in KS:
                hit, recall, mrr = _hit_recall_mrr(retrieved, rel, k)
                row[f"{rname}_hit@{k}"] = hit
                row[f"{rname}_recall@{k}"] = recall
                row[f"{rname}_mrr"] = mrr
            per_retriever_results[rname].append(row)
        per_query.append(row)

    summary: Dict[str, Any] = {"retrievers": {}}
    for rname, results in per_retriever_results.items():
        agg = aggregate(results, rname)
        # Normalize key names for the JSON summary block
        summary["retrievers"][rname] = {
            "queries": agg["queries"],
            "hit@1": agg["hit@1"],
            "hit@3": agg["hit@3"],
            "hit@5": agg["hit@5"],
            "hit@10": agg["hit@10"],
            "recall@1": agg["recall@1"],
            "recall@3": agg["recall@3"],
            "recall@5": agg["recall@5"],
            "recall@10": agg["recall@10"],
            "mrr": agg["mrr"],
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "master_dir": str(master_dir),
        "corpus": {"chunk_count": len(chunks), "source": "master_ontology_adapter+packet_derivation"},
        "golden": {
            "entities_total": len(golden),
            "queries_total": len(queries),
            "scored_entities": len(scored_keys),
            "scored_queries": len(per_query),
            "not_in_master_count": len(gaps),
            "not_in_master": gaps,
        },
        "retrievers_summary": summary["retrievers"],
        "per_query": per_query,
        "gate": evaluate_gate(summary["retrievers"], scored_queries=len(per_query)),
    }


def format_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# RAG Retrieval Golden Evaluation (Phase B)")
    lines.append("")
    lines.append(f"- 生成时间：{report['generated_at']}")
    lines.append(f"- 语料：{report['master_dir']}（当前 master，adapter 派生）")
    lines.append(f"- chunk 数：{report['corpus']['chunk_count']}")
    g = report["golden"]
    lines.append(
        f"- golden：实体 {g['entities_total']} 个，查询 {g['scored_queries']} 条"
        f"（scored {g['scored_entities']} 实体 / 未入 master {g['not_in_master_count']} 个）"
    )
    lines.append("")
    lines.append("| retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for rname in ["bm25", "vector", "structured", "graph"]:
        if rname in report["retrievers_summary"]:
            s = report["retrievers_summary"][rname]
            lines.append(
                f"| {rname} | {s['hit@1']} | {s['hit@3']} | {s['hit@5']} | {s['hit@10']} | "
                f"{s['recall@1']} | {s['recall@3']} | {s['recall@5']} | {s['recall@10']} | {s['mrr']} |"
            )
    lines.append("")
    gate = report.get("gate") or {}
    lines.append(f"## Retrieval Gate: **{gate.get('status', 'N/A')}**")
    lines.append("")
    lines.append("| retriever | metric | min | value | status |")
    lines.append("|---|---|---|---|---|")
    for c in gate.get("checks", []) or []:
        lines.append(
            f"| {c['retriever']} | {c['metric']} | {c['min']} | {c['value']} | "
            f"{'PASS' if c['passed'] else 'FAIL'} |"
        )
    lines.append("")
    if g["not_in_master"]:
        lines.append("## golden 未出现在当前 master（覆盖缺口，不计分）")
        lines.append("")
        for gap in g["not_in_master"]:
            lines.append(f"- {gap['sample_id']} :: {gap['name']}")
        lines.append("")
    lines.append("## 逐查询明细（top-5）")
    lines.append("")
    for q in report["per_query"]:
        lines.append(f"### {q['sample_id']} :: {q['query']}  (relevant={q['relevant']})")
        for rname in ("bm25", "vector"):
            lines.append(f"- **{rname}** retrieved: {[r['chunk_id'] for r in q.get(f'retrieved_{rname}', [])]}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master-dir", type=Path, default=MASTER_DIR)
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Base directory passed to VectorIndexManager (appends /vector_index internally).\n"
             "Default = scratch dir under outputs/runs/ontology/rag/eval/ (never the governed\n"
             "packet dir, which is protected by the manifest sha256 contract).",
    )
    ap.add_argument("--out", type=Path, default=None, help="JSON report output path")
    ap.add_argument(
        "--enforce-gate",
        action="store_true",
        help="Return exit code 1 if retrieval gate FAILS (CI/CD integration, regression hard gate).",
    )
    args = ap.parse_args()

    # Default: the eval vector index is a *scratch artifact* derived from the
    # current master, NOT a governed-packet member. It must never be written
    # into the governed packet directory — doing so corrupts the manifest
    # sha256 chain and raises DataContractError on next consumption
    # (regression 2026-09-01, GOV-20260901-101303 checksum mismatch).
    # Scratch base lives under the eval dir; VectorIndexManager appends
    # "vector_index" internally.
    if args.index_dir is not None:
        index_dir = args.index_dir
    else:
        index_dir = VECTOR_INDEX_RAG_DIR / "scratch_vector_index"

    out_dir = EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run_eval(args.master_dir, args.config, index_dir)

    out_path = args.out or (out_dir / f"retrieval_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = out_dir / "retrieval_eval_latest.json"
    latest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = out_dir / f"retrieval_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    md_path.write_text(format_markdown(report), encoding="utf-8")

    print(format_markdown(report))
    print(f"\n[JSON] {out_path}")
    print(f"[MD]   {md_path}")
    print(f"[VECTOR INDEX] {index_dir / 'vector_index'}")
    gate_passed = bool((report.get("gate") or {}).get("passed"))
    if args.enforce_gate:
        print(f"[Gate] enforce-gate ENABLED -> {'exit 0 (PASS)' if gate_passed else 'exit 1 (FAIL)'}")
        return 0 if gate_passed else 1
    print(f"[Gate] enforce-gate OFF -> exit 0 (gate_status={gate_passed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
