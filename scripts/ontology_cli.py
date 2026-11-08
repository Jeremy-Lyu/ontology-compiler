#!/usr/bin/env python3
"""Unified CLI for Ontology Compiler - All-in-one entry point.

This CLI provides:
1. Pipeline execution (compile documents to ontology)
2. RAG query (search and ask questions against ontology)
3. RAG build (build RAG index from ontology outputs)
4. Validate (schema and architecture validation)
5. Analyze (output analysis and layer consistency)
6. Health check
7. Status reporting
8. Cache management

Usage:
    # Compile documents to ontology
    python scripts/ontology_cli.py compile --folder /path/to/docs --domain AUTO
    
    # Query ontology with RAG
    python scripts/ontology_cli.py query "What is the Order Replication Process?"
    
    # Build RAG index
    python scripts/ontology_cli.py rag build --ontology-dir outputs/ontology
    
    # Validate schemas
    python scripts/ontology_cli.py validate schemas
    
    # Validate architecture
    python scripts/ontology_cli.py validate architecture
    
    # Analyze pipeline output
    python scripts/ontology_cli.py analyze output --run-dir outputs/runs/ontology/pipeline
    
    # Check layer consistency
    python scripts/ontology_cli.py analyze layers
    
    # Check system health
    python scripts/ontology_cli.py health
    
    # Show pipeline status
    python scripts/ontology_cli.py status
    
    # Clear document cache
    python scripts/ontology_cli.py cache --clear
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Pipeline Compilation
# ---------------------------------------------------------------------------

def _precreate_socketpair(retries: int = 20):
    """Pre-create a connected socket pair for asyncio's self-pipe.

    Winsock LSP corruption on Windows breaks ``socket.socketpair()``
    non-deterministically with ``WinError 10038``. We first try
    ``socketpair()`` with retries; if all retries fail, we fall back to
    a TCP loopback pair (``AF_INET`` sockets are not affected by the LSP).
    """
    import socket as _sock
    import time as _time

    # Strategy 1: retry socket.socketpair()
    for i in range(retries):
        try:
            return _sock.socketpair()
        except OSError:
            _time.sleep(0.01 * (i + 1))

    # Strategy 2: TCP loopback fallback (AF_INET sockets work even
    # when socketpair() / AF_UNIX is broken by Winsock LSP)
    for i in range(retries):
        try:
            srv = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM, 0)
            srv.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            cli = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM, 0)
            cli.connect(("127.0.0.1", port))
            conn, _ = srv.accept()
            srv.close()
            return conn, cli
        except OSError:
            _time.sleep(0.01 * (i + 1))

    raise OSError("socket creation failed after %d retries (socketpair + TCP loopback)" % retries)


def _patch_make_self_pipe(pre_ssock=None, pre_csock=None):
    """Patch BaseSelectorEventLoop to avoid socketpair().

    If pre-created sockets are available, use them. Otherwise, skip the
    self-pipe entirely (safe for single-threaded pipelines — the self-pipe
    is only needed for call_soon_threadsafe from other threads).
    """
    import asyncio.selector_events as _se
    import selectors as _selmod
    import select as _selectmod
    import time as _time

    def _make_self_pipe(self):
        if pre_ssock is not None and pre_csock is not None:
            self._ssock = pre_ssock
            self._csock = pre_csock
            self._ssock.setblocking(False)
            self._csock.setblocking(False)
            self._internal_fds += 1
            loop_reader = self._ssock.fileno()
            self._add_reader(loop_reader, self._read_from_self)
        else:
            # No sockets available — skip self-pipe (single-threaded safe)
            self._ssock = None
            self._csock = None

    def _close_self_pipe(self):
        if self._ssock is not None:
            try:
                self._remove_reader(self._ssock.fileno())
            except (AttributeError, OSError):
                pass
        self._ssock = None
        self._csock = None

    def _write_to_self(self):
        if self._csock is not None:
            try:
                self._csock.send(b'\0')
            except OSError:
                pass

    _se.BaseSelectorEventLoop._make_self_pipe = _make_self_pipe
    _se.BaseSelectorEventLoop._close_self_pipe = _close_self_pipe
    _se.BaseSelectorEventLoop._write_to_self = _write_to_self

    # Patch SelectSelector.select to handle empty FD lists.
    # On Windows, select.select() with empty lists returns WinError 10022.
    # When there are no FDs to watch, just sleep for the timeout.
    _SelCls = getattr(_selmod, "SelectSelector", None) or getattr(_selmod, "_SelectSelector", None)
    if _SelCls is not None:
        _orig_select = _SelCls.select

        def _safe_select(self, timeout=None):
            if not self._readers and not self._writers:
                if timeout is not None and timeout > 0:
                    _time.sleep(timeout)
                return []
            try:
                return _orig_select(self, timeout)
            except OSError:
                if timeout is not None and timeout > 0:
                    _time.sleep(timeout)
                return []

        _SelCls.select = _safe_select


def run_compile(args: argparse.Namespace) -> int:
    """Run ontology compilation pipeline using MainPipeline (production mainline)."""
    print(f"[CLI] Starting ontology compilation")
    print(f"[CLI] Source: {args.source}")
    print(f"[CLI] Domain: {args.domain}")

    try:
        from ontology_compiler.config import load_runtime_env
        loaded = load_runtime_env(force_reload=True)
        if loaded:
            print(f"[ENV] Loaded {len(loaded)} vars from config/env.env")
    except Exception as _exc:
        print(f"[ENV-WARN] Could not load environment config: {_exc}", file=sys.stderr)

    # Pre-create socketpair BEFORE asyncio needs it — Winsock LSP corruption
    # makes socketpair() fail non-determinically. If we can't create one,
    # patch _make_self_pipe to skip it entirely (single-threaded safe).
    pre_ssock = pre_csock = None
    try:
        pre_ssock, pre_csock = _precreate_socketpair()
        print(f"[CLI] [OK] Pre-created socketpair for asyncio")
    except OSError as exc:
        print(f"[CLI] [WARN] Could not pre-create socketpair: {exc!r}", file=sys.stderr)
        print(f"[CLI] [WARN] Patching asyncio to skip self-pipe (single-threaded mode)", file=sys.stderr)

    _patch_make_self_pipe(pre_ssock, pre_csock)

    # Use SelectorEventLoop (patched to use pre-created socketpair)
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    return asyncio.run(run_main_pipeline_async(args))


async def run_main_pipeline_async(args: argparse.Namespace) -> int:
    """Run MainPipeline asynchronously."""
    from src.pipeline.pipeline_router import get_production_pipeline

    # ============================================================
    # C-1: LLM HARD BUDGET GATE — apply CLI args BEFORE any LLM call.
    # Default 0 means "no cap", so unconfigured environments are
    # 100% backward compatible. Caps are enforced inside the gateway
    # *before* any provider is contacted, so they never silently
    # accumulate charge.
    # ============================================================
    max_cost_usd = float(getattr(args, "max_cost_usd", 0.0) or 0.0)
    max_calls = int(getattr(args, "max_calls", 0) or 0)
    max_total_tokens = int(getattr(args, "max_total_tokens", 0) or 0)
    degraded_offline = bool(getattr(args, "degraded_offline", False))

    budget_any_active = max_cost_usd > 0 or max_calls > 0 or max_total_tokens > 0
    if budget_any_active or degraded_offline:
        try:
            from src.application.bootstrap.llm_gateway import get_llm_gateway
            gateway = get_llm_gateway()
            if budget_any_active:
                gateway.set_budget_policy(
                    max_cost_usd=max_cost_usd,
                    max_calls=max_calls,
                    max_total_tokens=max_total_tokens,
                )
                print(
                    f"[GOV] [BudgetGate] Armed: max_cost=${max_cost_usd:.4f}, "
                    f"max_calls={max_calls}, max_tokens={max_total_tokens}"
                )
            if degraded_offline:
                print("[GOV] [DegradedMode] Forcing DEGRADE mode for LLM-dependent stages")
        except Exception as _exc:  # pragma: no cover - best-effort arming
            print(f"[GOV] [WARN] Could not arm LLM governance: {_exc!r}", file=sys.stderr)

    output_dir = str(PROJECT_ROOT / "outputs" / "runs" / "ontology" / "pipeline")

    context = {
        "source_path": args.source,
        "domain": args.domain,
        "output_dir": output_dir,
        "metadata": {
            "strict_governance": args.strict_governance,
            # C-1: propagate governance config so pipeline stages can
            # read budget / degraded flags without coupling to the CLI.
            "governance": {
                "max_cost_usd": max_cost_usd,
                "max_calls": max_calls,
                "max_total_tokens": max_total_tokens,
                "degraded_offline": degraded_offline,
            },
        },
    }

    pipeline = get_production_pipeline(context=context)
    
    pipeline_start = time.time()
    result = await pipeline.run()
    
    elapsed = round(time.time() - pipeline_start, 2)

    quality_status = getattr(result, "quality_status", None)
    quality_reason = getattr(result, "quality_reason", "")
    validation_passed = getattr(result, "validation_passed", False)
    quality_score = getattr(result, "quality_score", 0.0)
    quality_label = quality_status.value if hasattr(quality_status, "value") else str(quality_status or "unknown")

    if quality_label == "success":
        emoji = "✅"
    elif quality_label == "partial_success":
        emoji = "🟡"
    elif quality_label == "degraded_success":
        emoji = "🟠"
    else:
        emoji = "❌"

    print(f"\n[CLI] {emoji} Pipeline completed in {elapsed}s")
    print(f"[CLI] Status: {result.status.value}")
    print(f"[CLI] Quality status: {quality_label}")
    print(f"[CLI] Validation passed: {validation_passed}")
    print(f"[CLI] Gate / master quality score: {quality_score:.2f}")
    breakdown = {}
    try:
        meta = getattr(getattr(result, "context", None), "metadata", None) or {}
        breakdown = meta.get("quality_breakdown") or {}
    except Exception:
        breakdown = {}
    if breakdown:
        print(
            f"[CLI] File quality (extract): avg={breakdown.get('file_avg_score', 0):.2f} "
            f"min={breakdown.get('file_min_score', 0):.2f} "
            f"max={breakdown.get('file_max_score', 0):.2f} "
            f"files={breakdown.get('file_count', 0)}"
        )
        for row in (breakdown.get("files") or [])[:12]:
            print(
                f"[CLI]   · {row.get('file')}: score={row.get('quality_score')} "
                f"ok={row.get('validation_passed')} status={row.get('extraction_quality_status')}"
            )
        if breakdown.get("file_count", 0) > 12:
            print(f"[CLI]   · … {breakdown['file_count'] - 12} more files")
    if quality_reason:
        print(f"[CLI] Quality reason: {quality_reason}")
    print(f"[CLI] Publishable: {getattr(result, 'is_publishable', False)}")
    print(f"[CLI] Stages: {result.stats.completed_stages}/{result.stats.total_stages} completed")

    stats_dict = result.stats.to_dict() if hasattr(result.stats, 'to_dict') else {}
    for key, value in stats_dict.items():
        if key not in ['completed_stages', 'total_stages']:
            print(f"[CLI]   {key}: {value}")

    if result.errors:
        print(f"\n[CLI] Errors ({len(result.errors)}):")
        for error in result.errors[:5]:
            print(f"[CLI]   - {error}")

    if result.warnings:
        print(f"\n[CLI] Warnings ({len(result.warnings)}):")
        for warning in result.warnings[:5]:
            print(f"[CLI]   - {warning}")

    if quality_label == "success":
        return 0
    if quality_label in {"partial_success", "degraded_success"}:
        return 2
    return 1


# ---------------------------------------------------------------------------
# RAG Query
# ---------------------------------------------------------------------------

# =====================================================================






def run_query(args: argparse.Namespace) -> int:
    """Run RAG query against ontology using UnifiedRAGService (modern RAG stack)."""
    import asyncio
    from pathlib import Path
    from src.application.rag.core.unified_rag_service import UnifiedRAGService
    from src.infrastructure.config.rag_config import RAGPipelineConfig
    
    q = (args.query or "").strip()
    if not q:
        print("[CLI] [❌] Empty query")
        return 2
    print(f"[CLI] Querying ontology: {q}")
    
    # Load RAG configuration
    config = RAGPipelineConfig.from_env()
    
    # Initialize Unified RAG Service
    rag_service = UnifiedRAGService.from_config(config)
    
    try:
        # Execute search
        results = rag_service.search(q, top_k=12)
        
        print()
        print("=" * 80)
        print("  🔎 RAG 检索结果 (实体)")
        print("=" * 80)
        
        if not results:
            print("  (无相关实体，尝试其它关键词：例如 EBOM、MBOM、HERCULES、PLM、Quality 等)")
        else:
            for i, result in enumerate(results, 1):
                entity = result.entity
                score = result.score
                nid = entity.meta_id or "?"
                name = entity.meta_name or entity.name or "?"
                layer = entity.layer or "?"
                mcls = entity.meta_class or "?"
                desc = (entity.description or "").strip()
                if len(desc) > 160:
                    desc = desc[:157] + "..."
                qs = entity.quality_score
                aliases = list(entity.aliases or [])[:4]
                line1 = (
                    f"  [{i:2}] [{layer:4}] {nid:12}  {mcls:16}  {name}"
                )
                print(line1 + f"  score={score:5.1f}" + (f"  qs={qs:.2f}" if isinstance(qs, (int, float)) else ""))
                if aliases:
                    print(f"       Aliases: {', '.join(aliases)}")
                if desc:
                    print(f"       {desc}")
        
        return 0
        
    except Exception as e:
        print(f"[CLI] [❌] RAG query failed: {str(e)}")
        return 1


# ---------------------------------------------------------------------------
# RAG Build
# ---------------------------------------------------------------------------

def _build_chunks_from_tbh(tbh_data: dict) -> list:
    """Build RAG chunks from tbh_supplement.json nodes and relationships."""
    from src.application.rag.chunk_indexing import ChunkRecord
    
    chunks: list[ChunkRecord] = []
    nodes = tbh_data.get("nodes", [])
    relationships = tbh_data.get("relationships", [])
    
    rel_by_node: dict[str, list[str]] = {}
    for rel in relationships:
        src = rel.get("source", "")
        predicate = rel.get("predicate", rel.get("type", ""))
        target = rel.get("target", "")
        rel_by_node.setdefault(src, []).append(f"{predicate} -> {target}")
    
    for node in nodes:
        nid = node.get("meta_id", "")
        name = node.get("meta_name", "")
        meta_class = node.get("meta_class", "")
        meta_subclass = node.get("meta_subclass", "")
        desc = node.get("meta_description", "")
        domain = node.get("sandbox_domain", "")
        
        rels = rel_by_node.get(nid, [])
        rel_lines = "\n".join(f"  - {r}" for r in rels) if rels else "  (none)"
        
        content = (
            f"{meta_class}: {name}\n"
            f"ID: {nid}\n"
            f"Subclass: {meta_subclass}\n"
            f"Domain: {domain}\n"
            f"Description: {desc}\n"
            f"Relationships:\n{rel_lines}"
        ).strip()
        
        chunks.append(
            ChunkRecord(
                chunk_id=f"ONT_NODE_{nid}",
                chunk_type=f"ontology_{meta_class.lower()}",
                content=content,
                node_ids=[nid],
                linked_node_ids=[r.get("target", "") for r in relationships if r.get("source") == nid],
                meta={
                    "meta_class": meta_class,
                    "meta_subclass": meta_subclass,
                    "domain": domain,
                    "source": "tbh_supplement",
                },
            )
        )
    return chunks


def _build_chunks_from_l2(l2_data: dict) -> list:
    """Build RAG chunks from l2_business_scenarios.json."""
    from src.application.rag.chunk_indexing import ChunkRecord
    
    chunks: list[ChunkRecord] = []
    
    for scenario in l2_data.get("business_scenarios", []):
        name = scenario.get("scenario_name", "")
        content = (
            f"Business Scenario: {name}\n"
            f"Description: {scenario.get('description', '')}\n"
            f"Participants: {', '.join(scenario.get('participants', []))}\n"
            f"Triggers: {', '.join(scenario.get('triggers', []))}\n"
            f"Outcomes: {', '.join(scenario.get('outcomes', []))}\n"
            f"Key Metrics: {', '.join(scenario.get('key_metrics', []))}\n"
            f"Source: {scenario.get('source_document', '')}"
        ).strip()
        safe_id = name.upper().replace(" ", "_").replace("-", "_")[:60]
        chunks.append(
            ChunkRecord(
                chunk_id=f"L2_SCENARIO_{safe_id}",
                chunk_type="l2_business_scenario",
                content=content,
                node_ids=[],
                linked_node_ids=[],
                meta={"source_document": scenario.get("source_document", ""), "source": "l2_business_scenarios"},
            )
        )
    
    for logic in l2_data.get("business_logic", []):
        name = logic.get("logic_name", "")
        content = (
            f"Business Logic: {name}\n"
            f"Type: {logic.get('type', '')}\n"
            f"Description: {logic.get('description', '')}\n"
            f"Preconditions: {', '.join(logic.get('preconditions', []))}\n"
            f"Postconditions: {', '.join(logic.get('postconditions', []))}\n"
            f"Source: {logic.get('source_document', '')}"
        ).strip()
        safe_id = name.upper().replace(" ", "_").replace("-", "_")[:60]
        chunks.append(
            ChunkRecord(
                chunk_id=f"L2_LOGIC_{safe_id}",
                chunk_type="l2_business_logic",
                content=content,
                node_ids=[],
                linked_node_ids=[],
                meta={"logic_type": logic.get("type", ""), "source_document": logic.get("source_document", ""), "source": "l2_business_scenarios"},
            )
        )
    
    for definition in l2_data.get("definitions", []):
        term = definition.get("term", "")
        content = (
            f"Definition: {term}\n"
            f"Meaning: {definition.get('definition', '')}\n"
            f"Context: {definition.get('context', '')}\n"
            f"Related Terms: {', '.join(definition.get('related_terms', []))}\n"
            f"Source: {definition.get('source_document', '')}"
        ).strip()
        safe_id = term.upper().replace(" ", "_").replace("-", "_")[:60]
        chunks.append(
            ChunkRecord(
                chunk_id=f"L2_DEF_{safe_id}",
                chunk_type="l2_definition",
                content=content,
                node_ids=[],
                linked_node_ids=[],
                meta={"source_document": definition.get("source_document", ""), "source": "l2_business_scenarios"},
            )
        )
    
    return chunks


def _build_chunks_from_master(master: dict, source_file: str) -> list:
    """Build RAG chunks from master_ontology_v2.json."""
    from src.application.rag.chunk_indexing import ChunkRecord, build_chunks_from_domain_ontology
    
    chunks: list[ChunkRecord] = []
    nodes = master.get("nodes", [])
    edges = master.get("edges", [])
    
    domain_payload = {
        "schema_version": master.get("schema_version", "v1.0"),
        "nodes": nodes,
        "relationships": edges,
    }
    chunks.extend(build_chunks_from_domain_ontology(domain_payload, source_file))
    
    rel_by_node: dict[str, list[str]] = {}
    for e in edges:
        src = e.get("source", "")
        rel_by_node.setdefault(src, []).append(f"{e.get('type', '')} -> {e.get('target', '')}")
    
    for node in nodes:
        nid = node.get("meta_id", "")
        meta_class = node.get("meta_class", "")
        if meta_class in ("Capability",) or not nid:
            continue
        name = node.get("meta_name", "")
        meta_subclass = node.get("meta_subclass", "")
        desc = node.get("meta_description", "")
        domain = node.get("sandbox_domain", "")
        rels = rel_by_node.get(nid, [])
        rel_lines = "\n".join(f"  - {r}" for r in rels[:20]) if rels else "  (none)"
        
        content = (
            f"{meta_class}: {name}\n"
            f"ID: {nid}\n"
            f"Subclass: {meta_subclass}\n"
            f"Domain: {domain}\n"
            f"Description: {desc}\n"
            f"Relationships:\n{rel_lines}"
        ).strip()
        
        chunks.append(
            ChunkRecord(
                chunk_id=f"MASTER_NODE_{nid}",
                chunk_type=f"master_{meta_class.lower()}",
                content=content,
                node_ids=[nid],
                linked_node_ids=[e.get("target", "") for e in edges if e.get("source") == nid],
                meta={"meta_class": meta_class, "meta_subclass": meta_subclass, "domain": domain, "source": source_file},
            )
        )
    
    for scenario in master.get("business_scenarios", []):
        name = scenario.get("scenario_name", scenario.get("name", ""))
        domain = scenario.get("domain", "")
        content = (
            f"Business Scenario: {name}\n"
            f"Domain: {domain}\n"
            f"Description: {scenario.get('description', '')}\n"
            f"Participants: {', '.join(scenario.get('participants', []))}\n"
            f"Triggers: {', '.join(scenario.get('triggers', []))}\n"
            f"Outcomes: {', '.join(scenario.get('outcomes', []))}"
        ).strip()
        safe_id = name.upper().replace(" ", "_").replace("-", "_")[:60]
        chunks.append(
            ChunkRecord(
                chunk_id=f"MASTER_SCENARIO_{safe_id}",
                chunk_type="master_business_scenario",
                content=content,
                node_ids=[],
                linked_node_ids=[],
                meta={"domain": domain, "source": source_file},
            )
        )
    
    return chunks


def run_rag_build(args: argparse.Namespace) -> int:
    """Build RAG index from ontology outputs."""
    from src.application.rag.chunk_indexing import (
        ChunkRecord,
        build_inverted_index,
        compute_idf,
        write_chunks_jsonl,
        write_inverted_index,
        write_idf,
    )
    
    source = args.source
    if source is None:
        if args.run_dir:
            source = "extraction"
        else:
            source = "master"
    
    all_chunks: list[ChunkRecord] = []
    source_meta: dict = {}
    
    if source == "extraction":
        if not args.run_dir:
            print("[CLI] [❌] --run-dir is required for extraction mode")
            return 1
        
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"[CLI] [❌] Run directory not found: {run_dir}")
            return 1
        
        output_dir = Path(args.output_dir) if args.output_dir else run_dir / "rag_phase1"
        
        tbh_path = run_dir / "tbh_supplement.json"
        if tbh_path.exists():
            print("[CLI] Building chunks from tbh_supplement.json...")
            with open(tbh_path, "r", encoding="utf-8") as f:
                tbh_data = json.load(f)
            ont_chunks = _build_chunks_from_tbh(tbh_data)
            print(f"[CLI]   {len(ont_chunks)} ontology node chunks")
            all_chunks.extend(ont_chunks)
        else:
            print("[CLI] [WARN] tbh_supplement.json not found")
        
        l2_path = run_dir / "l2_business_scenarios.json"
        if l2_path.exists():
            print("[CLI] Building chunks from l2_business_scenarios.json...")
            with open(l2_path, "r", encoding="utf-8") as f:
                l2_data = json.load(f)
            l2_chunks = _build_chunks_from_l2(l2_data)
            print(f"[CLI]   {len(l2_chunks)} L2 chunks")
            all_chunks.extend(l2_chunks)
        else:
            print("[CLI] [WARN] l2_business_scenarios.json not found")
        
        source_meta = {"source_run": str(run_dir)}
    
    else:
        ontology_dir = Path(args.ontology_dir) if args.ontology_dir else PROJECT_ROOT / "outputs" / "runs" / "ontology" / "master_ontology"
        if not ontology_dir.exists():
            print(f"[CLI] [❌] Ontology directory not found: {ontology_dir}")
            return 1
        
        output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "runs" / "ontology" / "rag"
        
        sources: list[str] = []
        loaded_any = False
        
        # --- 格式1: master_ontology_v2.json (老格式) ---
        master_path = ontology_dir / "master_ontology_v2.json"
        if master_path.exists():
            print(f"[CLI] Loading {master_path}...")
            with open(master_path, "r", encoding="utf-8") as f:
                master = json.load(f)
            
            nodes = master.get("nodes", [])
            edges = master.get("edges", [])
            print(f"[CLI]   {len(nodes)} nodes, {len(edges)} edges")
            
            v2_chunks = _build_chunks_from_master(master, str(master_path))
            print(f"[CLI]   {len(v2_chunks)} chunks from master_ontology_v2.json")
            all_chunks.extend(v2_chunks)
            sources.append(str(master_path))
            loaded_any = True
        else:
            print(f"[CLI] [INFO] master_ontology_v2.json not found (skipping)")
        
        # --- 格式2: split-format (nodes.json + relationships.json) ---
        split_nodes_path = ontology_dir / "nodes.json"
        split_rels_path = ontology_dir / "relationships.json"
        if split_nodes_path.exists() and split_rels_path.exists():
            print(f"\n[CLI] Loading split-format files:")
            print(f"       - {split_nodes_path}")
            print(f"       - {split_rels_path}")
            try:
                with open(split_nodes_path, "r", encoding="utf-8") as f:
                    split_nodes = json.load(f)
                with open(split_rels_path, "r", encoding="utf-8") as f:
                    split_rels = json.load(f)
                
                # 兼容：nodes.json 可能是纯数组 or 含nodes字段的dict
                if isinstance(split_nodes, dict):
                    nodes_list = split_nodes.get("nodes", [])
                else:
                    nodes_list = split_nodes
                # 兼容：relationships.json 可能是纯数组 or 含relationships/edges字段的dict
                if isinstance(split_rels, dict):
                    edges_list = split_rels.get("relationships", split_rels.get("edges", []))
                else:
                    edges_list = split_rels
                
                print(f"[CLI]   {len(nodes_list)} nodes, {len(edges_list)} relationships")
                
                # 构造兼容 _build_chunks_from_master 的 master dict
                split_master = {
                    "schema_version": "split-v1",
                    "nodes": nodes_list,
                    "edges": edges_list,
                }
                
                # 尝试加载 instances (可选)
                instances_path = ontology_dir / "instances.json"
                if instances_path.exists():
                    try:
                        with open(instances_path, "r", encoding="utf-8") as f:
                            inst_data = json.load(f)
                        if isinstance(inst_data, dict):
                            inst_list = inst_data.get("instances", inst_data.get("nodes", []))
                        else:
                            inst_list = inst_data
                        if inst_list:
                            # L3 instances 以 "instances" 键注入，与 master v2 结构一致
                            split_master["instances"] = inst_list
                            print(f"[CLI]   + {len(inst_list)} instances loaded")
                    except Exception as e:
                        print(f"[CLI] [WARN] Load instances failed: {e}")
                
                split_chunks = _build_chunks_from_master(split_master, str(split_nodes_path))
                print(f"[CLI]   {len(split_chunks)} chunks from split-format")
                all_chunks.extend(split_chunks)
                sources.append(str(split_nodes_path))
                sources.append(str(split_rels_path))
                loaded_any = True
            except Exception as e:
                print(f"[CLI] [WARN] Split-format load failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            if not split_nodes_path.exists():
                print(f"[CLI] [INFO] nodes.json not found (skipping split-format)")
            else:
                print(f"[CLI] [INFO] relationships.json not found (skipping split-format)")
        
        # --- 格式3: master_ontology_unified.json (在ontology_dir同级或父级) ---
        # 候选路径：ontology_dir内部、ontology_dir父级、固定输出目录
        unified_candidates = [
            ontology_dir / "master_ontology_unified.json",
            ontology_dir.parent / "master_ontology_unified.json",
            PROJECT_ROOT / "outputs" / "runs" / "ontology" / "master_ontology_unified.json",
        ]
        unified_path: Path | None = None
        for uc in unified_candidates:
            if uc.exists():
                unified_path = uc
                break
        
        if unified_path is not None and not loaded_any:
            # 只有当 v2 和 split-format 都没成功加载时，才用 unified 避免重复
            print(f"\n[CLI] Loading {unified_path}...")
            try:
                with open(unified_path, "r", encoding="utf-8") as f:
                    unified_data = json.load(f)
                
                nodes_list = unified_data.get("nodes", [])
                edges_list = unified_data.get("edges", unified_data.get("relationships", []))
                print(f"[CLI]   {len(nodes_list)} nodes, {len(edges_list)} edges")
                
                unified_chunks = _build_chunks_from_master(unified_data, str(unified_path))
                print(f"[CLI]   {len(unified_chunks)} chunks from master_ontology_unified.json")
                all_chunks.extend(unified_chunks)
                sources.append(str(unified_path))
                loaded_any = True
            except Exception as e:
                print(f"[CLI] [WARN] Unified master load failed: {e}")
                import traceback
                traceback.print_exc()
        elif unified_path is not None:
            print(f"\n[CLI] [INFO] master_ontology_unified.json found but skipped (already loaded via v2/split)")
        
        # --- L2 scenarios 文件 ---
        l2_path = ontology_dir / "master_ontology_l2.json"
        if l2_path.exists():
            print(f"\n[CLI] Loading {l2_path}...")
            with open(l2_path, "r", encoding="utf-8") as f:
                l2_data = json.load(f)
            
            l2_chunks = _build_chunks_from_l2(l2_data)
            print(f"[CLI]   {len(l2_chunks)} chunks from master_ontology_l2.json")
            all_chunks.extend(l2_chunks)
            sources.append(str(l2_path))
        else:
            # 也尝试从 layer2_scenarios.json 加载 (split-format)
            l2_alt_path = ontology_dir / "layer2_scenarios.json"
            if l2_alt_path.exists():
                print(f"\n[CLI] Loading {l2_alt_path}...")
                try:
                    with open(l2_alt_path, "r", encoding="utf-8") as f:
                        l2_alt_data = json.load(f)
                    # layer2_scenarios.json 可能结构不同，尝试两种兼容方式
                    if isinstance(l2_alt_data, dict) and ("business_scenarios" in l2_alt_data or "business_logic" in l2_alt_data):
                        l2_chunks = _build_chunks_from_l2(l2_alt_data)
                    else:
                        # 把整个文件内容当作一个大的 scenario 集合包装
                        l2_chunks = _build_chunks_from_l2({
                            "business_scenarios": l2_alt_data if isinstance(l2_alt_data, list) else [],
                            "business_logic": [],
                        })
                    print(f"[CLI]   {len(l2_chunks)} chunks from layer2_scenarios.json")
                    all_chunks.extend(l2_chunks)
                    sources.append(str(l2_alt_path))
                except Exception as e:
                    print(f"[CLI] [WARN] layer2_scenarios.json load failed: {e}")
            else:
                print(f"[CLI] [INFO] master_ontology_l2.json / layer2_scenarios.json not found (skipping)")
        
        source_meta = {"sources": sources}
    
    if not all_chunks:
        print("[CLI] [❌] No chunks generated")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    seen: dict[str, ChunkRecord] = {}
    for c in all_chunks:
        seen[c.chunk_id] = c
    all_chunks = sorted(seen.values(), key=lambda c: c.chunk_id)
    print(f"\n[CLI] Total unique chunks: {len(all_chunks)}")
    
    type_counts: dict[str, int] = {}
    for c in all_chunks:
        type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1
    for t, n in sorted(type_counts.items()):
        print(f"[CLI]   {n:4d}  {t}")
    
    chunks_path = output_dir / "chunks.jsonl"
    write_chunks_jsonl(all_chunks, chunks_path)
    print(f"\n[CLI] Written: {chunks_path}")
    
    print("[CLI] Building inverted index...")
    inverted_index = build_inverted_index(all_chunks)
    idf = compute_idf(inverted_index, len(all_chunks))
    write_inverted_index(inverted_index, output_dir / "inverted_index.json")
    write_idf(idf, output_dir / "idf.json")
    print(f"[CLI]   {len(inverted_index)} tokens indexed")
    
    manifest = {
        "created_at": datetime.now().isoformat(),
        **source_meta,
        "outputs": {
            "chunks": str(chunks_path),
            "inverted_index": str(output_dir / "inverted_index.json"),
            "idf": str(output_dir / "idf.json"),
        },
        "stats": {
            "chunk_count": len(all_chunks),
            "token_count": len(inverted_index),
            "chunk_types": type_counts,
        },
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    print(f"\n[CLI] ✅ RAG artifacts built successfully!")
    print(f"[CLI] Output: {output_dir}")
    return 0


# ---------------------------------------------------------------------------
# Validate Commands
# ---------------------------------------------------------------------------

def validate_json_file(filepath: Path) -> tuple[bool, str]:
    """Validate JSON file syntax."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            json.load(f)
        return True, ""
    except JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def validate_schema_refs(filepath: Path) -> tuple[bool, str]:
    """Validate $ref references in schema files."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            schema = json.load(f)
        
        errors = []
        file_dir = filepath.parent
        
        def check_refs(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if key == "$ref":
                        ref_path = value
                        if ref_path.startswith("#"):
                            pass
                        elif "/" in ref_path:
                            parts = ref_path.split("#")[0]
                            resolved_path = file_dir / parts
                            if not resolved_path.exists():
                                errors.append(f"  {current_path}: $ref '{ref_path}' resolves to non-existent file: {resolved_path}")
                    else:
                        check_refs(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_refs(item, f"{path}[{i}]")
        
        check_refs(schema)
        
        if errors:
            return False, "\n".join(errors)
        return True, ""
    except Exception as e:
        return False, f"Ref validation error: {e}"


def run_validate_schemas(args: argparse.Namespace) -> int:
    """Validate all JSON schema files."""
    schemas_dir = PROJECT_ROOT / "docs" / "skills_v2" / "schemas"
    
    if not schemas_dir.exists():
        print(f"[CLI] [❌] Schemas directory not found at {schemas_dir}")
        return 1
    
    json_files = list(schemas_dir.rglob("*.json"))
    print(f"[CLI] Found {len(json_files)} JSON schema files")
    print("-" * 60)
    
    results = {"valid": 0, "invalid": 0, "errors": []}
    
    for filepath in sorted(json_files):
        rel_path = filepath.relative_to(schemas_dir)
        
        print(f"\n[CLI] Validating: {rel_path}")
        
        is_json_valid, json_error = validate_json_file(filepath)
        if not is_json_valid:
            print(f"  ❌ JSON parsing failed: {json_error}")
            results["invalid"] += 1
            results["errors"].append(f"{rel_path}: JSON parsing failed - {json_error}")
            continue
        
        is_ref_valid, ref_error = validate_schema_refs(filepath)
        if not is_ref_valid:
            print(f"  ❌ $ref validation failed:")
            print(ref_error)
            results["invalid"] += 1
            results["errors"].append(f"{rel_path}: $ref validation failed")
            continue
        
        print(f"  ✅ All checks passed")
        results["valid"] += 1
    
    print("\n" + "-" * 60)
    print(f"[CLI] Validation Summary:")
    print(f"[CLI]   Total files: {len(json_files)}")
    print(f"[CLI]   ✅ Valid: {results['valid']}")
    print(f"[CLI]   ❌ Invalid: {results['invalid']}")
    
    if results["errors"]:
        print(f"\n[CLI] Errors:")
        for error in results["errors"]:
            print(f"[CLI]   - {error}")
        return 1
    else:
        print("\n[CLI] ✅ All schemas are valid!")
        return 0


def run_validate_architecture(args: argparse.Namespace) -> int:
    """Validate architecture contracts and registries."""
    from src.infrastructure.config.contract_registry import get_contract_registry
    from src.infrastructure.config.rule_loader import get_rule_loader
    from src.pipeline.skill_pipeline_separation import get_skill_pipeline_bridge
    from src.pipeline.pipeline_router import list_pipelines, PipelineType
    
    print("[CLI] === 契约注册表验证 ===")
    registry = get_contract_registry()
    stats = registry.get_stats()
    
    print(f"[CLI] 总契约数: {stats['total_contracts']}")
    print(f"[CLI] Skills: {stats['skill_count']}")
    print(f"[CLI] Schemas: {stats['schema_count']}")
    print(f"[CLI] Configs: {stats['config_count']}")
    print(f"[CLI] Rules: {stats['rule_count']}")
    print(f"[CLI] Pipeline Steps: {stats['pipeline_step_count']}")
    
    print("\n[CLI] === Pipeline 路由验证 ===")
    pipelines = list_pipelines()
    for route, info in pipelines.items():
        type_label = {
            PipelineType.PRODUCTION.value: '生产主线',
            PipelineType.MIGRATION.value: '迁移支线',
            PipelineType.EXPERIMENTAL.value: '实验线',
        }.get(info.pipeline_type.value, '未知')
        deprecated_label = ' (已废弃)' if info.deprecated else ''
        print(f"[CLI]   {route.upper()}: {info.name} [{type_label}]{deprecated_label}")
    
    print("\n[CLI] === Skill/Pipeline 职责分离验证 ===")
    bridge = get_skill_pipeline_bridge()
    stats = bridge.get_stats()
    
    print(f"[CLI] 注册的 Skills: {stats['skills_count']}")
    print(f"[CLI] 注册的 Steps: {stats['steps_count']}")
    
    print("\n[CLI] ✅ 架构验证完成")
    return 0


# ---------------------------------------------------------------------------
# Analyze Commands
# ---------------------------------------------------------------------------

def run_analyze_output(args: argparse.Namespace) -> int:
    """Analyze pipeline output directory."""
    run_dir = Path(args.run_dir) if args.run_dir else PROJECT_ROOT / "outputs" / "runs" / "ontology" / "pipeline"
    
    if not run_dir.exists():
        print(f"[CLI] [❌] Run directory not found: {run_dir}")
        return 1
    
    print(f"[CLI] === Output Analysis: {run_dir} ===")
    
    asset_packets_dir = run_dir / "asset_packets" / "raw"
    if not asset_packets_dir.exists():
        print(f"[CLI] [WARN] asset_packets directory not found")
        return 0
    
    run_subdirs = list(asset_packets_dir.glob("MAIN-*"))
    if not run_subdirs:
        print(f"[CLI] [WARN] No run directories found")
        return 0
    
    output_dir = sorted(run_subdirs)[-1]
    print(f"[CLI] Analyzing: {output_dir.name}")
    
    print("\n[CLI] === Files ===")
    for f in sorted(output_dir.glob("*.json")):
        size = f.stat().st_size
        print(f"[CLI]   {f.name:<30s} {size:>8d} bytes")
    
    entities_path = output_dir / "entities.json"
    if entities_path.exists():
        with open(entities_path, "r", encoding="utf-8") as f:
            entities = json.load(f)
        print(f"\n[CLI] === Entities ({len(entities)}) ===")
        
        layer_dist = {}
        for e in entities:
            layer = e.get("layer", "unknown")
            layer_dist[layer] = layer_dist.get(layer, 0) + 1
        print(f"[CLI] Layer distribution: {layer_dist}")
        
        for e in entities[:10]:
            layer = e.get('layer', '')
            name = e.get('name', '')
            meta_class = e.get('meta_class', '')
            confidence = e.get('confidence', 0)
            desc = e.get('description', '')[:60]
            print(f"[CLI]   - {name} [{layer}] ({meta_class}) conf={confidence:.2f} | desc=\"{desc}\"")
    
    rels_path = output_dir / "relationships.json"
    if rels_path.exists():
        with open(rels_path, "r", encoding="utf-8") as f:
            rels = json.load(f)
        print(f"\n[CLI] === Relationships ({len(rels)}) ===")
        for r in rels[:10]:
            from_name = r.get('from_name', r.get('from', ''))
            to_name = r.get('to_name', r.get('to', ''))
            rel_type = r.get('type', '')
            confidence = r.get('confidence', 0)
            print(f"[CLI]   - {from_name} -> {to_name} [{rel_type}] conf={confidence:.2f}")
    
    evidence_path = output_dir / "evidence.json"
    if evidence_path.exists():
        with open(evidence_path, "r", encoding="utf-8") as f:
            evs = json.load(f)
        print(f"\n[CLI] === Evidence ({len(evs)}) ===")
        for i, ev in enumerate(evs[:5]):
            content = ev.get('content', '')[:80]
            raw_text = ev.get('raw_text', '')[:60]
            source = ev.get('source_file', '')
            confidence = ev.get('confidence', 0)
            print(f"[CLI]   {i+1}: content=\"{content}\" | raw=\"{raw_text}\" | source={Path(source).name if source else ''} | confidence={confidence:.2f}")
    
    print(f"\n[CLI] ✅ Output analysis complete")
    return 0


def run_analyze_folder(args: argparse.Namespace) -> int:
    """Analyze folder using UnifiedCompilationPipeline."""
    print("=" * 70, flush=True)
    print("  ONTOLOGY PIPELINE - Folder Analysis", flush=True)
    print("=" * 70, flush=True)
    print(f"  Target folder: {args.folder}", flush=True)
    print(f"  Domain: {args.domain}", flush=True)
    print(f"  Fast mode: {'Enabled' if args.fast else 'Disabled'}", flush=True)
    if args.question:
        print(f"  Analysis question: {args.question}", flush=True)
    print("=" * 70, flush=True)

    from ontology_compiler.services.unified_compilation_pipeline import (
        UnifiedCompilationService,
        CompilationContext,
    )

    path = Path(args.folder)
    is_folder = path.is_dir()
    
    ctx = CompilationContext(
        source_path=args.folder,
        is_folder=is_folder,
        domain=args.domain,
        fast_mode=args.fast,
        skip_rigor=False,
        analysis_question=args.question,
    )

    pipeline = UnifiedCompilationPipeline()
    
    print(f"\n  Pipeline steps: {[s.__class__.__name__ for s in pipeline.steps]}", flush=True)
    print(f"\n  Starting analysis... (this may take a few minutes)", flush=True)

    try:
        ctx = asyncio.run(pipeline.run(ctx))

        print("\n" + "=" * 70, flush=True)
        print("  Analysis Results", flush=True)
        print("=" * 70, flush=True)
        print(f"  Files processed: {len(ctx.fragments)}", flush=True)
        print(f"  Quality level: {ctx.quality_level}", flush=True)
        print(f"  Quality score: {ctx.quality_score:.3f}", flush=True)
        print(f"  Conflicts found: {len(ctx.conflicts)}", flush=True)
        print(f"  Deduplication stats: {ctx.deduplication_stats}", flush=True)
        
        if ctx.final_fragment:
            print(f"  Final nodes: {len(ctx.final_fragment.nodes)}", flush=True)
            print(f"  Final relationships: {len(ctx.final_fragment.relationships)}", flush=True)
        
        if ctx.warnings:
            print(f"\n  Warnings ({len(ctx.warnings)}):", flush=True)
            for w in ctx.warnings[:5]:
                print(f"    - {w}", flush=True)
        
        if ctx.errors:
            print(f"\n  Errors ({len(ctx.errors)}):", flush=True)
            for e in ctx.errors:
                print(f"    - {e}", flush=True)

        print("\n" + "=" * 70, flush=True)
        print("  Analysis completed!", flush=True)
        print("=" * 70, flush=True)
        
        return 0

    except Exception as e:
        print(f"\n  Analysis failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return 1


def run_analyze_layers(args: argparse.Namespace) -> int:
    """Check layer consistency."""
    layers_dir = PROJECT_ROOT / "outputs" / "runs" / "ontology" / "layers"
    
    if not layers_dir.exists():
        print(f"[CLI] [❌] Layers directory not found: {layers_dir}")
        return 1
    
    layer_map = {
        'l0_metadata.json': 'layer_L0.json',
        'l1_concepts.json': 'layer_L1.json',
        'l2_entities.json': 'layer_L2.json',
        'l25_rules.json': 'layer_L2.5.json',
        'l3_instances.json': 'layer_L3.json',
    }
    
    print("[CLI] === Layers Consistency Check ===")
    
    for merge_file, export_file in layer_map.items():
        merge_path = layers_dir / merge_file
        export_path = layers_dir / export_file
        
        if merge_path.exists() and export_path.exists():
            with open(merge_path, 'r', encoding='utf-8') as f:
                merge_data = json.load(f)
            with open(export_path, 'r', encoding='utf-8') as f:
                export_data = json.load(f)
            
            counts_match = len(merge_data) == len(export_data)
            print(f"[CLI] {merge_file} ({len(merge_data)}) vs {export_file} ({len(export_data)})")
            print(f"[CLI]   Count matches: {'✅' if counts_match else '❌'}")
        else:
            print(f"[CLI] {merge_file} or {export_file} not found")
    
    rel_dir = PROJECT_ROOT / "outputs" / "runs" / "ontology" / "relationships"
    if rel_dir.exists():
        print("\n[CLI] === Relationship Type Statistics ===")
        rel_types = {}
        total_rels = 0
        for rel_file in rel_dir.glob('relationships_*.json'):
            rel_type = rel_file.name.replace('relationships_', '').replace('.json', '')
            with open(rel_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            count = len(data)
            total_rels += count
            rel_types[rel_type] = count
        
        print(f"[CLI] Total relationship types: {len(rel_types)}")
        print(f"[CLI] Total relationships: {total_rels}")
        
        print("\n[CLI] Top 10 relationship types:")
        for rel_type, count in sorted(rel_types.items(), key=lambda x: -x[1])[:10]:
            print(f"[CLI]   {rel_type}: {count}")
    
    print("\n[CLI] ✅ Layer analysis complete")
    return 0


# ---------------------------------------------------------------------------
# Health & Status
# ---------------------------------------------------------------------------

def run_health(args: argparse.Namespace) -> int:
    """Check system health."""
    print("[CLI] Running health check...")
    
    from src.pipeline.stages import run_health
    
    DEFAULT_MASTER_PATH = PROJECT_ROOT / "outputs" / "runs" / "ontology" / "master_ontology"
    
    health_args = argparse.Namespace()
    health_args.master_path = str(DEFAULT_MASTER_PATH)
    
    result = run_health(health_args)
    print(f"[CLI] Health check: {'✅' if result.success else '❌'}")
    
    if result.data:
        for key, value in result.data.items():
            print(f"[CLI]   {key}: {value}")
    
    return 0 if result.success else 1


def run_status(args: argparse.Namespace) -> int:
    """Show pipeline status."""
    print("[CLI] Pipeline status:")
    
    master_path = PROJECT_ROOT / "outputs" / "runs" / "ontology" / "master_ontology"
    
    if master_path.exists():
        if master_path.is_dir():
            nodes_file = master_path / "nodes.json"
            rel_file = master_path / "relationships.json"
            
            if nodes_file.exists():
                with open(nodes_file, "r", encoding="utf-8") as f:
                    nodes = json.load(f)
                print(f"[CLI] Total nodes: {len(nodes)}")
                
                layer_dist = {}
                for n in nodes:
                    layer = n.get("layer", "unknown")
                    layer_dist[layer] = layer_dist.get(layer, 0) + 1
                print(f"[CLI] Layer distribution: {layer_dist}")
            
            if rel_file.exists():
                with open(rel_file, "r", encoding="utf-8") as f:
                    rels = json.load(f)
                print(f"[CLI] Total relationships: {len(rels)}")
    else:
        print("[CLI] No master ontology found")
    
    return 0


# ---------------------------------------------------------------------------
# Cache Management
# ---------------------------------------------------------------------------

def run_cache(args: argparse.Namespace) -> int:
    """Manage document cache."""
    try:
        from src.infrastructure.services.document_cache import DocumentCache
        
        cache = DocumentCache()
        stats = cache.get_cache_stats()
        
        print(f"[CLI] Cache stats:")
        print(f"[CLI]   Total entries: {stats['total_entries']}")
        print(f"[CLI]   Total size: {stats['total_size_mb']} MB")
        print(f"[CLI]   Cache dir: {stats['cache_dir']}")
        
        if args.clear:
            deleted = cache.cleanup_old_entries(days_to_keep=0)
            print(f"[CLI] Cleared {deleted} cache entries")
        
        if args.cleanup:
            deleted = cache.cleanup_old_entries(days_to_keep=args.cleanup)
            print(f"[CLI] Cleaned up {deleted} entries older than {args.cleanup} days")
        
        return 0
    except Exception as e:
        print(f"[CLI] [❌] Cache operation failed: {e}")
        return 1


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ontology_cli",
        description="Unified CLI for Ontology Compiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    compile_parser = subparsers.add_parser("compile", help="Compile documents to ontology")
    compile_parser.add_argument("--file", help="Single source file")
    compile_parser.add_argument("--folder", help="Source folder")
    compile_parser.add_argument("--domain", default="AUTO", help="Business domain (AUTO for auto-detection)")
    compile_parser.add_argument("--strict-governance", action="store_true", help="Enable strict governance")
    # C-1: LLM hard budget gates (0 = unlimited, the default safe backward-compat value)
    compile_parser.add_argument("--max-cost-usd", type=float, default=0.0,
                                help="LLM cost cap in USD (budget refuses further calls when reached; 0=unlimited)")
    compile_parser.add_argument("--max-calls", type=int, default=0,
                                help="LLM call-count cap (0=unlimited)")
    compile_parser.add_argument("--max-total-tokens", type=int, default=0,
                                help="LLM total token cap (prompt+completion; 0=unlimited)")
    compile_parser.add_argument("--degraded-offline", action="store_true",
                                help="Forces LLM-dependent stages into DEGRADE mode (run rule-based only)")
    
    query_parser = subparsers.add_parser("query", help="Query ontology with RAG")
    query_parser.add_argument("query", help="Query text")
    
    rag_parser = subparsers.add_parser("rag", help="RAG operations")
    rag_subparsers = rag_parser.add_subparsers(dest="rag_command")
    rag_build_parser = rag_subparsers.add_parser("build", help="Build RAG index")
    rag_build_parser.add_argument(
        "--source",
        choices=["extraction", "master"],
        help="Source type (auto-detected if omitted)",
    )
    rag_build_parser.add_argument("--run-dir", help="Run directory (extraction mode)")
    rag_build_parser.add_argument(
        "--ontology-dir",
        default="outputs/runs/ontology/master_ontology",
        help="Ontology directory (master mode)",
    )
    rag_build_parser.add_argument("--output-dir", help="Output directory")
    
    validate_parser = subparsers.add_parser("validate", help="Validation commands")
    validate_subparsers = validate_parser.add_subparsers(dest="validate_command")
    validate_subparsers.add_parser("schemas", help="Validate JSON schemas")
    validate_subparsers.add_parser("architecture", help="Validate architecture contracts")
    
    analyze_parser = subparsers.add_parser("analyze", help="Analysis commands")
    analyze_subparsers = analyze_parser.add_subparsers(dest="analyze_command")
    analyze_output_parser = analyze_subparsers.add_parser("output", help="Analyze pipeline output")
    analyze_output_parser.add_argument("--run-dir", help="Run directory to analyze")
    analyze_subparsers.add_parser("layers", help="Check layer consistency")
    analyze_subparsers.add_parser("services", help="Analyze service usage")
    analyze_folder_parser = analyze_subparsers.add_parser("folder", help="Analyze folder using UnifiedCompilationPipeline")
    analyze_folder_parser.add_argument("--folder", required=True, help="Path to folder to analyze")
    analyze_folder_parser.add_argument("--domain", default="SUPPLY_CHAIN", help="Domain context")
    analyze_folder_parser.add_argument("--fast", action="store_true", help="Fast mode (skip LLM)")
    analyze_folder_parser.add_argument("--question", help="Analysis question to guide file selection and prompt")
    
    subparsers.add_parser("health", help="Check system health")
    subparsers.add_parser("status", help="Show pipeline status")
    
    cache_parser = subparsers.add_parser("cache", help="Manage document cache")
    cache_parser.add_argument("--clear", action="store_true", help="Clear all cache")
    cache_parser.add_argument("--cleanup", type=int, help="Clean up entries older than N days")
    
    skill_parser = subparsers.add_parser("skill", help="Skill operations")
    skill_subparsers = skill_parser.add_subparsers(dest="skill_command")
    skill_run_parser = skill_subparsers.add_parser("run", help="Run a skill")
    skill_run_parser.add_argument("skill_name", help="Name of the skill to run")
    skill_run_parser.add_argument("context_json", help="Path to context JSON file")
    skill_subparsers.add_parser("list", help="List registered skills")
    
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_parser.add_argument("--run-dir", help="Run directory for full report")
    report_parser.add_argument("--json", help="Standalone JSON file for L2 report")
    report_parser.add_argument("--output", help="Output file path")
    report_parser.add_argument(
        "--type",
        choices=["extraction", "neo4j", "organize-bom"],
        default="extraction",
        help="Report type",
    )
    report_parser.add_argument("--env", default="config/env.env", help="Env file for neo4j")
    report_parser.add_argument("--filename", help="Custom filename for neo4j")
    report_parser.add_argument("--base-dir", help="Base directory for organize-bom")
    report_parser.add_argument("--cmd", choices=["check", "pagerank", "louvain"], help="Neo4j command")
    report_parser.add_argument("--limit", type=int, default=20, help="Limit for results")
    
    verify_parser = subparsers.add_parser("verify", help="Verification commands")
    verify_subparsers = verify_parser.add_subparsers(dest="verify_command")
    verify_subparsers.add_parser("export", help="Verify export consistency")
    verify_subparsers.add_parser("dual", help="Verify dual store consistency")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "compile":
        args.source = args.folder or args.file
        if not args.source:
            print("[CLI] [❌] Either --file or --folder is required")
            return 1
        return run_compile(args)
    
    elif args.command == "query":
        return run_query(args)
    
    elif args.command == "rag":
        if args.rag_command == "build":
            return run_rag_build(args)
        else:
            rag_parser.print_help()
            return 1
    
    elif args.command == "validate":
        if args.validate_command == "schemas":
            return run_validate_schemas(args)
        elif args.validate_command == "architecture":
            return run_validate_architecture(args)
        else:
            validate_parser.print_help()
            return 1
    
    elif args.command == "health":
        return run_health(args)
    
    elif args.command == "status":
        return run_status(args)
    
    elif args.command == "cache":
        return run_cache(args)
    
    elif args.command == "skill":
        if args.skill_command == "run":
            return run_skill(args)
        elif args.skill_command == "list":
            return list_skills(args)
        else:
            skill_parser.print_help()
            return 1
    
    elif args.command == "report":
        return run_generate_report(args)
    
    elif args.command == "verify":
        if args.verify_command == "export":
            return run_verify_export_consistency(args)
        elif args.verify_command == "dual":
            return run_verify_dual_store(args)
        else:
            verify_parser.print_help()
            return 1
    
    elif args.command == "analyze":
        if args.analyze_command == "output":
            return run_analyze_output(args)
        elif args.analyze_command == "layers":
            return run_analyze_layers(args)
        elif args.analyze_command == "services":
            return run_analyze_services(args)
        elif args.analyze_command == "folder":
            return run_analyze_folder(args)
        else:
            analyze_parser.print_help()
            return 1
    
    else:
        print(f"[CLI] [❌] Unknown command: {args.command}")
        return 1


# ---------------------------------------------------------------------------
# Skill Commands
# ---------------------------------------------------------------------------

def run_skill(args: argparse.Namespace) -> int:
    """Run a registered skill (P0-6-3: canonical harness discovery)."""
    from ontology_compiler.harness import build_default_skill_registry
    
    skill_name = args.skill_name
    context_path = Path(args.context_json)
    
    if not context_path.is_file():
        print(f"[CLI] [❌] Context file not found: {context_path}")
        return 1
    
    with open(context_path, "r", encoding="utf-8") as f:
        context = json.load(f)
    
    registry = build_default_skill_registry()
    skill = registry.get(skill_name)

    if skill is None:
        print(f"[CLI] [❌] Skill '{skill_name}' not found.")
        return 1
    
    result = skill.execute(context)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def list_skills(args: argparse.Namespace) -> int:
    """List registered skills (P0-6-3: canonical harness discovery)."""
    from ontology_compiler.harness import build_default_skill_registry

    registry = build_default_skill_registry()
    skills = [s.name for s in registry.list_skills(enabled_only=False)]
    print(f"[CLI] Registered skills ({len(skills)}):")
    for skill_name in sorted(skills):
        print(f"[CLI]   - {skill_name}")
    return 0


# ---------------------------------------------------------------------------
# Report Commands
# ---------------------------------------------------------------------------

def run_generate_report(args: argparse.Namespace) -> int:
    """Generate human-readable report."""
    from scripts.generate_report import (
        generate_from_run_dir,
        generate_from_json,
        _run_neo4j_report,
        _organize_bom_analysis,
    )
    
    if args.type == "neo4j":
        output_dir = Path(args.output or "outputs/reports")
        _run_neo4j_report(output_dir, filename=args.filename, env_path=args.env, cmd=args.cmd, limit=args.limit)
        return 0
    
    if args.type == "organize-bom":
        if not args.base_dir:
            print("[CLI] [❌] --base-dir is required for --type organize-bom")
            return 1
        _organize_bom_analysis(Path(args.base_dir))
        return 0
    
    if args.json:
        json_file = Path(args.json)
        if not json_file.exists():
            print(f"[CLI] [❌] Input JSON not found: {json_file}")
            return 1
        output_file = Path(args.output) if args.output else json_file.parent / "REPORT.md"
        generate_from_json(json_file, output_file)
        return 0
    
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"[CLI] [❌] Run directory not found: {run_dir}")
            return 1
        
        if (run_dir / "unified_summary.json").exists():
            output_file = Path(args.output) if args.output else None
            generate_from_run_dir(run_dir, output_file)
        elif (run_dir / "l2_business_scenarios.json").exists():
            json_file = run_dir / "l2_business_scenarios.json"
            output_file = Path(args.output) if args.output else run_dir / "REPORT.md"
            generate_from_json(json_file, output_file)
        else:
            print(f"[CLI] [❌] Neither unified_summary.json nor l2_business_scenarios.json found in: {run_dir}")
            return 1
        return 0
    
    else:
        print("[CLI] [❌] Either --run-dir or --json is required")
        return 1


# ---------------------------------------------------------------------------
# Verify Commands
# ---------------------------------------------------------------------------

def run_verify_export_consistency(args: argparse.Namespace) -> int:
    """Verify export consistency between unified file and layers directory."""
    import json
    
    unified_path = Path('outputs/runs/ontology/master_ontology_unified.json')
    layers_dir = Path('outputs/runs/ontology/layers')
    rels_dir = Path('outputs/runs/ontology/relationships')
    
    if not unified_path.exists():
        print(f"[CLI] [❌] Unified file not found: {unified_path}")
        return 1
    
    with open(unified_path, 'r', encoding='utf-8') as f:
        unified = json.load(f)
    
    print('=== 主文件数据 ===')
    print(f'总节点数: {len(unified.get("nodes", []))}')
    print(f'总关系数: {len(unified.get("relationships", []))}')
    
    layer_dist = {}
    for n in unified.get("nodes", []):
        layer = n.get("layer", "unknown")
        layer_dist[layer] = layer_dist.get(layer, 0) + 1
    print(f'节点按 layer 分布: {layer_dist}')
    
    print()
    print('=== layers 目录 ===')
    total_layer_nodes = 0
    if layers_dir.exists():
        for layer_file in sorted(layers_dir.glob('layer_*.json')):
            with open(layer_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            count = len(data)
            total_layer_nodes += count
            print(f'{layer_file.name}: {count} nodes')
        print(f'layers 目录总节点: {total_layer_nodes}')
    
    print()
    print('=== relationships 目录 ===')
    total_rels = 0
    if rels_dir.exists():
        for rel_file in sorted(rels_dir.glob('*.json')):
            with open(rel_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            count = len(data)
            total_rels += count
            print(f'{rel_file.name}: {count} relationships')
        print(f'relationships 目录总关系: {total_rels}')
    
    return 0


def run_verify_dual_store(args: argparse.Namespace) -> int:
    """Verify consistency between unified file and directory store."""
    import json
    
    unified_path = Path('outputs/runs/ontology/master_ontology_unified.json')
    store_dir = Path('outputs/runs/ontology/master_ontology')
    
    if not unified_path.exists():
        print(f"[CLI] [❌] Unified file not found: {unified_path}")
        return 1
    
    print('=== 单体文件 ===')
    with open(unified_path, 'r', encoding='utf-8') as f:
        unified = json.load(f)
    print(f'Nodes: {len(unified.get("nodes", []))}')
    print(f'Instances: {len(unified.get("instances", []))}')
    print(f'Relationships: {len(unified.get("relationships", []))}')
    print(f'L0: {len(unified.get("layer0_metadata", []))}')
    print(f'L1: {len(unified.get("layer1_concepts", []))}')
    print(f'L2: {len(unified.get("layer2_scenarios", []))}')
    
    print()
    print('=== 分层文件 ===')
    if store_dir.exists():
        for file in sorted(store_dir.glob('*.json')):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f'{file.name}: {len(data) if isinstance(data, list) else type(data).__name__}')
        
        print()
        print('=== 一致性检查 ===')
        if (store_dir / 'nodes.json').exists():
            with open(store_dir / 'nodes.json', 'r', encoding='utf-8') as f:
                dir_nodes = json.load(f)
            print(f'Nodes 一致: {len(unified["nodes"]) == len(dir_nodes)}')
    else:
        print('[CLI] [⚠️] 分层文件目录不存在')
    
    return 0


# ---------------------------------------------------------------------------
# Analyze Service Commands
# ---------------------------------------------------------------------------

def run_analyze_services(args: argparse.Namespace) -> int:
    """Analyze service usage across the project."""
    from collections import defaultdict
    
    services_dir = Path(__file__).parent.parent / "ontology_compiler" / "services"
    project_root = Path(__file__).parent.parent
    
    if not services_dir.exists():
        print(f"[CLI] [❌] Services directory not found at {services_dir}")
        return 1
    
    print("Analyzing service usage in ontology_compiler/services/")
    print("=" * 70)
    
    service_files = []
    for filepath in sorted(services_dir.rglob("*.py")):
        rel_path = filepath.relative_to(services_dir)
        if rel_path.parts[0] == "archived":
            continue
        if rel_path.name == "__init__.py":
            continue
        service_files.append((filepath, rel_path))
    
    def find_references(service_name: str) -> list[str]:
        references = []
        exclude_dirs = {"archived", ".git", "__pycache__", "node_modules", ".eggs", "*.egg-info"}
        
        for filepath in project_root.rglob("*.py"):
            rel_path = filepath.relative_to(project_root)
            
            excluded = False
            for exclude in exclude_dirs:
                if exclude in str(rel_path):
                    excluded = True
                    break
            if excluded:
                continue
            
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                patterns = [
                    f"from ontology_compiler.services.{service_name}",
                    f"from ontology_compiler.services import.*{service_name}",
                    f"import ontology_compiler.services.{service_name}",
                    f"from ontology_compiler.services.{service_name} import",
                ]
                
                for pattern in patterns:
                    if pattern in content:
                        references.append(str(rel_path))
                        break
            except Exception:
                continue
        
        return references
    
    usage_report = defaultdict(list)
    
    for filepath, rel_path in service_files:
        service_name = rel_path.stem if rel_path.parent.name == "services" else f"{rel_path.parent.name}.{rel_path.stem}"
        references = find_references(service_name)
        usage_report[len(references)].append((service_name, references))
    
    print(f"\n总服务数: {len(service_files)}")
    
    print("\n=== 未被引用的服务 ===")
    for service_name, refs in sorted(usage_report.get(0, [])):
        print(f"  - {service_name}")
    
    print("\n=== 被引用的服务 ===")
    for count in sorted(usage_report.keys()):
        if count == 0:
            continue
        print(f"\n引用次数: {count}")
        for service_name, refs in sorted(usage_report[count]):
            print(f"  - {service_name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())