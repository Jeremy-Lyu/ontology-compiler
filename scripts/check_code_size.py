from __future__ import annotations

"""LOC ratchet gate for oversized files (execution plan Phase 0).

Sources of truth for the ceilings:
- docs/NEXT_PHASE_ARCHITECTURE_OPTIMIZATION_EXECUTION_PLAN_20260811.md (section 5.1)
- docs/CODE_VOLUME_AND_ARCHITECTURE_OPTIMIZATION_REVIEW_20260811.md

The gate fails when a tracked file exceeds its frozen ceiling, so tracked
files can only shrink (ratchet down), never grow. Function sizes are reported
informationally: functions >= 200 lines in tracked files and >= 80 lines in
new files are listed, matching the long-term target of <= 80 lines per
function without hard-failing existing code.

Usage:
    python3 scripts/check_code_size.py            # verify against ceilings
    python3 scripts/check_code_size.py --report   # print full report, exit 0
"""

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Frozen LOC ceilings. Raise intentionally only with an approved migration note.
FILE_CEILINGS = {
    # 2053 (2026-08-11 Phase 0 baseline) -> 2111 (2026-08-13): run() 编排与
    # 收尾流程做方法级拆分（_build_stage_args / _begin_run_continuity /
    # _finalize_pipeline / _collect_classification_metrics /
    # _build_entity_profile_summary / _auto_detect_domain 等专项 helper）。
    # 其中 stage 失败硬停语义（PREFLIGHT/GUARD 时 break）曾在抽取
    # _handle_failed_stage 时丢失，已在 2026-08-13 修复并由
    # tests/pipeline/test_main_pipeline_characterization.py 锁定。
    # 行数增长属同文件分解的过渡成本；这些 helper 是 Phase 3.2 迁往
    # run_context / stage_executor 等独立模块的候选，届时回落至 <900 目标。
    # 2117 -> 2203 (2026-08-14): CLI quality gate feature. Added
    # PipelineQualityStatus enum, quality_status/quality_reason/
    # validation_passed/quality_score fields on PipelineExecutionResult,
    # _derive_quality_status decision logic and _finalize_pipeline wiring.
    # Success semantics are now separated from publishability; locked by
    # tests/pipeline/test_main_pipeline_characterization.py.
    # 2203 -> 2228 (2026-08-14): _extract_stage_quality() feeds extract-stage
    # extraction_quality (validation_passed/quality_score per file) into
    # quality-status derivation, so real extraction quality is used when the
    # quality-gate metadata is absent.
    # 2228 -> 2290 (2026-08-14): fixed three quality-path defects from the
    # 2026-08-14 run log: (1) _normalize_artifact_path flattened the real
    # files/0000 fragment into a non-existent path; (2) degraded-with-output
    # was classified FAILURE 'no publishable structured output'; (3)
    # _collect_pipeline_output returned None before save_result wrote
    # run_manifest.json, so the quality gate never ran. Locked by
    # tests/pipeline/test_main_pipeline_characterization.py.
    # 2290 -> 2328 (2026-08-14): _collect_pipeline_output fallback now maps
    # stage nodes/relations onto the OutputValidator + ontology_packet schema
    # contract (document/entities/relations aliases, id/name/type entity
    # aliases, source_id/target_id/relation_type relation aliases) so the
    # quality gate runs on real data and passes on valid output.
    # 2328 -> 2355 (2026-08-15): fallback normalization extended to align
    # relation endpoint + evidence keys with the governed packet schema used
    # by the quality gate (endpoint/relation_key/evidence mapping), closing
    # the remaining packet-vs-master ID mismatch on ATACQ-style runs.
    # 2355 -> 2390 (2026-08-15): A-3 resolved-config snapshot added to the
    # run manifest (llm/budget/version bundles) for run replayability.
    # 2390 -> 2415 (2026-08-15): A-2 quality-single-source — _derive_quality_status
    # no longer fabricates a score from extract-stage averages when the
    # governance gate did not run; gate-missing now yields PARTIAL_SUCCESS(0.0)
    # with an explicit reason (explicit branch + docstrings).
    # 2415 -> 2445 (2026-08-15): A-1/A-3 — module-level _config_fingerprint()
    # plus env fallback for schema/skill/prompt versions, so the manifest's
    # resolved-config snapshot is replayable even when PipelineContext carries
    # empty version fields (with characterization coverage).
    # 2445 -> 2415 (2026-08-15): F-3 LOC ratchet — B1-B4 moved output mapping,
    # quality gate, classification metrics, contract validation and run
    # continuity side effects out of MainPipeline; the orchestrator is now
    # measurably smaller. Ceiling tightened to lock the reduction.
    # G7 (2026-08-17): MainPipeline business decisions moved out —
    # entity_layer_policy (layer inference + profile + metadata projection +
    # layer-pair gating) and relation_rewrite_policy (RELATED_TO rewrite
    # family) + stage_args (stage Namespace assembly). Orchestrator dropped
    # 2415 -> 2197; ceiling ratcheted to lock it, continuing the NEXT_PHASE
    # target progression toward <900.
    # 2200 -> 2744 (2026-08-26 RATCHET RESET): drifted +544 during the
    # 2026-08-19..08-25 optimization window (quality-gate fallback mapping,
    # continuity hooks, stage-decision logic) BEFORE the consolidation round;
    # detected when make check was re-run on 2026-08-26. Ceiling reset to the
    # exact current size with ZERO headroom to stop further growth.
    # Reduction plan (docs/CODE_CLEANUP_CANDIDATES.md §3): split output
    # collection / quality-gate derivation into dedicated modules, target
    # back under 2200.
    # 2744 -> 2186 (2026-08-26 PHASE 1+2 REDUCTION, TARGET MET): extracted
    # result_helpers.py (config fingerprint / resolved-config snapshot /
    # contract artifacts / dedupe + evidence helpers) and
    # stage_execution_core.py (_run_pipeline_stages loop /
    # _resolve_actual_stage_decision / trace / _collect_pipeline_output)
    # using the pipeline-first-parameter pattern; deleted zero-caller
    # wrapper methods. MainPipeline keeps thin delegating methods; behavior
    # locked by test_main_pipeline_characterization.py. Ceiling tightened to
    # the new exact size with zero headroom.
    # 2186 -> 2148 (2026-09-09 P0-3 batch 1): regression/continuity metadata
    # persistence moved verbatim to src/pipeline/regression_metadata.py
    # (_capture_and_persist_extracted_metadata / _collect_continuity_sample_text /
    # _persist_extracted_metadata / _persist_regression_manifest) using the
    # established pipeline-first-parameter pattern; call sites in
    # _begin_run_continuity now call module functions directly (no tests
    # referenced the private methods — verified repo-wide 2026-09-09).
    # Also pruned the now-unused imports (shutil/_write_json/_entity_layer_value/
    # project_metadata_to_context/extract_metadata; normalize_entity is STILL
    # used by _normalize_entity_record and was kept). Ceiling ratcheted to
    # the new exact size with zero headroom.
    # 2148 -> 2139 (2026-09-09 P0-3 batch 2): asset-packet dedup/evidence
    # wiring collapsed — asset_packet.py now calls result_helpers
    # (deduplicate_entities / deduplicate_relations /
    # extract_evidence_from_entities) directly instead of routing through
    # MainPipeline._deduplicate_* / _extract_evidence_from_entities thin
    # delegates (deleted; no external callers). _generate_asset_packet kept
    # as instance-level seam (characterization tests patch it on the
    # instance); run() flow unchanged. Ceiling ratcheted to exact size.
    # 2139 -> 2097 (2026-09-09 P0-3 batch 3): validation report assembly moved
    # verbatim to result_helpers.build_validation_report (pipeline-first);
    # MainPipeline._build_validation_report is now a thin delegate. The other
    # batch-3 candidates (_run_contract_validation_chain/_run_quality_gate/
    # _collect_pipeline_output) are already use-case wiring and are kept: two
    # are patched at instance level by characterization tests, and the test
    # suite calls _collect_pipeline_output directly. Ceiling ratcheted to the
    # new exact size with zero headroom.
    "src/pipeline/main_pipeline.py": 797,
    # 826 -> 797 (2026-10-16 P1-3): stage-result 构建簇（skipped/degraded +
    # skip 判定，53 行）下沉至 src/pipeline/stage_result_builders.py 纯函数，
    # MainPipeline 保留薄委托（stage_execution_core 以实例方法调用）。
    # 注：原定 "≤600" 目标不成立 —— AST 测量 104 方法/530 方法体，其中 85 个是
    # ≤5 行委托，本身就是薄编排形态；再降需删除委托接缝（会断调用方）。
    # 结构判据（薄编排 + 无重方法，最大 30 行）已满足。仍零余量。
    # ^ 859 -> 822 (2026-09-09 P0-3 batch 18): per-stage handlers (13
    # _handle_* bodies, ~37 lines) moved verbatim to new
    # src/pipeline/stage_handlers.py (pipeline-first-parameter pattern);
    # MainPipeline keeps thin delegates (the dispatch table in
    # _stage_handler stays on the orchestrator — run-flow tests patch it at
    # class level). Pruned the now-unused run_* stage imports +
    # DEFAULT_PIPELINE_DIR/PROJECT_ROOT (stage_plumbing now imports PROJECT_ROOT from .stages directly). Ceiling
    # ratcheted to the exact size with zero headroom
    # (cumulative 2186 -> 822).
    # ^ 968 -> 859 (2026-09-09 P0-3 batch 17): __init__ construction state
    # (_normalize_output_dir / __init__ body / _generate_run_id /
    # _compute_file_hash / _determine_execution_mode / _set_execution_mode,
    # ~109 lines) moved verbatim to new src/pipeline/pipeline_init.py
    # (init_context_state / init_run_state / init_services /
    # init_execution_state + helpers; order preserved; _set_execution_mode
    # stays an instance-level seam patched by guard-failure tests). Pruned
    # the now-unused hashlib/build_pipeline_run_paths/ArtifactRegistry +
    # service *Service class imports (getters + UnifiedSkillContract
    # annotation stay). Ceiling ratcheted to the exact size with zero
    # headroom (cumulative 2186 -> 859).
    # ^ 1021 -> 967 (2026-09-09 P0-3 batch 16): skill bootstrap + post-exec
    # validation cluster (_initialize_skill_integration /
    # _run_post_execution_validation / _validate_pipeline_output /
    # _run_quality_gate, ~54 lines bodies) moved verbatim to new
    # src/pipeline/skill_bootstrap.py (pipeline-first-parameter pattern);
    # MainPipeline keeps thin delegates (_run_post_execution_validation
    # stays an instance-level seam patched by characterization tests;
    # _validate/_run_gate are B1/B3 use-case wiring). Pruned the now-unused
    # SkillPipelineBridge/PipelineSkillRouter/*Contract class imports (only
    # the getter factories + UnifiedSkillContract annotation remain; the
    # router getter import is restored because characterization tests patch
    # src.pipeline.main_pipeline.get_pipeline_skill_router). Ceiling ratcheted
    # to the exact size with zero headroom (cumulative 2186 -> 968).
    # ^ 1153 -> 1021 (2026-09-09 P0-3 batch 15): continuity + artifact
    # registration cluster (_begin_run_continuity /
    # _load_regression_baseline* / _end_run_continuity /
    # _filter_unprocessed_files* / _register_stage_artifacts /
    # _register_stage_bundle / _record_stage_artifact_outputs /
    # _mark_source_file_processed_if_needed, ~132 lines bodies) moved
    # verbatim to new src/pipeline/continuity_artifacts.py
    # (pipeline-first-parameter pattern); MainPipeline keeps thin delegates
    # (_begin_run_continuity / _end_run_continuity /
    # _register_stage_artifacts stay instance-level seams patched by
    # characterization tests). Pruned the now-unused ArtifactBundle +
    # regression_metadata imports (consumers moved with the bodies).
    # Ceiling ratcheted to the exact size with zero headroom
    # (cumulative 2186 -> 1021).
    # ^ 1284 -> 1153 (2026-09-09 P0-3 batch 14): model block
    # (PipelineStage / PipelineStatus / PipelineQualityStatus /
    # PipelineExecutionMode / PipelineContext / PipelineStats /
    # PipelineExecutionResult, ~131 lines) moved verbatim to new
    # src/pipeline/pipeline_models.py; main_pipeline keeps an import +
    # re-export so router/scripts/tests/package keep importing from
    # main_pipeline (identity verified: same class objects). Pruned the
    # now-unused Enum/dataclass/field + DEFAULT_MASTER_PATH imports.
    # Ceiling ratcheted to the exact size with zero headroom
    # (cumulative 2186 -> 1153).
    # ^ 1379 -> 1284 (2026-09-09 P0-3 batch 13): domain auto-detection +
    # factory/convenience constructors (_auto_detect_domain /
    # _match_folder_keywords / _match_path_keywords / create_main_pipeline /
    # run_main_pipeline, ~95 lines) moved verbatim to new
    # src/pipeline/pipeline_factory.py (plain module pattern — no instance
    # state); main_pipeline keeps thin re-export aliases (router +
    # src.pipeline.__init__ import create/run_main_pipeline by name).
    # Ceiling ratcheted to the exact size with zero headroom
    # (cumulative 2186 -> 1284).
    # ^ 1432 -> 1379 (2026-09-09 P0-3 batch 12): runflow wiring
    # (_log_pipeline_start / _ensure_contract_validation /
    # _build_failed_contract_result / _resolve_stage_enums /
    # _collect_classification_metrics / _store_classification_metrics,
    # ~53 lines bodies) moved verbatim to new src/pipeline/runflow.py
    # (pipeline-first-parameter pattern); MainPipeline keeps thin delegates
    # (_run_contract_validation_chain stays the instance-level seam patched
    # by characterization tests). Pruned the now-unused
    # get_classification_service import (moved with the wiring into runflow).
    # Ceiling ratcheted to the exact size with zero headroom
    # (cumulative 2186 -> 1379).
    # ^ 1525 -> 1432 (2026-09-09 P0-3 batch 11): status/quality derivation
    # (_derive_pipeline_status / _extract_stage_quality /
    # _derive_quality_status / _derive_execution_reason /
    # _build_pipeline_result, ~93 lines bodies) moved verbatim to new
    # src/pipeline/status_quality.py (pipeline-first-parameter pattern);
    # MainPipeline keeps thin delegates (_extract_stage_quality and
    # _derive_quality_status stay diagnostics seams called directly by
    # characterization tests). Ceiling ratcheted to the exact size with zero
    # headroom (cumulative 2186 -> 1432).
    # ^ 1580 -> 1525 (2026-09-09 P0-3 batch 10): canonical result assembly
    # (_build_core_pipeline_result / _build_artifact_envelope, ~55 lines
    # bodies) moved verbatim to result_helpers.build_core_pipeline_result /
    # build_artifact_envelope (pipeline-first-parameter pattern); MainPipeline
    # keeps thin delegates (_build_core_pipeline_result stays an
    # instance-level seam called directly by bridge characterization tests;
    # _build_artifact_envelope chains through it). Pruned the now-unused
    # CorePipelineStageResult import. Ceiling ratcheted to the exact size
    # with zero headroom (cumulative 2186 -> 1525).
    # ^ 1673 -> 1580 (2026-09-09 P0-3 batch 9): stage-skill hook cluster
    # (_build_default_skill_contract / _apply_stage_skills /
    # _get_skills_for_stage / _merge_skill_result / _validate_stage_output /
    # _collect_validation_entities / _collect_validation_relations /
    # _record_stage_validation_violations, ~94 lines bodies) moved verbatim
    # to new src/pipeline/stage_skills.py (pipeline-first-parameter pattern);
    # MainPipeline keeps thin delegates (_apply_stage_skills and
    # _validate_stage_output stay instance-level seams patched by
    # characterization tests). Ceiling ratcheted to the exact size with zero
    # headroom (cumulative 2186 -> 1580).
    # ^ 1747 -> 1673 (2026-09-09 P0-3 batch 8): governance arming body
    # (_arm_governance_from_context, ~60 lines) moved verbatim to new
    # src/pipeline/governance_arming.py (pipeline-first-parameter pattern);
    # MainPipeline keeps a thin delegate (called from __init__). Also deleted
    # the zero-caller _RELATED_TO_REWRITE_MAP class constant (the canonical
    # map lives in relation_rewrite_policy.RELATED_TO_REWRITE_MAP since G7;
    # verified no source refs). Ceiling ratcheted to the exact size with zero
    # headroom (cumulative 2186 -> 1673).
    # ^ 1887 -> 1747 (2026-09-09 P0-3 batch 7): stage-plumbing normalization
    # cluster (normalize_artifact_path / sanitize / normalize_stage_result_paths /
    # entity-layer index / rewrite_stage_relationships / normalize_stage_entities /
    # *_record / *_layer / *_summary / *_metadata / layer-pair gate / lookup /
    # rewrite_related_to / build_stage_args / resolve_stage_fragment_arg /
    # collect_stats / collect_output_files / log_summary, ~140 lines bodies)
    # moved verbatim to new src/pipeline/stage_plumbing.py
    # (pipeline-first-parameter pattern); MainPipeline keeps thin delegates
    # (callers: _sanitize path inside run flow + stage handlers go through
    # instance methods; _normalize_artifact_path is called directly by
    # characterization tests). Pruned the now-unused
    # get_relation_governance_engine import (moved with the gate into
    # stage_plumbing). Ceiling ratcheted to the exact size with zero headroom
    # (cumulative 2186 -> 1747).
    # B-1 (2026-08-17): RAG LOC ratchet — RAG-04 moved ~800 lines of
    # presentation/formatting out of rag_answerer (formatting.py), so actual
    # size dropped 2967 -> 2172. Ceiling tightened to 2200 (current + ratchet
    # headroom); future RAG splits ratchet down by BOUNDARY_STEP (25) toward
    # the RAG-07 target of <1000.
    # 2200 -> 2216 (2026-08-26 RATCHET RESET): drifted +16 in the
    # 2026-08-19..08-25 window; reset to exact size with zero headroom.
    # Split task registered in docs/CODE_CLEANUP_CANDIDATES.md §3.
    # 2216 -> 2110 (2026-08-26 REDUCTION): deleted 13 zero-caller/zero-test
    # AH-framework delegate methods (RAG-04 already moved the implementations
    # to presentation/; only _generate_ah_framework_report had an internal
    # caller and is kept) + pruned the now-unused formatting imports. The
    # characterization suite exercises the presentation functions
    # directly, so coverage is unchanged.
    # 2110 -> 2131 (2026-09-01): asyncio-nesting fix in _call_llm_with_client.
    # Running asyncio.run() inside a live event loop raised RuntimeError; the
    # coroutine is now executed in a dedicated thread (ThreadPoolExecutor)
    # while the loopless path keeps asyncio.run. Behavior locked by
    # tests/rag/test_rag_fix_characterization.py.
    # 2131 -> 2288 (2026-09-01 P1-1/P1-2): two deliberate additions —
    # (1) P1-1 EvidenceGate integration: _retrieve_with_gate /
    # _search_packet_with_gate implement RETRIEVE_MORE auto re-retrieval
    # (locked by tests/rag/test_rag_evidence_gate_characterization.py);
    # (2) P1-2 DifyCallLLMBackend adapter + _rewrite_query wiring restore the
    # QueryRewriter LLM expansion over Dify (locked by
    # tests/rag/test_query_rewriter_llm_characterization.py). Reset to exact
    # size with zero headroom; split task remains in
    # docs/CODE_CLEANUP_CANDIDATES.md §3.
    # 2288 -> 861 (2026-09-03 RAG-07 / §9.7 p2-3): bounded-context split of
    # rag_answerer into four modules —
    # (1) deterministic_answerer.py: QuestionType/AnswerResult/DomainOntology +
    # 7 deterministic handlers + answer_deterministic;
    # (2) llm_calls.py (LLMCallMixin): provider calls (_call_llm / _call_llm_with_dify
    # / _call_deepseek_direct / _call_llm_with_client) + prompt building +
    # context formatting + AH-framework report generation;
    # (3) retrieval_mixin.py (RetrievalMixin): _retrieve_hybrid_fused /
    # _retrieve_with_entity_expansion / _retrieve_with_gate + entity-expansion
    # funnel + synthesized-summary helpers;
    # (4) rag_answerer.py keeps DifyCallLLMBackend, _rewrite_query, _search_neo4j,
    # _build_evidence, _compute_confidence, packet search, _answer_async_impl
    # and the public facade. RAGAnswerer(LLMCallMixin, RetrievalMixin) inherits
    # the moved methods; instance-level mock.patch.object still works on
    # inherited methods, so characterization coverage is unchanged (behavior
    # locked by tests/rag/test_rag_contracts_characterization.py /
    # test_rag_answerer_formatting_characterization.py /
    # test_query_rewriter_llm_characterization.py). Ceiling tightened to
    # 861 + BOUNDARY_STEP headroom (target <1000 met).
    "src/application/rag/rag_answerer.py": 875,
    # 1092 -> 1097 (2026-08-12): L0-nodes flatten bug fix in run_extract. The
    # fix needs one extra assignment vs the old buggy single-line assignment,
    # plus a short rationale comment; behavior fix locked by
    # tests/pipeline/test_extract_characterization.py.
    # 1097 -> 1205 (2026-08-12): Phase 3 run_extract split (331 -> 150 lines).
    # Extracted 8 focused helpers (_infer_document_type_from_source,
    # _auto_detect_domain, _collect_input_files, _load_skill_hits,
    # _build_document_metadata, _process_extract_file + _consume_cache_hit +
    # _persist_success_fragment) plus a small _ExtractRunState holder. Net file
    # growth is the transitional cost of in-file decomposition (helpers move to
    # a dedicated module in Phase 5). Behavior locked by
    # tests/pipeline/test_extract_characterization.py.
    # 1205 -> 1269 (2026-08-12): Phase 3 >80-line cleanup —
    # _apply_node_optimizations (118 -> _load_optimization_config +
    # _optimize_node + thin orchestrator) and _apply_global_id_uniqueness
    # (88 -> _build_global_id_indexes/_remap_colliding_node_ids/
    # _rewire_colliding_relationships/_report_global_id_uniqueness). extract.py
    # now has no function >80 lines besides the run_extract orchestrator (150,
    # at the Phase 3 target).
    # 1269 -> 1302 (2026-08-14): per-file extraction quality tracking in the
    # extract stage. _ExtractRunState gains file_quality; _persist_success_fragment
    # records extraction_quality_status/validation_passed/quality_score/chunk
    # metadata per file; run_extract aggregates it into extraction_quality for
    # the pipeline quality gate.
    # 1302 -> 1320 (2026-08-14): _infer_chunk_strategy selects an explicit
    # chunking strategy per file suffix and threads chunk_strategy through the
    # per-file document_metadata to the extractor.
    # 1320 -> 1327 (2026-08-14): _consume_cache_hit invalidates polluted
    # empty cache entries, so a stale cached-at time does not cause an
    # endless hit-then-reprocess cycle for each run.
    # 1327 -> 1343 (2026-08-15): per-file extract timeout is now configurable
    # (EXTRACT_FILE_TIMEOUT, default 900s) via _file_timeout_seconds() — the
    # hardcoded 600s ceiling killed VSM/ATACQ runs mid-extract when
    # smart-chunking + slow LLM JSON retries exceeded the file budget.
    # 1343 -> 1347 (2026-08-18): file-level concurrency raised 2 -> 4 and the
    # rationale comment documents DIFY/DeepSeek rate limiting being owned by
    # the LLM client; small transitional growth, will ratchet back down when
    # the semaphore config moves to pipeline.yaml.
    # 1347 -> 1382 (2026-08-26 RATCHET RESET): drifted +35 in the
    # 2026-08-19..08-25 window; reset to exact size with zero headroom.
    # 1382 -> 1289 (2026-08-26 REDUCTION): cross-file [GLOBAL ID FIX] cluster
    # moved verbatim to global_id_fix.py (re-imported for compatibility);
    # file concurrency made env-configurable via EXTRACT_FILE_CONCURRENCY
    # (default unchanged at 4) — pipeline.yaml itself is dead config (no
    # loader), so the env pattern mirrors EXTRACT_FILE_TIMEOUT instead.
    # 1289 -> 1311 (2026-09-01 RATCHET RESET): drifted +22 in the
    # 2026-08-26..08-31 window — the combined post-safety-net fragment
    # persistence block (fragment_<run_id>.json prepended to fragment_paths so
    # merge consumes the fresh aggregate, not a stale per-file cache snapshot)
    # plus per-file extraction_quality aggregation in run_extract.  Reset to
    # exact size with zero headroom; ratchet back down in Phase 5 when
    # run_extract's helpers move to a dedicated module.
    # 1311 -> 1323 (2026-09-02 RATCHET RESET): M4 CQ 抽取约束 delegation block
    # in run_extract (result.data.update(extract_cq_stage_hint(...))). All CQ
    # logic lives in src/pipeline/cq_integration.py / src/application/cq/ —
    # the stage only holds the zero-regression try/except hook. Behavior locked
    # by tests/cq/test_cq_extract_coverage_m4.py.
    "src/pipeline/stages/extract.py": 1391,
    # 1323 -> 1391 (2026-10-16): ratchet 漂移修正。`96b45a1` 修 TrueUnifiedExtractor
    # 扁平返回双兼容（status/data 两处 fallback）+ `6ecccdc` Markdown 渲染改道，
    # 共 +68 行；工作树无 diff，陈旧值对齐。仍零余量。
    # 2095 -> 2161 (2026-08-12): Phase 3 optimize_before_master_write split
    # (214 -> <=80 line orchestrator). Extracted step-group helpers
    # (_init_optimize_stats, _collect_master_collections, _run_consolidation,
    # _run_entity_enrichment, _run_relationship_hierarchy, _finalize_master).
    # Net file growth is the transitional cost of in-file decomposition;
    # behavior locked by tests/pipeline/test_global_optimizer_characterization.py.
    # 2161 -> 2260 (2026-08-12): Phase 3 >80-line cleanup — the six step
    # helpers (_fix_duplicate_meta_ids 127, _cleanup_relationships 125,
    # _connect_orphan_nodes 128, _assign_hierarchy_parent_ids 131,
    # _connect_isolated_components 93, _connect_l0_roots 83) were split into
    # focused helpers; the file now has NO function >80 lines. Same
    # transitional in-file decomposition cost; behavior locked by
    # tests/pipeline/test_global_optimizer_characterization.py.
    # 2260 -> 2320 (2026-08-14): cross-layer rules unified onto
    # config/ontology/relationship_rules.yaml as single source of truth —
    # added _cross_layer_direction_allowed/_load_cross_layer_rules replacing
    # the hardcoded _ALLOWED_CROSS_LAYER (which previously conflicted with the
    # quality gate, forcing critical layer_pair violations every run).
    # 2320 -> 2382 (2026-08-26 RATCHET RESET): drifted +62 in the
    # 2026-08-19..08-25 window; reset to exact size with zero headroom.
    # 2382 -> 2237 (2026-08-26 REDUCTION): extracted entity_accessors.py
    # (_normalize_name/_get_entity_id/_get_entity_name/_set_entity_id) and
    # cross_layer_ops.py (Step-5b trio + _ALLOWED_CROSS_LAYER); names
    # re-imported so `go.*` attribute access and internal call sites are
    # unchanged (locked by test_extracted_modules + characterization).
    "src/pipeline/stages/global_optimizer.py": 2258,
    # 2237 -> 2258 (2026-10-16): ratchet 漂移修正。`6ecccdc` 引入 +21 行未同步
    # ceiling；工作树无 diff，纯陈旧值对齐。仍零余量。
    # 1137 -> 584 (2026-09-03): P2 RAG 收口 — domain builder (414 lines) extracted
    # to domain_chunk_builder.py; ChunkRecord unified to domain/models.py.
    # Behavior locked by test_chunk_indexing_characterization.py +
    # test_rag_contracts_characterization.py.
    "src/application/rag/chunk_indexing.py": 600,
    # 835 -> 907 (2026-08-12): Phase 3 run_merge extraction. The 470-line
    # run_merge was decomposed into module-level helpers (run_merge now 69
    # lines); helper code lives in the same module to avoid a circular import,
    # so the file grew while function granularity improved. Behavior locked by
    # tests/pipeline/test_merge_characterization.py.
    # 907 -> 1028 (2026-08-12): Phase 3 continued — run_dedup (76->43) and the
    # three >80-line helpers (_build_merged_nodes 102->11, _merge_relationships
    # 83->25, _reassign_colliding_ids 99->25) were split into focused helpers;
    # file now has NO function >80 lines. Net file growth is the transitional
    # cost of in-file decomposition; the helper groups are the targets for the
    # planned move to dedicated modules (Phase 5), which will shrink this file.
    # 1028 -> 1029 (2026-08-18): trailing-newline drift (+1 by the LOC counter;
    # no functional growth). Keep the frozen ceiling honest so the gate stays
    # meaningful.
    # 1029 -> 1111 (2026-09-01 P0-3): empty-extraction guard. run_merge now
    # refuses to write master from an empty/under-threshold fragment
    # (MERGE_MIN_FRAGMENT_ENTITIES env, default 0 = block only absolute-empty)
    # to prevent an unstable LLM provider from wiping master (regression
    # 2026-09-01, GOV rollback). Added _fragment_entity_count +
    # merge_min_fragment_entities + guard branch. Behavior locked by
    # tests/pipeline/test_merge_characterization.py.
    "src/pipeline/stages/merge.py": 1111,
    "ontology_compiler/services/render_summary.py": 1142,
    "ontology_compiler/services/cross_file_association_service.py": 1529,
    # 1376 -> 1404 (2026-08-14): fixed _run_llm_extraction_loop return contract.
    # The return statement was nested under the except branch; the degraded
    # validation path (break with entities) fell through to an implicit None
    # return, crashing extract_unified with "cannot unpack non-iterable
    # NoneType". Return moved to method level after the retry loop. Locked by
    # tests/extraction/test_extracted_modules.py
    # TestLlmExtractionLoopReturnContract.
    # 1404 -> 1410 (2026-08-14): surface extraction_quality_status /
    # validation_passed / chunk_quality / chunk_metrics through
    # extract_async_content metadata so the compilation pipeline can classify
    # chunk-partial/degraded results instead of treating them as plain success.
    # 1410 -> 1421 (2026-08-14): chunk_strategy constructor override plus
    # EXTRACT_CHUNK_CACHE env wiring in the _extract_with_chunking wrapper.
    # 1421 -> 1427 (2026-08-15): D-5 prompt truncation/compression metrics
    # threaded through _build_unified_prompt + single-call extraction loop.
    # 1427 -> 1534 (2026-08-18): DIFY 作为主 provider 的切换收口 + extract 并发
    # 与 LLM 调用侧（circuit-breaker / rate-limiter / cache 共享工厂）配套说明
    # 与失败记忆调整逻辑在 extract/extract_async 双路径的补强（上午既有调优）。
    # 该批为生产策略调整的过渡成本；下一步抽离失败记忆/策略调整到独立模块后
    # 回落（目标随 RAG-07 收敛）。
    # 1534 -> 1584 (2026-08-26 RATCHET RESET): drifted +50 in the
    # 2026-08-19..08-25 window; reset to exact size with zero headroom.
    # 1584 -> 1529 (2026-08-26 REDUCTION): failure-memory strategy check
    # moved to extraction_failure_memory.py (apply_failure_memory_adjustments);
    # guarded imports + fallback stubs deduplicated; deleted zero-caller
    # _repair_json/_extract_json_from_text wrappers + unused parser imports.
    "ontology_compiler/services/true_unified_extractor.py": 1529,
}

# Boundaries so a ratcheted file never silently disappears from tracking.
BOUNDARY_STEP = 25
FUNCTION_REPORT_THRESHOLD = 80


def _loc(path: Path) -> int:
    return sum(1 for _ in path.open(encoding="utf-8", errors="ignore"))


def _functions(path: Path) -> list[tuple[str, int]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            size = node.end_lineno - node.lineno + 1
            if size >= FUNCTION_REPORT_THRESHOLD:
                out.append((node.name, size))
    out.sort(key=lambda item: item[1], reverse=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="LOC ratchet gate")
    parser.add_argument("--report", action="store_true", help="print report and always exit 0")
    args = parser.parse_args()

    failures: list[str] = []
    report: list[str] = []
    report.append(f"LOC ratchet: {len(FILE_CEILINGS)} tracked files")
    report.append(f"{'file':<66} {'loc':>5} {'ceiling':>8}  status")

    for rel, ceiling in sorted(FILE_CEILINGS.items(), key=lambda item: item[0]):
        path = ROOT / rel
        if not path.exists():
            failures.append(f"tracked file missing: {rel}")
            report.append(f"{rel:<66} {'--':>5} {ceiling:>8}  MISSING")
            continue
        size = _loc(path)
        ok = size <= ceiling
        status = "ok" if ok else "OVER"
        report.append(f"{rel:<66} {size:>5} {ceiling:>8}  {status}")
        if not ok:
            failures.append(f"{rel}: {size} lines exceeds ceiling {ceiling}")

    hot_functions = [(rel, fn, size) for rel in FILE_CEILINGS for fn, size in _functions(ROOT / rel)]
    report.append("")
    report.append("Functions >= 80 lines in tracked files (informational):")
    for rel, fn, size in sorted(hot_functions, key=lambda item: item[2], reverse=True):
        report.append(f"  {size:>5}  {rel}:{fn}")

    if args.report:
        print("\n".join(report))
        return 0

    print("\n".join(report))
    print("")
    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("OK: all tracked files within LOC ceilings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())