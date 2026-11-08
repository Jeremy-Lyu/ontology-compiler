# scripts/ 分级清单（2026-10-16 诊断 P0-2，清单式分级：不搬文件，只定级）

> 教训：`6ecccdc` 批量搬运生产文件致门禁崩。物理分目录高风险，改清单约束。
> 本清单是唯一分级依据；新增脚本必须在此登记定级，否则 `check_docs_links` 可扩展检查。

## GATE（5，`make check` 直调，有单测）

| 脚本 | 门禁 |
|---|---|
| check_code_size.py | LOC ratchet |
| check_compat_usage_gate.py | compat 只减不增 |
| check_docs_links.py | 文档链接 |
| check_governed_publish.py | governed 发布 |
| check_import_boundary.py | import 方向 |

## CLI（3，用户入口）

ontology_cli.py / rag_cli.py / run_batch.py

## OPS（14，管线一部分，必须 --dry-run + 单测）

| 脚本 | 被谁 import |
|---|---|
| run_regression_samples.py | Makefile regression |
| retrieval_eval.py | run_regression_samples |
| build_governed_packet.py | tests + MasterStore 钩子；`--require-publishable` / `--certified` + publish_receipt |
| retire_dated_shims.py | 带日期垫片退役调度（text_utils / skill_registry / DualPipeline） |
| `python -m mcp.ontology_mcp.server` | 只读 MCP（search/get/answerable/readiness/ask）；见 `mcp/README.md` |
| gate_a_freshness_check.py | Gate A 结构保鲜（RELATED_TO / origin / 可答性 smoke；≠ 全量 LLM 回归） |
| analyze_master_origins.py | master `origin_kind` 注入 vs 抽取占比 |
| rebuild_rag_index.py | 手动运维 |
| cleanup_outputs_retention.py | retention-weekly |
| verify_safety_net_coverage.py | 回归验证 |
| golden_semantic_eval.py | run_regression_samples |
| compare_pipeline_results.py | 手动对比 |
| compat_usage_stats.py | check_compat_usage_gate |
| regression_artifact_diagnosis.py | 回归诊断 |
| regression_diagnosis.py | 回归诊断 |
| regression_manifest_smoke.py | make check |
| regression_test.py | 回归 |
| validate_real_samples.py | 真实验收 |

## ONEOFF（5，一次性分析，不进门禁，改前先问）

analyze_hierarchy.py / analyze_master_ontology_rag.py / build_pipeline_result.py /
export_governed_chunks_to_md.py / summarize_audit_artifacts.py

2026-10-20 移入 `scripts/ARCHIVED/`（无调用方，且写死了旧 Run ID 或本机路径）：
`merge_batch8.py`、`merge_batch10.py`（合并已内嵌进 `run_batch.py`）、
`run_batch1.py`、`inject_domain_mapping.py`、`optimize_master.py`、
`generate_optimization_report.py`、`run_rag_query.py`（对话入口是 `rag_cli.py chat`）。

> 注：`quality_observability.py` 定级 OPS（被 run_regression_samples:496,742 import），不是分析脚本。

## 新增（2026-10-16 P2-3c）

| 脚本 | 说明 |
|---|---|
| report_skill_registry_health.py | skill registry 健康报告（**只读、无副作用，故豁免 --dry-run**）；`--check` 退出 1 表示债务回退，可接入 CI。收敛 phantom(24) / discovery_errors(16) / evaluable(18) / routing integrity 四类发现 |
