# Scripts Directory

> 本体编译系统运行入口。只保留少量稳定 CLI 入口；复用逻辑下沉到 `src/` 与 `ontology_compiler/`。不再常用的脚本统一放入 `archive/`。

---

## 🔥 Batch 处理入口（推荐）

### 🚀 一行命令版：`run_batch.py`（Step1+2+3 全自动）

**下次处理新 batch 直接改 `--folder` 路径，其他全部自动：**

```bash
cd /path/to/ontology-compiler

# ⚠️ 只改这里：把 batch10 换成新的 batch 文件夹
python scripts/run_batch.py ^
  --folder "/path/to/docs/batch"
```

### ✅ 已验证通过的真实路径示例（直接复制改 batch 号即可）

```bash
# 【示例 1】BOM 目录下的 batch（batch10 验证通过，含 &）
python scripts/run_batch.py ^
  --folder "/path/to/docs/batch"

# 【示例 2】CIP BIS 目录下的 batch（batch1 验证通过，路径中间含空格「CIP BIS」，脚本已兼容）
python scripts/run_batch.py ^
  --folder "/path/to/docs/batch"
```

脚本自动完成一条龙：
```text
Step 1 compile  → Step 2 merge  → Step 3 回归测试 + 导出分层图
（提取文档）       （合并到master）    （14/14 PASS，生成 layer_L0~L3.json）
```

可选开关：
| 参数 | 说明 |
|------|------|
| `--file FILE` | 处理单个文档（和 `--folder` 二选一） |
| `--domain DOMAIN` | 指定业务域（默认 `AUTO`） |
| `--strict-governance` | 严格治理模式 |
| `--skip-regression` | 跳过回归测试 |
| `--skip-export` | 跳过分层图导出 |

> 💡 源码：[run_batch.py](scripts/run_batch.py)

---

### 🛠️ 手动版（3 步，高级/调试用）

处理新 batch 只需 **3 步**，所有脚本都已模板化：

```text
Step 1 compile  →  Step 2 merge_batchXX.py  →  Step 3 回归测试 + export
(提取文档)         (合并到 master)              (验证+生成分层图)
```

#### Step 1: 运行 Pipeline 提取

```bash
cd /path/to/ontology-compiler

# ⚠️ 修改这里：--folder 改成你要处理的 batch 文件夹
python scripts/ontology_cli.py compile ^
  --folder "/path/to/docs/batch" ^
  --domain AUTO
```

完成后记下输出末尾的 **Run ID**，例如 `MAIN-20260804-165231`

#### Step 2: 合并到 Master

合并已经写在 `run_batch.py` 里。新 batch 直接跑 `python scripts/run_batch.py --folder <目录>`，不要再复制 `merge_batch10.py`（该模板已归档）。

#### Step 3: 验证 + 导出

```bash
# 3.1 回归测试（必须 14/14 PASS）
python scripts/regression_test.py --master outputs/runs/ontology/master_ontology_unified.json

# 3.2 生成最新分层图（layer_L0.json ~ layer_L3.json）
python -c "from argparse import Namespace; from src.pipeline.stages.export import run_export; r = run_export(Namespace(master_path='outputs/runs/ontology/master_ontology_unified.json')); print(r.success, r.message)"
```

> 💡 详细说明见项目根目录的 [README.md](../README.md) 中的「Batch 快速处理流程」章节

---

## 🔎 RAG 查询（本体知识检索）

通过 `ontology_cli.py query` 对 master 本体做实体和关系检索。**无需构建向量索引**，直接从 `master_ontology_unified.json` 毫秒级返回结果。

```bash
cd /path/to/ontology-compiler

# 基本用法
python scripts/ontology_cli.py query "EBOM"
python scripts/ontology_cli.py query "HERCULES"
python scripts/ontology_cli.py query "PLM"
python scripts/ontology_cli.py query "Quality"
```

> 💡 提示：master 实体和描述以英文为主，建议用英文/缩写关键词；每条结果附带层级、ID、类型、得分、描述和该实体的 Top N 条关系（含方向 IN/OUT）。

---

## 推荐入口

| 脚本 | 用途 |
|------|------|
| **`run_batch.py`** | **🔥 一键 Batch 处理（Step1+2+3 一条龙，最常用！）** |
| `ontology_cli.py` | **统一CLI入口**（包含 compile / query / rag / validate / analyze 等） |
| `validate_real_samples.py` | 真实样本验证 |

---

## 常用命令

### Pipeline 编译

```bash
# 编译文档到本体
python scripts/ontology_cli.py compile --folder /path/to/docs --domain AUTO

# 查询本体（RAG）
python scripts/ontology_cli.py query "MBOM相关的流程和部门"

# 构建RAG索引
python scripts/ontology_cli.py rag build --ontology-dir outputs/runs/ontology

# 验证架构
python scripts/ontology_cli.py validate architecture

# 检查健康状态
python scripts/ontology_cli.py health
```

### 图谱优化与重新生成

```bash
# 从主数据集重新生成图谱（导出分层文件和关系）
python -c "from argparse import Namespace; from src.pipeline.stages.export import run_export; r = run_export(Namespace(master_path='outputs/runs/ontology/master_ontology_unified.json')); print(r.success, r.message)"

# 回归测试验证
python scripts/regression_test.py --master outputs/runs/ontology/master_ontology_unified.json
```

### RAG 检索 + Dify LLM 综合分析（🔥 默认，推荐）

> ⚙️ **配置来源**：[config/env.env](scripts/run_batch.py)
> `LLM_PROVIDER=dify` + `DIFY_API_KEY=<your-key>` + `DIFY_BASE_URL=<your-dify-base-url>`
> 优先级：Dify > UnifiedLLMClient > UnifiedLLMProvider，Dify 不可用时自动降级。

```bash
# 1) 构建 RAG 索引（首次使用或 master 更新后需要运行）
python scripts/rebuild_rag_index.py

# 2) 🔥 RAG 实体检索 + Dify LLM 结构化分析（输出：L0~L3 实体清单 + 关联关系 + 证据 + 置信度）
python scripts/rag_cli.py query "What is MBOM and how is it related to EBOM?"
python scripts/rag_cli.py query "PLM system related processes" --topk 8

# 3) 交互式 RAG Chat
python scripts/rag_cli.py chat
```

独立的 `run_rag_query.py` 已归档，对话入口用上面的 `rag_cli.py chat`。

> 💡 **Dify LLM 回答流程**：
> 1. 从 `master_ontology_unified.json` 构建的 RAG 索引（chunks.jsonl + 倒排索引）中按关键词/别名/ID 检索 topk 相关本体 chunks（节点、关系、L0~L3 层级）
> 2. 将「系统要求 + 检索到的本体上下文 + 用户问题」拼接后通过 `/v1/chat-messages` 发送到 Dify Chat App
> 3. Dify 返回结构化 JSON（L0_entities / L1 / L2 / L2.5 / L3_entities + relationships，每条含证据 + 置信度）
> 4. RAGAnswerer 自动将 JSON 渲染为中文 Markdown（实体定义、父类归属、关联关系图、证据溯源）
>
> 典型输出示例（MBOM vs EBOM）：
> - **实体**：L0 概念（EBOM / MBOM）→ L1 领域流程（Ebom MBOM Split / MBOM Update）→ L2 业务（变更传播 / 修订流程）→ L2.5 映射（消耗关系 / 模板使用）→ L3 实例（导入工序）
> - **关系**：EBOM ──associated_with──▶ MBOM（置信度 0.9）、L2-093 Change Propagation ──uses──▶ EBOM / MBOM 等 7 条

### 验证

```bash
# 真实样本验证
python scripts/validate_real_samples.py --folder "docs/BATCH2"
```

---

## 🧪 语义回归（生产样本，推荐）

### 完整 5 样本隔离回归

每个样本使用**独立 master 副本**，互不污染：

```bash
python scripts/run_regression_samples.py ^
  --production ^
  --html ^
  --sample-id mbom-du-v2 ^
  --sample-id ebom-mbom-2023 ^
  --sample-id production-inventory-pptx ^
  --sample-id production-inventory-md ^
  --sample-id doc-folder-all
```

输出：
```text
outputs/runs/ontology/pipeline/regression_reports/regression_summary_<时间戳>.json
outputs/runs/ontology/pipeline/regression_reports/regression_summary_<时间戳>.md
outputs/runs/ontology/pipeline/regression_reports/regression_summary_<时间戳>.html
```

### 启用 LLM candidate generation

候选实体/关系会接回主抽取 state（默认关闭）：

```bash
set EXTRACT_ENABLE_CANDIDATE_GEN=1   # PowerShell
# 或
export EXTRACT_ENABLE_CANDIDATE_GEN=1  # bash

python scripts/run_regression_samples.py --production --html --sample-id ebom-mbom-2023
```

### 聚焦回归（调试单条）

```bash
# 只跑 EBOM/MBOM，重点看 DERIVED_FROM
python scripts/run_regression_samples.py --production --html --sample-id ebom-mbom-2023

# 只跑库存 Markdown，重点看 Cycle Stock / Safety Stock
python scripts/run_regression_samples.py --production --html --sample-id production-inventory-md
```

### 断点排查

| 用途 | 命令 |
|------|------|
| 拆解单个样本来源与关系链路 | `python scripts/regression_artifact_diagnosis.py` |
| master 注入 vs 抽取占比 | `python scripts/analyze_master_origins.py` |
| Gate A 结构保鲜（无 LLM） | `python scripts/gate_a_freshness_check.py` |
| MCP 只读语义面 | `python -m mcp.ontology_mcp.server`（配置见 `mcp/README.md`） |
| 查看最新回归报告 | `outputs/runs/ontology/pipeline/regression_reports/` |
| 对比两次 pipeline 结果 | `python scripts/compare_pipeline_results.py --before <a> --after <b> --html` |

---

## 📄 单文件编译（如 Quality-ATACQ 1.pptx）

```bash
# 正常模式（需要 LLM provider 可用）
python scripts/ontology_cli.py compile ^
  --file "data/Quality-ATACQ 1.pptx" ^
  --domain SUPPLY_CHAIN

# 离线/降级模式（不依赖 LLM，仅规则抽取）
python scripts/ontology_cli.py compile ^
  --file "data/Quality-ATACQ 1.pptx" ^
  --domain SUPPLY_CHAIN ^
  --degraded-offline
```

> 注意：当前 DeepSeek/Dify provider 偶发 `[Errno 8] nodename nor servname provided`（网络/DNS）。恢复后可去掉 `--degraded-offline` 获得完整 extract。

---

## 🔍 诊断与门禁（修改代码后建议运行）

```bash
# 全量单元测试
python -m pytest tests/ -q

# doc 链接/路径完整性
python scripts/check_docs_links.py

# 代码行数上限
python scripts/check_code_size.py

# import 边界约束
python scripts/check_import_boundary.py

# 全部脚本语法检查
python -m py_compile scripts/*.py
```

---

## 目录结构

```text
scripts/
├── run_batch.py                 # 🔥 一键 Batch 处理（Step1+2+3 一条龙，含合并）
├── ontology_cli.py              # 统一 CLI 入口（compile / query / rag / validate / analyze / health）
├── regression_test.py           # 回归测试基线
├── analyze_hierarchy.py         # 层级结构分析
├── validate_real_samples.py     # 真实样本验证
├── build_pipeline_result.py     # 构建 pipeline_result.json
├── compare_pipeline_results.py  # 对比 pipeline 结果
├── run_regression_samples.py    # 生产样本语义回归（独立 master 隔离）
├── regression_artifact_diagnosis.py  # 来源/关系链路诊断
├── check_docs_links.py          # 门禁：doc 链接完整性
├── check_code_size.py           # 门禁：代码行数上限
├── check_import_boundary.py     # 门禁：import 边界
├── summarize_audit_artifacts.py # 审计产物摘要
├── rebuild_rag_index.py         # 重建 RAG 索引（实体 + 关系 Chunks + 倒排索引）
├── rag_cli.py                   # RAG 子入口薄壳（query/chat/build；build 复用 ontology_cli）
├── reports/                     # 报告生成模块
└── ARCHIVED/                    # 已归档脚本（一次性分析/修复/检查工具）
```

---

## 归档脚本

`ARCHIVED/` 目录包含已完成任务的一次性脚本、优化工具和临时检查脚本：

### 2026-10-20 归档（无调用方）

- `merge_batch8.py` / `merge_batch10.py` — 写死旧 Run ID；合并已内嵌进 `run_batch.py`
- `run_batch1.py` — 写死本机 Batch1 路径
- `inject_domain_mapping.py` — 写死本机 `domain_mapping.md` 路径的一次性注入
- `optimize_master.py` — 针对当时 7 个实体名的一次性清理，现由流水线 GlobalOptimizer 做
- `generate_optimization_report.py` — 绑定 2026-08-01 那份 backup
- `run_rag_query.py` — 对话入口改为 `rag_cli.py chat`

### 本次新增归档（8 个，2026-08-04）

- `apply_optimizations.py` - 全局优化脚本（功能已合并到 GlobalOptimizer 内部）
- `check_duplicate_meta_ids.py` - 重复 meta_id 检查（已合并到 GlobalOptimizer）
- `cleanup_duplicate_meta_ids.py` - 重复 meta_id 清理（已合并到 GlobalOptimizer）
- `cleanup_redundant_buckets.py` - 冗余分组清理（已合并到 GlobalOptimizer）
- `optimize_layer_classification.py` - 层级分类优化（已合并到 GlobalOptimizer）
- `parse_pipeline_result.py` - Pipeline 结果解析（功能已内联到 compare/report 脚本）
- `review_pending_entities.py` - 待审实体检阅（一次性任务已完成）
- `start_services.py` - 服务启动脚本（功能已内联到 ontology_cli 或不需要）

### 早期归档

- `analyze_batch4_time.py` - BATCH4时间分析
- `analyze_chunk_distribution.py` - Chunk分布分析
- `analyze_pptx_content.py` - PPTX内容分析
- `analyze_prompt_template.py` - Prompt模板分析
- `analyze_relationship_mismatch.py` - 关系匹配分析
- `analyze_remaining_missing_ids.py` - 缺失ID分析
- `check_batch4_text.py` - BATCH4文本检查
- `compare_batch4_optimization.py` - BATCH4优化对比
- `diagnose_deepseek_network.py` - Deepseek网络诊断
- `find_business_scenario_parent.py` - 业务场景父级查找
- `find_missing_parent_id.py` - 缺失parent_id查找
- `fix_missing_parent_id.py` - 缺失parent_id修复
- `fix_ontology_p2.py` - 本体P2修复
- `fix_pml_parent.py` - PML父级修复
- `normalize_skill_json.py` - Skill JSON规范化
- `restore_from_backup.py` - 备份恢复
- `sync_skill_schemas.py` - Skill Schema同步
- `test_deepseek_connectivity.py` - Deepseek连接测试
- `verify_chunk_optimization.py` - Chunk优化验证
- `verify_p2_fixes.py` - P2修复验证

### 2026-08-02 归档（一次性分析/修复/导出/检查脚本）

- `analyze_batch_performance.py` - 批次性能分析
- `analyze_l1_l2_detail.py` - L1/L2 层级详细分析
- `analyze_l2_l2.py` - L2/L2.5 层级分析
- `analyze_master_corrected.py` - Master 修正分析
- `analyze_results.py` - 结果分析
- `apply_final_v3.py` - V3 修复应用
- `apply_v4_fix.py` - V4 修复应用
- `distribution_analysis.py` - 分布分析
- `enhance_decoupling.py` - L1→L2 关系解耦增强
- `export_batch2_md.py` - BATCH2 导出
- `export_deep_fix_md.py` - 深度修复导出
- `export_final_md.py` - 最终导出
- `export_final_report.py` - 最终报告导出
- `export_hard_fix_full_list.py` - 硬修复清单导出
- `export_v4_md.py` - V4 导出
- `export_verification_report.py` - 验证报告导出
- `inspect_master.py` - Master 检查（含硬编码路径）
- `list_all_entities.py` - 实体列表（含硬编码 run_id）
- `list_full_details.py` - 完整详情列表（含硬编码 run_id）
- `test_hard_fixes.py` - 硬修复测试
- `test_llm_api.py` - LLM API 连接测试
- `update_prompts.py` - Prompt 更新
- `update_prompts_v31.py` - Prompt V31 更新
- `validate_cli.py` - CLI 验证工具
