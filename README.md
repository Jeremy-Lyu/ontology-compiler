# Ontology Compiler

把制造与供应链文档编译成分层本体，再拿这张图做检索和问答。

输入是 PPT、Word、PDF、Excel 或 Markdown。输出是带层级、关系和证据的知识图：实体归到 L0–L3，关系受白名单约束，每条结果能回到原文。语言模型负责抽取；规则负责在模型不可用时仍产出可用结果，并在合并阶段消掉重复实体。

生产入口只有一条：`scripts/ontology_cli.py compile` → `src/pipeline` 的 `MAIN` 流水线。

## 能力

- 把单份文档或整个文件夹抽成本体，并合并进同一份 master。
- 按领域规则识别制造 BOM、物流、工位作业（OIS / PII / WES）一类对象。
- 跨文件把同一事物的不同写法收成一个节点。
- 用 BM25 和向量混合检索，再按证据回答问题。
- 语言模型超时或没有密钥时，降级为纯规则抽取，流水线仍能跑完。
- **含义优先**：先做可答性预检，再答；覆盖不足就报缺口，不编造。
- **只读 MCP**：给 Cursor / 代理用的语义面（search / get / answerable / ask / readiness）。

## 特点

- **层和类是分开的。** L0 是主类（Process、System、Actor、Asset、Data 等），L1 是受控子类，L2 是业务对象，L2.5 是映射和规则，L3 是实例。语义说明在 `docs/ENTITY_CLASSIFICATION_SPEC.md`。
- **证据跟着节点走。** 抽取结果保留来源文件和原文片段，检索回答要能指回这些证据。
- **模型不是唯一路径。** `LLM_PROVIDER` 选 DeepSeek 或 Dify，失败时可切备用提供方。`--degraded-offline` 强制只走规则。
- **重复实体有固定消解顺序。** 先别名表，再子串，再模糊匹配，最后看邻居关系。低置信度只打标，不自动合并。
- **检索不依赖单一打分。** 关键词和向量各排一次名，再用倒数排名融合，避免 BM25 分数和余弦分数直接相加。
- **发布有收据。** governed packet 经 `publish_receipt` / `check_governed_publish` 校验后才宜对外消费。
- **机制绿 ≠ 业务绿。** `scripts/gate_a_freshness_check.py` 盯 RELATED_TO、来源拆分与 CQ 可答性漂移。

## 流水线

一次 `compile` 按这个顺序跑完。

| 阶段 | 做什么 |
|---|---|
| preflight | 读配置、确认输入文件 |
| extract | 解析正文、按长度分块、调用模型或规则抽取 |
| merge | 并入 master，处理 ID 冲突 |
| dedup | 去掉重复节点 |
| graph_fusion | 把碎片收成一张图 |
| relationship | 补关系；打开开关时做关系改写 |
| audit / dcs_audit | 审计覆盖和缺口 |
| guard | 质量门禁 |
| emergent_reasoning | 补跨层连接 |
| export | 写出分层结果和治理包 |
| document_report | 写单文档报告 |
| summarize | 写本次运行摘要 |

## 算法

**正文切块。** 按文件类型选策略：Markdown、Word、PPT 按标题和段落切；纯文本和 PDF 按上下文切；表格按固定长度切。超长文档先按段、再按句，最后才硬切，块与块之间保留重叠。阈值由 `CHUNK_THRESHOLD_CHARS` 控制，写在 `config/env.env`。

**抽取。** 模型按 `schemas/ontology_packet.schema.json` 返回 JSON。返回不是合法 JSON 时会在同一次调用里重试，次数由 `LLM_MAX_JSON_RETRIES` 控制。L2.5 领域规则在 `skills/l25/`：命中 MBOM、物流等关键词后，把对应约束送进抽取提示。

**实体消解（MSER）。** 实现在 `src/pipeline/stages/entity_resolution.py`，别名表是 `config/entity_aliases.yaml`。

1. 别名表精确命中，直接合并。
2. 较短名称是较长名称的子串（两边至少 4 个字符），视为同一对象。
3. 模糊匹配使用 rapidfuzz 的 `ratio`、`partial_ratio`、`token_sort_ratio`，取三者最高分。未安装 rapidfuzz 时这一档关闭。
4. 分数处于中间地带时，和邻居重叠、词元重叠一起加权：模糊 0.5、邻居 0.3、词元 0.2。达到阈值才自动合并，否则只标记待审。

**检索。** 治理包上建两套索引：倒排索引上的 BM25，以及本地 TF-IDF 向量的余弦相似度。可选换成 OpenAI embedding 或 sentence-transformers，由 `EMBEDDING_PROVIDER` 决定。两路名次用 Reciprocal Rank Fusion 合并，常数 k = 60。中文分词在安装了 jieba 时使用 jieba，否则退回按字切。Neo4j 图检索是可选通道，连接写在 `config/env.env` 的 `NEO4J_*`。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install rapidfuzz   # 实体模糊合并需要；不装则跳过这一档

cp config/env.example.env config/env.env
# 编辑 config/env.env：填入 LLM_PROVIDER、对应的 API key 和地址
```

编译一份文档：

```bash
python scripts/ontology_cli.py compile \
  --file "/path/to/document.pptx" \
  --domain AUTO
```

`AUTO` 会根据正文判断领域。制造和供应链材料通常会落到 `SUPPLY_CHAIN`。

没有模型时：

```bash
python scripts/ontology_cli.py compile \
  --file "/path/to/document.pptx" \
  --domain SUPPLY_CHAIN \
  --degraded-offline
```

提问（先可答性，再查询）：

```bash
python scripts/rag_cli.py meaning answerable "MBOM 如何由 EBOM 派生"
python scripts/rag_cli.py meaning search "MBOM"
python scripts/rag_cli.py query "MBOM 如何由 EBOM 派生"
python scripts/ontology_cli.py query "作业指导书和工艺卡是什么关系"
```

MCP（stdio，只读；配置见 `mcp/README.md`）：

```bash
python -m mcp.ontology_mcp.server
```

检查改动：

```bash
make check
python scripts/gate_a_freshness_check.py   # 结构保鲜；全量 LLM 回归另跑 make regression
```

`make check` 跑全部单元测试、字节码编译、回归冒烟，以及代码行数、导入边界、文档链接、兼容层用量四道门禁。

## 常用操作

```bash
# 整个目录
python scripts/ontology_cli.py compile --folder "/path/to/docs" --domain AUTO

# 编译、合并、回归一次做完
python scripts/run_batch.py --folder "/path/to/docs"

# 限制这次运行的模型花费
python scripts/ontology_cli.py compile \
  --file "/path/to/document.pptx" \
  --max-cost-usd 1 \
  --max-calls 20

# 打开关系改写（默认关闭，避免未确认时改写 RELATED_TO）
PIPELINE_SKILL_RELATION_REWRITE=1 \
  python scripts/ontology_cli.py compile --file "/path/to/document.pptx"

# 看最近一次运行是否健康
python scripts/ontology_cli.py health
python scripts/ontology_cli.py status
```

回归样本登记在 `config/regression_samples.json`：

```bash
python scripts/run_regression_samples.py --production --html --sample-id ebom-mbom-2023
```

## 操作时注意

- **密钥只放 `config/env.env`。** 这个文件不要提交。样例是 `config/env.example.env`。`LLM_PROVIDER=deepseek` 走 DeepSeek 直连；`dify` 走内部 Dify。`LLM_FALLBACK_PROVIDERS` 写成另一个提供方的名字，主提供方短暂失败时才会切过去。
- **PPT 里的图抽不出来。** 解析器读取形状里的文字和备注。布局图、图标如果没有文字框，对应页会几乎是空的。抽完先看 `outputs/runs/ontology/pipeline/<run-id>/artifacts/source-documents/` 里的 Markdown，确认正文齐不齐，再看实体对不对。
- **长文档会拆成多次模型调用。** 单文件总时长上限是 `EXTRACT_FILE_TIMEOUT`（秒）。块太多或 JSON 反复不合法时，先把 `CHUNK_THRESHOLD_CHARS` 调大，减少块数，而不是把重试次数调高。
- **模型上下文太大时先减检索条数。** Dify 报 body too large 时，下调 `DIFY_MAX_CONTEXT_CHARS`，或把 `--topk` 降到 4–6。
- **模糊合并依赖 rapidfuzz。** 没装时别名和子串仍然生效，拼写变体不会自动合并。
- **Neo4j 不是编译的前提。** 编译和本地检索不连数据库也能完成。只有要走图检索时才需要本机 `bolt://localhost:7687` 上的 Neo4j。
- **master 会跨次运行累积。** 本体写在 `outputs/runs/ontology/master/master_ontology_unified.json`。换一批互不相关的文档之前，先确认要不要沿用这份 master。

## 结果放在哪

| 路径 | 内容 |
|---|---|
| `outputs/runs/ontology/pipeline/<run-id>/summary_report.md` | 这次运行的实体数、关系数、分层 |
| `outputs/runs/ontology/pipeline/<run-id>/artifacts/source-documents/` | 实际送去抽取的正文 |
| `outputs/runs/ontology/master/master_ontology_unified.json` | 累积后的本体 |
| `outputs/asset_packets/governed/<run-id>/` | 治理后的实体、关系、证据，检索读的是这里 |
| `outputs/runs/ontology/reports/documents/` | 单文档可读报告 |

## 目录

```text
scripts/ontology_cli.py     编译、查询、健康检查
scripts/rag_cli.py          检索、含义面、对话
scripts/run_batch.py        目录批处理
scripts/build_governed_packet.py  发布可检索资产
mcp/                        只读 MCP 语义面
src/pipeline/               生产流水线
ontology_compiler/          抽取、分块、模型调用
src/application/rag/        BM25、向量索引、混合检索、semantic_surface
skills/l25/                 MBOM 与物流的领域抽取规则
skills/agent/               代理用 setup / management / query skills
config/                     环境样例、关系白名单、别名、回归样本
schemas/                    本体包、实体、关系、RAG 的契约
docs/                       仍被代码引用的说明，见 docs/README.md
AGENTS.md                   代理入口（含义优先）
tests/                      单元测试
```

契约以 schema 和 `config/` 为准。`docs/` 只保留代码还在引用的说明，入口是 `docs/README.md`。

## Limitations

- 这是制造工程语义层的**参考实现**，不是即插即用 SaaS。
- `dag_allowed=true` 只表示并行可行性评估通过，六角色仍串行执行。
- MCP 为本地 stdio、无鉴权；勿暴露到不可信网络。
- 全量业务回归需要配置 LLM；离线可跑 `make check` 与 Gate A 结构保鲜。
