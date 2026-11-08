# Ontology Compiler

制造 / 供应链文档 → **分层本体** → **可治理检索与问答**。

不是「把文档切块丢进向量库」。本项目先把 PPT / Word / PDF / Excel / Markdown 编译成带层级、关系与证据的 OntologyPacket，经合并、去重与质量门禁后发布为 **governed packet**；问答前先做**可答性预检**，覆盖不足就报缺口，不编造。

```text
文档 (pptx/md/docx/pdf/xlsx)
        │
        ▼
   MainPipeline (MAIN)          ← 唯一生产编排
        │  extract / merge / dedup / relation / guard / export
        ▼
   OntologyPacket + master      ← 分层实体 L0–L3 + 白名单关系 + 证据
        │
        ▼
   governed packet              ← 唯一检索资产（含 publish_receipt）
        │
        ├─ rag_cli meaning* / query     （人）
        └─ MCP 只读工具                 （代理）
```

生产入口只有一条：`scripts/ontology_cli.py compile` → `src/pipeline` 的 `MAIN`。

---

## 为什么要这样设计

| 常见做法 | 本项目的取舍 |
|---|---|
| 文档 → embedding → 相似度问答 | 文档 → **本体图** → 治理发布 → 可答性门禁 → 再答 |
| 关系随意抽、RELATED_TO 凑数 | 关系走 **白名单 / 家族改写**；RELATED_TO 占比可观测 |
| LLM 挂了整条链路停 | LLM **失败可降级**为规则抽取，流水线仍跑完 |
| 代理直接 dump 全库 JSON | 代理只碰 **含义面**（search / get / answerable / ask） |
| 「测试绿 = 业务绿」 | 另有 Gate A 保鲜：CQ 可答、RELATED_TO、来源拆分防漂 |

目标读者：要在制造工程里落地**可控语义层**的人（PLM / BOM / 工艺 / 供应链），以及要给 Agent 接只读知识面的人。

---

## 架构

### 三条主线

```text
① 编译主线（写）
   CLI compile
     → src/pipeline stages
     → ontology_compiler 抽取内核（被 pipeline 调用，不另开生产路由）
     → master ontology + export

② 发布主线（写 → 读的边界）
   master
     → build_governed_packet
     → packet.json + 索引 + publish_receipt
     → check_governed_publish

③ 消费主线（只读）
   governed packet
     → semantic_surface（含义 / 可答性）
     → hybrid RAG（BM25 + 向量 RRF）
     → MCP / rag_cli / ontology_cli query
```

### 双内核（有意冻结）

| 层 | 目录 | 职责 |
|---|---|---|
| 编排 | `src/pipeline/` | 唯一生产路由 MAIN；阶段编排、合并、门禁、导出 |
| 抽取 | `ontology_compiler/` | 分块、LLM/规则抽取、prompt、packet adapter |

两者**不物理合并**：pipeline 经明确 adapter 调 compiler；新抽取逻辑只进 compiler。避免「两边各加功能」的第二主线。

### 编译流水线（一次 `compile`）

| 阶段 | 做什么 |
|---|---|
| preflight | 读配置、确认输入 |
| extract | 解析正文 → 分块 → 模型或规则抽实体/关系 |
| merge | 并入 master，处理 ID 冲突 |
| dedup / entity resolution | 别名 → 子串 → 模糊 → 邻居加权合并 |
| graph_fusion | 碎片收成一张图 |
| relationship | 补边；可选关系改写（默认关） |
| audit / guard | 覆盖审计与质量门禁 |
| export | 分层结果 + 触发/配合 governed 资产 |

### 数据与契约

```text
输入文档
  → 中间：OntologyPacket（schemas/ontology_packet.schema.json）
  → 累积：master ontology
  → 对外消费：governed packet（唯一权威；legacy rag 目录禁止）
```

- **实体分层**：L0 主类 → L1 受控子类 → L2 业务对象 → L2.5 映射/规则 → L3 实例。说明书：`docs/ENTITY_CLASSIFICATION_SPEC.md`。
- **关系**：`config/ontology/relationship_rules.yaml` 白名单；优先具体类型，RELATED_TO 仅兜底。
- **证据**：节点/边保留来源文件与原文片段；回答应能指回证据。
- **来源标记**：`origin_kind` ∈ `input` / `injected` / `master_reused` / `derived`，避免「抽出了多少」与注入规则糊在一起。

### 消费面（含义优先）

```text
提问
  → check_answerable（概念 / 关系类型是否在 governed packet 里）
       ├─ 不足 → 返回 gaps，拒绝编造
       └─ 足够 → meaning search/get，或 RAG query，或 MCP ask_question
```

代理入口见 `AGENTS.md`；MCP 配置见 `mcp/README.md`。

---

## 能力一览

**编译与治理**

- 单文件 / 整目录编译，合并进同一份 master
- 领域规则识别制造 BOM、物流、工位作业（OIS / PII / WES）等
- 跨文件实体消解（别名表 + 模糊 + 图邻居）
- LLM 超时/无密钥时 `--degraded-offline` 纯规则路径
- 质量门禁、发布收据、`check_governed_publish`

**检索与问答**

- BM25 + 向量，Reciprocal Rank Fusion（k=60）
- 可答性预检后再答；`rag_cli meaning*` 只读含义面
- 可选 Neo4j（非编译前提）

**代理与工程**

- 只读 MCP：`search_entities` / `get_entity` / `check_answerable` / `ask_question` / `check_readiness`
- `make check`：单测 + 编译 + 回归冒烟 + LOC / import / docs / compat 门禁
- Gate A 结构保鲜：RELATED_TO、来源占比、CQ smoke（`scripts/gate_a_freshness_check.py`）

---

## 算法要点

**切块。** 按文件类型选策略（见 `docs/CHUNKING_STRATEGY_PROFILES.md`）。阈值 `CHUNK_THRESHOLD_CHARS`。

**抽取。** 模型按 packet schema 返回 JSON；失败同调用内重试。`skills/l25/` 在命中 MBOM/物流等关键词后把约束注入提示。

**实体消解（MSER）。** `src/pipeline/stages/entity_resolution.py` + `config/entity_aliases.yaml`：

1. 别名表精确命中 → 合并  
2. 子串（≥4 字符）→ 合并  
3. rapidfuzz 多指标取最高分（未装则跳过）  
4. 中间分 + 邻居/词元加权；够阈值才自动合并，否则待审  

**检索。** 治理包上建 BM25 倒排 + 本地/外部向量；两路名次 RRF 融合。

---

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install rapidfuzz   # 实体模糊合并；不装则跳过该档

cp config/env.example.env config/env.env
# 编辑：LLM_PROVIDER、对应 API key（无 key 可用 --degraded-offline）
```

```bash
# 编译
python scripts/ontology_cli.py compile \
  --file "/path/to/document.pptx" \
  --domain AUTO

# 无模型
python scripts/ontology_cli.py compile \
  --file "/path/to/document.pptx" \
  --domain SUPPLY_CHAIN \
  --degraded-offline

# 发布检索资产（若尚未由 pipeline 钩子触发）
python scripts/build_governed_packet.py
python scripts/check_governed_publish.py --require-receipt

# 先可答性，再查询
python scripts/rag_cli.py meaning answerable "MBOM 如何由 EBOM 派生"
python scripts/rag_cli.py meaning search "MBOM"
python scripts/rag_cli.py query "MBOM 如何由 EBOM 派生"

# MCP（stdio）
python -m mcp.ontology_mcp.server

# 门禁
make check
python scripts/gate_a_freshness_check.py
```

`AUTO` 会按正文判领域；制造/供应链材料通常落到 `SUPPLY_CHAIN`。

---

## 常用操作

```bash
python scripts/ontology_cli.py compile --folder "/path/to/docs" --domain AUTO
python scripts/run_batch.py --folder "/path/to/docs"

python scripts/ontology_cli.py compile \
  --file "/path/to/document.pptx" \
  --max-cost-usd 1 --max-calls 20

PIPELINE_SKILL_RELATION_REWRITE=1 \
  python scripts/ontology_cli.py compile --file "/path/to/document.pptx"

python scripts/ontology_cli.py health
python scripts/run_regression_samples.py --production --html --sample-id ebom-mbom-2023
```

回归样本登记在 `config/regression_samples.json`。

---

## 操作时注意

- **密钥只放 `config/env.env`**（勿提交）。样例是 `config/env.example.env`。
- **PPT 图抽不出文字。** 只读形状文字与备注；抽完先看 `artifacts/source-documents/` 里的 Markdown。
- **长文档多次模型调用。** 超时先调大 `CHUNK_THRESHOLD_CHARS` 减块数，而不是盲目加重视试。
- **Dify body too large：** 降 `DIFY_MAX_CONTEXT_CHARS` 或 `--topk`。
- **master 会跨次累积。** 换一批无关文档前确认是否沿用现有 master。
- **Neo4j 可选**；本地 hybrid 检索不依赖它。

---

## 结果放在哪

| 路径 | 内容 |
|---|---|
| `outputs/runs/ontology/pipeline/<run-id>/` | 本次运行摘要、源文档 Markdown |
| `outputs/runs/ontology/master/` | 累积 master |
| `outputs/runs/ontology/asset_packets/governed/<id>/` | **检索权威包**（packet、索引、收据） |
| `outputs/runs/ontology/reports/documents/` | 单文档可读报告 |

---

## 目录

```text
scripts/ontology_cli.py          编译 / 查询 / 健康检查
scripts/rag_cli.py               含义面 / 检索 / 对话
scripts/build_governed_packet.py 发布可检索资产
scripts/gate_a_freshness_check.py 业务结构保鲜
mcp/                             只读 MCP
src/pipeline/                    生产编排（MAIN）
ontology_compiler/               抽取内核
src/application/rag/             检索 + semantic_surface
skills/l25/                      领域抽取约束
skills/agent/                    代理 skills
config/                          白名单、别名、回归样本、环境样例
schemas/                         packet / 实体 / 关系契约
docs/                            仍被代码引用的说明书（见 docs/README.md）
AGENTS.md                        代理入口
tests/
```

可执行规则以 `config/` + `schemas/` 为准；`docs/` 只保留说明书层。

---

## Limitations

- 制造工程语义层的**参考实现**，不是即插即用 SaaS。
- `dag_allowed=true` 只表示并行**可行性**；六角色今日仍串行。
- MCP 为本地 stdio、无鉴权；勿暴露到不可信网络。
- 全量业务回归需要真实 LLM；离线可跑 `make check` 与 Gate A 结构检查。
