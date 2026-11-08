# AGENTS.md — 代理入口（含义优先）

给要把本仓库当语义层用的代理 / 人类协作者。

## 该读哪条 skill

| 场景 | Skill |
|------|--------|
| 环境未就绪、要跑 compile / 建 governed packet | [`skills/agent/ontology-setup/`](skills/agent/ontology-setup/) |
| 检查含义、门禁、认证与发布收据 | [`skills/agent/ontology-management/`](skills/agent/ontology-management/) |
| 提问、可答性预检、证据回答 | [`skills/agent/ontology-query/`](skills/agent/ontology-query/) |

生命周期：安装就绪 → 语义检查/门禁 → 提问与验证 → 有缺口就停，不要编造。

## 真值表（不要在 skill 里复制）

| 职责 | 唯一真值 | Skill 只做 |
|------|----------|------------|
| 文档入口 | `docs/README.md` | 指路，不抄全文 |
| 实体分类协议 | `docs/ENTITY_CLASSIFICATION_SPEC.md` | 引用条款 |
| 关系白名单 | `config/ontology/relationship_rules.yaml` | 引用，不内嵌 |
| Packet schema | `schemas/ontology_packet.schema.json` | 命名字段 |
| 生产路由 | `src/pipeline` MAIN + `scripts/ontology_cli.py` | 调用命令 |
| 检索资产 | governed packet（`scripts/build_governed_packet.py`） | 走 `rag_cli meaning/*` |
| 语义只读面 | `src/application/rag/semantic_surface.py` | 调用，不 dump master JSON |

## 含义优先规则

1. **先问能不能答**：`python scripts/rag_cli.py meaning answerable "..."`；不够就报缺口。
2. **只读含义**：`meaning search` / `meaning get`；禁止把 `master_ontology/*.json` 或整包 `packet.json` 当默认工具输出。
3. **写入走 CLI / 管道**：MCP/代理只读；编译与发布用 `ontology_cli` / `build_governed_packet.py`。
4. **发布要有收据**：`publish_receipt.json`；`certified` 只能在门禁通过且人工确认后设置。
5. **零件号在 L3**：实例型编码不得停在 L0/L1（质量规则 `layer_instance_discipline`）。

## 最小命令

```bash
# 就绪
python scripts/ontology_cli.py health
python scripts/build_governed_packet.py
python scripts/check_governed_publish.py --require-receipt

# 含义
python scripts/rag_cli.py meaning search "MBOM"
python scripts/rag_cli.py meaning answerable "MBOM 如何由 EBOM 派生"
python scripts/rag_cli.py query "MBOM 如何由 EBOM 派生"

# MCP（stdio，只读；Cursor 配置见 mcp/README.md）
python -m mcp.ontology_mcp.server

# Gate A 结构保鲜（无 LLM；全量回归另跑 make regression）
python scripts/gate_a_freshness_check.py
python scripts/analyze_master_origins.py
```
