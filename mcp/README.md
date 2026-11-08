# Ontology MCP（只读语义面）

最小 MCP stdio 服务，给 Cursor / 其他 MCP 客户端用。**无写路径、无鉴权**（本机 stdio 信任模型）。对外多租户前须另加鉴权。

## 工具

| 工具 | 作用 |
|------|------|
| `search_entities` | 按名称/别名/类型检索 |
| `get_entity` | 按 id 取实体 + 邻接关系 |
| `check_answerable` | 提问前可答性预检 |
| `check_readiness` | governed pointer / publish_receipt |
| `ask_question` | **经可答性门禁后再答**；不够则 `refused=true` + gaps，不编造 |

`ask_question` 返回证据草图（实体/关系类型），不调 LLM。生成式 RAG 仍用 `python scripts/rag_cli.py query "..."`（同样默认先过可答性门禁）。

## 启动

```bash
python -m mcp.ontology_mcp.server
```

## Cursor 配置

在 Cursor MCP 设置（或项目 `.cursor/mcp.json`）里注册 stdio 服务，把 `cwd` 指到本仓库根目录：

```json
{
  "mcpServers": {
    "ontology": {
      "command": "python",
      "args": ["-m", "mcp.ontology_mcp.server"],
      "cwd": "/ABS/PATH/TO/ontology_20260702"
    }
  }
}
```

建议代理顺序：`check_readiness` → `check_answerable` / `ask_question` → 需要细节再用 `search_entities` / `get_entity`。

## 边界

- 只读；编译/发布走 `ontology_cli` / `build_governed_packet.py`
- 不 dump `master_ontology/*.json` 或整包 `packet.json`
- 无网络鉴权；勿把 stdio 端口暴露到不可信网络
