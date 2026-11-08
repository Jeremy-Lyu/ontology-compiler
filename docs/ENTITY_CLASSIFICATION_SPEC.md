# ENTITY_CLASSIFICATION_SPEC — 实体分类与建模规范（重建 v1）

> 状态：**2026-08-26 重建，2026-10-20 补回原文**。2026-06-27 的判定顺序、主类特征和红线
> 在 ENTITY_CLASSIFICATION_SPEC_20260627（归档原文）。本文仍是代码消费的说明书：字段、层级、
> 时间纪律和 OntologyPacket 以本文为准。本文档依据以下现存资产与
> 方法论参考建立，作为实体分类/建模的**单一事实来源（说明书层）**：
>
> - ENTITY_CLASSIFICATION_REFACTOR_PLAN（归档，未随公开包发布）（资产化拆解方案）
> - ENTITY_CLASSIFICATION_MAINLINE_PLAN（归档，未随公开包发布）（主线计划）
> - `schemas/ontology_packet.schema.json`（抽取输出协议契约）
> - `schemas/ontology_entity.schema.yaml` + `src/domain/models/ontology_entity.py`（实体契约真值）
> - `config/ontology/relationship_rules.yaml`（关系族白名单/层对规则/证据分级）
> - 方法论参考：CQ 驱动设计、最少必要集、事件对象化/版本区间/三时间纪律（见
>   CODE_CLEANUP_CANDIDATES（归档，未随公开包发布） 同期的 Downloads 参考材料归纳）
>
> 分工原则（同 REFACTOR_PLAN）：**MD 只做语义说明，可执行规则一律在 YAML/schema/skill 资产中。**

---

## 1. 三条建模铁律

1. **节点只表达"是什么"**：分类（type/sub_type）、层级（layer）、自身属性。
2. **关系只表达"如何关联"**：支撑/实现/归属/映射/时态/依赖/追溯——禁止用关系承载属性语义。
3. **属性只表达"补充语义"**：编码/版本/状态/来源/证据/时效——禁止把查询结果落成属性。

## 2. 层级定义

| 层 | 名称 | 职责 | 说明 |
|---|---|---|---|
| L0 | 主类 | Capability / System / Actor / Asset / Data / Process / Concept 等 | 见 `config/ontology/entity_schema.yaml` `l0_classes` |
| L1 | 标准子类 | 主类下受控子类枚举 | 同上 `l1_subclasses` |
| L2 | 业务实体层 | 具体业务对象/场景 | 不同模块不得私自改写 L2 语义 |
| **L2.5** | **桥接层** | **映射/转换/规则/聚合/族群-模板-批次桥接** | 不是新主类；只产 rule/mapping/constraint 类节点 |
| L3 | 实例层 | 最终实例 | 必须挂到 L2/L2.5 父节点 |

### 2.1 layer_scope（强制）

每个实体/关系必须携带 `layer_scope`，二选一：

- `spec` —— 业务本体语义（进图谱、进 RAG 的部分）
- `impl` —— 抽取实现元数据（运行时辅助，不进业务视图）

目的：防止业务本体与实现元数据混写，保证后续维护与演化边界清晰。

## 3. 实体字段契约

唯一真值：`src/domain/models/ontology_entity.py` + `schemas/ontology_entity.schema.yaml`；
抽取输出协议中的实体定义见 `schemas/ontology_packet.schema.json` `$defs.entity`。

最小必要字段（新增字段必须先过"删除测试"——去掉它任何一条 CQ 都答不了才允许加）：

| 字段 | 语义 |
|---|---|
| id | 稳定标识（meta_id） |
| code | 业务编码（part_number/bom_id 等） |
| name | 规范名 |
| layer / layer_scope | 层级 / spec\|impl |
| type / sub_type | 主类 / 子类 |
| status / version | 生命周期状态 / 版本 |
| parent_id | 父节点 |
| source_file / source_span / evidence_ref | 溯源三件套 |
| confidence | 置信度 |
| attributes | 补充语义（open dict） |

## 4. 关系契约

- 类型必须来自 `config/ontology/relationship_rules.yaml` 白名单（7 族：STRUCTURAL /
  REALIZATION / GOVERNANCE / DEPENDENCY_TRACE / CONFIG_EFFECTIVITY / CHANGE_FLOW / GENERAL），
  按 `rewrite_priorities` 优先结构关系、后语义、后映射、最后兜底（RELATED_TO 仅兜底）。
- 每条关系必带：`confidence`、`evidence_ref`；时态关系另带 `effective_from/to`。
- `provenance` 枚举：`llm | skill | rule | inferred`。
- 层间合法性由 `relationship_rules.yaml` 的 `layer_pair_rules` 判定（与 QualityGate 镜像同一配置）。

## 5. 时间与版本纪律（五条模型纪律）

1. **事件对象化**：ECN/C-Notice 是实体，不是备注字段；
2. **版本区间**：所有版本带 valid_from/to，查询落在区间上；
3. **三时间独立**：T1 计划切换日 / T2 供应商供货日 / T3 实际装车日 三列独立，禁止合并；
   规范字段名（packet entity 可选属性）：`t1_planned_switch_date` / `t2_supplier_delivery_date` /
   `t3_actual_install_date`；版本区间用 `valid_from`/`valid_to`（业务有效时间）。
   注意区分：pipeline master 上由 `_populate_temporal_fields` 盖的运行时间戳属于**事务时间**（impl 侧），
   与上述业务有效时间（spec 侧）语义不同，禁止互相覆盖。
4. **事实锚**：以 VIN 装配记录为事实锚，理论 BOM 与实际装车分离；
5. **视图派生**：月用量/加权成本等聚合是查询结果（视图），不落库为属性。

## 6. 抽取输出协议（OntologyPacket）

顶层结构（真值：`schemas/ontology_packet.schema.json`）：

```
document, layer_scope, entities, relations, evidence, provenance, quality, warnings, assumptions
```

entities 内按 L0/L1/L2/L2.5/L3 分桶；entity/relation 字段见 §3/§4。

## 7. 消费方式

| 消费方 | 使用字段 |
|---|---|
| pipeline 校验 | 全字段（contract_registry 注册） |
| skill governance | relation_type 白名单 / 必填字段 |
| RAG 索引 | id/name/type/sub_type/description/evidence_ref/aliases/tags/relations_summary |
| 审计回放 | provenance / source_span / version / confidence |

## 8. Schema 演化治理

- 新增字段/子类 = 低风险（向后兼容）；修改语义 = 中风险（需迁移说明）；
- 删除/改类型 = 高风险（破坏性变更需一次性迁移 + characterization 测试更新）；
- 所有 schema 资产改动必须同步本文件的对应章节引用。

---

## 9. 代码怎么消费本文件

运行时**不解析**本 Markdown。本文件规定字段语义；可执行规则在 schema / YAML / 下面两个纯函数里。

| 本文件 | 代码 |
|---|---|
| §2 层级，§2.1 `layer_scope` 必带 | `review_ontology_packet`：`layer` 不在 L0/L1/L2/L2.5/L3 为 error；实体或关系缺 `layer_scope` 为 warning |
| §3 最小字段 | 硬错误只查 `id` / `name` / `layer` / `type`。`code`、`parent_id`、溯源三件套由 adapter 填空，不单独挡发布 |
| §3 字段名 | 抽取中间态仍可能是 `meta_id` / `meta_name` / `meta_class`。质量规则 `FieldCompletenessRule` 把它们当成同一字段，不另建一套必填名 |
| §4 关系 | `relation_type` 与两端点缺失为 error；端点对不上实体为 warning；缺 `confidence` 或 `evidence_ref` 为 warning；`provenance` 越出 `llm\|skill\|rule\|inferred` 为 error |
| §5 三时间 | `extraction_packet_adapter._TEMPORAL_ALIASES` 把别名收成 `t1_planned_switch_date` / `t2_supplier_delivery_date` / `t3_actual_install_date`，三列互不合并 |
| §6 packet | `adapt_extraction_json_to_packet` 把 `L*_entities` 收成 `entities[].layer` |
| 2026-06-27 原文的判定顺序和红线 | 不执行。主类枚举以 `config/ontology/entity_schema.yaml` 为准 |

`docs/architectures/ENTITY_CLASSIFICATION_SPEC.md` 这条路径不存在。引用一律写 `docs/ENTITY_CLASSIFICATION_SPEC.md`。

## 关联文档

- ENTITY_CLASSIFICATION_SPEC_20260627（归档原文） — 找回的原文（判定顺序 / 主类特征 / 红线）
- ENTITY_CLASSIFICATION_ROADMAP（归档，未随公开包发布） — 采用路线图（当前状态与任务）
- ENTITY_CLASSIFICATION_REFACTOR_PLAN（归档，未随公开包发布） — 资产化拆解方案
- CODE_CLEANUP_CANDIDATES（归档，未随公开包发布） — 遗留代码退役台账

## 附录 R：旧仓研究洞察摘录（2026-10-13，源 `ontology_2026/deepseek_tools/` 已归档）

> 源文件已整体归档（不删除），此处只摘仍有效的两条结论。原文见归档副本。

### R.1 L2.5 定位（源 `OEM_LAYERING_AND_PALANTIR_INSIGHTS.md` §1.2/§2.2）

- L2.5 = **实例桥接层 / 配置激活层**：配置激活、实例映射、某 VIN / 某 BOM / 某版本下的语义落地。
- 三段式归属：表结构 → L1（数据长什么样）；字段语义 → L2（业务含义）；
  字段实例激活 → L2.5（某版本下的落地值）。结构、语义、实例不混层。
- L3 专注治理与发布，不吸收业务解释职责。

现状对照：本 SPEC §L2.5 定义为"桥接层（映射/转换/规则）"，与洞察一致；
"字段实例激活"语义尚未显式写入，待补。

### R.2 双层级命名冲突（源 `ONTOLOGY_AND_PIPELINE_NAMING_AND_CLASSIFICATION_GUIDE.md` §1）

- 本体层级（L0/L1/L2/L3）与数据处理流程层级**不应同名**，当前混用是已知整改项。
- 改名前：新文档统一用"本体 Lx" vs "流程阶段"区分表述，避免歧义。

现状对照：整改未做，记为已知问题（不阻塞）。
