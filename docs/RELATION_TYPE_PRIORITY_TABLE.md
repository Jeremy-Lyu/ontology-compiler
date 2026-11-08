# Relation Type Priority Table

版本：v1.0
日期：2026-07-31

> 本文用于统一关系类型优先级与 rewrite 策略。
>
> 配套实现：`config/ontology/relationship_rules.yaml`、`ontology_compiler/services/relation_governance.py`

---

## 1. 优先级原则

关系抽取与重写遵循：

1. 结构关系优先
2. 实现/实现化关系次之
3. 治理/归属关系再次之
4. 依赖/证据链关系再次之
5. 配置/生效关系再次之
6. 变更流关系再次之
7. `RELATED_TO` 只作为兜底

---

## 2. 核心优先级表

| 优先级 | 关系类型 | 族 | 说明 |
|--------|----------|----|------|
| 1 | `CONTAINS` | STRUCTURAL | 结构包含关系 |
| 1 | `HAS_PART` | STRUCTURAL | 部分-整体关系 |
| 1 | `PART_OF` | STRUCTURAL | 反向组成关系 |
| 1 | `HAS_VARIANT` | STRUCTURAL | 变体/配置结构关系 |
| 1 | `DECOMPOSES_TO` | STRUCTURAL | 分解关系 |
| 2 | `REALIZES` | REALIZATION | 能力/设计的实现关系 |
| 2 | `DEPLOYS_TO` | REALIZATION | 部署到关系 |
| 2 | `SUPPORTS` | REALIZATION | 支持关系 |
| 2 | `EXECUTES` | REALIZATION | 执行关系 |
| 2 | `CONNECTED_TO` | REALIZATION | 连接关系 |
| 2 | `INTEGRATES_WITH` | REALIZATION | 集成关系 |
| 3 | `OWNS` | GOVERNANCE | 拥有/归属关系 |
| 3 | `MANAGES` | GOVERNANCE | 管理关系 |
| 3 | `MAINTAINS` | GOVERNANCE | 维护关系 |
| 3 | `ASSIGNED_TO` | GOVERNANCE | 指派关系 |
| 3 | `RESPONSIBLE_FOR` | GOVERNANCE | 负责关系 |
| 3 | `PARTICIPATES_IN` | GOVERNANCE | 参与关系 |
| 4 | `DEPENDS_ON` | DEPENDENCY_TRACE | 依赖关系 |
| 4 | `USES` | DEPENDENCY_TRACE | 使用关系 |
| 4 | `USED_IN` | DEPENDENCY_TRACE | 被用于关系 |
| 4 | `REFERENCES` | DEPENDENCY_TRACE | 引用关系 |
| 4 | `TRACES_TO` | DEPENDENCY_TRACE | 追溯关系 |
| 4 | `VERIFIES` | DEPENDENCY_TRACE | 验证关系 |
| 4 | `VALIDATES` | DEPENDENCY_TRACE | 校验关系 |
| 4 | `SATISFIES` | DEPENDENCY_TRACE | 满足关系 |
| 4 | `CONFORMS_TO` | DEPENDENCY_TRACE | 符合关系 |
| 4 | `COMPLIES_TO` | DEPENDENCY_TRACE | 遵从关系 |
| 5 | `APPLIES_TO` | CONFIG_EFFECTIVITY | 生效/适用关系 |
| 5 | `CONFIGURED_BY` | CONFIG_EFFECTIVITY | 配置驱动关系 |
| 5 | `EFFECTIVE_FOR` | CONFIG_EFFECTIVITY | 对象生效关系 |
| 5 | `OPTION_FOR` | CONFIG_EFFECTIVITY | 选项关系 |
| 5 | `SUBSTITUTES` | CONFIG_EFFECTIVITY | 替代关系 |
| 5 | `REPLACES` | CONFIG_EFFECTIVITY | 替换关系 |
| 5 | `SUPERSEDES` | CONFIG_EFFECTIVITY | 取代关系 |
| 5 | `ALIGNED_WITH` | CONFIG_EFFECTIVITY | 对齐关系 |
| 6 | `MODIFIES` | CHANGE_FLOW | 变更关系 |
| 6 | `DERIVED_FROM` | CHANGE_FLOW | 派生关系 |
| 6 | `GENERATES` | CHANGE_FLOW | 生成关系 |
| 6 | `RESULTS_IN` | CHANGE_FLOW | 结果关系 |
| 6 | `TRIGGERS` | CHANGE_FLOW | 触发关系 |
| 6 | `TRANSFORMS_TO` | CHANGE_FLOW | 转换关系 |
| 9 | `RELATED_TO` | GENERAL | 仅兜底 |

---

## 3. RELATED_TO 重写规则

当满足以下条件时，优先重写 `RELATED_TO`：

- 结构可判断时 → `CONTAINS` / `PART_OF` / `HAS_PART`
- 明确实现或运行依赖时 → `REALIZES` / `EXECUTES` / `DEPLOYS_TO`
- 明确归属/职责时 → `OWNS` / `MANAGES` / `ASSIGNED_TO`
- 明确引用/证据链时 → `REFERENCES` / `TRACES_TO` / `VALIDATES`
- 明确配置或生效时 → `APPLIES_TO` / `EFFECTIVE_FOR`
- 明确变更来源时 → `DERIVED_FROM` / `MODIFIES` / `TRANSFORMS_TO`

如果无法判断，则保留为 `RELATED_TO`，并必须写入：
- `rewrite_reason`
- `confidence`
- `evidence_ref`

---

## 4. 关系治理规则

- `RELATED_TO` 占比必须持续下降
- 同一对实体之间，优先保留最具体关系
- 具体关系冲突时，以证据更强者为准
- 关系必须符合 `relation_rules.yaml` 中的 family / layer pair / evidence 约束

---

## 5. 运行时检查建议

每次关系治理阶段输出：
- `relation_type_distribution`
- `related_to_share`
- `rewrite_count`
- `dropped_count`
- `family_distribution`

---

## 6. 更新规则

- 新增关系类型时，先补本文优先级表
- 再补 `config/ontology/relationship_rules.yaml`
- 再补测试样本与回归样本
