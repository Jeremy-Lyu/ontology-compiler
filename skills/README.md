# Skills 目录 - 构建指南

> **版本**: 2026-06-27  
> **架构**: 业务域聚合 + 功能内聚  
> **目标**: 避免无限扩展，保持可维护性

---

## 目录结构 (重组后 - 2026-06-27)

```
skills/
├── README.md                      # 本文件 - 构建指南
├── SKILL_REGISTRY.md              # 技能注册清单 (自动生成)
├── agent/                         # 代理入口三件套（见根目录 AGENTS.md）
│   ├── ontology-setup/
│   ├── ontology-management/
│   └── ontology-query/
├── __init__.py                    # 包初始化
├── skill_interface.py             # BaseSkill 接口
├── skill_metadata.py              # 技能元数据模型
├── skill_registry.py              # 技能注册中心
├── skill_router.py                # 技能路由
├── skill_memory.py                # 技能记忆
├── multi_agent_orchestrator.py    # 多Agent编排
├── mfg/                           # 制造域 (3 skills)
│   ├── bom-analyzer/              # BOM多时间序列分析
│   ├── build-inventory-field-catalog/
│   └── scene-analysis/
├── ontology/                      # 本体编译域 (5 skills)
│   ├── consensus-resolver/
│   ├── drift-detector/
│   ├── generate-ontology-patch/
│   ├── ontology-merger/
│   └── ontology-validator/
├── pipeline/                      # 管道编排域 (1 skill)
│   └── stage-runner/              # 统一 Stage 管理
├── rag/                           # RAG检索域 (4 skills)
│   ├── complex-rag-orchestrator/
│   ├── multi-agent-analysis/
│   ├── run-ontology-rag-eval/
│   └── run-ontology-rag-validate/
└── system/                        # 系统工具域 (5 skills)
    ├── capsule-publisher/
    ├── check-config/
    ├── cleanup-comp-runs/
    ├── switch_llm_provider/
    └── validate-output-contracts/
```

### 统计信息 (2026-06-27)

| 业务域 | Skills 数量 | 主要功能 |
|--------|-------------|---------|
| `mfg/` | 3 | BOM分析、库存管理、场景分析 |
| `ontology/` | 5 | 本体验证、合并、漂移检测 |
| `pipeline/` | 1 | 管道编排 |
| `rag/` | 4 | RAG检索、多Agent分析 |
| `system/` | 5 | 配置检查、发布、清理 |
| **总计** | **18** | - |

---

## 核心原则

### 1. 业务域聚合

**准则**: 相同业务域的技能放入同一目录

| 业务域 | 包含内容 | 示例 |
|--------|----------|------|
| `mfg/` | 制造相关 | BOM、库存、场景分析 |
| `ontology/` | 本体编译 | 合并、验证、漂移检测 |
| `pipeline/` | 管道编排 | Stage 管理 |
| `rag/` | 检索增强 | RAG评估、验证 |
| `system/` | 系统工具 | 发布、配置、清理 |

### 2. 功能内聚合

**准则**: 功能相似的技能合并为一个

**案例**: 7个 `stage-*` → 1个 `stage-runner`
```json
{
  "skill_id": "stage-runner",
  "sub_skills": [
    {"stage": "aggregate"},
    {"stage": "catalog"},
    {"stage": "guard"},
    {"stage": "health"},
    {"stage": "preflight"},
    {"stage": "publish"},
    {"stage": "rollback"}
  ]
}
```

### 3. 避免无限扩展

**限制规则**:
- 顶层业务域 ≤ 6 个
- 每个业务域内 skills ≤ 8 个
- 新 skill 必须先评估: 合并? 替换? 还是新增?

**新增决策流程**:
```
新技能需求
    ↓
是否有现有技能可扩展? → 是 → 扩展现有技能
    ↓ 否
是否属于现有业务域? → 是 → 放入对应域
    ↓ 否
是否需要新业务域? → 评估必要性 → 创建新业务域(需审批)
```

---

## Skill 规范

### 目录结构标准

```
skills/<domain>/<skill-name>/
├── .ontology-skill/
│   └── skill.json          # Skill 定义 (必需)
├── SKILL.md                # 使用文档 (必需)
├── scripts/                # 执行脚本 (可选)
└── tests/                  # 测试用例 (可选)
```

### skill.json 模板

```json
{
  "skill_id": "{domain}-{action}",
  "name": "{Domain} {Action}",
  "description": "{清晰描述技能用途}",
  "version": "1.0.0",
  "category": "{mfg|ontology|pipeline|rag|system}",
  "tags": ["tag1", "tag2"],
  "trigger": {
    "keywords": ["keyword1"],
    "file_patterns": ["*.csv"],
    "min_confidence": 0.7
  },
  "input_schema": {
    "type": "object",
    "required": ["param1"],
    "properties": {}
  },
  "output_schema": {
    "type": "object"
  },
  "scripts": {
    "main": "scripts/{skill_id}.py"
  }
}
```

---

## 使用方式

### 1. 通过 SkillRegistry 调用

```python
from ontology_compiler.harness import build_default_skill_registry

registry = build_default_skill_registry()

# 获取 skill
skill = registry.get_skill("bom-analyzer")

# 执行
result = skill.execute({
    "csv_files": ["TBS130_202606010000.csv"],
    "analysis_type": "full_analysis"
})
```

### 2. 直接脚本调用

```bash
# BOM分析
python -m scripts.diagnostics.run_bom_timeseries_analysis \
    --csv-dir Sample_data/TBS130/ \
    --analysis-type full_analysis

# Stage 执行
python -m scripts.pipeline.stage_runner --stage aggregate --run-dir outputs/run_001
```

### 3. Workflow 编排调用

```python
from skills.pipeline.stage_runner import StageRunner

runner = StageRunner()
runner.execute_stages([
    "preflight",
    "catalog",
    "aggregate",
    "guard",
    "publish"
])
```

---

## 重组映射 (2026-06-11)

| 原位置 | 新位置 | 操作 |
|--------|--------|------|
| `bom-analysis/` | `mfg/bom-analyzer/` | 移动 + 整合其他BOM技能 |
| `bom-change-analyzer/` | → 并入 `mfg/bom-analyzer/` | 合并 |
| `bom-hybrid-analyzer/` | → 并入 `mfg/bom-analyzer/` | 合并 |
| `build-inventory-field-catalog/` | `mfg/inventory-analyzer/` | 移动 + 重命名 |
| `scene-analysis/` | `mfg/scene-analyzer/` | 移动 |
| `stage-*/` (7个) | `pipeline/stage-runner/` | 合并 |
| `consensus-resolver/` | `ontology/consensus-resolver/` | 移动 |
| `drift-detector/` | `ontology/drift-detector/` | 移动 |
| `generate-ontology-patch/` | `ontology/generate-ontology-patch/` | 移动 |
| `ontology-merger/` | `ontology/ontology-merger/` | 移动 |
| `ontology-validator/` | `ontology/ontology-validator/` | 移动 |
| `complex-rag-orchestrator/` | `rag/complex-rag-orchestrator/` | 移动 |
| `run-ontology-rag-eval/` | `rag/run-ontology-rag-eval/` | 移动 |
| `run-ontology-rag-validate/` | `rag/run-ontology-rag-validate/` | 移动 |
| `capsule-publisher/` | `system/capsule-publisher/` | 移动 |
| `check-config/` | `system/check-config/` | 移动 |
| `cleanup-comp-runs/` | `system/cleanup-comp-runs/` | 移动 |
| `validate-output-contracts/` | `system/validate-output-contracts/` | 移动 |
| `example_skills/` | `examples/example-skills/` | 移动 + 重命名 |
| `multi-agent-analysis/` | `archived/skills_20260611/` | 归档 |
| `switch_llm_provider/` | `archived/skills_20260611/` | 归档 |

---

## 维护指南

### 添加新 Skill

1. **评估阶段**
   ```bash
   # 检查是否有相似技能
   ls skills/*/* | grep {keyword}
   
   # 检查业务域归属
   cat skills/README.md | grep {domain}
   ```

2. **创建阶段**
   ```bash
   # 使用模板创建
   cp -r skills/examples/example-skills skills/{domain}/{new-skill}
   
   # 编辑定义
   vim skills/{domain}/{new-skill}/.ontology-skill/skill.json
   vim skills/{domain}/{new-skill}/SKILL.md
   ```

3. **注册阶段**
   ```python
   # 在 ontology_compiler/harness/__init__.py 中注册
   from skills.{domain}.{new_skill} import NewSkill
   registry.register(NewSkill())
   ```

### 归档旧 Skill

```bash
# 移动至归档区
mv skills/{old-skill} skills/archived/skills_20260611/

# 更新 IMPROVEMENT_EXECUTION_TRACKER.md
echo "- {date}: 归档 {old-skill} → {reason}" >> docs/IMPROVEMENT_EXECUTION_TRACKER.md
```

### 合并 Skills

```bash
# 1. 创建目标 skill
mkdir -p skills/{domain}/{merged-skill}

# 2. 复制主要 skill 内容
cp -r skills/{domain}/{primary-skill}/* skills/{domain}/{merged-skill}/

# 3. 整合其他 skill 的功能
# 编辑 skill.json 添加 sub_skills 或 additional_capabilities

# 4. 归档被合并的 skills
mkdir -p skills/archived/{date}
mv skills/{domain}/{merged-skill-1} skills/archived/{date}/
mv skills/{domain}/{merged-skill-2} skills/archived/{date}/
```

---

## 最佳实践

### DO
- ✅ 按业务域组织 skills
- ✅ 功能相似则合并
- ✅ 每个 skill 必须有 SKILL.md
- ✅ 使用语义化的 skill_id (kebab-case)
- ✅ 在 skill.json 中定义清晰的 input/output schema

### DON'T
- ❌ 不要为每个小功能创建独立 skill
- ❌ 不要将 skill 放在业务域根目录 (必须放入子目录)
- ❌ 不要遗漏 trigger 定义 (影响自动路由)
- ❌ 不要硬编码路径 (使用相对路径或配置)

---

## 关联文档

| 文档 | 路径 | 说明 |
|------|------|------|
| 改进追踪器 | `docs/IMPROVEMENT_EXECUTION_TRACKER.md` | 重组记录 |
| Skill 接口 | `skills/skill_interface.py` | BaseSkill 定义 |
| Skill Registry | `skills/skill_registry.py` | 注册中心 |
| Skill Router | `skills/skill_router.py` | 路由逻辑 |
| 经验萃取 | `.notes/Context_initialization/bom_analysis_experience.md` | BOM分析方法论 |

---

## 变更记录

| 日期 | 变更 | 执行者 |
|------|------|--------|
| 2026-06-11 | Skills目录重组: 21个 → 17个，按业务域聚合 | System |
| 2026-06-11 | 合并BOM相关skills: 3个 → 1个 bom-analyzer | System |
| 2026-06-11 | 合并Stage skills: 7个 → 1个 stage-runner | System |
| 2026-06-11 | 创建本构建指南 | System |

---

*本指南作为 Skills 体系的核心文档，任何变更需同步更新。*
