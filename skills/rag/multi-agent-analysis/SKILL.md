# Multi-Agent Analysis Skill

## Description

Provides **unified analysis interface** that automatically routes queries to appropriate processing path:

- **Simple queries** (knowledge lookup): Direct RAG retrieval
- **Complex queries** (analysis/design): Multi-Agent orchestration (RAG + Data + Synthesis)

## When to Use

Use for complex analytical questions that require:
- Multi-system data correlation (e.g., OHF + PLUS + Partcache)
- Hypothesis validation with actual data
- Design recommendations (data models, process optimization)

Examples:
- "如何设计排产物料数据模型支持多场景？"
- "分析当前Call-off管理的问题和风险"
- "C-Note变更如何影响排产和物料计划？"

## Commands

> **Note**: `scripts/multi_agent_cli.py` 尚未实现，当前通过 `src/agent/orchestration/orchestrator.py` 编程式调用。

```python
# 编程式调用示例
from src.agent.orchestration.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator()
task = orchestrator.submit_task("analysis-001", "分析当前Call-off管理的问题和风险")
result = orchestrator.complete_task("analysis-001", output={"conclusions": [...], "confidence": 0.85})
```

## Architecture

```
User Query
    ↓
IntentClassifier (simple vs complex)
    ↓
    ├─ simple → LLMAnswerer (existing RAG)
    └─ complex → Orchestrator (3-round analysis)
                    ↓
        Round 1: RAG Agent (knowledge retrieval)
        Round 2: Data Agent (Layer 1/2 validation)
        Round 3: Synthesis Agent (conclusions)
                    ↓
              Structured Report
```

## Output

- Console: Markdown report with conclusions and confidence scores
- File (with `--save`): `outputs/multi_agent_analysis/analysis_<query>_<date>.md`

## Dependencies

- `src/application/rag/llm_answerer.py` (existing)
- `src/agent/multi_agent/` (new)
- `outputs/ontology/rag/` (RAG index)

## P0 Implementation

Current version (P0) provides:
- ✅ Intent classification (simple/complex)
- ✅ 3-round sequential orchestration
- ✅ RAG knowledge retrieval (reuses LLMAnswerer)
- ✅ Stub data validation (Layer 1/2 structure, awaits real data integration)
- ✅ Structured report generation

P1 Planned:
- Real Excel/CSV data integration for Data Agent
- Reviewer Agent for confidence validation
- Adaptive follow-up query generation
