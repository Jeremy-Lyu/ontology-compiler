"""Governance Module - 统一治理层

根据 PROJECT_ARCHITECTURE_AUDIT_REPORT_20260708.md 问题 P9 的要求：
- 将 audit、guard、quality_policy、relationship governance 收束到一个治理层

核心组件：
- QualityGate: 统一质量门禁
- GovernanceRulesEngine: 治理规则引擎

设计原则：
- 判定口径一致：所有质量判定使用同一套规则
- 统一入口：所有治理检查通过单一接口
- 可配置：阈值和规则可通过配置文件调整
- 可追溯：每次检查记录详细信息，便于问题追踪
"""
from __future__ import annotations

from .quality_gate import (
    QualityGate,
    GovernanceRulesEngine,
    QualityGrade,
    GateResult,
    IssueSeverity,
    IssueCategory,
    QualityIssue,
    QualityScore,
    QualityGateReport,
    get_governance_engine,
)

__all__ = [
    "QualityGate",
    "GovernanceRulesEngine",
    "QualityGrade",
    "GateResult",
    "IssueSeverity",
    "IssueCategory",
    "QualityIssue",
    "QualityScore",
    "QualityGateReport",
    "get_governance_engine",
]