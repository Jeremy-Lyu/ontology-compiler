"""
Enhanced Documentation Service for Pipeline Output

Improves documentation rationality and clarity across L0-L2.5 layers
with structured reporting and actionable insights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import json
from datetime import datetime
import re


class DocumentationLevel(Enum):
    """Documentation detail levels"""
    SUMMARY = "summary"
    DETAILED = "detailed" 
    COMPREHENSIVE = "comprehensive"


class LayerDocumentationType(Enum):
    """Types of layer-specific documentation"""
    L0_ARCHITECTURE = "L0_architecture"
    L1_DOMAIN = "L1_domain"
    L2_SCENARIOS = "L2_scenarios"
    L25_ANCHORS = "L25_anchors"


@dataclass
class LayerMetrics:
    """Metrics for a specific layer"""
    layer: str
    entity_count: int
    relationship_count: int
    quality_score: float
    accuracy_score: float
    coverage_score: float
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class CrossLayerInsight:
    """Cross-layer analysis insights"""
    insight_type: str
    description: str
    affected_layers: List[str]
    impact_level: str  # HIGH, MEDIUM, LOW
    actionable_recommendation: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class DocumentationReport:
    """Complete documentation report"""
    timestamp: str
    pipeline_version: str
    layer_metrics: Dict[str, LayerMetrics]
    cross_layer_insights: List[CrossLayerInsight]
    overall_quality_grade: str
    executive_summary: str
    detailed_analysis: str
    next_steps: List[str]
    appendices: Dict[str, Any] = field(default_factory=dict)


class EnhancedDocumentationService:
    """
    Service for generating enhanced, rational documentation output.
    
    Features:
    - Layer-specific structured documentation
    - Cross-layer insights and analysis
    - Actionable recommendations
    - Executive summaries for different stakeholders
    - Quality trend analysis
    """
    
    def __init__(self):
        self._quality_thresholds = {
            "excellent": 0.9,
            "good": 0.75,
            "acceptable": 0.6,
            "needs_improvement": 0.4
        }
    
    def generate_documentation_report(
        self,
        layer_data: Dict[str, Any],
        quality_metrics: Dict[str, Any],
        level: DocumentationLevel = DocumentationLevel.DETAILED
    ) -> DocumentationReport:
        """
        Generate comprehensive documentation report.
        
        Args:
            layer_data: Data for each layer (L0, L1, L2, L2.5)
            quality_metrics: Quality metrics for each layer
            level: Documentation detail level
            
        Returns:
            Complete documentation report
        """
        layer_metrics = self._extract_layer_metrics(layer_data, quality_metrics)
        cross_layer_insights = self._generate_cross_layer_insights(layer_metrics, layer_data)
        overall_grade = self._calculate_overall_grade(layer_metrics)
        executive_summary = self._generate_executive_summary(layer_metrics, cross_layer_insights)
        detailed_analysis = self._generate_detailed_analysis(layer_metrics, cross_layer_insights, level)
        next_steps = self._generate_next_steps(cross_layer_insights, layer_metrics)
        appendices = self._build_appendices(layer_data, quality_metrics, layer_metrics, cross_layer_insights)

        return self._build_documentation_report(
            layer_metrics=layer_metrics,
            cross_layer_insights=cross_layer_insights,
            overall_grade=overall_grade,
            executive_summary=executive_summary,
            detailed_analysis=detailed_analysis,
            next_steps=next_steps,
            appendices=appendices,
        )
    
    def _extract_layer_metrics(self, layer_data: Dict[str, Any], quality_metrics: Dict[str, Any]) -> Dict[str, LayerMetrics]:
        """Extract and normalize metrics for each layer."""
        return self._collect_layer_metrics(layer_data, quality_metrics)

    def _collect_layer_metrics(self, layer_data: Dict[str, Any], quality_metrics: Dict[str, Any]) -> Dict[str, LayerMetrics]:
        layer_metrics = {}

        for layer_name in ["L0", "L1", "L2", "L2.5"]:
            layer_info = layer_data.get(layer_name, {})
            quality_info = quality_metrics.get(layer_name.lower().replace(".", ""), {})
            entities = layer_info.get("entities", [])
            relationships = layer_info.get("relationships", [])
            quality_score = quality_info.get("overall_score", 0.0)
            accuracy_score = quality_info.get("accuracy", 0.0)
            coverage_score = quality_info.get("coverage", 0.0)
            issues = self._identify_layer_issues(layer_name, entities, relationships, quality_info)
            recommendations = self._generate_layer_recommendations(layer_name, issues, quality_score)

            layer_metrics[layer_name] = LayerMetrics(
                layer=layer_name,
                entity_count=len(entities),
                relationship_count=len(relationships),
                quality_score=quality_score,
                accuracy_score=accuracy_score,
                coverage_score=coverage_score,
                issues=issues,
                recommendations=recommendations,
            )

        return layer_metrics
    
    def _generate_cross_layer_insights(self, layer_metrics: Dict[str, LayerMetrics], layer_data: Dict[str, Any]) -> List[CrossLayerInsight]:
        """Generate insights from cross-layer analysis."""
        insights = []
        insights.extend(self._build_accuracy_insights(layer_metrics))
        insights.extend(self._build_entity_imbalance_insights(layer_metrics))
        insights.extend(self._build_quality_consistency_insights(layer_metrics))
        return insights

    def _build_accuracy_insights(self, layer_metrics: Dict[str, LayerMetrics]) -> List[CrossLayerInsight]:
        accuracy_trend = [layer_metrics[layer].accuracy_score for layer in ["L0", "L1", "L2", "L2.5"]]
        if len(accuracy_trend) > 1 and accuracy_trend[-1] < accuracy_trend[0] * 0.9:
            return [CrossLayerInsight(
                insight_type="ACCURACY_DECLINE",
                description=f"Pipeline accuracy declines from {accuracy_trend[0]:.3f} (L0) to {accuracy_trend[-1]:.3f} (L2.5)",
                affected_layers=["L0", "L1", "L2", "L2.5"],
                impact_level="HIGH",
                actionable_recommendation="Review L2.5 integration logic and validate cross-layer data flow",
                evidence=[f"L0 accuracy: {accuracy_trend[0]:.3f}", f"L2.5 accuracy: {accuracy_trend[-1]:.3f}"],
            )]
        return []

    def _build_entity_imbalance_insights(self, layer_metrics: Dict[str, LayerMetrics]) -> List[CrossLayerInsight]:
        entity_counts = [layer_metrics[layer].entity_count for layer in ["L0", "L1", "L2", "L2.5"]]
        if max(entity_counts) > min(entity_counts) * 5:
            return [CrossLayerInsight(
                insight_type="ENTITY_IMBALANCE",
                description=f"Significant entity count imbalance across layers (min: {min(entity_counts)}, max: {max(entity_counts)})",
                affected_layers=["L0", "L1", "L2", "L2.5"],
                impact_level="MEDIUM",
                actionable_recommendation="Investigate extraction completeness in underperforming layers",
                evidence=[
                    f"L0: {entity_counts[0]} entities",
                    f"L1: {entity_counts[1]} entities",
                    f"L2: {entity_counts[2]} entities",
                    f"L2.5: {entity_counts[3]} entities",
                ],
            )]
        return []

    def _build_quality_consistency_insights(self, layer_metrics: Dict[str, LayerMetrics]) -> List[CrossLayerInsight]:
        quality_scores = [layer_metrics[layer].quality_score for layer in ["L0", "L1", "L2", "L2.5"]]
        quality_variance = sum((q - sum(quality_scores) / len(quality_scores))**2 for q in quality_scores) / len(quality_scores)
        if quality_variance > 0.05:
            return [CrossLayerInsight(
                insight_type="QUALITY_INCONSISTENCY",
                description=f"High quality score variance ({quality_variance:.3f}) indicates inconsistent processing across layers",
                affected_layers=["L0", "L1", "L2", "L2.5"],
                impact_level="MEDIUM",
                actionable_recommendation="Standardize quality assessment criteria across all layers",
                evidence=[f"Quality variance: {quality_variance:.3f}"],
            )]
        return []
    
    def _calculate_overall_grade(self, layer_metrics: Dict[str, LayerMetrics]) -> str:
        """Calculate overall pipeline quality grade."""
        if not layer_metrics:
            return "NO_DATA"
        
        # Weight average of quality scores (L2.5 has highest weight)
        weights = {"L0": 0.15, "L1": 0.25, "L2": 0.3, "L2.5": 0.3}
        weighted_score = sum(
            layer_metrics[layer].quality_score * weights[layer] 
            for layer in weights if layer in layer_metrics
        )
        
        # Determine grade based on thresholds
        if weighted_score >= self._quality_thresholds["excellent"]:
            return "A"
        elif weighted_score >= self._quality_thresholds["good"]:
            return "B"
        elif weighted_score >= self._quality_thresholds["acceptable"]:
            return "C"
        elif weighted_score >= self._quality_thresholds["needs_improvement"]:
            return "D"
        else:
            return "F"
    
    def _generate_executive_summary(self, layer_metrics: Dict[str, LayerMetrics], insights: List[CrossLayerInsight]) -> str:
        """Generate executive summary for stakeholders."""
        total_entities = sum(m.entity_count for m in layer_metrics.values())
        avg_quality = self._average_quality(layer_metrics)
        high_impact_insights = [i for i in insights if i.impact_level == "HIGH"]

        summary = self._build_executive_summary_header(layer_metrics, total_entities, avg_quality)
        summary += self._build_executive_summary_findings(insights, high_impact_insights, layer_metrics)
        return summary

    def _average_quality(self, layer_metrics: Dict[str, LayerMetrics]) -> float:
        return sum(m.quality_score for m in layer_metrics.values()) / len(layer_metrics) if layer_metrics else 0

    def _build_executive_summary_header(self, layer_metrics: Dict[str, LayerMetrics], total_entities: int, avg_quality: float) -> str:
        return f"""Pipeline Analysis Summary
        
Overall Performance: Grade {self._calculate_overall_grade(layer_metrics)}
- Total Entities Processed: {total_entities:,}
- Average Quality Score: {avg_quality:.3f}
- Layers Analyzed: {len(layer_metrics)}

Key Findings:
"""

    def _build_executive_summary_findings(
        self,
        insights: List[CrossLayerInsight],
        high_impact_insights: List[CrossLayerInsight],
        layer_metrics: Dict[str, LayerMetrics],
    ) -> str:
        findings = ""
        if high_impact_insights:
            findings += f"- CRITICAL: {high_impact_insights[0].description}\n"
        findings += f"- Generated {len(insights)} cross-layer insights"
        findings += f"- Identified {sum(len(m.issues) for m in layer_metrics.values())} issues across layers"
        return findings
    
    def _generate_detailed_analysis(self, layer_metrics: Dict[str, LayerMetrics], insights: List[CrossLayerInsight], level: DocumentationLevel) -> str:
        """Generate detailed technical analysis."""
        analysis = self._build_detailed_analysis_header()
        analysis += self._build_layer_analysis_sections(layer_metrics)
        analysis += self._build_cross_layer_analysis_sections(insights)
        return analysis

    def _build_detailed_analysis_header(self) -> str:
        return "Detailed Pipeline Analysis\n" + "=" * 50 + "\n\n"

    def _build_layer_analysis_sections(self, layer_metrics: Dict[str, LayerMetrics]) -> str:
        sections = ""
        for layer_name, metrics in layer_metrics.items():
            sections += f"## {layer_name} Layer Analysis\n"
            sections += f"- Entity Count: {metrics.entity_count}\n"
            sections += f"- Relationship Count: {metrics.relationship_count}\n"
            sections += f"- Quality Score: {metrics.quality_score:.3f}\n"
            sections += f"- Accuracy Score: {metrics.accuracy_score:.3f}\n"
            sections += f"- Coverage Score: {metrics.coverage_score:.3f}\n"
            sections += self._build_issue_block(metrics.issues)
            sections += self._build_recommendation_block(metrics.recommendations)
            sections += "\n"
        return sections

    def _build_issue_block(self, issues: List[str]) -> str:
        if not issues:
            return ""
        block = "\nIssues Identified:\n"
        for issue in issues:
            block += f"- {issue}\n"
        return block

    def _build_recommendation_block(self, recommendations: List[str]) -> str:
        if not recommendations:
            return ""
        block = "\nRecommendations:\n"
        for rec in recommendations:
            block += f"- {rec}\n"
        return block

    def _build_cross_layer_analysis_sections(self, insights: List[CrossLayerInsight]) -> str:
        if not insights:
            return ""
        analysis = "## Cross-Layer Insights\n"
        for insight in insights:
            analysis += f"### {insight.insight_type} ({insight.impact_level})\n"
            analysis += f"{insight.description}\n"
            analysis += f"**Action:** {insight.actionable_recommendation}\n\n"
        return analysis
    
    def _generate_next_steps(self, insights: List[CrossLayerInsight], layer_metrics: Dict[str, LayerMetrics]) -> List[str]:
        """Generate actionable next steps."""
        next_steps = self._prioritize_insight_actions(insights)
        next_steps.extend(self._build_quality_focus_step(layer_metrics))
        next_steps.extend(self._build_monitoring_steps())
        return next_steps[:10]

    def _prioritize_insight_actions(self, insights: List[CrossLayerInsight]) -> List[str]:
        prioritized = []
        for insight in [i for i in insights if i.impact_level == "HIGH"] + [i for i in insights if i.impact_level == "MEDIUM"]:
            prioritized.append(insight.actionable_recommendation)
        return prioritized

    def _build_quality_focus_step(self, layer_metrics: Dict[str, LayerMetrics]) -> List[str]:
        low_quality_layers = [layer for layer, metrics in layer_metrics.items() if metrics.quality_score < 0.7]
        if low_quality_layers:
            return [f"Focus improvement efforts on layers: {', '.join(low_quality_layers)}"]
        return []

    def _build_monitoring_steps(self) -> List[str]:
        return [
            "Implement continuous monitoring for accuracy metrics",
            "Schedule weekly cross-layer consistency reviews",
        ]
    
    def _identify_layer_issues(self, layer_name: str, entities: List[Any], relationships: List[Any], quality_info: Dict[str, Any]) -> List[str]:
        """Identify issues specific to a layer."""
        issues = []
        if not entities:
            return ["No entities extracted"]

        issues.extend(self._check_entity_metadata(entities))
        issues.extend(self._check_relationship_completeness(relationships))
        issues.extend(self._check_quality_signals(quality_info))
        return issues

    def _check_entity_metadata(self, entities: List[Any]) -> List[str]:
        issues = []
        missing_ids = sum(1 for e in entities if not e.get("meta_id"))
        if missing_ids > 0:
            issues.append(f"{missing_ids} entities missing meta_id")
        missing_classes = sum(1 for e in entities if not e.get("meta_class"))
        if missing_classes > 0:
            issues.append(f"{missing_classes} entities missing meta_class")
        return issues

    def _check_relationship_completeness(self, relationships: List[Any]) -> List[str]:
        if not relationships:
            return []
        incomplete_rels = sum(1 for r in relationships if not all(k in r for k in ["from", "to", "type"]))
        return [f"{incomplete_rels} relationships missing required fields"] if incomplete_rels > 0 else []

    def _check_quality_signals(self, quality_info: Dict[str, Any]) -> List[str]:
        issues = []
        if quality_info.get("completeness", 1.0) < 0.5:
            issues.append("Low completeness score")
        if quality_info.get("accuracy", 1.0) < 0.6:
            issues.append("Poor accuracy score")
        return issues
    
    def _generate_layer_recommendations(self, layer_name: str, issues: List[str], quality_score: float) -> List[str]:
        """Generate recommendations for layer improvement."""
        recommendations = []
        if quality_score < 0.7:
            recommendations.append(f"Improve {layer_name} extraction quality (current: {quality_score:.3f})")
        recommendations.extend(self._build_layer_recommendation_by_issue(issues))
        return recommendations

    def _build_layer_recommendation_by_issue(self, issues: List[str]) -> List[str]:
        recommendations = []
        if "No entities extracted" in issues:
            recommendations.append("Review extraction configuration and source data format")
        if any("missing meta_id" in issue for issue in issues):
            recommendations.append("Enhance entity identification logic")
        if any("missing meta_class" in issue for issue in issues):
            recommendations.append("Improve schema classification accuracy")
        if any("relationships missing" in issue for issue in issues):
            recommendations.append("Strengthen relationship extraction rules")
        return recommendations
    
    def _get_pipeline_version(self) -> str:
        """Get current pipeline version."""
        return "v2.6.0"

    def _build_appendices(
        self,
        layer_data: Dict[str, Any],
        quality_metrics: Dict[str, Any],
        layer_metrics: Dict[str, LayerMetrics],
        insights: List[CrossLayerInsight],
    ) -> Dict[str, Any]:
        return {
            "source_layer_count": len(layer_data),
            "quality_metric_count": len(quality_metrics),
            "layer_summaries": {
                name: {
                    "entity_count": metrics.entity_count,
                    "relationship_count": metrics.relationship_count,
                    "quality_score": metrics.quality_score,
                }
                for name, metrics in layer_metrics.items()
            },
            "insight_types": [insight.insight_type for insight in insights],
        }

    def _build_documentation_report(
        self,
        *,
        layer_metrics: Dict[str, LayerMetrics],
        cross_layer_insights: List[CrossLayerInsight],
        overall_grade: str,
        executive_summary: str,
        detailed_analysis: str,
        next_steps: List[str],
        appendices: Dict[str, Any],
    ) -> DocumentationReport:
        return DocumentationReport(
            timestamp=datetime.now().isoformat(),
            pipeline_version=self._get_pipeline_version(),
            layer_metrics=layer_metrics,
            cross_layer_insights=cross_layer_insights,
            overall_quality_grade=overall_grade,
            executive_summary=executive_summary,
            detailed_analysis=detailed_analysis,
            next_steps=next_steps,
            appendices=appendices,
        )
    
    def export_to_markdown(self, report: DocumentationReport) -> str:
        """Export documentation report to Markdown format."""
        return self._build_markdown_report(report)

    def _build_markdown_report(self, report: DocumentationReport) -> str:
        lines = [
            "# Pipeline Documentation Report",
            f"",
            f"**Generated:** {report.timestamp}  ",
            f"**Pipeline Version:** {report.pipeline_version}  ",
            f"**Overall Grade:** {report.overall_quality_grade}",
            "",
            "## Executive Summary",
            "",
            report.executive_summary,
            "",
            "## Layer Metrics",
            "",
        ]

        for layer_name, metrics in report.layer_metrics.items():
            lines.extend(self._build_markdown_layer_block(layer_name, metrics))

        lines.extend(self._build_markdown_insights_block(report.cross_layer_insights))
        lines.extend(self._build_markdown_next_steps_block(report.next_steps))
        lines.extend([
            "## Detailed Analysis",
            "",
            report.detailed_analysis,
            "",
            "---",
            "*Report generated by Enhanced Documentation Service*",
        ])
        return "\n".join(lines)

    def _build_markdown_layer_block(self, layer_name: str, metrics: LayerMetrics) -> list[str]:
        lines = [
            f"### {layer_name}",
            f"- **Entities:** {metrics.entity_count}",
            f"- **Relationships:** {metrics.relationship_count}",
            f"- **Quality Score:** {metrics.quality_score:.3f}",
            f"- **Accuracy:** {metrics.accuracy_score:.3f}",
            f"- **Coverage:** {metrics.coverage_score:.3f}",
            "",
        ]
        if metrics.issues:
            lines.append("**Issues:**")
            lines.extend(f"- {issue}" for issue in metrics.issues)
            lines.append("")
        if metrics.recommendations:
            lines.append("**Recommendations:**")
            lines.extend(f"- {rec}" for rec in metrics.recommendations)
            lines.append("")
        return lines

    def _build_markdown_insights_block(self, insights: List[CrossLayerInsight]) -> list[str]:
        lines = ["## Cross-Layer Insights", ""]
        for insight in insights:
            lines.extend([
                f"### {insight.insight_type} ({insight.impact_level})",
                insight.description,
                "",
                f"**Action Required:** {insight.actionable_recommendation}",
                "",
            ])
        return lines

    def _build_markdown_next_steps_block(self, next_steps: List[str]) -> list[str]:
        lines = ["## Next Steps", ""]
        lines.extend(f"{i}. {step}" for i, step in enumerate(next_steps, 1))
        lines.append("")
        return lines
    
    def export_to_json(self, report: DocumentationReport) -> str:
        """Export documentation report to JSON format."""
        return self._serialize_report(report)

    def _serialize_report(self, report: DocumentationReport) -> str:
        return json.dumps(report.__dict__, default=str, indent=2)
