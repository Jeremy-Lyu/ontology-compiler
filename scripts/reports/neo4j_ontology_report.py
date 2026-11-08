"""Neo4j-backed ontology analysis report generation.

This module is the authoritative source for OntologyAnalyzer.
The legacy `scripts/neo4j_analytics.py` now imports from here with deprecation warnings.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict

try:
    from neo4j import GraphDatabase
except ImportError:
    raise ImportError("请先安装 neo4j 驱动: pip install neo4j")


class OntologyAnalyzer:
    """本体图谱分析器"""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    @classmethod
    def from_env(cls, env_path: str = "config/env.env"):
        """从环境文件创建分析器"""
        config = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "neo4j", "database": "neo4j"}

        if Path(env_path).exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("NEO4J_URI="):
                        config["uri"] = line.split("=", 1)[1].strip()
                    elif line.startswith("NEO4J_USERNAME="):
                        config["user"] = line.split("=", 1)[1].strip()
                    elif line.startswith("NEO4J_PASSWORD="):
                        config["password"] = line.split("=", 1)[1].strip()

        config["uri"] = os.getenv("NEO4J_URI", config["uri"])
        config["user"] = os.getenv("NEO4J_USER", config["user"])
        config["password"] = os.getenv("NEO4J_PASSWORD", config["password"])
        return cls(**config)

    def close(self):
        if self.driver:
            self.driver.close()

    def query(self, cypher: str, params: Dict = None) -> List[Dict]:
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, params or {})
            return [dict(r) for r in result]

    def get_stats(self) -> Dict:
        nodes = self.query("MATCH (n) RETURN labels(n)[0] as label, count(n) as cnt ORDER BY cnt DESC")
        rels = self.query("MATCH ()-[r]->() RETURN type(r) as type, count(r) as cnt ORDER BY cnt DESC")
        return {
            "total_nodes": sum(n["cnt"] for n in nodes),
            "total_relationships": sum(r["cnt"] for r in rels),
            "node_labels": {n["label"]: n["cnt"] for n in nodes},
            "relationship_types": {r["type"]: r["cnt"] for r in rels},
            "timestamp": datetime.now().isoformat(),
        }

    def get_domain_distribution(self, top_n: int = 20) -> List[Dict]:
        return self.query(f"""
            MATCH (n:L1_Concept) WHERE n.domain IS NOT NULL
            RETURN n.domain as domain, count(n) as count,
                   collect(n.name)[0..5] as samples
            ORDER BY count DESC LIMIT {top_n}
        """)

    def analyze_digital_thread(self) -> List[Dict]:
        return self.query("""
            MATCH (dl:DataLayer)
            OPTIONAL MATCH (dl)-[r:TRANSLATES_TO|SCHEDULES_TO|RECORDS_TO]->(next:DataLayer)
            RETURN dl.layer_type as layer, dl.table as table,
                   dl.key_fields as keys, count(r) as connections
            ORDER BY layer
        """)

    def get_data_flow(self) -> List[Dict]:
        return self.query("""
            MATCH path = (a:DataLayer)-[:TRANSLATES_TO|SCHEDULES_TO|RECORDS_TO*]->(b:DataLayer)
            RETURN [n in nodes(path) | n.table] as flow,
                   length(path) as hops
            LIMIT 10
        """)

    def get_mbom_concepts(self) -> List[Dict]:
        return self.query("""
            MATCH (n:MBOMConcept)
            OPTIONAL MATCH (n)-[r]->(m)
            RETURN n.name as concept, n.description as desc,
                   count(r) as rel_count
            ORDER BY rel_count DESC
        """)

    def get_central_entities(self, limit: int = 20) -> List[Dict]:
        return self.query(f"""
            MATCH (n:L1_Concept)
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(r) as degree
            RETURN n.id as id, n.name as name, n.domain as domain, degree
            ORDER BY degree DESC LIMIT {limit}
        """)

    def find_shortest_path(self, start_id: str, end_id: str) -> Dict:
        results = self.query("""
            MATCH path = shortestPath(
                (a {id: $start_id})-[:RELATED_TO|PART_OF*]-(b {id: $end_id})
            )
            RETURN [n in nodes(path) | n.name] as nodes,
                   length(path) as hops
        """, {"start_id": start_id, "end_id": end_id})
        return results[0] if results else {}

    def analyze_configuration_collapse(self) -> List[Dict]:
        return self.query("""
            MATCH (n:L1_Concept)
            WHERE n.description CONTAINS 'BOM' OR n.name CONTAINS 'BOM'
            OPTIONAL MATCH (n)-[r:PART_OF|CONTAINS]->(m)
            RETURN n.name as bom, n.description as desc,
                 collect(m.name)[0..10] as components
            LIMIT 20
        """)

    def analyze_temporal_sewing(self) -> List[Dict]:
        return self.query("""
            MATCH (dl:DataLayer)
            WHERE dl.key_fields CONTAINS 'ISSUE' OR dl.table CONTAINS 'TCP712'
            RETURN dl.table as table, dl.key_fields as keys,
                   dl.description as desc
        """)

    def analyze_quality_traceability(self) -> List[Dict]:
        return self.query("""
            MATCH path = (commercial:DataLayer {layer_type: 'commercial'})-[:TRANSLATES_TO|SCHEDULES_TO|RECORDS_TO*]->(archive:DataLayer {layer_type: 'archive'})
            RETURN [n in nodes(path) | n.table] as data_flow,
                   [r in relationships(path) | type(r)] as transformations,
                   length(path) as steps
            LIMIT 5
        """)

    def generate_report(self) -> Dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "stats": self.get_stats(),
            "domains": self.get_domain_distribution(10),
            "digital_thread": self.analyze_digital_thread(),
            "mbom_concepts": self.get_mbom_concepts(),
            "central_entities": self.get_central_entities(10),
        }

    def check_gds(self) -> bool:
        """检查 GDS 是否可用"""
        try:
            result = self.query("RETURN gds.version() as version")
            print(f"✅ GDS 版本: {result[0]['version']}")
            return True
        except Exception as e:
            print(f"❌ GDS 不可用，请安装插件: {e}")
            return False

    def pagerank(self, limit: int = 20):
        """PageRank 影响力分析"""
        try:
            self.query("CALL gds.graph.drop('onto', false) YIELD g RETURN 1")
        except Exception:
            pass

        self.query("""
            CALL gds.graph.project('onto', ['L1_Concept'], 
                {RELATED_TO: {orientation: 'UNDIRECTED'}})
            YIELD nodeCount, relationshipCount
            RETURN nodeCount, relationshipCount
        """)

        results = self.query(f"""
            CALL gds.pageRank.stream('onto') YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).id as id,
                   gds.util.asNode(nodeId).name as name,
                   score
            ORDER BY score DESC LIMIT {limit}
        """)

        self.query("CALL gds.graph.drop('onto')")
        return results

    def louvain(self, min_size: int = 3):
        """Louvain 社区检测"""
        try:
            self.query("CALL gds.graph.drop('onto', false) YIELD g RETURN 1")
        except Exception:
            pass

        self.query("""
            CALL gds.graph.project('onto', ['L1_Concept'], 'RELATED_TO')
            YIELD nodeCount, relationshipCount
        """)

        results = self.query(f"""
            CALL gds.louvain.stream('onto') YIELD nodeId, communityId
            WITH communityId, count(*) as size,
                 collect(gds.util.asNode(nodeId).name) as entities
            WHERE size >= {min_size}
            RETURN communityId, size, entities
            ORDER BY size DESC
        """)

        self.query("CALL gds.graph.drop('onto')")
        return results


def generate_markdown_report(analyzer: OntologyAnalyzer, output_path: Path) -> str:
    """Generate a Markdown analysis report from Neo4j graph statistics."""
    stats = analyzer.get_stats()
    domains = analyzer.get_domain_distribution(15)
    mbom = analyzer.get_mbom_concepts()
    thread = analyzer.analyze_digital_thread()
    central = analyzer.get_central_entities(15)

    report = f"""# 本体图谱分析报告

**生成时间**: {stats['timestamp']}
**数据来源**: Neo4j Graph Database
**分析实体**: {stats['total_nodes']:,} 节点 | {stats['total_relationships']:,} 关系

---

## 1. 数据库概览

### 1.1 总体统计

| 指标 | 数值 |
|------|------|
| 总节点数 | {stats['total_nodes']:,} |
| 总关系数 | {stats['total_relationships']:,} |
| 节点标签类型 | {len(stats['node_labels'])} |
| 关系类型 | {len(stats['relationship_types'])} |

### 1.2 节点标签分布

| 标签 | 数量 | 占比 |
|------|------|------|
"""

    total = stats['total_nodes']
    for label, count in list(stats['node_labels'].items())[:10]:
        pct = (count / total * 100) if total > 0 else 0
        report += f"| {label} | {count:,} | {pct:.1f}% |\n"

    report += """
### 1.3 关系类型分布

| 关系类型 | 数量 |
|----------|------|
"""

    for rel_type, count in list(stats['relationship_types'].items())[:10]:
        report += f"| {rel_type} | {count:,} |\n"

    report += f"""
---

## 2. 领域分布分析

### 2.1 Top {len(domains)} 领域

| 领域 | 实体数 | 占比 | 示例实体 |
|------|--------|------|----------|
"""

    total_l1 = sum(d['count'] for d in domains)
    for d in domains:
        pct = (d['count'] / total_l1 * 100) if total_l1 > 0 else 0
        samples = ', '.join(d['samples'][:3]) if d['samples'] else 'N/A'
        report += f"| {d['domain']} | {d['count']} | {pct:.1f}% | {samples} |\n"

    report += """
---

## 3. MBOM 核心概念分析

| 概念 | 描述 | 关联数 |
|------|------|--------|
"""

    for c in mbom:
        desc = c.get('desc', '')
        if len(desc) > 50:
            desc = desc[:50] + '...'
        report += f"| {c.get('concept', 'N/A')} | {desc} | {c.get('rel_count', 0)} |\n"

    report += """
---

## 4. 数字主线 (Digital Thread) 分析

| 层级 | 数据表 | 关键字段 | 连接数 |
|------|--------|----------|--------|
"""

    for t in thread:
        report += f"| {t.get('layer', 'N/A')} | {t.get('table', 'N/A')} | {t.get('keys', 'N/A')} | {t.get('connections', 0)} |\n"

    report += f"""
---

## 5. 实体中心性分析

| 实体ID | 名称 | 领域 | 连接度 |
|--------|------|------|--------|
"""

    for c in central:
        name = c.get('name', '')[:30] if c.get('name') else c.get('id', 'N/A')[:30]
        report += f"| {c.get('id', 'N/A')[:25]} | {name} | {c.get('domain', 'N/A')} | {c.get('degree', 0)} |\n"

    report += f"""
---

## 6. 关键发现

1. **实体完整性**: 已导入 {stats['total_nodes']:,} 个节点
2. **关系连通性**: {stats['total_relationships']:,} 条关系
3. **领域覆盖**: 主要覆盖 {len(domains)} 个业务领域

---

*报告由自动化分析系统生成*
"""
    return report


def save_report(report: str, output_dir: Path, filename: str | None = None) -> Path:
    """Persist report content to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ontology_analysis_report_{timestamp}.md"
    output_path = output_dir / filename
    output_path.write_text(report, encoding="utf-8")
    print(f"✅ 报告已保存: {output_path}")
    return output_path


def run_neo4j_report(
    output_dir: Path,
    *,
    filename: str | None = None,
    env_path: str = "config/env.env",
) -> Path:
    """Connect to Neo4j, generate report, and save to output_dir."""
    analyzer = OntologyAnalyzer.from_env(env_path)
    try:
        report = generate_markdown_report(analyzer, output_dir)
        return save_report(report, output_dir, filename)
    finally:
        analyzer.close()