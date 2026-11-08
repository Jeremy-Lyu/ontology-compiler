"""Neo4j 图数据库检索器

提供基于 Neo4j 的检索功能：
1. 关键词搜索 (L2Scenario/L2Logic/L2Definition)
2. 图遍历扩展 (获取相关实体和场景)
3. 全文索引搜索

Usage:
    retriever = Neo4jRetriever()
    results = retriever.keyword_search("Manufacturing Change Issue")
    context = retriever.graph_expand([r['id'] for r in results])
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# B-4 (2026-08-17): neo4j SDK import isolated to infrastructure provider.
from src.infrastructure.rag.neo4j_provider import get_graph_database_class

GraphDatabase = get_graph_database_class()


@dataclass
class Neo4jSearchResult:
    """Neo4j 搜索结果"""
    id: str
    name: str
    type: str  # OntologyNode, L2Scenario, L2Logic, L2Definition
    description: str
    score: float
    properties: Dict[str, Any]


class Neo4jRetriever:
    """Neo4j 图检索器
    
    自动从 config/env.env 读取连接配置：
    - NEO4J_URI=bolt://localhost:7687
    - NEO4J_USERNAME=neo4j
    - NEO4J_PASSWORD=12345678
    """
    
    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None,
        database: str = None
    ):
        if GraphDatabase is None:
            raise RuntimeError(
                "neo4j driver not installed. Run: pip install neo4j==5.15.0"
            )
        
        # 保留传入的参数
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        
        # 加载环境变量（向后兼容）
        self._load_env()
        
        # 解析配置，优先级：YAML > 环境变量 > 默认值
        self._resolve_config()
        
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password)
        )
        
        # 验证连接
        with self.driver.session() as session:
            result = session.run("RETURN 1 as test")
            result.single()
    
    def _load_yaml_config(self) -> Optional[Dict[str, Any]]:
        """从 YAML 配置加载连接参数。"""
        yaml_paths = [
            Path("config/neo4j_config.yaml"),
            Path("../config/neo4j_config.yaml"),
            Path(__file__).parent.parent.parent.parent / "config" / "neo4j_config.yaml",
        ]

        for yaml_path in yaml_paths:
            if yaml_path.exists():
                try:
                    import yaml
                    with open(yaml_path, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)
                    conn = cfg.get("connection", {})
                    return {
                        "uri": conn.get("uri"),
                        "username": conn.get("username"),
                        "password": conn.get("password"),
                        "database": conn.get("database"),
                    }
                except Exception:
                    pass
        return None

    def _load_env(self):
        """加载 config/env.env 环境变量（向后兼容）"""
        env_paths = [
            Path("config/env.env"),
            Path("../config/env.env"),
            Path("../../config/env.env"),
        ]
        
        for env_path in env_paths:
            if env_path.exists():
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            if key and value and os.getenv(key) is None:
                                os.environ[key] = value
                break

    def _resolve_config(self):
        """解析配置，优先级：YAML > 环境变量 > 默认值"""
        # 优先从 YAML 加载
        yaml_cfg = self._load_yaml_config()
        if yaml_cfg:
            self.uri = self.uri or yaml_cfg.get("uri") or os.getenv("NEO4J_URI", "bolt://localhost:7687")
            self.username = self.username or yaml_cfg.get("username") or os.getenv("NEO4J_USERNAME", "neo4j")
            self.password = self.password or yaml_cfg.get("password") or os.getenv("NEO4J_PASSWORD", "12345678")
            self.database = self.database or yaml_cfg.get("database") or os.getenv("NEO4J_DATABASE", "neo4j")
        else:
            # 回退到环境变量
            self.uri = self.uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
            self.username = self.username or os.getenv("NEO4J_USERNAME", "neo4j")
            self.password = self.password or os.getenv("NEO4J_PASSWORD", "12345678")
            self.database = self.database or os.getenv("NEO4J_DATABASE", "neo4j")
    
    def close(self):
        """关闭连接"""
        if hasattr(self, 'driver'):
            self.driver.close()
    
    def keyword_search(
        self,
        query: str,
        topk: int = 5,
        labels: List[str] = None
    ) -> List[Neo4jSearchResult]:
        """关键词搜索
        
        搜索 name, description 包含 query 的节点
        
        Args:
            query: 搜索关键词
            topk: 返回数量
            labels: 指定标签列表，如 ['L2Scenario', 'L2Logic']
        
        Returns:
            搜索结果列表
        """
        if labels is None:
            labels = ['L2Scenario', 'L2Logic', 'L2Definition', 'OntologyNode']
        
        results = []
        
        with self.driver.session() as session:
            for label in labels:
                try:
                    # 大小写不敏感搜索
                    cypher = f"""
                        MATCH (n:{label})
                        WHERE toLower(n.name) CONTAINS toLower($q)
                           OR toLower(n.description) CONTAINS toLower($q)
                           OR toLower(n.term) CONTAINS toLower($q)
                        RETURN n.id as id,
                               n.name as name,
                               n.description as description,
                               n.domain as domain,
                               n.category as category,
                               labels(n)[0] as node_type
                        LIMIT $limit
                    """
                    result = session.run(cypher, q=query, limit=topk)
                    
                    for record in result:
                        results.append(Neo4jSearchResult(
                            id=record["id"],
                            name=record["name"] or record["id"],
                            type=record["node_type"],
                            description=record["description"] or "",
                            score=1.0,  # 关键词匹配得分为 1
                            properties={
                                "domain": record.get("domain"),
                                "category": record.get("category")
                            }
                        ))
                except Exception as e:
                    # 某些标签可能没有特定属性，跳过
                    continue
        
        # 按相关性简单排序：名称完全匹配排前面
        results.sort(key=lambda r: (
            0 if query.lower() in r.name.lower() else 1,
            -len(r.description)  # 描述越长，信息越丰富
        ))
        
        return results[:topk]
    
    def fulltext_search(self, query: str, topk: int = 5) -> List[Neo4jSearchResult]:
        """全文索引搜索（需要预先创建全文索引）
        
        比 keyword_search 更快，支持更复杂的查询语法
        """
        results = []
        
        with self.driver.session() as session:
            try:
                result = session.run("""
                    CALL db.index.fulltext.queryNodes('ontology_fulltext', $query)
                    YIELD node, score
                    RETURN node.id as id,
                           node.name as name,
                           node.description as description,
                           node.term as term,
                           labels(node)[0] as node_type,
                           score
                    ORDER BY score DESC
                    LIMIT $limit
                """, query=query, limit=topk)
                
                for record in result:
                    results.append(Neo4jSearchResult(
                        id=record["id"],
                        name=record["name"] or record["term"] or record["id"],
                        type=record["node_type"],
                        description=record["description"] or "",
                        score=record["score"],
                        properties={}
                    ))
            except Exception as e:
                # 全文索引可能不存在，回退到关键词搜索
                print(f"[Neo4j] 全文索引不可用，回退到关键词搜索: {e}")
                return self.keyword_search(query, topk)
        
        return results
    
    def graph_expand(
        self,
        seed_ids: List[str],
        hops: int = 1
    ) -> Dict[str, Any]:
        """图遍历扩展上下文
        
        从种子节点出发，沿图关系遍历获取相关实体
        
        Args:
            seed_ids: 种子节点 ID 列表
            hops: 遍历跳数 (1-2)
        
        Returns:
            包含 entities, scenarios, relationships 的字典
        """
        with self.driver.session() as session:
            if hops == 1:
                # 1跳：直接关联的实体
                result = session.run("""
                    MATCH (s)
                    WHERE s.id IN $seed_ids
                    OPTIONAL MATCH (s)-[r:INVOLVES|RELATES_TO|SUPPORTS|DEFINES]-(n)
                    WHERE n:OntologyNode OR n:L2Scenario
                    RETURN s.id as seed_id,
                           s.name as seed_name,
                           collect(DISTINCT {
                               name: n.name,
                               type: labels(n)[0],
                               relation: type(r)
                           }) as related
                """, seed_ids=seed_ids)
                
                return [dict(record) for record in result]
            
            else:  # hops == 2
                # 2跳：实体 + 实体的邻居
                result = session.run("""
                    MATCH (s)
                    WHERE s.id IN $seed_ids
                    OPTIONAL MATCH (s)-[:INVOLVES]->(n:OntologyNode)
                    OPTIONAL MATCH (n)-[:RELATES_TO]-(related:OntologyNode)
                    RETURN s.name as scenario,
                           collect(DISTINCT n.name) as entities,
                           collect(DISTINCT related.name) as related_entities
                """, seed_ids=seed_ids)
                
                return [dict(record) for record in result]
    
    def get_entity_context(self, entity_name: str) -> Dict[str, Any]:
        """获取实体的完整上下文
        
        包括：属性、关系、涉及的 L2 场景
        """
        with self.driver.session() as session:
            # 实体基本信息
            entity_result = session.run("""
                MATCH (n:OntologyNode)
                WHERE n.name = $name OR n.id = $name
                RETURN n.id as id, n.name as name, n.meta_class as meta_class,
                       n.subclass as subclass, n.domain as domain,
                       n.description as description
                LIMIT 1
            """, name=entity_name)
            
            entity = dict(entity_result.single()) if entity_result.peek() else {}
            
            # 相关场景
            scenarios_result = session.run("""
                MATCH (n:OntologyNode)
                WHERE n.name = $name OR n.id = $name
                OPTIONAL MATCH (s:L2Scenario)-[:INVOLVES]->(n)
                RETURN collect(DISTINCT s.name) as scenarios
            """, name=entity_name)
            
            scenarios = list(scenarios_result.single().values())[0] if scenarios_result.peek() else []
            
            # 相关实体
            related_result = session.run("""
                MATCH (n:OntologyNode)
                WHERE n.name = $name OR n.id = $name
                OPTIONAL MATCH (n)-[r:RELATES_TO]-(related:OntologyNode)
                RETURN collect(DISTINCT {
                    name: related.name,
                    relation: r.type,
                    class: related.meta_class
                }) as related
            """, name=entity_name)
            
            related = list(related_result.single().values())[0] if related_result.peek() else []
            
            return {
                "entity": entity,
                "scenarios": scenarios,
                "related_entities": related
            }
    
    def hybrid_search(
        self,
        query: str,
        topk: int = 5
    ) -> Dict[str, Any]:
        """混合搜索：关键词 + 图上下文
        
        返回包含检索结果和图上下文的完整数据结构
        """
        # 1. 关键词搜索
        search_results = self.keyword_search(query, topk=topk)
        
        if not search_results:
            return {
                "results": [],
                "context": {},
                "query": query
            }
        
        # 2. 获取种子 ID
        seed_ids = [r.id for r in search_results]
        
        # 3. 图扩展上下文
        graph_context = self.graph_expand(seed_ids, hops=1)
        
        return {
            "results": [
                {
                    "id": r.id,
                    "name": r.name,
                    "type": r.type,
                    "description": r.description,
                    "score": r.score
                }
                for r in search_results
            ],
            "context": graph_context,
            "query": query
        }


def demo():
    """演示 Neo4j 检索功能"""
    print("=" * 60)
    print("Neo4j RAG 检索演示")
    print("=" * 60)
    
    try:
        retriever = Neo4jRetriever()
        
        # 测试查询
        queries = [
            "Manufacturing Change Issue",
            "MBOM",
            "Teamcenter"
        ]
        
        for query in queries:
            print(f"\n🔍 查询: {query}")
            print("-" * 40)
            
            results = retriever.keyword_search(query, topk=3)
            for i, r in enumerate(results, 1):
                print(f"{i}. [{r.type}] {r.name}")
                print(f"   {r.description[:80]}...")
        
        # 图扩展演示
        if results:
            print("\n📊 图上下文扩展:")
            print("-" * 40)
            seed_id = results[0].id
            context = retriever.graph_expand([seed_id])
            print(f"种子: {context[0].get('seed_name')}")
            related = context[0].get('related', [])
            print(f"关联实体: {len(related)} 个")
            for r in related[:3]:
                print(f"  - {r.get('name')} ({r.get('type')})")
        
        retriever.close()
        print("\n✅ 演示完成")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请确保:")
        print("  1. Neo4j 已启动 (docker ps)")
        print("  2. 已运行导入: python scripts/import_to_neo4j.py")
        print("  3. 配置正确: config/env.env")


if __name__ == "__main__":
    demo()
