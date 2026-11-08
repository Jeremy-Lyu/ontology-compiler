"""基于 master_ontology_unified.json 重建 RAG 索引"""
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.application.rag.chunk_indexing import (
    ChunkRecord,
    build_inverted_index,
    compute_idf,
    write_chunks_jsonl,
    write_inverted_index,
    write_idf,
)


# 常见缩写映射，用于增强检索
ACRONYM_MAP = {
    "bom": "BOM Bill of Materials",
    "mbom": "MBOM Manufacturing Bill of Materials 制造物料清单",
    "ebom": "EBOM Engineering Bill of Materials 工程物料清单",
    "plm": "PLM Product Lifecycle Management 产品生命周期管理",
    "mes": "MES Manufacturing Execution System 制造执行系统",
    "erp": "ERP Enterprise Resource Planning 企业资源规划",
    "wms": "WMS Warehouse Management System 仓库管理系统",
    "agv": "AGV Automated Guided Vehicle 自动导引车",
    "pvc": "PVC",
    "ois": "OIS Operation Instruction Sheet 操作指导书",
    "mpp": "MPP Manufacturing Process Planner 制造工艺规划",
    "ppa": "PPA Preliminary Production Part Address",
    "tcf": "TCF",
    "rdc": "RDC Regional Distribution Center 区域配送中心",
    "xps": "XPS",
    "bis": "BIS Business Information System 业务信息系统",
    "cip": "CIP Continuous Improvement Process 持续改进流程",
    "andon": "ANDON 安灯系统",
    "atacq": "ATACQ",
    "cmp": "CMP",
    "vpc": "VPC",
    "plc": "PLC Programmable Logic Controller 可编程逻辑控制器",
    "tcm": "TCM Total Cost Management 全面成本管理",
    "scm": "SCM Supply Chain Management 供应链管理",
    "sop": "SOP Standard Operating Procedure 标准作业程序",
    "bom管理": "BOM管理 物料清单管理",
    "物料管理": "物料管理 Material Management",
    "生产流程": "生产流程 Production Process",
    "工艺流程": "工艺流程 Process Flow",
}


def expand_name_for_search(name: str) -> str:
    """将合并的名称拆分为可检索的关键词变体"""
    if not name:
        return ""
    result = name
    name_lower = name.lower()
    
    # 检查是否包含已知缩写
    for acronym, full_form in ACRONYM_MAP.items():
        if acronym in name_lower:
            result += f" | {full_form}"
    
    # 拆分驼峰命名（如 Mbomsystem -> Mbom system）
    parts = re.findall(r'[A-Z][a-z]*|[0-9]+', name)
    if len(parts) > 1:
        result += f" | {' '.join(parts)}"
    
    # 如果名称中包含已知缩写的大写形式
    for acronym in ["BOM", "PLM", "MES", "ERP", "WMS", "AGV", "OIS", "MPP", "PPA", "PLC"]:
        if acronym.lower() in name_lower and acronym not in result:
            result += f" | {acronym}"
    
    return result


def build_chunks_from_unified(master: dict, source_file: str) -> list:
    """从 master_ontology_unified.json 构建 RAG chunks"""
    chunks: list[ChunkRecord] = []
    nodes = master.get("nodes", [])
    instances = master.get("instances", [])
    relationships = master.get("relationships", master.get("edges", []))

    # 构建节点映射
    node_by_id = {n.get("meta_id", n.get("entity_id", "")): n for n in nodes}
    instance_by_id = {inst.get("meta_id", inst.get("entity_id", "")): inst for inst in instances}
    
    # 构建节点->关系映射
    rel_by_node = defaultdict(list)
    for rel in relationships:
        src = rel.get("from", rel.get("source", ""))
        tgt = rel.get("to", rel.get("target", ""))
        rel_type = rel.get("type", rel.get("relation_type", "RELATED_TO"))
        rel_by_node[src].append(f"{rel_type} -> {tgt}")
        rel_by_node[tgt].append(f"<- {rel_type} from {src}")

    # 1. 为每个节点创建chunk（增强版：包含更多上下文）
    for node in nodes:
        nid = node.get("meta_id", node.get("entity_id", ""))
        if not nid:
            continue
        name = node.get("name", "")
        meta_class = node.get("meta_class", "")
        layer = node.get("layer", "")
        desc = node.get("description", "")
        domain = node.get("domain", "")
        status = node.get("status", "")
        parent = node.get("parent_class", node.get("parent_id", ""))
        evidence = node.get("evidence", {})
        if isinstance(evidence, dict):
            source_text = evidence.get("source_text", "")
        else:
            source_text = str(evidence)

        # 属性信息
        attrs = node.get("attributes", [])
        attr_lines = []
        for attr in attrs[:10]:
            if isinstance(attr, dict):
                attr_name = attr.get("name", "")
                attr_val = attr.get("value", "")
                attr_lines.append(f"  - {attr_name}: {attr_val}")

        rels = rel_by_node.get(nid, [])
        rel_lines = "\n".join(f"  - {r}" for r in rels[:20]) if rels else "  (none)"

        # 关键词变体（增强检索）
        search_terms = expand_name_for_search(name)

        content = (
            f"{meta_class}: {name}\n"
            f"Search Terms: {search_terms}\n"
            f"Layer: {layer}\n"
            f"ID: {nid}\n"
            f"Domain: {domain}\n"
            f"Status: {status}\n"
            f"Parent: {parent}\n"
            f"Description: {desc}\n"
            f"Evidence: {source_text}\n"
            f"Attributes:\n" + ("\n".join(attr_lines) if attr_lines else "  (none)") + "\n"
            f"Relationships:\n{rel_lines}"
        ).strip()

        chunks.append(
            ChunkRecord(
                chunk_id=f"UNIFIED_NODE_{nid}",
                chunk_type=f"unified_{layer.lower()}_{meta_class.lower()}",
                content=content,
                node_ids=[nid],
                linked_node_ids=[r.split(" -> ")[-1] for r in rels if " -> " in r],
                meta={
                    "meta_class": meta_class,
                    "layer": layer,
                    "domain": domain,
                    "source": source_file,
                },
            )
        )

    # 2. 为每个实例创建chunk
    for inst in instances:
        nid = inst.get("meta_id", inst.get("entity_id", ""))
        if not nid:
            continue
        name = inst.get("name", "")
        meta_class = inst.get("meta_class", "")
        desc = inst.get("description", "")
        parent = inst.get("parent_id", "")
        domain = inst.get("domain", "")

        rels = rel_by_node.get(nid, [])
        rel_lines = "\n".join(f"  - {r}" for r in rels[:20]) if rels else "  (none)"

        content = (
            f"Instance: {name}\n"
            f"Class: {meta_class}\n"
            f"ID: {nid}\n"
            f"Domain: {domain}\n"
            f"Parent: {parent}\n"
            f"Description: {desc}\n"
            f"Relationships:\n{rel_lines}"
        ).strip()

        chunks.append(
            ChunkRecord(
                chunk_id=f"UNIFIED_INST_{nid}",
                chunk_type="unified_l3_instance",
                content=content,
                node_ids=[nid],
                linked_node_ids=[],
                meta={"meta_class": meta_class, "source": source_file},
            )
        )

    # 3. 为每条关系创建摘要chunk（按source节点分组）
    rel_by_src = defaultdict(list)
    for rel in relationships:
        src = rel.get("from", rel.get("source", ""))
        rel_by_src[src].append(rel)

    for src_id, rels in rel_by_src.items():
        src_node = node_by_id.get(src_id) or instance_by_id.get(src_id)
        src_name = src_node.get("name", src_id) if src_node else src_id
        rel_lines = []
        for r in rels[:15]:
            tgt = r.get("to", r.get("target", ""))
            rtype = r.get("type", r.get("relation_type", "RELATED_TO"))
            tgt_node = node_by_id.get(tgt) or instance_by_id.get(tgt)
            tgt_name = tgt_node.get("name", tgt) if tgt_node else tgt
            rel_lines.append(f"  - {rtype} -> {tgt_name} ({tgt})")
        content = (
            f"Relationships for: {src_name} ({src_id})\n"
            + "\n".join(rel_lines)
        ).strip()
        chunks.append(
            ChunkRecord(
                chunk_id=f"UNIFIED_REL_{src_id}",
                chunk_type="unified_relationships",
                content=content,
                node_ids=[src_id],
                linked_node_ids=[r.get("to", r.get("target", "")) for r in rels[:15]],
                meta={"source": source_file},
            )
        )

    # 4. 创建MBOM/EBOM相关流程和系统的聚合chunk（增强检索）
    # 关键词列表用于识别相关实体
    mbom_ebom_keywords = {
        "mbom", "ebom", "bom", "物料清单", "制造", "生产", 
        "工艺", "工程", "plm", "mes", "erp", "wms",
        "物料", "清单", "bill of materials",
    }
    
    mbom_ebom_nodes = []
    for node in nodes:
        name = (node.get("name", "") + " " + node.get("description", "")).lower()
        if any(kw in name for kw in mbom_ebom_keywords):
            mbom_ebom_nodes.append(node)
    
    if mbom_ebom_nodes:
        content_lines = ["MBOM/EBOM Related Entities Summary:"]
        for node in mbom_ebom_nodes[:30]:
            nid = node.get("meta_id", "")
            name = node.get("name", "")
            meta_class = node.get("meta_class", "")
            layer = node.get("layer", "")
            desc = node.get("description", "")[:100]
            content_lines.append(f"- {name} ({meta_class}, Layer: {layer}, ID: {nid})")
            if desc:
                content_lines.append(f"  Description: {desc}...")
        
        chunks.append(
            ChunkRecord(
                chunk_id="UNIFIED_MBOM_EBOM_SUMMARY",
                chunk_type="unified_mbom_ebom_summary",
                content="\n".join(content_lines),
                node_ids=[n.get("meta_id", "") for n in mbom_ebom_nodes],
                linked_node_ids=[],
                meta={"source": source_file, "type": "MBOM_EBOM_SUMMARY"},
            )
        )

    # 5. 创建Process类型实体的流程链chunk
    process_nodes = [n for n in nodes if n.get("meta_class") == "Process"]
    if process_nodes:
        content_lines = ["Process Flow Chain Summary:"]
        for proc in process_nodes[:30]:
            nid = proc.get("meta_id", "")
            name = proc.get("name", "")
            layer = proc.get("layer", "")
            desc = proc.get("description", "")[:100]
            content_lines.append(f"- Process: {name} (Layer: {layer}, ID: {nid})")
            if desc:
                content_lines.append(f"  Description: {desc}...")
            
            # 添加相关关系
            rels = rel_by_node.get(nid, [])
            if rels:
                content_lines.append("  Related:")
                for r in rels[:5]:
                    content_lines.append(f"    {r}")
        
        chunks.append(
            ChunkRecord(
                chunk_id="UNIFIED_PROCESS_SUMMARY",
                chunk_type="unified_process_summary",
                content="\n".join(content_lines),
                node_ids=[n.get("meta_id", "") for n in process_nodes],
                linked_node_ids=[],
                meta={"source": source_file, "type": "PROCESS_SUMMARY"},
            )
        )

    # 6. 创建L2业务场景的流程上下文chunk
    l2_nodes = [n for n in nodes if n.get("layer") == "L2"]
    if l2_nodes:
        content_lines = ["L2 Business Scenario Context:"]
        for scen in l2_nodes[:30]:
            nid = scen.get("meta_id", "")
            name = scen.get("name", "")
            meta_class = scen.get("meta_class", "")
            desc = scen.get("description", "")[:100]
            content_lines.append(f"- Scenario: {name} ({meta_class}, ID: {nid})")
            if desc:
                content_lines.append(f"  Description: {desc}...")
            
            # 添加相关关系
            rels = rel_by_node.get(nid, [])
            if rels:
                content_lines.append("  Related:")
                for r in rels[:5]:
                    content_lines.append(f"    {r}")
        
        chunks.append(
            ChunkRecord(
                chunk_id="UNIFIED_L2_SCENARIO_SUMMARY",
                chunk_type="unified_l2_scenario_summary",
                content="\n".join(content_lines),
                node_ids=[n.get("meta_id", "") for n in l2_nodes],
                linked_node_ids=[],
                meta={"source": source_file, "type": "L2_SCENARIO_SUMMARY"},
            )
        )

    return chunks


def main():
    master_path = PROJECT_ROOT / "outputs" / "runs" / "ontology" / "master_ontology_unified.json"
    output_dir = PROJECT_ROOT / "outputs" / "runs" / "ontology" / "rag"

    print(f"[RAG BUILD] Loading {master_path}...")
    with open(master_path, "r", encoding="utf-8") as f:
        master = json.load(f)

    nodes = master.get("nodes", [])
    instances = master.get("instances", [])
    rels = master.get("relationships", master.get("edges", []))
    print(f"[RAG BUILD]   {len(nodes)} nodes, {len(instances)} instances, {len(rels)} relationships")

    print("[RAG BUILD] Building chunks...")
    chunks = build_chunks_from_unified(master, str(master_path))
    print(f"[RAG BUILD]   {len(chunks)} chunks generated")

    # 去重
    seen = {}
    for c in chunks:
        seen[c.chunk_id] = c
    all_chunks = sorted(seen.values(), key=lambda c: c.chunk_id)
    print(f"[RAG BUILD]   {len(all_chunks)} unique chunks")

    # 类型统计
    type_counts = defaultdict(int)
    for c in all_chunks:
        type_counts[c.chunk_type] += 1
    print("[RAG BUILD] Chunk type distribution:")
    for t, n in sorted(type_counts.items()):
        print(f"    {n:4d}  {t}")

    # 写入索引文件
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = output_dir / "chunks.jsonl"
    write_chunks_jsonl(all_chunks, chunks_path)
    print(f"\n[RAG BUILD] Written: {chunks_path}")

    print("[RAG BUILD] Building inverted index...")
    inverted_index = build_inverted_index(all_chunks)
    idf = compute_idf(inverted_index, len(all_chunks))
    write_inverted_index(inverted_index, output_dir / "inverted_index.json")
    write_idf(idf, output_dir / "idf.json")
    print(f"[RAG BUILD]   {len(inverted_index)} tokens indexed")

    # 写入manifest
    manifest = {
        "build_time": __import__("datetime").datetime.now().isoformat(),
        "source_file": str(master_path),
        "total_chunks": len(all_chunks),
        "total_nodes": len(nodes),
        "total_instances": len(instances),
        "total_relationships": len(rels),
        "chunk_types": dict(type_counts),
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[RAG BUILD] Written: {manifest_path}")

    print(f"\n[RAG BUILD] ✅ RAG index rebuilt successfully at {output_dir}")


if __name__ == "__main__":
    main()
