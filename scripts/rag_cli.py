#!/usr/bin/env python3
"""Ontology RAG CLI - 统一入口

支持功能:
    1. 查询 (query)   - 使用 LLM 增强 RAG 问答
    2. 构建 (build)   - 重新构建 RAG 索引
    3. 对话 (chat)    - 交互式问答模式
    4. 更新 (update)  - 对话式 Ontology 更新

用法:
    # 查询
    python rag_cli.py query "MBOM是什么"
    python rag_cli.py query "MBOM如何在工厂运作" --detailed

    # 构建索引
    python rag_cli.py build

    # 交互模式
    python rag_cli.py chat

    # 对话式更新
    python rag_cli.py update "添加一个新流程"

环境变量:
    自动加载 config/env.env
"""
import argparse
import sys
from pathlib import Path

# =========================================================================
# 与 ontology_cli.py 保持一致的 sys.path 初始化：
#   - 把项目根目录加入 sys.path，保证 `from scripts.xxx import yyy` 和
#     `from src.xxx import yyy` 两种包级导入风格都能工作。
#   - 同时保证脚本既可以 `python rag_cli.py`（在 scripts 目录内）也能
#     `python scripts/rag_cli.py`（在项目根目录）两种方式运行。
# =========================================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    # 兼容 `import ontology_cli` 的扁平导入写法
    sys.path.insert(0, str(Path(__file__).resolve().parent))

_PROJECT_ROOT_GLOBAL = _PROJECT_ROOT  # 导出给后续函数复用
del _PROJECT_ROOT


def cmd_query(args):
    """查询命令"""
    from src.application.rag.rag_answerer import LLMAnswerer
    from src.application.rag.governed_packets import default_governed_rag_dir
    from src.application.rag.semantic_surface import check_answerable

    print(f"\n🔍 问题: {args.question}")
    print("=" * 70)

    # 2026-09-01: RAG 检索必须以 governed packet 为权威 rag_dir
    # （legacy outputs/runs/ontology/rag 已被 DataContractError 硬禁止）。
    rag_dir = default_governed_rag_dir()

    if not getattr(args, "skip_answerable_check", False):
        report = check_answerable(args.question, packet_dir=rag_dir)
        print(
            f"🧭 可答性预检: answerable={report.answerable} "
            f"confidence={report.confidence:.2f} packet={report.packet_id or 'n/a'}"
        )
        if report.gaps:
            for gap in report.gaps:
                print(f"   · {gap}")
        if not report.answerable:
            print("\n⛔ 语义层覆盖不足，已跳过 LLM 作答。")
            print("   用 --skip-answerable-check 强制查询，或先补齐 governed packet。")
            print("   也可用: python scripts/rag_cli.py meaning answerable \"...\"")
            sys.exit(3)

    answerer = LLMAnswerer(rag_dir=rag_dir, use_neo4j=args.neo4j)
    
    try:
        if args.detailed:
            result = answerer.answer_detailed(args.question, topk=args.topk)
            
            print(f"\n🤖 模型: {result['model']}")
            print(f"📊 置信度: {result['confidence']:.2f}")
            print(f"📚 使用证据: {result['evidence_count']} 条\n")
            
            print("📝 答案:")
            print("-" * 70)
            print(result['answer'])
            print("-" * 70)
            
            print("\n📖 证据来源:")
            for i, ev in enumerate(result['evidence'], 1):
                print(f"  {i}. {ev['chunk_id']} ({ev['chunk_type']}, score: {ev['score']:.3f})")
        else:
            result = answerer.answer(args.question, topk=args.topk)
            
            print("\n📝 答案:")
            print("-" * 70)
            print(result.answer)
            print("-" * 70)
            
            print(f"\n📊 置信度: {result.confidence:.2f} | 模型: {result.model}")
    finally:
        answerer.close()


def cmd_meaning_search(args):
    import json
    from src.application.rag.semantic_surface import search_entities

    hits = search_entities(args.query, limit=args.limit)
    print(json.dumps([h.to_dict() for h in hits], ensure_ascii=False, indent=2))


def cmd_meaning_get(args):
    import json
    from src.application.rag.semantic_surface import get_entity

    payload = get_entity(args.entity_id)
    if payload is None:
        print(f"entity not found: {args.entity_id}")
        sys.exit(1)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_meaning_answerable(args):
    import json
    from src.application.rag.semantic_surface import check_answerable

    report = check_answerable(args.question)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    if not report.answerable:
        sys.exit(3)


def cmd_build(args):
    """构建命令 - 智能查找 master 目录并构建 RAG 索引"""
    print("🔨 构建 RAG 索引...")

    # =====================================================================
    # 双路径导入 fallback：
    #   - 首选：包级导入 `from scripts.ontology_cli import run_rag_build`
    #     （在 `python -m scripts.rag_cli` 或 pytest 下生效）
    #   - 回退：扁平导入 `import ontology_cli` + getattr
    #     （在直接 `python scripts/rag_cli.py` 运行时生效）
    # =====================================================================
    try:
        from scripts.ontology_cli import run_rag_build  # type: ignore[no-redef]
    except (ModuleNotFoundError, ImportError):
        import ontology_cli as _ontology_cli_mod  # type: ignore[no-redef]
        run_rag_build = _ontology_cli_mod.run_rag_build  # type: ignore[attr-defined]

    import argparse as _argparse
    from pathlib import Path as _Path

    project_root: _Path = _PROJECT_ROOT_GLOBAL

    # ===== 智能查找可用的 ontology master 目录 =====
    candidate_dirs = [
        project_root / "outputs" / "runs" / "ontology" / "master_ontology",
        project_root / "outputs" / "ontology" / "master_ontology",
        project_root / "outputs" / "ontology",
    ]
    # 额外探测：从最新的 pipeline Run 目录下加载 master
    pipeline_root = project_root / "outputs" / "runs" / "ontology" / "pipeline"
    if pipeline_root.exists():
        run_dirs = sorted(pipeline_root.glob("MAIN-*"), reverse=True)
        for rd in run_dirs:
            candidate_dirs.append(rd / "master_ontology")

    resolved_ontology_dir: _Path | None = None
    for cd in candidate_dirs:
        if not cd.exists():
            continue
        # 只要目录下存在任意一个 master JSON 文件即视为有效
        master_signals = [
            cd / "master_ontology_v2.json",
            cd / "master_ontology_l2.json",
            cd / "master_ontology_unified.json",
            cd / "nodes.json",
            cd.parent / "master_ontology_unified.json",
        ]
        if any(s.exists() for s in master_signals):
            resolved_ontology_dir = cd
            break

    if resolved_ontology_dir is None:
        print("[❌] 未找到任何可用的 master ontology 目录。")
        print("     请先运行 pipeline（scripts/run_batch.py）生成 master 产物。")
        print(f"     探测过的路径: {[str(p) for p in candidate_dirs[:5]]}")
        sys.exit(2)

    ontology_dir_str = str(resolved_ontology_dir)
    print(f"✅ 检测到 master 目录: {ontology_dir_str}")

    build_args = _argparse.Namespace()
    build_args.source = "master"
    build_args.run_dir = None
    build_args.ontology_dir = ontology_dir_str
    build_args.output_dir = None

    rc = run_rag_build(build_args)
    if rc != 0:
        sys.exit(rc)
    print("\n🎉 RAG 索引构建完成！现在可以运行:")
    print(f"   python scripts/rag_cli.py query \"MBOM是什么\"")
    print(f"   python scripts/rag_cli.py query \"supply chain流程\" --detailed")


def cmd_chat(args):
    """交互模式"""
    print("\n" + "=" * 70)
    print("🤖 Ontology RAG - 交互模式")
    print("=" * 70)
    print("\n提示:")
    print("  - 直接输入问题获取答案")
    print("  - 输入 /exit 退出")
    print("-" * 70 + "\n")
    
    from src.application.rag.rag_answerer import LLMAnswerer
    
    answerer = LLMAnswerer()
    
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        
        if user_input.lower() in ['/exit', 'quit', 'exit', 'q']:
            print("再见！")
            break
        
        if not user_input:
            continue
        
        try:
            print(f"\n🔍 {user_input}")
            print("-" * 70)
            
            result = answerer.answer(user_input, topk=5)
            print(result.answer)
            
            print(f"\n(置信度: {result.confidence:.2f})")
            print("=" * 70)
            
        except Exception as e:
            print(f"❌ 错误: {e}")


def cmd_update(args):
    """(REMOVED 2026-08-26) 对话式更新命令。

    原实现动态加载 `archived/examples/examples/ontology_chat_update_example.py`，
    该文件已不存在，命令为死代码。已从 CLI 子命令中移除；如需恢复对话式更新，
    应基于 `src/application/rag/` 的稳定契约重新设计，而不是引用归档示例。
    """
    raise SystemExit(
        "The 'update' command was removed (dead reference to archived example). "
        "See docs/CODE_CLEANUP_CANDIDATES.md."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Ontology RAG CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python rag_cli.py query "MBOM是什么"
  python rag_cli.py query "supply chain流程" --detailed
  python rag_cli.py chat
  python rag_cli.py build

说明:
  本脚本是 RAG 子入口（query/chat/build 薄壳）。compile/validate/analyze 等
  主能力请使用 scripts/ontology_cli.py。
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # query 命令
    query_parser = subparsers.add_parser('query', help='RAG 查询')
    query_parser.add_argument('question', help='查询问题')
    query_parser.add_argument('-d', '--detailed', action='store_true', help='详细模式')
    query_parser.add_argument('-k', '--topk', type=int, default=5, help='检索数量')
    query_parser.add_argument('--neo4j', action='store_true', help='启用 Neo4j 图数据库检索')
    query_parser.add_argument(
        '--skip-answerable-check',
        action='store_true',
        help='跳过语义层可答性预检（默认开启预检，覆盖不足则退出码 3）',
    )
    query_parser.set_defaults(func=cmd_query)
    
    # build 命令
    build_parser = subparsers.add_parser('build', help='构建 RAG 索引')
    build_parser.add_argument('-f', '--force', action='store_true', help='强制重建')
    build_parser.set_defaults(func=cmd_build)
    
    # chat 命令
    chat_parser = subparsers.add_parser('chat', help='交互式问答')
    chat_parser.set_defaults(func=cmd_chat)

    # meaning — 窄语义面（search / get / answerable），不 dump 存储
    meaning = subparsers.add_parser('meaning', help='语义层只读查询（含义优先）')
    meaning_sub = meaning.add_subparsers(dest='meaning_command')
    m_search = meaning_sub.add_parser('search', help='按名称/别名/类型检索实体')
    m_search.add_argument('query', help='检索词')
    m_search.add_argument('-k', '--limit', type=int, default=10)
    m_search.set_defaults(func=cmd_meaning_search)
    m_get = meaning_sub.add_parser('get', help='按 id 取实体及邻接关系')
    m_get.add_argument('entity_id', help='实体 id')
    m_get.set_defaults(func=cmd_meaning_get)
    m_ans = meaning_sub.add_parser('answerable', help='提问前可答性预检')
    m_ans.add_argument('question', help='自然语言问题')
    m_ans.set_defaults(func=cmd_meaning_answerable)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    if args.command == 'meaning' and not getattr(args, 'meaning_command', None):
        meaning.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
