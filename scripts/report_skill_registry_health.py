#!/usr/bin/env python3
"""Skill registry 健康报告（P2-3c, 2026-10-16）。

把本次整改中分散发现的四类问题收敛成一个只读视图：

1. **phantom skills** —— 声明了 `script_path` 但文件不存在（注册为 executor 却不可执行）
2. **discovery errors** —— manifest 校验失败、被静默跳过的 skill（category 误用目录名 / 缺 command）
3. **evaluable skills** —— 声明了 `output_schema`、可被 schema 驱动 evaluator 评估的 skill
4. **routing integrity** —— 被 `config/pipeline_skill_routing.json` 选中的 skill 是否可执行

只读、无副作用（故无 --dry-run）；退化为 0/1 退出码便于 CI 接入。

用法：
    python3 scripts/report_skill_registry_health.py            # 人类可读
    python3 scripts/report_skill_registry_health.py --json     # 机器可读
    python3 scripts/report_skill_registry_health.py --check    # 有新增债务则退出 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASELINE_PATH = PROJECT_ROOT / "config" / "quality" / "skill_manifest_integrity_baseline.json"
ROUTING_PATH = PROJECT_ROOT / "config" / "pipeline_skill_routing.json"


def _rel(p: str) -> str:
    try:
        return str(Path(p).relative_to(PROJECT_ROOT))
    except Exception:
        return p


def collect() -> dict:
    from ontology_compiler.harness.skill_registry import build_default_skill_registry

    reg = build_default_skill_registry()
    skills = reg.list_skills(enabled_only=False)

    phantom = sorted(s.name for s in skills if not (s.script_path and os.path.isfile(s.script_path)))
    evaluable = sorted(reg.evaluable_skill_names(enabled_only=False))
    errors = sorted(
        {
            f"{_rel(e['path'])} :: {e['error']}"
            for e in reg.get_discovery_errors()
        }
    )

    routed: list[str] = []
    routing_broken: list[str] = []
    if ROUTING_PATH.exists():
        cfg = json.loads(ROUTING_PATH.read_text(encoding="utf-8"))
        names = {s.name for s in skills}
        refs: set[str] = set()
        for rule in (cfg.get("routing_rules") or []) + (cfg.get("fallback_rules") or []):
            refs.update(rule.get("skills") or [])
        routed = sorted(refs)
        by_name = {s.name: s for s in skills}
        for name in routed:
            skill = by_name.get(name)
            if skill is None:
                routing_broken.append(f"{name} (未 discovery)")
            elif not (skill.script_path and os.path.isfile(skill.script_path)):
                routing_broken.append(f"{name} (script 缺失: {_rel(skill.script_path)})")

    return {
        "total_skills": len(skills),
        "enabled_skills": sum(1 for s in skills if s.enabled),
        "evaluable_skills": evaluable,
        "evaluable_count": len(evaluable),
        "phantom_skills": phantom,
        "discovery_errors": errors,
        "routed_skills": routed,
        "routing_broken": routing_broken,
    }


def _load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        return {}
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def check(report: dict, baseline: dict | None = None) -> list[str]:
    """返回相对基线的新增债务（空 = 无回退）。

    ``baseline`` 省略时从 ``BASELINE_PATH`` 读取（测试可显式传入）。
    """
    base = _load_baseline() if baseline is None else baseline
    problems: list[str] = []

    new_phantoms = sorted(set(report["phantom_skills"]) - set(base.get("phantom_skills", [])))
    if new_phantoms:
        problems.append(f"新增 phantom skill: {new_phantoms}")

    new_errors = sorted(set(report["discovery_errors"]) - set(base.get("discovery_errors", [])))
    if new_errors:
        problems.append(f"新增 discovery error: {new_errors}")

    if report["evaluable_count"] < base.get("evaluable_count", 0):
        problems.append(
            f"可评估 skill 数回退: {base.get('evaluable_count')} -> {report['evaluable_count']}"
        )

    if report["routing_broken"]:
        problems.append(f"路由引用了不可执行的 skill: {report['routing_broken']}")

    return problems


def _render(report: dict) -> str:
    lines = []
    lines.append("=" * 66)
    lines.append("Skill registry 健康报告")
    lines.append("=" * 66)
    lines.append(
        f"discovery: {report['total_skills']} skills "
        f"(enabled={report['enabled_skills']}) | "
        f"可评估={report['evaluable_count']}"
    )
    lines.append("")
    lines.append(f"[phantom] script 不存在: {len(report['phantom_skills'])}")
    for n in report["phantom_skills"]:
        lines.append(f"    - {n}")
    lines.append("")
    lines.append(f"[discovery errors] 被静默跳过: {len(report['discovery_errors'])}")
    for e in report["discovery_errors"]:
        lines.append(f"    - {e}")
    lines.append("")
    lines.append(f"[routed] 路由配置引用: {report['routed_skills']}")
    lines.append(
        f"[routed integrity] {'OK' if not report['routing_broken'] else 'BROKEN: ' + str(report['routing_broken'])}"
    )
    lines.append("")
    lines.append(f"[evaluable] {', '.join(report['evaluable_skills'])}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--check", action="store_true", help="exit 1 when debt grew vs baseline")
    args = parser.parse_args()

    report = collect()
    import sys as _sys

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_render(report))

    if args.check:
        problems = check(report)
        tail = _sys.stderr if args.json else _sys.stdout
        if problems:
            print("\nFAIL: skill registry 债务回退：", file=tail)
            for pp in problems:
                print(f"  - {pp}", file=tail)
            return 1
        print("\nOK: 无新增债务（相对基线）", file=tail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
