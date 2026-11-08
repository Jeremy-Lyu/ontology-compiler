#!/usr/bin/env python3
"""
一键 Batch 处理入口脚本（合并原 Step1/2/3）

📌 用法：
    python scripts/run_batch.py --folder "C:\\path\\to\\batch11"
    python scripts/run_batch.py --folder "C:\\...\\batch11" --domain AUTO

用户只要传 1 个参数（文档目录路径），剩下的全部自动：
    Step 1: 跑 ontology_cli compile（提取 entities/relationships）
    Step 2: 自动找到新生成的 Run ID，合并 fragments 到 master
    Step 3: 跑 14 项回归测试 + 重新导出分层图（layer_L0 ~ L3）

完全替代原来的「改路径compile → 复制 merge_batchXX.py 改 RUN_ID → 手动回归+export」3 步手动流程。
"""
from __future__ import annotations

# =====================================================================
# [WINDOWS FIX] WinError 10038（socket 不是 socket / socket.socketpair()
# 抛 OSError）happens when Python 3.11 on Windows uses the default
# SelectorEventLoop — its _make_self_pipe() call via socket.socketpair()
# is NOT safe inside enterprise-managed Windows sockets stacks
# (Winsock LSPs / VPN filters / corp security products) and fails with
# "[WinError 10038] An operation was attempted on something that is not
# a socket" on the very first asyncio.run() call.
#
# Fix: force WindowsProactorEventLoopPolicy BEFORE any other module
# touches asyncio.get_event_loop() / new_event_loop().
#
# ProactorEventLoop uses IOCP (I/O Completion Ports) internally instead
# of socket.socketpair() + select(), so it sidesteps the Winsock LSP
# filter corruption that breaks SelectorEventLoop on corp laptops.
#
# Must execute BEFORE any other `import asyncio` sub-importer runs —
# so we set the policy right after __future__ annotations.
# =====================================================================
import asyncio as _asyncio
import sys as _sys
import os as _os  # noqa: E402  (needed for IOCP probe kill-switch env var)
try:
    from pathlib import Path as _Path  # noqa: E402  (early import needed for IOCP probe banner below)
except Exception:  # pragma: no cover - pathlib is stdlib
    from pathlib import Path as _Path  # type: ignore

# =====================================================================
# 轻量级 env.env 早期加载：在 asyncio 策略设置和 IOCP 探测之前读取
# config/env.env 中的 RUN_BATCH_FORCE_SELECTOR_POLICY 等关键变量，
# 避免用户需要手动设置环境变量。
# =====================================================================
if _sys.platform == "win32":
    for _env_path in [
        _Path(__file__).resolve().parents[1] / "config" / "env.env",
        _Path(__file__).resolve().parents[1] / "env.env",
    ]:
        if _env_path.exists():
            try:
                with open(_env_path, "r", encoding="utf-8") as _f:
                    for _line in _f:
                        _line = _line.strip()
                        if _line and not _line.startswith("#") and "=" in _line:
                            _k, _v = _line.split("=", 1)
                            _k, _v = _k.strip(), _v.strip()
                            if _k and _k not in _os.environ:
                                _os.environ[_k] = _v
                break
            except Exception:
                pass

if _sys.platform == "win32":
    # 子进程兜底模式：强制用 WindowsSelectorEventLoopPolicy（绕过 Proactor 的 IOCP 问题）
    _force_selector = _os.environ.get("RUN_BATCH_FORCE_SELECTOR_POLICY", "").lower() in ("1", "true", "yes")
    try:
        if _force_selector:
            _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
            import warnings as _w2
            _w2.warn(
                "[run_batch.py] 已强制启用 WindowsSelectorEventLoopPolicy "
                "(RUN_BATCH_FORCE_SELECTOR_POLICY=1)",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            _asyncio.set_event_loop_policy(_asyncio.WindowsProactorEventLoopPolicy())
    except Exception as _e_pol:  # pragma: no cover - extremely rare
        import warnings as _w
        _w.warn(
            f"[run_batch.py] Could not set event loop policy "
            f"({_e_pol!r}). asyncio.run() calls below may fail with "
            f"WinError 10038; please run in PowerShell directly instead.",
            RuntimeWarning,
            stacklevel=2,
        )

# =====================================================================
# [WINDOWS FIX] 终极 asyncio 可用性探测 + subprocess 子进程重跑兜底
#
# 已知企业环境下的两种 Winsock LSP 污染模式：
#   模式 A：ProactorEventLoop 构造函数直接失败（IOCP 权限剥夺）
#        → 旧的 IOCP 探测能捕获，直接 exit(87)
#   模式 B：构造函数成功，但真跑 asyncio.run() 时 _selector 属性缺失
#        → 旧探测捕获不到，用户在 Step1 才看到 AttributeError
#
# 终极方案：在脚本顶部做一次"真刀真枪"的 asyncio 探测，实际跑一个
# asyncio.sleep(0.001)。若两种模式任一种触发，立即用 subprocess 启动
# 一个干净的 Python 子进程重跑整个脚本 —— 干净进程不会继承父进程的
# DLL 注入/LSP 挂钩污染，asyncio 基本都能正常工作。
#
# 子进程保护：通过 RUN_BATCH_SUBPROCESS_REEXEC=1 环境变量防止无限递归。
# =====================================================================
if _sys.platform == "win32":  # pragma: no cover - environment-specific
    _REEXEC_FLAG = "RUN_BATCH_SUBPROCESS_REEXEC"
    _is_reexec_child = _os.environ.get(_REEXEC_FLAG, "") == "1"

    if not _is_reexec_child:
        _asyncio_works = False
        _probe_error = None  # type: Optional[BaseException]  # noqa: UP007

        # KILL-SWITCH: skip the IOCP probe entirely via env var.
        # Set RUN_BATCH_SKIP_IOCP_PROBE=1 when you KNOW you are running in a
        # standalone PowerShell/CMD window (IOCP works) but the probe somehow
        # false-negatives — see _tmp_run_batch3.py which sets this automatically
        # because it is ALWAYS launched from a standalone shell by the user.
        # Also auto-skip when FORCE_SELECTOR_POLICY is set (no point probing
        # IOCP when we're going to force SelectorEventLoop anyway).
        _force_selector = _os.environ.get("RUN_BATCH_FORCE_SELECTOR_POLICY", "").lower() in ("1", "true", "yes")
        _skip_probe = _os.environ.get("RUN_BATCH_SKIP_IOCP_PROBE", "").lower() in ("1", "true", "yes")

        if _skip_probe or _force_selector:
            # 用户显式要求跳过 → 相信用户，直接走正常流程
            _asyncio_works = True
        else:
            try:
                # ====== 模式 A 检测：ProactorEventLoop 构造函数 ======
                _probe_loop = _asyncio.WindowsProactorEventLoop()
                try:
                    # ====== 模式 B 检测：真跑一个最小 await ======
                    # 只有实际跑 run_until_complete(sleep(...)) 才能
                    # 检测出 _selector 属性缺失的深层 Winsock 污染。
                    _probe_loop.run_until_complete(_asyncio.sleep(0.001))
                    _asyncio_works = True
                except AttributeError as _e_attr:
                    # 典型：'_Win32ProactorEventLoop' object has no attribute '_selector'
                    _probe_error = _e_attr
                except BaseException as _e_run:  # noqa: BLE001 - catch-all for probe
                    _probe_error = _e_run
                finally:
                    try:
                        _probe_loop.close()
                    except Exception:
                        pass
            except RuntimeError as _e_iocp:
                # 典型：RuntimeError: I/O completion port not available
                _probe_error = _e_iocp
            except BaseException as _e_ctor:  # noqa: BLE001 - catch-all for probe
                _probe_error = _e_ctor

        if not _asyncio_works:
            # ========== 探测失败 → 用干净子进程重跑 ==========
            import subprocess as _subprocess

            _self_script = str(_Path(__file__).resolve())
            _child_env = _os.environ.copy()
            _child_env[_REEXEC_FLAG] = "1"
            # 确保子进程不再重复做相同探测（kill-switch 防止递归）
            # 注意：保留 RUN_BATCH_SKIP_IOCP_PROBE 原值，不覆盖

            _warn_lines = [
                "",
                "=" * 80,
                "  ⚠️  检测到当前进程 asyncio 不可用 (Winsock LSP 污染)",
            ]
            if _probe_error is not None:
                _warn_lines.append(f"  原因: {type(_probe_error).__name__}: {_probe_error}")
            _warn_lines += [
                "  策略: 自动启动干净 Python 子进程重跑本脚本...",
                "  (子进程不会继承父进程的 DLL 注入/LSP 挂钩)",
                "=" * 80,
                "",
            ]
            print("\n".join(_warn_lines), file=_sys.stderr)
            _sys.stderr.flush()

            try:
                # 用干净子进程执行：继承 stdin/stdout/stderr，输出实时可见
                _completed = _subprocess.run(
                    [_sys.executable, _self_script, *_sys.argv[1:]],
                    env=_child_env,
                    check=False,
                )
                _sys.exit(_completed.returncode)
            except FileNotFoundError:
                print(f"[FATAL] 找不到 Python 解释器: {_sys.executable}", file=_sys.stderr)
                _sys.exit(2)
            except KeyboardInterrupt:
                print("\n[用户中断] 子进程被 Ctrl+C 终止", file=_sys.stderr)
                _sys.exit(130)


import argparse
import asyncio  # noqa: E402  (imported as _asyncio above for policy, re-imported here for API use)
import json
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PIPELINE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "runs" / "ontology" / "pipeline"
MASTER_PATH = PROJECT_ROOT / "outputs" / "runs" / "ontology" / "master_ontology_unified.json"


# =========================================================================
# Step 1: Compile Pipeline（同 ontology_cli.py run_compile）
# =========================================================================
def _run_asyncio_safe(coro) -> object:
    """[WINDOWS 多重兜底] 安全地运行一个 asyncio 协程。

    企业 Windows 环境下 asyncio 可能遭遇以下多种故障模式，本函数逐一兜底：
      尝试 1: 原生 asyncio.run()（WindowsProactorEventLoopPolicy 默认）
      尝试 2: 捕获 _selector 缺失 → 手动从 asyncio.windows_events 导入
              ProactorEventLoop + IocpProactor，补到 _selector/_proactor
      尝试 3: 切换 WindowsSelectorEventLoopPolicy 重跑
      尝试 4: 手动从 asyncio.windows_events 构造 SelectorEventLoop 跑
      尝试 5: subprocess 干净子进程重跑整个脚本（强制 SelectorPolicy + 防递归）

    Returns:
        协程的返回值。若走了子进程重跑（尝试 5），本函数永不返回（sys.exit）。
    """
    import subprocess as _sp

    # ====== 尝试 1: 原生 asyncio.run() ======
    try:
        return asyncio.run(coro)
    except AttributeError as _e1:
        _err_str = str(_e1).lower()
        if not (sys.platform == "win32"
                and ("proactoreventloop" in _err_str or "win32proactor" in _err_str)
                and "_selector" in _err_str):
            # 不是我们已知的 _selector 缺失 → 直接重抛
            raise
        print("[asyncio][FIX1] 检测到 _selector 缺失，尝试手动注入 IocpProactor...",
              file=sys.stderr)

    # ====== 尝试 2: 手动 new ProactorEventLoop (从 windows_events 导入) + 补 _selector ======
    if sys.platform == "win32":
        try:
            from asyncio import windows_events as _we  # type: ignore[attr-defined]
            loop2 = _we.ProactorEventLoop()
            if not hasattr(loop2, "_selector") or getattr(loop2, "_selector", None) is None:
                try:
                    _fresh_proactor = _we.IocpProactor()
                    loop2._proactor = _fresh_proactor  # type: ignore[attr-defined]
                    loop2._selector = _fresh_proactor  # type: ignore[attr-defined]
                    print("[asyncio][FIX1] 已手动注入新的 IocpProactor", file=sys.stderr)
                except Exception as _e_iocp:
                    print(f"[asyncio][FIX1] IocpProactor 注入失败: {_e_iocp}", file=sys.stderr)
            asyncio.set_event_loop(loop2)
            try:
                return loop2.run_until_complete(coro)
            finally:
                try:
                    loop2.close()
                except Exception:
                    pass
        except BaseException as _e2:
            print(f"[asyncio][FIX1] 修补 Proactor 仍失败: {type(_e2).__name__}: {_e2}",
                  file=sys.stderr)

    # ====== 尝试 3: 临时切换到 WindowsSelectorEventLoopPolicy ======
    if sys.platform == "win32":
        print("[asyncio][FIX2] 尝试切换到 WindowsSelectorEventLoopPolicy...",
              file=sys.stderr)
        _old_policy = asyncio.get_event_loop_policy()
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
            asyncio.set_event_loop(None)
            return asyncio.run(coro)
        except BaseException as _e3:
            print(f"[asyncio][FIX2] SelectorPolicy 仍失败: {type(_e3).__name__}: {_e3}",
                  file=sys.stderr)
        finally:
            try:
                asyncio.set_event_loop_policy(_old_policy)
            except Exception:
                pass

    # ====== 尝试 4: 直接从 windows_events 手动构造 SelectorEventLoop ======
    if sys.platform == "win32":
        try:
            from asyncio import windows_events as _we2  # type: ignore[attr-defined]
            print("[asyncio][FIX3] 尝试手动构造 SelectorEventLoop...", file=sys.stderr)
            loop4 = _we2.SelectorEventLoop()
            asyncio.set_event_loop(loop4)
            try:
                return loop4.run_until_complete(coro)
            finally:
                try:
                    loop4.close()
                except Exception:
                    pass
        except BaseException as _e4:
            print(f"[asyncio][FIX3] 手动 SelectorEventLoop 仍失败: {type(_e4).__name__}: {_e4}",
                  file=sys.stderr)

    # ====== 尝试 5: 启动干净 Python 子进程重跑整个脚本（强制 SelectorPolicy） ======
    _SELF_REEXEC_FLAG = "RUN_BATCH_STEP1_REEXEC_DONE"
    if os.environ.get(_SELF_REEXEC_FLAG) == "1":
        raise RuntimeError(
            "[asyncio][FATAL] 所有 5 层 asyncio 兜底策略均无效（含子进程重跑）。"
            "请尝试：(1) 全新 CMD/PowerShell 窗口运行；(2) 安装 Python 3.12+ 替换当前 3.11。"
        )

    print("[asyncio][FIX4] 启动干净 Python 子进程重跑整个脚本（强制 SelectorPolicy + 防递归）...",
          file=sys.stderr)
    sys.stderr.flush()

    _child_env = os.environ.copy()
    _child_env[_SELF_REEXEC_FLAG] = "1"
    _child_env["RUN_BATCH_FORCE_SELECTOR_POLICY"] = "1"
    _self_path = str(Path(__file__).resolve())

    try:
        _ret = _sp.run(
            [sys.executable, _self_path, *sys.argv[1:]],
            env=_child_env,
            check=False,
        )
        sys.exit(_ret.returncode)
    except FileNotFoundError:
        print(f"[asyncio][FATAL] 找不到 Python: {sys.executable}", file=sys.stderr)
        sys.exit(2)


def step1_compile(source_path: str, domain: str, strict_governance: bool,
                  max_cost_usd: float = 0.0, max_calls: int = 0, max_total_tokens: int = 0,
                  degraded_offline: bool = False) -> tuple[bool, float]:
    """Run ontology extraction pipeline. Returns (success, elapsed_seconds)."""
    try:
        from ontology_compiler.config import load_runtime_env
        loaded = load_runtime_env(force_reload=True)
        if loaded:
            print(f"[ENV] Loaded {len(loaded)} vars from config/env.env")
    except Exception as _exc:
        print(f"[ENV-WARN] Could not load environment config: {_exc}", file=sys.stderr)

    # ============================================================
    # C-1: LLM HARD BUDGET GATE — arm before ANY provider call.
    # Safe defaults (0) = unlimited, fully backward compatible.
    # ============================================================
    budget_any_active = max_cost_usd > 0 or max_calls > 0 or max_total_tokens > 0
    if budget_any_active or degraded_offline:
        try:
            from src.application.bootstrap.llm_gateway import get_llm_gateway
            gateway = get_llm_gateway()
            if budget_any_active:
                gateway.set_budget_policy(
                    max_cost_usd=max_cost_usd,
                    max_calls=max_calls,
                    max_total_tokens=max_total_tokens,
                )
                print(
                    f"[GOV] [BudgetGate] Armed: max_cost=${max_cost_usd:.4f}, "
                    f"max_calls={max_calls}, max_tokens={max_total_tokens}"
                )
            if degraded_offline:
                print("[GOV] [DegradedMode] Forcing DEGRADE mode for LLM-dependent stages")
        except Exception as _exc:  # pragma: no cover - best-effort arming
            print(f"[GOV] [WARN] Could not arm LLM governance: {_exc!r}", file=sys.stderr)

    args = Namespace(
        source=source_path,
        domain=domain,
        strict_governance=strict_governance,
    )

    t0 = time.time()
    try:
        from src.pipeline.pipeline_router import get_production_pipeline
    except Exception as e:
        print(f"[Step1][FATAL] 无法加载 production pipeline: {e}")
        return False, 0.0

    output_dir = str(PIPELINE_OUTPUT_DIR)
    context = {
        "source_path": args.source,
        "domain": args.domain,
        "output_dir": output_dir,
        "metadata": {
            "strict_governance": args.strict_governance,
            # C-1 governance propagation (same shape as ontology_cli context)
            "governance": {
                "max_cost_usd": float(max_cost_usd or 0.0),
                "max_calls": int(max_calls or 0),
                "max_total_tokens": int(max_total_tokens or 0),
                "degraded_offline": bool(degraded_offline),
            },
        },
    }
    pipeline = get_production_pipeline(context=context)
    result = _run_asyncio_safe(pipeline.run())
    elapsed = round(time.time() - t0, 2)

    ok = result.status.value == "success"
    tag = "✅" if ok else "❌"
    print(f"\n[Step1] {tag} Pipeline {result.status.value} in {elapsed}s")
    print(f"[Step1] Stages: {result.stats.completed_stages}/{result.stats.total_stages} completed")
    stats = result.stats.to_dict() if hasattr(result.stats, "to_dict") else {}
    for k, v in stats.items():
        if k not in ("completed_stages", "total_stages"):
            print(f"[Step1]   {k}: {v}")
    if result.errors:
        print(f"[Step1] Errors ({len(result.errors)}):")
        for err in result.errors[:5]:
            print(f"[Step1]   - {err}")
    return ok, elapsed


def find_latest_run_id(since_ts: float) -> Optional[str]:
    """Step1 完成后，根据修改时间找到最新创建的 MAIN-* 目录作为 Run ID。

    原理：把 PIPELINE_OUTPUT_DIR 下所有以 MAIN- 开头的目录按 mtime 降序，
    取第一个 mtime >= since_ts 的，就是刚跑完的这次。
    """
    candidates = []
    for d in PIPELINE_OUTPUT_DIR.glob("MAIN-*"):
        if not d.is_dir():
            continue
        mt = d.stat().st_mtime
        if mt >= since_ts - 5:  # 5 秒容错
            candidates.append((mt, d.name))
    if not candidates:
        # 没找到就退回取最新的那个（容错）
        all_main = sorted(
            (d.stat().st_mtime, d.name)
            for d in PIPELINE_OUTPUT_DIR.glob("MAIN-*")
            if d.is_dir()
        )
        if all_main:
            return all_main[-1][1]
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# =========================================================================
# Step 2: Merge fragments -> master（原 merge_batch10.py 逻辑已内嵌）
# =========================================================================
def step2_merge(run_id: str) -> bool:
    """Combine fragments from RUN_ID and merge into master_ontology_unified.json."""
    from src.pipeline.stages.merge import run_merge, _normalize_fragment

    run_dir = PIPELINE_OUTPUT_DIR / run_id
    fragment_dir = run_dir / "extract" / "files"
    if not fragment_dir.exists():
        print(f"[Step2][FATAL] Fragment 目录不存在: {fragment_dir}")
        return False

    print(f"\n[Step2][1/2] 组合 fragments: {fragment_dir}")
    combined: dict = {"nodes": [], "instances": [], "relationships": []}
    for frag_path in sorted(fragment_dir.glob("*/fragment_*.json")):
        with open(frag_path, "r", encoding="utf-8") as f:
            frag = json.load(f)
        n = len(frag.get("nodes", []))
        inst = len(frag.get("instances", []))
        rel = len(frag.get("relationships", []))
        combined["nodes"].extend(frag.get("nodes", []))
        combined["instances"].extend(frag.get("instances", []))
        combined["relationships"].extend(frag.get("relationships", []))
        combined.setdefault("source_files", []).append(frag.get("run_id", str(frag_path)))
        print(f"[Step2]   + {frag_path.name}:  nodes={n}  instances={inst}  rels={rel}")
    print(
        f"[Step2]   Total: nodes={len(combined['nodes'])}  "
        f"instances={len(combined['instances'])}  rels={len(combined['relationships'])}"
    )

    combined_frag = _normalize_fragment(combined)
    temp_frag_path = run_dir / f"fragment_{run_id}.json"
    with open(temp_frag_path, "w", encoding="utf-8") as f:
        json.dump(combined_frag, f, ensure_ascii=False, indent=2)
    print(f"[Step2]   组合后的 fragment 写入: {temp_frag_path}")

    print(f"\n[Step2][2/2] 执行 merge 到 master: {MASTER_PATH}")
    merge_args = Namespace(
        stage="full",
        domain="AUTO",
        output_dir=str(run_dir),
        run_dir=str(run_dir),
        master_path=str(MASTER_PATH),
        run_id=run_id,
        fragment=str(temp_frag_path),
    )
    result = run_merge(merge_args)
    ok = result.success
    tag = "✅" if ok else "❌"
    print(f"\n[Step2] {tag} Merge {result.message}")
    if result.data:
        for k in ("new_nodes", "merged_nodes", "merged_relationships",
                  "duplicate_nodes", "removed_dangling_relationships"):
            if k in result.data:
                print(f"[Step2]   {k}: {result.data[k]}")
        go = result.data.get("global_optimization", {})
        if go:
            summary = {k: go[k] for k in (
                "cross_layer_merged", "same_layer_merged", "mser_merged",
                "dangling_rels_fixed", "related_to_refined", "duplicate_meta_ids_fixed",
            ) if k in go}
            print(f"[Step2]   global_opt: {summary}")
    return ok


# =========================================================================
# Step 3: Regression Test + Export
# =========================================================================
def step3_regression() -> bool:
    """Run regression_test.py. Returns True if ALL TESTS PASSED."""
    print(f"\n[Step3][1/2] 运行回归测试: {MASTER_PATH}")
    try:
        from scripts.regression_test import compute_metrics, BASELINE, format_report
    except ImportError:
        # fallback 到 subprocess 调用（兼容 import 路径差异）
        import subprocess
        cp = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "regression_test.py"),
             "--master", str(MASTER_PATH)],
            cwd=str(PROJECT_ROOT),
        )
        return cp.returncode == 0

    metrics = compute_metrics(str(MASTER_PATH))
    # 借用 regression_test 的 format 逻辑
    passed_all = True
    lines = []
    lines.append("=" * 60)
    lines.append("  Regression Test Report")
    lines.append("=" * 60)
    lines.append(f"  {'Metric':<30s} {'Value':>10s}  {'Range':>20s}   {'Status':<7s}")
    lines.append("  " + "-" * 30 + " " + "-" * 10 + " " + "-" * 20 + " " + "-" * 7)
    for key, spec in BASELINE.items():
        val = metrics.get(key, 0)
        lo, hi = spec["min"], spec["max"]
        ok = lo <= val <= hi
        if not ok:
            passed_all = False
        status = "PASS" if ok else "FAIL"
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        lines.append(
            f"  {spec['description']:<30s} {val_str:>10s}  [{lo}, {hi}]   {status}"
        )
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"  [{'PASS' if passed_all else 'FAIL'}] ALL TESTS {'PASSED' if passed_all else 'FAILED'}")
    lines.append("=" * 60)
    print("\n".join(lines))

    report_path = PROJECT_ROOT / "outputs" / "runs" / "ontology" / "regression_report.json"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "passed": passed_all}, f, indent=2, ensure_ascii=False)
        print(f"[Step3]   Report saved: {report_path}")
    except Exception:
        pass
    return passed_all


def step3_export() -> bool:
    """Run export stage (master -> layer jsons)."""
    print(f"\n[Step3][2/2] 重新导出分层图 (layer_L0 ~ L3)")
    from src.pipeline.stages.export import run_export as run_export_stage
    args = Namespace(master_path=str(MASTER_PATH))
    result = run_export_stage(args)
    ok = result.success
    tag = "✅" if ok else "❌"
    print(f"[Step3] {tag} Export {result.message}")
    if result.data:
        summary = result.data.get("export_summary", {})
        if summary:
            print(f"[Step3]   nodes={summary.get('total_nodes')}  "
                  f"instances={summary.get('total_instances')}  "
                  f"relationships={summary.get('total_relationships')}")
            layers = summary.get("layers", {})
            if layers:
                print(f"[Step3]   layers: {layers}")
        if result.data.get("export_base_dir"):
            print(f"[Step3]   输出目录: {result.data['export_base_dir']}")
    return ok


# =========================================================================
# Main
# =========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run_batch",
        description="一键 Batch 处理：Step1 compile → Step2 merge → Step3 回归+导出，一条龙。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 最常用：只给文件夹路径，剩下全自动
  python scripts/run_batch.py --folder "C:\\Users\\...\\BOM\\batch10"

  # 指定 domain（默认 AUTO 自动检测）
  python scripts/run_batch.py --folder "C:\\...\\batch10" --domain SUPPLY_CHAIN
""",
    )
    src_grp = parser.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("--folder", help="要处理的 batch 文档所在的文件夹")
    src_grp.add_argument("--file", help="要处理的单个文档文件")
    parser.add_argument("--domain", default="AUTO", help="业务域 (默认 AUTO，自动检测)")
    parser.add_argument("--strict-governance", action="store_true", help="启用严格治理模式")
    parser.add_argument("--skip-regression", action="store_true", help="跳过 Step3.1 回归测试（不推荐）")
    parser.add_argument("--skip-export", action="store_true", help="跳过 Step3.2 导出分层图（不推荐）")
    # C-1: LLM hard budget gates — same defaults as ontology_cli.py (0 = unlimited, safe)
    parser.add_argument("--max-cost-usd", type=float, default=0.0,
                        help="LLM 预算上限（USD，到达后阻止后续调用；0=无限制）")
    parser.add_argument("--max-calls", type=int, default=0,
                        help="LLM 调用次数上限（0=无限制）")
    parser.add_argument("--max-total-tokens", type=int, default=0,
                        help="LLM 总 token 上限（prompt+completion；0=无限制）")
    parser.add_argument("--degraded-offline", action="store_true",
                        help="强制 LLM 相关阶段进入降级模式（仅规则处理，不调 LLM）")
    args = parser.parse_args()

    source_path = args.folder or args.file
    source_label = Path(source_path).name

    print("=" * 72, flush=True)
    print(f"  🔥 ONTOLOGY BATCH PIPELINE (Step1+2+3 合并版)")
    print("=" * 72, flush=True)
    print(f"  源路径: {source_path}", flush=True)
    print(f"  源标识: {source_label}", flush=True)
    print(f"  Domain: {args.domain}", flush=True)
    print(f"  Master: {MASTER_PATH}", flush=True)
    print("=" * 72, flush=True)

    overall_start = time.time()
    ts_before = time.time()

    # ---- Step1 -----------------------------------------------------------
    ok_s1, elapsed_s1 = step1_compile(
        source_path, args.domain, args.strict_governance,
        max_cost_usd=args.max_cost_usd,
        max_calls=args.max_calls,
        max_total_tokens=args.max_total_tokens,
        degraded_offline=args.degraded_offline,
    )
    if not ok_s1:
        print("\n[FATAL] Step1 compile 失败，终止流程。")
        return 1

    run_id = find_latest_run_id(ts_before)
    if not run_id:
        print("[FATAL] Step1 完成但无法定位 Run ID，无法继续 merge。")
        print(f"        请手动检查 {PIPELINE_OUTPUT_DIR} 下最新的 MAIN-* 目录。")
        return 1
    print(f"\n[INFO] 自动检测到 Run ID: {run_id}")

    # ---- Step2 -----------------------------------------------------------
    ok_s2 = step2_merge(run_id)
    if not ok_s2:
        print("\n[FATAL] Step2 merge 失败，终止流程（回归测试跳过）。")
        return 2

    # ---- Step3 -----------------------------------------------------------
    ok_s3_reg = True
    ok_s3_exp = True

    if not args.skip_regression:
        ok_s3_reg = step3_regression()
    else:
        print("\n[Step3][1/2] ⚠️  跳过回归测试 (--skip-regression)")

    if not args.skip_export:
        ok_s3_exp = step3_export()
    else:
        print("\n[Step3][2/2] ⚠️  跳过导出分层图 (--skip-export)")

    # ---- Summary ---------------------------------------------------------
    total = round(time.time() - overall_start, 2)
    passed_all = ok_s1 and ok_s2 and ok_s3_reg and ok_s3_exp
    tag = "🎉" if passed_all else "⚠️"

    print()
    print("=" * 72)
    print(f"  {tag} 全流程总结")
    print("=" * 72)
    print(f"  源: {source_label}")
    print(f"  Run ID: {run_id}")
    print(f"  Step1 Compile    : {'✅ PASS' if ok_s1  else '❌ FAIL'}  ({elapsed_s1}s)")
    print(f"  Step2 Merge      : {'✅ PASS' if ok_s2  else '❌ FAIL'}")
    print(f"  Step3 Regression : {'✅ PASS' if ok_s3_reg else '❌ FAIL'}{' (skipped)' if args.skip_regression else ''}")
    print(f"  Step3 Export     : {'✅ PASS' if ok_s3_exp else '❌ FAIL'}{' (skipped)' if args.skip_export else ''}")
    print(f"  总耗时: {total}s")
    print("=" * 72)
    if passed_all:
        print("  ✅ 全部完成！下次处理新 batch 只需把 --folder 改成新路径即可。")
    else:
        print("  ❌ 有失败步骤，请根据上面的日志排查。")
    print("=" * 72)
    return 0 if passed_all else 3


if __name__ == "__main__":
    sys.exit(main())
