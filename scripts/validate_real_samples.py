#!/usr/bin/env python3
"""Run real-sample validation against the ontology extraction pipeline.

This script is intentionally lightweight:
- reads plain text/markdown samples from a folder
- runs TrueUnifiedExtractor directly
- prints prompt length and layer counts
- optionally writes a JSON report

Usage:
    python scripts/validate_real_samples.py --samples ./samples --domain AUTO
    python scripts/validate_real_samples.py --samples ./samples --output report.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from ontology_compiler.services.true_unified_extractor import TrueUnifiedExtractor
from ontology_compiler.rigor.llm_backend import UnifiedLLMClient


@dataclass
class SampleResult:
    file: str
    prompt_chars: int
    l0: int
    l1: int
    l2: int
    l25: int
    l3: int
    relationships: int
    quality_score: float
    degraded: bool
    error: Optional[str] = None


async def run_sample(extractor: TrueUnifiedExtractor, path: Path, domain: Optional[str]) -> SampleResult:
    content = path.read_text(encoding="utf-8", errors="ignore")
    prompt = extractor._build_unified_prompt(content, domain=domain, use_context=True)
    try:
        result = await extractor.extract_all_layers(
            text=content,
            file_path=path,
            domain=domain,
            use_context=True,
            enable_validation=True,
            max_retries=1,
            fast_mode=False,
        )
        return SampleResult(
            file=str(path),
            prompt_chars=len(prompt),
            l0=len(result.l0_entities),
            l1=len(result.l1_entities),
            l2=len(result.l2_entities),
            l25=len(result.l25_entities),
            l3=len(result.l3_entities),
            relationships=len(result.relationships),
            quality_score=float(result.quality_score or 0.0),
            degraded=bool(result.degraded),
        )
    except Exception as exc:
        return SampleResult(
            file=str(path),
            prompt_chars=len(prompt),
            l0=0,
            l1=0,
            l2=0,
            l25=0,
            l3=0,
            relationships=0,
            quality_score=0.0,
            degraded=True,
            error=str(exc),
        )


async def main_async(args: argparse.Namespace) -> int:
    sample_dir = Path(args.samples)
    if not sample_dir.exists():
        print(f"[VALIDATE] Sample directory not found: {sample_dir}")
        return 1

    files = [p for p in sorted(sample_dir.rglob("*")) if p.suffix.lower() in {".txt", ".md"} and p.is_file()]
    if not files:
        print("[VALIDATE] No .txt/.md samples found")
        return 1

    provider = args.provider if args.provider != "AUTO" else None
    fallback_providers = args.fallback if args.fallback else None
    llm_client = UnifiedLLMClient(provider=provider, fallback_providers=fallback_providers)
    extractor = TrueUnifiedExtractor(llm_client=llm_client)
    results = []
    for path in files:
        print(f"[VALIDATE] Running sample: {path.name}")
        res = await run_sample(extractor, path, None if args.domain == "AUTO" else args.domain)
        results.append(res)
        print(
            f"[VALIDATE] prompt={res.prompt_chars} chars | L0={res.l0} L1={res.l1} L2={res.l2} L2.5={res.l25} L3={res.l3} rel={res.relationships} quality={res.quality_score:.2f} degraded={res.degraded}"
        )
        if res.error:
            print(f"[VALIDATE]   error: {res.error}")

    report = {
        "sample_count": len(results),
        "domain": args.domain,
        "results": [asdict(r) for r in results],
    }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[VALIDATE] Wrote report to {out}")

    failures = [r for r in results if r.error]
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate real samples for ontology extraction")
    parser.add_argument("--samples", required=True, help="Sample directory containing .txt/.md files")
    parser.add_argument("--domain", default="AUTO", help="Domain hint or AUTO")
    parser.add_argument("--provider", default="AUTO", help="Force LLM provider (deepseek, dify, AUTO)")
    parser.add_argument("--fallback", default="", help="Comma-separated fallback providers")
    parser.add_argument("--output", help="Optional JSON report output path")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
