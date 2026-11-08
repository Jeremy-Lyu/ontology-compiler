from __future__ import annotations

"""Smoke-test the regression manifest and runner contract.

This keeps the regression flow honest without requiring a full pipeline run.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

MANIFEST = Path("config/regression_samples.json")
RUNNER = Path("scripts/run_regression_samples.py")
MAKEFILE = Path("Makefile")
COMPARE = Path("scripts/compare_pipeline_results.py")


def _validate_golden_blocks(manifest: dict) -> None:
    """Validate every golden_semantic block is structurally usable.

    Each declared golden entity must have a name; each relation must have
    source/target/type; reject_relation_types must be non-empty when present.
    Fails the smoke test early so a malformed golden set can't silently
    degrade the semantic gate.
    """
    blocks = 0
    for group in manifest.get("sample_groups", []):
        for sample in group.get("samples", []):
            golden = sample.get("golden_semantic")
            if not golden:
                continue
            blocks += 1
            sample_id = sample.get("sample_id", "?")
            for ent in golden.get("entities", []):
                assert str(ent.get("name") or "").strip(), f"golden entity missing name in {sample_id}"
            for rel in golden.get("relations", []):
                assert str(rel.get("source") or "").strip(), f"golden relation missing source in {sample_id}"
                assert str(rel.get("target") or "").strip(), f"golden relation missing target in {sample_id}"
                assert rel.get("types"), f"golden relation missing types in {sample_id}"
            for rejected in golden.get("rejected_entities", []):
                assert str(rejected.get("name") or "").strip(), f"rejected entity missing name in {sample_id}"
    assert blocks > 0, "at least one golden_semantic block required (E-2)"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["version"] == "2.1"
    assert manifest["artifact_conventions"]["run_root"] == "outputs/runs/ontology/pipeline/"
    assert manifest["compare_with"]["before"].endswith("regression_summary_baseline.json")
    assert manifest["compare_with"]["after"].endswith("latest_regression_summary.json")

    # E-2 (2026-08-15): golden semantic blocks must be structurally valid.
    _validate_golden_blocks(manifest)

    runner_text = RUNNER.read_text(encoding="utf-8")
    assert "--compare-only" in runner_text
    assert "--write-baseline" in runner_text
    assert "--baseline-output" in runner_text
    assert "--before" in runner_text
    assert "--after" in runner_text
    assert "golden_semantic" in runner_text  # E: runner must read golden sets
    assert "golden_metrics" in runner_text   # E: runner must evaluate golden sets

    compare_text = COMPARE.read_text(encoding="utf-8")
    assert "CORE_METRICS" in compare_text
    assert "--html" in compare_text
    assert "golden_entity_recall" in compare_text  # E-2: compare covers semantics

    makefile_text = MAKEFILE.read_text(encoding="utf-8")
    assert "regression-baseline" in makefile_text
    assert "regression-compare" in makefile_text
    assert "regression-smoke" in makefile_text

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        before_path = tmp / "before.json"
        after_path = tmp / "after.json"
        compare_output = tmp / "compare.json"
        runner_output_dir = tmp / "runner-output"
        before_path.write_text(json.dumps({"results": [], "overall_status": "PASS"}, indent=2), encoding="utf-8")
        after_path.write_text(json.dumps({"results": [], "overall_status": "PASS"}, indent=2), encoding="utf-8")

        subprocess.run(
            [
                sys.executable,
                str(COMPARE),
                "--before",
                str(before_path),
                "--after",
                str(after_path),
                "--output",
                str(compare_output),
                "--html",
            ],
            check=True,
        )
        assert compare_output.exists()
        assert compare_output.with_suffix(".md").exists()
        assert compare_output.with_suffix(".html").exists()

        subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--compare-only",
                "--before",
                str(before_path),
                "--after",
                str(after_path),
                "--output-dir",
                str(runner_output_dir),
            ],
            check=True,
        )
        assert any(runner_output_dir.glob("regression_reports/compare_*.json"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
