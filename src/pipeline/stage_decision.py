"""StageDecision — selective execution gate for C-4.

C-4 task: introduce a ``StageDecision`` so the pipeline orchestrator can
choose per-stage between:

- ``RUN``             — execute the stage handler normally
- ``SKIP_UNCHANGED``  — skip because the upstream input + config bundle is
                        byte-identical to the last successful run and the
                        stage has no observable side effects beyond its
                        artifact (so the cached artifact is still valid)
- ``REUSE_ARTIFACT``  — skip execution but replay the previous run's
                        ``StageResult`` from the artifact registry, so
                        downstream stages see the same shape
- ``DEGRADE``         — skip LLM-dependent computation in degraded/offline
                        mode, producing an explicit degraded result
- ``BLOCK``           — refuse to run: a hard precondition failed and
                        continuing would produce misleading output

Design rules (C-4 acceptance: "cache hit 不会错误复用陈旧结果; barrier
阶段仍保持确定性"):

1. **Barrier stages are never skipped.** ``audit``, ``dcs_audit``, ``guard``
   must always execute so quality verdicts reflect the current run's data.
2. **A cache hit requires a version-bundle match.** If
   ``CacheVersionBundle`` of the cached run != current bundle, the decision
   falls back to ``RUN`` — never reuse a stale result.
3. **Decisions are deterministic and replayable.** Same inputs ⇒ same
   decision, so the manifest can record the decision trace.
4. **Only re-run failed chunks, not the whole chain.** A stage whose
   upstream artifact is unchanged and whose own last run succeeded can be
   ``REUSE_ARTIFACT``; a stage whose last run *failed* is always ``RUN``.
5. **DEGRADE only applies to LLM-dependent stages.** Deterministic stages
   (merge/dedup/export) always run even in offline mode.

This module is pure logic — no I/O, no imports of pipeline internals beyond
dataclasses — so it is fully unit-testable without a running pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

# P1 / C-4: canonical definitions live in core.contracts so both
# ontology_compiler and the application layer share the same class
# identity. Re-export here for backward compatibility.
from core.contracts import (  # noqa: F401
    CacheVersionBundle,
    compute_versioned_cache_key,
    version_bundles_match,
)

logger = logging.getLogger(__name__)


class StageDecision(str, Enum):
    """Per-stage execution decision (C-4)."""

    RUN = "RUN"
    SKIP_UNCHANGED = "SKIP_UNCHANGED"
    REUSE_ARTIFACT = "REUSE_ARTIFACT"
    DEGRADE = "DEGRADE"
    BLOCK = "BLOCK"


# Stages that must always execute so the quality verdict reflects the
# current run (determinism guarantee for barrier stages).
BARRIER_STAGES: frozenset[str] = frozenset({"audit", "dcs_audit", "guard"})

# Stages that call the LLM and therefore may be skipped in offline /
# degraded mode. Deterministic stages (merge, dedup, export, ...) always
# run because they have no LLM cost to avoid.
LLM_DEPENDENT_STAGES: frozenset[str] = frozenset(
    {"extract", "emergent_reasoning"}
)

# Stages whose output is a pure function of their input artifact and the
# config bundle (no external state, no LLM). These are the only stages
# eligible for SKIP_UNCHANGED / REUSE_ARTIFACT.
DETERMINISTIC_STAGES: frozenset[str] = frozenset(
    {"merge", "dedup", "graph_fusion", "relationship"}
)


@dataclass(frozen=True)
class StageDecisionInput:
    """Inputs to :func:`compute_stage_decision` (C-4).

    All fields are plain values so the decision is deterministic and
    replayable from the manifest.
    """

    stage: str
    execution_mode: str  # "production" | "degraded" | "offline"
    source_hash: str
    version_bundle: CacheVersionBundle
    cached_success: bool = False
    cached_source_hash: str = ""
    cached_version_bundle: Optional[CacheVersionBundle] = None
    has_cached_artifact: bool = False
    upstream_changed: bool = True
    retry_failed_chunks: bool = True


@dataclass
class StageDecisionResult:
    """Outcome of :func:`compute_stage_decision`."""

    decision: StageDecision
    reason: str
    stage: str
    version_bundle: CacheVersionBundle
    source_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "decision": self.decision.value,
            "reason": self.reason,
            "version_bundle": self.version_bundle.to_dict(),
            "source_hash": self.source_hash,
        }


def compute_stage_decision(inp: StageDecisionInput) -> StageDecisionResult:
    """Compute the per-stage execution decision (C-4 core logic).

    This is a pure function: identical inputs ⇒ identical output, so the
    decision trace can be recorded in the manifest and replayed.

    Decision precedence (first match wins):

    1. BLOCK — source hash missing (cannot fingerprint the input).
    2. RUN — barrier stage: always execute for deterministic quality verdicts.
    3. DEGRADE — LLM-dependent stage in degraded/offline mode.
    4. RUN — no cached artifact or upstream input changed since last run.
    5. RUN — cached run failed and retry-failed-chunks is enabled.
    6. REUSE_ARTIFACT — deterministic stage, same source hash + version bundle,
       cached artifact present, upstream unchanged.
    7. SKIP_UNCHANGED — deterministic stage, same source hash + version bundle,
       but no cached artifact to replay (skip is still safe because the stage
       is a pure function with no external side effects).
    8. RUN — fallback.
    """
    stage = inp.stage
    vb = inp.version_bundle

    # 1. BLOCK: cannot fingerprint the input.
    if not inp.source_hash:
        return StageDecisionResult(
            decision=StageDecision.BLOCK,
            reason="source hash missing; cannot fingerprint input for cache key",
            stage=stage,
            version_bundle=vb,
            source_hash=inp.source_hash,
        )

    # 2. Barrier stages always run — determinism guarantee.
    if stage in BARRIER_STAGES:
        return StageDecisionResult(
            decision=StageDecision.RUN,
            reason="barrier stage: must execute for deterministic quality verdict",
            stage=stage,
            version_bundle=vb,
            source_hash=inp.source_hash,
        )

    # 3. LLM-dependent stages may degrade in non-production mode.
    if stage in LLM_DEPENDENT_STAGES and inp.execution_mode != "production":
        return StageDecisionResult(
            decision=StageDecision.DEGRADE,
            reason=f"LLM-dependent stage in {inp.execution_mode} mode",
            stage=stage,
            version_bundle=vb,
            source_hash=inp.source_hash,
        )

    # Only deterministic stages are eligible for cache reuse.
    if stage not in DETERMINISTIC_STAGES:
        return StageDecisionResult(
            decision=StageDecision.RUN,
            reason="non-deterministic stage: always execute",
            stage=stage,
            version_bundle=vb,
            source_hash=inp.source_hash,
        )

    # 4. No cached artifact → must run.
    if not inp.has_cached_artifact:
        return StageDecisionResult(
            decision=StageDecision.RUN,
            reason="no cached artifact available",
            stage=stage,
            version_bundle=vb,
            source_hash=inp.source_hash,
        )

    # 5. Upstream changed → must run.
    if inp.upstream_changed:
        return StageDecisionResult(
            decision=StageDecision.RUN,
            reason="upstream artifact changed since last cached run",
            stage=stage,
            version_bundle=vb,
            source_hash=inp.source_hash,
        )

    # 6. Version bundle mismatch → stale, must run (never reuse stale).
    if not version_bundles_match(vb, inp.cached_version_bundle):
        return StageDecisionResult(
            decision=StageDecision.RUN,
            reason="version bundle mismatch; cached result is stale",
            stage=stage,
            version_bundle=vb,
            source_hash=inp.source_hash,
        )

    # 7. Source hash mismatch → stale, must run.
    if inp.cached_source_hash != inp.source_hash:
        return StageDecisionResult(
            decision=StageDecision.RUN,
            reason="source hash mismatch; cached result is stale",
            stage=stage,
            version_bundle=vb,
            source_hash=inp.source_hash,
        )

    # 8. Cached run failed → re-run (only failed chunks, not whole chain).
    if not inp.cached_success:
        if inp.retry_failed_chunks:
            return StageDecisionResult(
                decision=StageDecision.RUN,
                reason="previous run failed; re-running failed chunks only",
                stage=stage,
                version_bundle=vb,
                source_hash=inp.source_hash,
            )
        return StageDecisionResult(
            decision=StageDecision.BLOCK,
            reason="previous run failed and retry-failed-chunks disabled",
            stage=stage,
            version_bundle=vb,
            source_hash=inp.source_hash,
        )

    # 9. All preconditions met → reuse artifact.
    return StageDecisionResult(
        decision=StageDecision.REUSE_ARTIFACT,
        reason="source hash + version bundle match; cached artifact available",
        stage=stage,
        version_bundle=vb,
        source_hash=inp.source_hash,
    )


@dataclass
class StageDecisionTrace:
    """Accumulated decision trace for a full pipeline run (C-4 observability).

    Recorded into the manifest so a later review can see *why* each stage
    ran or was skipped — the "cache hit 不会错误复用陈旧结果" guarantee is
    auditable from the trace alone.
    """

    entries: list[StageDecisionResult] = field(default_factory=list)

    def record(self, result: StageDecisionResult) -> None:
        self.entries.append(result)

    def to_list(self) -> list[Dict[str, Any]]:
        return [e.to_dict() for e in self.entries]

    @property
    def reused_stages(self) -> list[str]:
        return [e.stage for e in self.entries if e.decision == StageDecision.REUSE_ARTIFACT]

    @property
    def run_stages(self) -> list[str]:
        return [e.stage for e in self.entries if e.decision == StageDecision.RUN]


__all__ = [
    "StageDecision",
    "StageDecisionInput",
    "StageDecisionResult",
    "StageDecisionTrace",
    "CacheVersionBundle",
    "BARRIER_STAGES",
    "LLM_DEPENDENT_STAGES",
    "DETERMINISTIC_STAGES",
    "version_bundles_match",
    "compute_stage_decision",
    "compute_versioned_cache_key",
]
