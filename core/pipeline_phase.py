"""Pipeline Phase Registry — stage → canonical business phase labels.

[W2 / 2026-06-20]

Defines 5 canonical business phases (PipelinePhase):
    P0_Raw / P1_Structure / P2_Semantics / P2_5_Instance / P3_Governance

Existing pipelines use their own stage naming (preflight / extract / catalog ...).
This module does NOT rename stages, but labels each stage with a phase tag for:
    - run summary grouping by phase
    - displaying "current business phase" in reports
    - providing constraint foundation for future state machine gates

Usage:
    from core.pipeline_phase import PipelinePhase, get_stage_phase_registry

    reg = get_stage_phase_registry()
    reg.phase_of("extract")              # PipelinePhase.P1_STRUCTURE
    reg.phase_of("unknown_stage")        # PipelinePhase.UNKNOWN
    reg.stages_in(PipelinePhase.P3_GOVERNANCE)
    reg.group_stages(["preflight","extract","merge","guard"])
        # {PipelinePhase.P0_RAW: ["preflight"],
        #  PipelinePhase.P1_STRUCTURE: ["extract"],
        #  PipelinePhase.P3_GOVERNANCE: ["merge","guard"]}
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from src.domain.models.enums import PipelinePhase


_DEFAULT_MAPPING_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "pipeline_phase_mapping.yaml"
)


# Standard ordering (for display only)
_DEFAULT_PHASE_ORDER: List[PipelinePhase] = [
    PipelinePhase.P0_RAW,
    PipelinePhase.P1_STRUCTURE,
    PipelinePhase.P2_SEMANTICS,
    PipelinePhase.P2_5_INSTANCE,
    PipelinePhase.P3_GOVERNANCE,
    PipelinePhase.UNKNOWN,
]


class StagePhaseRegistry:
    """Loads config/pipeline_phase_mapping.yaml and provides stage → phase lookup."""

    def __init__(self, mapping_path: Optional[Path] = None) -> None:
        self.mapping_path = Path(mapping_path) if mapping_path else _DEFAULT_MAPPING_PATH
        self._stage_to_phase: Dict[str, PipelinePhase] = {}
        self._phase_descriptions: Dict[PipelinePhase, str] = {}
        self._phase_order: List[PipelinePhase] = list(_DEFAULT_PHASE_ORDER)
        self._load()

    def _load(self) -> None:
        if not self.mapping_path.exists():
            return
        with self.mapping_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        raw_map = data.get("stage_to_phase", {}) or {}
        for stage, phase_value in raw_map.items():
            try:
                self._stage_to_phase[stage] = PipelinePhase(phase_value)
            except ValueError:
                self._stage_to_phase[stage] = PipelinePhase.UNKNOWN

        order = data.get("phase_order")
        if isinstance(order, list) and order:
            resolved = []
            for v in order:
                try:
                    resolved.append(PipelinePhase(v))
                except ValueError:
                    continue
            if PipelinePhase.UNKNOWN not in resolved:
                resolved.append(PipelinePhase.UNKNOWN)
            self._phase_order = resolved

        descs = data.get("phase_descriptions", {}) or {}
        for k, v in descs.items():
            try:
                self._phase_descriptions[PipelinePhase(k)] = str(v)
            except ValueError:
                continue

    def phase_of(self, stage: str) -> PipelinePhase:
        """Return the PipelinePhase for a stage; returns UNKNOWN if not configured."""
        return self._stage_to_phase.get(stage, PipelinePhase.UNKNOWN)

    def stages_in(self, phase: PipelinePhase) -> List[str]:
        """List all stages belonging to the specified phase."""
        return [s for s, p in self._stage_to_phase.items() if p == phase]

    def group_stages(self, stages: Iterable[str]) -> Dict[PipelinePhase, List[str]]:
        """Group a batch of stages by phase (preserves input order)."""
        grouped: Dict[PipelinePhase, List[str]] = {}
        for s in stages:
            phase = self.phase_of(s)
            grouped.setdefault(phase, []).append(s)
        ordered: Dict[PipelinePhase, List[str]] = {}
        for phase in self._phase_order:
            if phase in grouped:
                ordered[phase] = grouped[phase]
        return ordered

    def describe(self, phase: PipelinePhase) -> str:
        """Get human-readable description of phase."""
        return self._phase_descriptions.get(phase, phase.value)

    @property
    def phase_order(self) -> List[PipelinePhase]:
        return list(self._phase_order)

    @property
    def known_stages(self) -> List[str]:
        return list(self._stage_to_phase.keys())


@lru_cache(maxsize=1)
def get_stage_phase_registry() -> StagePhaseRegistry:
    """Global singleton, cached within process."""
    return StagePhaseRegistry()
