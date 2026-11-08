from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional
import json
import re


@dataclass
class ArtifactBundle:
    run_id: str
    stage: str
    artifact_root: str
    audit_root: str
    regression_root: str
    items: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "artifact_root": self.artifact_root,
            "audit_root": self.audit_root,
            "regression_root": self.regression_root,
            "items": self.items,
            "metadata": self.metadata,
            "created_at": datetime.now().isoformat(),
        }


_BUNDLE_FILENAME_RE = re.compile(r"^(?P<run_id>.+)-(?P<stage>[a-z_]+)-bundle\.json$")


class ArtifactRegistry:
    def __init__(self, artifact_root: str):
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.audit_root = self.artifact_root / "audit"
        self.regression_root = self.artifact_root / "regression"
        self.audit_root.mkdir(parents=True, exist_ok=True)
        self.regression_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ write
    def register_bundle(self, bundle: ArtifactBundle) -> str:
        path = self.artifact_root / f"{bundle.run_id}-{bundle.stage}-bundle.json"
        path.write_text(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def copy_audit(self, source_path: str, run_id: str, name: str) -> str:
        src = Path(source_path)
        dest = self.audit_root / run_id
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / name
        target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return str(target)

    # --------------------------------------------------------------- query API
    def load_bundle(self, run_id: str, stage: str) -> Optional[dict[str, Any]]:
        """Load a previously registered bundle JSON by (run_id, stage)."""
        path = self.artifact_root / f"{run_id}-{stage}-bundle.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @staticmethod
    def load_bundle_from_path(bundle_path: str | Path) -> Optional[dict[str, Any]]:
        """Load a bundle JSON from an arbitrary absolute path (cross-run lookup)."""
        p = Path(bundle_path)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def iter_bundle_paths(
        self,
        search_roots: Iterable[str | Path],
        *,
        stage: Optional[str] = None,
    ) -> list[Path]:
        """Scan arbitrary artifact roots for bundle files.

        This is the cross-run lookup primitive: given a list of sibling
        artifact directories (sibling runs under ``output_dir/runs/*/artifacts``)
        return every bundle path that matches *stage*.
        """
        results: list[Path] = []
        suffix = f"-{stage}-bundle.json" if stage else "-bundle.json"
        for root in search_roots:
            root_p = Path(root)
            if not root_p.is_dir():
                continue
            for child in root_p.iterdir():
                if not child.is_file():
                    continue
                if not child.name.endswith(suffix):
                    continue
                results.append(child)
        return results

    @staticmethod
    def parse_bundle_name(filename: str) -> Optional[tuple[str, str]]:
        """Return ``(run_id, stage)`` for a bundle filename, or None if invalid."""
        m = _BUNDLE_FILENAME_RE.match(filename)
        if not m:
            return None
        return m.group("run_id"), m.group("stage")
