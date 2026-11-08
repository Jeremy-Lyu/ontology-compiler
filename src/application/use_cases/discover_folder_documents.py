"""Application use-case for discovering readable documents in a folder."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.application.ports.document_ports import DocumentReaderPort


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoverFolderDocumentsRequest:
    """Request DTO for discovering readable documents in one folder tree."""

    folder_path: str
    max_files: int | None = None
    include_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None


class DiscoverFolderDocumentsUseCase:
    """Discover readable files in a folder using `DocumentReaderPort`."""

    def __init__(self, document_reader: DocumentReaderPort):
        self._document_reader = document_reader

    def execute(self, request: DiscoverFolderDocumentsRequest) -> list[str]:
        folder_path = request.folder_path
        if not isinstance(folder_path, str) or not folder_path.strip():
            raise ValueError("folder_path must be a non-empty string")

        root = Path(folder_path)
        if not root.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder_path}")
        if not root.is_dir():
            raise ValueError(f"Not a directory: {folder_path}")
        if request.max_files is not None and request.max_files <= 0:
            raise ValueError("max_files must be > 0 when provided")

        include_patterns = request.include_patterns or ["**/*"]
        exclude_patterns = request.exclude_patterns or []

        candidates: list[Path] = []
        seen: set[Path] = set()
        for pattern in include_patterns:
            path_iter: Iterable[Path]
            if request.max_files is None:
                path_iter = sorted(root.glob(pattern))
            else:
                path_iter = root.glob(pattern)
            for path in path_iter:
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                if any(path.match(exclude) for exclude in exclude_patterns):
                    continue
                if self._document_reader.can_read(str(path)):
                    candidates.append(path)

        if request.max_files is not None:
            sized_candidates: list[tuple[int, str, Path]] = []
            for path in candidates:
                try:
                    size = int(path.stat().st_size)
                except OSError as exc:
                    logger.warning(
                        "Failed to stat file for size-based ranking; treating as size=0: %s: %s",
                        path,
                        exc,
                    )
                    size = 0
                sized_candidates.append((size, str(path), path))

            sized_candidates.sort(key=lambda item: (-item[0], item[1]))
            return [str(item[2]) for item in sized_candidates[: request.max_files]]

        return [str(p) for p in candidates]

