from __future__ import annotations

from pathlib import Path


def pytest_ignore_collect(collection_path, config):
    path = Path(str(collection_path))
    if "tests/fixtures/extraction" in str(path):
        return True
    return False
