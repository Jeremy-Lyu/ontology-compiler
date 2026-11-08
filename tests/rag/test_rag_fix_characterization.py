"""Characterization tests for the 2026-09-01 RAG fixes.

Locks the behavior of two regression fixes so they cannot silently drift back:

1. asyncio nesting in ``RAGAnswerer._call_llm_with_client``:
   ``_answer_async_impl`` runs under ``asyncio.run`` and then calls the *sync*
   ``_call_llm`` -> ``_call_llm_with_client``. The old code unconditionally
   called ``asyncio.run(...)`` inside the already-running loop, which raised
   ``RuntimeError: asyncio.run() cannot be called from a running event loop``.
   The fix runs the coroutine in a dedicated thread when a loop is already
   running, while keeping the synchronous signature for loop-less callers.

2. governed-packet rag_dir resolution (``resolve_governed_rag_dir``):
   ``scripts/rag_cli.py cmd_query`` hardcoded the legacy rag dir, which now
   raises ``DataContractError`` under the Phase B hard constraint. The fix
   relies on ``resolve_governed_rag_dir`` to route every consumer to the
   governed packet. These tests lock that routing contract (hermetic, temp
   files, no real outputs).

No LLM / network access; the LLM client is a fake async callable.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.application.rag.governed_packets import (
    build_file_manifest,
    file_sha256,
    require_governed_packet,
    resolve_governed_rag_dir,
)
from src.application.rag.rag_answerer import LLMAnswerer
from src.domain.errors import DataContractError


class _FakeAsyncClient:
    """Minimal UnifiedLLMClient-shaped fake with an async call()."""

    provider = "fake"

    def __init__(self) -> None:
        self.calls = 0
        self.last_prompt = ""

    async def call(self, *, prompt: str, temperature: float, max_tokens: int) -> str:
        self.calls += 1
        self.last_prompt = prompt
        return "fake answer"


def _make_answerer(client) -> LLMAnswerer:
    """Build an LLMAnswerer instance without touching __init__/governed packet.

    `_call_llm_with_client` only needs the prompt-building helpers and the
    llm_client attribute, all of which are class-level / stateless.
    """
    answerer = object.__new__(LLMAnswerer)
    answerer.llm_client = client
    answerer.llm_provider = None
    answerer.dify_config = None
    # Bypass __init__ -> supply the prompt-config attrs __init__ would load
    # (non-empty so _classify_question / _build_llm_system_prompt are exercised).
    answerer._QUESTION_PATTERNS = {"GENERAL": []}
    answerer._SYSTEM_PROMPTS = {"GENERAL": "回答 {concept}\n来源: {sources}"}
    return answerer


class CallLlmWithClientAsyncNestingCharacterization(unittest.TestCase):
    def test_loopless_sync_call_returns_answer(self):
        client = _FakeAsyncClient()
        answerer = _make_answerer(client)
        out = answerer._call_llm_with_client("what is VTDS", "context")
        self.assertEqual(out, "fake answer")
        self.assertEqual(client.calls, 1)

    def test_inside_running_loop_does_not_raise_runtime_error(self):
        # Regression 2026-09-01: _answer_async_impl runs under asyncio.run and
        # then calls the sync _call_llm_with_client. Old code raised
        # RuntimeError("asyncio.run() cannot be called from a running event loop").
        client = _FakeAsyncClient()
        answerer = _make_answerer(client)

        async def _run() -> str:
            return answerer._call_llm_with_client("what is VTDS", "context")

        out = asyncio.run(_run())
        self.assertEqual(out, "fake answer")
        self.assertEqual(client.calls, 1)

    def test_failure_is_wrapped_in_runtime_error(self):
        class _Boom:
            provider = "boom"

            async def call(self, **kwargs):
                raise ConnectionError("boom")

        answerer = _make_answerer(_Boom())
        with self.assertRaises(RuntimeError) as ctx:
            answerer._call_llm_with_client("q", "ctx")
        self.assertIn("LLM call failed", str(ctx.exception))


def _write_mini_packet(packet_dir: Path) -> Path:
    """Create a minimal but fully valid governed packet in ``packet_dir``."""
    (packet_dir / "vector_index").mkdir(parents=True, exist_ok=True)
    (packet_dir / "chunks.jsonl").write_text('{"chunk_id": "ragc:L1-1", "content": "x"}\n', encoding="utf-8")
    (packet_dir / "inverted_index.json").write_text("{}", encoding="utf-8")
    (packet_dir / "idf.json").write_text("{}", encoding="utf-8")
    (packet_dir / "vector_index" / "metadata.json").write_text("{}", encoding="utf-8")
    (packet_dir / "packet.json").write_text("{}", encoding="utf-8")

    files = build_file_manifest(packet_dir, [
        "chunks.jsonl",
        "inverted_index.json",
        "idf.json",
        "vector_index/metadata.json",
        "packet.json",
    ])
    manifest = {
        "schema_version": "1.0",
        "packet_id": packet_dir.name,
        "built_at": "2026-09-01T00:00:00",
        "master_dir": str(packet_dir.parent),
        "master_hashes": {"nodes": "x", "relationships": "x"},
        "source_code_version": {},
        "files": [{"path": f.path, "size": f.size, "sha256": f.sha256} for f in files],
        "chunk_count": 1,
        "entity_count": 1,
        "bilingual_alias_groups": 0,
        "vector_dimension": 4096,
    }
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2)
    (packet_dir / "manifest.json").write_text(manifest_json, encoding="utf-8")
    (packet_dir / "manifest.sha256").write_text(
        file_sha256(packet_dir / "manifest.json"), encoding="utf-8"
    )
    require_governed_packet(packet_dir)  # self-check the fixture
    return packet_dir


class GovernedRagDirResolutionCharacterization(unittest.TestCase):
    """Locks the routing contract scripts/rag_cli.py now relies on."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self._governed_root = root / "asset_packets" / "governed"
        self._governed_root.mkdir(parents=True, exist_ok=True)
        self._packet_dir = self._write_packet(self._governed_root / "GOV-TEST-1")
        self._pointer = self._governed_root.parent / "latest_governed_packet.json"
        self._pointer.write_text(
            json.dumps({"packet_id": self._packet_dir.name}), encoding="utf-8"
        )
        self._legacy = root / "outputs" / "runs" / "ontology" / "rag"
        self._legacy.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_packet(self, packet_dir: Path) -> Path:
        return _write_mini_packet(packet_dir)

    def test_none_resolves_to_governed_packet_dir(self):
        with mock.patch("src.application.rag.governed_packets.GOVERNED_DIR", self._governed_root), \
             mock.patch("src.application.rag.governed_packets.LATEST_POINTER", self._pointer), \
             mock.patch("src.application.rag.governed_packets.LEGACY_RAG_DIR", self._legacy), \
             mock.patch("src.application.rag.governed_packets.ALLOW_LEGACY", False), \
             mock.patch("src.application.rag.governed_packets.GOVERNED_ONLY", True):
            resolved = resolve_governed_rag_dir(None)
            self.assertEqual(resolved.resolve(), self._packet_dir.resolve())

    def test_legacy_dir_raises_by_default(self):
        with mock.patch("src.application.rag.governed_packets.GOVERNED_DIR", self._governed_root), \
             mock.patch("src.application.rag.governed_packets.LATEST_POINTER", self._pointer), \
             mock.patch("src.application.rag.governed_packets.LEGACY_RAG_DIR", self._legacy), \
             mock.patch("src.application.rag.governed_packets.ALLOW_LEGACY", False), \
             mock.patch("src.application.rag.governed_packets.GOVERNED_ONLY", True):
            with self.assertRaises(DataContractError) as ctx:
                resolve_governed_rag_dir(self._legacy)
            self.assertIn("legacy rag_dir is forbidden", str(ctx.exception))

    def test_legacy_dir_allowed_when_escape_hatch_enabled(self):
        with mock.patch("src.application.rag.governed_packets.GOVERNED_DIR", self._governed_root), \
             mock.patch("src.application.rag.governed_packets.LATEST_POINTER", self._pointer), \
             mock.patch("src.application.rag.governed_packets.LEGACY_RAG_DIR", self._legacy), \
             mock.patch("src.application.rag.governed_packets.ALLOW_LEGACY", True), \
             mock.patch("src.application.rag.governed_packets.GOVERNED_ONLY", True):
            resolved = resolve_governed_rag_dir(self._legacy)
            self.assertEqual(resolved.resolve(), self._legacy.resolve())

    def test_explicit_governed_packet_dir_is_used(self):
        with mock.patch("src.application.rag.governed_packets.GOVERNED_DIR", self._governed_root), \
             mock.patch("src.application.rag.governed_packets.LATEST_POINTER", self._pointer), \
             mock.patch("src.application.rag.governed_packets.LEGACY_RAG_DIR", self._legacy), \
             mock.patch("src.application.rag.governed_packets.ALLOW_LEGACY", False), \
             mock.patch("src.application.rag.governed_packets.GOVERNED_ONLY", True):
            resolved = resolve_governed_rag_dir(self._packet_dir)
            self.assertEqual(resolved.resolve(), self._packet_dir.resolve())


if __name__ == "__main__":
    unittest.main()
