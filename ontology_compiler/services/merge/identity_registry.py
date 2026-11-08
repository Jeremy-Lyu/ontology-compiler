"""Identity registry for governance.

Provides DID-like identity management for publish/rollback authorization.
Implements IdentityRegistryPort for dependency injection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from core.errors import DataContractError
from core.contracts.ports import IdentityRegistryPort


@dataclass(frozen=True)
class IdentityRecord:
    subject_id: str
    did: str
    roles: list[str]
    trust_level: str
    scopes: list[str]
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IdentityRegistry(IdentityRegistryPort):
    """Local DID-like identity registry for governance checks.

    Implements IdentityRegistryPort for dependency injection.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path("config/identity_registry.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Any] | None = None
        self._cache_mtime: float | None = None

    def invalidate_cache(self) -> None:
        """Force next load() to re-read from disk."""
        self._cache = None
        self._cache_mtime = None

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            payload = {
                "schema_version": "v1.0",
                "identities": [
                    IdentityRecord(
                        subject_id="system-architect",
                        did="did:ontology:system-architect",
                        roles=["auto-domain-architect", "verified-user", "colleague"],
                        trust_level="high",
                        scopes=["l1", "l2", "publish", "rollback"],
                        status="active",
                    ).to_dict(),
                    IdentityRecord(
                        subject_id="external-importer",
                        did="did:ontology:external-importer",
                        roles=["external_importer"],
                        trust_level="restricted",
                        scopes=["external_knowledge"],
                        status="active",
                    ).to_dict(),
                ],
            }
            self.save(payload)
            self._cache = payload
            self._cache_mtime = self.path.stat().st_mtime if self.path.exists() else None
            return payload
        current_mtime = self.path.stat().st_mtime
        if self._cache is not None and self._cache_mtime == current_mtime:
            return self._cache
        with open(self.path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise DataContractError(f"invalid identity registry payload: {self.path}")
        payload.setdefault("schema_version", "v1.0")
        payload.setdefault("identities", [])
        self._cache = payload
        self._cache_mtime = current_mtime
        return payload

    def save(self, payload: dict[str, Any]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def get_identity(self, subject_id: str) -> dict[str, Any] | None:
        """Get identity record by subject_id (IdentityRegistryPort interface)."""
        payload = self.load()
        for item in payload.get("identities") or []:
            if isinstance(item, dict) and item.get("subject_id") == subject_id:
                return item
        return None

    def get_by_subject(self, subject_id: str) -> dict[str, Any] | None:
        """Alias for get_identity for backward compatibility."""
        return self.get_identity(subject_id)

    def check_scope(self, subject_id: str, scope: str) -> bool:
        """Check if identity has the specified scope (IdentityRegistryPort interface)."""
        identity = self.get_identity(subject_id)
        if identity is None:
            return False
        if str(identity.get("status") or "").lower() != "active":
            return False
        scopes = {str(s) for s in identity.get("scopes") or []}
        return scope in scopes

    def require_role(self, *, subject_id: str, allowed_roles: list[str]) -> dict[str, Any]:
        identity = self.get_by_subject(subject_id)
        if identity is None:
            raise DataContractError(f"identity not found: {subject_id}")
        if str(identity.get("status") or "").lower() != "active":
            raise DataContractError(f"identity inactive: {subject_id}")

        roles = {str(role) for role in identity.get("roles") or []}
        if not roles.intersection(set(allowed_roles)):
            raise DataContractError(
                f"identity {subject_id} missing required role; allowed={allowed_roles}, actual={sorted(roles)}"
            )
        return identity

    def require_scope(self, *, subject_id: str, required_scope: str) -> dict[str, Any]:
        identity = self.get_by_subject(subject_id)
        if identity is None:
            raise DataContractError(f"identity not found: {subject_id}")
        if str(identity.get("status") or "").lower() != "active":
            raise DataContractError(f"identity inactive: {subject_id}")

        scopes = {str(scope) for scope in identity.get("scopes") or []}
        if required_scope not in scopes:
            raise DataContractError(
                f"identity {subject_id} missing required scope: {required_scope}; actual={sorted(scopes)}"
            )
        return identity

    def build_publish_signature(self, *, subject_id: str, payload_sha256: str) -> str:
        identity = self.get_by_subject(subject_id)
        if identity is None:
            raise DataContractError(f"identity not found: {subject_id}")
        did = str(identity.get("did") or "").strip()
        if not did.startswith("did:"):
            raise DataContractError(f"identity {subject_id} invalid did")
        digest = str(payload_sha256 or "").strip().lower()
        if len(digest) != 64:
            raise DataContractError("payload_sha256 must be 64-char lowercase hex")
        return f"didsig:{did}:{digest}"

    def verify_publish_signature(
        self,
        *,
        subject_id: str,
        payload_sha256: str,
        signature: str,
        allowed_roles: list[str],
    ) -> dict[str, Any]:
        identity = self.require_role(subject_id=subject_id, allowed_roles=allowed_roles)
        self.require_scope(subject_id=subject_id, required_scope="publish")
        expected = self.build_publish_signature(subject_id=subject_id, payload_sha256=payload_sha256)
        if str(signature or "").strip() != expected:
            raise DataContractError("publish signature verification failed")
        return identity


