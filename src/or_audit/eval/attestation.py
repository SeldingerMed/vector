"""Executor attestation for hosted evaluation runs (Phase B5, OSS side).

Local job heads are unkeyed digests: tamper-evident, re-stampable. Cross-lab
trust needs the executor to stamp what it observed with a key the submitter
does not hold. This module defines the stamp and its verification; minting
happens in the hosted executor (private cloud tree), which holds the
operator secret. Nothing here mints without a secret, and nothing verifies
without the same one.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


class ExecutorAttestation(BaseModel):
    """HMAC stamp over observed execution provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    executor_id: str = Field(min_length=1, max_length=128)
    artifact_head: str = Field(min_length=64, max_length=64, pattern="^[0-9a-f]{64}$")
    backend: str = Field(min_length=1, max_length=32)
    world_pin: str = Field(min_length=1, max_length=128)
    nonce: str = Field(min_length=8, max_length=128)
    mac: str = Field(min_length=64, max_length=64, pattern="^[0-9a-f]{64}$")

    def payload(self) -> dict[str, Any]:
        return {
            "executor_id": self.executor_id,
            "artifact_head": self.artifact_head,
            "backend": self.backend,
            "world_pin": self.world_pin,
            "nonce": self.nonce,
        }


def attest(
    *,
    executor_id: str,
    artifact_head: str,
    backend: str,
    world_pin: str,
    nonce: str,
    secret: bytes,
) -> ExecutorAttestation:
    """Mint an attestation. Called by the hosted executor, never by evaluated code."""
    if not secret:
        raise ValueError("attestation needs a non-empty operator secret")
    stamp = ExecutorAttestation(
        executor_id=executor_id,
        artifact_head=artifact_head,
        backend=backend,
        world_pin=world_pin,
        nonce=nonce,
        mac="0" * 64,
    )
    mac = hmac.new(secret, _canonical(stamp.payload()), hashlib.sha256).hexdigest()
    return stamp.model_copy(update={"mac": mac})


def verify(attestation: ExecutorAttestation, *, secret: bytes) -> bool:
    """Check the stamp against the operator secret (constant-time)."""
    if not secret:
        return False
    expected = hmac.new(secret, _canonical(attestation.payload()), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, attestation.mac)
