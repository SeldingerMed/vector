"""Executor attestation contract: minting needs the secret, forgery fails."""

from __future__ import annotations

import pytest

from or_audit.eval.attestation import attest, verify


def test_attest_verify_roundtrip() -> None:
    stamp = attest(
        executor_id="machine0-eu-1",
        artifact_head="a" * 64,
        backend="real",
        world_pin="b" * 40,
        nonce="nonce-1234",
        secret=b"operator-secret",
    )
    assert verify(stamp, secret=b"operator-secret") is True
    assert verify(stamp, secret=b"wrong-secret") is False
    assert verify(stamp, secret=b"") is False


def test_tampered_stamp_fails_verification() -> None:
    stamp = attest(
        executor_id="machine0-eu-1",
        artifact_head="a" * 64,
        backend="real",
        world_pin="b" * 40,
        nonce="nonce-1234",
        secret=b"operator-secret",
    )
    forged = stamp.model_copy(update={"backend": "synthetic-stub"})
    assert verify(forged, secret=b"operator-secret") is False
    upgraded = stamp.model_copy(update={"artifact_head": "c" * 64})
    assert verify(upgraded, secret=b"operator-secret") is False


def test_minting_needs_a_secret() -> None:
    with pytest.raises(ValueError, match="non-empty operator secret"):
        attest(
            executor_id="e",
            artifact_head="a" * 64,
            backend="real",
            world_pin="b" * 40,
            nonce="nonce-1234",
            secret=b"",
        )
