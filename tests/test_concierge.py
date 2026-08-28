"""Tests for the hosted concierge: intake, assess, select, adapt.

The properties under test are refusals, not happy paths: a pickle upload never
loads, an unsafe sandbox has no representation, an SSRF target is refused even
when allowlisted, a draft capability cannot be scored, a plan emits no
composite, and an adaptation can never touch a verifier.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from or_audit.commands.concierge import register
from or_audit.concierge.adapt import (
    AdaptBudget,
    AdaptCandidate,
    AdaptObservation,
    FrozenPackage,
    ScenarioSpace,
    assert_frozen_before_scoring,
    assert_verifier_untouched,
    freeze_adapted_package,
    search_scenarios,
)
from or_audit.concierge.assess import (
    AssessProposal,
    ConfirmedCapability,
    ModelProbeReport,
    assert_confirmed,
    assess_model,
    confirm,
)
from or_audit.concierge.intake import (
    EndpointAllowlist,
    EndpointDescriptor,
    IntakeResult,
    ProbeRequest,
    ProbeResponse,
    SandboxPolicy,
    SigningKey,
    TenantKeyring,
    UploadManifest,
    intake_endpoint,
    intake_upload,
    manifest_signature,
    runtime_descriptor,
    sign_manifest,
    signed_payload,
    verify_manifest_signature,
)
from or_audit.concierge.select import (
    NO_COMPOSITE_FOOTER,
    EvalBudget,
    narrate_plan,
    narrate_results,
    select_eval_plan,
)
from or_audit.errors import TaskContractError
from or_audit.eval.contracts import (
    CapabilitySpec,
    InteractionMode,
    RuntimeDescriptor,
    RuntimeKind,
)
from or_audit.eval.integrity import tree_digest
from or_audit.eval.job import JobResult
from or_audit.eval.loader import load_agent, load_task
from or_audit.eval.provenance import (
    PROVENANCE_FILENAME,
    adaptation_tells,
    assert_public_leaderboard_eligible,
    assert_scoreable_package,
    content_digest,
    read_provenance,
    verifier_identity,
)
from or_audit.eval.runner import run_job

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_CATALOG = REPO_ROOT / "docs/examples/tasks"
VIDEO_TASK = TASK_CATALOG / "video-nextstep"
VIDEO_AGENT = REPO_ROOT / "docs/examples/agents/example-video-predictor"
COUNTERFACTUAL_TASK = TASK_CATALOG / "counterfactual-recovery"
COUNTERFACTUAL_AGENT = REPO_ROOT / "docs/examples/agents/example-counterfactual-world-model"
INTAKE_SOURCE = REPO_ROOT / "src/or_audit/concierge/intake.py"

SANDBOX = SandboxPolicy(cpu_quota=2.0, memory_bytes=1 << 30, disk_bytes=1 << 30)

#: Control-plane keys. Two tenants, so cross-tenant misuse is testable.
ACME_KEY = SigningKey(key_id="acme-2026-01", tenant="tenant-acme", secret=b"A" * 32)
RIVAL_KEY = SigningKey(key_id="rival-2026-01", tenant="tenant-rival", secret=b"R" * 32)


def _keyring() -> TenantKeyring:
    return TenantKeyring([ACME_KEY, RIVAL_KEY])


def _artifact(tmp_path: Path, name: str, body: bytes = b"tensor-bytes") -> Path:
    path = tmp_path / name
    path.write_bytes(body)
    return path


def _manifest(
    path: Path,
    *,
    upload_format: str = "safetensors",
    key: SigningKey | None = ACME_KEY,
    **overrides: object,
) -> UploadManifest:
    """A manifest for ``path``, signed by ``key`` unless ``key=None``."""
    body = path.read_bytes()
    fields: dict[str, object] = {
        "format": upload_format,
        "digest": hashlib.sha256(body).hexdigest(),
        "entrypoint": path.name,
        "size_bytes": len(body),
        "declared_by": key.tenant if key is not None else "tenant-acme",
    }
    fields.update(overrides)
    manifest = UploadManifest.model_validate(fields)
    return manifest if key is None else sign_manifest(manifest, key=key)


def _passing_intake(tmp_path: Path) -> IntakeResult:
    artifact = _artifact(tmp_path, "model.safetensors")
    result = intake_upload(_manifest(artifact), artifact, sandbox=SANDBOX, keyring=_keyring())
    assert result.passed, result.refusals
    return result


def _probe(response: ProbeResponse) -> Callable[[ProbeRequest], ProbeResponse]:
    def probe(request: ProbeRequest) -> ProbeResponse:
        assert request.payload["synthetic"] is True
        return response

    return probe


def _video_probe_report() -> ModelProbeReport:
    return ModelProbeReport(
        interface="video-predict",
        interaction_modes=(InteractionMode.SINGLE_TURN,),
        observations=("video-clip", "dsa-sequence"),
        outputs=("next-step", "structured-prediction", "release-audit"),
        features=("reasoning", "abstention"),
        modalities=("video-laparoscopic",),
        action_space="none: single-turn prediction",
        evidence=("probed /schema: accepts video-clip", "probed /predict: returns next-step"),
    )


def _confirmed(tmp_path: Path) -> ConfirmedCapability:
    proposal = assess_model(_passing_intake(tmp_path), probe=lambda _: _video_probe_report())
    return confirm(proposal, confirmed_by="tenant-reviewer@acme")


# --------------------------------------------------------------------------
# Capability 0: intake
# --------------------------------------------------------------------------


def test_intake_never_imports_a_pickle_family_module() -> None:
    tree = ast.parse(INTAKE_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {
        "pickle",
        "pickletools",
        "cloudpickle",
        "dill",
        "joblib",
        "torch",
        "marshal",
        "shelve",
        "importlib",
    }
    assert not imported & forbidden, f"intake.py must not import {sorted(imported & forbidden)}"


@pytest.mark.parametrize(
    ("upload_format", "filename"),
    [
        ("pickle", "model.pkl"),
        ("safetensors", "model.ckpt"),
        ("pytorch", "pytorch_model.bin"),
    ],
)
def test_pickle_family_upload_is_refused_because_loading_executes_code(
    tmp_path: Path, upload_format: str, filename: str
) -> None:
    artifact = _artifact(tmp_path, filename)
    result = intake_upload(
        _manifest(artifact, upload_format=upload_format),
        artifact,
        sandbox=SANDBOX,
        keyring=_keyring(),
    )
    assert not result.passed
    joined = " ".join(result.refusals)
    assert "deserialization executes code" in joined
    with pytest.raises(TaskContractError, match="failed intake"):
        runtime_descriptor(result)


def test_unlisted_format_is_refused() -> None:
    artifact_free = UploadManifest(
        format="tensorflow-savedmodel",
        digest="0" * 64,
        entrypoint="saved_model.pb",
        size_bytes=10,
        declared_by="tenant-acme",
    )
    result = intake_upload(artifact_free, Path("missing"), sandbox=SANDBOX, keyring=_keyring())
    assert not result.passed
    assert any("only [safetensors, onnx, gguf, container]" in item for item in result.refusals)


def test_safetensors_digest_mismatch_is_refused(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "model.safetensors", b"declared-bytes")
    manifest = _manifest(artifact, digest="a" * 64)
    result = intake_upload(manifest, artifact, sandbox=SANDBOX, keyring=_keyring())
    assert not result.passed
    assert any("digest mismatch" in item for item in result.refusals)
    assert result.digests["artifact"] == hashlib.sha256(b"declared-bytes").hexdigest()


def test_size_mismatch_is_refused(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "model.safetensors")
    result = intake_upload(
        _manifest(artifact, size_bytes=999_999), artifact, sandbox=SANDBOX, keyring=_keyring()
    )
    assert not result.passed
    assert any("manifest declares 999999" in item for item in result.refusals)


def test_custom_format_requires_a_digest_pinned_tenant_image(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "bundle.tar")
    bare = intake_upload(
        _manifest(artifact, upload_format="container"),
        artifact,
        sandbox=SANDBOX,
        keyring=_keyring(),
    )
    assert not bare.passed
    assert any("digest-pinned" in item for item in bare.refusals)

    pinned = intake_upload(
        _manifest(
            artifact,
            upload_format="container",
            image="registry.example.com/acme/loader",
            image_digest="b" * 64,
        ),
        artifact,
        sandbox=SANDBOX,
        keyring=_keyring(),
    )
    assert pinned.passed, pinned.refusals
    descriptor = runtime_descriptor(pinned)
    assert descriptor.kind is RuntimeKind.CONTAINER
    assert descriptor.image_digest == "b" * 64


def test_image_reference_without_a_digest_is_not_a_pin(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "bundle.tar")
    with pytest.raises(TaskContractError, match="not a pin"):
        _manifest(artifact, upload_format="container", image="registry.example.com/acme/loader")


def test_intake_refuses_a_sandbox_that_does_not_scan(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "model.safetensors")
    result = intake_upload(
        _manifest(artifact),
        artifact,
        sandbox=SandboxPolicy(artifact_scan=False),
        keyring=_keyring(),
    )
    assert not result.passed
    assert any("not a gate" in item for item in result.refusals)


def test_unsafe_sandbox_policy_cannot_be_constructed() -> None:
    with pytest.raises(TaskContractError, match="cannot allow egress"):
        SandboxPolicy(egress=True)
    with pytest.raises(TaskContractError, match="trust_remote_code"):
        SandboxPolicy(trust_remote_code=True)


# ---- manifest signing: trust never comes from the manifest ----------------


def test_unsigned_manifest_is_refused(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "model.safetensors")
    manifest = _manifest(artifact, key=None)
    assert manifest.signature == ""
    result = intake_upload(manifest, artifact, sandbox=SANDBOX, keyring=_keyring())
    assert not result.passed
    assert any("refusing unsigned upload manifest" in item for item in result.refusals)
    assert result.signer == "unsigned"


def test_a_forged_signature_is_not_recorded_as_an_unsigned_manifest(tmp_path: Path) -> None:
    """An operator reading the record must see which key was asserted and failed.

    Collapsing "signature did not verify" into "unsigned" would make a forgery
    look like an omission, which is the one distinction an incident review needs.
    """
    artifact = _artifact(tmp_path, "model.safetensors")
    signed = _manifest(artifact)
    forged = signed.model_copy(update={"entrypoint": "evil:load"})
    result = intake_upload(forged, artifact, sandbox=SANDBOX, keyring=_keyring())
    assert not result.passed
    assert result.signer_key_id == ""
    assert result.claimed_key_id == ACME_KEY.key_id
    assert result.signer == f"unverified (claimed key {ACME_KEY.key_id})"
    assert "unverified" in result.describe()

    unsigned = intake_upload(
        _manifest(artifact, key=None), artifact, sandbox=SANDBOX, keyring=_keyring()
    )
    assert unsigned.claimed_key_id == ""
    assert unsigned.signer == "unsigned"


def test_key_id_and_signature_must_be_declared_together(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "model.safetensors")
    with pytest.raises(TaskContractError, match="authenticates nothing"):
        _manifest(artifact, key=None, key_id="acme-2026-01")
    with pytest.raises(TaskContractError, match="names no verifier"):
        _manifest(artifact, key=None, signature="f" * 64)


def test_unknown_key_id_is_refused(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "model.safetensors")
    stranger = SigningKey(key_id="stranger-2026-01", tenant="tenant-acme", secret=b"S" * 32)
    manifest = _manifest(artifact, key=stranger)
    result = intake_upload(manifest, artifact, sandbox=SANDBOX, keyring=_keyring())
    assert not result.passed
    assert any("unknown key_id 'stranger-2026-01'" in item for item in result.refusals)


def test_key_registered_to_another_tenant_is_refused(tmp_path: Path) -> None:
    """tenant-rival signs correctly, but claims to be tenant-acme."""
    artifact = _artifact(tmp_path, "model.safetensors")
    manifest = _manifest(artifact, key=RIVAL_KEY, declared_by="tenant-acme")
    # The signature itself is valid: only the tenant binding is wrong.
    assert manifest.signature == manifest_signature(manifest, secret=RIVAL_KEY.secret)
    result = intake_upload(manifest, artifact, sandbox=SANDBOX, keyring=_keyring())
    assert not result.passed
    assert any(
        "registered to tenant 'tenant-rival' but the manifest declares 'tenant-acme'" in item
        for item in result.refusals
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("digest", "d" * 64),
        ("format", "onnx"),
        ("entrypoint", "other-model.safetensors"),
        ("size_bytes", 4096),
        ("declared_by", "tenant-rival"),
        ("image_digest", "e" * 64),
    ],
)
def test_tampering_with_any_signed_field_breaks_the_signature(
    tmp_path: Path, field: str, value: object
) -> None:
    artifact = _artifact(tmp_path, "model.safetensors")
    signed = _manifest(
        artifact,
        image="registry.example.com/acme/loader",
        image_digest="b" * 64,
    )
    tampered = signed.model_copy(update={field: value})
    assert signed_payload(tampered) != signed_payload(signed)
    assert tampered.signature == signed.signature

    signer, refusal = verify_manifest_signature(tampered, keyring=_keyring())
    assert signer is None
    assert refusal
    result = intake_upload(tampered, artifact, sandbox=SANDBOX, keyring=_keyring())
    assert not result.passed


def test_a_valid_signature_over_another_payload_does_not_transfer(tmp_path: Path) -> None:
    """A genuine signature lifted onto a different payload does not verify.

    Two ways to try it, two refusals: keep the victim's ``declared_by`` and the
    tenant binding catches it; keep the signer's tenant and the MAC catches it.
    """
    victim = _artifact(tmp_path, "model.safetensors")
    other = _artifact(tmp_path, "rival.safetensors", b"a-different-model")
    rival_signed = _manifest(other, key=RIVAL_KEY)
    stolen = {"key_id": rival_signed.key_id, "signature": rival_signed.signature}

    as_acme = _manifest(victim, key=None).model_copy(update=stolen)
    refused_binding = intake_upload(as_acme, victim, sandbox=SANDBOX, keyring=_keyring())
    assert not refused_binding.passed
    assert any("cannot assert an upload on" in item for item in refused_binding.refusals)

    as_rival = _manifest(victim, key=None, declared_by="tenant-rival").model_copy(update=stolen)
    refused_mac = intake_upload(as_rival, victim, sandbox=SANDBOX, keyring=_keyring())
    assert not refused_mac.passed
    assert any("signature does not verify" in item for item in refused_mac.refusals)


def test_verified_signer_is_recorded_and_bound_into_the_identity(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "model.safetensors")
    acme = intake_upload(_manifest(artifact), artifact, sandbox=SANDBOX, keyring=_keyring())
    assert acme.passed, acme.refusals
    assert (acme.signer_tenant, acme.signer_key_id) == (ACME_KEY.tenant, ACME_KEY.key_id)
    assert acme.signer == "tenant-acme#acme-2026-01"
    assert "signature verified" in acme.describe()

    # The same bytes asserted by a different tenant is a different execution.
    rival = intake_upload(
        _manifest(artifact, key=RIVAL_KEY), artifact, sandbox=SANDBOX, keyring=_keyring()
    )
    assert rival.passed, rival.refusals
    assert rival.identity != acme.identity
    assert runtime_descriptor(rival).identity != runtime_descriptor(acme).identity


def test_keyring_refuses_weak_secrets_and_rebinding_a_key_id() -> None:
    keyring = TenantKeyring()
    with pytest.raises(TaskContractError, match="at least 32"):
        keyring.register(key_id="acme-2026-01", tenant="tenant-acme", secret=b"short")
    keyring.register(key_id="acme-2026-01", tenant="tenant-acme", secret=b"A" * 32)
    with pytest.raises(TaskContractError, match="refusing to rebind"):
        keyring.register(key_id="acme-2026-01", tenant="tenant-rival", secret=b"R" * 32)
    assert keyring.key_ids == ("acme-2026-01",)


def test_keyring_never_reveals_a_secret_in_its_repr() -> None:
    keyring = _keyring()
    assert "secrets=<redacted>" in repr(keyring)
    assert ACME_KEY.secret.hex() not in repr(keyring)
    assert "secret=<redacted>" in repr(ACME_KEY)
    assert ACME_KEY.secret.hex() not in repr(ACME_KEY)


def test_keyring_from_records_rejects_non_hex_secrets() -> None:
    with pytest.raises(TaskContractError, match="not hex"):
        TenantKeyring.from_records([{"key_id": "k", "tenant": "t", "secret_hex": "not-hex-at-all"}])


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", "loopback"),
        ("169.254.169.254", "link-local"),
        ("10.0.0.5", "private"),
        ("[::1]", "loopback"),
    ],
)
def test_endpoint_intake_refuses_ssrf_ranges_even_when_allowlisted(
    host: str, expected: str
) -> None:
    bare_host = host.strip("[]")
    descriptor = EndpointDescriptor(
        id="tenant-endpoint",
        url=f"https://{host}/v1",
        model="acme-vlm",
        declared_by="tenant-acme",
    )
    probed: list[ProbeRequest] = []

    def probe(request: ProbeRequest) -> ProbeResponse:
        probed.append(request)
        return ProbeResponse(status_code=200, final_url=request.url)

    result = intake_endpoint(
        descriptor,
        # Allowlisting the target on purpose: the range check must dominate.
        allowlist=EndpointAllowlist(hosts=(bare_host,)),
        probe=probe,
    )
    assert not result.passed
    assert any(expected in item for item in result.refusals)
    assert probed == [], "a refused host must never be probed"


def test_endpoint_intake_refuses_an_off_allowlist_host() -> None:
    descriptor = EndpointDescriptor(
        id="tenant-endpoint",
        url="https://evil.example.net/v1",
        model="acme-vlm",
        declared_by="tenant-acme",
    )
    result = intake_endpoint(
        descriptor,
        allowlist=EndpointAllowlist(hosts=("infer.acme.example",)),
        probe=_probe(ProbeResponse(status_code=200, final_url="https://evil.example.net/v1")),
    )
    assert not result.passed
    assert any("not on the tenant allowlist" in item for item in result.refusals)


def test_endpoint_intake_refuses_an_off_allowlist_redirect() -> None:
    descriptor = EndpointDescriptor(
        id="tenant-endpoint",
        url="https://infer.acme.example/v1",
        model="acme-vlm",
        declared_by="tenant-acme",
    )
    response = ProbeResponse(
        status_code=200,
        final_url="https://exfil.example.net/v1",
        redirects=("https://exfil.example.net/v1",),
    )
    result = intake_endpoint(
        descriptor,
        allowlist=EndpointAllowlist(hosts=("infer.acme.example",)),
        probe=_probe(response),
    )
    assert not result.passed
    assert any("off-allowlist" in item for item in result.refusals)


def test_endpoint_intake_refuses_a_metadata_service_redirect() -> None:
    descriptor = EndpointDescriptor(
        id="tenant-endpoint",
        url="https://infer.acme.example/v1",
        model="acme-vlm",
        declared_by="tenant-acme",
    )
    response = ProbeResponse(
        status_code=200,
        final_url="https://169.254.169.254/latest/meta-data/",
        redirects=("https://169.254.169.254/latest/meta-data/",),
    )
    result = intake_endpoint(
        descriptor,
        allowlist=EndpointAllowlist(hosts=("infer.acme.example",)),
        probe=_probe(response),
    )
    assert not result.passed
    assert any("link-local" in item for item in result.refusals)


def test_endpoint_intake_refuses_plaintext() -> None:
    descriptor = EndpointDescriptor(
        id="tenant-endpoint",
        url="http://infer.acme.example/v1",
        model="acme-vlm",
        declared_by="tenant-acme",
    )
    result = intake_endpoint(
        descriptor,
        allowlist=EndpointAllowlist(hosts=("infer.acme.example",)),
        probe=_probe(ProbeResponse(status_code=200, final_url="http://infer.acme.example/v1")),
    )
    assert not result.passed
    assert any("not\nhttps" in item.replace(" ", "\n") for item in result.refusals)


def test_endpoint_intake_accepts_an_allowlisted_public_host_within_its_probe_budget() -> None:
    descriptor = EndpointDescriptor(
        id="tenant-endpoint",
        url="https://infer.acme.example/v1",
        model="acme-vlm",
        declared_by="tenant-acme",
        probe_budget=2,
    )
    calls: list[int] = []

    def probe(request: ProbeRequest) -> ProbeResponse:
        calls.append(request.index)
        assert request.payload["synthetic"] is True
        return ProbeResponse(
            status_code=200,
            final_url=request.url,
            schema_keys=("model", "input", "output"),
        )

    result = intake_endpoint(
        descriptor,
        allowlist=EndpointAllowlist(hosts=("INFER.acme.example",)),
        probe=probe,
    )
    assert result.passed, result.refusals
    assert len(calls) <= descriptor.probe_budget
    assert result.sandbox_report.tier == "sandbox"
    assert result.sandbox_report.probe_policy is not None
    assert result.sandbox_report.probe_policy.synthetic_inputs_only
    descriptor_out = runtime_descriptor(result)
    assert descriptor_out.kind is RuntimeKind.OPENAI_COMPATIBLE
    assert descriptor_out.base_url == "https://infer.acme.example/v1"


def test_probe_payload_must_be_marked_synthetic() -> None:
    with pytest.raises(TaskContractError, match="marked synthetic"):
        ProbeRequest(url="https://infer.acme.example/v1", index=0, payload={"frames": "real"})


def test_endpoint_intake_records_a_probe_transport_failure() -> None:
    descriptor = EndpointDescriptor(
        id="tenant-endpoint",
        url="https://infer.acme.example/v1",
        model="acme-vlm",
        declared_by="tenant-acme",
    )

    def probe(request: ProbeRequest) -> ProbeResponse:
        raise RuntimeError(f"connection reset on probe {request.index}")

    result = intake_endpoint(
        descriptor,
        allowlist=EndpointAllowlist(hosts=("infer.acme.example",)),
        probe=probe,
    )
    assert not result.passed
    assert any("connection reset" in item for item in result.refusals)


def test_runtime_descriptor_carries_the_intake_identity(tmp_path: Path) -> None:
    result = _passing_intake(tmp_path)
    descriptor = runtime_descriptor(result)
    # A plain kernel RuntimeDescriptor, not a concierge subclass: the fields are
    # real, so a package round-trip cannot silently drop them.
    assert type(descriptor) is RuntimeDescriptor
    assert descriptor.intake_identity == result.identity
    assert descriptor.sandbox_policy_digest == result.sandbox_report.policy_digest

    bare = RuntimeDescriptor(
        kind=descriptor.kind,
        entrypoint=descriptor.entrypoint,
        timeout_sec=descriptor.timeout_sec,
    )
    # The execution identity a scorecard records must change with the gate.
    assert descriptor.identity != bare.identity

    round_tripped = RuntimeDescriptor.model_validate(descriptor.model_dump(mode="json"))
    assert round_tripped == descriptor
    assert round_tripped.identity == descriptor.identity
    assert round_tripped.intake_identity == result.identity


def test_intake_result_cannot_disagree_with_its_own_findings() -> None:
    with pytest.raises(TaskContractError, match="cannot disagree"):
        IntakeResult(
            identity="c" * 64,
            kind="upload",
            format="safetensors",
            sandbox_report={"policy": {}},
            passed=True,
            refusals=("something was wrong",),
        )


# --------------------------------------------------------------------------
# Capability 1: assess
# --------------------------------------------------------------------------


def test_assess_refuses_a_failed_intake(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "model.ckpt")
    failed = intake_upload(_manifest(artifact), artifact, sandbox=SANDBOX, keyring=_keyring())
    with pytest.raises(TaskContractError, match="refusing to assess failed intake"):
        assess_model(failed, probe=lambda _: _video_probe_report())


def test_assess_refuses_a_probe_report_without_evidence() -> None:
    with pytest.raises(TaskContractError, match="no evidence"):
        ModelProbeReport(
            interface="video-predict",
            interaction_modes=(InteractionMode.SINGLE_TURN,),
            outputs=("next-step",),
            evidence=(),
        )


def test_proposal_is_inert_until_confirmed(tmp_path: Path) -> None:
    intake = _passing_intake(tmp_path)
    proposal = assess_model(intake, probe=lambda _: _video_probe_report())

    # Structural: a proposal is not a capability and has no satisfaction API.
    assert not issubclass(AssessProposal, CapabilitySpec)
    assert not hasattr(proposal, "satisfies")
    assert proposal.intake_identity == intake.identity
    assert "NOT USABLE until confirmed" in proposal.describe()

    # Runtime: the scored path refuses a draft. Type-level: mypy rejects it too,
    # which is why the ignore below is required rather than decorative.
    with pytest.raises(TaskContractError, match="unconfirmed assessment"):
        assert_confirmed(proposal)
    with pytest.raises(TaskContractError, match="unconfirmed assessment"):
        select_eval_plan(
            proposal,  # type: ignore[arg-type]
            catalog_paths=[VIDEO_TASK],
            budget=EvalBudget(max_total_trials=1),
        )

    confirmed = confirm(proposal, confirmed_by="tenant-reviewer@acme")
    assert isinstance(confirmed, ConfirmedCapability)
    assert not issubclass(ConfirmedCapability, AssessProposal)
    assert confirmed.proposal_digest == proposal.proposal_digest
    assert assert_confirmed(confirmed) is confirmed


def test_confirmation_must_name_a_party(tmp_path: Path) -> None:
    proposal = assess_model(_passing_intake(tmp_path), probe=lambda _: _video_probe_report())
    with pytest.raises((TaskContractError, ValueError)):
        confirm(proposal, confirmed_by="")


# --------------------------------------------------------------------------
# Capability 2: select and narrate
# --------------------------------------------------------------------------


def _pinned_catalog() -> list[Path]:
    """A fixed catalog: ordering assertions must not drift as tasks are added."""
    return [
        TASK_CATALOG / name
        for name in (
            "angiostress-dias",
            "broncho-airway-nav",
            "counterfactual-recovery",
            "laparoscopic-cholec-cvs",
            "lumen-nav-safe",
            "ortho-burr-safe",
            "video-nextstep",
        )
    ]


def test_plan_ranks_by_satisfaction_then_modality_and_respects_the_budget(
    tmp_path: Path,
) -> None:
    capability = _confirmed(tmp_path)
    plan = select_eval_plan(
        capability,
        catalog_paths=_pinned_catalog(),
        budget=EvalBudget(max_total_trials=6, max_trials_per_task=30),
    )

    # laparoscopic-cholec-cvs first: it is the only task whose modality the
    # agent declares. The rest of the video-predict family follows by id.
    assert [entry.task_id for entry in plan.entries] == [
        "laparoscopic-cholec-cvs",
        "angiostress-dias",
    ]
    assert plan.entries[0].modality_rank == 0
    assert plan.entries[1].modality_rank == 1
    assert [entry.trials for entry in plan.entries] == [5, 1]
    assert plan.total_trials == 6

    refusals = " | ".join(plan.refusals)
    for unsatisfied in ("broncho-airway-nav", "lumen-nav-safe", "ortho-burr-safe"):
        assert f"{unsatisfied}: capability" in refusals
    assert "does not satisfy interface" in refusals
    assert "interaction mode closed-loop not in [single-turn]" in refusals
    assert "video-nextstep: trial budget exhausted" in refusals
    for entry in plan.entries:
        assert entry.task_digest == tree_digest(Path(entry.path))
        assert entry.gates


def test_plan_never_exceeds_its_own_budget(tmp_path: Path) -> None:
    capability = _confirmed(tmp_path)
    plan = select_eval_plan(
        capability,
        catalog_paths=[VIDEO_TASK],
        budget=EvalBudget(max_total_trials=2, max_trials_per_task=1),
    )
    assert plan.total_trials == 1
    assert plan.entries[0].trials == 1


def test_narrate_plan_reports_refusals_gates_and_abstention_first(tmp_path: Path) -> None:
    capability = _confirmed(tmp_path)
    plan = select_eval_plan(
        capability, catalog_paths=_pinned_catalog(), budget=EvalBudget(max_total_trials=6)
    )
    text = narrate_plan(plan)
    assert text.index("REFUSED CANDIDATES") < text.index("SAFETY GATES AND ABSTENTION")
    assert text.index("SAFETY GATES AND ABSTENTION") < text.index("PER-WORLD ROWS")
    assert NO_COMPOSITE_FOOTER in text
    planned_gates = [gate for entry in plan.entries for gate in entry.gates]
    assert planned_gates
    for gate in planned_gates:
        assert gate in text


def test_narrate_results_reports_gates_and_abstention_before_worlds_and_no_composite(
    tmp_path: Path,
) -> None:
    result = run_job(
        task=load_task(VIDEO_TASK),
        task_dir=VIDEO_TASK,
        agent=load_agent(VIDEO_AGENT),
        agent_dir=VIDEO_AGENT,
        out=tmp_path / "job",
        n=3,
    )
    text = narrate_results([result])

    assert text.index("SAFETY GATES") < text.index("ABSTENTION")
    assert text.index("ABSTENTION") < text.index("PER-WORLD ROWS")
    assert "unsafe_prediction" in text
    assert f"unassessable in {result.headline_unassessable} of {result.n}" in text
    assert "failed a hard gate" in text
    assert result.head in text
    assert NO_COMPOSITE_FOOTER in text
    lowered = text.lower()
    for forbidden in (
        "composite score",
        "overall score",
        "aggregate score",
        "total score",
        "mean reward",
        "average reward",
    ):
        assert forbidden not in lowered


def test_narrate_results_tolerates_no_results() -> None:
    text = narrate_results([])
    assert "(no result)" in text
    assert NO_COMPOSITE_FOOTER in text


# --------------------------------------------------------------------------
# Capability 3: adapt
# --------------------------------------------------------------------------


def _parent_copy(tmp_path: Path, source: Path = COUNTERFACTUAL_TASK) -> Path:
    parent = tmp_path / "parent"
    shutil.copytree(source, parent, ignore=shutil.ignore_patterns("__pycache__"))
    return parent


def _degrading(
    hard_id: str, *, abstain_id: str = ""
) -> Callable[[AdaptCandidate], AdaptObservation]:
    def evaluate(candidate: AdaptCandidate) -> AdaptObservation:
        if candidate.id == abstain_id:
            return AdaptObservation(n=4, gate_failures=4, abstentions=4, headline_true=0)
        if candidate.id == hard_id:
            return AdaptObservation(n=4, gate_failures=3, abstentions=0, headline_true=0)
        return AdaptObservation(n=4, gate_failures=0, abstentions=0, headline_true=4)

    return evaluate


def test_search_scenarios_never_mutates_the_source_package(tmp_path: Path) -> None:
    parent = _parent_copy(tmp_path)
    before = tree_digest(parent)
    task = load_task(parent)
    candidate_ids: list[str] = []

    def evaluate(candidate: AdaptCandidate) -> AdaptObservation:
        candidate_ids.append(candidate.id)
        return AdaptObservation(n=2, gate_failures=1, abstentions=0, headline_true=1)

    report = search_scenarios(
        parent,
        space=ScenarioSpace(seeds=(7,), max_perturbations_per_candidate=1),
        evaluate=evaluate,
        budget=AdaptBudget(max_candidates=32, trials_per_candidate=2),
    )
    assert tree_digest(parent) == before
    assert report.parent_digest == before
    assert report.evaluated == len(candidate_ids) > len(task.scenarios)
    # Derived candidates are new objects: the declared scenarios are untouched.
    assert {scenario.id for scenario in load_task(parent).scenarios} == {
        scenario.id for scenario in task.scenarios
    }
    assert any("seed7" in item for item in candidate_ids)


def test_search_orders_by_degradation_and_refuses_unassessable_candidates(
    tmp_path: Path,
) -> None:
    parent = _parent_copy(tmp_path)
    space = ScenarioSpace(max_perturbations_per_candidate=1)
    ids = [candidate.id for candidate in _candidate_ids(parent, space)]
    hard, abstaining = ids[0], ids[-1]
    report = search_scenarios(
        parent,
        space=space,
        evaluate=_degrading(hard, abstain_id=abstaining),
        budget=AdaptBudget(max_candidates=32, trials_per_candidate=4),
    )
    assert report.ordering[0].candidate.id == hard
    assert report.ordering[0].honest
    assert report.hardest is not None
    assert report.hardest.candidate.id == hard
    assert report.ordering[-1].candidate.id == abstaining
    assert not report.ordering[-1].honest
    assert any("honesty bound" in item for item in report.refusals)
    assert "hardest honest first" in report.describe()


def _candidate_ids(parent: Path, space: ScenarioSpace) -> list[AdaptCandidate]:
    seen: list[AdaptCandidate] = []

    def evaluate(candidate: AdaptCandidate) -> AdaptObservation:
        seen.append(candidate)
        return AdaptObservation(n=1, gate_failures=0, abstentions=0, headline_true=1)

    search_scenarios(
        parent,
        space=space,
        evaluate=evaluate,
        budget=AdaptBudget(max_candidates=32, trials_per_candidate=1),
    )
    return seen


def test_search_refuses_a_knob_the_task_does_not_declare(tmp_path: Path) -> None:
    parent = _parent_copy(tmp_path)
    with pytest.raises(TaskContractError, match="declares no perturbation"):
        search_scenarios(
            parent,
            space=ScenarioSpace(perturbation_ids=("invented-collapse",)),
            evaluate=lambda _: AdaptObservation(
                n=1, gate_failures=0, abstentions=0, headline_true=1
            ),
            budget=AdaptBudget(max_candidates=4),
        )


def test_search_records_the_candidate_budget_it_could_not_spend(tmp_path: Path) -> None:
    parent = _parent_copy(tmp_path)
    report = search_scenarios(
        parent,
        space=ScenarioSpace(seeds=(11, 12, 13), max_perturbations_per_candidate=1),
        evaluate=lambda _: AdaptObservation(n=1, gate_failures=0, abstentions=0, headline_true=1),
        budget=AdaptBudget(max_candidates=2, trials_per_candidate=1),
    )
    assert report.evaluated == 2
    assert any("candidate budget exhausted" in item for item in report.refusals)


def test_freeze_writes_a_bumped_quarantined_digest_pinned_package(tmp_path: Path) -> None:
    parent = _parent_copy(tmp_path)
    task = load_task(parent)
    frozen = freeze_adapted_package(
        parent,
        scenarios=list(task.scenarios),
        perturbations=list(task.perturbations),
        out=tmp_path / "adapted",
    )
    frozen_dir = Path(frozen.path)

    assert frozen.task_version == "1-adapted1" != task.task_version
    assert frozen.parent_task_id == task.id
    assert frozen.parent_digest == tree_digest(parent)
    assert frozen.authored_by == "agent"
    assert frozen.public_leaderboard_eligible is False
    assert frozen.digest == tree_digest(frozen_dir)

    provenance = json.loads((frozen_dir / PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert provenance["authored_by"] == "agent"
    assert provenance["public_leaderboard_eligible"] is False
    assert provenance["parent"] == {
        "task_id": task.id,
        "task_version": task.task_version,
        "digest": frozen.parent_digest,
        # Pinned so a moved gate is caught even after the parent is deleted.
        "verifier_identity": verifier_identity(parent),
    }
    # The self-pin covers everything a scored run reads, and only that.
    assert provenance["content_digest"] == content_digest(frozen_dir)
    assert provenance["content_digest"] != frozen.digest

    reloaded = load_task(frozen_dir)
    assert reloaded.task_version == frozen.task_version
    assert (frozen_dir / "verifier.py").read_bytes() == (parent / "verifier.py").read_bytes()
    assert_verifier_untouched(parent, frozen_dir)
    assert assert_frozen_before_scoring(frozen) is frozen


def test_freeze_cannot_mint_leaderboard_eligibility(tmp_path: Path) -> None:
    with pytest.raises(TaskContractError, match="Tier-1 conformance"):
        FrozenPackage(
            path=str(tmp_path),
            task_id="t",
            task_version="1-adapted1",
            digest="d",
            parent_task_id="t",
            parent_task_version="1",
            parent_digest="p",
            public_leaderboard_eligible=True,
        )


def test_tampered_verifier_is_refused(tmp_path: Path) -> None:
    parent = _parent_copy(tmp_path)
    task = load_task(parent)
    frozen = freeze_adapted_package(
        parent,
        scenarios=list(task.scenarios),
        out=tmp_path / "adapted",
    )
    frozen_dir = Path(frozen.path)
    verifier = frozen_dir / "verifier.py"
    verifier.write_text(
        verifier.read_text(encoding="utf-8") + "\n# a scenario author is not a gate author\n",
        encoding="utf-8",
    )
    with pytest.raises(TaskContractError, match="never verifiers"):
        assert_verifier_untouched(parent, frozen_dir)


def test_retuned_gate_is_refused(tmp_path: Path) -> None:
    parent = _parent_copy(tmp_path)
    task = load_task(parent)
    frozen = freeze_adapted_package(
        parent,
        scenarios=list(task.scenarios),
        out=tmp_path / "adapted",
    )
    frozen_dir = Path(frozen.path)
    toml_path = frozen_dir / "task.toml"
    original = toml_path.read_text(encoding="utf-8")
    gate = task.verifier.gates[0]
    tampered = original.replace(
        f'id = "{gate.id}"',
        f'id = "{gate.id}-relaxed"',
        1,
    )
    assert tampered != original
    toml_path.write_text(tampered, encoding="utf-8")
    with pytest.raises(TaskContractError, match="never verifiers"):
        assert_verifier_untouched(parent, frozen_dir)


def test_editing_a_frozen_package_makes_it_unscoreable(tmp_path: Path) -> None:
    parent = _parent_copy(tmp_path)
    task = load_task(parent)
    frozen = freeze_adapted_package(
        parent,
        scenarios=list(task.scenarios),
        out=tmp_path / "adapted",
    )
    (Path(frozen.path) / "instruction.md").write_text("edited after freezing\n", encoding="utf-8")
    with pytest.raises(TaskContractError, match="edited after freezing"):
        assert_frozen_before_scoring(frozen)


def test_unfrozen_adaptation_cannot_be_scored() -> None:
    with pytest.raises(TaskContractError, match="expected a FrozenPackage"):
        assert_frozen_before_scoring({"path": "somewhere", "scenarios": []})


def test_freeze_refuses_to_write_inside_its_parent(tmp_path: Path) -> None:
    parent = _parent_copy(tmp_path)
    task = load_task(parent)
    with pytest.raises(TaskContractError, match="inside its parent"):
        freeze_adapted_package(
            parent,
            scenarios=list(task.scenarios),
            out=parent / "adapted",
        )


def test_freeze_refuses_an_empty_scenario_set(tmp_path: Path) -> None:
    parent = _parent_copy(tmp_path)
    with pytest.raises(TaskContractError, match="no scenario"):
        freeze_adapted_package(parent, scenarios=[], out=tmp_path / "adapted")


# --------------------------------------------------------------------------
# The provenance boundary: the quarantine the N11 invariants describe
# --------------------------------------------------------------------------


def _frozen_dir(tmp_path: Path) -> tuple[Path, Path]:
    """A parent package and a frozen adaptation of all its declared scenarios."""
    parent = _parent_copy(tmp_path)
    task = load_task(parent)
    frozen = freeze_adapted_package(
        parent,
        scenarios=list(task.scenarios),
        perturbations=list(task.perturbations),
        out=tmp_path / "adapted",
    )
    return parent, Path(frozen.path)


def _run(task_dir: Path, out: Path) -> JobResult:
    return run_job(
        task=load_task(task_dir),
        task_dir=task_dir,
        agent=load_agent(COUNTERFACTUAL_AGENT),
        agent_dir=COUNTERFACTUAL_AGENT,
        out=out,
        n=1,
    )


def test_a_curated_package_presents_no_provenance_and_is_not_a_suspect(tmp_path: Path) -> None:
    """No provenance.json means no adaptation claim, so there is nothing to verify."""
    parent = _parent_copy(tmp_path)
    assert read_provenance(parent) is None
    assert assert_scoreable_package(parent) is None
    # Returns None; the assertion is that it does not raise.
    assert_public_leaderboard_eligible(parent)


def test_a_frozen_adaptation_still_runs(tmp_path: Path) -> None:
    """The boundary refuses tampering, not adaptation: the honest package scores."""
    _parent, frozen_dir = _frozen_dir(tmp_path)
    result = _run(frozen_dir, tmp_path / "job")
    assert result.task_version == "1-adapted1"


def test_a_verifier_edited_after_freezing_is_refused_at_scoring(tmp_path: Path) -> None:
    """The reviewer's probe: freeze, edit the verifier, then try to score it.

    Before the boundary existed, ``run_job`` simply hashed the edited tree as a
    new task digest and scored it.
    """
    _parent, frozen_dir = _frozen_dir(tmp_path)
    verifier = frozen_dir / "verifier.py"
    verifier.write_text(
        verifier.read_text(encoding="utf-8") + "\n# a scenario author is not a gate author\n",
        encoding="utf-8",
    )
    with pytest.raises(TaskContractError, match="never verifiers"):
        assert_scoreable_package(frozen_dir)
    with pytest.raises(TaskContractError, match="never verifiers"):
        _run(frozen_dir, tmp_path / "job")


def test_a_retuned_gate_after_freezing_is_refused_at_scoring(tmp_path: Path) -> None:
    """A gate moved in task.toml is caught by the pinned parent verifier identity."""
    parent, frozen_dir = _frozen_dir(tmp_path)
    gate = load_task(parent).verifier.gates[0]
    toml_path = frozen_dir / "task.toml"
    original = toml_path.read_text(encoding="utf-8")
    tampered = original.replace(f'id = "{gate.id}"', f'id = "{gate.id}-relaxed"', 1)
    assert tampered != original
    toml_path.write_text(tampered, encoding="utf-8")
    with pytest.raises(TaskContractError, match="never verifiers"):
        _run(frozen_dir, tmp_path / "job")


def test_the_parent_verifier_pin_outlives_the_parent_package(tmp_path: Path) -> None:
    """The refusal must not depend on the parent still being on disk."""
    parent, frozen_dir = _frozen_dir(tmp_path)
    shutil.rmtree(parent)
    assert assert_scoreable_package(frozen_dir) is not None
    verifier = frozen_dir / "verifier.py"
    verifier.write_text(verifier.read_text(encoding="utf-8") + "\n# sweetened\n", encoding="utf-8")
    with pytest.raises(TaskContractError, match="never verifiers"):
        assert_scoreable_package(frozen_dir)


def test_a_scenario_edited_after_freezing_is_refused_at_scoring(tmp_path: Path) -> None:
    """Not only the verifier: the self-pin covers every file a run reads."""
    _parent, frozen_dir = _frozen_dir(tmp_path)
    (frozen_dir / "instruction.md").write_text("edited after freezing\n", encoding="utf-8")
    with pytest.raises(TaskContractError, match="edited after freezing"):
        _run(frozen_dir, tmp_path / "job")


def test_a_frozen_package_cannot_relabel_its_own_authorship(tmp_path: Path) -> None:
    """`--authored-by human` used to produce an agent package that denied it."""
    _parent, frozen_dir = _frozen_dir(tmp_path)
    record = json.loads((frozen_dir / PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    record["authored_by"] = "human"
    (frozen_dir / PROVENANCE_FILENAME).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(TaskContractError, match="authored_by 'human'"):
        assert_scoreable_package(frozen_dir)
    with pytest.raises(TaskContractError, match="authored_by 'human'"):
        _run(frozen_dir, tmp_path / "job")


def test_a_frozen_package_cannot_grant_itself_leaderboard_eligibility(tmp_path: Path) -> None:
    _parent, frozen_dir = _frozen_dir(tmp_path)
    record = json.loads((frozen_dir / PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    record["public_leaderboard_eligible"] = True
    (frozen_dir / PROVENANCE_FILENAME).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(TaskContractError, match="cannot grant it to itself"):
        assert_scoreable_package(frozen_dir)


def test_a_legacy_provenance_record_is_refused(tmp_path: Path) -> None:
    """A record written before the self-pin existed cannot establish its claim."""
    _parent, frozen_dir = _frozen_dir(tmp_path)
    path = frozen_dir / PROVENANCE_FILENAME
    record = json.loads(path.read_text(encoding="utf-8"))
    record["format_version"] = "1"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(TaskContractError, match="format_version '1'"):
        assert_scoreable_package(frozen_dir)


def test_a_provenance_record_missing_its_content_pin_is_refused(tmp_path: Path) -> None:
    _parent, frozen_dir = _frozen_dir(tmp_path)
    path = frozen_dir / PROVENANCE_FILENAME
    record = json.loads(path.read_text(encoding="utf-8"))
    del record["content_digest"]
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(TaskContractError, match="not a readable package provenance record"):
        assert_scoreable_package(frozen_dir)


def test_an_unreadable_provenance_file_is_refused_not_ignored(tmp_path: Path) -> None:
    _parent, frozen_dir = _frozen_dir(tmp_path)
    (frozen_dir / PROVENANCE_FILENAME).write_text("{ not json", encoding="utf-8")
    with pytest.raises(TaskContractError, match="not readable JSON"):
        assert_scoreable_package(frozen_dir)


def test_a_quarantined_package_is_refused_at_public_leaderboard_ingestion(
    tmp_path: Path,
) -> None:
    """The flag is honoured where it matters, and the refusal names the package."""
    _parent, frozen_dir = _frozen_dir(tmp_path)
    with pytest.raises(TaskContractError) as excinfo:
        assert_public_leaderboard_eligible(frozen_dir)
    message = str(excinfo.value)
    assert "refusing to publish counterfactual-recovery@1-adapted1" in message
    assert "Tier-1 conformance" in message


def test_leaderboard_ingestion_checks_integrity_before_the_quarantine(tmp_path: Path) -> None:
    """A quarantine flag on an edited package says nothing about what ran."""
    _parent, frozen_dir = _frozen_dir(tmp_path)
    verifier = frozen_dir / "verifier.py"
    verifier.write_text(verifier.read_text(encoding="utf-8") + "\n# sweetened\n", encoding="utf-8")
    with pytest.raises(TaskContractError, match="never verifiers"):
        assert_public_leaderboard_eligible(frozen_dir)


def _rename_version(task_dir: Path, new_version: str) -> None:
    """Move a package out of the '-adaptedN' lineage, changing its identity."""
    toml_path = task_dir / "task.toml"
    text = toml_path.read_text(encoding="utf-8")
    replaced = text.replace(
        f'task_version = "{load_task(task_dir).task_version}"', f'task_version = "{new_version}"', 1
    )
    assert replaced != text
    toml_path.write_text(replaced, encoding="utf-8")


def test_curated_packages_carry_no_adaptation_tells() -> None:
    """The tells must not fire on honest packages: a false refusal blocks users."""
    for package in sorted(TASK_CATALOG.iterdir()):
        if not (package / "task.toml").is_file():
            continue
        try:
            task = load_task(package)
        except TaskContractError:  # a package broken for unrelated reasons
            continue
        assert adaptation_tells(package, task) == (), package.name


def test_a_frozen_package_carries_its_lineage_in_its_task_version(tmp_path: Path) -> None:
    _parent, frozen_dir = _frozen_dir(tmp_path)
    tells = adaptation_tells(frozen_dir)
    assert any("-adaptedN" in tell for tell in tells)


def test_deleting_provenance_no_longer_escapes_leaderboard_ingestion(tmp_path: Path) -> None:
    """The bypass no local check can close is at least made loud.

    Deleting the record used to make an adaptation indistinguishable from a
    curated package. It still defeats the *content* pin — nothing local can
    stop that — but the package keeps the adapted lineage in its identity, and
    publishing something that claims that lineage without the evidence for it
    is refused by name.
    """
    _parent, frozen_dir = _frozen_dir(tmp_path)
    (frozen_dir / PROVENANCE_FILENAME).unlink()
    # Scoring is deliberately unaffected: there is no claim left to verify.
    assert assert_scoreable_package(frozen_dir) is None
    with pytest.raises(TaskContractError) as excinfo:
        assert_public_leaderboard_eligible(frozen_dir)
    message = str(excinfo.value)
    assert "carries the marks of an agent-authored adaptation" in message
    assert "task_version '1-adapted1'" in message
    assert PROVENANCE_FILENAME in message


def test_escaping_the_tells_costs_the_package_its_adapted_identity(tmp_path: Path) -> None:
    """The deliberate escape hatch, and the price Main named for taking it.

    An honest author who really does want an unadapted package under this name
    can have one — by renaming it out of the lineage, which severs exactly the
    parent linkage a deletion was trying to keep.
    """
    _parent, frozen_dir = _frozen_dir(tmp_path)
    (frozen_dir / PROVENANCE_FILENAME).unlink()
    _rename_version(frozen_dir, "2")
    assert adaptation_tells(frozen_dir) == ()
    # Returns None; the assertion is that it no longer refuses.
    assert_public_leaderboard_eligible(frozen_dir)


def test_a_search_derived_scenario_is_a_tell_even_after_renaming(tmp_path: Path) -> None:
    """Renaming the version is not enough when the search left its own mark."""
    parent = _parent_copy(tmp_path)
    scenario = load_task(parent).scenarios[0]
    # Exactly what adapt._derive_scenario writes for a re-seeded candidate.
    derived = scenario.model_copy(update={"id": f"{scenario.id}-seed7", "seed": 7})
    frozen = freeze_adapted_package(parent, scenarios=[derived], out=tmp_path / "adapted")
    frozen_dir = Path(frozen.path)
    (frozen_dir / PROVENANCE_FILENAME).unlink()
    _rename_version(frozen_dir, "2")
    tells = adaptation_tells(frozen_dir)
    assert tells == (
        f"scenario '{scenario.id}-seed7' is named for its own seed (7), the shape a "
        "search-derived scenario takes",
    )
    with pytest.raises(TaskContractError, match="named for its own seed"):
        assert_public_leaderboard_eligible(frozen_dir)


def test_a_scenario_whose_name_and_seed_disagree_is_not_a_tell(tmp_path: Path) -> None:
    """Matched on the id/seed pair, so somebody's own naming is not a refusal."""
    parent = _parent_copy(tmp_path)
    scenario = load_task(parent).scenarios[0]
    named = scenario.model_copy(update={"id": f"{scenario.id}-seed7", "seed": 3})
    frozen = freeze_adapted_package(parent, scenarios=[named], out=tmp_path / "adapted")
    frozen_dir = Path(frozen.path)
    (frozen_dir / PROVENANCE_FILENAME).unlink()
    _rename_version(frozen_dir, "2")
    assert adaptation_tells(frozen_dir) == ()


def test_frozen_package_refuses_a_relabelled_authorship_in_memory() -> None:
    with pytest.raises(TaskContractError, match="not a caller's choice"):
        FrozenPackage(
            path="somewhere",
            task_id="t",
            task_version="1-adapted1",
            digest="d",
            parent_task_id="t",
            parent_task_version="1",
            parent_digest="p",
            authored_by="human",
        )


def test_freeze_takes_no_authorship_argument() -> None:
    """The authorship class is not a parameter, so no caller can pass one."""
    assert "authored_by" not in inspect.signature(freeze_adapted_package).parameters


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------


def _keyring_file(tmp_path: Path) -> Path:
    """The control plane's own key material, on its own disk."""
    path = tmp_path / "keyring.json"
    path.write_text(
        json.dumps(
            {
                "keys": [
                    {
                        "key_id": key.key_id,
                        "tenant": key.tenant,
                        "secret_hex": key.secret.hex(),
                    }
                    for key in (ACME_KEY, RIVAL_KEY)
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="surgeval")
    sub = parser.add_subparsers(dest="command", required=True)
    register(sub)
    return parser


def test_cli_intake_prints_refusals_first_and_exits_one(tmp_path: Path, capsys) -> None:
    artifact = _artifact(tmp_path, "model.ckpt")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_manifest(artifact).model_dump_json(indent=2) + "\n", encoding="utf-8")
    args = _parser().parse_args(
        [
            "concierge",
            "intake",
            "--manifest",
            str(manifest_path),
            "--artifact",
            str(artifact),
            "--keyring",
            str(_keyring_file(tmp_path)),
            "--out",
            str(tmp_path / "intake.json"),
        ]
    )
    assert args.func(args) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("REFUSED: ")
    assert "deserialization executes code" in captured.err
    assert "REFUSED" in captured.out


def test_cli_intake_assess_select_round_trip(tmp_path: Path, capsys) -> None:
    artifact = _artifact(tmp_path, "model.safetensors")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_manifest(artifact).model_dump_json(indent=2) + "\n", encoding="utf-8")
    intake_json = tmp_path / "intake.json"
    parser = _parser()

    args = parser.parse_args(
        [
            "concierge",
            "intake",
            "--manifest",
            str(manifest_path),
            "--artifact",
            str(artifact),
            "--keyring",
            str(_keyring_file(tmp_path)),
            "--out",
            str(intake_json),
        ]
    )
    assert args.func(args) == 0

    probe_path = tmp_path / "probe.json"
    probe_path.write_text(_video_probe_report().model_dump_json(indent=2) + "\n", encoding="utf-8")
    capability_json = tmp_path / "capability.json"
    args = parser.parse_args(
        [
            "concierge",
            "assess",
            "--intake",
            str(intake_json),
            "--probe-report",
            str(probe_path),
            "--out",
            str(capability_json),
            "--confirmed-by",
            "tenant-reviewer@acme",
        ]
    )
    assert args.func(args) == 0
    assert ConfirmedCapability.model_validate_json(capability_json.read_text(encoding="utf-8"))

    args = parser.parse_args(
        [
            "concierge",
            "select",
            "--capability",
            str(capability_json),
            "--catalog",
            str(TASK_CATALOG),
            "--budget",
            "6",
        ]
    )
    # Exit 1: candidates were refused, and the plan says so before anything else.
    assert args.func(args) == 1
    captured = capsys.readouterr()
    assert "does not satisfy interface" in captured.err
    assert NO_COMPOSITE_FOOTER in captured.out


def test_cli_adapt_freezes_a_quarantined_package(tmp_path: Path, capsys) -> None:
    parent = _parent_copy(tmp_path)
    args = _parser().parse_args(
        [
            "concierge",
            "adapt",
            "--task",
            str(parent),
            "--out",
            str(tmp_path / "adapted"),
        ]
    )
    assert args.func(args) == 0
    captured = capsys.readouterr()
    assert "QUARANTINED" in captured.out
    assert "1-adapted1" in captured.out
    assert_verifier_untouched(parent, tmp_path / "adapted")
    assert assert_scoreable_package(tmp_path / "adapted") is not None


def test_cli_adapt_has_no_authored_by_flag(tmp_path: Path) -> None:
    """`concierge adapt --authored-by human` produced an agent package that
    denied being one, defeating the quarantine keyed on that field."""
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "concierge",
                "adapt",
                "--task",
                str(tmp_path / "parent"),
                "--out",
                str(tmp_path / "adapted"),
                "--authored-by",
                "human",
            ]
        )
