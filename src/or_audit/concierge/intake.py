"""Capability 0: the pre-Assess intake gate for untrusted models and endpoints.

An uploaded weight file and a tenant-registered endpoint are hostile inputs
until proven otherwise, and they are hostile in different ways. A weight file
in a pickle-family format *is code*: loading it executes whatever the author
put in its reduce protocol, in whatever process opened it. An arbitrary
endpoint URL is an SSRF primitive: it aims probing traffic at whatever the
caller can reach, including a customer's clinical network and the cloud
instance-metadata service.

This module is therefore deliberately incapable of the dangerous operation:
**nothing here deserializes an upload.** The only thing done to upload bytes is
streaming them through SHA-256. There is no ``pickle``, ``torch``, ``joblib``,
``dill``, or ``cloudpickle`` import in this module, and ``tests/test_concierge``
asserts that by parsing this file's imports — so the property survives later
edits instead of resting on this docstring.

A manifest is only a *claim* until it is authenticated. ``declared_by`` is
free text and the format, digest, and entrypoint are exactly what an attacker
would want to choose, so intake verifies an HMAC-SHA256 signature over the
whole manifest against a :class:`TenantKeyring` the *control plane* owns and
passes in. Trust therefore never comes from the manifest: an unsigned
manifest, an unknown key, a key registered to a different tenant, or a single
flipped byte anywhere in the signed payload all refuse, and the verified
signer is recorded in the result so the eventual execution identity names who
asserted the upload.

Refusals are *returned*, not raised. An intake gate must report every finding
for one artifact, and the record of a rejected upload is itself a deliverable.
Nothing downstream can act on a rejected artifact, because
:func:`runtime_descriptor` is the only bridge from an intake result to an
executable descriptor and it refuses any result that did not pass.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Protocol, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from or_audit.audit.canonical import canonical_bytes, digest
from or_audit.errors import TaskContractError
from or_audit.eval.contracts import (
    RuntimeDescriptor,
    RuntimeKind,
    SHA256Hex,
    Slug,
)

#: Upload formats whose load path is data-only, plus ``container`` for a
#: tenant-supplied, digest-pinned image that loads its own custom format inside
#: the sandbox. Anything else is refused by name.
UPLOAD_FORMAT_ALLOWLIST: Final[tuple[str, ...]] = ("safetensors", "onnx", "gguf", "container")

#: Declared formats whose loaders execute author-controlled code. Values name
#: the mechanism, so the refusal explains itself to whoever hits it.
_EXECUTABLE_FORMATS: Final[dict[str, str]] = {
    "pickle": "Python pickle",
    "pkl": "Python pickle",
    "cloudpickle": "cloudpickle (pickle with extra reach)",
    "dill": "dill (pickle with extra reach)",
    "joblib": "joblib (pickle container)",
    "pytorch": "torch.load (pickle archive)",
    "torch": "torch.load (pickle archive)",
    "pt": "torch.load (pickle archive)",
    "pth": "torch.load (pickle archive)",
    "checkpoint": "a framework checkpoint (pickle archive)",
    "ckpt": "a framework checkpoint (pickle archive)",
    "keras": "Keras HDF5 (Lambda layers execute)",
    "h5": "Keras HDF5 (Lambda layers execute)",
}

#: Artifact suffixes with the same problem, checked independently of the
#: declared format: a ``.ckpt`` file relabelled ``safetensors`` is still a
#: program.
_EXECUTABLE_SUFFIXES: Final[dict[str, str]] = {
    ".pt": "torch.load (pickle archive)",
    ".pth": "torch.load (pickle archive)",
    ".ckpt": "a framework checkpoint (pickle archive)",
    ".pkl": "Python pickle",
    ".pickle": "Python pickle",
    ".bin": "a torch pickle archive (the historical HF weight layout)",
    ".joblib": "joblib (pickle container)",
    ".dill": "dill (pickle with extra reach)",
    ".h5": "Keras HDF5 (Lambda layers execute)",
}

_DESERIALIZATION_ADVICE: Final = (
    "deserialization executes code at load time, so this artifact is a program "
    "rather than weights; re-export to safetensors, or declare format "
    '"container" and supply a digest-pinned image that loads it inside the '
    "no-egress sandbox"
)

#: Synthetic probe body. Marked so a real observation can never be sent by
#: accident: :class:`ProbeRequest` refuses an unmarked payload.
SYNTHETIC_PROBE_PAYLOAD: Final[dict[str, Any]] = {
    "synthetic": True,
    "source": "surgeval-concierge-intake",
    "note": "schema probe only; contains no task, patient, or dataset content",
}

_CHUNK_BYTES: Final = 1 << 20


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class IntakeKind(StrEnum):
    """Which hostile-input path an intake result came from."""

    UPLOAD = "upload"
    ENDPOINT = "endpoint"


class SandboxPolicy(_Frozen):
    """Execution tier for an untrusted upload.

    ``egress`` and ``trust_remote_code`` are fields that must be ``False``
    rather than absent constants: the sandbox report has to *state* the
    posture for an auditor, and an implicit absence attests nothing. Because
    construction refuses the unsafe values, an unsafe policy has no
    representation — there is no object to pass, log, or accidentally default
    into.
    """

    egress: bool = False
    cpu_quota: Annotated[float, Field(gt=0.0, le=1024.0)] = 2.0
    memory_bytes: Annotated[int, Field(ge=1 << 20)] = 8 << 30
    disk_bytes: Annotated[int, Field(ge=1 << 20)] = 16 << 30
    #: Pinned device class, or ``""`` for a CPU-only tier.
    gpu: str = ""
    trust_remote_code: bool = False
    artifact_scan: bool = True

    @model_validator(mode="after")
    def _unsafe_states_have_no_representation(self) -> Self:
        if self.egress:
            raise TaskContractError(
                "a model-intake sandbox cannot allow egress: the artifact is "
                "untrusted code until it has been gated, and an egressing "
                "sandbox turns an upload into a pivot into the tenant network"
            )
        if self.trust_remote_code:
            raise TaskContractError(
                "trust_remote_code=True executes repository-authored Python "
                "during model load; it is exactly the intake gate this class "
                "exists to hold, so it has no permitted value here"
            )
        return self


class ProbePolicy(_Frozen):
    """Bounded, synthetic-only probing tier for a registered endpoint.

    Endpoint intake needs *some* egress — that is what a reachability probe
    is — so it is not represented as a :class:`SandboxPolicy` with the rule
    quietly relaxed. It is a separate tier with the narrow allowance written
    down: named hosts only, a bounded probe count, synthetic inputs only, and
    ``tier="sandbox"`` recording that probing never runs from the control
    plane.
    """

    tier: Literal["sandbox"] = "sandbox"
    allowed_hosts: tuple[str, ...]
    max_probes: Annotated[int, Field(ge=1, le=64)] = 3
    synthetic_inputs_only: bool = True

    @model_validator(mode="after")
    def _bounded_and_synthetic(self) -> Self:
        if not self.allowed_hosts:
            raise TaskContractError("a probe policy must name the host(s) it may reach")
        if not self.synthetic_inputs_only:
            raise TaskContractError(
                "endpoint probing sends synthetic inputs only: a probe carrying "
                "task or dataset content would hand evaluation material to an "
                "unaudited third-party endpoint"
            )
        return self


class SandboxReport(_Frozen):
    """What ran where, under which posture, with what findings."""

    tier: Literal["sandbox"] = "sandbox"
    policy: SandboxPolicy | None = None
    probe_policy: ProbePolicy | None = None
    probes: Annotated[int, Field(ge=0)] = 0
    findings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _names_a_tier(self) -> Self:
        if self.policy is None and self.probe_policy is None:
            raise TaskContractError("a sandbox report must carry the policy the work ran under")
        return self

    @property
    def policy_digest(self) -> str:
        """Digest of the posture, for binding into an execution identity."""
        return digest(
            {
                "tier": self.tier,
                "policy": self.policy.model_dump(mode="json") if self.policy else None,
                "probe_policy": (
                    self.probe_policy.model_dump(mode="json") if self.probe_policy else None
                ),
            }
        )


class UploadManifest(_Frozen):
    """Tenant-signed declaration accompanying an uploaded artifact.

    ``format`` is a declared slug rather than a closed enum on purpose: the
    allowlist is enforced by :func:`intake_upload`, which can then *record*
    why a pickle upload was rejected. A type that cannot represent the hostile
    input cannot produce an audit record of refusing it. For the same reason an
    unsigned manifest is representable: intake has to be able to refuse one on
    the record.
    """

    format: Slug
    digest: SHA256Hex
    entrypoint: Annotated[str, StringConstraints(min_length=1, max_length=400)]
    size_bytes: Annotated[int, Field(ge=1)]
    declared_by: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    #: Tenant-supplied runtime image, required when ``format == "container"``.
    image: str = ""
    image_digest: str = ""
    #: Control-plane-minted key id naming which tenant secret signed this
    #: manifest. Part of the signed payload, so it cannot be swapped either.
    key_id: Annotated[str, StringConstraints(max_length=120)] = ""
    #: Lowercase-hex HMAC-SHA256 over :func:`signed_payload`. Malformed values
    #: are not a construction error: they fail verification and become a
    #: refusal, which is the auditable outcome.
    signature: Annotated[str, StringConstraints(max_length=200)] = ""

    @model_validator(mode="after")
    def _pins_are_complete(self) -> Self:
        if bool(self.image) != bool(self.image_digest):
            raise TaskContractError(
                "image and image_digest must be declared together: an image "
                "reference without a content digest is not a pin, and a mutable "
                "tag is a different program tomorrow"
            )
        if bool(self.key_id) != bool(self.signature):
            raise TaskContractError(
                "key_id and signature must be declared together: a signature "
                "without a key id names no verifier, and a key id without a "
                "signature authenticates nothing"
            )
        return self


#: Minimum shared-secret length. A 256-bit key matches the MAC it feeds; a
#: short shared secret is a guessable one, and this gate is the only thing
#: standing between a stranger and "trusted tenant upload".
MIN_SIGNING_SECRET_BYTES: Final = 32


@dataclass(frozen=True, repr=False)
class SigningKey:
    """One control-plane-minted key. ``repr`` never shows the secret."""

    key_id: str
    tenant: str
    secret: bytes

    def __repr__(self) -> str:
        return f"SigningKey(key_id={self.key_id!r}, tenant={self.tenant!r}, secret=<redacted>)"


class TenantKeyring:
    """Control-plane-owned signing keys, keyed by ``key_id``.

    Deliberately not a pydantic model and deliberately not reachable from a
    manifest: the keyring is passed *into* :func:`intake_upload` by whoever
    operates the control plane, so authentication authority never travels with
    the artifact being authenticated. ``key_id`` is globally unique, which is
    what lets intake tell "unknown key" apart from "key belongs to a different
    tenant" — two different attacks that deserve two different refusals.
    """

    __slots__ = ("_keys",)

    def __init__(self, keys: Iterable[SigningKey] = ()) -> None:
        self._keys: dict[str, SigningKey] = {}
        for key in keys:
            self.register(key_id=key.key_id, tenant=key.tenant, secret=key.secret)

    def register(self, *, key_id: str, tenant: str, secret: bytes) -> SigningKey:
        """Mint or re-assert one key. Rebinding a key id is refused."""
        if not key_id or not tenant:
            raise TaskContractError("a signing key needs both a key_id and a tenant")
        if len(secret) < MIN_SIGNING_SECRET_BYTES:
            raise TaskContractError(
                f"signing secret for {key_id!r} is {len(secret)} bytes; at least "
                f"{MIN_SIGNING_SECRET_BYTES} are required for an HMAC-SHA256 key"
            )
        existing = self._keys.get(key_id)
        if existing is not None and existing.tenant != tenant:
            raise TaskContractError(
                f"refusing to rebind key {key_id!r} from tenant {existing.tenant!r} "
                f"to {tenant!r}: a key id identifies one tenant for the life of the "
                "signatures it produced"
            )
        key = SigningKey(key_id=key_id, tenant=tenant, secret=secret)
        self._keys[key_id] = key
        return key

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> TenantKeyring:
        """Build a keyring from ``{key_id, tenant, secret_hex}`` records."""
        keyring = cls()
        for index, record in enumerate(records):
            raw = str(record.get("secret_hex", ""))
            try:
                secret = bytes.fromhex(raw)
            except ValueError as exc:
                raise TaskContractError(
                    f"keyring entry {index} secret_hex is not hex: {exc}"
                ) from exc
            keyring.register(
                key_id=str(record.get("key_id", "")),
                tenant=str(record.get("tenant", "")),
                secret=secret,
            )
        return keyring

    def lookup(self, key_id: str) -> SigningKey | None:
        return self._keys.get(key_id)

    @property
    def key_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._keys))

    def __contains__(self, key_id: object) -> bool:
        return key_id in self._keys

    def __len__(self) -> int:
        return len(self._keys)

    def __repr__(self) -> str:
        return f"TenantKeyring(key_ids={list(self.key_ids)}, secrets=<redacted>)"


def signed_payload(manifest: UploadManifest) -> dict[str, Any]:
    """Every declared field except the signature itself.

    Whole-manifest coverage rather than a hand-listed subset: a signature that
    covers only some fields invites exactly the substitution it exists to stop
    (swap the container image, keep the signature), and a subset would silently
    stop covering any field added later.
    """
    payload = manifest.model_dump(mode="json")
    payload.pop("signature", None)
    return payload


def manifest_signature(manifest: UploadManifest, *, secret: bytes) -> str:
    """HMAC-SHA256 over the RFC 8785 canonical form of the signed payload.

    JCS rather than ad-hoc ``json.dumps`` so a tenant's signer can be written
    in another language and still produce the same bytes.
    """
    return hmac.new(secret, canonical_bytes(signed_payload(manifest)), hashlib.sha256).hexdigest()


def sign_manifest(manifest: UploadManifest, *, key: SigningKey) -> UploadManifest:
    """Return a copy of ``manifest`` signed by ``key``. Used by the tenant side."""
    signed = manifest.model_copy(update={"key_id": key.key_id, "signature": ""})
    return signed.model_copy(update={"signature": manifest_signature(signed, secret=key.secret)})


def verify_manifest_signature(
    manifest: UploadManifest,
    *,
    keyring: TenantKeyring,
) -> tuple[SigningKey | None, str]:
    """Return the verified signer, or an empty signer and one refusal.

    The comparison is :func:`hmac.compare_digest`; the tenant binding is
    checked against the keyring's record for the key id, never against the
    manifest's own ``declared_by``.
    """
    if not manifest.signature or not manifest.key_id:
        return None, (
            "refusing unsigned upload manifest: declared_by "
            f"{manifest.declared_by!r} is an unauthenticated claim, and format, "
            "digest, and entrypoint are exactly the fields an attacker would "
            "choose. Sign the manifest with a control-plane-minted tenant key"
        )
    key = keyring.lookup(manifest.key_id)
    if key is None:
        return None, (
            f"refusing upload manifest signed with unknown key_id "
            f"{manifest.key_id!r}: the control plane holds no secret for it, so "
            "the signature cannot be checked at all"
        )
    if key.tenant != manifest.declared_by:
        return None, (
            f"refusing upload manifest: key {manifest.key_id!r} is registered to "
            f"tenant {key.tenant!r} but the manifest declares "
            f"{manifest.declared_by!r}; one tenant cannot assert an upload on "
            "another's behalf"
        )
    expected = manifest_signature(manifest, secret=key.secret)
    if not hmac.compare_digest(expected, manifest.signature):
        return None, (
            f"refusing upload manifest: signature does not verify under key "
            f"{manifest.key_id!r}. Every declared field is covered, so a changed "
            "format, digest, entrypoint, size, image pin, or tenant invalidates it"
        )
    return key, ""


class EndpointDescriptor(_Frozen):
    """Tenant-registered inference endpoint offered for evaluation."""

    id: Slug
    url: Annotated[str, StringConstraints(min_length=8, max_length=500)]
    model: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    declared_by: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    probe_budget: Annotated[int, Field(ge=1, le=16)] = 3


class EndpointAllowlist(_Frozen):
    """Hosts a tenant registered, normalized for exact-match comparison."""

    hosts: tuple[str, ...]

    @model_validator(mode="after")
    def _non_empty(self) -> Self:
        if not self.hosts:
            raise TaskContractError(
                "an empty endpoint allowlist cannot be satisfied; register the "
                "host explicitly rather than probing whatever a URL names"
            )
        return self

    def permits(self, host: str) -> bool:
        return host.strip().lower() in {entry.strip().lower() for entry in self.hosts}


class ProbeRequest(_Frozen):
    """One bounded, synthetic probe the transport is asked to perform."""

    url: str
    index: Annotated[int, Field(ge=0)]
    payload: dict[str, Any] = Field(default_factory=lambda: dict(SYNTHETIC_PROBE_PAYLOAD))

    @model_validator(mode="after")
    def _payload_is_marked_synthetic(self) -> Self:
        if self.payload.get("synthetic") is not True:
            raise TaskContractError(
                "an intake probe payload must be marked synthetic: probing an "
                "unaudited endpoint with real observations exports evaluation "
                "material off-platform"
            )
        return self


class ProbeResponse(_Frozen):
    """What the injected transport observed. No network code lives here."""

    status_code: Annotated[int, Field(ge=100, le=599)]
    #: URL the transport actually reached, after any redirect.
    final_url: str
    #: Redirect chain, in order, as offered by the endpoint.
    redirects: tuple[str, ...] = ()
    #: Observed response schema keys. Content is not retained.
    schema_keys: tuple[str, ...] = ()


class EndpointProbe(Protocol):
    """Injected probe transport. Intake itself performs no network I/O."""

    def __call__(self, request: ProbeRequest, /) -> ProbeResponse: ...


class IntakeResult(_Frozen):
    """Outcome of the gate: an identity, what was checked, and every refusal.

    ``passed`` is not independently settable — it must equal "no refusals" —
    so a passed result carrying an unresolved refusal cannot be constructed.
    """

    identity: SHA256Hex
    kind: IntakeKind
    format: str
    digests: dict[str, str] = Field(default_factory=dict)
    sandbox_report: SandboxReport
    passed: bool
    refusals: tuple[str, ...] = ()
    #: Upload facts carried forward to a runtime descriptor.
    entrypoint: str = ""
    image: str = ""
    image_digest: str = ""
    #: Verified signer of the upload manifest, taken from the keyring record
    #: rather than from the manifest, and folded into ``identity`` so the
    #: eventual execution identity names who asserted this artifact.
    signer_tenant: str = ""
    signer_key_id: str = ""
    #: Key id the manifest *claimed*, verified or not. Recorded so a refused
    #: intake shows which key was asserted; trust never derives from this field.
    claimed_key_id: str = ""
    #: Endpoint facts carried forward to a runtime descriptor.
    base_url: str = ""
    model: str = ""

    @model_validator(mode="after")
    def _passed_means_no_refusals(self) -> Self:
        if self.passed is bool(self.refusals):
            raise TaskContractError(
                f"intake result {self.identity} claims passed={self.passed} with "
                f"{len(self.refusals)} refusal(s); a gate result cannot disagree "
                "with its own findings"
            )
        return self

    def describe(self) -> str:
        lines = [
            f"Intake {self.kind.value} {self.identity}",
            f"  format   {self.format}",
            f"  signer   {self.signer}",
            f"  verdict  {'PASSED' if self.passed else 'REFUSED'}",
        ]
        for name, value in sorted(self.digests.items()):
            lines.append(f"  digest   {name}={value}")
        for finding in self.sandbox_report.findings:
            lines.append(f"  checked  {finding}")
        for refusal in self.refusals:
            lines.append(f"  REFUSED: {refusal}")
        return "\n".join(lines)

    @property
    def signer(self) -> str:
        """``tenant#key_id`` of the *verified* signer, else why there is none.

        A failed verification is not the same state as an unsigned manifest, and
        reporting both as ``unsigned`` would make a forged signature look like a
        missing one in the record an operator reads.
        """
        if self.signer_key_id:
            return f"{self.signer_tenant}#{self.signer_key_id}"
        if self.claimed_key_id:
            return f"unverified (claimed key {self.claimed_key_id})"
        return "unsigned"


def _file_sha256(path: Path) -> tuple[str, int]:
    """Stream bytes through SHA-256. The bytes are never interpreted."""
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            hasher.update(chunk)
    return hasher.hexdigest(), size


def _executable_format_refusal(label: str, mechanism: str) -> str:
    return f"refusing {label}: {mechanism} — {_DESERIALIZATION_ADVICE}"


def _identity(payload: dict[str, Any]) -> str:
    return digest(payload)


def intake_upload(
    manifest: UploadManifest,
    path: Path | str,
    *,
    sandbox: SandboxPolicy,
    keyring: TenantKeyring,
) -> IntakeResult:
    """Gate an uploaded artifact without ever loading it.

    Checks, in order: the manifest carries a signature that verifies under a
    control-plane key registered to the declaring tenant; the artifact exists
    and its bytes hash to the declared digest; the declared format is on the
    allowlist; neither the format nor the file suffix names a code-executing
    deserializer; a custom format carries a digest-pinned tenant image; and the
    sandbox tier actually scans artifacts. Every failure is recorded as a
    refusal so the rejection is auditable, and a refused result cannot become a
    runtime descriptor.

    ``keyring`` is a required argument rather than module state: the authority
    that authenticates an upload belongs to whoever operates the control plane,
    and it must not be reachable from the artifact being authenticated.
    """
    artifact = Path(path)
    refusals: list[str] = []
    findings: list[str] = [
        "upload bytes were hashed, never deserialized",
        f"sandbox tier: egress={sandbox.egress}, "
        f"trust_remote_code={sandbox.trust_remote_code}, "
        f"cpu_quota={sandbox.cpu_quota}, memory_bytes={sandbox.memory_bytes}, "
        f"disk_bytes={sandbox.disk_bytes}, gpu={sandbox.gpu or 'none'}",
    ]
    digests: dict[str, str] = {}

    signer, signature_refusal = verify_manifest_signature(manifest, keyring=keyring)
    if signature_refusal:
        refusals.append(signature_refusal)
    else:
        findings.append(
            f"manifest signature verified: every declared field is covered by "
            f"HMAC-SHA256 under key {manifest.key_id} registered to "
            f"{signer.tenant if signer else '(none)'}"
        )

    declared_format = manifest.format.lower()
    mechanism = _EXECUTABLE_FORMATS.get(declared_format)
    if mechanism is not None:
        refusals.append(
            _executable_format_refusal(f"declared format {manifest.format!r}", mechanism)
        )
    elif declared_format not in UPLOAD_FORMAT_ALLOWLIST:
        allowed = ", ".join(UPLOAD_FORMAT_ALLOWLIST)
        refusals.append(
            f"refusing declared format {manifest.format!r}: only [{allowed}] are "
            "accepted, because every other loader we would have to reach for "
            "performs deserialization that executes code at load time"
        )

    suffix_mechanism = _EXECUTABLE_SUFFIXES.get(artifact.suffix.lower())
    if suffix_mechanism is not None:
        refusals.append(
            _executable_format_refusal(f"artifact suffix {artifact.suffix!r}", suffix_mechanism)
        )
    else:
        findings.append(f"artifact suffix {artifact.suffix or '(none)'} is not a pickle family")

    if not artifact.is_file():
        refusals.append(f"refusing upload: no artifact at {artifact}")
    else:
        actual, size = _file_sha256(artifact)
        digests["artifact"] = actual
        if actual != manifest.digest:
            refusals.append(
                f"refusing upload: artifact digest mismatch (manifest declares "
                f"{manifest.digest}, bytes hash to {actual}); the signed manifest "
                "does not describe these bytes"
            )
        else:
            findings.append(f"artifact digest matches the manifest ({actual})")
        if size != manifest.size_bytes:
            refusals.append(
                f"refusing upload: artifact is {size} bytes, manifest declares "
                f"{manifest.size_bytes}"
            )

    if declared_format == "container":
        if not manifest.image_digest:
            refusals.append(
                "refusing upload: a custom format must ship as a digest-pinned "
                "tenant-supplied container image that loads it inside the "
                "no-egress sandbox; we do not point our own loader at an "
                "unknown format"
            )
        else:
            digests["image"] = manifest.image_digest
            findings.append(
                f"tenant runtime image is digest-pinned ({manifest.image}@{manifest.image_digest})"
            )

    if not sandbox.artifact_scan:
        refusals.append(
            "refusing upload: the sandbox tier disabled artifact_scan, and an "
            "intake gate that does not scan the artifact is not a gate"
        )

    report = SandboxReport(policy=sandbox, findings=tuple(findings))
    identity = _identity(
        {
            "kind": IntakeKind.UPLOAD.value,
            "format": declared_format,
            "declared_by": manifest.declared_by,
            "entrypoint": manifest.entrypoint,
            "digests": dict(digests),
            "policy": report.policy_digest,
            # Verified signer, from the keyring record. A different asserting
            # party is a different execution identity.
            "signer": {
                "tenant": signer.tenant if signer else "",
                "key_id": signer.key_id if signer else "",
            },
        }
    )
    return IntakeResult(
        identity=identity,
        kind=IntakeKind.UPLOAD,
        format=declared_format,
        digests=digests,
        sandbox_report=report,
        passed=not refusals,
        refusals=tuple(refusals),
        entrypoint=manifest.entrypoint,
        image=manifest.image,
        image_digest=manifest.image_digest,
        signer_tenant=signer.tenant if signer else "",
        signer_key_id=signer.key_id if signer else "",
        claimed_key_id=manifest.key_id,
    )


def _host_range_refusal(host: str) -> str | None:
    """Refuse a literal address in a range no tenant endpoint may live in.

    Run *regardless* of allowlist membership: the allowlist is tenant-supplied
    input, so letting an entry vouch for ``169.254.169.254`` would make the
    SSRF check bypassable by the same party it defends against.
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return None
    blocked = {
        "loopback": address.is_loopback,
        "link-local": address.is_link_local,
        "private": address.is_private,
        "multicast": address.is_multicast,
        "reserved": address.is_reserved,
        "unspecified": address.is_unspecified,
    }
    hit = sorted(name for name, flag in blocked.items() if flag)
    if not hit:
        return None
    return (
        f"refusing endpoint host {host}: {', '.join(hit)} address. Probing it "
        "would aim our traffic at infrastructure the tenant does not own — the "
        "cloud metadata service and a customer's clinical network both live in "
        "these ranges — and an allowlist entry cannot vouch for them"
    )


def _split_host(url: str) -> tuple[str, str]:
    """Return ``(scheme, host)`` for a URL, with an empty host if unparseable."""
    parts = urlsplit(url)
    return parts.scheme.lower(), (parts.hostname or "").strip().lower()


def _redirect_refusals(
    response: ProbeResponse,
    *,
    allowlist: EndpointAllowlist,
    index: int,
) -> list[str]:
    refusals: list[str] = []
    for hop in (*response.redirects, response.final_url):
        if not hop:
            continue
        _, host = _split_host(hop)
        if not host:
            refusals.append(f"refusing probe {index}: unparseable redirect target {hop!r}")
            continue
        range_refusal = _host_range_refusal(host)
        if range_refusal is not None:
            refusals.append(f"probe {index} redirect: {range_refusal}")
        elif not allowlist.permits(host):
            refusals.append(
                f"refusing probe {index}: endpoint redirected to off-allowlist "
                f"host {host!r}; a redirect is the endpoint choosing our "
                "destination, which is the SSRF the allowlist exists to stop"
            )
    return refusals


def intake_endpoint(
    descriptor: EndpointDescriptor,
    *,
    allowlist: EndpointAllowlist,
    probe: EndpointProbe,
    policy: ProbePolicy | None = None,
) -> IntakeResult:
    """Gate a tenant-registered endpoint from the sandbox tier.

    Allowlisted hosts only; TLS only; no loopback, link-local, private,
    multicast, or reserved literal; no off-allowlist redirect; and at most
    ``descriptor.probe_budget`` probes carrying synthetic inputs. ``probe`` is
    injected — this module performs no network I/O, so the transport (and its
    tier) is the caller's, and the report records that probing ran from the
    sandbox rather than the control plane.
    """
    refusals: list[str] = []
    scheme, host = _split_host(descriptor.url)
    probe_policy = policy or ProbePolicy(
        allowed_hosts=tuple(allowlist.hosts),
        max_probes=descriptor.probe_budget,
    )
    findings: list[str] = [
        "probing ran from the sandbox tier, never from the control plane",
        f"probe budget {probe_policy.max_probes}, synthetic inputs only",
    ]

    if scheme != "https":
        refusals.append(
            f"refusing endpoint {descriptor.id}: scheme {scheme or '(none)'} is not "
            "https; a plaintext hop exposes the tenant's model traffic and lets "
            "a network position rewrite our destination"
        )
    if not host:
        refusals.append(f"refusing endpoint {descriptor.id}: {descriptor.url!r} names no host")

    range_refusal = _host_range_refusal(host) if host else None
    if range_refusal is not None:
        refusals.append(range_refusal)
    elif host and not allowlist.permits(host):
        refusals.append(
            f"refusing endpoint host {host!r}: not on the tenant allowlist "
            f"[{', '.join(allowlist.hosts)}]. A name we did not register is a "
            "destination we cannot vouch for, and DNS is not a trust boundary"
        )

    probes = 0
    if not refusals:
        for index in range(probe_policy.max_probes):
            request = ProbeRequest(url=descriptor.url, index=index)
            try:
                response = probe(request)
            except Exception as exc:  # transport failure is a finding, not a crash
                refusals.append(f"refusing endpoint {descriptor.id}: probe {index} failed: {exc}")
                break
            probes += 1
            refusals.extend(_redirect_refusals(response, allowlist=allowlist, index=index))
            if response.status_code >= 400:
                refusals.append(
                    f"refusing endpoint {descriptor.id}: probe {index} answered "
                    f"HTTP {response.status_code}; an endpoint that cannot serve a "
                    "synthetic schema probe cannot be assessed"
                )
            if refusals:
                break
            findings.append(
                f"probe {index}: HTTP {response.status_code} from {response.final_url} "
                f"with keys [{', '.join(response.schema_keys) or 'none'}]"
            )
            if response.schema_keys:
                break
    findings.append(f"probes performed: {probes} of at most {probe_policy.max_probes}")

    report = SandboxReport(
        probe_policy=probe_policy,
        probes=probes,
        findings=tuple(findings),
    )
    identity = _identity(
        {
            "kind": IntakeKind.ENDPOINT.value,
            "endpoint": descriptor.id,
            "url": descriptor.url,
            "model": descriptor.model,
            "declared_by": descriptor.declared_by,
            "policy": report.policy_digest,
        }
    )
    return IntakeResult(
        identity=identity,
        kind=IntakeKind.ENDPOINT,
        format="endpoint",
        digests={},
        sandbox_report=report,
        passed=not refusals,
        refusals=tuple(refusals),
        base_url=descriptor.url,
        model=descriptor.model,
    )


def runtime_descriptor(
    result: IntakeResult,
    *,
    timeout_sec: float = 120.0,
) -> RuntimeDescriptor:
    """Convert an intake-*passed* result into a pinned runtime descriptor.

    The only bridge out of intake. A failed result raises, so a refused upload
    has no executable representation. The descriptor populates the kernel's
    ``intake_identity`` and ``sandbox_policy_digest`` fields, both covered by
    ``RuntimeDescriptor.identity`` — so the eventual scorecard's execution
    identity changes when the gate, the signer, or the isolation tier changes,
    and an ``AgentPackage`` round-trip cannot drop them.
    """
    if not result.passed:
        detail = "; ".join(result.refusals)
        raise TaskContractError(
            f"refusing to build a runtime descriptor from failed intake {result.identity}: {detail}"
        )
    intake_identity = result.identity
    policy_digest = result.sandbox_report.policy_digest
    if result.kind is IntakeKind.ENDPOINT:
        return RuntimeDescriptor(
            kind=RuntimeKind.OPENAI_COMPATIBLE,
            model=result.model,
            base_url=result.base_url,
            timeout_sec=timeout_sec,
            intake_identity=intake_identity,
            sandbox_policy_digest=policy_digest,
        )
    if result.image_digest:
        return RuntimeDescriptor(
            kind=RuntimeKind.CONTAINER,
            image=result.image,
            image_digest=result.image_digest,
            entrypoint=result.entrypoint,
            timeout_sec=timeout_sec,
            intake_identity=intake_identity,
            sandbox_policy_digest=policy_digest,
        )
    return RuntimeDescriptor(
        kind=RuntimeKind.LOCAL,
        entrypoint=result.entrypoint,
        timeout_sec=timeout_sec,
        intake_identity=intake_identity,
        sandbox_policy_digest=policy_digest,
    )
