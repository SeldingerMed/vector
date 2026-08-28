"""Hosted agentic concierge: intake -> assess -> select -> adapt.

An evaluation agent on top of the open rails, not a dashboard and not a
feature. The four capabilities ship in order, and capability 0 is a security
gate rather than a convenience:

``intake``
    Uploaded weights and linked endpoints are hostile until proven otherwise.
    Nothing here deserializes an upload, and no probe leaves the sandbox tier.
``assess``
    The agent probes an intake-passed model and *drafts* a capability. The
    kernel keeps authority: ``or_audit.eval.bind.assert_bind`` decides
    satisfaction, and a named party confirms before the first scored run.
``select``
    Confirmed capabilities map onto the catalog by interface satisfaction
    first, and results are narrated with gates and abstention first and no
    composite anywhere.
``adapt``
    The agent searches a task's declared scenario space toward the hardest
    honest test, then freezes any adaptation into a new versioned,
    digest-pinned package that is quarantined from public leaderboards and
    whose verifier is byte-identical to its parent's.
"""

from __future__ import annotations

from or_audit.concierge.adapt import (
    AdaptBudget,
    AdaptCandidate,
    AdaptObservation,
    AdaptReport,
    AdaptRow,
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
    ModelProbe,
    ModelProbeReport,
    assert_confirmed,
    assess_model,
    confirm,
)
from or_audit.concierge.intake import (
    MIN_SIGNING_SECRET_BYTES,
    UPLOAD_FORMAT_ALLOWLIST,
    EndpointAllowlist,
    EndpointDescriptor,
    EndpointProbe,
    IntakeKind,
    IntakeResult,
    ProbePolicy,
    ProbeRequest,
    ProbeResponse,
    SandboxPolicy,
    SandboxReport,
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
    EvalPlan,
    PlanEntry,
    difficulty_rank,
    modality_rank,
    narrate_plan,
    narrate_results,
    phi_rank,
    select_eval_plan,
)

__all__ = [
    "MIN_SIGNING_SECRET_BYTES",
    "NO_COMPOSITE_FOOTER",
    "UPLOAD_FORMAT_ALLOWLIST",
    "AdaptBudget",
    "AdaptCandidate",
    "AdaptObservation",
    "AdaptReport",
    "AdaptRow",
    "AssessProposal",
    "ConfirmedCapability",
    "EndpointAllowlist",
    "EndpointDescriptor",
    "EndpointProbe",
    "EvalBudget",
    "EvalPlan",
    "FrozenPackage",
    "IntakeKind",
    "IntakeResult",
    "ModelProbe",
    "ModelProbeReport",
    "PlanEntry",
    "ProbePolicy",
    "ProbeRequest",
    "ProbeResponse",
    "SandboxPolicy",
    "SandboxReport",
    "ScenarioSpace",
    "SigningKey",
    "TenantKeyring",
    "UploadManifest",
    "assert_confirmed",
    "assert_frozen_before_scoring",
    "assert_verifier_untouched",
    "assess_model",
    "confirm",
    "difficulty_rank",
    "freeze_adapted_package",
    "intake_endpoint",
    "intake_upload",
    "manifest_signature",
    "modality_rank",
    "narrate_plan",
    "narrate_results",
    "phi_rank",
    "runtime_descriptor",
    "search_scenarios",
    "select_eval_plan",
    "sign_manifest",
    "signed_payload",
    "verify_manifest_signature",
]
