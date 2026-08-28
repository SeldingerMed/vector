"""v0.3 contract validators that were unhit after the framework landed."""

from __future__ import annotations

import pytest

from or_audit.errors import TaskContractError
from or_audit.eval.contracts import (
    CapabilitySpec,
    InteractionMode,
    InterfaceSpec,
    RuntimeDescriptor,
    RuntimeKind,
    legacy_capability,
    legacy_interface,
)
from or_audit.eval.vector import MetricOutcome


def test_closed_loop_interface_requires_an_action() -> None:
    with pytest.raises(TaskContractError, match="must declare an action"):
        InterfaceSpec(id="gym-policy", interaction_mode=InteractionMode.CLOSED_LOOP)


def test_single_turn_interface_requires_output() -> None:
    with pytest.raises(TaskContractError, match="needs output"):
        InterfaceSpec(id="video-predict", interaction_mode=InteractionMode.SINGLE_TURN)


def test_capability_satisfies_declared_schemas() -> None:
    interface = InterfaceSpec(
        id="gym-policy",
        interaction_mode=InteractionMode.CLOSED_LOOP,
        observations=("gym-obs",),
        actions=("insertion_twist",),
        features=("safety",),
    )
    matching = CapabilitySpec(
        interface="gym-policy",
        interaction_modes=(InteractionMode.CLOSED_LOOP,),
        observations=("gym-obs",),
        actions=("insertion_twist",),
        features=("safety",),
    )
    assert matching.satisfies(interface)
    missing_feature = CapabilitySpec(
        interface="gym-policy",
        interaction_modes=(InteractionMode.CLOSED_LOOP,),
        observations=("gym-obs",),
        actions=("insertion_twist",),
    )
    assert not missing_feature.satisfies(interface)


def test_capability_requires_an_interaction_mode() -> None:
    with pytest.raises(TaskContractError, match="no interaction mode"):
        CapabilitySpec(interface="gym-policy", interaction_modes=())


def test_runtime_identity_validators() -> None:
    with pytest.raises(TaskContractError, match="command or entrypoint"):
        RuntimeDescriptor(kind=RuntimeKind.LOCAL)
    with pytest.raises(TaskContractError, match="model and revision"):
        RuntimeDescriptor(kind=RuntimeKind.HUGGINGFACE, model="org/name")
    with pytest.raises(TaskContractError, match="model and base_url"):
        RuntimeDescriptor(kind=RuntimeKind.OPENAI_COMPATIBLE, model="gpt")
    with pytest.raises(TaskContractError, match="image and image_digest"):
        RuntimeDescriptor(kind=RuntimeKind.CONTAINER, image="img")
    pinned = RuntimeDescriptor(
        kind=RuntimeKind.CONTAINER,
        image="ghcr.io/example/agent",
        image_digest="sha256:abc",
    )
    assert len(pinned.identity) == 64
    hf = RuntimeDescriptor(kind=RuntimeKind.HUGGINGFACE, model="org/model", revision="abc123")
    api = RuntimeDescriptor(
        kind=RuntimeKind.OPENAI_COMPATIBLE,
        model="local-model",
        base_url="http://127.0.0.1:8000/v1",
    )
    local = RuntimeDescriptor(kind=RuntimeKind.LOCAL, entrypoint="policy.py:load_policy")
    trusted = RuntimeDescriptor(
        kind=RuntimeKind.TRUSTED_IN_PROCESS,
        command=("python", "-m", "or_audit.eval.plugin_host"),
    )
    assert hf.model == "org/model"
    assert api.base_url.endswith("/v1")
    assert local.entrypoint
    assert trusted.command


def test_legacy_port_translation() -> None:
    video = legacy_interface({"id": "video-predict", "prediction": "next-step"})
    assert video.interaction_mode is InteractionMode.SINGLE_TURN
    assert video.outputs == ("next-step",)
    gym = legacy_interface({"id": "gym-policy"})
    assert gym.actions == ("continuous-action",)
    with pytest.raises(TaskContractError, match="unknown legacy port"):
        legacy_interface({"id": "not-a-port"})
    assert legacy_capability("video-predict").schema_wildcard is True
    assert legacy_capability("gym-policy").interface == "gym-policy"
    with pytest.raises(TaskContractError, match="unknown legacy port"):
        legacy_capability("not-a-port")


def test_metric_outcome_infers_legacy_kind_and_rejects_mismatches() -> None:
    assert MetricOutcome(id="flag", value=True).kind is not None
    assert MetricOutcome(id="label", value="safe").kind is not None
    assert MetricOutcome(id="pen", value=0.2).kind is not None
    assert MetricOutcome(id="unknown", value=None).value is None
    with pytest.raises(TaskContractError, match="boolean metric"):
        MetricOutcome(id="flag", value="no", kind="boolean")
    with pytest.raises(TaskContractError, match="continuous metric"):
        MetricOutcome(id="pen", value="far", kind="continuous")
    with pytest.raises(TaskContractError, match="categorical metric"):
        MetricOutcome(id="label", value=True, kind="categorical")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_metric_value_is_refused_on_the_published_record(value: float) -> None:
    """A diverged solver's number is not a measurement, and must not become one.

    Original defect: ``MetricOutcome`` accepted ``float('nan')``, so a verifier
    could return one, pass validation, and then blow up unhandled inside
    ``compute_head`` - ``canonical_json`` cannot render a non-finite float. The
    canonicalizer was the only thing standing in the way and it was not a
    contract, so the operator got a canonicalization error from deep in the
    digest code instead of a statement about their metric.
    ``score_context`` records null (unassessable) instead; reaching this
    validator means a construction path that bypassed it.
    """
    with pytest.raises(TaskContractError, match="is not a measurement"):
        MetricOutcome(id="force", value=value, kind="continuous")
    # The honest record for the same evidence still constructs.
    assert MetricOutcome(id="force", value=None, kind="continuous").value is None


def test_every_non_finite_entry_point_into_the_job_head_is_refused() -> None:
    """The digest layer's refusal is a backstop, not the contract.

    ``canonical_json`` cannot render ``nan``/``inf``, so every one of these
    used to abort head computation with a ValueError from three frames down.
    Two fixes have now depended on that refusal by accident; these are the
    remaining fields that reach a canonical digest from world, verifier, or
    task input.
    """
    from or_audit.eval.task import ProjectionSpec
    from or_audit.eval.vector import GateOutcome, TrialVector, project

    # `score_context` clamps confidence with min(max(c, 0.0), 1.0), which does
    # not catch nan: every comparison against nan is False.
    assert min(max(float("nan"), 0.0), 1.0) != min(max(float("nan"), 0.0), 1.0)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(TaskContractError, match="is not a number"):
            GateOutcome(id="g", status="pass", confidence=bad)
    assert GateOutcome(id="g", status="pass", confidence=None).confidence is None

    # TOML spells `inf` and `nan`, so a declared reward can be non-finite.
    vector = TrialVector(
        task_id="t",
        task_version="0",
        agent_identity="a",
        seed=0,
        gates=(),
        metrics=(MetricOutcome(id="raw_success", value=True, headline=True),),
    )
    with pytest.raises(TaskContractError, match="a reward must be a finite number"):
        project(
            vector, ProjectionSpec(id="p", source_metric="raw_success", true_value=float("inf"))
        )
    with pytest.raises(TaskContractError, match="a reward must be a finite number"):
        project(
            vector,
            ProjectionSpec(id="p", source_metric="raw_success", false_value=float("nan")),
        )
    finite = ProjectionSpec(id="p", source_metric="raw_success", require_false_metrics=())
    assert project(vector, finite) == 1.0
