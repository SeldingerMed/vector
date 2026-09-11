"""Phase F3: declared branching capability, measured-fork proof, honest refusal.

Every environment here is a deterministic scripted fake — no external stack —
so the probes' verdicts are reproducible on CI. The suite pins the four
contract rules:

* ``measure_branch_support`` earns ``measured_fork`` only from byte-equal twin
  rollouts, ``prefix_replay`` only from within-tolerance agreement, and
  ``not_supported`` whenever the twin pair diverges;
* ``branch_from`` verifies every arm's prefix segment against the reference
  replay before comparing anything, and a world that cannot replay its own
  prefix is refused outright with **no candidate comparisons reported** —
  the refusal, not the comparison, is the deliverable;
* ``require_branch_support`` fails closed: a bare ``declared`` claim may drive
  arms but can never certify equality, and ``not_supported``/``unspecified``
  are states a world reports, not mechanisms a caller may require;
* ``WorldCapabilities.branching`` rides the same task-vs-adapter cross-check
  as every other gate, so a task cannot grant itself a fork mechanism its
  installed adapter withholds.
"""

from __future__ import annotations

import json
import math
from typing import Any, ClassVar, cast

import pytest

from or_audit.errors import TaskContractError
from or_audit.eval import worlds as _worlds
from or_audit.eval.branching import (
    DRIVING_MECHANISM,
    FORK_EQUALITY_MECHANISM,
    BranchEvidence,
    branch_from,
    measure_branch_support,
    require_branch_support,
)
from or_audit.eval.enums import WorldKind
from or_audit.eval.worlds import (
    BUILTIN_WORLD_CAPABILITIES,
    BranchingSupport,
    WorldCapabilities,
    WorldKindSpec,
    branching_at_least,
    register_world_kind,
    resolve_world_capabilities,
)


def _score(history: list[tuple[float, ...]]) -> float:
    total = 0.0
    for i, action in enumerate(history):
        for k, value in enumerate(action):
            total += (i + 1) * (k + 1) * value
    return total


class DeterministicEnv:
    """Info is a pure function of (seed, action history): replayable exactly."""

    def __init__(self) -> None:
        self.history: list[tuple[float, ...]] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.seed = seed
        self.history = []
        return {"x": float(seed or 0)}, {"origin": seed}

    def step(self, action):
        self.history.append(tuple(float(v) for v in action))
        value = _score(self.history)
        return {"x": value}, value * 0.1, False, False, {"h": value}

    def close(self) -> None:
        pass


class WobblyEnv(DeterministicEnv):
    """Reset draws hidden state from a global counter: a seed does NOT restore it."""

    counter: ClassVar[list[int]] = [0]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        WobblyEnv.counter[0] += 1
        self.bias = float(WobblyEnv.counter[0])
        self.history = []
        return {"x": self.bias}, {"origin": self.bias}

    def step(self, action):
        self.history.append(tuple(float(v) for v in action))
        value = self.bias + _score(self.history)
        return {"x": value}, value * 0.1, False, False, {"h": value}


class JitterEnv(DeterministicEnv):
    """Twins differ by 1e-9 only: within any declared tolerance, never byte-equal."""

    parity: ClassVar[list[int]] = [0]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        JitterEnv.parity[0] += 1
        self.bias = 1e-9 * (JitterEnv.parity[0] % 2)
        self.history = []
        return {"x": 0.0}, {"origin": 0}

    def step(self, action):
        self.history.append(tuple(float(v) for v in action))
        value = self.bias + _score(self.history)
        return {"x": value}, value * 0.1, False, False, {"h": value}


class ThirdInstanceDriftEnv(DeterministicEnv):
    """Instances 1-2 replay the prefix; the third adds a hidden bias during it.

    A scripted way to make the *candidate* arm's prefix disagree with the
    reference pair while the reference pair itself agrees — which is exactly
    the situation the per-arm prefix check exists to catch.
    """

    constructed: ClassVar[list[int]] = [0]

    def __init__(self) -> None:
        ThirdInstanceDriftEnv.constructed[0] += 1
        self.bias = 1.0 if ThirdInstanceDriftEnv.constructed[0] >= 3 else 0.0
        self.history = []

    def step(self, action):
        self.history.append(tuple(float(v) for v in action))
        value = self.bias + _score(self.history)
        return {"x": value}, value * 0.1, False, False, {"h": value}


class ObsJitterEnv(DeterministicEnv):
    """Twins differ only in the *observation* channel: reward and info are fixed.

    The observation-only sibling of ``JitterEnv``: an engine whose
    nondeterminism is visible nowhere but in the bytes the policy actually
    conditions on. If a probe measured only reward/info it would certify a
    fork this world cannot provide.
    """

    counter: ClassVar[list[int]] = [0]

    def step(self, action):
        self.history.append(tuple(float(v) for v in action))
        ObsJitterEnv.counter[0] += 1
        return (
            {"x": 1e-9 * ObsJitterEnv.counter[0]},
            0.5,
            False,
            False,
            {"h": 0.0},
        )


class ObsBlindProbeEnv(DeterministicEnv):
    """Identical reward/info streams; the observation scales by a class knob.

    Two instances configured with different multipliers are, from the
    reward/info side, the *same world*: any difference in what the policy can
    see lives in the observation. If arm digests were blind to observations,
    these two runs would digest identically and a probe would be certifying
    equality of worlds the policy can tell apart.
    """

    multiplier: ClassVar[float] = 1.0

    def step(self, action):
        self.history.append(tuple(float(v) for v in action))
        value = _score(self.history)
        return {"x": self.multiplier * value}, 0.5, False, False, {"h": 0.0}


PREFIX: list[list[float]] = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
CAND_A: list[list[float]] = [[0.9, 0.9]]
CAND_B: list[list[float]] = [[0.1, 0.1]]


# --------------------------------------------------------------------------
# measure_branch_support: the ladder is earned, never granted
# --------------------------------------------------------------------------


def test_byte_identical_twins_earn_measured_fork() -> None:
    evidence = measure_branch_support(DeterministicEnv, seed=7, prefix_actions=PREFIX)
    assert evidence.support is BranchingSupport.MEASURED_FORK
    assert evidence.prefix_equal
    assert evidence.identical_fork_digests
    assert evidence.max_float_delta == 0.0
    assert evidence.first_difference == ""


def test_hidden_state_at_reset_defeats_the_fork() -> None:
    """A seed that does not restore hidden state must not certify a fork.

    The wobbly env's info differs from the first record onward, so the probe
    must report ``not_supported`` and say why in ``first_difference``.
    """
    evidence = measure_branch_support(WobblyEnv, seed=7, prefix_actions=PREFIX)
    assert evidence.support is BranchingSupport.NOT_SUPPORTED
    assert not evidence.prefix_equal
    assert not evidence.identical_fork_digests
    assert "seed" in evidence.first_difference


def test_observation_only_nondeterminism_cannot_certify_a_fork() -> None:
    """The false-certification hole: noise hidden in the observation channel.

    Reward and info are constants here, so a probe that ignored the
    observation would see byte-equal traces and hand out ``measured_fork`` to
    a world whose policy input differs between twins run to run. With the
    observation recorded, the digest mismatch is caught; and even under a
    generous tolerance the world stops at ``prefix_replay`` — the noisy rung
    never earns the equality-certifying rung.
    """
    ObsJitterEnv.counter[0] = 0
    strict = measure_branch_support(ObsJitterEnv, seed=3, prefix_actions=PREFIX[:2])
    assert strict.support is BranchingSupport.NOT_SUPPORTED
    assert not strict.identical_fork_digests
    assert strict.max_float_delta == pytest.approx(2e-9, abs=1e-12)
    assert "diverged in observation" in strict.first_difference

    ObsJitterEnv.counter[0] = 0
    lenient = measure_branch_support(
        ObsJitterEnv, seed=3, prefix_actions=PREFIX[:2], tolerance=1e-6
    )
    assert lenient.support is BranchingSupport.PREFIX_REPLAY
    assert not lenient.identical_fork_digests


def test_within_tolerance_is_prefix_replay_not_measured_fork() -> None:
    """Float-noise engines reach the middle rung only, and only when they say so.

    With no tolerance the 1e-9 jitter is a divergence (byte equality is the
    price of the top rung, so nothing excuses it silently); with an explicit
    tolerance the probe records a faithful-but-noisy replay and still withholds
    the equality-certifying rung.
    """
    strict = measure_branch_support(JitterEnv, seed=1, prefix_actions=PREFIX[:2])
    assert strict.support is BranchingSupport.NOT_SUPPORTED
    assert strict.max_float_delta == pytest.approx(1e-9, abs=1e-12)

    lenient = measure_branch_support(JitterEnv, seed=1, prefix_actions=PREFIX[:2], tolerance=1e-6)
    assert lenient.support is BranchingSupport.PREFIX_REPLAY
    assert lenient.prefix_equal
    assert not lenient.identical_fork_digests


def test_empty_prefix_still_proves_the_rung_it_claims() -> None:
    """Zero shared steps is a legal degenerate branch point (step 0).

    The twin reset record is inside the trace, so ``measured_fork`` still
    requires the two instances to agree on their reset information.
    """
    evidence = measure_branch_support(DeterministicEnv, seed=4, prefix_actions=[])
    assert evidence.support is BranchingSupport.MEASURED_FORK
    wobbly = measure_branch_support(WobblyEnv, seed=4, prefix_actions=[])
    assert wobbly.support is BranchingSupport.NOT_SUPPORTED


def test_measurement_outranks_a_loud_declaration() -> None:
    """``claimed`` records the strongest standing claim so callers see the gap.

    A capability block claiming ``measured_fork`` is contradicted by a world
    whose twins diverge: the evidence must keep ``claimed`` at the claim and
    ``support`` at the measurement.
    """
    loud = WorldCapabilities(
        physics=True, closed_loop=True, branching=BranchingSupport.MEASURED_FORK
    )
    evidence = measure_branch_support(WobblyEnv, seed=9, prefix_actions=PREFIX, capabilities=loud)
    assert evidence.support is BranchingSupport.NOT_SUPPORTED
    assert evidence.claimed is BranchingSupport.MEASURED_FORK


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -0.5])
def test_tolerance_must_be_a_real_number(bad: float) -> None:
    with pytest.raises(TaskContractError):
        measure_branch_support(DeterministicEnv, seed=1, prefix_actions=PREFIX, tolerance=bad)


def test_failing_env_factory_is_a_contract_error_not_a_crash() -> None:
    def broken() -> DeterministicEnv:
        raise RuntimeError("driver gone")

    with pytest.raises(TaskContractError):
        measure_branch_support(broken, seed=1, prefix_actions=PREFIX)


# --------------------------------------------------------------------------
# branch_from: compare from a common state, or refuse
# --------------------------------------------------------------------------


def test_branches_diverge_from_a_shared_prefix() -> None:
    result = branch_from(
        DeterministicEnv, seed=7, prefix_actions=PREFIX, candidates=[CAND_A, CAND_B]
    )
    assert result["supported"]
    assert result["refused"] == ""
    assert result["evidence"].candidate_diverged
    assert len({arm["digest"] for arm in result["arms"]}) == 2
    assert all(arm["prefix_matches_reference"] and arm["replayed_equal"] for arm in result["arms"])
    # the arms really share the prefix: same plan starts, different endings
    assert result["arms"][0]["plan"][: len(PREFIX)] == result["arms"][1]["plan"][: len(PREFIX)]


def test_a_world_that_cannot_replay_its_prefix_is_refused_not_compared() -> None:
    """The core F3 refusal: no candidate verdicts are emitted at all.

    An honest answer on a world that cannot fork is "I cannot tell you", so the
    result must carry zero arm digests — reporting comparisons from states that
    already diverged before the branch point would be fabricated evidence.
    """
    result = branch_from(WobblyEnv, seed=3, prefix_actions=PREFIX, candidates=[CAND_A, CAND_B])
    assert not result["supported"]
    assert result["arms"] == []
    assert result["evidence"].support is BranchingSupport.NOT_SUPPORTED
    assert not result["evidence"].candidate_diverged
    assert "seed alone does not restore" in result["refused"]


def test_candidate_arm_whose_prefix_drifts_is_refused() -> None:
    """Fork pair agrees, but the candidate replays the prefix differently.

    The per-arm prefix check must catch drift that the fork-pair comparison
    structurally cannot see (it only sees the reference pair), and the whole
    branch set is refused rather than reporting one drifting arm's number.
    """
    ThirdInstanceDriftEnv.constructed[0] = 0
    result = branch_from(
        ThirdInstanceDriftEnv, seed=2, prefix_actions=PREFIX, candidates=[CAND_A, CAND_B]
    )
    assert not result["supported"]
    assert result["arms"] == []
    assert result["evidence"].support is BranchingSupport.NOT_SUPPORTED


def test_identical_candidate_plans_measure_nothing() -> None:
    """Distinct-but-equal plans that all digest the same mean the intervention
    changed nothing: ``candidate_diverged`` stays False, so a caller cannot
    publish a "counterfactual" that never compared anything."""
    result = branch_from(
        DeterministicEnv,
        seed=7,
        prefix_actions=PREFIX,
        candidates=[[[0.2, 0.2]], [[0.2, 0.2]]],
    )
    assert result["supported"]
    assert not result["evidence"].candidate_diverged


def test_arm_digests_are_sensitive_to_the_observation_channel() -> None:
    """Same plan, two worlds distinguishable only through the observation.

    Reward and info are constant and action-independent in both variants, so
    an observation-blind digest would hash the two arms the same and certify
    cross-world equality the policy could refute. The digests must therefore
    track what the policy actually conditions on.
    """
    ObsBlindProbeEnv.multiplier = 1.0
    plain = branch_from(ObsBlindProbeEnv, seed=7, prefix_actions=PREFIX, candidates=[CAND_A])
    ObsBlindProbeEnv.multiplier = 4.0
    scaled = branch_from(ObsBlindProbeEnv, seed=7, prefix_actions=PREFIX, candidates=[CAND_A])
    assert plain["supported"]
    assert scaled["supported"]
    assert plain["arms"][0]["digest"] != scaled["arms"][0]["digest"]


def test_observation_only_replay_drift_is_refused_not_certified() -> None:
    """The false-certification hole on the arm side of ``branch_from``.

    Each replay pass of a candidate plan jitters only the observation (reward
    and info are constants). If the trace ignored observations, the two passes
    would digest byte-equal, every arm would report ``replayed_equal``, and
    the branch set would be certified ``measured_fork`` although the policy
    input differs run to run. With the observation in the digest, the arm's
    prefix fails the byte-strict replay against the reference and the whole
    comparison is refused with zero arms — the only honest outcome.
    """
    ObsJitterEnv.counter[0] = 0
    result = branch_from(
        ObsJitterEnv, seed=7, prefix_actions=PREFIX, candidates=[CAND_A], tolerance=1e-6
    )
    assert not result["supported"]
    assert result["arms"] == []
    assert result["evidence"].support is BranchingSupport.NOT_SUPPORTED
    assert "replay" in result["refused"]


def test_arm_traces_are_json_safe() -> None:
    result = branch_from(DeterministicEnv, seed=7, prefix_actions=PREFIX, candidates=[CAND_A])
    json.dumps(result["arms"])  # must not raise
    assert result["arms"][0]["plan"] == [*PREFIX, [pytest.approx(0.9), pytest.approx(0.9)]]


def test_numpy_actions_digest_by_content_not_identity() -> None:
    """Array actions must not split a digest by object identity.

    Two arms whose actions differ only in *representation* (array vs list of
    the same numbers) have to digest equal, or an intervention encoded one way
    would look like a divergence and the other way would hide one.
    """
    pytest.importorskip("numpy")
    import numpy as np

    with_array = branch_from(
        DeterministicEnv, seed=5, prefix_actions=PREFIX, candidates=[[np.array([0.7, 0.8])]]
    )
    with_list = branch_from(
        DeterministicEnv, seed=5, prefix_actions=PREFIX, candidates=[[[0.7, 0.8]]]
    )
    assert with_array["arms"][0]["digest"] == with_list["arms"][0]["digest"]


def test_branch_from_rejects_unusable_inputs() -> None:
    with pytest.raises(TaskContractError):
        branch_from(DeterministicEnv, seed=1, prefix_actions=PREFIX, candidates=[])
    with pytest.raises(TaskContractError):
        branch_from(
            DeterministicEnv, seed=1, prefix_actions=PREFIX, candidates=CAND_A, tolerance=-1.0
        )
    with pytest.raises(TaskContractError):
        branch_from(cast(Any, "not-a-factory"), seed=1, prefix_actions=PREFIX, candidates=CAND_A)


# --------------------------------------------------------------------------
# require_branch_support: fail closed
# --------------------------------------------------------------------------


def _caps(branching: BranchingSupport, **flags: bool) -> WorldCapabilities:
    return WorldCapabilities(physics=True, closed_loop=True, branching=branching, **flags)


def test_no_capability_source_is_a_refusal() -> None:
    with pytest.raises(TaskContractError, match="capability declarations"):
        require_branch_support(None, DRIVING_MECHANISM)


def test_unspecified_support_refuses_every_mechanism() -> None:
    """The default must be the safe one: nothing declared, nothing permitted."""
    for mechanism in (DRIVING_MECHANISM, FORK_EQUALITY_MECHANISM):
        with pytest.raises(TaskContractError):
            require_branch_support(_caps(BranchingSupport.UNSPECIFIED), mechanism)


def test_a_bare_declaration_can_drive_but_never_certify() -> None:
    declared = _caps(BranchingSupport.DECLARED)
    require_branch_support(declared, DRIVING_MECHANISM)  # must not raise
    with pytest.raises(TaskContractError, match="measured evidence"):
        require_branch_support(declared, FORK_EQUALITY_MECHANISM)


def test_measured_backing_is_the_only_route_to_certification() -> None:
    require_branch_support(_caps(BranchingSupport.MEASURED_FORK), FORK_EQUALITY_MECHANISM)
    # prefix_replay drives but does not certify: byte equality was never observed
    replay = _caps(BranchingSupport.PREFIX_REPLAY)
    require_branch_support(replay, DRIVING_MECHANISM)
    with pytest.raises(TaskContractError):
        require_branch_support(replay, FORK_EQUALITY_MECHANISM)


@pytest.mark.parametrize("state", [BranchingSupport.NOT_SUPPORTED, BranchingSupport.UNSPECIFIED])
def test_reported_states_are_not_requestable(state: BranchingSupport) -> None:
    """``not_supported``/``unspecified`` are things a world says, not asks."""
    with pytest.raises(TaskContractError, match="not a branching mechanism"):
        require_branch_support(_caps(BranchingSupport.MEASURED_FORK), state)


def test_strength_order_is_total_and_measured_is_strongest() -> None:
    order = [
        BranchingSupport.NOT_SUPPORTED,
        BranchingSupport.UNSPECIFIED,
        BranchingSupport.DECLARED,
        BranchingSupport.PREFIX_REPLAY,
        BranchingSupport.MEASURED_FORK,
    ]
    for i, weaker in enumerate(order):
        for j, stronger in enumerate(order):
            assert branching_at_least(stronger, weaker) is (j >= i)


# --------------------------------------------------------------------------
# BranchEvidence guards
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_evidence_rejects_unusable_tolerance(bad: float) -> None:
    with pytest.raises(TaskContractError):
        BranchEvidence(
            support=BranchingSupport.PREFIX_REPLAY,
            prefix_equal=True,
            identical_fork_digests=False,
            candidate_diverged=False,
            tolerance=bad,
        )


def test_evidence_refuses_an_unbacked_measured_fork() -> None:
    """``measured_fork`` without the two digest witnesses is a fabrication."""
    with pytest.raises(TaskContractError, match="earned from two"):
        BranchEvidence(
            support=BranchingSupport.MEASURED_FORK,
            prefix_equal=True,
            identical_fork_digests=False,
            candidate_diverged=False,
        )
    with pytest.raises(TaskContractError, match="earned from two"):
        BranchEvidence(
            support=BranchingSupport.MEASURED_FORK,
            prefix_equal=False,
            identical_fork_digests=True,
            candidate_diverged=True,
        )


# --------------------------------------------------------------------------
# Capability surface: declaration cross-checked against the adapter
# --------------------------------------------------------------------------


def test_branching_rides_the_task_versus_adapter_crosscheck() -> None:
    """A task may not grant itself a fork mechanism the adapter withholds.

    ``gates()`` includes the branching promotion flag, so a task declaring
    ``measured_fork`` against an adapter that declared nothing must be refused
    by ``resolve_world_capabilities`` — the same positional cross-check that
    already guards physics/closed-loop overstating.
    """
    try:
        register_world_kind(
            WorldKindSpec(
                kind="f3-probe-world",
                capabilities=WorldCapabilities(physics=True, closed_loop=True),
                provider="tests",
            )
        )
        greedy = WorldCapabilities(
            physics=True, closed_loop=True, branching=BranchingSupport.MEASURED_FORK
        )
        with pytest.raises(TaskContractError, match="disagrees with the installed"):
            resolve_world_capabilities("f3-probe-world", greedy)
        # under-claiming stays legal: declining a mechanism you have is safe
        resolved = resolve_world_capabilities(
            "f3-probe-world", WorldCapabilities(physics=True, closed_loop=True)
        )
        assert resolved.branching is BranchingSupport.UNSPECIFIED
    finally:
        # Remove only what this test added. ``reset_default_world_kinds()`` would
        # wipe the entry-point-discovered adapter identities that sibling suites
        # (test_world_kinds.py) assert, turning an isolated probe into cross-file
        # contamination.
        _worlds._WORLD_KIND_REGISTRY.pop("f3-probe-world", None)


def test_branch_claims_require_a_stepped_world() -> None:
    """``closed_loop=False`` means the runner cannot step the world at all,
    so there is no trajectory to fork — the claim is refused at construction."""
    for mechanism in (
        BranchingSupport.DECLARED,
        BranchingSupport.PREFIX_REPLAY,
        BranchingSupport.MEASURED_FORK,
    ):
        with pytest.raises(TaskContractError, match="closed_loop"):
            WorldCapabilities(physics=True, closed_loop=False, branching=mechanism)


def test_measured_fork_claim_requires_physics() -> None:
    """A fork of a traceless synthetic stepper certifies nothing about dynamics;
    pairing the top rung with non-physics state would let it ride into the
    physics-oracle gate for free."""
    with pytest.raises(TaskContractError, match="physics"):
        WorldCapabilities(closed_loop=True, physics=False, branching=BranchingSupport.MEASURED_FORK)
    # declarations (no measurement claimed) remain allowed on non-physics worlds
    WorldCapabilities(closed_loop=True, physics=False, branching=BranchingSupport.DECLARED)


def test_lumen_gym_declares_prefix_replay_and_nothing_louder() -> None:
    """The one built-in claim is the one the pinned stack demonstrates
    (tests/test_lumen_branch.py); the measured rung is never declared from a
    table, and no built-in ships ``measured_fork``."""
    assert (
        BUILTIN_WORLD_CAPABILITIES[WorldKind.LUMEN_GYM].branching is BranchingSupport.PREFIX_REPLAY
    )
    for kind, capabilities in BUILTIN_WORLD_CAPABILITIES.items():
        assert capabilities.branching is not BranchingSupport.MEASURED_FORK, kind
        if capabilities.branching is not BranchingSupport.UNSPECIFIED:
            assert capabilities.closed_loop, kind


def test_gates_tuple_carries_the_promotion_flag() -> None:
    assert WorldCapabilities().gates().count(True) == 0
    promoted = WorldCapabilities(closed_loop=True, branching=BranchingSupport.DECLARED)
    assert promoted.gates()[-1] is True
    assert WorldCapabilities(closed_loop=True).gates()[-1] is False
