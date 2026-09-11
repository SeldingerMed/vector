"""Branching support for counterfactual evaluation: probe, gate, and refusal.

Phase F3. A counterfactual interface asks what would have happened had the
policy acted differently at some observed step. Answering it honestly needs a
world that can *fork*: replay the shared prefix exactly, then diverge the
candidates from a common state. The kernel ships no verified
``snapshot``/``restore`` primitive — the simulation bridges' ``snapshot()``
methods are read-only reports of live state, not restorable checkpoints — so
this module measures the one mechanism the pinned stack actually demonstrates
(deterministic prefix replay, see ``tests/test_lumen_branch.py``) and refuses,
rather than assumes, everything else.

Three rules hold it together, and they are the rules conformance applies to
determinism:

* **recorded, never assumed.** :attr:`BranchingSupport.MEASURED_FORK` is
  earned only when two independent forked rollouts of one seed produce
  byte-identical canonical digests;
* **a seed is not a state.** :func:`branch_from` re-runs the shared prefix on
  every arm and verifies each arm's prefix segment digests identically to the
  reference replay before comparing anything. A prefix that fails to replay
  means the world cannot fork at all, and the whole branch set is reported as
  a refusal (:attr:`BranchingSupport.NOT_SUPPORTED`) with no candidate
  comparisons — not a quiet side-by-side of traces from divergent states;
* **a declaration is not a measurement.** :func:`require_branch_support` fails
  closed when the mechanism a caller needs is not backed by the world's
  declared support, including when a bare ``declared`` claim is all there is
  and the caller needs equality-certifying evidence.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Sequence
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator

from or_audit.audit.canonical import digest as canonical_digest
from or_audit.errors import TaskContractError
from or_audit.eval.gym_world import GymEnv, jsonable
from or_audit.eval.worlds import BranchingSupport, WorldCapabilities, branching_at_least

#: Minimum support a caller needs to *drive* side-by-side arms from a shared
#: prefix: at least an authoritative declaration that prefix replay works.
DRIVING_MECHANISM = BranchingSupport.DECLARED
#: Support a caller needs to *certify* branch equivalence (e.g. publishing a
#: counterfactual claim): only a measured, byte-equal fork pair qualifies.
FORK_EQUALITY_MECHANISM = BranchingSupport.MEASURED_FORK
#: Mechanisms a caller may demand of a world. ``not_supported``/``unspecified``
#: are states a world reports, never things one can require of it.
REQUESTABLE_MECHANISMS = frozenset(
    {
        BranchingSupport.DECLARED,
        BranchingSupport.PREFIX_REPLAY,
        BranchingSupport.MEASURED_FORK,
    }
)

#: A zero-argument factory for one fresh environment instance. Deliberately
#: not :data:`or_audit.eval.gym_world.GymFactory` (which takes a task): a
#: probe must not be able to mutate shared world state between arms, so each
#: arm is built by calling this afresh.
BranchEnvFactory = Callable[[], GymEnv]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BranchEvidence(_Frozen):
    """What a prefix-replay probe actually observed. Mirrors DeterminismEvidence.

    ``support`` is what the probe *earned*, kept separate from the strongest
    level either the task or its adapter *claimed*, so a claim the probe does
    not back stays visible as ``claimed > support`` in the evidence instead of
    being silently overwritten by it.
    """

    #: Level the probe earned from the rollouts it ran.
    support: BranchingSupport
    #: Strongest level the task or its installed adapter claims; ``unspecified``
    #: means nothing was claimed and there is nothing to violate.
    claimed: BranchingSupport = BranchingSupport.UNSPECIFIED
    tolerance: float = 0.0
    #: Every replay of the shared prefix produced equal canonical digests.
    prefix_equal: bool
    #: The two independent forked rollouts of one seed produced byte-identical
    #: canonical digests. The fork pair *is* the twin replay pair.
    identical_fork_digests: bool
    #: True when the candidate arms were distinguishable at all: at least two
    #: arms digested differently. ``False`` on distinct plans means the
    #: intervention changed nothing and the comparison measured nothing.
    candidate_diverged: bool
    #: Largest absolute float delta between any two repeated rollouts.
    max_float_delta: float = 0.0
    #: First difference tolerance does not excuse; ``""`` when none.
    first_difference: str = ""

    @model_validator(mode="after")
    def _tolerance_is_a_finite_measurement(self) -> Self:
        """``inf``/``nan`` tolerance would certify two unrelated traces as equal.

        ``delta <= inf`` holds for every pair of floats, so an infinite
        tolerance is a comparison that observed nothing while still reporting a
        passing measurement. Neither value is a JSON number, so the evidence
        would serialize as non-standard ``Infinity``/``NaN``.
        """
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise TaskContractError(
                f"branching tolerance must be a finite non-negative float, got {self.tolerance!r}"
            )
        return self

    @model_validator(mode="after")
    def _earned_fork_needs_a_real_fork_pair(self) -> Self:
        """``measured_fork`` is a claim about byte equality of two rollouts.

        Reporting it without the matching digest evidence would let an
        unmeasured world inherit the strongest level in the schema — the exact
        failure mode the determinism ladder exists to prevent.
        """
        if self.support is BranchingSupport.MEASURED_FORK and not (
            self.prefix_equal and self.identical_fork_digests
        ):
            raise TaskContractError(
                "cannot report 'measured_fork' without prefix_equal=True and "
                "identical_fork_digests=True: the level is earned from two "
                "byte-identical forked rollouts or not at all"
            )
        return self


def _claimed_support(
    capabilities: WorldCapabilities | None,
    declared: BranchingSupport | None,
) -> BranchingSupport:
    """Strongest standing claim, from the task declaration and the adapter alike.

    Both are claims a published artifact carries, and the probe must be able
    to contradict either, so it is the *strongest* claim that gets compared
    against the measurement — the ``_declared_determinism`` rule from
    conformance.
    """
    strongest = BranchingSupport.UNSPECIFIED
    for claim in (capabilities.branching if capabilities is not None else None, declared):
        if claim is not None and not branching_at_least(strongest, claim):
            strongest = claim
    return strongest


def _normalize_action(action: Any) -> Any:
    """Reduce an action to a hashable, digest-stable form.

    numpy arrays are accepted by gym ``step`` but hash by identity and digest
    through ``repr``, so two element-equal arrays would compare unequal. They
    are canonicalized through ``tolist`` here so arm plans and digests see
    content, not object identity.
    """
    tolist = getattr(action, "tolist", None)
    if callable(tolist):
        return tolist()
    return action


def _float_delta(left: Any, right: Any) -> float:
    """Largest absolute float delta between two canonically-matched values.

    Walks mappings and sequences pairwise; non-numeric leaves contribute
    nothing here (a structural difference is caught by the digest comparison,
    which is stricter than any tolerance can be). Bools are skipped: a flipped
    ``terminated`` flag is already a digest-level difference.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return 0.0
    if isinstance(left, float | int) and isinstance(right, float | int):
        return abs(float(left) - float(right))
    if isinstance(left, dict) and isinstance(right, dict):
        worst = 0.0
        for key in left.keys() & right.keys():
            worst = max(worst, _float_delta(left[key], right[key]))
        return worst
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        worst = 0.0
        for a, b in zip(left, right, strict=False):
            worst = max(worst, _float_delta(a, b))
        return worst
    return 0.0


def _trace_delta(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]) -> float:
    """Max float delta across two aligned rollout traces, step by step."""
    worst = 0.0
    for a, b in zip(left, right, strict=False):
        worst = max(worst, _float_delta(a, b))
    return worst


def _rollout(
    env: GymEnv,
    *,
    seed: int,
    options: dict[str, Any] | None,
    actions: Sequence[Any],
    max_steps: int,
) -> list[dict[str, Any]]:
    """Drive ``env`` with ``actions`` and return the per-step trace.

    Deliberately does not route through :func:`or_audit.eval.gym_world.run_gym_episode`:
    that driver injects harness perturbations and stamps policy-side
    bookkeeping meant for *scoring*. A probe must observe the engine's own
    bytes — anything sanitized before the digest is measured can hide the very
    nondeterminism being tested, and an injected perturbation would compare
    perturbation schedules instead of worlds. The reset record is part of the
    trace, because hidden state chosen at reset is exactly what "a seed
    restores the state" would smuggle past a prefix-only comparison.
    Termination honours ``terminated``/``truncated`` the way the runner does.
    """
    _, reset_info = env.reset(seed=seed, options=options)
    trace: list[dict[str, Any]] = [{"reset_info": jsonable(reset_info), "seed": seed}]
    for index, action in enumerate(actions):
        if index >= max_steps:
            break
        _, reward, terminated, truncated, info = env.step(action)
        trace.append(
            {
                "action": jsonable(_normalize_action(action)),
                "reward": jsonable(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": jsonable(info) if isinstance(info, dict) else jsonable({"info": info}),
            }
        )
        if terminated or truncated:
            break
    return trace


def _fresh_rollout(
    env_factory: BranchEnvFactory,
    *,
    seed: int,
    options: dict[str, Any] | None,
    actions: Sequence[Any],
    max_steps: int,
    label: str,
) -> list[dict[str, Any]]:
    """One rollout on a brand-new environment instance.

    Every arm gets its own instance: an engine with hidden mutable state that
    shared an instance across arms would let an earlier arm's stepping leak
    into a later arm's trace, which digests as *support for* branching that
    was never actually provided.
    """
    try:
        env = env_factory()
    except BaseException as exc:
        raise TaskContractError(
            f"branching probe could not construct the {label} environment: {exc!r}"
        ) from exc
    try:
        return _rollout(env, seed=seed, options=options, actions=actions, max_steps=max_steps)
    except TaskContractError:
        raise
    except BaseException as exc:
        raise TaskContractError(
            f"branching probe could not complete the {label} rollout: {exc!r}"
        ) from exc
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            with contextlib.suppress(BaseException):
                close()


def _first_difference(first: Sequence[dict[str, Any]], second: Sequence[dict[str, Any]]) -> str:
    """Canonical summary of the first step where two traces part company."""
    for index, (a, b) in enumerate(zip(first, second, strict=False)):
        if canonical_digest(a) != canonical_digest(b):
            if "action" in a:
                return f"step {index} action {a['action']!r}"
            return f"reset record of step {index}"
    if len(first) != len(second):
        return f"traces end at different steps ({len(first)} vs {len(second)})"
    return "traces differ"


def _compare_fork_pair(
    first: Sequence[dict[str, Any]],
    second: Sequence[dict[str, Any]],
    *,
    claimed: BranchingSupport,
    tolerance: float,
) -> BranchEvidence:
    """Grade one twin pair of rollouts into evidence; shared by both probes."""
    byte_equal = canonical_digest(list(first)) == canonical_digest(list(second))
    delta = _trace_delta(first, second)
    if byte_equal:
        return BranchEvidence(
            support=BranchingSupport.MEASURED_FORK,
            claimed=claimed,
            tolerance=tolerance,
            prefix_equal=True,
            identical_fork_digests=True,
            candidate_diverged=False,
            max_float_delta=0.0,
        )
    if delta <= tolerance and len(first) == len(second):
        # Byte-different but every float within tolerance: faithful enough to
        # drive arms, not enough to certify equality, so it stops one rung
        # below the top of the ladder.
        return BranchEvidence(
            support=BranchingSupport.PREFIX_REPLAY,
            claimed=claimed,
            tolerance=tolerance,
            prefix_equal=True,
            identical_fork_digests=False,
            candidate_diverged=False,
            max_float_delta=delta,
            first_difference=f"twin digests differ within tolerance {tolerance:g}",
        )
    return BranchEvidence(
        support=BranchingSupport.NOT_SUPPORTED,
        claimed=claimed,
        tolerance=tolerance,
        prefix_equal=False,
        identical_fork_digests=False,
        candidate_diverged=False,
        max_float_delta=delta,
        first_difference=(
            f"a seed alone does not restore this world's state: twin rollouts diverged "
            f"at {_first_difference(first, second)} beyond tolerance {tolerance:g}"
        ),
    )


def require_branch_support(
    capabilities: WorldCapabilities | None, mechanism: BranchingSupport
) -> None:
    """Fail closed unless the world's declared support backs ``mechanism``.

    This is the gate every counterfactual-branching caller passes through
    before forking anything. It refuses, in order: a caller demanding a
    mechanism that is not a mechanism at all (``not_supported``/``unspecified``
    are states a world reports, never things one can require of one); a
    missing capability source; and any support weaker than what is asked. So
    a lone :attr:`BranchingSupport.DECLARED` claim may drive arms
    (:data:`DRIVING_MECHANISM`) but can never certify equality
    (:data:`FORK_EQUALITY_MECHANISM`), which needs the measured rung.
    """
    if mechanism not in REQUESTABLE_MECHANISMS:
        raise TaskContractError(
            f"{mechanism.value!r} is not a branching mechanism a caller can require: "
            f"ask for one of {', '.join(sorted(item.value for item in REQUESTABLE_MECHANISMS))}"
        )
    if capabilities is None:
        raise TaskContractError(
            f"branching mechanism {mechanism.value!r} needs capability declarations, but the "
            "world has none: install the world adapter or declare "
            "[environment.capabilities].branching in the task package"
        )
    support = capabilities.branching
    if support is BranchingSupport.DECLARED and mechanism is FORK_EQUALITY_MECHANISM:
        raise TaskContractError(
            f"{mechanism.value!r} branching needs measured evidence, but the world "
            f"only declares {support.value!r}: an author's assertion may drive arms, "
            "yet it can never certify that two branches were equal. Run "
            "measure_branch_support against this adapter and record the result, or "
            "serve the comparison without an equality claim"
        )
    if not branching_at_least(support, mechanism):
        raise TaskContractError(
            f"world cannot back {mechanism.value!r} branching: it declares "
            f"{support.value!r}, which is weaker. Refusing to compare branches on a world "
            "that may not reproduce the shared prefix — any such comparison would be "
            "fabricated evidence"
        )


def measure_branch_support(
    env_factory: BranchEnvFactory,
    *,
    seed: int,
    prefix_actions: Sequence[Any],
    tolerance: float = 0.0,
    options: dict[str, Any] | None = None,
    max_steps: int = 10_000,
    capabilities: WorldCapabilities | None = None,
    declared: BranchingSupport | None = None,
) -> BranchEvidence:
    """Measure whether one world can fork a rollout at a shared prefix.

    Runs the same ``prefix_actions`` from the same ``seed`` twice, each on a
    fresh instance, and grades what it saw: byte-identical twins earn
    :attr:`BranchingSupport.MEASURED_FORK`; agreement only inside
    ``tolerance`` earns :attr:`BranchingSupport.PREFIX_REPLAY` (faithful enough
    to drive arms, not to certify equality); anything else reports
    :attr:`BranchingSupport.NOT_SUPPORTED` with the first difference.
    ``claimed`` carries the strongest standing claim so ``claimed > support``
    is visible to callers. Deterministic engines earn the top rung; nobody is
    granted it.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise TaskContractError(f"tolerance must be a finite non-negative float, got {tolerance!r}")
    if not callable(env_factory):
        raise TaskContractError("env_factory must be callable (a zero-argument factory)")
    prefix = [_normalize_action(action) for action in prefix_actions]
    first = _fresh_rollout(
        env_factory,
        seed=seed,
        options=options,
        actions=prefix,
        max_steps=max_steps,
        label="fork",
    )
    second = _fresh_rollout(
        env_factory,
        seed=seed,
        options=options,
        actions=prefix,
        max_steps=max_steps,
        label="twin",
    )
    return _compare_fork_pair(
        first,
        second,
        claimed=_claimed_support(capabilities, declared),
        tolerance=tolerance,
    )


def branch_from(
    env_factory: BranchEnvFactory,
    *,
    seed: int,
    prefix_actions: Sequence[Any],
    candidates: Sequence[Sequence[Any]],
    tolerance: float = 0.0,
    options: dict[str, Any] | None = None,
    max_steps: int = 10_000,
    capabilities: WorldCapabilities | None = None,
    declared: BranchingSupport | None = None,
) -> dict[str, Any]:
    """Drive counterfactual candidates from one shared prefix, or refuse.

    For each candidate an arm re-runs the shared prefix on a fresh instance and
    continues with the candidate's own actions, and every arm is rolled twice
    so its own reproducibility is measured, not assumed. Before any candidate
    outcome is trusted, each arm's prefix segment must digest identically to
    the reference replay of that prefix — **a seed alone does not restore a
    full mutable state**, so an arm whose prefix fails to replay was branched
    from a different world state than it claims. When that happens the whole
    branch set is refused with :attr:`BranchingSupport.NOT_SUPPORTED`
    evidence and *no* candidate comparisons are reported, rather than quietly
    comparing traces from divergent states.

    Returns a plain, JSON-safe dict::

        {"supported": bool, "refused": str, "evidence": BranchEvidence,
         "arms": [{"index", "plan", "digest", "steps", "replayed_equal",
                   "prefix_matches_reference", "max_float_delta"}]}

    ``candidate_diverged`` on the evidence is the honesty check on the whole
    experiment: distinct candidate plans that all digest equal means the
    intervention changed nothing and the comparison measured nothing.
    """
    if not callable(env_factory):
        raise TaskContractError("env_factory must be callable (a zero-argument factory)")
    if not candidates:
        raise TaskContractError("branch_from needs at least one candidate action sequence")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise TaskContractError(f"tolerance must be a finite non-negative float, got {tolerance!r}")
    prefix = [_normalize_action(action) for action in prefix_actions]
    claimed = _claimed_support(capabilities, declared)

    # The fork pair: two independent instances replaying the same prefix from
    # the same seed. Their equality is what every arm's branch point is graded
    # against, so it is established before a single candidate runs.
    reference = _fresh_rollout(
        env_factory,
        seed=seed,
        options=options,
        actions=prefix,
        max_steps=max_steps,
        label="reference",
    )
    twin = _fresh_rollout(
        env_factory,
        seed=seed,
        options=options,
        actions=prefix,
        max_steps=max_steps,
        label="reference-twin",
    )
    fork = _compare_fork_pair(reference, twin, claimed=claimed, tolerance=tolerance)
    if not fork.prefix_equal:
        return {
            "supported": False,
            "refused": (
                "the shared prefix did not replay identically on two independent instances: "
                "a seed alone does not restore this world's state, so counterfactual "
                "branches from it would compare trajectories that already diverged before "
                "the branch point"
            ),
            "evidence": fork,
            "arms": [],
        }

    reference_digest = canonical_digest(reference)
    prefix_len = len(reference)
    arms: list[dict[str, Any]] = []
    digests: list[str] = []
    worst = fork.max_float_delta
    for index, candidate in enumerate(candidates):
        plan = prefix + [_normalize_action(action) for action in candidate]
        label = f"candidate-{index}"
        first_pass = _fresh_rollout(
            env_factory,
            seed=seed,
            options=options,
            actions=plan,
            max_steps=max_steps,
            label=f"{label} (a)",
        )
        second_pass = _fresh_rollout(
            env_factory,
            seed=seed,
            options=options,
            actions=plan,
            max_steps=max_steps,
            label=f"{label} (b)",
        )
        arm_prefix = first_pass[:prefix_len]
        if canonical_digest(list(arm_prefix)) != reference_digest:
            delta = max(worst, _trace_delta(arm_prefix, reference))
            return {
                "supported": False,
                "refused": (
                    f"candidate {index} replayed the shared prefix differently from the "
                    "reference rollout, so its branch point is not the state the reference "
                    "arm had; refusing the comparison instead of reporting a fabricated "
                    "counterfactual"
                ),
                "evidence": BranchEvidence(
                    support=BranchingSupport.NOT_SUPPORTED,
                    claimed=claimed,
                    tolerance=tolerance,
                    prefix_equal=False,
                    identical_fork_digests=fork.identical_fork_digests,
                    candidate_diverged=False,
                    max_float_delta=delta,
                    first_difference=(
                        f"candidate {index} diverged from the reference prefix before the "
                        "branch point"
                    ),
                ),
                "arms": [],
            }
        replayed_equal = canonical_digest(first_pass) == canonical_digest(second_pass)
        replay_delta = _trace_delta(first_pass, second_pass)
        worst = max(worst, replay_delta)
        digest = canonical_digest(first_pass)
        digests.append(digest)
        arms.append(
            {
                "index": index,
                "plan": jsonable(plan),
                "digest": digest,
                "steps": len(first_pass),
                "replayed_equal": replayed_equal,
                "prefix_matches_reference": True,
                "max_float_delta": replay_delta,
            }
        )

    everything_exact = fork.identical_fork_digests and all(arm["replayed_equal"] for arm in arms)
    evidence = BranchEvidence(
        support=(
            BranchingSupport.MEASURED_FORK if everything_exact else BranchingSupport.PREFIX_REPLAY
        ),
        claimed=claimed,
        tolerance=tolerance,
        prefix_equal=True,
        identical_fork_digests=everything_exact,
        candidate_diverged=len(set(digests)) > 1,
        max_float_delta=worst,
    )
    return {"supported": True, "refused": "", "evidence": evidence, "arms": arms}
