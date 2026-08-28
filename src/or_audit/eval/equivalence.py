"""The §2.6 shelf-equivalence artifact: comparability earned, not schema'd.

A shared ``TrialVector`` schema does **not** make scores from different worlds
comparable. Two worlds can carry the same gate id, the same metric names and the
same headline while differing in observation/action interface, anatomy, scenario
distribution, dynamics fidelity, termination semantics, and what their safety
state physically means. Reporting "policy A is safer than policy B across
worlds" on that basis would be a fabricated claim wearing a schema.

So cross-world comparability is a separate, published, digest-pinned artifact
with four requirements, each of which must hold before any surface may order
across worlds:

1. **Task equivalence** — matched objective, initial-state distribution, and
   termination semantics, declared in the artifact rather than assumed.
2. **Gate equivalence** — each hard gate maps to the same physical quantity in
   the same unit, with per-engine calibration evidence that the thresholds bite
   at comparable physical events. Identically *named* gates are not equivalent
   gates.
3. **Scenario-distribution alignment** — anatomy/difficulty matched, or
   explicitly stratified with the strata named.
4. **An external referent** — agreement with a ground truth neither engine owns
   (a phantom, a bench, a registry), measured as rank correlation against a
   declared floor and *recomputed here* from the declared rankings so the
   number cannot be asserted into existence.

Until such an artifact validates, every public surface reports per-world rows
only; see :mod:`or_audit.eval.shelf`.
"""

from __future__ import annotations

import json
import math
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from or_audit.audit.canonical import digest
from or_audit.errors import ScoreContractError, TaskContractError
from or_audit.eval.task import NonEmpty, Slug

#: Declared floats are recomputed; this is the slack allowed on the round-trip.
_CORRELATION_TOLERANCE = 1e-9

Statement = Annotated[str, StringConstraints(min_length=1, max_length=2000)]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^(|[0-9a-f]{64})$")]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GateRef(Protocol):
    """One gate a shelf world actually runs, as this module needs to see it.

    Structural rather than imported: :mod:`or_audit.eval.shelf` already imports
    this module to gate its cross-world collapse, so the manifest type it owns
    (``ShelfGateRef``) cannot be imported back the other way.
    """

    @property
    def gate_id(self) -> str: ...

    @property
    def unit(self) -> str: ...


class DeclaredMatch(_Frozen):
    """One explicitly declared equivalence claim and whether it holds.

    ``matched=False`` is a legitimate, publishable state: it says the authors
    looked and found the two worlds differ. It simply does not unlock ordering.
    """

    statement: Statement
    matched: bool


class TaskEquivalence(_Frozen):
    """§2.6 requirement 1: the two worlds pose the same task."""

    objective: DeclaredMatch
    initial_state_distribution: DeclaredMatch
    termination: DeclaredMatch

    def failures(self) -> tuple[str, ...]:
        return tuple(
            f"task equivalence: {name} is declared unmatched ({claim.statement})"
            for name, claim in (
                ("objective", self.objective),
                ("initial-state distribution", self.initial_state_distribution),
                ("termination semantics", self.termination),
            )
            if not claim.matched
        )


class GateCalibration(_Frozen):
    """Per-engine evidence that one gate's threshold bites at the shared event."""

    world_id: Slug
    #: The physical quantity this engine's threshold is expressed over.
    physical_quantity: NonEmpty
    #: The unit that quantity is measured in on this engine.
    unit: NonEmpty
    threshold: float
    #: Whether the calibration showed the threshold firing at the gate's
    #: declared physical event on this engine.
    bites_at_declared_event: bool
    #: How that was measured. An unbacked claim is not calibration.
    evidence: Statement


class GateEquivalence(_Frozen):
    """§2.6 requirement 2: one hard gate, the same physics on both engines."""

    gate_id: Slug
    #: The physical quantity the gate binds to, e.g. "wall contact force".
    physical_quantity: NonEmpty
    #: The unit the threshold is expressed in, e.g. "newton".
    unit: NonEmpty
    #: The physical event the threshold is calibrated to fire at.
    physical_event: NonEmpty
    calibration: tuple[GateCalibration, ...]

    def failures(self, world_pair: tuple[str, str]) -> tuple[str, ...]:
        """Reasons this gate is not equivalent across ``world_pair``."""
        reasons: list[str] = []
        by_world = {entry.world_id: entry for entry in self.calibration}
        if len(by_world) != len(self.calibration):
            reasons.append(f"gate '{self.gate_id}': duplicate per-world calibration entries")
        for world_id in world_pair:
            entry = by_world.get(world_id)
            if entry is None:
                reasons.append(
                    f"gate '{self.gate_id}': no calibration evidence for world '{world_id}'"
                )
                continue
            if entry.unit != self.unit:
                reasons.append(
                    f"gate '{self.gate_id}': world '{world_id}' calibrates in "
                    f"{entry.unit!r} but the gate declares {self.unit!r} — identically "
                    f"named gates in different units are not the same gate"
                )
            if entry.physical_quantity != self.physical_quantity:
                reasons.append(
                    f"gate '{self.gate_id}': world '{world_id}' measures "
                    f"{entry.physical_quantity!r} but the gate declares "
                    f"{self.physical_quantity!r}"
                )
            if not entry.bites_at_declared_event:
                reasons.append(
                    f"gate '{self.gate_id}': world '{world_id}' does not fire at the "
                    f"declared event {self.physical_event!r}"
                )
        return tuple(reasons)


class ScenarioAlignment(_Frozen):
    """§2.6 requirement 3: matched anatomy/difficulty, or declared strata."""

    mode: Literal["matched", "stratified"]
    statement: Statement
    strata: tuple[NonEmpty, ...] = ()

    def failures(self) -> tuple[str, ...]:
        if self.mode == "stratified" and not self.strata:
            return (
                "scenario alignment: mode is 'stratified' but no strata are declared; "
                "name the strata the comparison is made within",
            )
        if self.mode == "matched" and self.strata:
            return (
                "scenario alignment: mode is 'matched' but strata are declared; "
                "a stratified comparison must say so",
            )
        return ()


class ExternalReferent(_Frozen):
    """§2.6 requirement 4: agreement with a ground truth neither engine owns."""

    #: What the referent is: "phantom", "cadaver", "real-data bench", ...
    kind: NonEmpty
    description: Statement
    #: Per-world ordering of the same subjects, best first.
    world_rankings: dict[str, tuple[NonEmpty, ...]]
    #: The referent's own ordering of those subjects, best first.
    referent_ranking: tuple[NonEmpty, ...]
    #: Declared agreement (the weakest per-world Spearman); recomputed at
    #: validation so a published number cannot drift from its own rankings.
    rank_correlation: float = Field(ge=-1.0, le=1.0)
    #: Floor the agreement must clear. A non-positive floor would be no
    #: evidence of agreement at all, so it is refused by type.
    min_rank_correlation: float = Field(gt=0.0, le=1.0)


class Publication(_Frozen):
    """The artifact's own identity: a comparability claim is itself citable."""

    artifact_id: NonEmpty
    #: SHA-256 over this artifact with the digest field blanked; empty until
    #: :func:`write_equivalence_artifact` publishes it.
    digest: Sha256Hex = ""


class EquivalenceArtifact(_Frozen):
    """A published claim that two worlds on one shelf are comparable."""

    shelf_id: Slug
    task_family: Slug
    #: Exactly two shelf world ids. Comparability is pairwise; a shelf-wide
    #: claim is a set of pairwise artifacts, not a broader assertion.
    world_pair: tuple[Slug, Slug]
    task_equivalence: TaskEquivalence
    gate_equivalence: tuple[GateEquivalence, ...]
    scenario_alignment: ScenarioAlignment
    external_referent: ExternalReferent
    published_as: Publication

    @model_validator(mode="after")
    def _distinct_worlds(self) -> Self:
        """One world is not a pair; comparing it with itself measures nothing.

        Left unchecked, ``("w", "w")`` satisfies every per-world check (both
        halves resolve to the same shelf entry, the referent agrees with itself
        perfectly) and unlocks a "cross-world" ordering computed from one world.
        """
        first, second = self.world_pair
        if first == second:
            raise TaskContractError(
                f"equivalence artifact names world '{first}' twice: comparability is a claim "
                "about two distinct worlds, and a world is trivially comparable with itself"
            )
        return self


class EquivalenceVerdict(_Frozen):
    """Outcome of :func:`validate_equivalence`, naming exactly what failed."""

    valid: bool
    artifact_id: str
    shelf_id: str
    task_family: str
    world_pair: tuple[str, str]
    #: The four §2.6 requirements plus the publication pin, in a stable order.
    requirements: dict[str, bool]
    failures: tuple[str, ...]
    #: Weakest per-world rank correlation against the referent, when computable.
    computed_rank_correlation: float | None = None
    #: Whether the artifact was checked against a shelf's real gate manifest.
    #: False means the gate claims were only checked against themselves.
    shelf_gates_checked: bool = False

    @property
    def failed_requirements(self) -> tuple[str, ...]:
        return tuple(name for name, ok in self.requirements.items() if not ok)


def _average_ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks with ties resolved to their average rank."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        average = (start + stop) / 2.0 + 1.0
        for position in range(start, stop + 1):
            ranks[order[position]] = average
        start = stop + 1
    return ranks


def spearman_rank_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman's rho with average ranks for ties.

    Refuses rather than returning a placeholder when the correlation is
    undefined (mismatched lengths, fewer than two observations, or one side
    entirely tied): a fabricated 0.0 there would read as "measured no
    agreement" when nothing was measured at all.
    """
    if len(a) != len(b):
        raise ScoreContractError(
            f"rank correlation needs equal-length sequences, got {len(a)} and {len(b)}"
        )
    if len(a) < 2:
        raise ScoreContractError("rank correlation needs at least two observations")
    ranks_a = _average_ranks(a)
    ranks_b = _average_ranks(b)
    n = len(ranks_a)
    mean_a = sum(ranks_a) / n
    mean_b = sum(ranks_b) / n
    deviations_a = [value - mean_a for value in ranks_a]
    deviations_b = [value - mean_b for value in ranks_b]
    variance_a = sum(value * value for value in deviations_a)
    variance_b = sum(value * value for value in deviations_b)
    if variance_a == 0.0 or variance_b == 0.0:
        raise ScoreContractError(
            "rank correlation is undefined when one ranking ties every subject"
        )
    covariance = sum(x * y for x, y in zip(deviations_a, deviations_b, strict=True))
    return covariance / math.sqrt(variance_a * variance_b)


def _referent_failures(artifact: EquivalenceArtifact) -> tuple[tuple[str, ...], float | None]:
    """Recompute the referent agreement and report why it fails, if it does."""
    referent = artifact.external_referent
    reasons: list[str] = []
    subjects = referent.referent_ranking
    if len(set(subjects)) != len(subjects):
        reasons.append("external referent: the referent ranking repeats a subject")
    if len(subjects) < 2:
        reasons.append(
            "external referent: at least two ranked subjects are needed for an agreement claim"
        )
    declared_worlds = set(referent.world_rankings)
    if declared_worlds != set(artifact.world_pair):
        reasons.append(
            f"external referent: rankings are declared for {sorted(declared_worlds)} but the "
            f"world pair is {list(artifact.world_pair)}"
        )
    for world_id in artifact.world_pair:
        ranking = referent.world_rankings.get(world_id)
        if ranking is None:
            continue
        # A world ranking must be a permutation of the referent's subjects. Set
        # equality alone admits a tuple that repeats one, and the position map
        # below would then keep the last occurrence only - silently moving every
        # other subject while still producing a correlation that can pass.
        if len(set(ranking)) != len(ranking):
            reasons.append(
                f"external referent: world '{world_id}' repeats a subject in its ranking"
            )
        elif len(ranking) != len(subjects) or set(ranking) != set(subjects):
            reasons.append(
                f"external referent: world '{world_id}' ranks a different subject set than "
                f"the referent does"
            )
    if reasons:
        return tuple(reasons), None

    referent_positions = [float(index) for index in range(len(subjects))]
    correlations: dict[str, float] = {}
    for world_id in artifact.world_pair:
        ranking = referent.world_rankings[world_id]
        position_of = {subject: index for index, subject in enumerate(ranking)}
        world_positions = [float(position_of[subject]) for subject in subjects]
        correlations[world_id] = spearman_rank_correlation(world_positions, referent_positions)
    weakest_world = min(correlations, key=lambda key: correlations[key])
    computed = correlations[weakest_world]
    if computed < referent.min_rank_correlation:
        reasons.append(
            f"external referent: world '{weakest_world}' agrees with the {referent.kind} "
            f"referent at rho={computed:.4g}, below the declared minimum "
            f"{referent.min_rank_correlation:.4g}"
        )
    if abs(computed - referent.rank_correlation) > _CORRELATION_TOLERANCE:
        reasons.append(
            f"external referent: declared rank_correlation={referent.rank_correlation:.4g} "
            f"does not match the rankings, which give rho={computed:.4g}"
        )
    return tuple(reasons), computed


def _publication_failures(artifact: EquivalenceArtifact) -> tuple[str, ...]:
    expected = equivalence_digest(artifact)
    if not artifact.published_as.digest:
        return (
            f"publication: artifact '{artifact.published_as.artifact_id}' is unpublished; "
            f"write it with write_equivalence_artifact to pin digest {expected}",
        )
    if artifact.published_as.digest != expected:
        return (
            f"publication: declared digest {artifact.published_as.digest} does not match "
            f"the artifact content, which digests to {expected}",
        )
    return ()


def _shelf_gate_failures(
    artifact: EquivalenceArtifact, world_gates: Mapping[str, Sequence[GateRef]]
) -> tuple[str, ...]:
    """Check the claimed gates against the gates the shelf worlds really run.

    Internal consistency is not coverage: an artifact can calibrate a gate no
    task declares, or cover one hard gate while omitting the others, and still
    read as a clean comparability claim. Matching is by id *and* unit, because
    an identically named gate in another unit is a different gate.
    """
    reasons: list[str] = []
    claimed = {gate.gate_id: gate.unit for gate in artifact.gate_equivalence}
    for world_id in artifact.world_pair:
        manifest = world_gates.get(world_id)
        if manifest is None:
            reasons.append(
                f"shelf gates: no gate manifest for world '{world_id}'; the artifact's gate "
                "claims cannot be checked against what that world runs"
            )
            continue
        declared = {ref.gate_id: ref.unit for ref in manifest}
        for gate_id, unit in sorted(declared.items()):
            if gate_id not in claimed:
                reasons.append(
                    f"shelf gates: world '{world_id}' runs hard gate '{gate_id}' ({unit!r}), "
                    "which the artifact claims no equivalence for"
                )
            elif claimed[gate_id] != unit:
                reasons.append(
                    f"shelf gates: world '{world_id}' declares gate '{gate_id}' in unit "
                    f"{unit!r} but the artifact maps it to {claimed[gate_id]!r}"
                )
        for gate_id in sorted(set(claimed) - set(declared)):
            reasons.append(
                f"shelf gates: the artifact claims gate '{gate_id}', which world "
                f"'{world_id}' does not declare"
            )
    return tuple(reasons)


def validate_equivalence(
    artifact: EquivalenceArtifact,
    *,
    world_gates: Mapping[str, Sequence[GateRef]] | None = None,
) -> EquivalenceVerdict:
    """Check the four §2.6 requirements and the artifact's own publication pin.

    ``world_gates`` is the shelf's per-world gate manifest. Without it the
    artifact is only checked against itself, and the verdict says as much
    through :attr:`EquivalenceVerdict.shelf_gates_checked`: a claim about gates
    the shelf never runs is not evidence of comparability, so no caller may
    unlock cross-world ordering on an unchecked verdict.
    """
    task_failures = artifact.task_equivalence.failures()
    gate_failures: list[str] = []
    if not artifact.gate_equivalence:
        gate_failures.append(
            "gate equivalence: no gate is mapped; a comparability claim over a shelf "
            "with hard gates must map every gate to a physical quantity and unit"
        )
    seen: set[str] = set()
    for gate in artifact.gate_equivalence:
        # One gate, one mapping: duplicates would collapse in the coverage check.
        if gate.gate_id in seen:
            gate_failures.append(
                f"gate equivalence: gate '{gate.gate_id}' is mapped more than once"
            )
        seen.add(gate.gate_id)
        gate_failures.extend(gate.failures(artifact.world_pair))
    shelf_gate_failures = () if world_gates is None else _shelf_gate_failures(artifact, world_gates)
    scenario_failures = artifact.scenario_alignment.failures()
    referent_failures, computed = _referent_failures(artifact)
    publication_failures = _publication_failures(artifact)

    requirements = {
        "task_equivalence": not task_failures,
        "gate_equivalence": not gate_failures,
    }
    if world_gates is not None:
        requirements["shelf_gates"] = not shelf_gate_failures
    requirements["scenario_alignment"] = not scenario_failures
    requirements["external_referent"] = not referent_failures
    requirements["publication"] = not publication_failures
    failures = (
        *task_failures,
        *gate_failures,
        *shelf_gate_failures,
        *scenario_failures,
        *referent_failures,
        *publication_failures,
    )
    return EquivalenceVerdict(
        valid=not failures,
        artifact_id=artifact.published_as.artifact_id,
        shelf_id=artifact.shelf_id,
        task_family=artifact.task_family,
        world_pair=artifact.world_pair,
        requirements=requirements,
        failures=failures,
        computed_rank_correlation=computed,
        shelf_gates_checked=world_gates is not None,
    )


def equivalence_payload(artifact: EquivalenceArtifact) -> dict[str, Any]:
    """JSON-shaped artifact content with its own digest field blanked."""
    payload = artifact.model_dump(mode="json")
    payload["published_as"] = {**payload["published_as"], "digest": ""}
    return payload


def equivalence_digest(artifact: EquivalenceArtifact) -> str:
    """Content digest of the artifact, excluding the digest field itself."""
    return digest(equivalence_payload(artifact))


def write_equivalence_artifact(artifact: EquivalenceArtifact, path: Path) -> EquivalenceArtifact:
    """Publish the artifact as deterministic JSON pinned by its own digest.

    Refuses to overwrite a declared digest that disagrees with the content:
    silently re-pinning an edited artifact under its old identity would let a
    citation point at text that no longer says what it said.
    """
    content_digest = equivalence_digest(artifact)
    declared = artifact.published_as.digest
    if declared and declared != content_digest:
        raise TaskContractError(
            f"equivalence artifact '{artifact.published_as.artifact_id}' declares digest "
            f"{declared} but its content digests to {content_digest}; clear the digest to "
            f"republish under a new pin"
        )
    published = artifact.model_copy(
        update={"published_as": artifact.published_as.model_copy(update={"digest": content_digest})}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(published.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return published


def load_equivalence_artifact(path: Path | str) -> EquivalenceArtifact:
    """Load an equivalence artifact from TOML or JSON."""
    source = Path(path)
    if not source.is_file():
        raise TaskContractError(f"missing equivalence artifact: {source}")
    text = source.read_text(encoding="utf-8")
    try:
        if source.suffix == ".toml":
            raw: Any = tomllib.loads(text)
        elif source.suffix == ".json":
            raw = json.loads(text)
        else:
            raise TaskContractError(
                f"an equivalence artifact is .toml or .json, got {source.suffix or source.name!r}"
            )
    except (tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
        # A syntax error is a contract failure like any other, and the caller
        # (surgeval shelf equivalence check) reports refusals, not tracebacks.
        raise TaskContractError(
            f"equivalence artifact {source} is not parseable {source.suffix.lstrip('.')}: {exc}"
        ) from exc
    try:
        return EquivalenceArtifact.model_validate(raw)
    except (ValidationError, TaskContractError) as exc:
        raise TaskContractError(f"equivalence artifact {source} failed validation: {exc}") from exc
