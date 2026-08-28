"""Capability 3: search a task's declared scenario space, then freeze the result.

Two rules make adaptive stress testing safe to sell, and both are enforced
here rather than described:

* **No in-place mutation, ever.** :func:`search_scenarios` derives new
  ``ScenarioSpec``/``PerturbationSpec`` values from the declared ones and
  verifies the source package's ``tree_digest`` is unchanged when it returns.
* **Nothing scored runs against an unfrozen adaptation.**
  :func:`freeze_adapted_package` writes a *new* versioned, digest-pinned
  package whose provenance records ``authored_by: agent`` and
  ``public_leaderboard_eligible: false``; :func:`assert_frozen_before_scoring`
  re-hashes the directory, so an edit after freezing is not scoreable.

The search aims at the hardest *honest* test. A candidate whose abstention rate
exceeds the budget is refused rather than celebrated: a trial the verifier
cannot assess is not a harder test, it is an unmeasured one, and ranking by it
would reward breaking the instrumentation.

The concierge can author scenarios. It can never author verifiers:
:func:`assert_verifier_untouched` refuses any difference in the verifier file,
gates, metrics, headline, or projection between parent and frozen package.
"""

from __future__ import annotations

import json
import shutil
import tomllib
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from math import isfinite
from pathlib import Path
from typing import Annotated, Any, Final, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from or_audit.audit.canonical import digest
from or_audit.errors import TaskContractError
from or_audit.eval.contracts import PerturbationSpec, ScenarioSpec, Slug
from or_audit.eval.integrity import file_sha256, tree_digest
from or_audit.eval.loader import load_task
from or_audit.eval.task import TaskSpec

#: Keys :func:`freeze_adapted_package` is permitted to change in ``task.toml``.
#: Everything else must re-emit byte-for-byte equal in meaning, checked by
#: reparsing the file we just wrote.
_MUTABLE_TASK_KEYS: Final[frozenset[str]] = frozenset(
    {"task_version", "scenarios", "perturbations"}
)

_CACHE_IGNORES: Final = shutil.ignore_patterns(
    ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "*.pyc", "*.pyo"
)

PROVENANCE_FILENAME: Final = "provenance.json"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AdaptCandidate(_Frozen):
    """One derived initial condition. Never a mutation of a declared one."""

    id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    scenario: ScenarioSpec
    perturbations: tuple[PerturbationSpec, ...] = ()

    @property
    def candidate_digest(self) -> str:
        return digest(self.model_dump(mode="json"))


class AdaptObservation(_Frozen):
    """What the injected evaluator measured for one candidate."""

    n: Annotated[int, Field(ge=1)]
    gate_failures: Annotated[int, Field(ge=0)]
    abstentions: Annotated[int, Field(ge=0)]
    headline_true: Annotated[int, Field(ge=0)]
    notes: str = ""

    @model_validator(mode="after")
    def _counts_fit_n(self) -> Self:
        for name, value in (
            ("gate_failures", self.gate_failures),
            ("abstentions", self.abstentions),
            ("headline_true", self.headline_true),
        ):
            if value > self.n:
                raise TaskContractError(
                    f"observation reports {name}={value} over n={self.n} trial(s)"
                )
        return self


class AdaptBudget(_Frozen):
    """Search ceiling, plus the honesty bound on abstention."""

    max_candidates: Annotated[int, Field(ge=1, le=10_000)]
    trials_per_candidate: Annotated[int, Field(ge=1, le=10_000)] = 5
    #: Above this abstention rate a candidate is refused, not ranked: an
    #: unassessable trial is not a harder test.
    max_abstention_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.2


class ScenarioSpace(_Frozen):
    """Which corner of the declared space to walk.

    Every field selects among, or derives from, what the *task* declares. There
    is no field for inventing a perturbation kind the world does not implement:
    that would be a new task, authored through the package path, not a stress
    knob.
    """

    scenario_ids: tuple[Slug, ...] = ()
    perturbation_ids: tuple[Slug, ...] = ()
    #: Extra seeds to re-instantiate each selected scenario under.
    seeds: tuple[Annotated[int, Field(ge=0)], ...] = ()
    include_unperturbed: bool = True
    max_perturbations_per_candidate: Annotated[int, Field(ge=0, le=8)] = 1


class AdaptRow(_Frozen):
    """One evaluated candidate and its place in the degradation ordering."""

    candidate: AdaptCandidate
    n: Annotated[int, Field(ge=1)]
    gate_failure_rate: float
    abstention_rate: float
    success_rate: float
    honest: bool
    reason: str = ""
    notes: str = ""


class AdaptReport(_Frozen):
    """Degradation ordering over a task's declared space, hardest first."""

    task_id: str
    parent_digest: str
    budget: AdaptBudget
    evaluated: Annotated[int, Field(ge=0)]
    ordering: tuple[AdaptRow, ...] = ()
    refusals: tuple[str, ...] = ()

    @property
    def hardest(self) -> AdaptRow | None:
        """Hardest *honest* candidate, or ``None`` if none qualified."""
        return next((row for row in self.ordering if row.honest), None)

    def describe(self) -> str:
        lines = [f"Adaptation search over {self.task_id} (parent {self.parent_digest})"]
        for refusal in self.refusals:
            lines.append(f"  REFUSED: {refusal}")
        lines.append("  degradation ordering (hardest honest first):")
        for index, row in enumerate(self.ordering, start=1):
            flag = "honest" if row.honest else f"REFUSED ({row.reason})"
            lines.append(
                f"    {index}. {row.candidate.id} gate-fail {row.gate_failure_rate:.3f} "
                f"success {row.success_rate:.3f} abstain {row.abstention_rate:.3f} [{flag}]"
            )
        if self.hardest is None:
            lines.append("  no honest candidate: nothing may be frozen from this search")
        return "\n".join(lines)


class FrozenPackage(_Frozen):
    """A digest-pinned, agent-authored adaptation. Quarantined by construction."""

    path: str
    task_id: str
    task_version: str
    digest: str
    parent_task_id: str
    parent_task_version: str
    parent_digest: str
    authored_by: Annotated[str, StringConstraints(min_length=1, max_length=80)] = "agent"
    public_leaderboard_eligible: bool = False
    scenario_ids: tuple[str, ...] = ()
    perturbation_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _quarantined(self) -> Self:
        if self.public_leaderboard_eligible:
            raise TaskContractError(
                f"refusing to mint public-leaderboard eligibility for "
                f"{self.task_id}: an adaptation is eligible only after promotion "
                "through the same Tier-1 conformance any wrap faces; the authoring "
                "step cannot grant it to itself"
            )
        return self


def _selected_scenarios(task: TaskSpec, space: ScenarioSpace) -> tuple[ScenarioSpec, ...]:
    if not space.scenario_ids:
        return task.scenarios
    known = {scenario.id: scenario for scenario in task.scenarios}
    unknown = [item for item in space.scenario_ids if item not in known]
    if unknown:
        raise TaskContractError(
            f"task {task.id} declares no scenario(s) {sorted(unknown)}; the search "
            "space is the task's own declared space"
        )
    return tuple(known[item] for item in space.scenario_ids)


def _selected_perturbations(task: TaskSpec, space: ScenarioSpace) -> tuple[PerturbationSpec, ...]:
    if not space.perturbation_ids:
        return task.perturbations
    known = {perturbation.id: perturbation for perturbation in task.perturbations}
    unknown = [item for item in space.perturbation_ids if item not in known]
    if unknown:
        raise TaskContractError(
            f"task {task.id} declares no perturbation(s) {sorted(unknown)}; a stress "
            "knob the world does not implement is a new task, not an adaptation"
        )
    return tuple(known[item] for item in space.perturbation_ids)


def _derive_scenario(scenario: ScenarioSpec, seed: int) -> ScenarioSpec:
    """Return a NEW scenario at ``seed``; the declared one is left untouched."""
    if seed == scenario.seed:
        return scenario
    return scenario.model_copy(update={"id": f"{scenario.id}-seed{seed}", "seed": seed})


def _candidates(task: TaskSpec, space: ScenarioSpace) -> list[AdaptCandidate]:
    scenarios = _selected_scenarios(task, space)
    perturbations = _selected_perturbations(task, space)
    if not scenarios:
        raise TaskContractError(
            f"task {task.id} declares no scenarios, so it has no declared space to "
            "search; author a scenario package before adapting it"
        )
    seeds: list[int] = []
    for seed in space.seeds:
        if seed not in seeds:
            seeds.append(seed)
    out: list[AdaptCandidate] = []
    for scenario in scenarios:
        seed_variants = [scenario.seed, *[seed for seed in seeds if seed != scenario.seed]]
        for seed in seed_variants:
            derived = _derive_scenario(scenario, seed)
            applicable = [
                perturbation
                for perturbation in perturbations
                if perturbation.scenario_id in (None, scenario.id)
            ]
            sizes = range(
                0 if space.include_unperturbed else 1,
                min(space.max_perturbations_per_candidate, len(applicable)) + 1,
            )
            for size in sizes:
                for chosen in combinations(applicable, size):
                    label = "+".join(item.id for item in chosen) or "unperturbed"
                    out.append(
                        AdaptCandidate(
                            id=f"{derived.id}@{label}",
                            scenario=derived,
                            perturbations=chosen,
                        )
                    )
    return out


def search_scenarios(
    task_dir: Path | str,
    *,
    space: ScenarioSpace,
    evaluate: Callable[[AdaptCandidate], AdaptObservation],
    budget: AdaptBudget,
) -> AdaptReport:
    """Walk the declared space toward the hardest honest test.

    The source package is read-only here: candidates are new objects derived
    from declared specs, and the package's ``tree_digest`` is re-checked before
    returning, so an evaluator that writes into the task directory is caught
    instead of silently changing what a parent digest means.
    """
    root = Path(task_dir)
    task = load_task(root)
    parent_digest = tree_digest(root)
    candidates = _candidates(task, space)
    refusals: list[str] = []
    if len(candidates) > budget.max_candidates:
        refusals.append(
            f"candidate budget exhausted: {len(candidates) - budget.max_candidates} of "
            f"{len(candidates)} candidate(s) not evaluated (max_candidates="
            f"{budget.max_candidates})"
        )
        candidates = candidates[: budget.max_candidates]

    rows: list[AdaptRow] = []
    for candidate in candidates:
        observation = evaluate(candidate)
        abstention_rate = observation.abstentions / observation.n
        honest = True
        reason = ""
        if observation.n > budget.trials_per_candidate:
            honest = False
            reason = (
                f"evaluator ran {observation.n} trial(s) where the budget allows "
                f"{budget.trials_per_candidate}"
            )
        elif abstention_rate > budget.max_abstention_rate:
            honest = False
            reason = (
                f"abstention rate {abstention_rate:.3f} exceeds the honesty bound "
                f"{budget.max_abstention_rate:.3f}: a trial the verifier cannot "
                "assess is an unmeasured test, not a harder one"
            )
        if not honest:
            refusals.append(f"{candidate.id}: {reason}")
        rows.append(
            AdaptRow(
                candidate=candidate,
                n=observation.n,
                gate_failure_rate=observation.gate_failures / observation.n,
                abstention_rate=abstention_rate,
                success_rate=observation.headline_true / observation.n,
                honest=honest,
                reason=reason,
                notes=observation.notes,
            )
        )

    ordering = tuple(
        sorted(
            rows,
            key=lambda row: (
                not row.honest,
                -row.gate_failure_rate,
                row.success_rate,
                row.candidate.id,
            ),
        )
    )
    after = tree_digest(root)
    if after != parent_digest:
        raise TaskContractError(
            f"scenario search mutated its source package {root} "
            f"({parent_digest} -> {after}); adaptation derives new packages and "
            "never edits the parent, or a published digest stops meaning anything"
        )
    return AdaptReport(
        task_id=task.id,
        parent_digest=parent_digest,
        budget=budget,
        evaluated=len(rows),
        ordering=ordering,
        refusals=tuple(refusals),
    )


def _toml_scalar(value: Any, *, path: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not isfinite(value):
            raise TaskContractError(f"cannot serialize non-finite {path} to TOML")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise TaskContractError(f"cannot serialize {path} of type {type(value).__name__} to TOML")


def _is_inline(value: Any) -> bool:
    """Whether a mapping can be emitted as an inline table."""
    return isinstance(value, Mapping) and all(
        not isinstance(item, Mapping | list | tuple) for item in value.values()
    )


def _toml_value(value: Any, *, path: str) -> str:
    if isinstance(value, Mapping):
        body = ", ".join(
            f"{key} = {_toml_value(item, path=f'{path}.{key}')}" for key, item in value.items()
        )
        return "{" + body + "}"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(item, path=f"{path}[]") for item in value) + "]"
    return _toml_scalar(value, path=path)


def _is_table_array(value: Any) -> bool:
    return (
        isinstance(value, list | tuple)
        and bool(value)
        and all(isinstance(item, Mapping) for item in value)
    )


def _dump_toml(data: Mapping[str, Any], *, prefix: str = "") -> str:
    """Emit a deterministic TOML document.

    Deliberately small: the callers reparse what this writes and compare it to
    the intended structure, so an unrepresentable value fails loudly instead of
    quietly changing a task.
    """
    body: list[str] = []
    tables: list[str] = []
    for key, value in data.items():
        path = f"{prefix}{key}"
        if _is_table_array(value):
            for item in value:
                tables.append(f"\n[[{path}]]")
                tables.append(_dump_toml(item, prefix=f"{path}."))
        elif isinstance(value, Mapping) and not _is_inline(value):
            tables.append(f"\n[{path}]")
            tables.append(_dump_toml(value, prefix=f"{path}."))
        else:
            body.append(f"{key} = {_toml_value(value, path=path)}")
    return "\n".join([*body, *tables]).strip("\n") + "\n"


def _bump_version(version: str) -> str:
    """Bump a task version into a distinct adapted lineage."""
    base, _, tail = version.rpartition("-adapted")
    lineage = bool(base) and tail.isdigit()
    bumped = f"{base}-adapted{int(tail) + 1}" if lineage else f"{version}-adapted1"
    if len(bumped) > 32:
        raise TaskContractError(
            f"adapted task_version {bumped!r} exceeds 32 characters; rename the "
            "parent lineage before adapting it again"
        )
    return bumped


def _verifier_files(task: TaskSpec) -> tuple[str, ...]:
    entrypoint = task.verifier.entrypoint
    module = entrypoint.split(":", 1)[0] if entrypoint else ""
    return tuple(name for name in (module, "verifier.toml") if name)


def assert_verifier_untouched(parent_dir: Path | str, frozen_dir: Path | str) -> None:
    """Refuse any difference in the verifier, its gates, or the projection.

    The concierge authors scenarios. If it could also move a gate or retune a
    projection, an "adapted" package would be a different claim wearing the
    parent's name, and comparing the two results would be meaningless.
    """
    parent_root = Path(parent_dir)
    frozen_root = Path(frozen_dir)
    parent = load_task(parent_root)
    frozen = load_task(frozen_root)
    for name in _verifier_files(parent):
        parent_file = parent_root / name
        frozen_file = frozen_root / name
        if not parent_file.is_file() and not frozen_file.is_file():
            continue
        if not parent_file.is_file() or not frozen_file.is_file():
            raise TaskContractError(
                f"verifier file {name} exists in only one of {parent_root} and "
                f"{frozen_root}; an adaptation cannot add or drop verifier content"
            )
        parent_hash = file_sha256(parent_file)
        frozen_hash = file_sha256(frozen_file)
        if parent_hash != frozen_hash:
            raise TaskContractError(
                f"refusing adaptation: verifier file {name} differs from the parent "
                f"({parent_hash} -> {frozen_hash}). The concierge can author "
                "scenarios, never verifiers"
            )
    if parent.verifier.model_dump(mode="json") != frozen.verifier.model_dump(mode="json"):
        raise TaskContractError(
            f"refusing adaptation of {parent.id}: the verifier declaration (gates, "
            "metrics, headline, abstention) differs from the parent. The concierge "
            "can author scenarios, never verifiers"
        )
    parent_projection = parent.projection.model_dump(mode="json") if parent.projection else None
    frozen_projection = frozen.projection.model_dump(mode="json") if frozen.projection else None
    if parent_projection != frozen_projection:
        raise TaskContractError(
            f"refusing adaptation of {parent.id}: the projection differs from the "
            "parent, so the adapted package would report a different reward for the "
            "same behaviour"
        )


def assert_frozen_before_scoring(package: object) -> FrozenPackage:
    """Return the package if it is still exactly what was frozen; else refuse."""
    if not isinstance(package, FrozenPackage):
        raise TaskContractError(
            f"refusing to score an unfrozen adaptation: expected a FrozenPackage, "
            f"got {type(package).__name__}. Freezing into a versioned, "
            "digest-pinned package precedes the first scored trial"
        )
    actual = tree_digest(Path(package.path))
    if actual != package.digest:
        raise TaskContractError(
            f"refusing to score {package.task_id}: package at {package.path} was "
            f"edited after freezing (pinned {package.digest}, now {actual})"
        )
    return package


def freeze_adapted_package(
    task_dir: Path | str,
    *,
    scenarios: Sequence[ScenarioSpec],
    perturbations: Sequence[PerturbationSpec] = (),
    out: Path | str,
    authored_by: str = "agent",
) -> FrozenPackage:
    """Write a new, versioned, digest-pinned package for an adaptation."""
    root = Path(task_dir).resolve()
    target = Path(out).resolve()
    parent = load_task(root)
    parent_digest = tree_digest(root)
    if not scenarios:
        raise TaskContractError(
            f"refusing to freeze an adaptation of {parent.id} with no scenario: an "
            "adaptation is an initial-condition claim, and an empty one is not one"
        )
    if target == root or root in target.parents:
        raise TaskContractError(
            f"refusing to write the adapted package inside its parent ({target}); "
            "the parent package must stay byte-identical"
        )
    if target.exists() and any(target.iterdir()):
        raise TaskContractError(
            f"refusing to overwrite existing package directory {target}; an adapted "
            "package is a new version, not an edit"
        )

    shutil.copytree(root, target, ignore=_CACHE_IGNORES, dirs_exist_ok=True)
    parent_data = tomllib.loads((root / "task.toml").read_text(encoding="utf-8"))
    data: dict[str, Any] = dict(parent_data)
    data["task_version"] = _bump_version(parent.task_version)
    data["scenarios"] = [
        scenario.model_dump(mode="json", exclude_none=True) for scenario in scenarios
    ]
    if perturbations:
        data["perturbations"] = [
            perturbation.model_dump(mode="json", exclude_none=True)
            for perturbation in perturbations
        ]
    else:
        data.pop("perturbations", None)

    text = _dump_toml(data)
    reparsed = tomllib.loads(text)
    if reparsed != data:
        raise TaskContractError(
            f"refusing to freeze {parent.id}: the re-emitted task.toml does not "
            "reparse to the intended structure, so the adapted package would not be "
            "the package that was checked"
        )
    for key, value in parent_data.items():
        if key not in _MUTABLE_TASK_KEYS and reparsed.get(key) != value:
            raise TaskContractError(
                f"refusing to freeze {parent.id}: re-emitting task.toml changed "
                f"{key!r}, and an adaptation may only change scenarios, "
                "perturbations, and the task version"
            )
    (target / "task.toml").write_text(text, encoding="utf-8")

    scenario_ids = tuple(scenario.id for scenario in scenarios)
    perturbation_ids = tuple(perturbation.id for perturbation in perturbations)
    provenance = {
        "format_version": "1",
        "authored_by": authored_by,
        "public_leaderboard_eligible": False,
        "quarantine_reason": (
            "agent-authored scenario package: excluded from public leaderboards "
            "until promoted through the same Tier-1 conformance any wrap faces"
        ),
        "parent": {
            "task_id": parent.id,
            "task_version": parent.task_version,
            "digest": parent_digest,
        },
        "task_id": parent.id,
        "task_version": data["task_version"],
        "scenarios": list(scenario_ids),
        "perturbations": list(perturbation_ids),
    }
    (target / PROVENANCE_FILENAME).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    frozen = load_task(target)
    if frozen.task_version == parent.task_version:
        raise TaskContractError(
            f"refusing to freeze {parent.id}: the adapted package kept the parent "
            "task_version, so two different packages would answer to one identity"
        )
    assert_verifier_untouched(root, target)
    return FrozenPackage(
        path=str(target),
        task_id=frozen.id,
        task_version=frozen.task_version,
        digest=tree_digest(target),
        parent_task_id=parent.id,
        parent_task_version=parent.task_version,
        parent_digest=parent_digest,
        authored_by=authored_by,
        scenario_ids=scenario_ids,
        perturbation_ids=perturbation_ids,
    )
