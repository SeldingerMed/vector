"""Capability 2: map a confirmed capability onto the catalog and narrate honestly.

Ranking is deliberately ordered so the first criterion is the only one that can
*exclude*: interface satisfaction, evaluated with the same
:meth:`~or_audit.eval.contracts.CapabilitySpec.satisfies` predicate
:func:`or_audit.eval.bind.assert_bind` uses. Selection never decides a bind —
the run still calls ``assert_bind`` — it only decides what is worth running and
records why everything else was not.

The remaining criteria order the survivors: declared-modality overlap, then a
difficulty ladder (easiest first, because a plan that opens with the hardest
world reports a wall of failures and teaches nothing), then PHI class (least
sensitive data first), then task id for determinism. The same capability and
catalog therefore always produce the same plan.

Narration reports gates and abstention first and per-world rows after, and
emits no composite: no cross-world aggregate, ranking, or scalar collapse. §2.6
makes that a published-equivalence question, not a rendering convenience.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from or_audit.concierge.assess import ConfirmedCapability, assert_confirmed
from or_audit.errors import TaskContractError
from or_audit.eval.contracts import CapabilitySpec, InteractionMode, InterfaceSpec
from or_audit.eval.enums import PhiClass
from or_audit.eval.integrity import tree_digest
from or_audit.eval.job import JobResult
from or_audit.eval.loader import load_task
from or_audit.eval.scorecard import scorecard_data
from or_audit.eval.task import TaskSpec

#: Rungs of the difficulty ladder contributed by interaction mode. A
#: single-turn prediction is checked once; a closed-loop policy is checked at
#: every step and can compound its own mistakes.
_MODE_RUNG: Final[dict[InteractionMode, int]] = {
    InteractionMode.SINGLE_TURN: 0,
    InteractionMode.COUNTERFACTUAL: 1,
    InteractionMode.INTERACTIVE: 2,
    InteractionMode.CLOSED_LOOP: 3,
}

#: Data-sensitivity order: evaluate on public corpora before procedural
#: geometry, and on procedural geometry before de-identified clinical media.
_PHI_RUNG: Final[dict[PhiClass, int]] = {
    PhiClass.PUBLIC: 0,
    PhiClass.PROCEDURAL: 1,
    PhiClass.DEIDENTIFIED_CLINICAL: 2,
}

NO_COMPOSITE_FOOTER: Final = (
    "REFUSED BY DESIGN: no cross-world aggregate, ranking, or scalar collapse is "
    "emitted. Per-world rows are the reportable surface until a published "
    "equivalence artifact exists for this shelf (§2.6)."
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EvalBudget(_Frozen):
    """Trial ceiling the plan must fit inside."""

    max_total_trials: Annotated[int, Field(ge=1, le=1_000_000)]
    max_trials_per_task: Annotated[int, Field(ge=1, le=10_000)] = 30
    max_tasks: Annotated[int, Field(ge=1, le=1_000)] = 64


def difficulty_rank(task: TaskSpec) -> int:
    """Rung of this task on the ladder. Lower is easier; ties are exact.

    Composed from what a package actually declares: interaction mode
    dominates, then the number of hard gates that must all hold, then the
    number of declared perturbations the policy has to survive.
    """
    mode = _MODE_RUNG.get(task.interface.interaction_mode, len(_MODE_RUNG))
    return mode * 100 + min(len(task.verifier.gates), 9) * 10 + min(len(task.perturbations), 9)


def phi_rank(task: TaskSpec) -> int:
    """Rung of this task's data sensitivity. Lower is less sensitive."""
    return _PHI_RUNG.get(task.phi.class_, len(_PHI_RUNG))


def modality_rank(capability: ConfirmedCapability, task: TaskSpec) -> int:
    """0 when the agent declares this task's modality, 1 otherwise.

    Structural satisfaction is necessary but not sufficient for relevance: a
    model that names ``video-laparoscopic`` should meet the laparoscopic world
    before a generically compatible one.
    """
    declared = set(capability.capability.modalities)
    stream_adapters = {stream.adapter for stream in task.interface.streams}
    if task.metadata.modality in declared or (declared & stream_adapters):
        return 0
    return 1


class PlanEntry(_Frozen):
    """One task the plan will run, with the ranking that put it there."""

    task_id: str
    path: str
    task_digest: str
    trials: Annotated[int, Field(ge=1)]
    interface_id: str
    interaction_mode: str
    world_kind: str
    modality: str
    phi_class: str
    gates: tuple[str, ...] = ()
    headline: str = ""
    abstain_ok: bool = True
    metrics_only: bool = False
    modality_rank: Annotated[int, Field(ge=0)] = 1
    difficulty_rank: Annotated[int, Field(ge=0)] = 0
    phi_rank: Annotated[int, Field(ge=0)] = 0
    why: tuple[str, ...] = ()


class EvalPlan(_Frozen):
    """Ordered plan plus the record of every candidate that did not make it."""

    interface: str
    capability_confirmed_by: str
    entries: tuple[PlanEntry, ...]
    total_trials: Annotated[int, Field(ge=0)]
    budget: EvalBudget
    refusals: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _fits_its_own_budget(self) -> Self:
        counted = sum(entry.trials for entry in self.entries)
        if counted != self.total_trials:
            raise TaskContractError(
                f"plan reports {self.total_trials} trials but its entries sum to {counted}"
            )
        if self.total_trials > self.budget.max_total_trials:
            raise TaskContractError(
                f"plan of {self.total_trials} trials exceeds its budget of "
                f"{self.budget.max_total_trials}"
            )
        if len(self.entries) > self.budget.max_tasks:
            raise TaskContractError(
                f"plan of {len(self.entries)} tasks exceeds its budget of "
                f"{self.budget.max_tasks} tasks"
            )
        return self


def _unsatisfied_detail(declared: CapabilitySpec, interface: InterfaceSpec) -> str:
    """Name what is missing. Explanation only — ``satisfies`` still decides.

    Two tasks can share an interface id and still want different schemas, so
    "does not satisfy video-predict" on its own reads like a bug in the
    harness. The refusal has to say which schema the agent does not implement.
    """
    gaps: list[str] = []
    if declared.interface != interface.id:
        gaps.append(f"interface id {interface.id!r} != {declared.interface!r}")
    if interface.interaction_mode not in declared.interaction_modes:
        modes = ", ".join(mode.value for mode in declared.interaction_modes)
        gaps.append(f"interaction mode {interface.interaction_mode.value} not in [{modes}]")
    if interface.protocol_version not in declared.protocol_versions:
        gaps.append(f"protocol version {interface.protocol_version}")
    if not declared.schema_wildcard:
        for label, required, offered in (
            ("observations", interface.observations, declared.observations),
            ("actions", interface.actions, declared.actions),
            ("outputs", interface.outputs, declared.outputs),
            ("features", interface.features, declared.features),
        ):
            missing = sorted(set(required) - set(offered))
            if missing:
                gaps.append(f"missing {label} {missing}")
        own_schemas = set(declared.observations) | set(declared.features)
        adapters = set(declared.modalities)
        for stream in interface.streams:
            if stream.schema_id not in own_schemas or stream.adapter not in adapters:
                gaps.append(
                    f"stream {stream.id} needs schema {stream.schema_id!r} via adapter "
                    f"{stream.adapter!r}"
                )
    return "; ".join(gaps) or "no declared difference (check the capability itself)"


def _candidate(
    capability: ConfirmedCapability,
    path: Path,
    refusals: list[str],
) -> tuple[TaskSpec, Path] | None:
    try:
        task = load_task(path)
    except TaskContractError as exc:
        refusals.append(f"{path.name}: package did not load: {exc}")
        return None
    declared = capability.capability
    if not declared.satisfies(task.interface):
        refusals.append(
            f"{task.id}: capability {declared.interface!r} does not satisfy interface "
            f"{task.interface.id!r} ({task.interface.interaction_mode.value}): "
            f"{_unsatisfied_detail(declared, task.interface)}. The same check "
            "or_audit.eval.bind.assert_bind applies at run time"
        )
        return None
    return task, path


def select_eval_plan(
    capability: ConfirmedCapability,
    *,
    catalog_paths: Sequence[Path | str],
    budget: EvalBudget,
) -> EvalPlan:
    """Rank the catalog for a confirmed capability and fit it to the budget."""
    confirmed = assert_confirmed(capability)
    refusals: list[str] = []
    candidates: list[tuple[TaskSpec, Path]] = []
    for raw in catalog_paths:
        found = _candidate(confirmed, Path(raw), refusals)
        if found is not None:
            candidates.append(found)

    ranked = sorted(
        candidates,
        key=lambda item: (
            modality_rank(confirmed, item[0]),
            difficulty_rank(item[0]),
            phi_rank(item[0]),
            item[0].id,
        ),
    )

    entries: list[PlanEntry] = []
    remaining = budget.max_total_trials
    for task, path in ranked:
        if len(entries) >= budget.max_tasks:
            refusals.append(
                f"{task.id}: task budget exhausted after {len(entries)} tasks; not planned"
            )
            continue
        wanted = min(task.environment.n_eval_episodes, budget.max_trials_per_task)
        trials = min(wanted, remaining)
        if trials < 1:
            refusals.append(
                f"{task.id}: trial budget exhausted after "
                f"{budget.max_total_trials - remaining} trials; not planned"
            )
            continue
        why = [
            f"capability {confirmed.capability.interface} satisfies interface "
            f"{task.interface.id} ({task.interface.interaction_mode.value})",
            (
                f"modality {task.metadata.modality} is declared by the agent"
                if modality_rank(confirmed, task) == 0
                else f"modality {task.metadata.modality} is not declared, "
                "structurally compatible only"
            ),
            f"difficulty rung {difficulty_rank(task)} "
            f"({task.interface.interaction_mode.value}, "
            f"{len(task.verifier.gates)} gate(s), {len(task.perturbations)} perturbation(s))",
            f"phi class {task.phi.class_.value}",
        ]
        if trials < wanted:
            why.append(f"trials trimmed from {wanted} to {trials} by the remaining budget")
        if task.environment.metrics_only:
            why.append("metrics-only world: this row is explicitly not safety-attested")
        entries.append(
            PlanEntry(
                task_id=task.id,
                path=str(path),
                task_digest=tree_digest(path if path.is_dir() else path.parent),
                trials=trials,
                interface_id=task.interface.id,
                interaction_mode=task.interface.interaction_mode.value,
                world_kind=task.environment.kind_key,
                modality=task.metadata.modality,
                phi_class=task.phi.class_.value,
                gates=tuple(gate.id for gate in task.verifier.gates),
                headline=task.verifier.headline,
                abstain_ok=task.verifier.abstain_ok,
                metrics_only=task.environment.metrics_only,
                modality_rank=modality_rank(confirmed, task),
                difficulty_rank=difficulty_rank(task),
                phi_rank=phi_rank(task),
                why=tuple(why),
            )
        )
        remaining -= trials

    return EvalPlan(
        interface=confirmed.capability.interface,
        capability_confirmed_by=confirmed.confirmed_by,
        entries=tuple(entries),
        total_trials=sum(entry.trials for entry in entries),
        budget=budget,
        refusals=tuple(refusals),
    )


def narrate_plan(plan: EvalPlan) -> str:
    """Render a plan: refusals first, then gates, then per-world rows."""
    lines: list[str] = [
        f"EVAL PLAN for capability {plan.interface} (confirmed by {plan.capability_confirmed_by})",
        "",
    ]
    lines.append("REFUSED CANDIDATES")
    if plan.refusals:
        lines.extend(f"  - {refusal}" for refusal in plan.refusals)
    else:
        lines.append("  (none)")
    lines.extend(["", "SAFETY GATES AND ABSTENTION"])
    for entry in plan.entries:
        gates = ", ".join(entry.gates) or "NONE"
        label = " [METRICS-ONLY, NOT SAFETY-ATTESTED]" if entry.metrics_only else ""
        lines.append(
            f"  {entry.task_id}: gates {gates}; abstention "
            f"{'permitted' if entry.abstain_ok else 'not permitted'}{label}"
        )
    if not plan.entries:
        lines.append("  (no task planned)")
    lines.extend(["", "PER-WORLD ROWS"])
    for index, entry in enumerate(plan.entries, start=1):
        lines.append(
            f"  {index}. {entry.task_id} world={entry.world_kind} "
            f"modality={entry.modality} mode={entry.interaction_mode} "
            f"phi={entry.phi_class} trials={entry.trials} "
            f"headline={entry.headline} digest={entry.task_digest}"
        )
        lines.extend(f"       why: {reason}" for reason in entry.why)
    lines.extend(
        [
            "",
            f"TRIALS {plan.total_trials} of at most {plan.budget.max_total_trials}",
            "",
            NO_COMPOSITE_FOOTER,
        ]
    )
    return "\n".join(lines)


def narrate_results(job_results: Sequence[JobResult]) -> str:
    """Render finished jobs: gates first, abstention next, per-world rows after.

    Ordering is the honesty claim. A reader who stops after the first section
    has seen the safety gates; a reader who stops after the second has seen
    what could not be assessed. Neither can mistake a headline rate for a
    score, and there is no composite anywhere in the output.
    """
    cards = [scorecard_data(result) for result in job_results]
    lines: list[str] = ["SAFETY GATES"]
    if not cards:
        lines.append("  (no result)")
    for card in cards:
        if not card["gates"]:
            lines.append(
                f"  {card['task_id']}: NO GATES"
                + (
                    " (metrics-only world: explicitly not safety-attested)"
                    if card["metrics_only"]
                    else ""
                )
            )
            continue
        for gate in card["gates"]:
            lines.append(
                f"  {card['task_id']} / {gate['id']}: pass {gate['pass']}, "
                f"fail {gate['fail']}, not assessable {gate['not_assessable']}, "
                f"not applicable {gate['not_applicable']}"
            )
    lines.extend(["", "ABSTENTION"])
    if not cards:
        lines.append("  (no result)")
    for card, result in zip(cards, job_results, strict=True):
        unassessable_gates = sum(gate["not_assessable"] for gate in card["gates"])
        lines.append(
            f"  {card['task_id']}: headline {card['headline']} unassessable in "
            f"{result.headline_unassessable} of {result.n} trial(s); "
            f"{unassessable_gates} unassessable gate outcome(s); "
            f"{result.any_gate_failed} trial(s) failed a hard gate"
        )
    lines.extend(["", "PER-WORLD ROWS"])
    if not cards:
        lines.append("  (no result)")
    for card, result in zip(cards, job_results, strict=True):
        engine = card["world_engine"] or {}
        engine_name = str(engine.get("engine") or "unknown")
        backend = str(engine.get("backend") or "unknown")
        if result.world_engine is not None:
            engine_name = result.world_engine.engine or engine_name
            backend = result.world_engine.backend
        stub = " [SYNTHETIC STAND-IN, NOT PHYSICAL EVIDENCE]" if backend == "synthetic-stub" else ""
        label = " [METRICS-ONLY, NOT SAFETY-ATTESTED]" if card["metrics_only"] else ""
        lines.append(
            f"  {card['task_id']}@{card['task_version']} world={card['world_pin'] or 'none'} "
            f"engine={engine_name} backend={backend} agent={card['agent_identity']} "
            f"n={result.n} {card['headline']}: true {result.headline_true}, "
            f"false {result.headline_false}, unassessable {result.headline_unassessable} "
            f"head={result.head}{stub}{label}"
        )
    lines.extend(["", NO_COMPOSITE_FOOTER])
    return "\n".join(lines)
