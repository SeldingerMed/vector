"""Cartesian jobs: agents x tasks x n, written as a parent directory of pair jobs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from or_audit.audit.canonical import digest
from or_audit.errors import TaskContractError
from or_audit.eval.agent import AgentPackage
from or_audit.eval.contracts import InteractionMode
from or_audit.eval.gym_world import GymFactory
from or_audit.eval.job import JobResult
from or_audit.eval.job_config import BUILTIN_RANDOM, EvaluationStageSpec, ResolvedJob
from or_audit.eval.loader import load_agent, load_task
from or_audit.eval.predict import load_items
from or_audit.eval.runner import assert_trial_capacity, builtin_random_agent, replay_job, run_job
from or_audit.eval.task import ProjectionSpec, TaskSpec


class PairRecord(BaseModel):
    """One (task, agent) cell in a cartesian job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    task_version: str
    agent_id: str
    dir: str
    n: Annotated[int, Field(ge=1)]
    head: str


class CartesianManifest(BaseModel):
    """Parent-directory index. Pair jobs underneath still have their own heads."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format_version: str
    id: str
    n: Annotated[int, Field(ge=1)] | None = None
    pairs: tuple[PairRecord, ...]
    projection: ProjectionSpec | None = None
    stage: EvaluationStageSpec | None = None
    observed_units: Annotated[int, Field(ge=1)] | None = None
    gate_outcome: Literal["passed", "failed", "not-assessable", "unknown"] | None = None
    head: str = ""


def pair_dir_name(task_id: str, agent_id: str) -> str:
    """Filesystem-safe directory for one pair. Slash in ``org/name`` becomes a hyphen."""
    return f"{task_id}__{agent_id.replace('/', '-')}"


def manifest_head_payload(manifest: CartesianManifest) -> dict[str, Any]:
    """Bytes that define cartesian replay identity. Excludes ``head`` itself."""
    dumped = manifest.model_dump(mode="json")
    dumped.pop("head", None)
    if manifest.stage is None:
        # Preserve v0.3 manifest heads written before optional stage metadata.
        dumped.pop("stage", None)
        dumped.pop("observed_units", None)
        dumped.pop("gate_outcome", None)
    elif manifest.observed_units is None:
        dumped.pop("observed_units", None)
    return dumped


def compute_manifest_head(manifest: CartesianManifest) -> str:
    """SHA-256 of the canonical cartesian payload."""
    return digest(manifest_head_payload(manifest))


def read_manifest(out: Path) -> CartesianManifest:
    """Load ``manifest.json`` from a cartesian parent directory."""
    path = out / "manifest.json"
    if not path.is_file():
        msg = f"missing manifest.json in {out}"
        raise TaskContractError(msg)
    return CartesianManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def write_manifest(out: Path, manifest: CartesianManifest) -> None:
    """Write ``manifest.json``."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )


def iter_job_dirs(path: Path) -> tuple[Path, ...]:
    """Job directories to export or replay: cartesian pairs, or a single job."""
    if (path / "manifest.json").is_file():
        manifest = read_manifest(path)
        if not manifest.pairs:
            msg = f"cartesian job {path} lists no pairs"
            raise TaskContractError(msg)
        return tuple(path / pair.dir for pair in manifest.pairs)
    if (path / "result.json").is_file():
        return (path,)
    msg = (
        f"{path} is neither a job directory nor a cartesian job "
        f"(missing result.json and manifest.json)"
    )
    raise TaskContractError(msg)


def _agent_from_ref(ref: str) -> tuple[AgentPackage, Path | None]:
    if ref == BUILTIN_RANDOM:
        return builtin_random_agent(), None
    path = Path(ref)
    agent_dir = path if path.is_dir() else path.parent
    return load_agent(path), agent_dir


def _independent_case_count(
    planned: list[tuple[Path, TaskSpec, AgentPackage, Path | None, str, int | None]],
    stage: EvaluationStageSpec,
) -> int:
    """Count task-owned cases once, even when several agents evaluate them."""
    tasks = {
        (str(root), task.id): (root, task, trials) for root, task, _, _, _, trials in planned
    }.values()
    modes = {task.harness.interaction_mode for _, task, _ in tasks}
    if len(modes) != 1:
        raise TaskContractError(f"stage {stage.name} cannot mix interaction modes")
    mode = next(iter(modes))
    if mode is InteractionMode.CLOSED_LOOP:
        if stage.independent_case_key != "$seed":
            raise TaskContractError(
                f"closed-loop stage {stage.name} independent_case_key must be '$seed'"
            )
        return len(
            {
                (stage.independent_case_groups.get(task.id, task.id), seed)
                for _, task, trials in tasks
                for seed in range(trials or 0)
            }
        )
    if stage.independent_case_key == "$seed":
        raise TaskContractError(
            f"dataset-backed stage {stage.name} must name an input field as independent_case_key"
        )
    cases: set[str] = set()
    for root, task, trials in tasks:
        for item in load_items(root / task.environment.inputs_path)[:trials]:
            if stage.independent_case_key not in item:
                raise TaskContractError(
                    f"task {task.id} input {item['id']!r} has no independent-case field "
                    f"{stage.independent_case_key!r}"
                )
            cases.add(
                json.dumps(item[stage.independent_case_key], sort_keys=True, separators=(",", ":"))
            )
    return len(cases)


def _observed_units(result: JobResult, source: str) -> int:
    if source == "trials":
        return result.n
    if source == "trajectory-steps":
        return sum(len(trial.trajectory) for trial in result.trials)
    metric_id = source.removeprefix("metric:")
    total = 0
    for trial in result.trials:
        metric = trial.vector.metric(metric_id)
        value = metric.value if metric is not None else None
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or value < 0
            or int(value) != value
        ):
            raise TaskContractError(
                f"stage unit source {source!r} must resolve to a non-negative integer "
                f"for every trial"
            )
        total += int(value)
    return total


def run_cartesian_job(
    resolved: ResolvedJob,
    *,
    out: Path,
    n: int | None = None,
    gym_factory: GymFactory | None = None,
) -> CartesianManifest:
    """Run every (task, agent) pair into ``out / pair_dir`` and write a manifest."""
    n_eval = n if n is not None else resolved.config.n
    if n_eval is not None and n_eval < 1:
        msg = f"job {resolved.config.id} n must be >= 1, got {n_eval}"
        raise TaskContractError(msg)

    planned: list[tuple[Path, TaskSpec, AgentPackage, Path | None, str, int | None]] = []
    used_dirs: set[str] = set()
    for task_ref, task_path in zip(resolved.config.tasks, resolved.task_paths, strict=True):
        task_dir = task_path if task_path.is_dir() else task_path.parent
        task: TaskSpec = load_task(task_path)
        pair_trials = n_eval if n_eval is not None else resolved.config.task_trials.get(task_ref)
        if resolved.config.projection is not None and task.projection != resolved.config.projection:
            raise TaskContractError(
                f"job {resolved.config.id} projection does not match the "
                f"task-declared projection for {task.id}"
            )
        for agent_ref in resolved.agent_refs:
            agent, agent_dir = _agent_from_ref(agent_ref)
            dirname = pair_dir_name(task.id, agent.id)
            if dirname in used_dirs:
                msg = (
                    f"job {resolved.config.id} pair directory {dirname!r} collides; "
                    f"two (task, agent) cells mapped to the same path"
                )
                raise TaskContractError(msg)
            used_dirs.add(dirname)
            planned.append((task_dir, task, agent, agent_dir, dirname, pair_trials))

    stage = resolved.config.stage
    if stage is not None:
        if any(trials is None for *_, trials in planned):
            raise TaskContractError(f"stage {stage.name} must declare job n or task_trials")
        scheduled = sum(trials or 0 for *_, trials in planned)
        if stage.unit_source == "trials" and scheduled != stage.target_units:
            raise TaskContractError(
                f"stage {stage.name} targets {stage.target_units} {stage.evaluation_unit} "
                f"but tasks x agents x n schedules {scheduled}"
            )
        supported_scenarios = {
            value
            for _, task, _, _, _, _ in planned
            for value in (task.id, *(scenario.id for scenario in task.scenarios))
        }
        unsupported_scenarios = set(stage.scenarios) - supported_scenarios
        if unsupported_scenarios:
            raise TaskContractError(
                f"stage {stage.name} names unsupported scenarios {sorted(unsupported_scenarios)}"
            )
        supported_events = {
            value
            for _, task, _, _, _, _ in planned
            for perturbation in task.perturbations
            for value in (perturbation.id, perturbation.kind)
        }
        unsupported_events = set(stage.event_injections) - supported_events
        if unsupported_events:
            raise TaskContractError(
                f"stage {stage.name} names unsupported injected events {sorted(unsupported_events)}"
            )
        task_ids = {task.id for _, task, _, _, _, _ in planned}
        if stage.independent_case_groups and set(stage.independent_case_groups) != task_ids:
            raise TaskContractError(
                f"stage {stage.name} independent_case_groups keys must exactly match task ids"
            )
        for task_dir, task, _, _, _, trials in planned:
            assert_trial_capacity(task, task_dir, trials or 0)
        observed_cases = _independent_case_count(planned, stage)
        if observed_cases != stage.independent_cases:
            raise TaskContractError(
                f"stage {stage.name} declares {stage.independent_cases} independent cases but "
                f"{stage.independent_case_key!r} identifies {observed_cases}"
            )

    pairs: list[PairRecord] = []
    outcomes: list[str] = []
    observed_units = 0
    for task_dir, task, agent, agent_dir, dirname, pair_trials in planned:
        result: JobResult = run_job(
            task=task,
            task_dir=task_dir,
            agent=agent,
            agent_dir=agent_dir,
            out=out / dirname,
            n=pair_trials,
            gym_factory=gym_factory,
        )
        pairs.append(
            PairRecord(
                task_id=result.task_id,
                task_version=result.task_version,
                agent_id=agent.id,
                dir=dirname,
                n=result.n,
                head=result.head,
            )
        )
        outcomes.append(result.gate_outcome)
        if stage is not None:
            observed_units += _observed_units(result, stage.unit_source)

    if stage is not None and observed_units != stage.target_units:
        raise TaskContractError(
            f"stage {stage.name} observed {observed_units} {stage.evaluation_unit}, "
            f"expected exactly {stage.target_units}"
        )

    gate_outcome = None
    if stage is not None:
        gate_outcome = (
            "failed"
            if "failed" in outcomes
            else "not-assessable"
            if "not-assessable" in outcomes
            else "unknown"
            if "unknown" in outcomes
            else "passed"
        )

    manifest = CartesianManifest(
        format_version=resolved.config.format_version,
        id=resolved.config.id,
        n=n_eval,
        pairs=tuple(pairs),
        projection=resolved.config.projection,
        stage=stage,
        observed_units=observed_units if stage is not None else None,
        gate_outcome=gate_outcome,
    )
    stamped = manifest.model_copy(update={"head": compute_manifest_head(manifest)})
    write_manifest(out, stamped)
    return stamped


def replay_cartesian(
    out: Path,
    *,
    load_task: Callable[[Path], TaskSpec],
    load_agent: Callable[[Path], AgentPackage],
    gym_factory: GymFactory | None = None,
) -> CartesianManifest:
    """Replay every pair job and require the stored manifest head to match."""
    previous = read_manifest(out)
    for pair in previous.pairs:
        rerun = replay_job(
            out / pair.dir,
            load_task=load_task,
            load_agent=load_agent,
            gym_factory=gym_factory,
        )
        if rerun.head != pair.head:
            msg = (
                f"cartesian pair {pair.dir} head mismatch: manifest {pair.head} reran {rerun.head}"
            )
            raise TaskContractError(msg)
    current = read_manifest(out)
    if compute_manifest_head(current) != current.head:
        msg = "cartesian manifest stamped a head that does not match its payload"
        raise TaskContractError(msg)
    if current.head != previous.head:
        msg = f"cartesian replay head mismatch: stored {previous.head} reran {current.head}"
        raise TaskContractError(msg)
    return current
