"""Job directory: the Harbor trial/job layout, with a vector instead of reward.txt."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from or_audit.audit.canonical import digest
from or_audit.errors import TaskContractError
from or_audit.eval.agent import AgentPackage
from or_audit.eval.contracts import MetricKind
from or_audit.eval.enums import WorldKind
from or_audit.eval.integrity import tree_digest
from or_audit.eval.task import TaskSpec
from or_audit.eval.trace import ProceduralTrace
from or_audit.eval.vector import TrialVector


class TrialRecord(BaseModel):
    """One trial on disk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: Annotated[int, Field(ge=0)]
    vector: TrialVector
    trajectory: ProceduralTrace = Field(default_factory=lambda: ProceduralTrace(()))
    projection: float | None = None
    projection_spec_digest: str = ""


class WorldEngineProvenance(BaseModel):
    """Head-covered attestation of which world/backend produced observations.

    Bound into ``JobResult.world_engine`` so replay/export can attest the
    backend from the authenticated result, never from a mutable
    ``config.json``. ``backend`` is a closed literal over the three declared
    states (``real`` / ``synthetic-stub`` / ``unknown``); any other value is a
    validation error, and unknown reporter fields are rejected, so a
    typo-ad-hoc value cannot silently pass the export provenance gate.

    ``adapter_id``/``adapter_digest`` record the world adapter that produced
    the observations, taken from the kernel's world-kind registry rather than
    from the bridge's own report. A third-party adapter that changes content
    therefore changes the head, even when the task and world pin are unchanged.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    engine: str
    backend: Literal["real", "synthetic-stub", "unknown"]
    backend_version: str = ""
    world_pin: str = ""
    adapter_id: str = ""
    adapter_digest: str = ""
    #: Tier-0 honesty label carried from ``WorldSpec.metrics_only``: this row is
    #: explicitly not safety-attested. Head-covered so a published artifact
    #: cannot drop the label after the fact.
    metrics_only: bool = False


class JobResult(BaseModel):
    """Aggregated job. Never a lone mean-reward."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format_version: str = "3"
    task_id: str
    task_version: str
    agent_identity: str
    world_pin: str
    #: Head-covered engine provenance recorded at run time (engine, backend,
    #: backend_version). Binding it here means export/replay can attest the
    #: backend from the authenticated result, never from a mutable config file.
    world_engine: WorldEngineProvenance | None = None
    interface_id: str = ""
    interaction_mode: str = ""
    runtime_identity: str = ""
    projection_identity: str = ""
    task_digest: str
    agent_digest: str
    n: Annotated[int, Field(ge=1)]
    headline: str
    trials: tuple[TrialRecord, ...]
    headline_true: int
    headline_false: int
    headline_unassessable: int
    any_gate_failed: int
    unique_trajectories: Annotated[int, Field(ge=1)] | None = None
    duplicate_trajectories: Annotated[int, Field(ge=0)] | None = None
    gate_outcome: Literal["passed", "failed", "not-assessable", "unknown"] = "unknown"
    claim_footer: str = ""
    head: str = ""

    @model_validator(mode="after")
    def _counts_match(self) -> Self:
        if len(self.trials) != self.n:
            msg = f"job n={self.n} but {len(self.trials)} trials"
            raise TaskContractError(msg)
        return self


def agent_identity(agent: AgentPackage) -> str:
    """Stable identity for a trial row."""
    pin = agent.weights_pin or "none"
    return f"{agent.id}@{agent.agent_version}+{pin}"


def _vector_dict(vector: TrialVector) -> dict[str, Any]:
    return vector.model_dump(mode="json")


def job_head_payload(result: JobResult) -> dict[str, Any]:
    """The bytes that define replay identity. Excludes ``head`` itself."""
    dumped = result.model_dump(mode="json")
    dumped.pop("head", None)
    return dumped


def compute_head(result: JobResult) -> str:
    """SHA-256 of the canonical job payload."""
    return digest(job_head_payload(result))


def verify_head(result: JobResult) -> bool:
    """Return whether ``result.head`` is the canonical head of its own payload."""
    return compute_head(result) == result.head


def assert_publishable(
    task: TaskSpec,
    trials: tuple[TrialRecord, ...],
    claim_footer: str,
) -> None:
    """Refuse a result that would hide injury or drop the AngioStress claim boundary."""
    metric_ids = {m.id for m in task.verifier.metrics}
    if "safe_success" in metric_ids:
        for trial in trials:
            if trial.vector.metric("safe_success") is None:
                msg = (
                    "refusing to publish a result that omits safe_success; "
                    "that is the CathSim failure mode BUILD.md forbids"
                )
                raise TaskContractError(msg)
    if task.environment.kind is WorldKind.ANGIOSTRESS_CONTRACT and not claim_footer:
        msg = (
            "refusing to publish an AngioStress result without the claim footer; "
            "BUILD.md P2 treats a missing boundary as an invalid scorecard"
        )
        raise TaskContractError(msg)


def assemble_job_result(
    *,
    task: TaskSpec,
    agent: AgentPackage,
    trials: tuple[TrialRecord, ...],
    task_digest: str,
    agent_digest: str,
    claim_footer: str = "",
    world_engine: dict[str, Any] | None = None,
) -> JobResult:
    """Build a publishable job result and stamp its head."""
    assert_publishable(task, trials, claim_footer)
    world_engine_model: WorldEngineProvenance | None = None
    if world_engine is not None:
        world_engine_model = WorldEngineProvenance(**world_engine)
    headline_definition = task.metric(task.verifier.headline)
    headline_true = 0
    headline_false = 0
    headline_unassessable = 0
    gate_failed = 0
    gate_unassessable = 0
    for trial in trials:
        value = trial.vector.headline.value
        if value is None:
            headline_unassessable += 1
        elif headline_definition.kind is MetricKind.BOOLEAN:
            if value is True:
                headline_true += 1
            else:
                headline_false += 1
        if trial.vector.any_gate_failed:
            gate_failed += 1
        if trial.vector.any_gate_unassessable:
            gate_unassessable += 1
    unique_trajectories = len({digest(list(trial.trajectory)) for trial in trials})
    result = JobResult(
        task_id=task.id,
        task_version=task.task_version,
        agent_identity=agent_identity(agent),
        world_pin=task.environment.world_pin,
        world_engine=world_engine_model,
        interface_id=task.interface.id,
        interaction_mode=task.harness.interaction_mode.value,
        runtime_identity=agent.runtime_identity,
        projection_identity=task.projection.identity if task.projection else "",
        task_digest=task_digest,
        agent_digest=agent_digest,
        n=len(trials),
        headline=task.verifier.headline,
        trials=trials,
        headline_true=headline_true,
        headline_false=headline_false,
        headline_unassessable=headline_unassessable,
        any_gate_failed=gate_failed,
        unique_trajectories=unique_trajectories,
        duplicate_trajectories=len(trials) - unique_trajectories,
        gate_outcome=(
            "failed" if gate_failed else "not-assessable" if gate_unassessable else "passed"
        ),
        claim_footer=claim_footer,
    )
    return result.model_copy(update={"head": compute_head(result)})


def resolve_bundle_path(job_dir: Path, raw: object, *, label: str) -> Path:
    """Resolve a relative bundle path without allowing traversal or symlink escape."""
    root = job_dir.resolve()
    path = Path(str(raw))
    if path.is_absolute():
        raise TaskContractError(f"bundle {label} path must be relative: {path}")
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise TaskContractError(f"bundle {label} path escapes job directory: {path}") from exc
    return candidate


def _copy_package(source: Path, target: Path) -> None:
    """Copy a package into a job bundle, refusing a target nested inside it.

    A nested target makes ``copytree`` walk the tree it is writing into, which
    does not fail cleanly: it builds a path one level deeper per iteration
    until the filesystem refuses the name, then reports a wall of nested paths
    that names neither the cause nor the fix. Someone running
    ``surgeval conformance ./task --out ./task/conf`` deserves a sentence.
    """
    resolved_source = source.resolve()
    resolved_target = target.resolve()
    if resolved_source == resolved_target:
        return
    if resolved_source in resolved_target.parents:
        raise TaskContractError(
            f"output directory {target} is inside the package being copied ({source}): "
            "the bundle copy would recurse into its own output. Fix: put --out somewhere "
            "outside the package directory."
        )
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        target,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
            "*.pyc",
            "*.pyo",
        ),
    )


def write_job(
    out: Path,
    *,
    config: dict[str, Any],
    result: JobResult,
    task_dir: Path,
    agent_dir: Path | None,
) -> None:
    """Write a Harbor-shaped job directory."""
    out.mkdir(parents=True, exist_ok=True)
    bundle = out / "bundle"
    task_target = bundle / "task"
    _copy_package(task_dir, task_target)
    if tree_digest(task_target) != result.task_digest:
        raise TaskContractError("copied task package digest does not match result")
    if agent_dir is not None:
        agent_target = bundle / "agent"
        _copy_package(agent_dir, agent_target)
        if tree_digest(agent_target) != result.agent_digest:
            raise TaskContractError("copied agent package digest does not match result")
    manifest = {
        "format_version": "2",
        "task": {"path": "bundle/task", "digest": result.task_digest},
        "agent": (
            {"path": "bundle/agent", "digest": result.agent_digest}
            if agent_dir is not None
            else {"path": None, "digest": result.agent_digest}
        ),
        "runtime_identity": result.runtime_identity,
    }
    (out / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (out / "result.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    for trial in result.trials:
        trial_dir = out / f"trial-{result.task_id}-{trial.seed}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(
            json.dumps(_vector_dict(trial.vector), indent=2) + "\n", encoding="utf-8"
        )
        (trial_dir / "trajectory.json").write_text(
            json.dumps(list(trial.trajectory), indent=2) + "\n", encoding="utf-8"
        )
        if trial.projection is not None:
            (trial_dir / "projection.json").write_text(
                json.dumps(
                    {
                        "projection": trial.projection,
                        "projection_identity": result.projection_identity,
                        "projection_spec_digest": trial.projection_spec_digest,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
    from or_audit.eval.scorecard import write_scorecards

    world_engine = config.get("world_engine")
    write_scorecards(
        out,
        result,
        world_engine=world_engine if isinstance(world_engine, dict) else None,
    )


def read_job_result(out: Path) -> JobResult:
    """Load ``result.json`` from a job directory."""
    path = out / "result.json"
    if not path.is_file():
        msg = f"missing result.json in {out}"
        raise TaskContractError(msg)
    return JobResult.model_validate(json.loads(path.read_text(encoding="utf-8")))


def read_job_config(out: Path) -> dict[str, Any]:
    """Load ``config.json`` from a job directory."""
    path = out / "config.json"
    if not path.is_file():
        msg = f"missing config.json in {out}"
        raise TaskContractError(msg)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"config.json in {out} is not an object"
        raise TaskContractError(msg)
    return data
