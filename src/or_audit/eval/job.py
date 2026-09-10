"""Job directory: the Harbor trial/job layout, with a vector instead of reward.txt."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from or_audit.audit.canonical import digest
from or_audit.errors import TaskContractError
from or_audit.eval.agent import AgentPackage
from or_audit.eval.contracts import MetricKind
from or_audit.eval.enums import WorldKind
from or_audit.eval.integrity import tree_digest
from or_audit.eval.split import validate_subgroups
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
    case_id: str = ""
    patient_id: str = ""
    site_id: str = ""
    episode_id: str = ""
    subgroups: dict[str, str] = Field(default_factory=dict)

    @field_validator("subgroups", mode="before")
    @classmethod
    def _check_subgroups(cls, value: object) -> dict[str, str]:
        return validate_subgroups(value) if value is not None else {}


class TrialBinding(BaseModel):
    """Head-covered trial metadata binding case, patient, site, and cohort subgroups."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = ""
    patient_id: str = ""
    site_id: str = ""
    episode_id: str = ""
    subgroups: dict[str, str] = Field(default_factory=dict)

    @field_validator("subgroups", mode="before")
    @classmethod
    def _check_subgroups(cls, value: object) -> dict[str, str]:
        return validate_subgroups(value) if value is not None else {}


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
    independent_cases: Annotated[int, Field(ge=1)] | None = None
    split_manifest_digest: str = ""
    split: str = ""
    independent_case_unit: str = ""
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
    independent_cases: int | None = None,
    split_manifest_digest: str = "",
    split: str = "",
    independent_case_unit: str = "",
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
        independent_cases=independent_cases,
        split_manifest_digest=split_manifest_digest,
        split=split,
        independent_case_unit=independent_case_unit,
        gate_outcome=(
            "failed"
            if gate_failed
            else "not-assessable"
            if gate_unassessable or not task.verifier.gates
            else "passed"
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


def _atomic_write_text(path: Path, text: str) -> None:
    """Write crash-atomically: readers never see a half-written file.

    Uses mkstemp (O_EXCL) in the target directory so a pre-created symlink
    cannot redirect the write, then renames over the target.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def write_job_skeleton(
    out: Path,
    *,
    config: dict[str, Any],
    task_dir: Path,
    agent_dir: Path | None,
    task_digest: str,
    agent_digest: str,
) -> None:
    """Bundle packages and config before the first trial runs.

    A killed run leaves the bundle, the config, and every completed trial
    behind, which is exactly what resume needs. Result and scorecards are
    still written only on completion.
    """
    out.mkdir(parents=True, exist_ok=True)
    bundle = out / "bundle"
    task_target = bundle / "task"
    _copy_package(task_dir, task_target)
    if tree_digest(task_target) != task_digest:
        raise TaskContractError("copied task package digest does not match")
    if agent_dir is not None:
        agent_target = bundle / "agent"
        _copy_package(agent_dir, agent_target)
        if tree_digest(agent_target) != agent_digest:
            raise TaskContractError("copied agent package digest does not match")
    runtime_identity = config.get("runtime_identity", "")
    manifest = {
        "format_version": "2",
        "task": {"path": "bundle/task", "digest": task_digest},
        "agent": (
            {"path": "bundle/agent", "digest": agent_digest}
            if agent_dir is not None
            else {"path": None, "digest": agent_digest}
        ),
        "runtime_identity": runtime_identity,
    }
    _atomic_write_text(out / "bundle.json", json.dumps(manifest, indent=2) + "\n")
    _atomic_write_text(out / "config.json", json.dumps(config, indent=2) + "\n")


def write_trial(
    out: Path,
    task_id: str,
    trial: TrialRecord,
    projection_identity: str,
    world_engine: dict[str, Any] | None = None,
) -> None:
    """Persist one completed trial atomically (crash-safe resume unit).

    Writes trajectory, projection, provenance, and binding first, then writes
    result.json last as the commit marker for the trial.
    """
    trial_dir = out / f"trial-{task_id}-{trial.seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        trial_dir / "trajectory.json", json.dumps(list(trial.trajectory), indent=2) + "\n"
    )
    if trial.projection is not None:
        _atomic_write_text(
            trial_dir / "projection.json",
            json.dumps(
                {
                    "projection": trial.projection,
                    "projection_identity": projection_identity,
                    "projection_spec_digest": trial.projection_spec_digest,
                }
            )
            + "\n",
        )
    if world_engine is not None:
        _atomic_write_text(
            trial_dir / "provenance.json",
            json.dumps(world_engine, indent=2) + "\n",
        )
    binding = TrialBinding(
        case_id=trial.case_id,
        patient_id=trial.patient_id,
        site_id=trial.site_id,
        episode_id=trial.episode_id,
        subgroups=trial.subgroups,
    )
    if (
        binding.case_id
        or binding.patient_id
        or binding.site_id
        or binding.episode_id
        or binding.subgroups
    ):
        _atomic_write_text(
            trial_dir / "binding.json",
            json.dumps(binding.model_dump(mode="json"), indent=2) + "\n",
        )
    _atomic_write_text(
        trial_dir / "result.json", json.dumps(_vector_dict(trial.vector), indent=2) + "\n"
    )


def read_partial_trials(
    out: Path, task_id: str, *, require_binding: bool = False
) -> tuple[list[TrialRecord], dict[str, Any] | None]:
    """Read committed trials from an interrupted job directory.

    Reads trial directories created by :func:`write_trial` when the job-level
    ``result.json`` is missing. Corrupt committed trials raise
    :class:`TaskContractError`; incomplete uncommitted directories (missing
    ``result.json``) are ignored as interrupted writes.
    """
    records: list[TrialRecord] = []
    observed_provenance: dict[str, Any] | None = None
    trials_with_prov = 0
    trials_without_prov = 0
    trials_with_binding = 0
    trials_without_binding = 0
    for trial_dir in sorted(out.glob(f"trial-{task_id}-*")):
        vector_path = trial_dir / "result.json"
        trajectory_path = trial_dir / "trajectory.json"
        # Incomplete uncommitted trial directory: result.json is the commit marker.
        if not vector_path.is_file():
            continue
        try:
            seed = int(trial_dir.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        # A trial with result.json was committed; a missing trajectory is corruption.
        if not trajectory_path.is_file():
            raise TaskContractError(
                f"trial dir {trial_dir.name} is corrupt: committed trial is missing trajectory.json"
            )
        try:
            vector = TrialVector.model_validate(json.loads(vector_path.read_text(encoding="utf-8")))
            trajectory = ProceduralTrace.model_validate(
                json.loads(trajectory_path.read_text(encoding="utf-8"))
            )
        except ValueError as exc:
            raise TaskContractError(f"trial dir {trial_dir.name} is corrupt: {exc}") from exc
        prov_path = trial_dir / "provenance.json"
        if prov_path.is_file():
            trials_with_prov += 1
            try:
                prov_data = json.loads(prov_path.read_text(encoding="utf-8"))
                prov_model = WorldEngineProvenance.model_validate(prov_data)
            except (json.JSONDecodeError, ValueError) as exc:
                raise TaskContractError(
                    f"trial dir {trial_dir.name} is corrupt: invalid provenance.json ({exc})"
                ) from exc
            prov_dict = prov_model.model_dump(mode="json")
            if observed_provenance is not None and observed_provenance != prov_dict:
                raise TaskContractError(
                    f"trial dir {trial_dir.name} has conflicting provenance "
                    f"against prior trials in {out.name}"
                )
            observed_provenance = prov_dict
        else:
            trials_without_prov += 1
        projection_path = trial_dir / "projection.json"
        projection = None
        projection_spec_digest = ""
        if projection_path.is_file():
            try:
                payload = json.loads(projection_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError) as exc:
                raise TaskContractError(
                    f"trial dir {trial_dir.name} is corrupt: invalid projection.json ({exc})"
                ) from exc
            if not isinstance(payload, dict):
                raise TaskContractError(
                    f"trial dir {trial_dir.name} is corrupt: projection.json is not an object"
                )
            projection = payload.get("projection")
            projection_spec_digest = str(payload.get("projection_spec_digest", ""))
        binding_path = trial_dir / "binding.json"
        binding = TrialBinding()
        if binding_path.is_file():
            trials_with_binding += 1
            try:
                b_raw = json.loads(binding_path.read_text(encoding="utf-8"))
                binding = TrialBinding.model_validate(b_raw)
            except (json.JSONDecodeError, ValueError) as exc:
                raise TaskContractError(
                    f"trial dir {trial_dir.name} is corrupt: invalid binding.json ({exc})"
                ) from exc
        else:
            trials_without_binding += 1
            if require_binding:
                raise TaskContractError(
                    f"trial dir {trial_dir.name} is missing binding.json required by job split"
                )
        records.append(
            TrialRecord(
                seed=seed,
                vector=vector,
                trajectory=trajectory,
                projection=projection,
                projection_spec_digest=projection_spec_digest,
                case_id=binding.case_id,
                patient_id=binding.patient_id,
                site_id=binding.site_id,
                episode_id=binding.episode_id,
                subgroups=binding.subgroups,
            )
        )
    if trials_with_prov > 0 and trials_without_prov > 0:
        raise TaskContractError(
            f"job dir {out.name} has mixed provenance: {trials_with_prov} trial(s) have "
            f"provenance.json while {trials_without_prov} trial(s) are missing it"
        )
    if trials_with_binding > 0 and trials_without_binding > 0:
        raise TaskContractError(
            f"job dir {out.name} has mixed binding metadata: {trials_with_binding} trial(s) have "
            f"binding.json while {trials_without_binding} trial(s) are missing it"
        )
    return sorted(records, key=lambda record: record.seed), observed_provenance


def write_job(
    out: Path,
    *,
    config: dict[str, Any],
    result: JobResult,
    task_dir: Path,
    agent_dir: Path | None,
) -> None:
    """Write a Harbor-shaped job directory."""
    write_job_skeleton(
        out,
        config=config,
        task_dir=task_dir,
        agent_dir=agent_dir,
        task_digest=result.task_digest,
        agent_digest=result.agent_digest,
    )
    for trial in result.trials:
        write_trial(
            out,
            result.task_id,
            trial,
            result.projection_identity,
            world_engine=(
                result.world_engine.model_dump(mode="json") if result.world_engine else None
            ),
        )
    from or_audit.eval.scorecard import write_scorecards

    world_engine = config.get("world_engine")
    write_scorecards(
        out,
        result,
        world_engine=world_engine if isinstance(world_engine, dict) else None,
    )
    _atomic_write_text(
        out / "result.json", json.dumps(result.model_dump(mode="json"), indent=2) + "\n"
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
