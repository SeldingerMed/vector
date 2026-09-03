"""``job.toml``: Harbor's cartesian product of agents x tasks x n."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from or_audit.errors import TaskContractError
from or_audit.eval.loader import _read_toml
from or_audit.eval.task import ProjectionSpec, Slug

NonEmptyPath = Annotated[str, StringConstraints(min_length=1, max_length=500)]
UnitSource = Annotated[
    str,
    StringConstraints(pattern=r"^(?:trials|trajectory-steps|metric:[a-z0-9][a-z0-9_-]*)$"),
]

BUILTIN_RANDOM = "random"

StageName = Literal["integration-smoke", "pilot", "qualification", "stress"]
_STAGE_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "integration-smoke": (),
    "pilot": ("integration-smoke",),
    "qualification": ("integration-smoke", "pilot"),
    "stress": ("integration-smoke", "pilot", "qualification"),
}


class EvaluationStageSpec(BaseModel):
    """Provider-neutral execution contract for one gated evaluation stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: StageName
    evaluation_unit: NonEmptyPath
    unit_source: UnitSource = "trials"
    target_units: Annotated[int, Field(ge=1)]
    independent_case_unit: NonEmptyPath
    independent_case_key: NonEmptyPath
    independent_cases: Annotated[int, Field(ge=1)]
    independent_case_groups: dict[NonEmptyPath, NonEmptyPath] = Field(default_factory=dict)
    scenarios: tuple[NonEmptyPath, ...]
    event_injections: tuple[NonEmptyPath, ...] = ()
    operator_contexts: tuple[NonEmptyPath, ...]
    stop_conditions: tuple[NonEmptyPath, ...]
    prerequisites: tuple[StageName, ...] = ()

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.independent_cases > self.target_units:
            raise TaskContractError(
                f"stage {self.name} declares {self.independent_cases} independent cases "
                f"but only {self.target_units} evaluation units"
            )
        for label, values in (
            ("scenarios", self.scenarios),
            ("operator_contexts", self.operator_contexts),
            ("stop_conditions", self.stop_conditions),
        ):
            if not values:
                raise TaskContractError(f"stage {self.name} must declare {label}")
            if len(set(values)) != len(values):
                raise TaskContractError(f"stage {self.name} declares duplicate {label}")
        if len(set(self.event_injections)) != len(self.event_injections):
            raise TaskContractError(f"stage {self.name} declares duplicate event_injections")
        expected = _STAGE_PREREQUISITES[self.name]
        if self.prerequisites != expected:
            raise TaskContractError(
                f"stage {self.name} prerequisites must be {list(expected)}, "
                f"got {list(self.prerequisites)}"
            )
        if self.name == "stress" and not self.event_injections:
            raise TaskContractError("stage stress must declare injected events")
        return self


class JobConfig(BaseModel):
    """Loadable cartesian job. Paths are as written; resolve against the file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format_version: Annotated[str, StringConstraints(min_length=1, max_length=16)]
    id: Slug
    n: Annotated[int, Field(ge=1)] | None = None
    tasks: tuple[NonEmptyPath, ...]
    task_trials: dict[str, Annotated[int, Field(ge=1)]] = Field(default_factory=dict)
    agents: tuple[NonEmptyPath, ...]
    projection: ProjectionSpec | None = None
    stage: EvaluationStageSpec | None = None

    @model_validator(mode="after")
    def _non_empty_unique(self) -> Self:
        if not self.tasks:
            msg = f"job {self.id} must list at least one task"
            raise TaskContractError(msg)
        if not self.agents:
            msg = f"job {self.id} must list at least one agent"
            raise TaskContractError(msg)
        if len(set(self.tasks)) != len(self.tasks):
            msg = f"job {self.id} lists the same task path twice"
            raise TaskContractError(msg)
        if len(set(self.agents)) != len(self.agents):
            msg = f"job {self.id} lists the same agent twice"
            raise TaskContractError(msg)
        if self.n is not None and self.task_trials:
            raise TaskContractError(f"job {self.id} cannot declare both n and task_trials")
        if self.task_trials and set(self.task_trials) != set(self.tasks):
            raise TaskContractError(f"job {self.id} task_trials keys must exactly match tasks")
        return self


@dataclass(frozen=True)
class ResolvedJob:
    """``JobConfig`` with paths resolved against the ``job.toml`` directory."""

    config: JobConfig
    toml_path: Path
    root: Path
    task_paths: tuple[Path, ...]
    agent_refs: tuple[str, ...]


def _job_toml(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_dir():
        return resolved / "job.toml"
    return resolved


def load_job_config(path: Path | str) -> JobConfig:
    """Load ``job.toml`` without resolving paths."""
    toml_path = _job_toml(Path(path))
    data = _read_toml(toml_path)
    try:
        return JobConfig.model_validate(data)
    except TaskContractError:
        raise
    except ValidationError as exc:
        msg = f"job {toml_path} failed validation: {exc}"
        raise TaskContractError(msg) from exc
    except Exception as exc:
        msg = f"job {toml_path} failed validation: {exc}"
        raise TaskContractError(msg) from exc


def _resolve_agent_ref(root: Path, ref: str) -> str:
    if ref == BUILTIN_RANDOM:
        return BUILTIN_RANDOM
    candidate = (root / ref).resolve()
    if candidate.exists():
        return str(candidate)
    msg = f"job agent {ref!r} is not {BUILTIN_RANDOM!r} and does not exist at {candidate}"
    raise TaskContractError(msg)


def resolve_job(path: Path | str) -> ResolvedJob:
    """Load a job and resolve task/agent paths relative to the file."""
    toml_path = _job_toml(Path(path))
    config = load_job_config(toml_path)
    root = toml_path.parent.resolve()
    task_paths: list[Path] = []
    for raw in config.tasks:
        resolved = (root / raw).resolve()
        if not resolved.exists():
            msg = f"job {config.id} task path does not exist: {raw} ({resolved})"
            raise TaskContractError(msg)
        task_paths.append(resolved)
    agent_refs = tuple(_resolve_agent_ref(root, raw) for raw in config.agents)
    return ResolvedJob(
        config=config,
        toml_path=toml_path,
        root=root,
        task_paths=tuple(task_paths),
        agent_refs=agent_refs,
    )
