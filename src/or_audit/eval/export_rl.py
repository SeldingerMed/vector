"""Export vector jobs as task-declared, versioned projection records for RL.

The vector remains authoritative. Each row carries the declarative projection
rule and digest and is recomputed from the stored vector; stored or caller-made
floats are never trusted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from or_audit.errors import ScoreContractError, TaskContractError
from or_audit.eval.cartesian import iter_job_dirs
from or_audit.eval.enums import ProjectionId
from or_audit.eval.integrity import tree_digest
from or_audit.eval.job import (
    JobResult,
    read_job_config,
    read_job_result,
    resolve_bundle_path,
    verify_head,
)
from or_audit.eval.loader import load_task
from or_audit.eval.sim.base import BACKEND_REAL, BACKEND_SYNTHETIC_STUB
from or_audit.eval.task import ProjectionSpec, TaskSpec
from or_audit.eval.vector import project


class RlExportRecord(BaseModel):
    """One episode in an RL dump. Not a leaderboard row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    task_version: str
    agent_identity: str
    world_pin: str
    seed: Annotated[int, Field(ge=0)]
    episode_id: str
    projection_id: str
    projection_version: str
    projection_digest: str
    projection_rule: dict[str, Any]
    projection: float


def _spec_for_task(task: TaskSpec, projection_id: ProjectionId | str) -> ProjectionSpec:
    requested = projection_id.value if isinstance(projection_id, ProjectionId) else projection_id
    if task.projection is None:
        raise TaskContractError(
            f"task {task.id} has no declared projection or diverged gating rule"
        )
    declared = (
        task.projection.id.value
        if isinstance(task.projection.id, ProjectionId)
        else task.projection.id
    )
    if declared != requested:
        raise TaskContractError(f"task {task.id} declares {declared!r}, not {requested!r}")
    return task.projection


def _assert_physical_backend(job_dir: Path, result: JobResult) -> None:
    """Refuse rewards not attested as real physics.

    Provenance is read from the head-covered ``JobResult.world_engine``, never
    from the mutable ``config.json``. Export positively requires ``backend ==
    BACKEND_REAL``; synthetic stand-ins, unattested (``unknown``) runs, and
    missing provenance are all refused.
    """
    world_engine = result.world_engine
    if world_engine is None:
        raise TaskContractError(
            f"{job_dir.name} records no head-bound world-engine backend "
            "(JobResult.world_engine missing); refusing to export rewards from "
            "unattested physics"
        )
    if world_engine.backend == BACKEND_SYNTHETIC_STUB:
        engine = world_engine.engine or "unknown"
        raise TaskContractError(
            f"{job_dir.name} ran against a synthetic stand-in for the {engine} world "
            f'(world_engine.backend="{BACKEND_SYNTHETIC_STUB}"); refusing to '
            f"export rewards from fabricated physics. Re-run the job against a real "
            f"simulation backend, or drop environment.synthetic_stub from the task."
        )
    if world_engine.backend != BACKEND_REAL:
        engine = world_engine.engine or "unknown"
        raise TaskContractError(
            f"{job_dir.name} ran with unattested physics "
            f'(world_engine.backend="{world_engine.backend}"); refusing to export '
            f"rewards from an environment that reported no real simulation backend. "
            f"Re-run the job against a real simulation backend."
        )


def export_job_records(
    job_dir: Path, *, projection_id: ProjectionId | str
) -> tuple[RlExportRecord, ...]:
    """Recompute one job's trials into RL records."""
    config = read_job_config(job_dir)
    result = read_job_result(job_dir)
    # The result head must verify before we trust any of its provenance or
    # projections; world_engine is bound by that head, not by config.json.
    if not verify_head(result):
        raise TaskContractError(f"{job_dir.name} result head does not verify; refusing to export")
    _assert_physical_backend(job_dir, result)
    task_dir = resolve_bundle_path(job_dir, config["task_dir"], label="task")
    if tree_digest(task_dir) != config.get("task_digest"):
        raise TaskContractError(f"{job_dir.name} bundled task digest does not match config")
    task = load_task(task_dir)
    spec = _spec_for_task(task, projection_id)
    records: list[RlExportRecord] = []
    for trial in result.trials:
        recomputed = project(trial.vector, spec)
        if trial.projection is not None and trial.projection != recomputed:
            msg = (
                f"{job_dir.name} seed {trial.seed}: stored projection "
                f"{trial.projection} disagrees with {spec.identity}={recomputed}; "
                f"export-rl recomputes from the vector and will not ship a "
                f"homemade float"
            )
            raise ScoreContractError(msg)
        records.append(
            RlExportRecord(
                task_id=result.task_id,
                task_version=result.task_version,
                agent_identity=result.agent_identity,
                world_pin=result.world_pin,
                seed=trial.seed,
                episode_id=f"{result.task_id}-{trial.seed}",
                projection_id=spec.id,
                projection_version=spec.version,
                projection_digest=spec.rule_digest,
                projection_rule=spec.model_dump(mode="json"),
                projection=recomputed,
            )
        )
    return tuple(records)


def export_rl(
    path: Path,
    *,
    projection_id: ProjectionId | str,
    out: Path,
) -> int:
    """Write jsonl for a job directory or a cartesian parent. Returns episode count."""
    records: list[RlExportRecord] = []
    for job_dir in iter_job_dirs(path):
        records.extend(export_job_records(job_dir, projection_id=projection_id))
    if not records:
        msg = f"{path} exported no episodes"
        raise TaskContractError(msg)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.model_dump(mode="json"), sort_keys=True) for record in records]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(records)
