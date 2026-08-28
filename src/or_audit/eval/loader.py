"""Load v0.3 packages and normalize v0.2 task, dataset, and agent layouts."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from or_audit.errors import TaskContractError
from or_audit.eval.agent import AgentPackage
from or_audit.eval.contracts import RuntimeKind
from or_audit.eval.dataset import DatasetSpec, TasksetSpec
from or_audit.eval.integrity import file_sha256, package_file
from or_audit.eval.task import TaskSpec


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TaskContractError(f"missing {path.name}: {path}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _task_root(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_dir():
        return resolved
    if resolved.name == "task.toml":
        return resolved.parent
    raise TaskContractError(f"a task is a directory (or its task.toml), got {path}")


def _load_versioned_records(root: Path, value: Any, *, label: str) -> Any:
    """Resolve optional scenario/perturbation JSON records inside a task package."""
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        return value
    records = []
    for raw_path in value:
        path = package_file(root, raw_path, label=label)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TaskContractError(f"{label} {path} is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise TaskContractError(f"{label} {path} must contain one object")
        records.append(payload)
    return records


def _verify_calibration(root: Path, calibration: Any, *, gate: str) -> None:
    """Verify a calibration artifact is package-contained and digest-verified."""
    artifact = calibration.artifact
    path = package_file(root, artifact, label=f"gate {gate} calibration artifact")
    if not path.is_file():
        raise TaskContractError(
            f"gate {gate} calibration artifact {artifact} is missing from the task package"
        )
    actual = file_sha256(path)
    if not _safe_eq(actual, calibration.digest):
        raise TaskContractError(
            f"gate {gate} calibration artifact {artifact} digest mismatch "
            f"(declared {calibration.digest}, actual {actual})"
        )


def _safe_eq(a: str, b: str) -> bool:
    # Constant-time-ish compare over equal-length hex digests.
    if len(a) != len(b):
        return False
    return a == b


def _verify_calibrations(root: Path, task: TaskSpec) -> None:
    """Verify every declared calibration artifact at load time."""
    for gate in task.verifier.gates:
        if gate.calibration is not None:
            _verify_calibration(root, gate.calibration, gate=gate.id)
        basis = gate.threshold_basis
        if basis is not None and basis.calibration is not None:
            _verify_calibration(root, basis.calibration, gate=gate.id)


def _verify_streams(task: TaskSpec) -> None:
    """Verify every stream's adapter plugin is known and digest-pinned."""
    from or_audit.eval.adapters import adapter_revision

    for stream in task.interface.streams:
        actual = adapter_revision(stream.adapter)
        if actual == "":
            raise TaskContractError(
                f"task {task.id} stream {stream.id!r} uses unknown or unpinned "
                f"adapter {stream.adapter!r} (no content pin available)"
            )
        if not _safe_eq(actual, stream.adapter_digest):
            raise TaskContractError(
                f"task {task.id} stream {stream.id!r} adapter {stream.adapter!r} "
                "content digest mismatch "
                f"(task pins {stream.adapter_digest}, registry has {actual})"
            )


def _verify_world_adapter(task: TaskSpec) -> None:
    """Verify a declared world-adapter pin against the registered adapter.

    A task that pins its world adapter is refused when the adapter is absent,
    was published by a different module, or has different content than the pin
    — the same discipline ``_verify_streams`` applies to modality adapters.
    """
    from or_audit.eval.sim import world_kind_spec  # registers built-in adapters

    world = task.environment
    if not world.adapter:
        return
    spec = world_kind_spec(world.kind)
    if spec is None or not spec.adapter_id:
        raise TaskContractError(
            f"task {task.id} pins world adapter {world.adapter!r} for world kind "
            f"{world.kind_key!r}, but no adapter is registered for that kind; "
            "install the world adapter distribution"
        )
    if spec.adapter_id != world.adapter:
        raise TaskContractError(
            f"task {task.id} pins world adapter {world.adapter!r} but world kind "
            f"{world.kind_key!r} is served by {spec.adapter_id!r}"
        )
    if not _safe_eq(spec.adapter_digest, world.adapter_digest):
        raise TaskContractError(
            f"task {task.id} world adapter {world.adapter!r} content digest mismatch "
            f"(task pins {world.adapter_digest}, installed adapter is "
            f"{spec.adapter_digest})"
        )


def load_task(path: Path | str) -> TaskSpec:
    root = _task_root(Path(path))
    data = _read_toml(root / "task.toml")
    instruction_path = root / "instruction.md"
    if not instruction_path.is_file():
        raise TaskContractError(f"task {root} is missing instruction.md")
    data["instruction"] = instruction_path.read_text(encoding="utf-8").strip()
    data["scenarios"] = _load_versioned_records(root, data.get("scenarios", []), label="scenario")
    data["perturbations"] = _load_versioned_records(
        root, data.get("perturbations", []), label="perturbation"
    )
    verifier_path = root / "verifier.toml"
    if verifier_path.is_file():
        extra = _read_toml(verifier_path)
        if "projection" in extra:
            data["projection"] = extra["projection"]
    try:
        task = TaskSpec.model_validate(data)
    except TaskContractError:
        raise
    except Exception as exc:
        raise TaskContractError(f"task {root} failed validation: {exc}") from exc
    _verify_calibrations(root, task)
    _verify_streams(task)
    _verify_world_adapter(task)
    return task


def _taskset_paths(path: Path | str) -> tuple[Path, Path, dict[str, Any]]:
    requested = Path(path).resolve()
    root = requested if requested.is_dir() else requested.parent
    if requested.is_dir():
        taskset_path = root / "taskset.toml"
        toml_path = taskset_path if taskset_path.is_file() else root / "dataset.toml"
    else:
        toml_path = requested
    return root, toml_path, _read_toml(toml_path)


def load_taskset(path: Path | str) -> TasksetSpec:
    """Load a canonical taskset or a v0.2 dataset directory."""
    root, toml_path, data = _taskset_paths(path)
    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise TaskContractError(f"taskset {toml_path} must list at least one task path")
    tasks = tuple(load_task((root / str(entry)).resolve()) for entry in raw_tasks)
    payload = {key: value for key, value in data.items() if key != "tasks"}
    try:
        spec = TasksetSpec.model_validate({**payload, "tasks": tasks})
    except TaskContractError:
        raise
    except Exception as exc:
        raise TaskContractError(f"taskset {toml_path} failed validation: {exc}") from exc
    spec.check_tasks()
    return spec


def load_dataset(path: Path | str) -> DatasetSpec:
    """Compatibility alias for :func:`load_taskset`."""
    return load_taskset(path)


def taskset_task_paths(path: Path | str) -> tuple[Path, ...]:
    root, toml_path, data = _taskset_paths(path)
    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise TaskContractError(f"taskset {toml_path} must list at least one task path")
    return tuple((root / str(entry)).resolve() for entry in raw_tasks)


def dataset_task_paths(path: Path | str) -> tuple[Path, ...]:
    """Compatibility alias for :func:`taskset_task_paths`."""
    return taskset_task_paths(path)


def _agent_root(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_dir():
        return resolved
    if resolved.name == "agent.toml":
        return resolved.parent
    raise TaskContractError(f"an agent is a directory (or its agent.toml), got {path}")


def load_agent(path: Path | str) -> AgentPackage:
    root = _agent_root(Path(path))
    data = _read_toml(root / "agent.toml")
    try:
        agent = AgentPackage.model_validate(data)
        if agent.runtime is not None and agent.runtime.kind is RuntimeKind.TRUSTED_IN_PROCESS:
            raise TaskContractError(
                "trusted-in-process is reserved for injected test runtimes; "
                "package agents must use an isolated runtime"
            )
        if agent.weights_path:
            weights = package_file(root, agent.weights_path, label="agent weights")
            actual = file_sha256(weights)
            if actual != agent.weights_pin:
                raise TaskContractError(
                    f"agent {agent.id} weights digest mismatch: "
                    f"declared {agent.weights_pin}, actual {actual}"
                )
        return agent
    except TaskContractError:
        raise
    except Exception as exc:
        raise TaskContractError(f"agent {root} failed validation: {exc}") from exc
