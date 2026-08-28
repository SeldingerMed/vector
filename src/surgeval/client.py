"""SurgEval Developer SDK Client.

High-level Python API for evaluating surgical, robotic, and procedural AI models
across diverse modalities (laparoscopy, bronchoscopy, fluoroscopy, orthopedics).
"""

from __future__ import annotations

import contextlib
import hashlib
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

import cloudpickle

from or_audit.errors import TaskContractError
from or_audit.eval.agent import AgentPackage
from or_audit.eval.contracts import CapabilitySpec, InteractionMode
from or_audit.eval.job import JobResult
from or_audit.eval.loader import load_agent, load_task, load_taskset
from or_audit.eval.runner import run_job
from or_audit.eval.task import TaskSpec
from surgeval.decorators import (
    agent_kind_for,
    binding_for,
    capability_toml,
    entrypoint_symbol_for,
    is_agent,
)


class _Synthesis(NamedTuple):
    """What the synthesized package must declare for one in-memory model."""

    capability: CapabilitySpec
    agent_id: str
    version: str
    kind: str
    entrypoint: str


def _resolve_synthesis(
    agent_obj: Any,
    interface_id: str,
    required_mode: InteractionMode,
) -> _Synthesis:
    """Resolve what a synthesized package declares for an in-memory model.

    A decorated class carries an inferred capability; anything else (a raw
    wrapper, a bare object with ``predict``) has declared nothing the kernel can
    check, so it binds as an explicit wildcard.
    """
    entrypoint = entrypoint_symbol_for(required_mode)
    if is_agent(agent_obj):
        binding = binding_for(agent_obj)
        if not binding.supports(required_mode):
            needed = (
                "act(observation, step=...)"
                if required_mode is InteractionMode.CLOSED_LOOP
                else "predict(item)"
            )
            raise TaskContractError(
                f"{binding.cls_name} implements {', '.join(binding.methods)}, which cannot "
                f"drive a {required_mode.value} task. Implement {needed}, or narrow the "
                "task selection to a mode this class supports."
            )
        return _Synthesis(
            capability=binding.capability_for_interface(interface_id),
            agent_id=binding.agent_id,
            version=binding.version,
            kind=binding.kind_for(required_mode),
            entrypoint=entrypoint,
        )
    return _Synthesis(
        capability=CapabilitySpec(
            interface=interface_id,
            interaction_modes=(required_mode,),
            schema_wildcard=True,
        ),
        agent_id="",
        version="0",
        kind=agent_kind_for(required_mode),
        entrypoint=entrypoint,
    )


def _synthesize_agent_bundle(
    agent_obj: Any,
    interface_id: str,
    interaction_mode: InteractionMode,
    target_dir: Path,
    *,
    strict_schemas: bool = False,
) -> tuple[AgentPackage, Path]:
    """Synthesize a complete local AgentPackage directory for an in-memory model."""
    agent_dir = target_dir / "synthesized_agent"
    agent_dir.mkdir(parents=True, exist_ok=True)

    synthesis = _resolve_synthesis(agent_obj, interface_id, interaction_mode)
    if strict_schemas and synthesis.capability.schema_wildcard:
        raise TaskContractError(
            f"strict_schemas=True refuses a wildcard binding for "
            f"{type(agent_obj).__name__} on interface {interface_id!r}: the model "
            "declares no schemas, so nothing verifies that it speaks this task's "
            "data shapes. Decorate the class with @surgeval.agent(...) and declare "
            "observations / actions / outputs / features as class attributes, or "
            "drop strict_schemas to accept an unverified wildcard binding."
        )

    # Force cloudpickle to serialize the class definition inline rather than
    # by reference, so the bundle is self-contained and portable.
    model_module_name = type(agent_obj).__module__
    if model_module_name not in ("builtins", "__main__"):
        model_module = sys.modules.get(model_module_name)
        if model_module is not None:
            with contextlib.suppress(Exception):
                cloudpickle.register_pickle_by_value(model_module)

    model_bytes = cloudpickle.dumps(agent_obj)
    weights_pin = hashlib.sha256(model_bytes).hexdigest()

    (agent_dir / "model.pkl").write_bytes(model_bytes)

    agent_id = synthesis.agent_id or f"custom/sdk-agent-{weights_pin[:12]}"
    runner_code = """import os
import sys
from pathlib import Path

import cloudpickle

for p in [os.getcwd(), str(Path(__file__).resolve().parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

with open(Path(__file__).parent / "model.pkl", "rb") as f:
    _instance = cloudpickle.load(f)

class _Runtime:
    def predict(self, item):
        fn = getattr(_instance, "predict", _instance)
        return fn(item)

    def act(self, obs, step=0):
        fn = getattr(_instance, "act", getattr(_instance, "predict", _instance))
        try:
            return fn(obs, step=step)
        except TypeError:
            return fn(obs)

    def reset(self, *, seed=None):
        fn = getattr(_instance, "reset", None)
        if callable(fn):
            try:
                fn(seed=seed)
            except TypeError:
                fn()
def load_predictor(*, root=None, weights_path=None, weights=None):
    return _Runtime()

def load_policy(*, root=None, weights_path=None, weights=None):
    return _Runtime()
"""
    (agent_dir / "runner.py").write_text(runner_code, encoding="utf-8")

    agent_toml = "\n".join(
        [
            'format_version = "2"',
            f'id = "{agent_id}"',
            f'agent_version = "{synthesis.version}"',
            f'kind = "{synthesis.kind}"',
            f'weights_pin = "{weights_pin}"',
            'weights_path = "model.pkl"',
            "",
            capability_toml(synthesis.capability),
            "[runtime]",
            'kind = "local"',
            'protocol_version = "1"',
            f'entrypoint = "runner.py:{synthesis.entrypoint}"',
            "timeout_sec = 120.0",
            "",
        ]
    )
    (agent_dir / "agent.toml").write_text(agent_toml, encoding="utf-8")

    agent_pkg = load_agent(agent_dir)
    return agent_pkg, agent_dir


def _assert_declared_schemas(agent_pkg: AgentPackage, interface_id: str) -> None:
    """Refuse a wildcard capability when the caller asked for a verified binding."""
    capability = agent_pkg.capability_for(interface_id)
    if capability is not None and capability.schema_wildcard:
        raise TaskContractError(
            f"strict_schemas=True refuses a wildcard binding: agent {agent_pkg.id} "
            f"declares schema_wildcard on interface {interface_id!r}, so nothing "
            "verifies that it speaks this task's data shapes. Declare "
            "observations / actions / outputs / features on the capability."
        )


def evaluate(
    agent: AgentPackage | Path | str | Any,
    task_or_taskset: TaskSpec | Path | str,
    *,
    task_dir: Path | str | None = None,
    out: Path | str | None = None,
    n: int | None = None,
    interface_id: str | None = None,
    strict_schemas: bool = False,
) -> JobResult:
    """Evaluate an agent or model policy on a procedural task.

    Args:
        agent: AgentPackage, path to agent package directory, or Python model instance.
        task_or_taskset: TaskSpec object or path to task directory.
        task_dir: Optional explicit task directory (used when task_or_taskset is a TaskSpec).
        out: Directory to store replayable evaluation artifacts.
        n: Number of evaluation episodes (defaults to task specification).
        interface_id: Interface ID override if agent is a raw policy.
        strict_schemas: Refuse instead of binding a wildcard capability. The
            zero-config on-ramp synthesizes a wildcard binding for a model that
            declares no schemas; set this when you want the kernel to prove the
            agent speaks the task's data shapes before a single trial runs.

    Returns:
        JobResult containing verifiable trial vectors, hard gate outcomes,
        typed metrics, and cryptographic artifact heads.
    """
    # Resolve task
    task: TaskSpec
    resolved_task_dir: Path
    if isinstance(task_or_taskset, TaskSpec):
        task = task_or_taskset
        resolved_task_dir = Path(task_dir).resolve() if task_dir else Path.cwd()
    else:
        task_path = Path(task_or_taskset).resolve()
        task = load_task(task_path)
        resolved_task_dir = (
            Path(task_dir).resolve()
            if task_dir
            else (task_path if task_path.is_dir() else task_path.parent)
        )
    # Context manager for temporary agent directory if synthesized
    with tempfile.TemporaryDirectory(prefix="surgeval-agent-") as agent_tmp:
        agent_pkg: AgentPackage
        agent_dir: Path | None

        if isinstance(agent, (str, Path)):
            agent_path = Path(agent).resolve()
            agent_pkg = load_agent(agent_path)
            agent_dir = agent_path if agent_path.is_dir() else agent_path.parent
            if strict_schemas:
                _assert_declared_schemas(agent_pkg, interface_id or task.interface.id)
        elif isinstance(agent, AgentPackage):
            agent_pkg = agent
            agent_dir = None
            if strict_schemas:
                _assert_declared_schemas(agent_pkg, interface_id or task.interface.id)
        else:
            # In-memory Python instance or wrapper
            iface = interface_id or task.interface.id
            agent_pkg, agent_dir = _synthesize_agent_bundle(
                agent_obj=agent,
                interface_id=iface,
                interaction_mode=task.interface.interaction_mode,
                target_dir=Path(agent_tmp),
                strict_schemas=strict_schemas,
            )

        if out is not None:
            out_path = Path(out).resolve()
            out_path.mkdir(parents=True, exist_ok=True)
            return run_job(
                task=task,
                task_dir=resolved_task_dir,
                agent=agent_pkg,
                agent_dir=agent_dir,
                out=out_path,
                n=n,
            )

        with tempfile.TemporaryDirectory(prefix="surgeval-run-") as run_tmp:
            out_path = Path(run_tmp)
            return run_job(
                task=task,
                task_dir=resolved_task_dir,
                agent=agent_pkg,
                agent_dir=agent_dir,
                out=out_path,
                n=n,
            )


__all__ = [
    "evaluate",
    "load_agent",
    "load_task",
    "load_taskset",
]
