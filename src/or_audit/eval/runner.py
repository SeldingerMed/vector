"""Execute bound task and agent packages through v0.3 harness interaction modes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from or_audit.audit.canonical import digest
from or_audit.errors import TaskContractError
from or_audit.eval.agent import AgentPackage
from or_audit.eval.bind import assert_bind
from or_audit.eval.contracts import CapabilitySpec, InteractionMode
from or_audit.eval.enums import AgentKind, PortId, WorldKind
from or_audit.eval.gym_world import (
    GymFactory,
    assert_perturbations_applied,
    make_gym,
    run_gym_episode,
    sample_action,
    split_perturbations,
)
from or_audit.eval.integrity import tree_digest
from or_audit.eval.job import (
    JobResult,
    TrialRecord,
    agent_identity,
    assemble_job_result,
    compute_head,
    read_job_config,
    read_job_result,
    resolve_bundle_path,
    write_job,
)
from or_audit.eval.plugins import (
    load_policy_runtime,
    load_predictor_runtime,
    load_verifier_runtime,
)
from or_audit.eval.predict import index_items, load_claim_footer, load_items
from or_audit.eval.provenance import assert_scoreable_package
from or_audit.eval.reconstitute import assert_trajectory_matches_vector
from or_audit.eval.sim import (
    BACKEND_UNKNOWN,
    get_simulation_engine,
    world_kind_key,
    world_kind_spec,
)
from or_audit.eval.task import TaskSpec
from or_audit.eval.trace import ProceduralTrace
from or_audit.eval.vector import project
from or_audit.eval.verifier import score_context

SAFETY_MAX_PEN = 0.3


def assert_trial_capacity(task: TaskSpec, task_dir: Path, n: int) -> None:
    """Refuse dataset-backed runs that request more trials than exist."""
    if task.harness.interaction_mode is InteractionMode.CLOSED_LOOP:
        return
    available = len(load_items(task_dir / task.environment.inputs_path))
    if n > available:
        raise TaskContractError(
            f"task {task.id} has {available} input items; cannot execute {n:,} distinct trials"
        )


def stream_adapters(task: TaskSpec) -> dict[str, Any]:
    """Resolve every stream's adapter, keyed by stream id (raises on unknown)."""
    from or_audit.eval.adapters import require_adapter

    return {s.id: require_adapter(s.adapter) for s in task.interface.streams}


def _source_parts(locator: str) -> list[str]:
    """Split a source locator into path parts (``$`` -> whole observation)."""
    if locator == "$":
        return []
    if locator.startswith("/"):
        return [
            part.replace("~1", "/").replace("~0", "~") for part in locator[1:].split("/") if part
        ]
    return locator.split(".")


def _get_source(item: dict[str, Any], locator: str) -> Any:
    """Address the observation slice a stream consumes, or raise if missing."""
    if locator == "$":
        return item
    cur: Any = item
    for part in _source_parts(locator):
        if not isinstance(cur, dict) or part not in cur:
            raise TaskContractError(f"stream source {locator!r} not present in observation")
        cur = cur[part]
    return cur


def preprocess_observation(
    task: TaskSpec, adapters: dict[str, Any], item: dict[str, Any]
) -> dict[str, Any]:
    """Compose each stream's processed source slice into a fresh payload.

    Every stream reads its slice from the *original* observation and its
    output lands under its own stream id — never written back into the shared
    observation, so ``source=\"$\"`` streams cannot consume each other's
    output, and the agent always sees which channel produced a value.
    Closed-loop observations are commonly ndarrays, so JSONable scalars, lists
    and arrays are accepted, not just dicts.
    """
    composed: dict[str, Any] = {}
    if not task.interface.streams:
        return item
    for stream in task.interface.streams:
        adapter = adapters.get(stream.id)
        if adapter is None:
            continue
        source = _get_source(item, stream.source)
        processed = adapter.preprocess_observation(source)
        if isinstance(processed, dict):
            normalized = processed
        elif hasattr(processed, "tolist"):  # ndarray / array-like
            normalized = processed.tolist()
        elif is_dataclass(processed) and not isinstance(processed, type):
            normalized = asdict(processed)
        else:
            normalized = processed
        composed[stream.id] = normalized
    return composed


def _close(runtime: object | None) -> None:
    close = getattr(runtime, "close", None)
    if callable(close):
        close()


def _engine_provenance(task: TaskSpec, env: object | None) -> dict[str, Any]:
    """Attest which engine produced the observations; an absent reporter is not omission.

    Adapter identity is taken from the kernel's world-kind registry, never from
    the bridge's own report, so a bridge cannot understate or misname the
    adapter that ran. The values land in the head-covered
    ``JobResult.world_engine``.

    Every field here describes what was *observed*, so an absent reporter
    leaves them empty. ``world_pin`` in particular used to default to
    ``task.environment.world_pin``: the field conformance reads as evidence
    was being populated from the declaration it is evidence about, which is
    self-certification with extra steps. The declared pin stays available on
    ``JobResult.world_pin``; unobserved is spelled ``""`` and read as "cannot
    verify", never as "matches".
    """
    reported: dict[str, Any] = {
        "engine": world_kind_key(task.environment.kind),
        "backend": BACKEND_UNKNOWN,
        "backend_version": "",
        "world_pin": "",
    }
    reporter = getattr(env, "engine_provenance", None)
    if callable(reporter):
        from_bridge = reporter()
        if isinstance(from_bridge, dict):
            reported = {str(key): value for key, value in from_bridge.items()}
    spec = world_kind_spec(task.environment.kind)
    reported["adapter_id"] = spec.adapter_id if spec else ""
    reported["adapter_digest"] = spec.adapter_digest if spec else ""
    reported["metrics_only"] = task.environment.metrics_only
    return reported


def builtin_random_agent(interface_id: str = "gym-policy") -> AgentPackage:
    capability = CapabilitySpec(
        interface=interface_id,
        interaction_modes=(InteractionMode.CLOSED_LOOP,),
        schema_wildcard=True,
    )
    return AgentPackage(
        format_version="1",
        id="seldingermed/random",
        agent_version="0",
        port=PortId.GYM_POLICY,
        kind=AgentKind.RANDOM.value,
        capabilities=(capability,),
    )


def run_job(
    *,
    task: TaskSpec,
    task_dir: Path,
    agent: AgentPackage,
    agent_dir: Path | None,
    out: Path,
    n: int | None = None,
    gym_factory: GymFactory | None = None,
) -> JobResult:
    assert_bind(task, agent)
    task.assert_runnable()
    # A package presenting concierge provenance must still be the package that was
    # frozen: an adaptation whose verifier, gates, or projection moved after
    # freezing would otherwise be hashed here as a brand-new task and scored.
    assert_scoreable_package(task_dir)
    task_package_digest = tree_digest(task_dir)
    agent_package_digest = (
        tree_digest(agent_dir) if agent_dir is not None else digest(agent.model_dump(mode="json"))
    )
    episodes = n if n is not None else task.environment.n_eval_episodes
    if episodes < 1:
        raise TaskContractError(f"n must be >= 1, got {episodes}")
    assert_trial_capacity(task, task_dir, episodes)
    extra: dict[str, Any] = {
        "interaction_mode": task.harness.interaction_mode.value,
        "world_engine": _engine_provenance(task, None),
    }
    if task.interface.streams:
        extra["streams"] = [
            {"id": s.id, "adapter": s.adapter, "schema": s.schema_id}
            for s in task.interface.streams
        ]
    binding_cap = next(
        (c for c in agent.capabilities if c.interface == task.interface.id and c.schema_wildcard),
        None,
    )
    if binding_cap is not None:
        extra["binding_mode"] = "wildcard"
    if task.harness.interaction_mode is InteractionMode.CLOSED_LOOP:
        result, safety, provenance = _run_closed_loop(
            task=task,
            task_dir=task_dir,
            agent=agent,
            agent_dir=agent_dir,
            task_digest=task_package_digest,
            agent_digest=agent_package_digest,
            n=episodes,
            gym_factory=gym_factory,
        )
        extra["safety_max_pen"] = safety
        extra["world_engine"] = provenance
    elif task.harness.interaction_mode is InteractionMode.SINGLE_TURN:
        result = _run_single_turn(
            task=task,
            task_dir=task_dir,
            agent=agent,
            agent_dir=agent_dir,
            task_digest=task_package_digest,
            agent_digest=agent_package_digest,
            n=episodes,
        )
    elif task.harness.interaction_mode is InteractionMode.INTERACTIVE:
        result = _run_interactive(
            task=task,
            task_dir=task_dir,
            agent=agent,
            agent_dir=agent_dir,
            task_digest=task_package_digest,
            agent_digest=agent_package_digest,
            n=episodes,
        )
    elif task.harness.interaction_mode is InteractionMode.COUNTERFACTUAL:
        result = _run_counterfactual(
            task=task,
            task_dir=task_dir,
            agent=agent,
            agent_dir=agent_dir,
            task_digest=task_package_digest,
            agent_digest=agent_package_digest,
            n=episodes,
        )
    else:  # pragma: no cover - enum exhaustiveness
        raise TaskContractError(f"unsupported harness mode {task.harness.interaction_mode}")
    config = {
        "format_version": "2",
        "task_id": task.id,
        "task_dir": "bundle/task",
        "agent_id": agent.id,
        "agent_dir": "bundle/agent" if agent_dir is not None else None,
        "task_digest": task_package_digest,
        "agent_digest": agent_package_digest,
        "runtime_identity": agent.runtime_identity,
        "n": result.n,
        "world_pin": task.environment.world_pin,
        "interface": task.interface.id,
        **extra,
    }
    write_job(out, config=config, result=result, task_dir=task_dir, agent_dir=agent_dir)
    return result


def _run_closed_loop(
    *,
    task: TaskSpec,
    task_dir: Path,
    agent: AgentPackage,
    agent_dir: Path | None,
    n: int,
    task_digest: str,
    agent_digest: str,
    gym_factory: GymFactory | None,
) -> tuple[JobResult, float, dict[str, Any]]:
    if agent.kind not in {AgentKind.RANDOM.value, AgentKind.POLICY.value}:
        raise TaskContractError(f"closed-loop runner does not implement kind={agent.kind}")
    policy = None
    if agent.kind == AgentKind.POLICY.value:
        if agent_dir is None:
            raise TaskContractError(f"policy agent {agent.id} has no package directory")
        policy = load_policy_runtime(agent_dir, agent.entrypoint, agent.weights_path, agent.runtime)
    verifier = load_verifier_runtime(task_dir, task.verifier.entrypoint)
    if gym_factory is not None:
        env = gym_factory(task)
    else:
        sim_engine = get_simulation_engine(task)
        env = sim_engine if sim_engine is not None else make_gym(task)
    provenance = _engine_provenance(task, env)
    identity = agent_identity(agent)
    adapters = stream_adapters(task)
    unwrapped = getattr(env, "unwrapped", env)
    nested = getattr(unwrapped, "_env", unwrapped)
    safety = float(getattr(nested, "safety_max_pen", SAFETY_MAX_PEN))
    trials = []
    try:
        for seed in range(n):
            if policy is not None:
                policy.reset(seed=seed)
            scenario = next(
                (candidate for candidate in task.scenarios if candidate.seed == seed),
                None,
            )
            perturbations = tuple(
                perturbation
                for perturbation in task.perturbations
                if perturbation.scenario_id is None
                or (scenario is not None and perturbation.scenario_id == scenario.id)
            )
            harness_perturbations, world_perturbations = split_perturbations(perturbations)
            reset_options = (
                {
                    "or_audit": {
                        "scenario": (
                            scenario.model_dump(mode="json") if scenario is not None else None
                        ),
                        "perturbations": [
                            perturbation.model_dump(mode="json")
                            for perturbation in world_perturbations
                        ],
                    }
                }
                if scenario is not None or world_perturbations
                else None
            )

            def action_fn(
                world: Any,
                observation: Any,
                step: int,
                *,
                episode_seed: int = seed,
            ) -> Any:
                if policy is None:
                    return sample_action(world, seed=episode_seed, step=step)
                return policy.act(preprocess_observation(task, adapters, observation), step=step)

            info, steps = run_gym_episode(
                env,
                seed=seed,
                action_fn=action_fn,
                max_steps=task.harness.max_steps,
                reset_options=reset_options,
                harness_perturbations=harness_perturbations,
            )
            assert_perturbations_applied(steps, perturbations)
            trace_steps = []
            for index, raw_step in enumerate(steps):
                trace_step = dict(raw_step)
                if index == 0 and scenario is not None:
                    trace_step["scenario"] = scenario
                active_perturbations = tuple(
                    perturbation
                    for perturbation in perturbations
                    if (
                        perturbation.at_step == index
                        or (perturbation.at_step is None and index == 0)
                    )
                )
                if active_perturbations:
                    trace_step["perturbations"] = active_perturbations
                trace_steps.append(trace_step)
            trace = ProceduralTrace.from_steps(
                trace_steps,
                mode=InteractionMode.CLOSED_LOOP,
            )
            vector = score_context(
                task=task,
                task_dir=task_dir,
                agent_identity=identity,
                seed=seed,
                context={
                    "kind": "gym-policy",
                    "info": info,
                    "trajectory": list(trace),
                    "safety_max_pen": safety,
                },
                runtime=verifier,
            )
            projection = project(vector, task.projection) if task.projection else None
            trials.append(
                TrialRecord(
                    seed=seed,
                    vector=vector,
                    trajectory=trace,
                    projection=projection,
                    projection_spec_digest=(task.projection.rule_digest if task.projection else ""),
                )
            )
    finally:
        _close(policy)
        _close(verifier)
        _close(env)
    return (
        assemble_job_result(
            task=task,
            agent=agent,
            trials=tuple(trials),
            task_digest=task_digest,
            world_engine=provenance,
            agent_digest=agent_digest,
        ),
        safety,
        provenance,
    )


def _run_predictions(
    *,
    task: TaskSpec,
    task_dir: Path,
    agent: AgentPackage,
    agent_dir: Path | None,
    task_digest: str,
    agent_digest: str,
    n: int,
    mode: InteractionMode,
) -> JobResult:
    if agent_dir is None:
        raise TaskContractError(f"agent {agent.id} has no package directory")
    inputs = load_items(task_dir / task.environment.inputs_path)
    labels = index_items(load_items(task_dir / task.environment.labels_path))
    predictor = load_predictor_runtime(
        agent_dir, agent.entrypoint, agent.weights_path, agent.runtime
    )
    verifier = load_verifier_runtime(task_dir, task.verifier.entrypoint)
    identity = agent_identity(agent)
    adapters = stream_adapters(task)
    trials = []
    try:
        for seed, item in enumerate(inputs[:n]):
            item_id = str(item["id"])
            if item_id not in labels:
                raise TaskContractError(f"task {task.id} has no label for item {item_id!r}")
            agent_input = preprocess_observation(task, adapters, item)
            prediction = predictor.predict(agent_input)
            context_kind = (
                "counterfactual" if mode is InteractionMode.COUNTERFACTUAL else "video-predict"
            )
            context = {
                "kind": context_kind,
                "input": item,
                "label": labels[item_id],
                "prediction": prediction,
            }
            vector = score_context(
                task=task,
                task_dir=task_dir,
                agent_identity=identity,
                seed=seed,
                context=context,
                runtime=verifier,
            )
            trace_payload: dict[str, Any] = {
                **context,
                "obs": agent_input,
                "output": prediction,
                "transition": {"oracle_evidence": labels[item_id]},
            }
            scenario = next(
                (
                    candidate
                    for candidate in task.scenarios
                    if candidate.inputs.get("item") == item_id
                ),
                None,
            )
            perturbations = tuple(
                perturbation
                for perturbation in task.perturbations
                if perturbation.scenario_id is None
                or (scenario is not None and perturbation.scenario_id == scenario.id)
            )
            if scenario is not None:
                trace_payload["scenario"] = scenario
            if perturbations:
                trace_payload["perturbations"] = perturbations
            if isinstance(prediction.get("uncertainty"), int | float):
                trace_payload["uncertainty"] = prediction["uncertainty"]
            if isinstance(prediction.get("abstain"), bool):
                trace_payload["abstained"] = prediction["abstain"]
            for event_name in (
                "evidence",
                "failure",
                "recovery",
                "handoff",
                "tool",
                "timing",
            ):
                if event_name in prediction:
                    trace_payload[event_name] = prediction[event_name]
                elif event_name in item:
                    trace_payload[event_name] = item[event_name]
            trace = ProceduralTrace.from_steps([trace_payload], mode=mode)
            projection = project(vector, task.projection) if task.projection else None
            trials.append(
                TrialRecord(
                    seed=seed,
                    vector=vector,
                    trajectory=trace,
                    projection=projection,
                    projection_spec_digest=(task.projection.rule_digest if task.projection else ""),
                )
            )
    finally:
        _close(predictor)
        _close(verifier)
    footer = ""
    if task.environment.kind is WorldKind.ANGIOSTRESS_CONTRACT:
        footer = load_claim_footer(task_dir / task.environment.contract_path)
    return assemble_job_result(
        task=task,
        agent=agent,
        trials=tuple(trials),
        task_digest=task_digest,
        agent_digest=agent_digest,
        claim_footer=footer,
        world_engine=_engine_provenance(task, None),
    )


def _run_single_turn(**kwargs: Any) -> JobResult:
    return _run_predictions(**kwargs, mode=InteractionMode.SINGLE_TURN)


def _run_interactive(
    *,
    task: TaskSpec,
    task_dir: Path,
    agent: AgentPackage,
    agent_dir: Path | None,
    task_digest: str,
    agent_digest: str,
    n: int,
) -> JobResult:
    if agent_dir is None:
        raise TaskContractError(f"agent {agent.id} has no package directory")
    inputs = load_items(task_dir / task.environment.inputs_path)
    labels = index_items(load_items(task_dir / task.environment.labels_path))
    predictor = load_predictor_runtime(
        agent_dir,
        agent.entrypoint,
        agent.weights_path,
        agent.runtime,
    )
    verifier = load_verifier_runtime(task_dir, task.verifier.entrypoint)
    identity = agent_identity(agent)
    trials = []
    try:
        for seed, item in enumerate(inputs[:n]):
            item_id = str(item["id"])
            if item_id not in labels:
                raise TaskContractError(f"task {task.id} has no label for item {item_id!r}")
            turns = item.get("turns")
            if not isinstance(turns, list) or not turns:
                raise TaskContractError(
                    f"interactive task {task.id} item {item_id!r} needs non-empty turns"
                )
            if len(turns) > task.harness.max_steps:
                raise TaskContractError(
                    f"interactive task {task.id} item {item_id!r} has {len(turns)} turns, "
                    f"above max_steps={task.harness.max_steps}"
                )
            history: list[dict[str, Any]] = []
            trace_payloads: list[dict[str, Any]] = []
            for turn_index, observation in enumerate(turns):
                request = {
                    "id": item_id,
                    "turn": observation,
                    "turn_index": turn_index,
                    "history": history,
                }
                prediction = predictor.predict(request)
                history.append({"observation": observation, "output": prediction})
                trace_payload: dict[str, Any] = {
                    "kind": "interactive",
                    "obs": observation,
                    "output": prediction,
                    "transition": {"history_length": len(history)},
                }
                if isinstance(prediction.get("uncertainty"), int | float):
                    trace_payload["uncertainty"] = prediction["uncertainty"]
                if isinstance(prediction.get("abstain"), bool):
                    trace_payload["abstained"] = prediction["abstain"]
                for event_name in (
                    "evidence",
                    "failure",
                    "recovery",
                    "handoff",
                    "tool",
                    "timing",
                ):
                    if event_name in prediction:
                        trace_payload[event_name] = prediction[event_name]
                trace_payloads.append(trace_payload)
                if prediction.get("done") is True:
                    break
            final_prediction = history[-1]["output"]
            context = {
                "kind": "interactive",
                "input": item,
                "label": labels[item_id],
                "prediction": final_prediction,
                "history": history,
            }
            trace_payloads[-1].update(context)
            trace_payloads[-1]["transition"] = {
                "history_length": len(history),
                "terminal": True,
                "oracle_evidence": labels[item_id],
            }
            vector = score_context(
                task=task,
                task_dir=task_dir,
                agent_identity=identity,
                seed=seed,
                context=context,
                runtime=verifier,
            )
            trace = ProceduralTrace.from_steps(
                trace_payloads,
                mode=InteractionMode.INTERACTIVE,
            )
            projection = project(vector, task.projection) if task.projection else None
            trials.append(
                TrialRecord(
                    seed=seed,
                    vector=vector,
                    trajectory=trace,
                    projection=projection,
                    projection_spec_digest=(task.projection.rule_digest if task.projection else ""),
                )
            )
    finally:
        _close(predictor)
        _close(verifier)
    footer = ""
    if task.environment.kind is WorldKind.ANGIOSTRESS_CONTRACT:
        footer = load_claim_footer(task_dir / task.environment.contract_path)
    return assemble_job_result(
        task=task,
        agent=agent,
        trials=tuple(trials),
        task_digest=task_digest,
        agent_digest=agent_digest,
        claim_footer=footer,
        world_engine=_engine_provenance(task, None),
    )


def _run_counterfactual(**kwargs: Any) -> JobResult:
    return _run_predictions(**kwargs, mode=InteractionMode.COUNTERFACTUAL)


def replay_job(
    out: Path,
    *,
    load_task: Callable[[Path], TaskSpec],
    load_agent: Callable[[Path], AgentPackage],
    gym_factory: GymFactory | None = None,
) -> JobResult:
    config = read_job_config(out)
    previous = read_job_result(out)
    task_dir = resolve_bundle_path(out, config["task_dir"], label="task")
    if tree_digest(task_dir) != config.get("task_digest"):
        raise TaskContractError("bundled task digest does not match config")
    task = load_task(task_dir)
    agent_dir_raw = config.get("agent_dir")
    if agent_dir_raw:
        agent_dir = resolve_bundle_path(out, agent_dir_raw, label="agent")
        if tree_digest(agent_dir) != config.get("agent_digest"):
            raise TaskContractError("bundled agent digest does not match config")
        agent = load_agent(agent_dir)
    else:
        agent_dir = None
        agent = builtin_random_agent(interface_id=str(config.get("interface") or task.interface.id))
        if digest(agent.model_dump(mode="json")) != config.get("agent_digest"):
            raise TaskContractError("builtin agent digest does not match config")
    assert_trajectory_matches_vector(
        out,
        task=task,
        task_dir=task_dir,
        result=previous,
        config=config,
    )
    rerun = run_job(
        task=task,
        task_dir=task_dir,
        agent=agent,
        agent_dir=agent_dir,
        out=out,
        n=int(config["n"]),
        gym_factory=gym_factory,
    )
    if rerun.head != previous.head:
        raise TaskContractError(f"replay head mismatch: stored {previous.head} reran {rerun.head}")
    if compute_head(rerun) != rerun.head:
        raise TaskContractError("rerun stamped a head that does not match its payload")
    return rerun
