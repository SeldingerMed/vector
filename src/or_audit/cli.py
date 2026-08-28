"""Command-line entry points.

``demo`` / ``verify-audit`` / ``describe-rule``
    The original credentialing-mode tools. Still work; they are not the wedge.
``tasks validate`` / ``tasks describe`` / ``datasets validate`` /
``agents validate`` / ``bind`` / ``run`` / ``replay`` / ``export-rl``
    BUILD.md P0-P3: Harbor-shaped eval contract, gym-policy and video-predict
    runners, cartesian ``job.toml``, and a versioned RL projection dump.

Written against ``argparse`` rather than a CLI framework.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from or_audit.audit.trail import AuditTrail
from or_audit.commands import register_all
from or_audit.decision.rule import DecisionRule
from or_audit.demo import run_demo
from or_audit.domain.enums import ThresholdOwner
from or_audit.errors import AuditChainError, ScoreContractError, TaskContractError
from or_audit.eval.agent import AgentPackage
from or_audit.eval.bind import assert_bind
from or_audit.eval.loader import (
    load_agent,
    load_task,
    load_taskset,
    taskset_task_paths,
)
from or_audit.eval.registry import (
    DEFAULT_REGISTRY,
    load_registry,
    materialize_entry,
    pull_entry,
    resolve_entry,
)
from or_audit.eval.runner import builtin_random_agent, replay_job, run_job
from or_audit.version import PACKAGE_VERSION

_DISCLAIMER = (
    "NOTE: synthetic data. This repository holds no clinical media, and the "
    "detectors are screening heuristics, not validated classifiers. Nothing "
    "here is evidence of clinical performance (PLAN.md sections 8, 9, V)."
)


def _demo(args: argparse.Namespace) -> int:
    """Run the synthetic end-to-end demo."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(args.workdir) if args.workdir else Path(tmp)
        workdir.mkdir(parents=True, exist_ok=True)
        outcome = run_demo(workdir, episodes=args.episodes)

        print(outcome.report.render())
        print()
        print("AUDIT TRAIL")
        print(f"  entries      {len(outcome.trail)}")
        print(f"  head         {outcome.trail.head_hash}")
        try:
            outcome.trail.verify(
                expected_head=outcome.trail.head_hash, expected_length=len(outcome.trail)
            )
            print("  verification intact (chain + pinned head + pinned length)")
        except AuditChainError as exc:  # pragma: no cover - defensive
            print(f"  verification FAILED: {exc}")
            return 1
        print()
        print("DE-IDENTIFICATION")
        print(f"  segments dropped {outcome.deid_frames_dropped}")
        print(f"  regions masked   {outcome.deid_boxes_masked}")
        if args.audit_log:
            outcome.trail.to_jsonl(Path(args.audit_log))
            print(f"  audit log written to {args.audit_log}")
        print()
        print(_DISCLAIMER)
    return 0


def _verify_audit(args: argparse.Namespace) -> int:
    """Verify an audit log, optionally against a pinned head."""
    path = Path(args.path)
    if not path.is_file():
        print(f"no such audit log: {path}", file=sys.stderr)
        return 2
    try:
        trail = AuditTrail.from_jsonl(path, verify=False)
    except AuditChainError as exc:
        print(f"UNREADABLE: {exc}", file=sys.stderr)
        return 1
    try:
        trail.verify(
            expected_head=args.expected_head,
            expected_length=args.expected_length,
        )
    except AuditChainError as exc:
        print(f"BROKEN: {exc}", file=sys.stderr)
        return 1
    print(f"intact: {len(trail)} entries, head {trail.head_hash}")
    if args.expected_head is None:
        print(
            "  WARNING: no --expected-head supplied. Tail truncation is not "
            "detectable from the chain alone; pin the head externally for this "
            "check to mean anything."
        )
        if not args.allow_unpinned:
            print(
                "  INCOMPLETE: refusing to report a clean result from an "
                "unpinned check. Supply --expected-head, or --allow-unpinned to "
                "acknowledge that truncation was not checked.",
                file=sys.stderr,
            )
            return 3
    return 0


def _describe_rule(args: argparse.Namespace) -> int:
    """Print a decision rule for pre-registration."""
    rule = DecisionRule(
        version=args.version,
        threshold_owner=ThresholdOwner(args.threshold_owner),
        threshold_provenance=args.provenance,
        min_proficiency_fraction=args.min_proficiency,
        min_assessable_items=args.min_items,
    )
    print(rule.describe())
    return 0


def _tasks_validate(args: argparse.Namespace) -> int:
    """Load a task directory and exit 0 only if the contract holds."""
    try:
        task = load_task(Path(args.path))
    except TaskContractError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    try:
        task.assert_runnable()
        runnable = "runnable"
    except TaskContractError:
        runnable = "valid (not runnable)"
    print(f"valid: {task.id}@{task.task_version} {runnable}")
    return 0


def _tasks_describe(args: argparse.Namespace) -> int:
    """Print a task the way Harbor would print a task.toml."""
    try:
        task = load_task(Path(args.path))
    except TaskContractError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(task.describe())
    print()
    print(task.instruction)
    return 0


def _tasksets_validate(args: argparse.Namespace) -> int:
    """Load a taskset and every task it names."""
    try:
        taskset = load_taskset(Path(args.path))
    except TaskContractError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(
        f"valid: {taskset.id}@{taskset.taskset_version} "
        f"{len(taskset.tasks)} task(s), headline {taskset.headline}"
    )
    return 0


def _registry_list(args: argparse.Namespace) -> int:
    try:
        index = load_registry(args.registry)
        entries = index.tasksets if args.registry_kind == "taskset" else index.agents
        for entry in sorted(entries, key=lambda item: item.reference):
            print(f"{entry.reference}\t{entry.digest}\t{entry.repository}@{entry.ref}")
        return 0
    except TaskContractError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1


def _registry_pull(args: argparse.Namespace) -> int:
    try:
        index = load_registry(args.registry)
        entry = resolve_entry(index, kind=args.registry_kind, ref=args.reference)
        target = pull_entry(entry, Path(args.out))
        print(f"pulled: {entry.reference} -> {target}")
        return 0
    except TaskContractError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1


def _agents_validate(args: argparse.Namespace) -> int:
    """Load an org/name agent package."""
    try:
        agent = load_agent(Path(args.path))
    except TaskContractError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(
        f"valid: {agent.id}@{agent.agent_version} "
        f"capabilities={','.join(cap.interface for cap in agent.capabilities)}"
    )
    return 0


def _bind(args: argparse.Namespace) -> int:
    """Refuse a pair unless one agent capability satisfies the task interface."""
    try:
        task = load_task(Path(args.task))
        agent = load_agent(Path(args.agent))
        assert_bind(task, agent)
    except TaskContractError as exc:
        print(f"INCOMPATIBLE: {exc}", file=sys.stderr)
        return 1
    print(f"bind: {agent.id} -> {task.id} interface={task.interface.id}")
    return 0


def _resolve_agent(spec: str, *, registry_source: str) -> tuple[AgentPackage, Path | None]:
    if spec == "random":
        return builtin_random_agent(), None
    path = Path(spec)
    if not path.exists() and "@" in spec:
        entry = resolve_entry(load_registry(registry_source), kind="agent", ref=spec)
        path = materialize_entry(entry)
    return load_agent(path), path if path.is_dir() else path.parent


def _run(args: argparse.Namespace) -> int:
    """Bind and run a task, taskset, or cartesian job.toml."""
    n_sources = sum(bool(flag) for flag in (args.task, args.taskset, args.job))
    if n_sources != 1:
        print(
            "run requires exactly one of -t/--task, -s/--taskset, or -c/--job",
            file=sys.stderr,
        )
        return 2
    if not args.job and not args.agent:
        print("run requires -a/--agent unless -c/--job", file=sys.stderr)
        return 2
    if args.job and args.agent:
        print(
            "run -c/--job takes agents from the job file; omit -a/--agent",
            file=sys.stderr,
        )
        return 2
    try:
        n = args.n if args.n else None
        if args.job:
            from or_audit.eval.cartesian import run_cartesian_job
            from or_audit.eval.job_config import resolve_job

            manifest = run_cartesian_job(
                resolve_job(Path(args.job)),
                out=Path(args.out),
                n=n,
            )
            print(f"ran: {manifest.id} pairs={len(manifest.pairs)} head {manifest.head}")
            return 0
        agent, agent_dir = _resolve_agent(args.agent, registry_source=args.registry)
        if args.task:
            task_path = Path(args.task)
            task_dir = task_path if task_path.is_dir() else task_path.parent
            result = run_job(
                task=load_task(task_path),
                task_dir=task_dir,
                agent=agent,
                agent_dir=agent_dir,
                out=Path(args.out),
                n=n,
            )
            print(f"ran: {result.task_id} n={result.n} head {result.head}")
            return 0
        taskset_path = Path(args.taskset)
        if not taskset_path.exists() and "@" in args.taskset:
            entry = resolve_entry(
                load_registry(args.registry),
                kind="taskset",
                ref=args.taskset,
            )
            taskset_path = materialize_entry(entry)
        load_taskset(taskset_path)
        out_root = Path(args.out)
        for task_path in taskset_task_paths(taskset_path):
            task = load_task(task_path)
            task_dir = task_path if task_path.is_dir() else task_path.parent
            result = run_job(
                task=task,
                task_dir=task_dir,
                agent=agent,
                agent_dir=agent_dir,
                out=out_root / task.id,
                n=n,
            )
            print(f"ran: {result.task_id} n={result.n} head {result.head}")
        return 0
    except (TaskContractError, ScoreContractError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1


def _replay(args: argparse.Namespace) -> int:
    """Re-run a job (or cartesian parent) and require the stored head to match."""
    path = Path(args.path)
    try:
        if (path / "manifest.json").is_file():
            from or_audit.eval.cartesian import replay_cartesian

            manifest = replay_cartesian(path, load_task=load_task, load_agent=load_agent)
            head = manifest.head
            label = manifest.id
        else:
            result = replay_job(path, load_task=load_task, load_agent=load_agent)
            head = result.head
            label = result.task_id
    except (TaskContractError, ScoreContractError) as exc:
        print(f"REPLAY FAILED: {exc}", file=sys.stderr)
        return 1
    if args.expect_head and args.expect_head != head:
        print(
            f"REPLAY FAILED: expected head {args.expect_head}, got {head}",
            file=sys.stderr,
        )
        return 1
    print(f"replay matched: {label} head {head}")
    return 0


def _export_rl(args: argparse.Namespace) -> int:
    """Dump a task-declared versioned projection JSONL recomputed from vectors."""
    from or_audit.eval.export_rl import export_rl

    try:
        n = export_rl(
            Path(args.path),
            projection_id=args.projection,
            out=Path(args.out),
        )
    except (TaskContractError, ScoreContractError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"exported: {n} episodes -> {args.out}")
    return 0


def _leaderboard_build(args: argparse.Namespace) -> int:
    from or_audit.eval.leaderboard import write_leaderboard

    try:
        data = write_leaderboard([Path(path) for path in args.paths], Path(args.out))
    except (TaskContractError, ScoreContractError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"leaderboard: {len(data['rows'])} row(s) -> {args.out}")
    return 0


def _cloud_serve(args: argparse.Namespace) -> int:
    """Run the optional Vector Cloud control plane."""
    if (args.allow_anonymous or args.enable_local) and args.host not in {
        "127.0.0.1",
        "::1",
        "localhost",
    }:
        print("cloud local development may only bind a loopback host", file=sys.stderr)
        return 2
    os.environ["VECTOR_CLOUD_DB"] = args.db
    os.environ["VECTOR_CLOUD_DATA"] = args.data
    os.environ["VECTOR_CLOUD_PACKAGE_ROOT"] = str(Path(args.package_root).resolve())
    if args.enable_local:
        os.environ["VECTOR_CLOUD_ENABLE_LOCAL"] = "1"
    else:
        os.environ.pop("VECTOR_CLOUD_ENABLE_LOCAL", None)
    if args.allow_anonymous:
        os.environ["VECTOR_CLOUD_ALLOW_ANONYMOUS"] = "1"
    else:
        os.environ.pop("VECTOR_CLOUD_ALLOW_ANONYMOUS", None)
    try:
        import uvicorn

        from or_audit.cloud.api import app_from_env

        app = app_from_env()
    except ImportError:
        print('cloud dependencies missing; install "surgeval[cloud]"', file=sys.stderr)
        return 1
    except TaskContractError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _cloud_worker(args: argparse.Namespace) -> int:
    """Run one provider job and return its evidence to the control plane."""
    del args
    from or_audit.cloud.worker import run_from_env

    return run_from_env()


def _adapters_list(args: argparse.Namespace) -> int:
    del args
    from or_audit.eval.adapters import list_adapters, reset_default_adapters

    adapters = list_adapters()
    if not adapters:
        reset_default_adapters()
        adapters = list_adapters()
    print(f"Registered Modality Adapters ({len(adapters)}):")
    for mod, cls_name in sorted(adapters.items()):
        print(f"  {mod:<25} -> {cls_name}")
    return 0


def _sim_list(args: argparse.Namespace) -> int:
    del args
    from or_audit.eval.sim import list_simulation_engines, reset_default_simulation_engines

    engines = list_simulation_engines()
    if not engines:
        reset_default_simulation_engines()
        engines = list_simulation_engines()
    print(f"Registered Simulation Engines ({len(engines)}):")
    for kind, factory_name in sorted(engines.items()):
        print(f"  {kind:<25} -> {factory_name}")
    return 0


def _sim_kinds(args: argparse.Namespace) -> int:
    """Print the world-kind registry: what each kind is eligible for, and who serves it."""
    del args
    from or_audit.eval.sim import list_world_kinds, world_adapter_discovery

    kinds = list_world_kinds()
    print(f"Registered World Kinds ({len(kinds)}):")
    for kind, spec in kinds.items():
        capabilities = spec.capabilities
        flags = ",".join(
            name
            for name, enabled in (
                ("physics", capabilities.physics),
                ("closed-loop", capabilities.closed_loop),
                ("counterfactual", capabilities.counterfactual),
            )
            if enabled
        )
        print(f"  {kind:<25} {flags or '(no eligibility)'}")
        print(f"    determinism {capabilities.determinism_class.value}")
        print(f"    adapter     {spec.adapter_identity}")
        if spec.provider:
            print(f"    provider    {spec.provider}")
    failures = tuple(item for item in world_adapter_discovery() if not item.ok)
    if failures:
        print()
        print(f"Failed world-kind plugins ({len(failures)}):")
        for failure in failures:
            print(f"  {failure.name}: {failure.error}")
        return 1
    return 0


def _cli_prog() -> str:
    """Return the installed command name for help output."""
    if not sys.argv:
        return "surgeval"
    stem = Path(sys.argv[0]).stem
    if stem in {"surgeval", "vector", "or-audit"}:
        return stem
    return "surgeval"


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog=prog or _cli_prog(),
        description="Vector evaluation harness for procedural medical AI. "
        "Also still runs the synthetic credentialing demo.",
    )
    parser.add_argument("--version", action="version", version=PACKAGE_VERSION)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the synthetic end-to-end pipeline")
    demo.add_argument("--episodes", type=int, default=8, help="episodes to assess")
    demo.add_argument("--workdir", help="directory for redacted output (default: temporary)")
    demo.add_argument("--audit-log", help="write the audit trail to this JSONL path")
    demo.set_defaults(func=_demo)

    verify = sub.add_parser("verify-audit", help="verify an audit log")
    verify.add_argument("path", help="JSONL audit log")
    verify.add_argument(
        "--expected-head", help="externally pinned head hash; required to detect truncation"
    )
    verify.add_argument("--expected-length", type=int, help="externally pinned entry count")
    verify.add_argument(
        "--allow-unpinned",
        action="store_true",
        help=(
            "accept a verification with no pinned head. Exits 3 without this, "
            "because an unpinned check cannot detect truncation and a clean "
            "exit code would overstate what was verified."
        ),
    )
    verify.set_defaults(func=_verify_audit)

    describe = sub.add_parser("describe-rule", help="print a decision rule for publication")
    describe.add_argument("--version", default="1")
    describe.add_argument(
        "--threshold-owner",
        default=ThresholdOwner.CUSTOMER.value,
        choices=[owner.value for owner in ThresholdOwner],
    )
    describe.add_argument(
        "--provenance", default="Credentialing committee minute", help="threshold provenance"
    )
    describe.add_argument("--min-proficiency", type=float, default=0.85)
    describe.add_argument("--min-items", type=int, default=5)
    describe.set_defaults(func=_describe_rule)

    tasks = sub.add_parser("tasks", help="validate or describe a Harbor-shaped eval task")
    tasks_sub = tasks.add_subparsers(dest="tasks_command", required=True)
    tasks_validate = tasks_sub.add_parser("validate", help="load and check a task directory")
    tasks_validate.add_argument("path", help="task directory or task.toml")
    tasks_validate.set_defaults(func=_tasks_validate)
    tasks_describe = tasks_sub.add_parser("describe", help="print a task's contract")
    tasks_describe.add_argument("path", help="task directory or task.toml")
    tasks_describe.set_defaults(func=_tasks_describe)

    tasksets = sub.add_parser("tasksets", help="validate a versioned task collection")
    tasksets_sub = tasksets.add_subparsers(dest="tasksets_command", required=True)
    tasksets_validate = tasksets_sub.add_parser("validate", help="load a taskset and its tasks")
    tasksets_validate.add_argument("path", help="taskset directory or taskset.toml")
    tasksets_validate.set_defaults(func=_tasksets_validate)
    tasksets_list = tasksets_sub.add_parser("list", help="list registry tasksets")
    tasksets_list.add_argument("--registry", default=DEFAULT_REGISTRY)
    tasksets_list.set_defaults(func=_registry_list, registry_kind="taskset")
    tasksets_pull = tasksets_sub.add_parser("pull", help="pull a verified registry taskset")
    tasksets_pull.add_argument("reference", help="org/name@version")
    tasksets_pull.add_argument("--registry", default=DEFAULT_REGISTRY)
    tasksets_pull.add_argument("--out", required=True)
    tasksets_pull.set_defaults(func=_registry_pull, registry_kind="taskset")

    datasets = sub.add_parser("datasets", help="compatibility alias for tasksets")
    datasets_sub = datasets.add_subparsers(dest="datasets_command", required=True)
    datasets_validate = datasets_sub.add_parser("validate")
    datasets_validate.add_argument("path")
    datasets_validate.set_defaults(func=_tasksets_validate)
    datasets_list = datasets_sub.add_parser("list")
    datasets_list.add_argument("--registry", default=DEFAULT_REGISTRY)
    datasets_list.set_defaults(func=_registry_list, registry_kind="taskset")
    datasets_pull = datasets_sub.add_parser("pull")
    datasets_pull.add_argument("reference")
    datasets_pull.add_argument("--registry", default=DEFAULT_REGISTRY)
    datasets_pull.add_argument("--out", required=True)
    datasets_pull.set_defaults(func=_registry_pull, registry_kind="taskset")

    agents = sub.add_parser("agents", help="validate an org/name agent package")
    agents_sub = agents.add_subparsers(dest="agents_command", required=True)
    agents_validate = agents_sub.add_parser("validate", help="load and check an agent directory")
    agents_validate.add_argument("path", help="agent directory or agent.toml")
    agents_validate.set_defaults(func=_agents_validate)
    agents_list = agents_sub.add_parser("list", help="list registry agents")
    agents_list.add_argument("--registry", default=DEFAULT_REGISTRY)
    agents_list.set_defaults(func=_registry_list, registry_kind="agent")
    agents_pull = agents_sub.add_parser("pull", help="pull a verified registry agent")
    agents_pull.add_argument("reference", help="org/name@version")
    agents_pull.add_argument("--registry", default=DEFAULT_REGISTRY)
    agents_pull.add_argument("--out", required=True)
    agents_pull.set_defaults(func=_registry_pull, registry_kind="agent")

    bind = sub.add_parser(
        "bind",
        help="check that an agent capability satisfies a task interface",
    )
    bind.add_argument("task", help="task directory or task.toml")
    bind.add_argument("agent", help="agent directory or agent.toml")
    bind.set_defaults(func=_bind)

    run = sub.add_parser("run", help="evaluate an agent on a task, taskset, or job.toml")
    run.add_argument("-t", "--task", help="task directory or task.toml")
    run.add_argument("-s", "--taskset", "-d", "--dataset", help="taskset directory or taskset.toml")
    run.add_argument("-c", "--job", help="job.toml cartesian product of agents x tasks")
    run.add_argument("-a", "--agent", help="agent directory, or 'random' (required unless -c)")
    run.add_argument(
        "-n",
        "--n",
        type=int,
        default=0,
        help="episodes (CLI > job.toml n > task n_eval_episodes)",
    )
    run.add_argument("--out", required=True, help="job output directory")
    run.add_argument(
        "--registry",
        default=DEFAULT_REGISTRY,
        help="registry.json path or URL for org/name@version references",
    )
    run.set_defaults(func=_run)
    leaderboard = sub.add_parser(
        "leaderboard",
        help="build a static task-scoped safety-vector leaderboard",
    )
    leaderboard.add_argument("paths", nargs="+", help="job directories or parents")
    leaderboard.add_argument("--out", required=True, help="static output directory")
    leaderboard.set_defaults(func=_leaderboard_build)

    replay = sub.add_parser("replay", help="re-run a job and match its stored head")
    replay.add_argument("path", help="job directory or cartesian parent")
    replay.add_argument("--expect-head", help="require this job or manifest head")
    replay.set_defaults(func=_replay)

    export_rl = sub.add_parser(
        "export-rl",
        help="write a versioned projection jsonl from a job (not a leaderboard row)",
    )
    export_rl.add_argument("path", help="job directory or cartesian parent")
    export_rl.add_argument(
        "--projection",
        required=True,
        help="task-declared projection id; arbitrary code and stored floats are refused",
    )
    export_rl.add_argument("--out", required=True, help="jsonl output path")
    export_rl.set_defaults(func=_export_rl)

    cloud = sub.add_parser("cloud", help="run the optional Vector Cloud control plane")
    cloud_sub = cloud.add_subparsers(dest="cloud_command", required=True)
    cloud_serve = cloud_sub.add_parser("serve", help="serve the job API")
    cloud_serve.add_argument("--host", default="127.0.0.1")
    cloud_serve.add_argument("--port", type=int, default=8787)
    cloud_serve.add_argument("--db", default=".vector-cloud/jobs.sqlite")
    cloud_serve.add_argument("--data", default=".vector-cloud/jobs")
    cloud_serve.add_argument("--package-root", default=".")
    cloud_serve.add_argument(
        "--enable-local",
        action="store_true",
        help="development only: execute task packages on this host",
    )
    cloud_serve.add_argument(
        "--allow-anonymous",
        action="store_true",
        help="development only: disable bearer auth (loopback only)",
    )
    cloud_serve.set_defaults(func=_cloud_serve)
    cloud_worker = cloud_sub.add_parser("worker", help="run one configured provider job")
    cloud_worker.set_defaults(func=_cloud_worker)

    adapters = sub.add_parser("adapters", help="inspect registered modality adapters")
    adapters_sub = adapters.add_subparsers(dest="adapters_command", required=True)
    adapters_list = adapters_sub.add_parser("list", help="list registered modality adapters")
    adapters_list.set_defaults(func=_adapters_list)

    sim = sub.add_parser("sim", help="inspect registered simulation engines")
    sim_sub = sim.add_subparsers(dest="sim_command", required=True)
    sim_list = sim_sub.add_parser("list", help="list registered simulation engine bridges")
    sim_list.set_defaults(func=_sim_list)
    sim_kinds = sim_sub.add_parser(
        "kinds", help="list registered world kinds, capabilities, and adapter identities"
    )
    sim_kinds.set_defaults(func=_sim_kinds)

    register_all(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
