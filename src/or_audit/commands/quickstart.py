"""``surgeval quickstart`` — first vector in minutes, on any machine.

N10 item 1 and the instrumented half of N1. This runs the CPU-only reference
pair end to end (no GPU, no sim dependencies, no config-file editing), prints
the resulting vector, and reports the number that item 1 is actually about:
**time to first vector**, in seconds.

That measurement is the point. "Install is easy" is an assertion; a printed
wall-clock number on the user's own machine is evidence, and it regresses
visibly when someone adds a heavyweight import to the startup path. So the
harness stack is imported *inside* the timed handler rather than at module
scope: the eval loader and runner are the dominant startup cost, and a metric
that excluded them would flatter us. What the number does not cover is
interpreter startup and argparse, which no in-process clock can see; it is a
lower bound on wall clock, not an estimate of it.

The command refuses rather than degrades in two places: when the reference
packages are not on disk (neither shipped in the installed distribution nor
present in a checkout), and when the job's own head does not verify. A
quickstart that printed a vector nobody can replay would teach the wrong
lesson on first contact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from or_audit.errors import ScoreContractError, TaskContractError

if TYPE_CHECKING:
    from or_audit.eval.job import JobResult

DEFAULT_OUT = ".quickstart"


def _resolve_packages(args: argparse.Namespace) -> tuple[Path, Path]:
    """The task/agent pair to run: overrides first, then the shipped reference."""
    if args.task and args.agent:
        return Path(args.task), Path(args.agent)
    from or_audit.install.doctor import require_reference_paths

    task_default, agent_default = require_reference_paths()
    return (
        Path(args.task) if args.task else task_default,
        Path(args.agent) if args.agent else agent_default,
    )


def _gate_line(gate: dict[str, Any]) -> str:
    parts = [f"pass {gate['pass']}", f"fail {gate['fail']}"]
    if gate["not_assessable"]:
        parts.append(f"not_assessable {gate['not_assessable']}")
    if gate["not_applicable"]:
        parts.append(f"not_applicable {gate['not_applicable']}")
    return f"  gate   {gate['id']}: " + ", ".join(parts)


def _metric_line(metric: dict[str, Any]) -> str:
    label = f"{metric['id']}{'*' if metric['headline'] else ''}"
    unit = f" {metric['unit']}" if metric["unit"] else ""
    head = f"  metric {label} [{metric['kind']}, {metric['direction']}{unit}]: "
    if metric["kind"] == "boolean":
        rate = metric["rate"]
        body = f"true {metric['true']}, false {metric['false']}"
        if rate is not None:
            body += f", rate {rate:.3f}"
    elif metric["kind"] == "continuous":
        mean = metric["mean"]
        body = "no assessable value" if mean is None else f"mean {mean:.4g}"
        if mean is not None:
            body += f", min {metric['min']:.4g}, max {metric['max']:.4g}"
    else:
        body = ", ".join(f"{key} {value}" for key, value in metric["counts"].items()) or "no values"
    if metric["unassessable"]:
        body += f", unassessable {metric['unassessable']}"
    return head + body


def _print_summary(result: JobResult, out: Path, elapsed: float) -> None:
    """The vector as gates and metrics, then the numbers a newcomer needs.

    Rendered through ``scorecard_data`` rather than by reading trials here, so
    the first thing a user ever sees obeys the same no-composite-scalar rule as
    every published surface.
    """
    from or_audit.eval.scorecard import METRICS_ONLY_HEADLINE, scorecard_data

    engine = result.world_engine.model_dump(mode="json") if result.world_engine else None
    data = scorecard_data(result, world_engine=engine)
    print(f"vector: {data['task_id']}@{data['task_version']} n={data['n']}")
    print(f"  agent  {data['agent_identity']}")
    print(f"  world  {data['world_pin'] or '(no world pin)'}")
    if data["metrics_only"]:
        print(f"  {METRICS_ONLY_HEADLINE}")
    for gate in data["gates"]:
        print(_gate_line(gate))
    for metric in data["metrics"]:
        print(_metric_line(metric))
    if data["claim_footer"]:
        print(f"  footer {data['claim_footer']}")
    print(f"  head   {data['head']}")
    print()
    print(f"time to first vector: {elapsed:.2f}s")
    print(f"job dir: {out}")
    print(f"reproduce: surgeval replay {out} --expect-head {result.head}")


def _quickstart(args: argparse.Namespace) -> int:
    """Run the CPU-only reference task and report time-to-first-vector."""
    started = time.perf_counter()
    out = Path(args.out)
    try:
        task_dir, agent_dir = _resolve_packages(args)
        from or_audit.eval.job import verify_head
        from or_audit.eval.loader import load_agent, load_task
        from or_audit.eval.runner import run_job

        result = run_job(
            task=load_task(task_dir),
            task_dir=task_dir,
            agent=load_agent(agent_dir),
            agent_dir=agent_dir,
            out=out,
            n=args.n,
        )
        if not verify_head(result):
            raise TaskContractError(
                f"quickstart job at {out} does not verify its own head, so the vector is not "
                "replayable. Fix: re-run into a clean --out directory, and report this if it "
                "recurs — a non-verifying head is a harness defect, not a user error"
            )
    except (TaskContractError, ScoreContractError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started
    if args.json:
        print(
            json.dumps(
                {
                    "time_to_first_vector_sec": elapsed,
                    "task_id": result.task_id,
                    "head": result.head,
                    "out": str(out),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    _print_summary(result, out, elapsed)
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``quickstart`` command."""
    parser = sub.add_parser(
        "quickstart",
        help="run the CPU-only reference task and print time-to-first-vector",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"job directory to write (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--task", help="override the reference task package path")
    parser.add_argument("--agent", help="override the reference agent package path")
    parser.add_argument(
        "-n",
        type=int,
        help="episodes to run (default: the task's declared n_eval_episodes)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit {time_to_first_vector_sec, task_id, head, out} instead of the summary",
    )
    parser.set_defaults(func=_quickstart)
