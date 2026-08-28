"""``surgeval conformance``: measure a task package against the Tier-1 rules.

Prints one line per check, the *measured* determinism class, the adapter
identity that produced the observations, and the resulting tier. ``--require-
tier1`` turns the tier into an exit code, which is what a wrap's CI uses to
keep a curated package from silently degrading to Tier 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from or_audit.errors import ScoreContractError, TaskContractError
from or_audit.eval.conformance import (
    DEFAULT_TOLERANCE,
    ConformanceReport,
    run_conformance,
    write_conformance_report,
)


def _agent_dir(spec: str | None) -> Path | None:
    """Resolve ``-a`` to a package directory; ``random`` means the builtin agent."""
    if spec is None or spec == "random":
        return None
    path = Path(spec)
    return path if path.is_dir() else path.parent


def _print_report(report: ConformanceReport, *, out: Path) -> None:
    print(f"conformance: {report.task_id}@{report.task_version}")
    print(f"  world        {report.world_kind}")
    print(f"  adapter      {report.adapter_identity}")
    print(f"  adapter pin  {'declared' if report.adapter_pinned else 'MISSING'}")
    if report.metrics_only:
        print("  metrics-only declared: Tier 0 by §2.2, not safety-attested")
    print(f"  determinism  {report.determinism_class.value} (measured, tol {report.tolerance:g})")
    for check in report.checks:
        print(f"  [{'pass' if check.passed else 'FAIL'}] {check.id}: {check.detail}")
    print(f"  tier         {report.tier} — {report.tier_reason}")
    print(f"  report       {out / 'conformance.json'}")


def _conformance(args: argparse.Namespace) -> int:
    """Run the four checks, write the report, and gate on the tier if asked."""
    out = Path(args.out)
    try:
        report = run_conformance(
            Path(args.task),
            agent_dir=_agent_dir(args.agent),
            n=args.n,
            workdir=out,
            tolerance=args.tolerance,
        )
        write_conformance_report(report, out)
    except (TaskContractError, ScoreContractError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    _print_report(report, out=out)
    if args.require_tier1 and report.tier == 0:
        print(f"REFUSED: {report.tier_reason}", file=sys.stderr)
        return 1
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``conformance`` subcommand."""
    parser = sub.add_parser(
        "conformance",
        help="measure a task package against the Tier-1 conformance rules",
        description=(
            "Run gate-state availability, license audit, evidence replay, and a "
            "two-run execution-determinism measurement against a task package."
        ),
    )
    parser.add_argument("task", help="task directory or task.toml")
    parser.add_argument(
        "-a",
        "--agent",
        help="agent directory, or 'random' for the builtin random policy (default: random)",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=2,
        help="trials per run; the job runs twice to measure determinism (default: 2)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="directory for conformance.json/md and the two measured job runs",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=(
            "largest float delta two runs' trajectories may show and still be called "
            f"tolerance-deterministic (default: {DEFAULT_TOLERANCE:g})"
        ),
    )
    parser.add_argument(
        "--require-tier1",
        action="store_true",
        help="exit 1 unless the package earns Tier 1",
    )
    parser.set_defaults(func=_conformance)
