"""``surgeval export-verifiers``: a task package as a training environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from or_audit.errors import ScoreContractError, TaskContractError


def _export_verifiers(args: argparse.Namespace) -> int:
    """Generate a verifiers-style environment whose reward is a gated projection."""
    from or_audit.eval.export_verifiers import export_verifiers_environment

    out = Path(args.out)
    try:
        export = export_verifiers_environment(
            Path(args.path),
            out=out,
            projection_id=args.projection,
        )
    except (TaskContractError, ScoreContractError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"exported: {export.environment_id} -> {out}")
    for relative in export.paths:
        print(f"  {out / relative}")
    print(f"  task       {export.task_id}@{export.task_version} ({export.task_digest})")
    print(f"  projection {export.projection_identity}")
    print(f"  world      {export.world_kind} pin={export.world_pin or '(unpinned)'}")
    # The one thing a training team must not misread about this artifact.
    print(
        "WARNING: the exported scalar is a projection of a safety vector, not a "
        "score. A failed hard gate projects to 0 and the full vector is logged; "
        "do not report it as a result."
    )
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``export-verifiers`` subcommand."""
    parser = sub.add_parser(
        "export-verifiers",
        help="export a task as a verifiers-style environment (train-time surface)",
    )
    parser.add_argument("path", help="task package directory (or its task.toml)")
    parser.add_argument(
        "--projection",
        required=True,
        help="task-declared projection id; a mismatched or absent projection is refused",
    )
    parser.add_argument("--out", required=True, help="environment package output directory")
    parser.set_defaults(func=_export_verifiers)
