"""``surgeval doctor`` — N10 item 3.

Thin by design: the diagnosis lives in :mod:`or_audit.install.doctor` so that
the cloud control plane and CI can call it without going through argparse.
This module owns only the two decisions a CLI owns — which worlds the user
asserted they want working, and whether to print for a human or a machine.
"""

from __future__ import annotations

import argparse
import json
import sys

from or_audit.errors import TaskContractError
from or_audit.install.doctor import run_doctor


def _doctor(args: argparse.Namespace) -> int:
    """Print per-check diagnosis; exit 1 when a required check failed."""
    try:
        report = run_doctor(packages=args.world or None)
    except TaskContractError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.render())
    return report.exit_code()


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``doctor`` command."""
    parser = sub.add_parser(
        "doctor",
        help="check this machine and print the fix for anything broken",
    )
    parser.add_argument(
        "--world",
        action="append",
        metavar="ID",
        help="require this catalog world to be working (repeatable)",
    )
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    parser.set_defaults(func=_doctor)
