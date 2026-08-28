"""``surgeval wrap``: self-serve scaffolding for a third-party world (N3).

The subcommand is deliberately thin. Everything that can refuse — an unpinned
world, an uncited threshold, a gate with no reported signal behind it — refuses
in :mod:`or_audit.eval.wrap`, so the same discipline applies to a programmatic
caller as to the CLI.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from or_audit.errors import TaskContractError
from or_audit.eval.wrap import GateMapping, WrapRequest, scaffold_wrap

_GATE_GRAMMAR = """gate mapping grammar (--gate, repeatable):

  ID=SIGNAL:EXPR@THRESHOLD:UNIT:CITATION   numeric-threshold gate
  ID=SIGNAL:EXPR                           boolean-signal gate (no number to cite)

  ID         gate id, a slug
  SIGNAL     the engine `info` key this wrap read, e.g. contact_force_n
  EXPR       fail condition over that signal only, e.g. contact_force_n > 1.5
  THRESHOLD  the number the gate bites at
  UNIT       physical unit of THRESHOLD, e.g. N, mm, mGy
  CITATION   normative source for THRESHOLD (may contain ':')

Quote the whole value once for your shell; nothing inside is unquoted for you, so
an EXPR wrapped in its own quotes becomes a string literal and is refused.

examples:

  --gate 'needle_force=contact_force_n:contact_force_n > 1.5@1.5:N:ORBIT needle
      handover force envelope v1, Table 2'
  --gate 'workspace=workspace_violation:workspace_violation == true'

A numeric threshold with no citation is refused: an unexplained safety number is
a heuristic wearing a unit. A wrap that maps no gates at all is refused unless it
ships --metrics-only, which labels the package explicitly not safety-attested.
"""


def _gate_mapping(raw: str) -> GateMapping:
    """Parse one ``--gate`` value, refusing anything malformed."""
    gate_id, assign, rest = raw.partition("=")
    if not assign or not gate_id or not rest:
        raise argparse.ArgumentTypeError(
            f"--gate {raw!r} is malformed: expected ID=SIGNAL:EXPR[@THRESHOLD:UNIT:CITATION]"
        )
    signal, colon, tail = rest.partition(":")
    if not colon or not signal or not tail:
        raise argparse.ArgumentTypeError(
            f"--gate {raw!r} is malformed: expected SIGNAL:EXPR after '='"
        )
    threshold: float | None = None
    unit = ""
    citation = ""
    if "@" in tail:
        expression, _, basis = tail.rpartition("@")
        parts = basis.split(":", 2)
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"--gate {raw!r} is malformed: a threshold needs THRESHOLD:UNIT:CITATION after '@'"
            )
        raw_threshold, unit, citation = parts
        try:
            threshold = float(raw_threshold)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--gate {raw!r} threshold {raw_threshold!r} is not a number"
            ) from None
        if not unit or not citation:
            raise argparse.ArgumentTypeError(
                f"--gate {raw!r} is malformed: THRESHOLD needs both a UNIT and a "
                "CITATION; an uncited safety number is refused"
            )
    else:
        expression = tail
    if not expression:
        raise argparse.ArgumentTypeError(f"--gate {raw!r} declares no fail condition")
    try:
        return GateMapping(
            id=gate_id,
            signal=signal,
            fail_when=expression,
            threshold=threshold,
            unit=unit,
            citation=citation,
        )
    except TaskContractError as exc:
        raise argparse.ArgumentTypeError(f"--gate {raw!r}: {exc}") from exc
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--gate {raw!r} is not a valid mapping: {exc}") from exc


def _parameters(raw: list[str] | None) -> dict[str, bool | int | float | str]:
    """Parse ``--param NAME=VALUE`` pairs into TOML-shaped scalars.

    Reuses TOML's own scalar rules rather than inventing a second syntax,
    because these values land verbatim in the generated ``task.toml``'s
    ``[environment].parameters`` and are forwarded to the real env
    constructor. A value that survives here but not there would be a
    generator that writes packages the runner cannot load.
    """
    parsed: dict[str, bool | int | float | str] = {}
    for item in raw or ():
        name, _, value = item.partition("=")
        if not name.strip() or not _:
            raise argparse.ArgumentTypeError(f"--param {item!r} is not NAME=VALUE")
        try:
            loaded = tomllib.loads(f"v = {value}")["v"]
        except tomllib.TOMLDecodeError:
            loaded = value
        if not isinstance(loaded, bool | int | float | str):
            raise argparse.ArgumentTypeError(
                f"--param {name.strip()!r} must be a scalar, got {type(loaded).__name__}"
            )
        parsed[name.strip()] = loaded
    return parsed


def _wrap(args: argparse.Namespace) -> int:
    """Scaffold a wrap package, or refuse and say what would make it publishable."""
    try:
        request = WrapRequest(
            env_id=args.env_id,
            task_id=args.task_id,
            world_kind=args.world_kind,
            world_pin=args.world_pin,
            license=args.license,
            interface_id=args.interface,
            modality=args.modality,
            source_repo=args.source_repo,
            gate_mappings=tuple(args.gate or ()),
            metrics_only=args.metrics_only,
            synthetic_stub=args.synthetic_stub,
            n_eval_episodes=args.episodes,
            max_steps=args.max_steps,
            parameters=_parameters(args.params),
        )
        result = scaffold_wrap(request, Path(args.out))
    except (TaskContractError, argparse.ArgumentTypeError) as exc:
        # `_parameters` runs here rather than as an argparse `type=` callback,
        # so its refusals surface after parsing is finished. Uncaught, a plain
        # `--param typo` printed a traceback instead of the refusal this
        # command promises for every other contract violation.
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"wrapped: {request.env_id} -> {result.root}")
    for path in result.files:
        print(f"  {path}")
    adapter = result.adapter_id or "(no adapter installed; capabilities declared in-package)"
    print(f"  world      {result.world_kind} pin={result.world_pin}")
    print(f"  adapter    {adapter}")
    if result.metrics_only:
        print("  gates      (none) metrics-only: explicitly not safety-attested")
    else:
        print(f"  gates      {', '.join(result.gate_ids)}")
    print(f"next: {result.conformance_command}")
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``wrap`` subcommand."""
    parser = sub.add_parser(
        "wrap",
        help="scaffold a task package around a third-party world",
        description=(
            "Scaffold a task-package skeleton for a third-party env: pinned world, "
            "mapped safety signals, cited thresholds, and a verifier that defaults "
            "nothing. Generation is deterministic and overwrites in place."
        ),
        epilog=_GATE_GRAMMAR,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("env_id", metavar="ENV", help="upstream env id, e.g. SurRoL/NeedleReach-v0")
    parser.add_argument("--task-id", required=True, help="slug for the generated task package")
    parser.add_argument(
        "--world-pin",
        required=True,
        help="immutable revision of the wrapped world; a wrap that cannot pin is refused",
    )
    parser.add_argument("--out", required=True, help="directory to write the package into")
    parser.add_argument("--world-kind", default="gym", help="world kind hosting the env (gym)")
    parser.add_argument("--interface", default="gym-policy", help="task interface id (gym-policy)")
    parser.add_argument(
        "--modality", default="robotic-kinematics", help="procedural modality (robotic-kinematics)"
    )
    parser.add_argument("--license", required=True, help="SPDX license id of the wrapped env")
    parser.add_argument("--source-repo", default="", help="upstream repository URL")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="the env reports no gate-able safety state: ship it labelled not safety-attested",
    )
    parser.add_argument(
        "--synthetic-stub",
        action="store_true",
        help="permit a non-physical stand-in; artifacts are stamped and RL export refuses them",
    )
    parser.add_argument(
        "--gate",
        action="append",
        type=_gate_mapping,
        metavar="SPEC",
        help="map a hard gate onto a reported signal (see grammar below); repeatable",
    )
    parser.add_argument(
        "-n", "--episodes", type=int, default=5, help="evaluation episodes per run (5)"
    )
    parser.add_argument("--max-steps", type=int, default=100, help="harness step limit (100)")
    parser.add_argument(
        "--param",
        action="append",
        metavar="NAME=VALUE",
        dest="params",
        help="world construction parameter, repeatable. Required when a bound signal's "
        "audited kind depends on construction (e.g. --param with_board_collision=true "
        "for LapGym's collision_with_board). Parsed as TOML scalars: true/false, "
        "numbers, else string.",
    )
    parser.set_defaults(func=_wrap)
