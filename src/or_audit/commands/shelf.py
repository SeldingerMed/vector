"""``surgeval shelf``: build a per-world shelf, check equivalence, rank.

The command surface mirrors the kernel's posture: building and ranking a shelf
is always available, ordering *across* worlds is not — it requires a validated
§2.6 equivalence artifact and says exactly what is missing on refusal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from or_audit.errors import ScoreContractError, TaskContractError
from or_audit.eval.equivalence import (
    EquivalenceArtifact,
    load_equivalence_artifact,
    validate_equivalence,
)
from or_audit.eval.shelf import (
    CROSS_WORLD_REFUSAL,
    ShelfGateRef,
    build_shelf,
    load_shelf_report,
    load_shelf_spec,
    refuse_cross_world_aggregate,
    shelf_ranking,
)


def _refuse(message: str) -> int:
    print(f"REFUSED: {message}", file=sys.stderr)
    return 1


def _shelf_build(args: argparse.Namespace) -> int:
    """Build a shelf's per-world and per-bench rows from verified job bundles."""
    try:
        spec = load_shelf_spec(Path(args.spec))
        report = build_shelf(spec, [Path(path) for path in args.jobs], out=Path(args.out))
    except (TaskContractError, ScoreContractError) as exc:
        return _refuse(str(exc))
    for world in report.worlds:
        print(f"world {world.entry.world_id}: {len(world.rows)} row(s) [{world.entry.task_id}]")
    for bench in report.benches:
        print(f"bench {bench.entry.task_id}: {len(bench.rows)} row(s)")
    print(f"shelf: {spec.id} -> {args.out}")
    print(f"note: {CROSS_WORLD_REFUSAL}")
    return 0


def _load_artifact(path: str) -> EquivalenceArtifact | int:
    try:
        return load_equivalence_artifact(Path(path))
    except TaskContractError as exc:
        return _refuse(str(exc))


def _equivalence_check(args: argparse.Namespace) -> int:
    """Report whether an equivalence artifact meets all four §2.6 requirements."""
    artifact = _load_artifact(args.artifact)
    if isinstance(artifact, int):
        return artifact
    # Without a shelf, the gate map is only checked against itself. That is a
    # requirement nobody ran, so this command must not print it as one that
    # passed - and `--shelf` is how the caller gets the real answer.
    world_gates: dict[str, tuple[ShelfGateRef, ...]] | None = None
    if args.shelf:
        try:
            report = load_shelf_report(Path(args.shelf))
            refuse_cross_world_aggregate(
                report,
                task_family=artifact.task_family,
                operation="check",
                equivalence=artifact,
            )
        except (TaskContractError, ScoreContractError) as exc:
            return _refuse(str(exc))
        world_gates = {
            world.entry.world_id: world.gate_manifest
            for world in report.worlds
            if world.gate_manifest is not None
        }
    verdict = validate_equivalence(artifact, world_gates=world_gates)
    print(f"artifact: {verdict.artifact_id}")
    print(f"shelf: {verdict.shelf_id}  family: {verdict.task_family}")
    print(f"worlds: {verdict.world_pair[0]} <-> {verdict.world_pair[1]}")
    for requirement, ok in verdict.requirements.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {requirement}")
    if not verdict.shelf_gates_checked:
        print("  [not run] shelf_gates — pass --shelf <built shelf> to check this")
        print("            artifact's gate map against that shelf's gate manifest")
    if verdict.computed_rank_correlation is not None:
        print(f"referent rank correlation: {verdict.computed_rank_correlation:.4g}")
    if verdict.valid:
        if verdict.shelf_gates_checked:
            print("equivalence: VALID — cross-world ordering is unlocked for this pair")
        else:
            print(
                "equivalence: VALID against its own declarations — cross-world ordering "
                "stays refused until the shelf_gates check above runs"
            )
        return 0
    for failure in verdict.failures:
        print(f"  - {failure}", file=sys.stderr)
    return _refuse(f"equivalence artifact fails {', '.join(verdict.failed_requirements)}")


def _print_orders(ranking: dict[str, Any]) -> None:
    for world in ranking["per_world"]:
        print(f"world {world['world_id']} ({world['task_family']}):")
        _print_order(world["order"])
    for bench in ranking["benches"]:
        print(f"bench {bench['task_id']} (real data):")
        _print_order(bench["order"])


def _gate_label(entry: dict[str, Any]) -> str:
    """Say whether gates were verified, never just whether one failed.

    A Tier-0 metrics-only world has no hard gates to fail, so ``any_gate_failed``
    is falsy for a row that checked no safety state at all. Reading "gates-pass"
    off that is the row claiming a verification it never ran.
    """
    if entry["safety_attested"]:
        return "gate-failed" if entry["any_gate_failed"] else "gates-pass"
    if entry["metrics_only"]:
        return "gates-not-attested (Tier-0 metrics-only world, no safety state to gate)"
    return "gates-not-attested (task declares no hard gates)"


def _backend_label(entry: dict[str, Any]) -> str:
    """Name the world that produced the row; absence is unattested, not real.

    ``surgeval shelf rank`` reads as a leaderboard, so a synthetic-stub or
    provenance-less row must not print like a real-world one. The shelf HTML
    already flags both; this is the same fact on the CLI.
    """
    backend = entry["backend"]
    if backend is None:
        return "backend=UNATTESTED"
    if backend == "synthetic-stub":
        return "backend=SYNTHETIC-STUB"
    return f"backend={backend}"


def _print_order(order: list[dict[str, Any]]) -> None:
    if not order:
        print("  (no rows)")
    for entry in order:
        value = entry["headline_value"]
        shown = "—" if value is None else f"{value:.4g}"
        print(
            f"  {entry['rank']}. {entry['agent_identity']} {entry['headline']}={shown} "
            f"{_gate_label(entry)} {_backend_label(entry)} "
            f"unassessable={entry['unassessable']}"
        )


def _shelf_rank(args: argparse.Namespace) -> int:
    """Print per-world orderings; cross-world only under a validated artifact."""
    try:
        report = load_shelf_report(Path(args.path))
    except (TaskContractError, ScoreContractError) as exc:
        return _refuse(str(exc))

    artifact: EquivalenceArtifact | None = None
    if args.equivalence:
        loaded = _load_artifact(args.equivalence)
        if isinstance(loaded, int):
            return loaded
        artifact = loaded
    elif args.cross_world:
        return _refuse(
            f"cross-world ordering requested for shelf '{report.spec.id}' without "
            f"--equivalence: {CROSS_WORLD_REFUSAL}"
        )

    try:
        ranking = shelf_ranking(report, equivalence=artifact)
    except (TaskContractError, ScoreContractError) as exc:
        return _refuse(str(exc))

    _print_orders(ranking)
    cross = ranking["cross_world"]
    if cross is None:
        print(f"cross-world: refused ({CROSS_WORLD_REFUSAL})")
        return 0
    pair = " <-> ".join(cross["world_pair"])
    print(f"cross-world ({pair}, family {cross['task_family']}):")
    for position, entry in enumerate(cross["order"], start=1):
        ranks = ", ".join(f"{world}={rank}" for world, rank in entry["world_ranks"].items())
        print(
            f"  {position}. {entry['agent_identity']} "
            f"mean-rank={entry['mean_world_rank']:.4g} ({ranks})"
        )
    excluded = cross["excluded_partial_coverage"]
    if excluded:
        print(f"  excluded (not run in both worlds): {', '.join(excluded)}")
    print(f"  licensed by: {json.dumps(cross['equivalence_artifact'], sort_keys=True)}")
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``shelf`` command group to the top-level subparsers."""
    shelf = sub.add_parser("shelf", help="build, check, and rank a per-world benchmark shelf")
    shelf_sub = shelf.add_subparsers(dest="shelf_command", required=True)

    build = shelf_sub.add_parser("build", help="build per-world rows from verified job bundles")
    build.add_argument("spec", help="shelf.toml (or the directory containing one)")
    build.add_argument(
        "--jobs",
        nargs="+",
        required=True,
        help="job directories or parents; must cover every declared real-data bench",
    )
    build.add_argument("--out", required=True, help="static output directory")
    build.set_defaults(func=_shelf_build)

    equivalence = shelf_sub.add_parser(
        "equivalence",
        help="work with §2.6 cross-world equivalence artifacts",
    )
    equivalence_sub = equivalence.add_subparsers(dest="equivalence_command", required=True)
    check = equivalence_sub.add_parser(
        "check",
        help="validate an equivalence artifact and list any failed requirement",
    )
    check.add_argument("artifact", help="equivalence artifact (.toml or .json)")
    check.add_argument(
        "--shelf",
        help="built shelf.json to check the artifact's gate map against; without it "
        "the shelf_gates requirement is reported as not run, never as passed",
    )
    check.set_defaults(func=_equivalence_check)

    rank = shelf_sub.add_parser("rank", help="print per-world orderings for a built shelf")
    rank.add_argument("path", help="shelf.json (or the directory containing one)")
    rank.add_argument(
        "--equivalence",
        help="validated equivalence artifact that unlocks one cross-world ordering",
    )
    rank.add_argument(
        "--cross-world",
        action="store_true",
        help="require a cross-world ordering; refused without a validated --equivalence",
    )
    rank.set_defaults(func=_shelf_rank)
