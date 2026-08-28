"""``surgeval worlds`` — browse the catalog and plan a world install.

Three subcommands, one job each: ``list`` shows the shelf and, crucially, its
*pin state* per row; ``info`` shows one row in full including the sources the
disposition was decided from; ``install`` prints the exact commands it would
run.

``install`` defaults to a dry run. That is not timidity: the interesting half
of an installer is what it would do to your machine, and a ladder whose
default is "execute" cannot be inspected before it runs.
"""

from __future__ import annotations

import argparse
import sys

from or_audit.errors import TaskContractError
from or_audit.eval.sim import world_kind_spec
from or_audit.install.catalog import (
    Disposition,
    InstallStrategy,
    WorldPackage,
    iter_packages,
    world_package,
)
from or_audit.install.installer import execute_install, plan_install

_COLUMNS = (
    "id",
    "domain",
    "engine",
    "strategy",
    "disposition",
    "license",
    "pin",
    "gates",
)


def _row(pkg: WorldPackage) -> tuple[str, ...]:
    return (
        pkg.id,
        pkg.domain,
        pkg.engine,
        pkg.strategy.value,
        pkg.disposition.value,
        pkg.license,
        pkg.pin_state,
        "metrics-only" if pkg.metrics_only else "safety",
    )


def _render_table(rows: list[tuple[str, ...]]) -> str:
    widths = [max(len(row[i]) for row in [_COLUMNS, *rows]) for i in range(len(_COLUMNS))]
    lines = []
    for row in [_COLUMNS, *rows]:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    return "\n".join(lines)


def _worlds_list(args: argparse.Namespace) -> int:
    """Print the catalog, including the WATCH/SKIP rows and their pin state."""
    try:
        packages = iter_packages(
            disposition=Disposition(args.disposition) if args.disposition else None,
            strategy=InstallStrategy(args.strategy) if args.strategy else None,
        )
    except TaskContractError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    if not packages:
        print("no catalog entries match that filter")
        return 0
    print(_render_table([_row(pkg) for pkg in packages]))
    installable = sum(1 for pkg in packages if pkg.installable)
    print(
        f"\n{len(packages)} world(s); {installable} with terms that permit a wrap and a pinned "
        "artifact. That is a check on this catalog's data, not on your machine: run `surgeval "
        "doctor` for whether this host can run them, and `surgeval worlds install <id>` for the "
        "exact commands or the specific missing artifact. A `gates` column reading "
        "`metrics-only` means that world exposes no physically-grounded safety signal, so tasks "
        "on it may not declare hard gates (§2.2)."
    )
    return 0


def _worlds_info(args: argparse.Namespace) -> int:
    """Print one catalog row in full, plus whether its kernel kind is live."""
    try:
        pkg = world_package(args.id)
    except TaskContractError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(pkg.describe())
    if pkg.world_kind:
        spec = world_kind_spec(pkg.world_kind)
        if spec is None:
            print(f"  registry    world kind {pkg.world_kind!r} is not registered in this install")
        else:
            provider = spec.provider or "-"
            print(f"  registry    adapter {spec.adapter_identity} (provider {provider})")
    print(f"  installable {'yes' if pkg.installable else 'no'}")
    return 0


def _worlds_install(args: argparse.Namespace) -> int:
    """Plan (and optionally run) the install ladder for one world."""
    try:
        pkg = world_package(args.id)
        plan = plan_install(pkg, accept_vendor_eula=args.accept_vendor_eula)
    except TaskContractError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(plan.render())
    dry_run = not args.execute
    outcome = execute_install(plan, dry_run=dry_run)
    if outcome.dry_run:
        print(f"\ndry run: {len(outcome.commands)} command(s) not executed (--execute to run)")
        return 0
    if not outcome.ok:
        failed = outcome.commands[len(outcome.exit_codes) - 1]
        print(
            f"REFUSED: install of {pkg.id!r} failed at `{' '.join(failed)}` "
            f"(exit {outcome.exit_codes[-1]}); run `surgeval doctor --world {pkg.id}` for the fix",
            file=sys.stderr,
        )
        return 1
    print(f"\ninstalled: {pkg.id} ({plan.strategy.value})")
    return 0


def _worlds(args: argparse.Namespace) -> int:
    del args
    print("worlds requires a subcommand: list, info, or install", file=sys.stderr)
    return 2


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``worlds`` command group."""
    worlds = sub.add_parser("worlds", help="browse and install catalog worlds")
    worlds.set_defaults(func=_worlds)
    worlds_sub = worlds.add_subparsers(dest="worlds_command")

    listing = worlds_sub.add_parser("list", help="list catalog worlds and their pin state")
    listing.add_argument(
        "--disposition",
        choices=[item.value for item in Disposition],
        help="only rows with this Appendix B disposition",
    )
    listing.add_argument(
        "--strategy",
        choices=[item.value for item in InstallStrategy],
        help="only rows using this install strategy",
    )
    listing.set_defaults(func=_worlds_list)

    info = worlds_sub.add_parser("info", help="describe one catalog world")
    info.add_argument("id", help="catalog world id")
    info.set_defaults(func=_worlds_info)

    install = worlds_sub.add_parser("install", help="plan (or run) a world install")
    install.add_argument("id", help="catalog world id")
    install.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="print the commands without running them (default)",
    )
    install.add_argument(
        "--execute",
        action="store_true",
        help="actually run the planned commands",
    )
    install.add_argument(
        "--accept-vendor-eula",
        action="store_true",
        help="acknowledge the vendor's EULA for a vendor-runtime world (required for those)",
    )
    install.set_defaults(func=_worlds_install)
