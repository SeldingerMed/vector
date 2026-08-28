"""``surgeval concierge`` — intake, assess, select, and adapt from the CLI.

Every subcommand prints refusals first, on stderr, prefixed ``REFUSED: ``, and
exits 1 when anything was refused. The control plane never probes or loads a
model here: ``intake`` only hashes bytes, and ``assess`` consumes a probe report
that a sandboxed prober produced, so the CLI cannot become the thing the intake
gate exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from or_audit.concierge.adapt import freeze_adapted_package
from or_audit.concierge.assess import (
    ConfirmedCapability,
    ModelProbeReport,
    assess_model,
    confirm,
)
from or_audit.concierge.intake import (
    IntakeResult,
    SandboxPolicy,
    TenantKeyring,
    UploadManifest,
    intake_upload,
)
from or_audit.concierge.select import (
    EvalBudget,
    narrate_plan,
    select_eval_plan,
)
from or_audit.errors import TaskContractError
from or_audit.eval.loader import load_task


def _print_refusals(refusals: tuple[str, ...] | list[str]) -> None:
    for refusal in refusals:
        print(f"REFUSED: {refusal}", file=sys.stderr)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TaskContractError(f"{path} must contain one JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _catalog_paths(catalog: Path) -> list[Path]:
    """Every task package directly under a catalog directory."""
    if (catalog / "task.toml").is_file():
        return [catalog]
    return sorted(child for child in catalog.iterdir() if (child / "task.toml").is_file())


def _keyring(path: Path) -> TenantKeyring:
    """Load the control plane's signing keys. Never taken from a manifest."""
    payload = _read_json(path)
    records = payload.get("keys")
    if not isinstance(records, list):
        raise TaskContractError(
            f"{path} must hold a 'keys' list of "
            "{key_id, tenant, secret_hex} records owned by the control plane"
        )
    return TenantKeyring.from_records([record for record in records if isinstance(record, dict)])


def _intake(args: argparse.Namespace) -> int:
    """Hash and gate an uploaded artifact. Never deserializes it."""
    manifest_path = Path(args.manifest)
    try:
        manifest = UploadManifest.model_validate(_read_json(manifest_path))
        keyring = _keyring(Path(args.keyring))
        sandbox = SandboxPolicy(
            cpu_quota=args.cpu_quota,
            memory_bytes=args.memory_bytes,
            disk_bytes=args.disk_bytes,
            gpu=args.gpu,
        )
    except FileNotFoundError as exc:
        print(f"no such file: {exc}", file=sys.stderr)
        return 2
    except (TaskContractError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    result = intake_upload(manifest, Path(args.artifact), sandbox=sandbox, keyring=keyring)
    _print_refusals(result.refusals)
    print(result.describe())
    if args.out:
        _write_json(Path(args.out), result.model_dump(mode="json"))
        print(f"  written  {args.out}")
    return 1 if result.refusals else 0


def _assess(args: argparse.Namespace) -> int:
    """Draft a capability from a sandboxed probe report. Confirmation is separate."""
    try:
        intake = IntakeResult.model_validate(_read_json(Path(args.intake)))
        report = ModelProbeReport.model_validate(_read_json(Path(args.probe_report)))
    except FileNotFoundError as exc:
        print(f"no such file: {exc}", file=sys.stderr)
        return 2
    except (TaskContractError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    try:
        proposal = assess_model(intake, probe=lambda _: report)
        payload: dict[str, Any] = proposal.model_dump(mode="json")
        if args.confirmed_by:
            payload = confirm(proposal, confirmed_by=args.confirmed_by).model_dump(mode="json")
    except (TaskContractError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    _write_json(Path(args.out), payload)
    print(proposal.describe())
    if args.confirmed_by:
        print(f"  CONFIRMED by {args.confirmed_by}")
    else:
        print(
            "  proposal only: pass --confirmed-by to record the party accepting it "
            "before the first scored run"
        )
    print(f"  written  {args.out}")
    return 0


def _select(args: argparse.Namespace) -> int:
    """Rank a catalog for a confirmed capability inside a trial budget."""
    catalog = Path(args.catalog)
    if not catalog.is_dir():
        print(f"no such catalog directory: {catalog}", file=sys.stderr)
        return 2
    try:
        capability = ConfirmedCapability.model_validate(_read_json(Path(args.capability)))
        budget = EvalBudget(
            max_total_trials=args.budget,
            max_trials_per_task=args.per_task,
        )
        plan = select_eval_plan(
            capability,
            catalog_paths=_catalog_paths(catalog),
            budget=budget,
        )
    except FileNotFoundError as exc:
        print(f"no such file: {exc}", file=sys.stderr)
        return 2
    except (TaskContractError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    _print_refusals(plan.refusals)
    print(narrate_plan(plan))
    if args.out:
        _write_json(Path(args.out), plan.model_dump(mode="json"))
        print(f"\nwritten  {args.out}")
    return 1 if plan.refusals else 0


def _adapt(args: argparse.Namespace) -> int:
    """Freeze declared scenarios into a new, quarantined, digest-pinned package."""
    task_dir = Path(args.task)
    try:
        task = load_task(task_dir)
        selected = {scenario.id: scenario for scenario in task.scenarios}
        chosen_ids = tuple(args.scenario) if args.scenario else tuple(selected)
        unknown = [item for item in chosen_ids if item not in selected]
        if unknown:
            raise TaskContractError(f"task {task.id} declares no scenario(s) {sorted(unknown)}")
        scenarios = [selected[item] for item in chosen_ids]
        perturbations = [
            perturbation
            for perturbation in task.perturbations
            if not args.perturbation or perturbation.id in set(args.perturbation)
        ]
        frozen = freeze_adapted_package(
            task_dir,
            scenarios=scenarios,
            perturbations=perturbations,
            out=Path(args.out),
        )
    except (TaskContractError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"Frozen adapted package {frozen.task_id}@{frozen.task_version}")
    print(f"  path       {frozen.path}")
    print(f"  digest     {frozen.digest}")
    print(f"  parent     {frozen.parent_task_id}@{frozen.parent_task_version}")
    print(f"  parent dig {frozen.parent_digest}")
    print(f"  authored   {frozen.authored_by}")
    print(f"  scenarios  {', '.join(frozen.scenario_ids)}")
    print(f"  perturb.   {', '.join(frozen.perturbation_ids) or '(none)'}")
    print(
        "  QUARANTINED: not eligible for a public leaderboard until promoted "
        "through Tier-1 conformance. The verifier is byte-identical to the parent's."
    )
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``concierge`` command group to a subparser action."""
    concierge = sub.add_parser(
        "concierge",
        help="hosted agentic concierge: intake, assess, select, adapt",
    )
    concierge_sub = concierge.add_subparsers(dest="concierge_command", required=True)

    intake = concierge_sub.add_parser(
        "intake", help="gate an uploaded model artifact without loading it"
    )
    intake.add_argument("--manifest", required=True, help="signed upload manifest (JSON)")
    intake.add_argument("--artifact", required=True, help="uploaded artifact path")
    intake.add_argument(
        "--keyring",
        required=True,
        help=(
            "control-plane signing keys (JSON: {'keys': [{key_id, tenant, secret_hex}]}); "
            "required, because an unauthenticated manifest is only a claim"
        ),
    )
    intake.add_argument("--out", help="write the intake result to this JSON path")
    intake.add_argument("--cpu-quota", type=float, default=2.0, help="sandbox CPU cores")
    intake.add_argument("--memory-bytes", type=int, default=8 << 30, help="sandbox memory quota")
    intake.add_argument("--disk-bytes", type=int, default=16 << 30, help="sandbox disk quota")
    intake.add_argument("--gpu", default="", help="pinned sandbox device class")
    intake.set_defaults(func=_intake)

    assess = concierge_sub.add_parser(
        "assess", help="draft a capability from a sandboxed probe report"
    )
    assess.add_argument("--intake", required=True, help="intake result (JSON)")
    assess.add_argument(
        "--probe-report",
        required=True,
        help="probe report produced inside the sandbox (JSON)",
    )
    assess.add_argument("--out", required=True, help="write the proposal to this JSON path")
    assess.add_argument(
        "--confirmed-by",
        default="",
        help="record a named party confirming the proposal (otherwise it stays inert)",
    )
    assess.set_defaults(func=_assess)

    select = concierge_sub.add_parser(
        "select", help="rank a task catalog for a confirmed capability"
    )
    select.add_argument("--capability", required=True, help="confirmed capability (JSON)")
    select.add_argument("--catalog", required=True, help="directory of task packages")
    select.add_argument("--budget", type=int, required=True, help="total trial budget")
    select.add_argument("--per-task", type=int, default=30, help="per-task trial ceiling")
    select.add_argument("--out", help="write the plan to this JSON path")
    select.set_defaults(func=_select)

    adapt = concierge_sub.add_parser(
        "adapt", help="freeze declared scenarios into a new quarantined package"
    )
    adapt.add_argument("--task", required=True, help="parent task package directory")
    adapt.add_argument("--out", required=True, help="new package directory to write")
    adapt.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="declared scenario id to freeze (repeatable; default all)",
    )
    adapt.add_argument(
        "--perturbation",
        action="append",
        default=[],
        help="declared perturbation id to freeze (repeatable; default all)",
    )
    # No --authored-by: authorship is the class that carries the quarantine, and
    # a package that could call itself human-authored could leave it.
    adapt.set_defaults(func=_adapt)


__all__ = ["register"]
