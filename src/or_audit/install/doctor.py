"""``surgeval doctor``: per-check diagnosis that prints the fix, not a trace.

N10 item 3. The design constraint is a single sentence: a user on a fresh
machine must be able to act on the output without reading Python. Two
structural consequences:

* Every failing check carries a non-empty ``fix``, enforced in the check
  constructor. A failure without a remedy is a stack trace with better
  formatting, which is the thing this command exists to replace.
* An unprobeable condition reports ``unknown``, never ``ok`` and never
  ``fail``. We cannot prove a GPU is absent from inside a shell with no
  ``nvidia-smi``, so we say so instead of guessing in either direction. An
  ``unknown`` on a *required* check still fails the command: the label stays
  honest about what we learned, and the exit code stays honest about the fact
  that we did not learn it. An unprobed requirement is not a satisfied one.

Optional worlds are advisory by default: a bare CPU machine with no SOFA and
no Isaac is a *healthy* machine for the quickstart path, so scanning the whole
catalog never fails the exit code. Asking about a specific world
(``--world <id>``) makes that world's checks required, because then the user
has asserted they want it working.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import or_audit
from or_audit.errors import TaskContractError
from or_audit.eval.sim import AdapterDiscovery, list_world_kinds, world_adapter_discovery
from or_audit.install.catalog import (
    Disposition,
    FirstPartyInstall,
    PipExtraInstall,
    PrebuiltContainerInstall,
    SourceBuildInstall,
    VendorRuntimeInstall,
    WorldCatalog,
    WorldPackage,
    load_catalog,
)
from or_audit.install.installer import CONTAINER_RUNTIMES, plan_refusal
from or_audit.version import PACKAGE_VERSION

#: Minimum interpreter the distribution declares in ``pyproject.toml``.
MIN_PYTHON = (3, 11)

#: The CPU-only reference pair ``surgeval quickstart`` runs, and the same pair
#: the doctor smoke-checks. One definition so the two surfaces cannot drift.
#:
#: Two layouts, because the pair lives in two places: a checkout keeps it under
#: ``docs/examples`` (where it is also documentation), and the wheel
#: force-includes it under ``or_audit/_examples`` so that
#: ``uv tool install surgeval && surgeval quickstart`` needs no clone.
PACKAGED_EXAMPLES_DIRNAME = "_examples"
PACKAGED_TASK_RELPATH = Path("tasks/video-nextstep")
PACKAGED_AGENT_RELPATH = Path("agents/example-video-predictor")
REFERENCE_TASK_RELPATH = Path("docs/examples/tasks/video-nextstep")
REFERENCE_AGENT_RELPATH = Path("docs/examples/agents/example-video-predictor")

_MISSING_EXAMPLES_FIX = (
    "neither the installed distribution nor the working directory carries the CPU-only "
    "reference packages. Fix: re-install a wheel built from this project (the reference pair "
    "ships under or_audit/_examples), run from a checkout "
    "(git clone https://github.com/SeldingerMed/vector && cd vector), or pass explicit "
    "--task/--agent package paths."
)


class CheckStatus(StrEnum):
    """Outcome of one diagnostic."""

    OK = "ok"
    FAIL = "fail"
    #: Genuinely unprobeable here (no GPU tooling, no container runtime).
    UNKNOWN = "unknown"
    #: Not installed and not asked for; nothing to diagnose.
    SKIPPED = "skipped"


@dataclass(frozen=True)
class DoctorCheck:
    """One diagnostic and, when it fails, what to do about it."""

    id: str
    status: CheckStatus
    detail: str
    fix: str = ""
    #: Whether a failure of this check fails the command.
    required: bool = True

    def __post_init__(self) -> None:
        if self.status in {CheckStatus.FAIL, CheckStatus.UNKNOWN} and not self.fix.strip():
            # Structural, not defensive: the promise of this command is that
            # every non-green line is actionable. A check that cannot say what
            # to do has no business reporting a problem.
            raise TaskContractError(
                f"doctor check {self.id!r} reports {self.status.value!r} without a fix"
            )

    @property
    def blocking(self) -> bool:
        """Whether this check fails the command.

        ``unknown`` blocks a required check. The caller named this world, so
        "we could not probe it" is not a pass: reporting success for a check
        that verified nothing is the failure mode this command exists to
        prevent. Advisory unknowns stay advisory, so a bare CPU machine
        scanning the whole shelf still exits 0.
        """
        return self.required and self.status in {CheckStatus.FAIL, CheckStatus.UNKNOWN}

    def render(self) -> str:
        line = f"[{self.status.value:>7}] {self.id}: {self.detail}"
        if self.fix and self.status is not CheckStatus.OK:
            line += f"\n          fix: {self.fix}"
        return line

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "detail": self.detail,
            "fix": self.fix,
            "required": self.required,
        }


@dataclass(frozen=True)
class DoctorReport:
    """Every check, in run order."""

    checks: tuple[DoctorCheck, ...] = field(default_factory=tuple)

    @property
    def failures(self) -> tuple[DoctorCheck, ...]:
        """Required checks that failed or went unproven — they set the exit code."""
        return tuple(check for check in self.checks if check.blocking)

    @property
    def advisories(self) -> tuple[DoctorCheck, ...]:
        """Non-blocking problems: optional worlds and unprobeable conditions."""
        return tuple(
            check
            for check in self.checks
            if not check.blocking and check.status in {CheckStatus.FAIL, CheckStatus.UNKNOWN}
        )

    @property
    def ok(self) -> bool:
        return not self.failures

    def exit_code(self) -> int:
        """0 when every required check was proven, 1 otherwise."""
        return 0 if self.ok else 1

    def render(self) -> str:
        lines = [check.render() for check in self.checks]
        if self.ok:
            lines.append(f"doctor: ok ({len(self.advisories)} advisory)")
        else:
            lines.append(f"doctor: {len(self.failures)} required check(s) not satisfied")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
            "failures": [check.id for check in self.failures],
            "advisories": [check.id for check in self.advisories],
        }


def _candidate_pairs() -> tuple[tuple[Path, Path], ...]:
    """Where the reference pair might live, most authoritative first.

    The wheel's own copy wins over a checkout: it is content-pinned with the
    installed distribution, so a quickstart run from inside some unrelated
    clone still measures the packages the user actually installed.
    """
    package_dir = Path(or_audit.__file__).resolve().parent
    packaged = package_dir / PACKAGED_EXAMPLES_DIRNAME
    candidates = [(packaged / PACKAGED_TASK_RELPATH, packaged / PACKAGED_AGENT_RELPATH)]
    # src-layout checkout, install root, and the caller's working directory.
    for root in (package_dir.parents[1], package_dir.parent, Path.cwd().resolve()):
        candidates.append((root / REFERENCE_TASK_RELPATH, root / REFERENCE_AGENT_RELPATH))
    return tuple(candidates)


def find_reference_paths() -> tuple[Path, Path] | None:
    """The CPU-only reference task/agent pair, or ``None`` when not shipped."""
    for task, agent in _candidate_pairs():
        if task.is_dir() and agent.is_dir():
            return task, agent
    return None


def require_reference_paths() -> tuple[Path, Path]:
    """The reference pair, or refuse with the actionable install fix."""
    found = find_reference_paths()
    if found is None:
        searched = ", ".join(str(task) for task, _ in _candidate_pairs())
        raise TaskContractError(f"{_MISSING_EXAMPLES_FIX} Searched: {searched}")
    return found


def _module_present(module: str) -> bool:
    """Whether ``module`` is importable, without importing the world itself."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


#: Seconds one local image inspection may take. A runtime that cannot answer a
#: metadata question in this long has not shown the image is there, and the
#: doctor's job is to report that rather than to wait for it.
IMAGE_INSPECT_TIMEOUT_S = 20.0

#: A runtime that cannot reach its own daemon exits nonzero exactly as it does
#: for a genuinely absent image. Reading absence out of that would be inventing
#: evidence, so these phrases downgrade the answer to "unprobeable".
_RUNTIME_UNREACHABLE = re.compile(
    r"cannot connect|connection refused|permission denied|is the docker daemon running"
    r"|daemon is not running|no such file or directory",
    re.IGNORECASE,
)


def _available_container_runtime() -> str | None:
    """Path of the first container runtime on PATH, or ``None`` when there is none.

    Unlike :func:`or_audit.install.installer.default_container_runtime`, which
    names ``docker`` so a plan is printable anywhere, this refuses to name a
    runtime it did not find: the doctor is asking whether one exists.
    """
    for runtime in CONTAINER_RUNTIMES:
        found = shutil.which(runtime)
        if found is not None:
            return found
    return None


def _image_present(runtime: str, reference: str) -> bool | None:
    """Whether ``runtime`` already holds ``reference`` locally, or ``None``.

    ``image inspect`` is a local metadata read, so this stays inside the
    offline-after-fetch contract: the doctor diagnoses, it never pulls. Argv is
    built from catalog data, never from user strings.

    ``None`` means the question could not be asked at all - no daemon, no
    permission, exec failure, timeout - which is a different claim from the
    runtime answering "no such image", and the two get different statuses.
    """
    argv = [runtime, "image", "inspect", reference]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=IMAGE_INSPECT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode == 0:
        return True
    if _RUNTIME_UNREACHABLE.search(completed.stderr or ""):
        return None
    return False


def _python_check() -> DoctorCheck:
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info[:2] >= MIN_PYTHON:
        return DoctorCheck("python", CheckStatus.OK, f"{version} at {sys.executable}")
    wanted = ".".join(str(part) for part in MIN_PYTHON)
    return DoctorCheck(
        "python",
        CheckStatus.FAIL,
        f"{version} is older than {wanted}",
        fix=f"install Python {wanted}+ and re-install surgeval into it (uv tool install surgeval)",
    )


def _package_check() -> DoctorCheck:
    return DoctorCheck(
        "surgeval",
        CheckStatus.OK,
        f"{PACKAGE_VERSION} from {Path(or_audit.__file__).resolve().parent}",
    )


def _world_kinds_check() -> DoctorCheck:
    kinds = list_world_kinds()
    if not kinds:
        return DoctorCheck(
            "world-kinds",
            CheckStatus.FAIL,
            "no world kinds are registered",
            fix=(
                "the built-in registry failed to initialise; re-install surgeval "
                "(uv tool install --reinstall surgeval) and report the failure if it persists"
            ),
        )
    attached = sum(1 for spec in kinds.values() if spec.adapter_identity != "unattached")
    return DoctorCheck(
        "world-kinds",
        CheckStatus.OK,
        f"{len(kinds)} registered ({attached} with a digest-pinned adapter): "
        + ", ".join(sorted(kinds)),
    )


def _discovery_checks(discovery: Sequence[AdapterDiscovery]) -> tuple[DoctorCheck, ...]:
    """One check per world-adapter plugin, so a bad plugin names itself.

    Discovery failures are recorded rather than raised by the kernel, which
    means nothing surfaces them unless a command asks. This is that command.
    """
    checks: list[DoctorCheck] = []
    failed = [item for item in discovery if not item.ok]
    for item in failed:
        checks.append(
            DoctorCheck(
                f"world-adapter:{item.name}",
                CheckStatus.FAIL,
                f"entry point failed to load: {item.error}",
                fix=(
                    f"the distribution publishing the {item.name!r} world adapter is broken or "
                    "collides with a registered kind; re-install or uninstall that distribution "
                    "(the rest of the harness keeps working without it)"
                ),
            )
        )
    ok_names = [item.kind or item.name for item in discovery if item.ok]
    checks.append(
        DoctorCheck(
            "world-adapters",
            CheckStatus.OK,
            f"{len(ok_names)} plugin adapter(s) loaded"
            + (f": {', '.join(sorted(ok_names))}" if ok_names else "")
            + (f"; {len(failed)} failed (see above)" if failed else ""),
        )
    )
    return tuple(checks)


def _reference_check() -> DoctorCheck:
    found = find_reference_paths()
    if found is None:
        return DoctorCheck(
            "reference-task",
            CheckStatus.FAIL,
            "the CPU-only reference task/agent pair is not on disk",
            fix=_MISSING_EXAMPLES_FIX,
        )
    task_dir, agent_dir = found
    from or_audit.eval.loader import load_agent, load_task

    try:
        task = load_task(task_dir)
        agent = load_agent(agent_dir)
    except TaskContractError as exc:
        return DoctorCheck(
            "reference-task",
            CheckStatus.FAIL,
            f"{task_dir} does not load: {exc}",
            fix=(
                "the reference packages on disk have been edited or truncated; restore them "
                "from a clean checkout (git checkout -- docs/examples)"
            ),
        )
    return DoctorCheck(
        "reference-task",
        CheckStatus.OK,
        f"{task.id}@{task.task_version} + {agent.id}@{agent.agent_version} load; "
        f"run `surgeval quickstart` for a vector",
    )


def _install_fix(pkg: WorldPackage, *, dry_run: bool = False) -> str:
    """A fix a user can actually run: the install command, or what blocks it.

    Naming ``surgeval worlds install <id>`` for a row that command refuses -
    unrecorded license, denied terms, or an off-shelf disposition - sends the
    user to a dead end. The catalog knows which case it is, so the fix says it.
    """
    refusal = plan_refusal(pkg)
    if refusal is not None:
        return refusal
    suffix = " --dry-run for the pinned source and build docs" if dry_run else ""
    return f"surgeval worlds install {pkg.id}{suffix}"


def _image_probe_check(
    pkg: WorldPackage,
    *,
    check_id: str,
    runtime: str,
    reference: str,
    what: str,
    required: bool,
) -> DoctorCheck:
    """Grade a container world on whether ``reference`` is actually present.

    Tool presence is not image presence. This check used to report ``ok`` as
    soon as ``docker`` was on PATH, which claimed an install it had never
    looked for; the runtime is now asked, offline, about the exact tag or
    digest the catalog names.
    """
    present = _image_present(runtime, reference)
    runtime_name = Path(runtime).name
    if present:
        return DoctorCheck(
            check_id,
            CheckStatus.OK,
            f"{runtime_name} has {what} {reference}",
            required=required,
        )
    if present is None:
        return DoctorCheck(
            check_id,
            CheckStatus.UNKNOWN,
            f"{runtime_name} could not be asked whether {what} {reference} is present",
            fix=(
                f"make {runtime_name} usable (`{runtime_name} info` must succeed), then re-run "
                f"`surgeval doctor --world {pkg.id}`"
            ),
            required=required,
        )
    if required:
        return DoctorCheck(
            check_id,
            CheckStatus.FAIL,
            f"{runtime_name} does not have {what} {reference}",
            fix=_install_fix(pkg),
        )
    return DoctorCheck(
        check_id,
        CheckStatus.SKIPPED,
        f"{what} not pulled (optional; run `surgeval doctor --world {pkg.id}` for the fix)",
        required=False,
    )


def _world_check(pkg: WorldPackage, *, required: bool) -> DoctorCheck:
    """One best-effort probe per world, chosen by install strategy."""
    check_id = f"world:{pkg.id}"
    spec = pkg.install
    if isinstance(spec, FirstPartyInstall | PipExtraInstall):
        module = spec.verify_import
        if not module:
            return DoctorCheck(
                check_id,
                CheckStatus.SKIPPED,
                f"{pkg.strategy.value} world declares no verify_import",
                required=required,
            )
        if _module_present(module):
            return DoctorCheck(
                check_id,
                CheckStatus.OK,
                f"{module} is importable ({pkg.engine})",
                required=required,
            )
        if required:
            return DoctorCheck(
                check_id,
                CheckStatus.FAIL,
                f"{module} is not importable",
                fix=_install_fix(pkg),
            )
        return DoctorCheck(
            check_id,
            CheckStatus.SKIPPED,
            f"not installed (optional; run `surgeval doctor --world {pkg.id}` for the fix)",
            required=False,
        )
    if isinstance(spec, SourceBuildInstall):
        module = spec.verify_import
        if not module:
            return DoctorCheck(
                check_id,
                CheckStatus.SKIPPED,
                f"source-build world declares no verify_import ({pkg.engine})",
                required=required,
            )
        if _module_present(module):
            return DoctorCheck(
                check_id,
                CheckStatus.OK,
                f"{module} is importable, so you have built {pkg.engine} yourself",
                required=required,
            )
        # Never a failure, even when named: the catalog explicitly hands this
        # build to the user, so reporting it as broken would blame the machine
        # for work we never claimed to do.
        return DoctorCheck(
            check_id,
            CheckStatus.SKIPPED,
            f"{module} not importable; {pkg.engine} is a user-built runtime "
            f"({', '.join(spec.requires) or 'see build docs'})",
            fix=_install_fix(pkg, dry_run=True),
            required=False,
        )
    if isinstance(spec, PrebuiltContainerInstall):
        runtime = _available_container_runtime()
        if runtime is None:
            return DoctorCheck(
                check_id,
                CheckStatus.UNKNOWN,
                f"no container runtime found, so the {pkg.engine} image cannot be checked",
                fix=(
                    f"install one of {', '.join(CONTAINER_RUNTIMES)} and re-run "
                    f"`surgeval doctor --world {pkg.id}`"
                ),
                required=required,
            )
        if not spec.pinned:
            return DoctorCheck(
                check_id,
                CheckStatus.UNKNOWN,
                f"container runtime present ({runtime}) but the catalog has no image digest",
                fix=(
                    f"nothing to check yet: the {pkg.id} image is unpublished. Track its digest "
                    "landing in catalog.toml"
                ),
                required=required,
            )
        return _image_probe_check(
            pkg,
            check_id=check_id,
            runtime=runtime,
            reference=spec.reference,
            what="pinned image",
            required=required,
        )
    if isinstance(spec, VendorRuntimeInstall):
        vendor = spec.vendor or "vendor"
        if shutil.which("nvidia-smi") is None:
            return DoctorCheck(
                check_id,
                CheckStatus.UNKNOWN,
                f"no GPU driver tooling on PATH, so the {vendor} runtime cannot be probed",
                fix=(
                    f"{pkg.engine} needs a {vendor} GPU with drivers and container toolkit; "
                    f"on a GPU host re-run `surgeval doctor --world {pkg.id}`"
                ),
                required=required,
            )
        # A vendor-runtime world *is* the vendor's container, so a GPU driver
        # that answered is a prerequisite met, not the world found. Without a
        # container runtime there is nothing here that could hold the image.
        runtime = _available_container_runtime()
        if runtime is None:
            return DoctorCheck(
                check_id,
                CheckStatus.UNKNOWN,
                f"GPU driver tooling present, but no container runtime, so the {vendor} "
                f"runtime image cannot be checked",
                fix=(
                    f"install one of {', '.join(CONTAINER_RUNTIMES)} (plus the {vendor} "
                    f"container toolkit) and re-run `surgeval doctor --world {pkg.id}`"
                ),
                required=required,
            )
        if not spec.pinned:
            return DoctorCheck(
                check_id,
                CheckStatus.UNKNOWN,
                f"container runtime present ({runtime}) but the catalog names no pinned "
                f"{vendor} runtime version, so no image reference can be verified",
                fix=(
                    f"record the pinned {vendor} runtime version, named in its container "
                    f"reference, for {pkg.id} in catalog.toml"
                ),
                required=required,
            )
        return _image_probe_check(
            pkg,
            check_id=check_id,
            runtime=runtime,
            reference=spec.container,
            what=f"{vendor} runtime {spec.pinned_version} image",
            required=required,
        )
    raise TaskContractError(f"unhandled install strategy for world {pkg.id!r}")


def _requested_world_checks(
    catalog: WorldCatalog,
    requested: Sequence[str],
) -> tuple[DoctorCheck, ...]:
    checks: list[DoctorCheck] = []
    for world_id in requested:
        pkg = catalog.get(world_id)
        if pkg is None:
            checks.append(
                DoctorCheck(
                    f"world:{world_id}",
                    CheckStatus.FAIL,
                    f"unknown world package {world_id!r}",
                    fix=f"run `surgeval worlds list`; the catalog has: {', '.join(catalog.ids)}",
                )
            )
            continue
        checks.append(_world_check(pkg, required=True))
    return tuple(checks)


def run_doctor(
    *,
    packages: Sequence[str] | None = None,
    catalog: WorldCatalog | None = None,
    discovery: Sequence[AdapterDiscovery] | None = None,
) -> DoctorReport:
    """Diagnose this machine.

    ``packages`` names worlds the user asserted they want working; those
    checks are required. With ``packages=None`` the whole shelf is probed
    advisorily, so a healthy CPU-only machine still exits 0.

    ``discovery`` is injectable: adapter-discovery failures are recorded once
    at import time, and a caller (or a test) needs to be able to hand in a
    specific report rather than mutate global plugin state.
    """
    resolved = catalog or load_catalog()
    report = discovery if discovery is not None else world_adapter_discovery()
    checks: list[DoctorCheck] = [_python_check(), _package_check(), _world_kinds_check()]
    checks.extend(_discovery_checks(report))
    checks.append(_reference_check())
    if packages is None:
        for pkg in resolved.worlds:
            if pkg.disposition in {Disposition.SHIPPED, Disposition.WRAP}:
                checks.append(_world_check(pkg, required=False))
    else:
        checks.extend(_requested_world_checks(resolved, packages))
    return DoctorReport(checks=tuple(checks))
