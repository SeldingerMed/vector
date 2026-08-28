"""Install planner and executor for the N10 ladder.

Two functions carry the whole surface: :func:`plan_install` turns a catalog
row into an explicit, ordered argv list, and :func:`execute_install` runs it.
The split exists so that the interesting half — *what exactly would run on
your machine* — is inspectable, testable, and printable without touching the
network. ``dry_run`` therefore defaults to ``True``: an install ladder whose
default is "execute" is not idiot-proof, it is a footgun.

Every refusal here names the missing artifact and the fix. In particular:

* an unpinned container image is refused, because an image without a digest is
  not an install, it is a moving target;
* a vendor runtime is never redistributed — we emit the *vendor's* pull and
  require explicit EULA acknowledgement, and we refuse to ask for that
  acknowledgement at all when the catalog has not recorded the terms;
* a world whose license we have not audited is refused, because directing a
  fetch under unknown terms is not something the harness will do on a user's
  behalf.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from or_audit.errors import TaskContractError
from or_audit.eval.licensing import LicenseStatus
from or_audit.install.catalog import (
    Disposition,
    FirstPartyInstall,
    InstallStrategy,
    PipExtraInstall,
    PrebuiltContainerInstall,
    SourceBuildInstall,
    VendorRuntimeInstall,
    WorldPackage,
)
from or_audit.version import PACKAGE_VERSION

#: A command runner: takes argv, returns a process exit code. Injectable so
#: tests (and ``--dry-run``) never shell out.
Runner = Callable[[Sequence[str]], int]

#: Container runtimes we know how to drive, in preference order.
CONTAINER_RUNTIMES = ("docker", "podman")


@dataclass(frozen=True)
class InstallStep:
    """One command, with the reason it exists."""

    argv: tuple[str, ...]
    purpose: str
    #: Whether the step reaches the network. The N10 contract is
    #: offline-after-fetch, so verification steps must be network-free.
    network: bool = True

    def render(self) -> str:
        return " ".join(self.argv)


@dataclass(frozen=True)
class InstallPlan:
    """The full, ordered command list for one world, plus what the user owns."""

    world_id: str
    strategy: InstallStrategy
    steps: tuple[InstallStep, ...]
    #: Things the *user* has accepted or must do; never executed by us.
    acknowledgements: tuple[str, ...] = field(default_factory=tuple)
    #: Work only a human can complete, with the upstream reference for each.
    #: A plan carrying these is never executable: see :func:`execute_install`.
    #: Emitting a command sequence that ends in a guaranteed-failing import is
    #: how a tool moves someone else's build failure into our exit code.
    manual: tuple[str, ...] = field(default_factory=tuple)

    @property
    def commands(self) -> tuple[tuple[str, ...], ...]:
        return tuple(step.argv for step in self.steps)

    @property
    def executable(self) -> bool:
        """Whether we can carry this plan to completion ourselves."""
        return not self.manual

    def render(self) -> str:
        """Exactly what would run, in order, one command per line."""
        lines = [f"install plan: {self.world_id} ({self.strategy.value})"]
        for note in self.acknowledgements:
            lines.append(f"  ack: {note}")
        for index, step in enumerate(self.steps, start=1):
            scope = "network" if step.network else "local"
            lines.append(f"  {index}. [{scope}] {step.purpose}")
            lines.append(f"     $ {step.render()}")
        if not self.steps:
            lines.append("  (nothing to run)")
        for note in self.manual:
            lines.append(f"  manual: {note}")
        if self.manual:
            lines.append("  this plan is not executable: the manual steps above are yours")
        return "\n".join(lines)


@dataclass(frozen=True)
class InstallOutcome:
    """What actually happened, or what a dry run would have done."""

    world_id: str
    dry_run: bool
    commands: tuple[tuple[str, ...], ...]
    #: Exit codes in command order; empty for a dry run. Short on first
    #: failure, because continuing past a failed pull installs nothing useful.
    exit_codes: tuple[int, ...] = field(default_factory=tuple)
    ok: bool = True


def default_pip_argv() -> tuple[str, ...]:
    """``uv pip install`` when uv is available, else this interpreter's pip."""
    if shutil.which("uv"):
        return ("uv", "pip", "install")
    return (sys.executable, "-m", "pip", "install")


def default_container_runtime() -> str:
    """First available container runtime, defaulting to ``docker`` by name.

    Defaulting to a name that may not exist is deliberate: the plan must be
    printable on a machine with no container runtime at all, and
    ``surgeval doctor`` is where a missing runtime gets reported with a fix.
    """
    for runtime in CONTAINER_RUNTIMES:
        if shutil.which(runtime):
            return runtime
    return CONTAINER_RUNTIMES[0]


def _disposition_fix(pkg: WorldPackage) -> str:
    """The honest remedy for an off-shelf row, which is not always promotion.

    A generic "promote it to 'wrap'" would be a false remedy for the two cases
    the 2026-08 audit actually produced: a row blocked by recorded terms cannot
    be promoted by doing more engineering, and a row with no runnable artifact
    has nothing to promote. Naming the wrong remedy is worse than naming none,
    because the user spends the afternoon before discovering it.
    """
    verdict = pkg.license_verdict
    if verdict.status is not LicenseStatus.ALLOWED and pkg.license_verified:
        return (
            f"Fix: nothing to install, and promotion is not an engineering task - the recorded "
            f"terms ({pkg.license}) are classified {verdict.status.value}: {verdict.reason}."
        )
    if pkg.disposition is Disposition.SKIP:
        return (
            "Fix: nothing to install and nothing to promote; this row is recorded as a coverage "
            "gap, not a wrap target. Closing it means a different world, or a first-party one."
        )
    return (
        "Fix: nothing to install; promote it to 'wrap' in catalog.toml first, which requires "
        "the license audit, the gate mapping, and the determinism measurement."
    )


def _refuse_disposition(pkg: WorldPackage) -> None:
    if pkg.disposition in {Disposition.SHIPPED, Disposition.WRAP}:
        return
    reason = pkg.risks.strip() or pkg.notes.strip() or "see catalog"
    raise TaskContractError(
        f"world {pkg.id!r} has disposition {pkg.disposition.value!r}: it is a survey row, "
        f"not a shelf item (Appendix B: {reason}). {_disposition_fix(pkg)}"
    )


def _refuse_license(pkg: WorldPackage) -> None:
    if not pkg.license_verified:
        raise TaskContractError(
            f"world {pkg.id!r} records license {pkg.license!r}: we will not direct a fetch under "
            "terms we have not read. Fix: audit the upstream LICENSE and record the SPDX "
            f"expression for {pkg.id!r} in catalog.toml (step one of every wrap, §2.4)."
        )
    verdict = pkg.license_verdict
    if verdict.status is LicenseStatus.ALLOWED:
        return
    raise TaskContractError(
        f"world {pkg.id!r} is licensed {pkg.license!r}, classified {verdict.status.value}: "
        f"{verdict.reason}. Reading the terms is what established this, so the audit is done "
        "and the answer is no. Fix: nothing to install; a wrap needs terms that permit "
        "redistribution, and this row should be disposition 'skip' with the license as the "
        "reason."
    )


def plan_refusal(pkg: WorldPackage) -> str | None:
    """Why ``plan_install`` would refuse this world, or ``None`` when it plans.

    Callers that *print a fix* consult this so the command they name is one that
    runs. ``surgeval doctor`` told users to run ``surgeval worlds install lumen``
    for a row whose license is unrecorded, and that command refuses - a fix that
    fails is worse than no fix, because it costs a round trip to discover.

    The verdict is taken by running the refusals rather than re-deriving their
    conditions, so this cannot drift away from what ``plan_install`` does.
    """
    try:
        _refuse_disposition(pkg)
        _refuse_license(pkg)
    except TaskContractError as exc:
        return str(exc)
    return None


def _source_build_steps(
    pkg: WorldPackage,
    spec: SourceBuildInstall,
) -> tuple[tuple[InstallStep, ...], tuple[str, ...]]:
    """Fetch the pinned source; hand the build back with the upstream reference.

    Returns ``(steps, manual)``. The split is the point. We can honestly do the
    clone and the checkout, so those are steps. We cannot do the build: every
    SOFA world needs a different plugin set compiled against a different engine
    revision, and ``sofa_env``'s own ``setup.py`` refuses any machine that is
    not ``x86_64`` running Python 3.10 - a constraint verified upstream, not
    guessed. Emitting a ``cmake`` line we have never run, or an ``import`` that
    is certain to fail because nothing has been built yet, would convert their
    build problem into our non-zero exit code.
    """
    if not spec.repo:
        raise TaskContractError(
            f"source-build world {pkg.id!r} records no repo. "
            "Fix: record the upstream repository in catalog.toml."
        )
    if not spec.pinned:
        raise TaskContractError(
            f"source-build world {pkg.id!r} has commit {spec.commit or '(empty)'!r}: a branch "
            "name is not a pin, and an unpinned source build reproduces nothing. Fix: record "
            f"the full 40-char commit SHA for {spec.repo} in catalog.toml."
        )
    if not spec.build_docs:
        raise TaskContractError(
            f"source-build world {pkg.id!r} records no build_docs, so the plan would end with "
            "a checkout and no way to build it. Fix: record the upstream build instructions "
            "URL in catalog.toml."
        )
    steps = (
        InstallStep(
            argv=("git", "clone", "--filter=blob:none", spec.repo, f"world-{pkg.id}"),
            purpose=f"fetch {pkg.display_name} source",
        ),
        InstallStep(
            argv=("git", "-C", f"world-{pkg.id}", "checkout", spec.commit),
            purpose=f"check out the pinned revision {spec.commit[:12]}",
            network=False,
        ),
    )
    manual = [
        f"build the runtime yourself following {spec.build_docs}"
        + (f" (requires {', '.join(spec.requires)})" if spec.requires else "")
    ]
    if spec.runtime_licenses:
        manual.append(
            f"that runtime closure is {', '.join(spec.runtime_licenses)}, which is why Vector "
            "publishes no image for this world"
        )
    if spec.verify_import:
        manual.append(
            f"once built, `python -c 'import {spec.verify_import}'` is the check that it worked"
        )
    return steps, tuple(manual)


def _first_party_steps(pkg: WorldPackage, spec: FirstPartyInstall) -> tuple[InstallStep, ...]:
    if not spec.verify_import:
        raise TaskContractError(
            f"first-party world {pkg.id!r} declares no verify_import, so an install cannot be "
            "verified. Fix: record the module the world imports as in catalog.toml."
        )
    return (
        InstallStep(
            argv=(sys.executable, "-c", f"import {spec.verify_import}"),
            purpose=(
                f"verify the first-party world imports "
                f"({spec.distribution or 'shipped with the harness'})"
            ),
            network=False,
        ),
    )


def _pip_extra_steps(
    pkg: WorldPackage,
    spec: PipExtraInstall,
    *,
    pip_argv: Sequence[str],
) -> tuple[InstallStep, ...]:
    if not spec.packages and not spec.extras:
        raise TaskContractError(
            f"pip-extra world {pkg.id!r} declares neither extras nor packages. "
            "Fix: record the extra and the pinned upstream packages in catalog.toml."
        )
    if spec.unpinned:
        missing = ", ".join(spec.unpinned)
        raise TaskContractError(
            f"pip-extra world {pkg.id!r} has unpinned package(s): {missing}. An unpinned "
            "install is not replayable, which is the whole point of the catalog. "
            "Fix: record the resolved version for each package in catalog.toml."
        )
    if not spec.verify_import:
        raise TaskContractError(
            f"pip-extra world {pkg.id!r} declares no verify_import, so a successful pip run "
            "would prove nothing. Fix: record the module the world imports as in catalog.toml."
        )
    targets: list[str] = []
    if spec.extras:
        targets.append(f"surgeval[{','.join(spec.extras)}]=={PACKAGE_VERSION}")
    targets.extend(spec.specs)
    return (
        InstallStep(
            argv=(*pip_argv, *targets),
            purpose=f"install {pkg.display_name} at its recorded pins",
        ),
        InstallStep(
            argv=(sys.executable, "-c", f"import {spec.verify_import}"),
            purpose="verify the world imports in this interpreter",
            network=False,
        ),
    )


def _prebuilt_container_steps(
    pkg: WorldPackage,
    spec: PrebuiltContainerInstall,
    *,
    runtime: str,
) -> tuple[InstallStep, ...]:
    if not spec.image:
        raise TaskContractError(
            f"prebuilt-container world {pkg.id!r} records no image. "
            "Fix: publish the image and record its reference in catalog.toml."
        )
    if not spec.pinned:
        raise TaskContractError(
            f"prebuilt-container world {pkg.id!r} has image_digest "
            f"{spec.image_digest or '(empty)'!r}: an unpinned image is not an install, it is a "
            f"moving target. Fix: publish {spec.image} and record its sha256:<64 hex> digest "
            "in catalog.toml."
        )
    return (
        InstallStep(
            argv=(runtime, "pull", spec.reference),
            purpose=f"pull the digest-pinned {pkg.display_name} image",
        ),
        InstallStep(
            argv=(runtime, "image", "inspect", spec.reference),
            purpose="verify the pinned image is present locally (offline after fetch)",
            network=False,
        ),
    )


def _vendor_runtime_steps(
    pkg: WorldPackage,
    spec: VendorRuntimeInstall,
    *,
    runtime: str,
    accept_vendor_eula: bool,
) -> tuple[InstallStep, ...]:
    vendor = spec.vendor or "the vendor"
    if not spec.container:
        raise TaskContractError(
            f"vendor-runtime world {pkg.id!r} records no container reference. "
            f"Fix: record {vendor}'s own container reference in catalog.toml."
        )
    if not spec.eula_url:
        raise TaskContractError(
            f"vendor-runtime world {pkg.id!r} records no eula_url, and we will not ask you to "
            f"accept terms we have not shown you. Fix: record {vendor}'s EULA URL for "
            f"{pkg.id!r} in catalog.toml."
        )
    if not accept_vendor_eula:
        raise TaskContractError(
            f"world {pkg.id!r} runs on {vendor}'s runtime, which we must not redistribute. "
            f"Read {spec.eula_url} and re-run with --accept-vendor-eula to have the installer "
            f"drive {vendor}'s own container pull."
        )
    if not spec.pinned_version:
        raise TaskContractError(
            f"vendor-runtime world {pkg.id!r} has no pinned_version, so the pull cannot be "
            f"verified. Fix: record the pinned {vendor} runtime version in catalog.toml."
        )
    if not spec.pinned:
        raise TaskContractError(
            f"vendor-runtime world {pkg.id!r} pins version {spec.pinned_version!r} but its "
            f"container reference {spec.container!r} does not name it, so the pull would fetch "
            "an unverifiable runtime. Fix: reference the pinned tag in catalog.toml."
        )
    steps: list[InstallStep] = []
    if spec.login_registry:
        steps.append(
            InstallStep(
                argv=(runtime, "login", spec.login_registry),
                purpose=f"authenticate to {spec.login_registry} with your own {vendor} account",
            )
        )
    steps.append(
        InstallStep(
            argv=(runtime, "pull", spec.container),
            purpose=f"pull {vendor}'s runtime {spec.pinned_version} (their image, their terms)",
        )
    )
    steps.append(
        InstallStep(
            argv=(runtime, "image", "inspect", spec.container),
            purpose=f"verify the pinned {vendor} runtime {spec.pinned_version} is present",
            network=False,
        )
    )
    return tuple(steps)


def plan_install(
    pkg: WorldPackage,
    *,
    accept_vendor_eula: bool = False,
    pip_argv: Sequence[str] | None = None,
    container_runtime: str | None = None,
) -> InstallPlan:
    """Explicit command list for installing one world, or a refusal.

    ``pip_argv`` and ``container_runtime`` are injectable so a plan is
    reproducible off this machine: printing a plan must not depend on whether
    the printing machine happens to have ``uv`` or ``podman``.
    """
    _refuse_disposition(pkg)
    _refuse_license(pkg)
    spec = pkg.install
    runtime = container_runtime or default_container_runtime()
    acknowledgements: tuple[str, ...] = ()
    manual: tuple[str, ...] = ()
    if isinstance(spec, FirstPartyInstall):
        steps = _first_party_steps(pkg, spec)
    elif isinstance(spec, PipExtraInstall):
        steps = _pip_extra_steps(pkg, spec, pip_argv=pip_argv or default_pip_argv())
    elif isinstance(spec, PrebuiltContainerInstall):
        steps = _prebuilt_container_steps(pkg, spec, runtime=runtime)
    elif isinstance(spec, SourceBuildInstall):
        steps, manual = _source_build_steps(pkg, spec)
    else:
        steps = _vendor_runtime_steps(
            pkg,
            spec,
            runtime=runtime,
            accept_vendor_eula=accept_vendor_eula,
        )
        acknowledgements = (
            f"you accepted {spec.vendor or 'the vendor'}'s EULA at {spec.eula_url}; "
            "Vector redistributes none of their runtime",
        )
    return InstallPlan(
        world_id=pkg.id,
        strategy=pkg.strategy,
        steps=steps,
        acknowledgements=acknowledgements,
        manual=manual,
    )


def _subprocess_runner(argv: Sequence[str]) -> int:
    """Default runner. Argv is built from catalog data, never from user strings."""
    return subprocess.call(list(argv))


def execute_install(
    plan: InstallPlan,
    *,
    dry_run: bool = True,
    runner: Runner | None = None,
) -> InstallOutcome:
    """Run a plan, or record it without running anything.

    ``dry_run=True`` is the default on purpose: the caller must ask for
    execution explicitly. A dry run touches nothing - not even the injected
    runner - so its outcome is a faithful transcript of the plan.

    A plan carrying manual work is refused rather than half-run. Cloning a
    source-build world and then reporting a failure the user was always going
    to hit would blame our exit code for their un-built runtime; printing the
    plan and stopping keeps the boundary where it belongs.
    """
    commands = plan.commands
    if dry_run:
        return InstallOutcome(world_id=plan.world_id, dry_run=True, commands=commands)
    if not plan.executable:
        raise TaskContractError(
            f"world {plan.world_id!r} installs by {plan.strategy.value}, which we cannot carry "
            "to completion: "
            + "; ".join(plan.manual)
            + ". Fix: run the plan yourself with --dry-run to get the pinned commands, then "
            "follow the upstream build reference."
        )
    run = runner or _subprocess_runner
    codes: list[int] = []
    for argv in commands:
        code = run(argv)
        codes.append(code)
        if code != 0:
            break
    return InstallOutcome(
        world_id=plan.world_id,
        dry_run=False,
        commands=commands,
        exit_codes=tuple(codes),
        ok=len(codes) == len(commands) and all(code == 0 for code in codes),
    )
