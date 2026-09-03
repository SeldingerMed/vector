"""N10 install-and-launch ladder: catalog, installer, doctor, quickstart.

These tests defend the ladder's refusals and its one measured promise. The
refusals are the product: an unpinned image, an unaudited license, or an
unacknowledged vendor EULA must stop the install and say why, because the
alternative is a user who thinks they reproduced something they did not.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

import or_audit
from or_audit.commands import doctor as doctor_cmd
from or_audit.commands import quickstart as quickstart_cmd
from or_audit.commands import worlds as worlds_cmd
from or_audit.errors import TaskContractError
from or_audit.eval.job import read_job_result, verify_head
from or_audit.eval.sim import AdapterDiscovery
from or_audit.install import doctor as doctor_mod
from or_audit.install.catalog import (
    UNVERIFIED,
    Disposition,
    FirstPartyInstall,
    InstallStrategy,
    PinnedPackage,
    PipExtraInstall,
    PrebuiltContainerInstall,
    VendorRuntimeInstall,
    WorldCatalog,
    WorldPackage,
    iter_packages,
    load_catalog,
    world_package,
)
from or_audit.install.doctor import (
    CheckStatus,
    DoctorCheck,
    DoctorReport,
    find_reference_paths,
    run_doctor,
)
from or_audit.install.installer import (
    InstallOutcome,
    InstallPlan,
    execute_install,
    plan_install,
)

PINNED_DIGEST = "sha256:" + "a" * 64


class FakeRunner:
    """Records argv instead of executing it."""

    def __init__(self, code: int = 0) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.code = code

    def __call__(self, argv: Sequence[str]) -> int:
        self.calls.append(tuple(argv))
        return self.code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="surgeval")
    sub = parser.add_subparsers(dest="command")
    for module in (worlds_cmd, quickstart_cmd, doctor_cmd):
        module.register(sub)
    return parser


def _pip_world(**overrides: object) -> WorldPackage:
    """A pip-extra world with the audit and pins a completed wrap would have."""
    payload: dict[str, object] = {
        "id": "pinned-pip-world",
        "display_name": "Pinned Pip World",
        "domain": "Endovascular",
        "engine": "MuJoCo",
        "disposition": Disposition.WRAP,
        "license": "MIT",
        "world_kind": "gym",
        "world_pin": "world-rev-1",
        "safety_evidence": "fixture: env reports info.max_pen",
        "metrics_only": False,
        "install": PipExtraInstall(
            strategy=InstallStrategy.PIP_EXTRA,
            extras=("cathsim",),
            packages=(
                PinnedPackage(name="cathsim", version="1.2.3"),
                PinnedPackage(name="mujoco", version="3.1.6"),
            ),
            verify_import="cathsim",
        ),
    }
    payload.update(overrides)
    return WorldPackage.model_validate(payload)


def _container_world(*, digest: str) -> WorldPackage:
    return WorldPackage(
        id="pinned-container-world",
        display_name="Pinned Container World",
        domain="Laparoscopic",
        engine="SOFA",
        disposition=Disposition.WRAP,
        license="Apache-2.0",
        world_kind="sofa",
        world_pin="world-rev-1",
        safety_evidence="fixture: env reports info.wall_force_n",
        metrics_only=False,
        install=PrebuiltContainerInstall(
            strategy=InstallStrategy.PREBUILT_CONTAINER,
            image="ghcr.io/seldingermed/vector-world-lapgym",
            image_digest=digest,
        ),
    )


def _vendor_world(
    *,
    eula_url: str = "https://vendor.example/eula",
    version: str = "4.5.0",
) -> WorldPackage:
    return WorldPackage(
        id="vendor-world",
        display_name="Vendor World",
        domain="dVRK manipulation",
        engine="Isaac Lab",
        disposition=Disposition.WRAP,
        license="Apache-2.0",
        world_kind="isaac-lab",
        world_pin="world-rev-1",
        safety_evidence="fixture: shaped reward only, no force channel",
        install=VendorRuntimeInstall(
            strategy=InstallStrategy.VENDOR_RUNTIME,
            vendor="NVIDIA",
            container=f"nvcr.io/nvidia/isaac-sim:{version}",
            login_registry="nvcr.io",
            eula_url=eula_url,
            pinned_version=version,
        ),
    )


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------


def test_catalog_loads_and_every_entry_validates():
    catalog = load_catalog()
    assert catalog.catalog_version
    assert catalog.worlds
    assert len(set(catalog.ids)) == len(catalog.ids)
    for pkg in catalog.worlds:
        assert pkg.display_name
        assert pkg.domain
        assert pkg.engine
        assert pkg.strategy is pkg.install.strategy
        assert pkg.pin_state in {"pinned", "unpinned", "world-unpinned", "install-unpinned"}
        # Notes or risks: every row must say why it has the disposition it has.
        assert pkg.notes.strip() or pkg.risks.strip()


def test_catalog_uses_only_strategies_it_can_honour():
    """Every row's strategy is real, and an empty strategy is a finding.

    ``pip-extra`` deliberately has no rows. The 2026-08 audit checked the two
    candidates that had one: ``cathsim`` and ``surrol`` are both absent from
    PyPI (404), so the previous pip-extra rows named packages that could never
    resolve. Anyone re-adding one must first confirm the distribution is
    actually published - and must not reach for a same-sounding name, because
    ``pip install steve`` fetches an unrelated static-site generator, not stEVE.
    """
    used = {pkg.strategy for pkg in load_catalog().worlds}
    assert used == {
        InstallStrategy.FIRST_PARTY,
        InstallStrategy.SOURCE_BUILD,
        InstallStrategy.VENDOR_RUNTIME,
    }
    for strategy in used:
        assert iter_packages(strategy=strategy)


def test_catalog_records_the_named_worlds_with_audited_dispositions():
    """Dispositions after verification, which is not the same as after survey.

    ``cathsim`` is the one that moved: Appendix B called it a wrap target, and
    reading its LICENSE (CC-BY-NC-SA-4.0, plus a field-of-use TERMS.md) settled
    that we may not ship it. The engineering was never the blocker - the env
    runs and reports real contact forces - so this row exists to stop a future
    reader from "fixing" the disposition back.
    """
    expected = {
        "lumen": Disposition.SHIPPED,
        "steve": Disposition.WRAP,
        "orbit-surgical": Disposition.WRAP,
        "surrol": Disposition.WRAP,
        "lapgym": Disposition.WRAP,
        "sonogym": Disposition.WRAP,
        "surgicai": Disposition.WRAP,
        "surgical-gym": Disposition.WRAP,
        "sofagym": Disposition.WATCH,
        "vr-caps": Disposition.WATCH,
        "cathsim": Disposition.SKIP,
        "dvrl": Disposition.SKIP,
        "rl-cataract": Disposition.SKIP,
    }
    for world_id, disposition in expected.items():
        assert world_package(world_id).disposition is disposition


def test_catalog_never_fabricates_a_pin_or_a_license():
    """Unverified data must be labelled, not guessed.

    Any entry claiming an audited license or a complete pin set is asserting
    something a human checked; this test is the tripwire that keeps such a
    claim from arriving by accident with the rest of a diff.
    """
    for pkg in load_catalog().worlds:
        if not pkg.license_verified:
            assert pkg.license == UNVERIFIED
        if isinstance(pkg.install, PrebuiltContainerInstall) and pkg.install.image_digest:
            assert pkg.install.image_digest.startswith("sha256:")
        if isinstance(pkg.install, VendorRuntimeInstall) and pkg.install.pinned_version:
            assert pkg.install.pinned_version in pkg.install.container


def test_unknown_world_id_refuses_with_the_known_ids():
    with pytest.raises(TaskContractError, match="unknown world package"):
        world_package("not-a-world")


def test_duplicate_ids_refuse_at_load(tmp_path: Path):
    path = tmp_path / "catalog.toml"
    entry = """
[[worlds]]
id = "dup"
display_name = "Dup"
domain = "d"
engine = "e"
disposition = "wrap"
safety_evidence = "fixture"
[worlds.install]
strategy = "source-build"
"""
    path.write_text(f'catalog_version = "1"\n{entry}{entry}', encoding="utf-8")
    with pytest.raises(TaskContractError, match="duplicate world package id"):
        load_catalog(path)


# --------------------------------------------------------------------------
# plan_install
# --------------------------------------------------------------------------


def test_pip_extra_plan_is_pinned_argv():
    plan = plan_install(_pip_world(), pip_argv=("pip", "install"))
    assert plan.strategy is InstallStrategy.PIP_EXTRA
    install_argv = plan.commands[0]
    assert install_argv[:2] == ("pip", "install")
    assert "cathsim==1.2.3" in install_argv
    assert "mujoco==3.1.6" in install_argv
    assert any(arg.startswith("surgeval[cathsim]==") for arg in install_argv)
    # Every requirement carries a version: an unpinned spec would not replay.
    assert all("==" in arg for arg in install_argv[2:])
    # The last step verifies the import locally, so a green pip run is not
    # mistaken for a working world.
    assert plan.steps[-1].network is False
    assert "import cathsim" in " ".join(plan.steps[-1].argv)


def test_pip_extra_refuses_an_unpinned_package():
    world = _pip_world(
        install=PipExtraInstall(
            strategy=InstallStrategy.PIP_EXTRA,
            packages=(PinnedPackage(name="cathsim", version=""),),
            verify_import="cathsim",
        )
    )
    with pytest.raises(TaskContractError, match="unpinned package"):
        plan_install(world, pip_argv=("pip", "install"))


def test_container_plan_pulls_the_digest_pinned_reference():
    plan = plan_install(_container_world(digest=PINNED_DIGEST), container_runtime="docker")
    pull = plan.commands[0]
    assert pull[:2] == ("docker", "pull")
    assert pull[2].endswith(f"@{PINNED_DIGEST}")
    assert plan.steps[-1].network is False


def test_container_refuses_an_undigested_image():
    with pytest.raises(TaskContractError, match="moving target"):
        plan_install(_container_world(digest=""), container_runtime="docker")


def test_container_refuses_a_malformed_digest():
    with pytest.raises(TaskContractError, match="moving target"):
        plan_install(_container_world(digest="sha256:deadbeef"), container_runtime="docker")


def test_vendor_runtime_requires_explicit_eula_acceptance():
    world = _vendor_world()
    with pytest.raises(TaskContractError, match="--accept-vendor-eula"):
        plan_install(world, container_runtime="docker")
    plan = plan_install(world, container_runtime="docker", accept_vendor_eula=True)
    assert plan.acknowledgements
    assert "vendor.example/eula" in plan.acknowledgements[0]


def test_vendor_runtime_pulls_the_vendors_image_and_verifies_the_pin():
    plan = plan_install(_vendor_world(), container_runtime="docker", accept_vendor_eula=True)
    rendered = plan.render()
    assert ("docker", "login", "nvcr.io") in plan.commands
    assert ("docker", "pull", "nvcr.io/nvidia/isaac-sim:4.5.0") in plan.commands
    assert ("docker", "image", "inspect", "nvcr.io/nvidia/isaac-sim:4.5.0") in plan.commands
    # We drive the vendor's own pull; nothing in the plan republishes it.
    assert "4.5.0" in rendered


def test_vendor_runtime_refuses_when_the_eula_is_not_recorded():
    world = _vendor_world(eula_url="")
    with pytest.raises(TaskContractError, match="no eula_url"):
        plan_install(world, container_runtime="docker", accept_vendor_eula=True)


def test_vendor_runtime_refuses_when_the_reference_does_not_name_the_pin():
    world = WorldPackage(
        id="vendor-world",
        display_name="Vendor World",
        domain="dVRK",
        engine="Isaac Lab",
        disposition=Disposition.WRAP,
        license="Apache-2.0",
        safety_evidence="fixture: no force channel",
        install=VendorRuntimeInstall(
            strategy=InstallStrategy.VENDOR_RUNTIME,
            vendor="NVIDIA",
            container="nvcr.io/nvidia/isaac-sim:latest",
            eula_url="https://vendor.example/eula",
            pinned_version="4.5.0",
        ),
    )
    with pytest.raises(TaskContractError, match="does not name it"):
        plan_install(world, container_runtime="docker", accept_vendor_eula=True)


def test_first_party_plan_is_a_local_verification_only():
    world = WorldPackage(
        id="lumen",
        display_name="Lumen",
        domain="Endovascular",
        engine="Lumen",
        disposition=Disposition.SHIPPED,
        license="Apache-2.0",
        safety_evidence="fixture: task.toml gates bind info.unsafe",
        metrics_only=False,
        install=FirstPartyInstall(distribution="seldinger-lumen", verify_import="lumen"),
    )
    plan = plan_install(world)
    assert len(plan.steps) == 1
    assert plan.steps[0].network is False


def test_install_refuses_an_unaudited_license():
    with pytest.raises(TaskContractError, match="terms we have not read"):
        plan_install(_pip_world(license=UNVERIFIED), pip_argv=("pip", "install"))


@pytest.mark.parametrize("disposition", [Disposition.WATCH, Disposition.SKIP])
def test_install_refuses_a_survey_row(disposition: Disposition):
    with pytest.raises(TaskContractError, match="not a shelf item"):
        plan_install(_pip_world(disposition=disposition), pip_argv=("pip", "install"))


def test_every_catalog_entry_either_plans_or_names_its_missing_artifact():
    """No entry may fail silently or with an unactionable message."""
    unactionable: list[str] = []
    for pkg in load_catalog().worlds:
        message = ""
        try:
            plan_install(pkg, container_runtime="docker", pip_argv=("pip", "install"))
        except TaskContractError as exc:
            message = str(exc)
        if message and "Fix:" not in message and "re-run with" not in message:
            unactionable.append(f"{pkg.id}: {message}")
    assert not unactionable


# --------------------------------------------------------------------------
# execute_install
# --------------------------------------------------------------------------


def test_dry_run_records_the_plan_and_runs_nothing():
    plan = plan_install(_pip_world(), pip_argv=("pip", "install"))
    runner = FakeRunner()
    outcome = execute_install(plan, dry_run=True, runner=runner)
    assert outcome.dry_run
    assert outcome.ok
    assert outcome.commands == plan.commands
    assert outcome.exit_codes == ()
    assert runner.calls == []


def test_execute_defaults_to_dry_run():
    plan = plan_install(_pip_world(), pip_argv=("pip", "install"))
    runner = FakeRunner()
    assert execute_install(plan, runner=runner).dry_run
    assert runner.calls == []


def test_execute_runs_exactly_the_planned_commands_in_order():
    plan = plan_install(_pip_world(), pip_argv=("pip", "install"))
    runner = FakeRunner()
    outcome = execute_install(plan, dry_run=False, runner=runner)
    assert tuple(runner.calls) == plan.commands
    assert outcome.ok
    assert outcome.exit_codes == (0,) * len(plan.commands)


def test_execute_stops_at_the_first_failure():
    plan = plan_install(_container_world(digest=PINNED_DIGEST), container_runtime="docker")
    runner = FakeRunner(code=1)
    outcome = execute_install(plan, dry_run=False, runner=runner)
    assert not outcome.ok
    assert len(runner.calls) == 1
    assert outcome.exit_codes == (1,)


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def test_doctor_report_on_this_machine_is_actionable():
    report = run_doctor()
    assert report.checks
    for check in report.checks:
        if check.status in {CheckStatus.FAIL, CheckStatus.UNKNOWN}:
            assert check.fix.strip(), check.id
    assert {"python", "surgeval", "world-kinds", "reference-task"} <= {
        check.id for check in report.checks
    }
    assert report.exit_code() == (0 if report.ok else 1)


def test_doctor_surfaces_an_adapter_discovery_failure():
    broken = AdapterDiscovery(name="thirdparty-world", ok=False, error="No module named 'nope'")
    report = run_doctor(discovery=(broken,))
    failure = next(check for check in report.checks if check.id == "world-adapter:thirdparty-world")
    assert failure.status is CheckStatus.FAIL
    assert "No module named" in failure.detail
    assert "uninstall" in failure.fix
    assert not report.ok
    assert report.exit_code() == 1


def test_doctor_optional_worlds_do_not_fail_a_healthy_machine():
    report = run_doctor(discovery=())
    world_checks = [check for check in report.checks if check.id.startswith("world:")]
    assert world_checks
    assert all(not check.required for check in world_checks)
    assert report.ok


def test_doctor_requires_a_named_world():
    """A named world is graded rather than advisory; ``lumen`` is required.

    What the fix *says* is covered by
    ``test_every_doctor_fix_that_names_a_command_is_a_command_that_runs`` and
    ``test_a_blocked_world_gets_the_blocker_as_its_fix``; here the contract is
    only that naming a world makes its check count.
    """
    report = run_doctor(packages=["lumen"], discovery=())
    check = next(item for item in report.checks if item.id == "world:lumen")
    assert check.required
    if check.status is CheckStatus.FAIL:
        assert check.fix
        assert not report.ok


def test_every_doctor_fix_that_names_a_command_is_a_command_that_runs():
    """A fix that fails costs a round trip to discover, so none may.

    The trap this closes was live: doctor told users to run
    ``surgeval worlds install lumen``, and that command refuses because the
    row's license is unrecorded. The test does not pattern-match the string -
    it parses the world id back out and asks the planner whether the command
    would refuse.
    """
    ids = [pkg.id for pkg in load_catalog().worlds]
    report = run_doctor(packages=ids, discovery=())
    named = 0
    for check in report.checks:
        fix = check.fix or ""
        match = re.search(r"surgeval worlds install ([\w.-]+)", fix)
        if not check.id.startswith("world:") or match is None:
            continue
        named += 1
        pkg = world_package(match.group(1))
        # Would the command exit 0? plan_install raises exactly when it refuses.
        plan_install(pkg, pip_argv=("pip", "install"), accept_vendor_eula=True)
    # Guard against the test passing because nothing names a command any more.
    assert named >= 3


def test_a_blocked_world_gets_the_blocker_as_its_fix():
    """When no command helps, the fix is the blocker, stated once."""
    report = run_doctor(packages=["lumen", "cathsim"], discovery=())
    lumen = next(item for item in report.checks if item.id == "world:lumen")
    assert "worlds install lumen" in (lumen.fix or "")
    cathsim = next(item for item in report.checks if item.id == "world:cathsim")
    assert not cathsim.required
    assert "worlds install" not in (cathsim.fix or "")
    assert "CC-BY-NC-SA-4.0" in (cathsim.fix or "")


def test_doctor_names_the_catalog_when_asked_about_an_unknown_world():
    report = run_doctor(packages=["nope"], discovery=())
    check = next(item for item in report.checks if item.id == "world:nope")
    assert check.status is CheckStatus.FAIL
    assert "worlds list" in check.fix
    assert not report.ok


def test_a_failing_check_without_a_fix_is_refused():
    """The command's promise is structural, not a style guideline."""
    with pytest.raises(TaskContractError, match="without a fix"):
        DoctorCheck("bogus", CheckStatus.FAIL, "something broke")


def _fake_which(*present: str) -> Callable[..., str | None]:
    """PATH lookup that finds exactly ``present``, so probes do not read this machine."""

    def which(name: str, *_args: object, **_kwargs: object) -> str | None:
        return f"/usr/bin/{name}" if name in present else None

    return which


def _one_world_catalog(pkg: WorldPackage) -> WorldCatalog:
    return WorldCatalog(catalog_version="test", worlds=(pkg,))


def test_a_required_unknown_check_fails_the_command():
    """The fail-open shape: a required world nobody could probe used to exit 0.

    ``unknown`` stays the printed label - it is honest, and distinct from
    ``fail`` - but an unprobed requirement is not a satisfied one.
    """
    check = DoctorCheck(
        "world:orbit-surgical",
        CheckStatus.UNKNOWN,
        "no GPU driver tooling on PATH, so the NVIDIA runtime cannot be probed",
        fix="on a GPU host re-run `surgeval doctor --world orbit-surgical`",
    )
    report = DoctorReport(checks=(check,))
    assert check.blocking
    assert report.failures == (check,)
    assert not report.ok
    assert report.exit_code() == 1
    assert "unknown" in report.render()


def test_an_advisory_unknown_check_stays_advisory():
    """A CPU-only machine scanning the whole shelf is still a healthy machine."""
    check = DoctorCheck(
        "world:orbit-surgical",
        CheckStatus.UNKNOWN,
        "no GPU driver tooling on PATH",
        fix="on a GPU host re-run `surgeval doctor --world orbit-surgical`",
        required=False,
    )
    report = DoctorReport(checks=(check,))
    assert not check.blocking
    assert report.advisories == (check,)
    assert report.ok
    assert report.exit_code() == 0


def test_doctor_inspects_the_pinned_image_before_calling_a_container_world_ok(
    monkeypatch: pytest.MonkeyPatch,
):
    pkg = _container_world(digest=PINNED_DIGEST)
    monkeypatch.setattr("or_audit.install.doctor.shutil.which", _fake_which("docker"))
    asked: list[tuple[str, str]] = []

    def present(runtime: str, reference: str) -> bool:
        asked.append((runtime, reference))
        return True

    monkeypatch.setattr(doctor_mod, "_image_present", present)
    report = run_doctor(packages=[pkg.id], catalog=_one_world_catalog(pkg), discovery=())
    check = next(item for item in report.checks if item.id == f"world:{pkg.id}")
    assert check.status is CheckStatus.OK
    spec = pkg.install
    assert isinstance(spec, PrebuiltContainerInstall)
    # The exact digest-pinned reference, not the bare image name.
    assert asked == [("/usr/bin/docker", f"{spec.image}@{PINNED_DIGEST}")]
    assert report.exit_code() == 0


def test_doctor_fails_a_required_container_world_whose_image_is_absent(
    monkeypatch: pytest.MonkeyPatch,
):
    """Reviewer's probe: docker on PATH, image never pulled, and doctor said ok."""
    pkg = _container_world(digest=PINNED_DIGEST)
    monkeypatch.setattr("or_audit.install.doctor.shutil.which", _fake_which("docker"))
    monkeypatch.setattr(doctor_mod, "_image_present", lambda _runtime, _reference: False)
    report = run_doctor(packages=[pkg.id], catalog=_one_world_catalog(pkg), discovery=())
    check = next(item for item in report.checks if item.id == f"world:{pkg.id}")
    assert check.status is CheckStatus.FAIL
    assert check.fix
    assert report.exit_code() == 1


def test_doctor_cannot_grade_a_container_world_when_the_runtime_will_not_answer(
    monkeypatch: pytest.MonkeyPatch,
):
    """A daemon that never answered proves neither presence nor absence."""
    pkg = _container_world(digest=PINNED_DIGEST)
    monkeypatch.setattr("or_audit.install.doctor.shutil.which", _fake_which("docker"))
    monkeypatch.setattr(doctor_mod, "_image_present", lambda _runtime, _reference: None)
    report = run_doctor(packages=[pkg.id], catalog=_one_world_catalog(pkg), discovery=())
    check = next(item for item in report.checks if item.id == f"world:{pkg.id}")
    assert check.status is CheckStatus.UNKNOWN
    assert "docker info" in check.fix
    assert report.exit_code() == 1


def test_doctor_requires_a_container_runtime_for_a_vendor_world(
    monkeypatch: pytest.MonkeyPatch,
):
    """Reviewer's probe: nvidia-smi present, no runtime, no image, still ok."""
    pkg = _vendor_world()
    monkeypatch.setattr("or_audit.install.doctor.shutil.which", _fake_which("nvidia-smi"))
    monkeypatch.setattr(
        doctor_mod,
        "_image_present",
        lambda _runtime, _reference: pytest.fail("no runtime, so nothing to inspect"),
    )
    report = run_doctor(packages=[pkg.id], catalog=_one_world_catalog(pkg), discovery=())
    check = next(item for item in report.checks if item.id == f"world:{pkg.id}")
    assert check.status is not CheckStatus.OK
    assert "no container runtime" in check.detail
    assert report.exit_code() == 1


def test_doctor_inspects_the_vendor_image_when_a_runtime_is_available(
    monkeypatch: pytest.MonkeyPatch,
):
    pkg = _vendor_world()
    monkeypatch.setattr("or_audit.install.doctor.shutil.which", _fake_which("nvidia-smi", "podman"))
    asked: list[tuple[str, str]] = []

    def present(runtime: str, reference: str) -> bool:
        asked.append((runtime, reference))
        return False

    monkeypatch.setattr(doctor_mod, "_image_present", present)
    absent = run_doctor(packages=[pkg.id], catalog=_one_world_catalog(pkg), discovery=())
    check = next(item for item in absent.checks if item.id == f"world:{pkg.id}")
    assert check.status is CheckStatus.FAIL
    assert absent.exit_code() == 1
    spec = pkg.install
    assert isinstance(spec, VendorRuntimeInstall)
    assert asked == [("/usr/bin/podman", spec.container)]

    monkeypatch.setattr(doctor_mod, "_image_present", lambda _runtime, _reference: True)
    found = run_doctor(packages=[pkg.id], catalog=_one_world_catalog(pkg), discovery=())
    check = next(item for item in found.checks if item.id == f"world:{pkg.id}")
    assert check.status is CheckStatus.OK
    assert found.exit_code() == 0


def test_the_image_probe_reads_local_metadata_and_never_pulls(
    monkeypatch: pytest.MonkeyPatch,
):
    """``image inspect`` only, with a daemon outage kept distinct from absence."""
    calls: list[tuple[str, ...]] = []
    answers = [
        (0, ""),
        (1, "Error: No such image: img@sha256:0"),
        (1, "Cannot connect to the Docker daemon at unix:///x.sock"),
    ]

    def fake_run(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(argv))
        # Offline by construction: a timeout is set and no pull is ever run.
        assert kwargs["timeout"] == doctor_mod.IMAGE_INSPECT_TIMEOUT_S
        code, stderr = answers.pop(0)
        return subprocess.CompletedProcess(list(argv), code, stdout="", stderr=stderr)

    monkeypatch.setattr("or_audit.install.doctor.subprocess.run", fake_run)
    assert doctor_mod._image_present("docker", "img@sha256:0") is True
    assert doctor_mod._image_present("docker", "img@sha256:0") is False
    # A runtime that could not be reached has not proven the image is absent.
    assert doctor_mod._image_present("docker", "img@sha256:0") is None
    assert calls == [("docker", "image", "inspect", "img@sha256:0")] * 3


def test_the_image_probe_reports_unknown_when_the_runtime_cannot_be_run(
    monkeypatch: pytest.MonkeyPatch,
):
    def explode(_argv: Sequence[str], **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="docker", timeout=1)

    monkeypatch.setattr("or_audit.install.doctor.subprocess.run", explode)
    assert doctor_mod._image_present("docker", "img@sha256:0") is None


def test_doctor_json_report_is_machine_readable(capsys: pytest.CaptureFixture[str]):
    parser = _parser()
    args = parser.parse_args(["doctor", "--json"])
    code = args.func(args)
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"ok", "checks", "failures", "advisories"}
    assert report["ok"] is (code == 0)
    for check in report["checks"]:
        assert set(check) == {"id", "status", "detail", "fix", "required"}
        if check["status"] in {"fail", "unknown"}:
            assert check["fix"]


# --------------------------------------------------------------------------
# quickstart
# --------------------------------------------------------------------------


def test_quickstart_produces_a_verifying_vector(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    out = tmp_path / "job"
    parser = _parser()
    args = parser.parse_args(["quickstart", "--out", str(out)])
    assert args.func(args) == 0
    captured = capsys.readouterr().out
    assert "time to first vector:" in captured
    assert f"surgeval replay {out}" in captured
    result = read_job_result(out)
    assert verify_head(result)
    assert result.task_id == "video-nextstep"
    assert result.head in captured


def test_quickstart_json_shape(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    out = tmp_path / "job"
    parser = _parser()
    args = parser.parse_args(["quickstart", "--out", str(out), "--json"])
    assert args.func(args) == 0
    record = json.loads(capsys.readouterr().out)
    assert set(record) == {"time_to_first_vector_sec", "task_id", "head", "out"}
    assert record["task_id"] == "video-nextstep"
    assert record["out"] == str(out)
    assert record["head"] == read_job_result(out).head
    assert record["time_to_first_vector_sec"] > 0


def test_quickstart_refuses_a_missing_package_with_an_actionable_message(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    parser = _parser()
    args = parser.parse_args(
        [
            "quickstart",
            "--out",
            str(tmp_path / "job"),
            "--task",
            str(tmp_path / "absent-task"),
            "--agent",
            str(tmp_path / "absent-agent"),
        ]
    )
    assert args.func(args) == 1
    assert "REFUSED:" in capsys.readouterr().err


def test_reference_packages_are_on_disk_in_a_checkout():
    found = find_reference_paths()
    assert found is not None
    task_dir, agent_dir = found
    assert (task_dir / "task.toml").is_file()
    assert (agent_dir / "agent.toml").is_file()


def _fake_installed_package(tmp_path: Path) -> Path:
    """A stand-in ``or_audit`` install directory carrying ``_examples``.

    This is the wheel layout the pyproject ``force-include`` produces, built
    from the in-tree originals so the copies cannot drift from what ships.
    """
    checkout_task, checkout_agent = doctor_mod.find_reference_paths() or (None, None)
    assert checkout_task is not None
    assert checkout_agent is not None
    package_dir = tmp_path / "site-packages" / "or_audit"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    packaged = package_dir / doctor_mod.PACKAGED_EXAMPLES_DIRNAME
    shutil.copytree(checkout_task, packaged / doctor_mod.PACKAGED_TASK_RELPATH)
    shutil.copytree(checkout_agent, packaged / doctor_mod.PACKAGED_AGENT_RELPATH)
    return package_dir


def test_reference_pair_resolves_from_the_installed_package_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """``uv tool install surgeval && surgeval quickstart`` needs no checkout.

    The wheel force-includes the reference pair under ``or_audit/_examples``,
    and that copy must win over any ``docs/examples`` the caller happens to be
    standing in: the vector should describe the packages the user installed.
    """
    package_dir = _fake_installed_package(tmp_path)
    monkeypatch.setattr(or_audit, "__file__", str(package_dir / "__init__.py"))
    found = doctor_mod.find_reference_paths()
    assert found is not None
    task_dir, agent_dir = found
    assert task_dir == package_dir / doctor_mod.PACKAGED_EXAMPLES_DIRNAME / "tasks/video-nextstep"
    assert (task_dir / "task.toml").is_file()
    assert (agent_dir / "agent.toml").is_file()


def test_quickstart_runs_against_the_installed_package_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    package_dir = _fake_installed_package(tmp_path)
    monkeypatch.setattr(or_audit, "__file__", str(package_dir / "__init__.py"))
    out = tmp_path / "job"
    parser = _parser()
    args = parser.parse_args(["quickstart", "--out", str(out), "--json"])
    assert args.func(args) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["task_id"] == "video-nextstep"
    assert record["time_to_first_vector_sec"] > 0
    result = read_job_result(out)
    assert verify_head(result)
    assert result.head == record["head"]


def test_quickstart_refuses_when_no_layout_carries_the_reference_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """A wheel with no examples and no checkout must say what to do."""
    empty = tmp_path / "site-packages" / "or_audit"
    empty.mkdir(parents=True)
    monkeypatch.setattr(or_audit, "__file__", str(empty / "__init__.py"))
    monkeypatch.chdir(tmp_path)
    parser = _parser()
    args = parser.parse_args(["quickstart", "--out", str(tmp_path / "job")])
    assert args.func(args) == 1
    err = capsys.readouterr().err
    assert "REFUSED:" in err
    assert "or_audit/_examples" in err
    assert "--task/--agent" in err
    assert "Searched:" in err


def test_packaged_example_paths_match_the_wheel_force_include():
    """The constants and the packaging rule are one contract, in two files.

    ``pyproject.toml`` decides where the reference pair lands in the wheel and
    this module decides where to look for it. A silent rename in either place
    would turn `surgeval quickstart` on a fresh install into a refusal, so the
    mapping is asserted rather than trusted.
    """
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as handle:
        config = tomllib.load(handle)
    include = config["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    prefix = f"or_audit/{doctor_mod.PACKAGED_EXAMPLES_DIRNAME}"
    assert include[str(doctor_mod.REFERENCE_TASK_RELPATH)] == (
        f"{prefix}/{doctor_mod.PACKAGED_TASK_RELPATH}"
    )
    assert include[str(doctor_mod.REFERENCE_AGENT_RELPATH)] == (
        f"{prefix}/{doctor_mod.PACKAGED_AGENT_RELPATH}"
    )


# --------------------------------------------------------------------------
# worlds command
# --------------------------------------------------------------------------


def test_worlds_list_shows_pin_state_and_the_watch_rows(capsys: pytest.CaptureFixture[str]):
    parser = _parser()
    args = parser.parse_args(["worlds", "list"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "source-build" in out
    assert "vendor-runtime" in out
    assert "first-party" in out
    assert "watch" in out
    assert "skip" in out
    assert "unverified" in out


def test_worlds_list_filters(capsys: pytest.CaptureFixture[str]):
    parser = _parser()
    args = parser.parse_args(["worlds", "list", "--strategy", "vendor-runtime"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "orbit-surgical" in out
    assert "cathsim" not in out


def test_worlds_info_prints_sources_and_registry_state(capsys: pytest.CaptureFixture[str]):
    parser = _parser()
    args = parser.parse_args(["worlds", "info", "steve"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "github.com/lkarstensen/stEVE" in out
    assert "source-build" in out
    assert "installable no" in out


def test_worlds_info_unknown_id_refuses(capsys: pytest.CaptureFixture[str]):
    parser = _parser()
    args = parser.parse_args(["worlds", "info", "nope"])
    assert args.func(args) == 1
    assert "REFUSED:" in capsys.readouterr().err


def test_worlds_install_refuses_a_survey_row_with_a_reachable_remedy(
    capsys: pytest.CaptureFixture[str],
):
    """An off-shelf row refuses, and the named remedy must be one that exists.

    Three distinct cases, because a generic "promote it to 'wrap'" is a false
    remedy for two of them: ``cathsim`` is engineering-ready but licensed
    CC-BY-NC-SA-4.0 (no amount of work promotes it), ``dvrl`` has no runnable
    artifact to promote, and only ``vr-caps`` is waiting on the audit that
    promotion actually requires.
    """
    parser = _parser()
    expected = {
        "cathsim": "promotion is not an engineering task",
        "sofagym": "promotion is not an engineering task",
        "dvrl": "nothing to promote",
        "vr-caps": "promote it to 'wrap'",
    }
    for world_id, remedy in expected.items():
        args = parser.parse_args(["worlds", "install", world_id])
        assert args.func(args) == 1
        err = capsys.readouterr().err
        assert "REFUSED:" in err
        assert "not a shelf item" in err
        assert remedy in err, f"{world_id} refusal named the wrong remedy"
    # The blocked rows must say which terms blocked them, not just that they did.
    args = parser.parse_args(["worlds", "install", "cathsim"])
    assert args.func(args) == 1
    assert "CC-BY-NC-SA-4.0" in capsys.readouterr().err


def test_worlds_install_plans_a_source_build_row(capsys: pytest.CaptureFixture[str]):
    """The shelf's source-build rows plan; only their build steps are manual."""
    parser = _parser()
    args = parser.parse_args(["worlds", "install", "steve"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "install plan: steve (source-build)" in out
    assert "git clone" in out
    assert "manual:" in out
    assert "this plan is not executable" in out


def test_worlds_without_a_subcommand_is_a_usage_error(capsys: pytest.CaptureFixture[str]):
    parser = _parser()
    args = parser.parse_args(["worlds"])
    assert args.func(args) == 2
    assert "requires a subcommand" in capsys.readouterr().err


# --------------------------------------------------------------------------
# worlds command: dry-run/execute coherence (CliCoherence)
# --------------------------------------------------------------------------


class _RecordingInstall:
    """Stands in for ``execute_install`` and records the intent it was handed."""

    def __init__(self) -> None:
        self.dry_runs: list[bool] = []

    def __call__(self, plan: InstallPlan, *, dry_run: bool = True) -> InstallOutcome:
        self.dry_runs.append(dry_run)
        return InstallOutcome(world_id=plan.world_id, dry_run=dry_run, commands=plan.commands)


@pytest.fixture
def recorded_install(monkeypatch: pytest.MonkeyPatch) -> _RecordingInstall:
    """Replace the installer so these tests can never reach a real runtime."""
    recorder = _RecordingInstall()
    monkeypatch.setattr(worlds_cmd, "execute_install", recorder)
    return recorder


def test_worlds_install_refuses_dry_run_and_execute_together(
    recorded_install: _RecordingInstall, capsys: pytest.CaptureFixture[str]
):
    """The dangerous reading used to win: --dry-run --execute installed anyway.

    argparse rejects the pair before a plan exists, so the installer is never
    reached at all - not even in dry-run mode.
    """
    parser = _parser()
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["worlds", "install", "orbit-surgical", "--dry-run", "--execute"])
    assert exit_info.value.code == 2
    assert "not allowed with argument --dry-run" in capsys.readouterr().err
    assert recorded_install.dry_runs == [], "a contradictory request must install nothing"


def test_worlds_install_dry_run_flag_is_load_bearing(recorded_install: _RecordingInstall):
    """--dry-run decides the outcome itself, rather than only being the default."""
    parser = _parser()
    args = parser.parse_args(
        ["worlds", "install", "orbit-surgical", "--accept-vendor-eula", "--dry-run"]
    )
    assert args.func(args) == 0
    assert recorded_install.dry_runs == [True]


def test_worlds_install_defaults_to_a_dry_run(recorded_install: _RecordingInstall):
    parser = _parser()
    args = parser.parse_args(["worlds", "install", "orbit-surgical", "--accept-vendor-eula"])
    assert args.func(args) == 0
    assert recorded_install.dry_runs == [True]


def test_worlds_install_execute_is_the_only_way_to_run_commands(
    recorded_install: _RecordingInstall,
):
    parser = _parser()
    args = parser.parse_args(
        ["worlds", "install", "orbit-surgical", "--accept-vendor-eula", "--execute"]
    )
    assert args.func(args) == 0
    assert recorded_install.dry_runs == [False]


def test_worlds_install_execute_refuses_a_manual_plan_instead_of_tracebacking(
    capsys: pytest.CaptureFixture[str],
):
    """`--execute` on a plan with manual steps is a refusal, not a stack trace.

    Every source-build row is non-executable, so this is the common path, not an
    edge: the installer's ``TaskContractError`` used to escape the command and
    reach the user as a traceback with no exit code contract at all.
    """
    parser = _parser()
    args = parser.parse_args(["worlds", "install", "steve", "--execute"])
    assert args.func(args) == 1
    captured = capsys.readouterr()
    assert "install plan: steve (source-build)" in captured.out
    assert "REFUSED:" in captured.err
    assert "--dry-run" in captured.err, "the refusal must name the command that works"
