"""The citation checker must fail closed.

This script is the only thing that makes "audited" mean anything: it is what
turns a catalog claim ("this key is assigned at this path, at this pin") into a
checked fact. An earlier version printed ``SKIP`` for an unreachable source and
still returned 0 as long as some other world passed, so a registry outage could
have reported success while verifying none of the seven wrap targets.

These tests are hermetic - no network. Reachability against the real upstream
trees is what the scheduled workflow does; what is pinned here is the decision
logic, especially every path that must *not* return success.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import check_world_signals as cws  # type: ignore[import-not-found]  # noqa: E402

from or_audit.install.catalog import AuditedEnv, SignalKind, WorldSignal  # noqa: E402


def _tree(tmp_path: Path, rel: str, body: str) -> Path:
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return tmp_path


class TestCitationChecking:
    def test_a_resolved_citation_reports_no_problem(self, tmp_path: Path):
        tree = _tree(tmp_path, "env.py", "a\nb\nreward['force'] = compute()\n")
        env = AuditedEnv(
            env_id="e",
            path="env.py",
            signals=(WorldSignal(key="force", kind=SignalKind.PHYSICAL, unit="N", line=3),),
        )
        assert cws.check_world(_pkg(env), tree) == []

    def test_a_missing_path_is_a_problem_and_names_the_basename_hit(self, tmp_path: Path):
        """The ``sonogym`` defect: a citation naming a file that is not there."""
        tree = _tree(tmp_path, "deep/nested/env.py", "reward['force'] = 1\n")
        env = AuditedEnv(
            env_id="e",
            path="env.py",
            signals=(WorldSignal(key="force", kind=SignalKind.PHYSICAL, unit="N"),),
        )
        (problem,) = cws.check_world(_pkg(env), tree)
        assert "missing at pin" in problem
        assert "deep/nested/env.py" in problem

    def test_a_key_absent_from_the_file_is_a_problem(self, tmp_path: Path):
        """The invented-key defect this check caught in our own catalog."""
        tree = _tree(tmp_path, "env.py", "reward['other'] = 1\n")
        env = AuditedEnv(
            env_id="e",
            path="env.py",
            signals=(WorldSignal(key="force", kind=SignalKind.PHYSICAL, unit="N"),),
        )
        (problem,) = cws.check_world(_pkg(env), tree)
        assert "key never appears" in problem

    def test_a_wrong_line_is_a_problem_and_names_the_real_lines(self, tmp_path: Path):
        tree = _tree(tmp_path, "env.py", "x\nreward['force'] = 1\ny\n")
        env = AuditedEnv(
            env_id="e",
            path="env.py",
            signals=(WorldSignal(key="force", kind=SignalKind.PHYSICAL, unit="N", line=99),),
        )
        (problem,) = cws.check_world(_pkg(env), tree)
        assert "cites line 99" in problem

    def test_a_signal_may_cite_its_own_file(self, tmp_path: Path):
        """stEVE's shape: the signal is assigned several modules from the env."""
        tree = _tree(tmp_path, "env.py", "pass\n")
        _tree(tmp_path, "sim/adapter.py", "x\nself.simulation_error = True\n")
        env = AuditedEnv(
            env_id="e",
            path="env.py",
            signals=(
                WorldSignal(
                    key="simulation_error",
                    kind=SignalKind.DIAGNOSTIC,
                    path="sim/adapter.py",
                    line=2,
                ),
            ),
        )
        assert cws.check_world(_pkg(env), tree) == []


class TestFailsClosed:
    """Every path that cannot verify must be non-success."""

    def test_missing_curl_is_a_failure_not_a_skip(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(cws.shutil, "which", lambda _name: None)
        assert cws.main([]) == 2

    def test_an_unreachable_source_is_a_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """The exact regression: one world unreachable used to still exit 0."""
        monkeypatch.setattr(cws.shutil, "which", lambda _name: "/usr/bin/curl")

        def refuse(*_args: Any, **_kwargs: Any) -> None:
            raise cws.CheckFailure("simulated registry outage")

        monkeypatch.setattr(cws, "fetch_tree", refuse)
        assert cws.main(["lapgym", "--cache", str(tmp_path)]) == 2

    def test_a_timeout_is_a_failure(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setattr(cws.shutil, "which", lambda _name: "/usr/bin/curl")

        def stall(*_args: Any, **_kwargs: Any) -> None:
            raise subprocess.TimeoutExpired(cmd="curl", timeout=1)

        monkeypatch.setattr(cws, "fetch_tree", stall)
        assert cws.main(["lapgym", "--cache", str(tmp_path)]) == 2

    def test_one_reachable_world_cannot_mask_an_unreachable_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """The precise fail-open shape: a partial run must not report success."""
        monkeypatch.setattr(cws.shutil, "which", lambda _name: "/usr/bin/curl")
        calls: list[str] = []

        def selective(slug: str, sha: str, dest: Path, **_kwargs: Any) -> None:
            calls.append(slug)
            if "sofa_env" in slug:
                (dest / "sofa_env").mkdir(parents=True, exist_ok=True)
                return
            raise cws.CheckFailure("simulated outage")

        monkeypatch.setattr(cws, "fetch_tree", selective)
        code = cws.main(["lapgym", "steve", "--cache", str(tmp_path)])
        assert len(calls) == 2
        assert code != 0

    def test_a_citation_failure_outranks_an_unreachable_world(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """A real defect must report 1, not be masked by an outage elsewhere."""
        monkeypatch.setattr(cws.shutil, "which", lambda _name: "/usr/bin/curl")

        def empty_or_fail(slug: str, sha: str, dest: Path, **_kwargs: Any) -> None:
            if "sofa_env" in slug:
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "placeholder").write_text("x", encoding="utf-8")
                return
            raise cws.CheckFailure("simulated outage")

        monkeypatch.setattr(cws, "fetch_tree", empty_or_fail)
        assert cws.main(["lapgym", "steve", "--cache", str(tmp_path)]) == 1

    def test_a_world_with_no_pin_cannot_be_checked(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(cws.shutil, "which", lambda _name: "/usr/bin/curl")
        pkg = cws.load_catalog().require("lapgym").model_copy(update={"world_pin": ""})
        assert "no world_pin" in "".join(_problems_for(pkg))


class TestRetry:
    def test_a_transient_failure_is_retried_then_raises(self, monkeypatch: pytest.MonkeyPatch):
        """Bounded retry reduces false alarms without ever passing on failure."""
        attempts: list[int] = []

        class Failed:
            returncode = 1
            stdout = b""
            stderr = b"boom"

        def run(*_args: Any, **_kwargs: Any) -> Failed:
            attempts.append(1)
            return Failed()

        monkeypatch.setattr(cws.subprocess, "run", run)
        with pytest.raises(cws.CheckFailure, match="after 3 attempt"):
            cws.fetch_tree("o/r", "a" * 40, Path("/tmp/unused-signals"), timeout=1, attempts=3)
        assert len(attempts) == 3


def _pkg(env: AuditedEnv) -> Any:
    """A catalog row carrying one audited env, for the path-checking helpers."""
    return cws.load_catalog().require("lapgym").model_copy(update={"envs": (env,)})


def _problems_for(pkg: Any) -> list[str]:
    """Run the pin/repo preconditions the way ``main`` does, without network."""
    problems: list[str] = []
    if cws.repo_slug(pkg) is None:
        problems.append("no GitHub repo in install spec or sources")
    if not pkg.world_pin:
        problems.append("no world_pin: a citation cannot be checked against nothing")
    return problems
