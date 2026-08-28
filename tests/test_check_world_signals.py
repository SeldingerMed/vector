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

import json
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

    def test_missing_git_is_a_failure_not_a_skip(self, monkeypatch: pytest.MonkeyPatch):
        """``git`` is the fetcher now: content addressing is what authenticates."""
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


class TestOccurrenceIsNotPublication:
    """A key on the line is not a key published at the line.

    ``docs/CONFORMANCE.md`` says a resolved citation proves the key is assigned
    and published to ``info``/``extras``. Two probes got through weaker
    versions of this: an ``orbit-surgical`` file whose cited lines held only
    the comments ``# time_out`` / ``# object_dropping``, and - once tokenizing
    replaced substring matching - a cited line holding only the one-word
    docstring for the same key.
    """

    def _published(self, tmp_path: Path, body: str, line: int = 1) -> list[str]:
        tree = _tree(tmp_path, "env.py", body)
        env = AuditedEnv(
            env_id="e",
            path="env.py",
            signals=(WorldSignal(key="time_out", kind=SignalKind.BOOKKEEPING, line=line),),
        )
        problems: list[str] = cws.check_world(_pkg(env), tree)
        return problems

    def test_a_key_only_in_a_comment_does_not_resolve(self, tmp_path: Path):
        """The first probe, at the exact cited lines."""
        body = ["pass"] * 200
        body[176] = "# time_out"
        body[178] = "# object_dropping"
        tree = _tree(tmp_path, "env.py", "\n".join(body) + "\n")
        env = AuditedEnv(
            env_id="Isaac-Lift-Needle-PSM-v0",
            path="env.py",
            signals=(
                WorldSignal(key="time_out", kind=SignalKind.BOOKKEEPING, line=177),
                WorldSignal(key="object_dropping", kind=SignalKind.GEOMETRIC, line=179),
            ),
        )
        problems = cws.check_world(_pkg(env), tree)
        assert len(problems) == 2
        assert all("key never appears" in problem for problem in problems)

    def test_a_one_word_docstring_does_not_resolve(self, tmp_path: Path):
        """The second probe: a lone literal is prose, not a publication."""
        (problem,) = self._published(tmp_path, '"""time_out"""\n')
        assert "key never appears" in problem

    def test_a_key_only_read_does_not_resolve(self, tmp_path: Path):
        (problem,) = self._published(tmp_path, "time_out = 0\nif time_out:\n    pass\n", line=2)
        assert "not a publication on cited line 2" in problem

    def test_a_key_only_passed_as_an_argument_does_not_resolve(self, tmp_path: Path):
        (problem,) = self._published(tmp_path, "log(time_out)\n")
        assert "key never appears" in problem

    def test_a_key_inside_a_longer_identifier_does_not_resolve(self, tmp_path: Path):
        (problem,) = self._published(tmp_path, "time_out_total = 1\n")
        assert "key never appears" in problem

    def test_prose_naming_the_key_does_not_resolve(self, tmp_path: Path):
        (problem,) = self._published(tmp_path, '"""Publishes time_out to info."""\n')
        assert "key never appears" in problem

    def test_a_read_subscript_does_not_publish(self, tmp_path: Path):
        """``x = info["time_out"]`` reads the key; it does not assign it."""
        (problem,) = self._published(tmp_path, 'x = info["time_out"]\n')
        assert "key never appears" in problem

    @pytest.mark.parametrize(
        "body",
        [
            "time_out = DoneTerm(func=mdp.time_out)",
            "time_out += 1",
            "time_out: DoneTerm = DoneTerm()",
            "self.time_out = True",
            'self.extras["time_out"] = value',
            'self.info = {"time_out": False}',
            'info.update({"time_out": value})',
        ],
        ids=[
            "name-assignment",
            "augmented-assignment",
            "annotated-assignment",
            "attribute-store",
            "subscript-store",
            "dict-literal-assigned",
            "dict-literal-in-a-call",
        ],
    )
    def test_every_published_shape_the_catalog_cites_resolves(self, tmp_path: Path, body: str):
        """Strict must not mean so strict that a real citation is refused."""
        assert self._published(tmp_path, body + "\n") == []

    def test_a_dict_key_in_a_return_resolves(self, tmp_path: Path):
        """SurRoL's shape: the info dict is built in the ``step`` return path."""
        body = "def step(self):\n    return {'time_out': self._done()}\n"
        assert self._published(tmp_path, body, line=2) == []

    def test_the_real_lines_are_named_when_the_cited_line_is_wrong(self, tmp_path: Path):
        (problem,) = self._published(tmp_path, "# time_out\npass\ntime_out = 1\n")
        assert "not a publication on cited line 1" in problem
        assert "on [3]" in problem

    def test_a_non_python_citation_needs_a_word_boundary(self, tmp_path: Path):
        tree = _tree(tmp_path, "env.yaml", "time_outs: 1\n")
        env = AuditedEnv(
            env_id="e",
            path="env.yaml",
            signals=(WorldSignal(key="time_out", kind=SignalKind.BOOKKEEPING, line=1),),
        )
        (problem,) = cws.check_world(_pkg(env), tree)
        assert "key never appears" in problem
        assert cws.evidence_lines("time_out: 1\n", "time_out", python=False, published=True) == [1]

    def test_a_file_that_does_not_parse_as_python_is_a_problem(self, tmp_path: Path):
        """An unreadable citation is unchecked, and unchecked is not verified."""
        (problem,) = self._published(tmp_path, 'time_out = "unterminated\n')
        assert "does not parse as Python" in problem


class TestUnpublishedSignalsAreCheckedAsCode:
    """An unpublished row claims the opposite of a publication.

    ``surrol`` records ``getContactPoints`` as computed and *discarded*, and
    ``surgical-gym``'s absence markers cite a reward write and a step-count
    read. Demanding a publication shape there would refuse exactly the rows
    that record the honest negative - so the rule is "code, not prose", and it
    is chosen by ``published``, not by the checker's mood.
    """

    def _unpublished(self, tmp_path: Path, body: str, key: str, line: int) -> list[str]:
        tree = _tree(tmp_path, "env.py", body)
        env = AuditedEnv(
            env_id="e",
            path="env.py",
            signals=(
                WorldSignal(
                    key=key,
                    kind=SignalKind.PHYSICAL,
                    unit="contacts",
                    line=line,
                    published=False,
                ),
            ),
        )
        problems: list[str] = cws.check_world(_pkg(env), tree)
        return problems

    def test_a_discarded_call_resolves(self, tmp_path: Path):
        """SurRoL psm_env.py:262 verbatim in shape."""
        body = "points_1 = p.getContactPoints(bodyA=psm.body, linkIndexA=6)\n"
        assert self._unpublished(tmp_path, body, "getContactPoints", 1) == []

    def test_a_read_in_a_termination_expression_resolves(self, tmp_path: Path):
        """Surgical Gym psm.py:273 in shape."""
        body = "self.reset_buf = where(self.progress_buf >= self._max, a, b)\n"
        assert self._unpublished(tmp_path, body, "progress_buf", 1) == []

    def test_prose_still_does_not_resolve(self, tmp_path: Path):
        """The weaker rule is still a rule: a docstring is not code."""
        (problem,) = self._unpublished(tmp_path, '"""getContactPoints"""\n', "getContactPoints", 1)
        assert "key never appears" in problem
        assert "as code" in problem

    def test_the_same_line_would_fail_a_published_signal(self, tmp_path: Path):
        """The asymmetry is the point, so it is pinned rather than implied."""
        body = "points_1 = p.getContactPoints(bodyA=psm.body)\n"
        assert self._unpublished(tmp_path, body, "getContactPoints", 1) == []
        tree = _tree(tmp_path, "published.py", body)
        env = AuditedEnv(
            env_id="e",
            path="published.py",
            signals=(
                WorldSignal(
                    key="getContactPoints",
                    kind=SignalKind.PHYSICAL,
                    unit="contacts",
                    line=1,
                ),
            ),
        )
        (problem,) = cws.check_world(_pkg(env), tree)
        assert "as a publication" in problem


class TestExplicitlyRequestedWorlds:
    """A world the user named must not report success on nothing."""

    def test_a_named_world_with_no_audited_envs_is_a_usage_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        """The probe: ``lumen`` printed "resolved for 0 of 0" and exited 0."""
        monkeypatch.setattr(cws.shutil, "which", lambda _name: "/usr/bin/git")
        monkeypatch.setattr(cws, "fetch_tree", _never)
        assert cws.main(["lumen", "--cache", str(tmp_path)]) == 2
        out = capsys.readouterr().out
        assert "USAGE" in out
        assert "lumen" in out

    def test_the_default_selection_still_skips_unaudited_worlds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        """Naming a world is a request; the default sweep is not."""
        monkeypatch.setattr(cws.shutil, "which", lambda _name: "/usr/bin/git")

        def refuse(*_args: Any, **_kwargs: Any) -> None:
            raise cws.CheckFailure("simulated outage")

        monkeypatch.setattr(cws, "fetch_tree", refuse)
        assert cws.main(["--cache", str(tmp_path)]) == 2
        out = capsys.readouterr().out
        assert "USAGE" not in out
        assert "lumen" not in out


class TestEmptySignalSurface:
    """Verifying zero things is not verifying."""

    def test_an_empty_cited_file_is_a_problem(self, tmp_path: Path):
        """The probe: a zero-byte cached file reported "0 signal(s) resolved"."""
        tree = _tree(tmp_path, "env.py", "")
        env = AuditedEnv(env_id="e", path="env.py")
        (problem,) = cws.check_world(_pkg(env), tree)
        assert "is empty at the pin" in problem

    def test_an_env_pinning_no_line_cannot_be_verified_by_any_fetch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """Not a citation failure (1) - an unverifiable target (2)."""
        monkeypatch.setattr(cws.shutil, "which", lambda _name: "/usr/bin/git")
        monkeypatch.setattr(cws, "fetch_tree", _never)
        monkeypatch.setattr(cws, "load_catalog", _catalog_without_absence_markers)
        assert cws.main(["surgical-gym", "--cache", str(tmp_path)]) == 2

    def test_absence_markers_are_checked_exactly_like_signals(self, tmp_path: Path):
        """An unchecked absence record would be the same defect in better clothes."""
        tree = _tree(tmp_path, "task.py", "rew_buf = 1\n# progress_buf\n")
        env = AuditedEnv(
            env_id="surgicalgym.tasks.psm:PSM",
            path="task.py",
            absence_markers=(
                WorldSignal(key="rew_buf", kind=SignalKind.BOOKKEEPING, line=1, published=False),
                WorldSignal(
                    key="progress_buf", kind=SignalKind.BOOKKEEPING, line=2, published=False
                ),
            ),
        )
        (problem,) = cws.check_world(_pkg(env), tree)
        assert "absence marker 'progress_buf'" in problem
        assert "key never appears" in problem

    def test_the_real_surgical_gym_row_pins_lines(self):
        """The catalog's one empty-surface env states its emptiness."""
        (env,) = cws.load_catalog().require("surgical-gym").envs
        assert env.signals == ()
        assert [marker.line for marker in env.absence_markers] == [266, 273]
        assert cws.unpinned_envs(cws.load_catalog().require("surgical-gym")) == []


class TestContentAddressedFetch:
    """``sha`` must authenticate the tree, not just address a URL."""

    def test_a_pin_the_remote_does_not_carry_is_refused(self, tmp_path: Path):
        """The probe: an unrelated tarball was extracted and labelled as the pin."""
        _repo(tmp_path / "unrelated", {"fabricated.py": "time_out = 1\n"})
        cws.GIT_REMOTE = str(tmp_path / "unrelated")
        try:
            with pytest.raises(cws.CheckFailure, match="after 1 attempt"):
                cws.fetch_tree(
                    "orbit-surgical/orbit-surgical",
                    "6e47534f7d412e4be523116f250c992a63146883",
                    tmp_path / "dest",
                    timeout=60,
                    attempts=1,
                )
        finally:
            cws.GIT_REMOTE = _PRISTINE_REMOTE
        assert not (tmp_path / "dest" / "fabricated.py").exists()

    def test_the_requested_pin_is_checked_out_and_immediately_reusable(self, tmp_path: Path):
        """A fetched tree vouches for itself: git, not a file it wrote."""
        sha = _repo(tmp_path / "src", {"pkg/env.py": "force = 1\n"})
        cws.GIT_REMOTE = str(tmp_path / "src")
        try:
            cws.fetch_tree("o/r", sha, tmp_path / "dest", timeout=60, attempts=1)
        finally:
            cws.GIT_REMOTE = _PRISTINE_REMOTE
        assert (tmp_path / "dest" / "pkg/env.py").read_text(encoding="utf-8") == "force = 1\n"
        assert cws.cache_rejection(tmp_path / "dest", sha, timeout=60) == ""

    def test_a_leftover_file_is_not_layered_under_a_fetch(self, tmp_path: Path):
        """A cached fabrication must not survive the refetch that replaces it."""
        sha = _repo(tmp_path / "src", {"pkg/env.py": "force = 1\n"})
        dest = tmp_path / "dest"
        (dest / "pkg").mkdir(parents=True)
        (dest / "pkg" / "fabricated.py").write_text("force = 1\n", encoding="utf-8")
        cws.GIT_REMOTE = str(tmp_path / "src")
        try:
            cws.fetch_tree("o/r", sha, dest, timeout=60, attempts=1)
        finally:
            cws.GIT_REMOTE = _PRISTINE_REMOTE
        assert not (dest / "pkg" / "fabricated.py").exists()

    def test_the_fetch_only_ever_runs_git(self, monkeypatch: pytest.MonkeyPatch):
        """No ``curl | tar``, so no hostile-tarball extraction surface either."""
        argvs: list[list[str]] = []

        class Failed:
            returncode = 1
            stdout = b""
            stderr = b"boom"

        def run(argv: list[str], **_kwargs: Any) -> Failed:
            argvs.append(argv)
            return Failed()

        monkeypatch.setattr(cws.subprocess, "run", run)
        with pytest.raises(cws.CheckFailure):
            cws.fetch_tree("o/r", "a" * 40, Path("/tmp/unused-signals"), timeout=1, attempts=1)
        assert argvs
        assert all(argv[0] == "git" for argv in argvs)
        assert not hasattr(cws, "CODELOAD")


class TestCacheIsAuthenticatedByGit:
    """A cache entry is evidence only when git says it is the pin.

    The reviewer's probe was a cache holding one fabricated ORBIT file with the
    two cited strings. The first fix wrote a marker file beside it, which is no
    fix at all: the marker lives in the same writable directory as the content
    it vouches for, so whoever forges the tree forges the marker. git derives
    the answer from content instead, and nothing local can carry a sha over
    different bytes without a SHA-1 collision.
    """

    def test_a_hand_made_directory_is_never_reused(self, tmp_path: Path):
        """The probe, verbatim."""
        tree = tmp_path / "orbit-surgical"
        tree.mkdir()
        (tree / "fabricated.py").write_text(
            "time_out = DoneTerm(x)\nobject_dropping = DoneTerm(y)\n", encoding="utf-8"
        )
        assert "not a git repository" in cws.cache_rejection(tree, "a" * 40, timeout=60)

    def test_a_well_formed_marker_beside_the_fabrication_does_not_help(self, tmp_path: Path):
        """The probe's second round: forge the tree, then forge its receipt."""
        tree = tmp_path / "orbit-surgical"
        tree.mkdir()
        (tree / "fabricated.py").write_text("time_out = DoneTerm(x)\n", encoding="utf-8")
        for marker in (
            tree.parent / "orbit-surgical.fetch.json",
            tree / ".world-signals-fetch.json",
        ):
            marker.write_text(
                json.dumps(
                    {
                        "slug": "orbit-surgical/orbit-surgical",
                        "pin": "6e47534f7d412e4be523116f250c992a63146883",
                        "complete": True,
                    }
                ),
                encoding="utf-8",
            )
        rejection = cws.cache_rejection(
            tree, "6e47534f7d412e4be523116f250c992a63146883", timeout=60
        )
        assert "not a git repository" in rejection

    def test_a_repo_at_another_pin_is_not_reused(self, tmp_path: Path):
        sha = _repo(tmp_path / "w", {"env.py": "force = 1\n"})
        assert cws.cache_rejection(tmp_path / "w", sha, timeout=60) == ""
        assert "not the requested" in cws.cache_rejection(tmp_path / "w", "b" * 40, timeout=60)

    def test_an_untracked_addition_invalidates_reuse(self, tmp_path: Path):
        """The fabricated-cache probe *is* an untracked addition."""
        sha = _repo(tmp_path / "w", {"env.py": "force = 1\n"})
        (tmp_path / "w" / "fabricated.py").write_text("time_out = 1\n", encoding="utf-8")
        assert "worktree not clean" in cws.cache_rejection(tmp_path / "w", sha, timeout=60)

    def test_a_gitignored_addition_also_invalidates_reuse(self, tmp_path: Path):
        """``--ignored``: an attacker does not get to pick a filename that hides."""
        sha = _repo(tmp_path / "w", {"env.py": "force = 1\n", ".gitignore": "hidden/\n"})
        (tmp_path / "w" / "hidden").mkdir()
        (tmp_path / "w" / "hidden" / "env.py").write_text("time_out = 1\n", encoding="utf-8")
        assert "worktree not clean" in cws.cache_rejection(tmp_path / "w", sha, timeout=60)

    def test_an_edit_to_a_tracked_file_invalidates_reuse(self, tmp_path: Path):
        sha = _repo(tmp_path / "w", {"env.py": "force = 1\n"})
        (tmp_path / "w" / "env.py").write_text("time_out = 1\n", encoding="utf-8")
        assert "worktree not clean" in cws.cache_rejection(tmp_path / "w", sha, timeout=60)

    def test_an_unusable_cache_entry_is_refetched_by_main(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """A stale cache is normal: refetch, and fail only if the refetch fails."""
        monkeypatch.setattr(cws.shutil, "which", lambda _name: "/usr/bin/git")
        tree = tmp_path / "lapgym"
        tree.mkdir()
        (tree / "fabricated.py").write_text("x = 1\n", encoding="utf-8")
        fetched: list[str] = []

        def record(slug: str, _sha: str, _dest: Path, **_kwargs: Any) -> None:
            fetched.append(slug)
            raise cws.CheckFailure("simulated outage")

        monkeypatch.setattr(cws, "fetch_tree", record)
        assert cws.main(["lapgym", "--cache", str(tmp_path)]) == 2
        assert fetched == ["ScheiklP/sofa_env"]


_PRISTINE_REMOTE = cws.GIT_REMOTE
_REAL_LOAD_CATALOG = cws.load_catalog


def _never(*_args: Any, **_kwargs: Any) -> None:
    """A fetch that must not happen: the refusal precedes any network."""
    raise AssertionError("fetched a tree that could not be verified anyway")


def _catalog_without_absence_markers(*_args: Any, **_kwargs: Any) -> Any:
    """The catalog as it was before ``surgical-gym`` stated its empty surface."""
    catalog = _REAL_LOAD_CATALOG()
    pkg = catalog.require("surgical-gym")
    stripped = pkg.model_copy(
        update={"envs": (pkg.envs[0].model_copy(update={"absence_markers": ()}),)}
    )
    worlds = tuple(stripped if world.id == pkg.id else world for world in catalog.worlds)
    return catalog.model_copy(update={"worlds": worlds})


def _repo(root: Path, files: dict[str, str]) -> str:
    """A one-commit local git repo, and the sha of that commit."""
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for rel, body in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            "pinned",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, check=True
    )
    return head.stdout.decode().strip()


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
