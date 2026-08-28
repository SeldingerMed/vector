"""Verify every audited env citation resolves at its world's pin.

This is the check whose absence let two real defects ship in the 2026-08
catalog, and both were the same shape - a citation that reads as evidence but
points at code the pinned revision does not contain:

* ``surrol`` cited ``surrol/gym/surrol_env.py`` while pinned to the default
  branch, a VPPV research monorepo with six divergent vendored copies of
  ``surrol/`` and no top-level package. The pin did not identify a world.
* ``sonogym`` cited ``robot_US_guided_surgery.py``, transcribing the directory
  name; the file is ``robotic_US_guided_surgery.py``.

A prose ``safety_evidence`` field cannot be checked by a machine, so it was
never checked at all. A path plus a line plus a key can be, and that is the
whole point of :class:`or_audit.install.catalog.AuditedEnv`.

The check fetches each pinned tree from codeload and asserts, per audited env:

1. ``path`` exists at ``world_pin``;
2. every signal's key appears in its file;
3. when a signal names a ``line``, the key appears on it.

Fails closed. Exit ``0`` means every requested world was actually fetched and
checked; ``1`` means a citation does not resolve; ``2`` means a world could
not be checked at all (unreachable source, missing ``curl``, no pin). An
unverified target is never reported as a pass, because the only thing this
script exists to provide is the assurance that the claims were checked - and
it runs on a schedule, never as a PR gate, so failing costs nothing but
attention.

Usage::

    python scripts/check_world_signals.py            # every audited world
    python scripts/check_world_signals.py surrol      # one world
    python scripts/check_world_signals.py --cache DIR # reuse fetched trees
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from or_audit.install.catalog import (  # noqa: E402
    AuditedEnv,
    WorldPackage,
    WorldSignal,
    load_catalog,
)

CODELOAD = "https://codeload.github.com/{slug}/tar.gz/{sha}"


class CheckFailure(Exception):
    """A citation does not resolve at the pin."""


def repo_slug(pkg: WorldPackage) -> str | None:
    """``owner/name`` for a GitHub-hosted world, or ``None``.

    Prefers the install spec's repo, because that is the revision the plan
    actually fetches; falls back to the first GitHub source reference for
    vendor-runtime rows, whose code lives on GitHub while their runtime does
    not.
    """
    candidates = [getattr(pkg.install, "repo", "") or "", *pkg.source]
    for ref in candidates:
        if "github.com" not in ref:
            continue
        parts = [p for p in urlparse(ref).path.split("/") if p]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1].removesuffix('.git')}"
    return None


def fetch_tree(slug: str, sha: str, dest: Path, *, timeout: int, attempts: int = 2) -> None:
    """Extract one pinned tarball, or raise with the transport's own message.

    Retries once by default. A bounded retry is not fail-open: it reduces
    false alarms from a registry hiccup while still requiring the fetch to
    succeed before anything is called verified.
    """
    dest.mkdir(parents=True, exist_ok=True)
    url = CODELOAD.format(slug=slug, sha=sha)
    last = ""
    for _attempt in range(attempts):
        curl = subprocess.run(
            ["curl", "-sSL", "--fail", "--max-time", str(timeout), url],
            capture_output=True,
            timeout=timeout + 30,
        )
        if curl.returncode != 0:
            last = f"fetch failed: {curl.stderr.decode()[:200]}"
            continue
        tar = subprocess.run(
            ["tar", "xz", "-C", str(dest), "--strip-components=1"],
            input=curl.stdout,
            capture_output=True,
            timeout=timeout + 30,
        )
        if tar.returncode != 0:
            last = f"extract failed: {tar.stderr.decode()[:200]}"
            continue
        return
    raise CheckFailure(f"{slug}@{sha[:12]} after {attempts} attempt(s): {last}")


def check_signal(tree: Path, env: AuditedEnv, signal: WorldSignal) -> list[str]:
    """Problems with one signal's citation. Empty means it resolved."""
    rel = signal.path or env.path
    target = tree / rel
    if not target.is_file():
        hits = [str(p.relative_to(tree)) for p in tree.rglob(Path(rel).name)][:3]
        hint = f" (basename found at: {', '.join(hits)})" if hits else ""
        return [f"signal {signal.key!r}: path {rel!r} does not exist at the pin{hint}"]

    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    problems: list[str] = []
    if not any(signal.key in line for line in lines):
        problems.append(f"signal {signal.key!r}: key never appears in {rel}")
        return problems
    if signal.line:
        if signal.line > len(lines):
            problems.append(
                f"signal {signal.key!r}: cites line {signal.line} but {rel} has {len(lines)}"
            )
        elif signal.key not in lines[signal.line - 1]:
            found = [n for n, line in enumerate(lines, 1) if signal.key in line][:4]
            problems.append(
                f"signal {signal.key!r}: not on cited line {signal.line} of {rel} "
                f"(appears on {found})"
            )
    return problems


def check_world(pkg: WorldPackage, tree: Path) -> list[str]:
    """Every citation problem for one world."""
    problems: list[str] = []
    for env in pkg.envs:
        if not (tree / env.path).is_file():
            hits = [str(p.relative_to(tree)) for p in tree.rglob(Path(env.path).name)][:3]
            hint = f" (basename found at: {', '.join(hits)})" if hits else ""
            problems.append(f"env {env.env_id!r}: path {env.path!r} missing at pin{hint}")
            continue
        for signal in env.signals:
            problems.extend(f"env {env.env_id!r}: {p}" for p in check_signal(tree, env, signal))
    return problems


def main(argv: list[str] | None = None) -> int:
    """Verify every requested world, or fail.

    Exit codes: ``0`` only when every requested world was actually verified,
    ``1`` when a citation does not resolve, ``2`` when a world could not be
    checked at all. The last one is the point of this function's shape: an
    earlier version printed ``SKIP`` for a fetch failure and still returned
    0 as long as some other world passed, so a registry outage could have
    reported success while checking none of the claims. This job is scheduled
    and manual, never a PR gate, so there is nothing to protect by going
    green on an unverified target - and "unverified" is not "verified".
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("worlds", nargs="*", help="world ids (default: every audited world)")
    parser.add_argument("--cache", type=Path, help="reuse/keep fetched trees here")
    parser.add_argument("--timeout", type=int, default=300, help="per-fetch seconds")
    parser.add_argument("--attempts", type=int, default=2, help="fetch attempts per world (2)")
    args = parser.parse_args(argv)

    catalog = load_catalog()
    wanted = args.worlds or [pkg.id for pkg in catalog.worlds if pkg.envs]
    audited = [catalog.require(world_id) for world_id in wanted]
    expected = [pkg for pkg in audited if pkg.envs]

    if shutil.which("curl") is None:
        print(
            "UNVERIFIED: curl not found, so no citation could be checked against a pinned "
            f"tree. {len(expected)} world(s) went unverified. Install curl and re-run; this "
            "is reported as a failure because an unchecked claim is not a checked one."
        )
        return 2 if expected else 0

    holder = args.cache or Path(tempfile.mkdtemp(prefix="world-signals-"))
    holder.mkdir(parents=True, exist_ok=True)
    failures: dict[str, list[str]] = {}
    unverified: dict[str, str] = {}
    verified: list[str] = []

    try:
        for pkg in audited:
            if not pkg.envs:
                print(f"{pkg.id}: no audited envs; nothing to check")
                continue
            slug = repo_slug(pkg)
            if slug is None:
                failures[pkg.id] = ["no GitHub repo in install spec or sources"]
                continue
            if not pkg.world_pin:
                failures[pkg.id] = ["no world_pin: a citation cannot be checked against nothing"]
                continue
            tree = holder / pkg.id
            try:
                if not tree.exists() or not any(tree.iterdir()):
                    fetch_tree(
                        slug, pkg.world_pin, tree, timeout=args.timeout, attempts=args.attempts
                    )
            except (CheckFailure, subprocess.TimeoutExpired) as exc:
                unverified[pkg.id] = str(exc)
                print(f"UNVERIFIED {pkg.id}: {exc}")
                continue
            problems = check_world(pkg, tree)
            if problems:
                failures[pkg.id] = problems
                print(f"FAIL {pkg.id} ({slug}@{pkg.world_pin[:12]})")
                for problem in problems:
                    print(f"  - {problem}")
            else:
                verified.append(pkg.id)
                signals = sum(len(env.signals) for env in pkg.envs)
                print(
                    f"ok   {pkg.id} ({slug}@{pkg.world_pin[:12]}): "
                    f"{len(pkg.envs)} env(s), {signals} signal(s) resolved"
                )
    finally:
        if args.cache is None:
            shutil.rmtree(holder, ignore_errors=True)

    if failures:
        print(f"\n{len(failures)} world(s) carry citations that do not resolve at their pin.")
        if unverified:
            print(f"{len(unverified)} further world(s) could not be checked: {sorted(unverified)}")
        return 1
    if unverified:
        print(
            f"\n{len(unverified)} of {len(expected)} world(s) could not be checked "
            f"({', '.join(sorted(unverified))}), so this run verifies nothing about them. "
            "Re-run when the source is reachable."
        )
        return 2
    if len(verified) != len(expected):
        missing = sorted({pkg.id for pkg in expected} - set(verified))
        print(f"\nInternal error: {missing} were neither verified nor reported. Refusing to pass.")
        return 2
    print(f"\nAll citations resolved for {len(verified)} of {len(expected)} world(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
