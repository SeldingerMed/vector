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

The tree is fetched by content address - ``git fetch <sha>``, so git verifies
the objects it receives hash to the pin that was asked for - and then, per
audited env:

1. ``path`` exists at ``world_pin`` and is not empty;
2. a ``published`` signal's key is *published* at its cited line: assigned to
   that name, stored under that key in a dict, or assigned to that attribute.
   Occurring on the line is not enough - a comment, a docstring, a read, or an
   argument mentions a key without publishing it;
3. an unpublished signal's key occurs at its cited line *as code* rather than
   prose, because that row's claim is the opposite one: upstream computes the
   quantity and never surfaces it;
4. the env pins at least one line, whether a signal or - for a surface that is
   genuinely empty - an ``absence_markers`` reading. An env that pins nothing
   cannot be verified by fetching anything.

Fails closed. Exit ``0`` means every requested world was actually fetched and
checked; ``1`` means a citation does not resolve; ``2`` means a world could
not be checked at all (unreachable source, missing ``git``, no pin, nothing
line-pinned to check, or a named world with no audited envs at all). An
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
import ast
import os
import re
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

#: Fetch remote for a GitHub-hosted world. A template so tests can point it at
#: a local repository; substituting it cannot fabricate a pin, because the
#: fetch is content-addressed - see :func:`fetch_tree`.
GIT_REMOTE = "https://github.com/{slug}.git"

#: No credential prompt: a private or renamed repo must fail, not block.
GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}


class CheckFailure(Exception):
    """A citation could not be resolved, or its tree could not be fetched."""


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


def _git(args: list[str], tree: Path, *, timeout: int) -> str | None:
    """``git -C tree <args>`` stdout, or ``None`` when git refused."""
    done = subprocess.run(
        ["git", "-C", str(tree), *args], capture_output=True, timeout=timeout, env=GIT_ENV
    )
    return None if done.returncode != 0 else done.stdout.decode(errors="replace").strip()


def cache_rejection(tree: Path, sha: str, *, timeout: int) -> str:
    """Why this cache entry cannot be reused, or ``""`` when it can be.

    Any nonempty directory used to count as a fetched tree, so a hand-made
    ``<cache>/<world-id>`` holding one file with the cited strings in it passed
    as the verified pin. A marker file cannot fix that: it lives in the same
    writable directory as the content it vouches for, so whoever can forge the
    tree can forge the marker - a claim about a claim, which is the shape of
    the prose ``safety_evidence`` this whole check replaced.

    So git decides, from content: the entry must be a repository whose ``HEAD``
    *is* the requested sha, and whose worktree is clean including ignored
    files, so a tracked edit and an untracked fabrication both invalidate
    reuse. Nothing here is self-asserted - git derives the sha from the object
    hashes, and no locally constructed repository can carry this sha over
    different content without a SHA-1 collision.

    The residual limit, on record: an actor who can rewrite ``.git`` itself can
    make a local check agree with them (``assume-unchanged``, a rewritten
    index). SHA-1 content addressing is the trust root, not the cache
    directory's permissions - so point ``--cache`` somewhere only you can
    write, or omit it and fetch into a fresh temporary directory.

    A rejection is not a failure: the caller refetches, and only a failed
    refetch is reported as unverified.
    """
    if not tree.is_dir() or not any(tree.iterdir()):
        return "nothing cached"
    if not (tree / ".git").exists():
        return "not a git repository, so nothing ties these files to a pin"
    head = _git(["rev-parse", "HEAD"], tree, timeout=timeout)
    if head is None:
        return "no HEAD: no completed checkout"
    if head != sha:
        return f"HEAD is {head[:12]}, not the requested {sha[:12]}"
    dirty = _git(["status", "--porcelain", "--ignored"], tree, timeout=timeout)
    if dirty is None:
        return "git could not read the worktree state"
    if dirty:
        entries = dirty.splitlines()
        return f"worktree not clean: {len(entries)} change(s), e.g. {entries[0].strip()!r}"
    return ""


def fetch_tree(slug: str, sha: str, dest: Path, *, timeout: int, attempts: int = 2) -> None:
    """Check out one pinned tree by content address, or raise git's own message.

    ``git fetch <url> <sha>`` is the whole argument: git verifies that the
    objects it receives hash to the SHA that was asked for, so a substituted
    response cannot be labelled as this pin. The previous ``curl | tar`` used
    ``sha`` only to build a URL and extracted whatever bytes came back, which
    meant an unrelated tarball verified as the pin - and handed a hostile
    tarball an extraction surface on the way.

    ``dest`` is emptied first, because a leftover file that git's checkout does
    not overwrite would sit in the tree that is about to be read as evidence.
    ``core.symlinks`` is written to the repo config rather than passed to one
    command, so that a cited path cannot traverse a symlink out of the tree
    *and* a later ``git status`` agrees with how the worktree was written.

    Retries once by default. A bounded retry is not fail-open: it reduces
    false alarms from a registry hiccup while still requiring the fetch to
    succeed before anything is called verified.
    """
    url = GIT_REMOTE.format(slug=slug)
    steps = (
        ("init", ["git", "init", "-q", str(dest)]),
        ("config", ["git", "-C", str(dest), "config", "core.symlinks", "false"]),
        (
            "fetch",
            ["git", "-C", str(dest), "fetch", "-q", "--depth", "1", "--no-tags", url, sha],
        ),
        ("checkout", ["git", "-C", str(dest), "checkout", "-q", "FETCH_HEAD"]),
    )
    last = ""
    for _attempt in range(attempts):
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)
        last = ""
        for label, argv in steps:
            done = subprocess.run(argv, capture_output=True, timeout=timeout, env=GIT_ENV)
            if done.returncode != 0:
                last = f"{label} failed: {done.stderr.decode(errors='replace')[:200]}"
                break
        if last:
            continue
        head = _git(["rev-parse", "HEAD"], dest, timeout=timeout)
        if head != sha:
            last = f"checked out {head or '(unknown)'}, not the requested pin"
            continue
        return
    raise CheckFailure(f"{slug}@{sha[:12]} after {attempts} attempt(s): {last}")


def _dict_key_lines(node: ast.AST, key: str) -> set[int]:
    """Lines where ``key`` is a literal key of a dict inside ``node``."""
    return {
        item.lineno
        for child in ast.walk(node)
        if isinstance(child, ast.Dict)
        for item in child.keys
        if isinstance(item, ast.Constant) and item.value == key
    }


def _subscript_key_lines(node: ast.AST, key: str) -> set[int]:
    """Lines where ``key`` indexes a subscript inside ``node``, read or write."""
    lines: set[int] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript):
            index = child.slice
            if isinstance(index, ast.Constant) and index.value == key:
                lines.add(index.lineno)
    return lines


def _store_lines(target: ast.expr, key: str) -> set[int]:
    """Lines where an assignment target stores into ``key``.

    Three shapes, all of which the catalog really cites: the bare name
    (``time_out = DoneTerm(...)``), the attribute (``self.simulation_error =
    True``), and the dict key (``self.extras["cost"] = ...``). The store
    context is what matters, so ``info[force] = 1`` does not publish ``force``.
    """
    lines: set[int] = set()
    for node in ast.walk(target):
        if isinstance(node, ast.Name | ast.Attribute) and isinstance(node.ctx, ast.Store):
            spelled = node.id if isinstance(node, ast.Name) else node.attr
            if spelled == key:
                lines.add(node.lineno)
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            index = node.slice
            if isinstance(index, ast.Constant) and index.value == key:
                lines.add(index.lineno)
    return lines


def publication_lines(module: ast.Module, key: str) -> set[int]:
    """Lines where ``key`` is published, not merely mentioned.

    ``docs/CONFORMANCE.md`` says a resolved citation proves the key is
    *assigned and published to* ``info``/``extras``. Occurrence cannot prove
    that: ``# time_out``, a lone ``'time_out'`` string literal, ``if
    time_out:`` and ``log(time_out)`` all put the key on the line without
    publishing it. So the cited line has to carry an assignment, an augmented
    assignment, an attribute store, a dict-key store, or a literal dict key in
    something returned, assigned, or handed to a call
    (``info.update({...})``).

    What this does *not* establish, stated so the boundary is on record: that
    the container reached ``info``/``extras``. Proving that needs dataflow -
    upstream really writes ``reward_features[...]`` and ``self.extras[...]``
    and ``self.info = {...}``, so a whitelist of container names would refuse
    most of the catalog's true citations. The machine checks that the key is
    *stored under that key at that line*; which mapping is published is the
    first-hand read the catalog records.
    """
    lines: set[int] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                lines |= _store_lines(target, key)
            if node.value is not None:
                lines |= _dict_key_lines(node.value, key)
        elif isinstance(node, ast.Return) and node.value is not None:
            lines |= _dict_key_lines(node.value, key)
        elif isinstance(node, ast.Call):
            for argument in [*node.args, *(kw.value for kw in node.keywords)]:
                lines |= _dict_key_lines(argument, key)
    return lines


def _identifier_lines(node: ast.AST, key: str) -> set[int]:
    """Lines where ``key`` is spelled as an identifier inside ``node``."""
    lines: set[int] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            spelled = child.id
        elif isinstance(child, ast.Attribute):
            spelled = child.attr
        elif isinstance(child, ast.keyword | ast.arg):
            spelled = child.arg or ""
        elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            spelled = child.name
        else:
            continue
        if spelled == key:
            lines.add(child.lineno)
    return lines


def code_lines(module: ast.Module, key: str) -> set[int]:
    """Lines where ``key`` occurs as code rather than as prose.

    The rule for an *unpublished* row, whose claim is the opposite of a
    publication: upstream computes the quantity and never surfaces it. SurRoL
    calls ``p.getContactPoints(...)`` and discards the result, and Surgical
    Gym reads ``self.progress_buf`` to terminate; demanding a publication
    shape would refuse exactly the rows that record the honest negative. A
    docstring or a bare literal still resolves nothing, because a string is
    only code here when it is a dict key or a subscript index.
    """
    return (
        _dict_key_lines(module, key)
        | _subscript_key_lines(module, key)
        | _identifier_lines(module, key)
    )


def evidence_lines(text: str, key: str, *, python: bool, published: bool) -> list[int]:
    """1-based lines that carry evidence for ``key``, strongest rule first.

    Raises:
        CheckFailure: the file does not parse as Python. An unreadable
            citation is an unchecked one, and unchecked is not verified.
    """
    if not python:
        # No lexer is right for every non-Python file; a word boundary is at
        # least stricter than a substring.
        word = re.compile(rf"(?<![0-9A-Za-z_]){re.escape(key)}(?![0-9A-Za-z_])")
        return [n for n, line in enumerate(text.splitlines(), 1) if word.search(line)]
    try:
        module = ast.parse(text)
    except (SyntaxError, ValueError) as exc:
        raise CheckFailure(f"does not parse as Python ({exc})") from exc
    found = publication_lines(module, key) if published else code_lines(module, key)
    return sorted(found)


def check_signal(
    tree: Path, env: AuditedEnv, signal: WorldSignal, *, label: str = "signal"
) -> list[str]:
    """Problems with one citation. Empty means it resolved."""
    rel = signal.path or env.path
    target = tree / rel
    if not target.is_file():
        hits = [str(p.relative_to(tree)) for p in tree.rglob(Path(rel).name)][:3]
        hint = f" (basename found at: {', '.join(hits)})" if hits else ""
        return [f"{label} {signal.key!r}: path {rel!r} does not exist at the pin{hint}"]

    text = target.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return [f"{label} {signal.key!r}: cited file {rel!r} is empty at the pin"]
    shape = "a publication" if signal.published else "code"
    try:
        found = evidence_lines(
            text, signal.key, python=target.suffix == ".py", published=signal.published
        )
    except CheckFailure as exc:
        return [f"{label} {signal.key!r}: {rel} {exc}"]

    if not found:
        return [f"{label} {signal.key!r}: key never appears in {rel} as {shape}"]
    if signal.line:
        total = len(text.splitlines())
        if signal.line > total:
            return [f"{label} {signal.key!r}: cites line {signal.line} but {rel} has {total}"]
        if signal.line not in found:
            return [
                f"{label} {signal.key!r}: not {shape} on cited line {signal.line} of {rel} "
                f"(appears as {shape} on {found[:4]})"
            ]
    return []


def unpinned_envs(pkg: WorldPackage) -> list[str]:
    """Envs that pin no line, so no fetch can verify them.

    Not a citation failure - nothing is claimed wrongly - and not a pass
    either: ``surgical-gym`` has one audited env with no signals, and a
    zero-byte cached file used to make it report "0 signal(s) resolved" after
    reading not one line. An honestly empty signal surface is stated with
    ``absence_markers``, never by omission.
    """
    return [
        f"env {env.env_id!r} pins no line: no signal and no absence_marker to check, so "
        "fetching the tree would verify nothing. Record the reward / termination sites "
        "the env was read at in absence_markers"
        for env in pkg.envs
        if not any(cited.line for cited in (*env.signals, *env.absence_markers))
    ]


def check_world(pkg: WorldPackage, tree: Path) -> list[str]:
    """Every citation problem for one world."""
    problems: list[str] = []
    for env in pkg.envs:
        target = tree / env.path
        if not target.is_file():
            hits = [str(p.relative_to(tree)) for p in tree.rglob(Path(env.path).name)][:3]
            hint = f" (basename found at: {', '.join(hits)})" if hits else ""
            problems.append(f"env {env.env_id!r}: path {env.path!r} missing at pin{hint}")
            continue
        if target.stat().st_size == 0:
            problems.append(f"env {env.env_id!r}: path {env.path!r} is empty at the pin")
            continue
        cited: list[tuple[WorldSignal, str]] = [(s, "signal") for s in env.signals]
        cited += [(m, "absence marker") for m in env.absence_markers]
        for signal, label in cited:
            problems.extend(
                f"env {env.env_id!r}: {p}" for p in check_signal(tree, env, signal, label=label)
            )
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
    if args.worlds:
        audited = [catalog.require(world_id) for world_id in args.worlds]
        # The default selection may skip an unaudited world; a named one is a
        # request to verify something, and here there is nothing to verify.
        unaudited = [pkg.id for pkg in audited if not pkg.envs]
        if unaudited:
            print(
                f"USAGE: no audited envs for {', '.join(unaudited)}, so naming them asks "
                "this check to verify nothing at all. Audit the world first, or drop the "
                "name to check every audited world."
            )
            return 2
    else:
        audited = [pkg for pkg in catalog.worlds if pkg.envs]
    if not audited:
        print("UNVERIFIED: no audited worlds to check, so this run establishes nothing.")
        return 2

    if shutil.which("git") is None:
        print(
            "UNVERIFIED: git not found, so no citation could be checked against a pinned "
            f"tree. {len(audited)} world(s) went unverified. Install git and re-run; this "
            "is reported as a failure because an unchecked claim is not a checked one."
        )
        return 2

    holder = args.cache or Path(tempfile.mkdtemp(prefix="world-signals-"))
    holder.mkdir(parents=True, exist_ok=True)
    failures: dict[str, list[str]] = {}
    unverified: dict[str, str] = {}
    verified: list[str] = []

    try:
        for pkg in audited:
            slug = repo_slug(pkg)
            if slug is None:
                failures[pkg.id] = ["no GitHub repo in install spec or sources"]
                continue
            if not pkg.world_pin:
                failures[pkg.id] = ["no world_pin: a citation cannot be checked against nothing"]
                continue
            gaps = unpinned_envs(pkg)
            if gaps:
                unverified[pkg.id] = "; ".join(gaps)
                print(f"UNVERIFIED {pkg.id}: {unverified[pkg.id]}")
                continue
            tree = holder / pkg.id
            try:
                rejection = cache_rejection(tree, pkg.world_pin, timeout=args.timeout)
                if rejection:
                    if tree.exists():
                        print(f"     {pkg.id}: refetching, cache unusable ({rejection})")
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
                markers = sum(len(env.absence_markers) for env in pkg.envs)
                absence = f", {markers} absence marker(s)" if markers else ""
                print(
                    f"ok   {pkg.id} ({slug}@{pkg.world_pin[:12]}): "
                    f"{len(pkg.envs)} env(s), {signals} signal(s) resolved{absence}"
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
            f"\n{len(unverified)} of {len(audited)} world(s) could not be checked "
            f"({', '.join(sorted(unverified))}), so this run verifies nothing about them. "
            "Re-run when the source is reachable."
        )
        return 2
    if len(verified) != len(audited):
        missing = sorted({pkg.id for pkg in audited} - set(verified))
        print(f"\nInternal error: {missing} were neither verified nor reported. Refusing to pass.")
        return 2
    print(f"\nAll citations resolved for {len(verified)} of {len(audited)} world(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
