"""SPDX license classification for the curated artifact and its runtime closure.

Two consumers, one table:

* ``scripts/check_license_allowlist.py`` gates SurgEval's own *runtime*
  dependency closure, because the harness ships as open-core with commercial
  and attestation tiers above it;
* the Tier-1 conformance suite (:mod:`or_audit.eval.conformance`) gates every
  *wrapped world* we curate, because "some envs carry restrictive or
  contaminating licenses" is a real property of this catalog (next.md §2.1) and
  a per-env license audit is part of the artifact, not a chore beside it.

The two share :data:`ALLOWLIST` deliberately: a license we would not accept in
our own wheel is not a license we can redistribute a wrapped world under.

Three verdicts, never two. ``restricted`` is a reviewed refusal with a named
reason; ``unknown`` is the honest state for an identifier nobody has reviewed,
and it is a refusal too — guessing that an unrecognized identifier is
permissive is exactly how contamination enters a curated catalog. Adding an
identifier to :data:`ALLOWLIST` or :data:`DENYLIST` is the intended governance
mechanism: it happens in review, not at runtime.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from or_audit.errors import TaskContractError

#: name -> SPDX license expression for SurgEval's own runtime closure. A
#: dependency's *absence here is a hard failure*: adding a runtime dependency
#: without an explicit reviewed entry is rejected, forcing a deliberate review.
THIRD_PARTY: dict[str, str] = {
    "pydantic": "MIT",
    "pydantic-core": "MIT",
    "annotated-types": "MIT",
    "typing-extensions": "PSF-2.0",
    "typing-inspection": "MIT",
    "numpy": "BSD-3-Clause AND MIT AND 0BSD AND Zlib AND CC0-1.0",
    "cloudpickle": "BSD-3-Clause",
}

#: Licenses acceptable for commercial redistribution (permissive, sublicensable).
ALLOWLIST: frozenset[str] = frozenset(
    {
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "Apache-2.0",
        "PSF-2.0",
        "0BSD",
        "Zlib",
        "ISC",
        "CC0-1.0",
    }
)

#: Reviewed refusals: SPDX identifier -> why redistribution is blocked. These
#: are the licenses that make an otherwise attractive world unshippable inside
#: a curated artifact; the reason names the class so the fix is obvious.
DENYLIST: Mapping[str, str] = {
    # Reciprocal copyleft: derived and linked work inherits the license.
    "GPL-2.0": "reciprocal copyleft",
    "GPL-2.0-only": "reciprocal copyleft",
    "GPL-2.0-or-later": "reciprocal copyleft",
    "GPL-3.0": "reciprocal copyleft",
    "GPL-3.0-only": "reciprocal copyleft",
    "GPL-3.0-or-later": "reciprocal copyleft",
    # Network copyleft: reaches a hosted control plane, not just a redistribution.
    "AGPL-3.0": "network copyleft",
    "AGPL-3.0-only": "network copyleft",
    "AGPL-3.0-or-later": "network copyleft",
    # Weak copyleft: still constrains how the artifact may be combined/shipped.
    "LGPL-2.0": "weak copyleft",
    "LGPL-2.0-only": "weak copyleft",
    "LGPL-2.0-or-later": "weak copyleft",
    "LGPL-2.1": "weak copyleft",
    "LGPL-2.1-only": "weak copyleft",
    "LGPL-2.1-or-later": "weak copyleft",
    "LGPL-3.0": "weak copyleft",
    "LGPL-3.0-only": "weak copyleft",
    "LGPL-3.0-or-later": "weak copyleft",
    # File-level copyleft: dev/build only, never in a redistributed runtime.
    "MPL-1.1": "file-level copyleft",
    "MPL-2.0": "file-level copyleft",
    "MPL-2.0-no-copyleft-exception": "file-level copyleft",
    "EPL-1.0": "file-level copyleft",
    "EPL-2.0": "file-level copyleft",
    "CDDL-1.0": "file-level copyleft",
    "CDDL-1.1": "file-level copyleft",
    "OSL-3.0": "reciprocal copyleft",
    "QPL-1.0": "reciprocal copyleft",
    # Source-available, not open: field-of-use and non-compete restrictions.
    "SSPL-1.0": "source-available with service restrictions",
    "BUSL-1.1": "source-available with a non-compete term",
    "Elastic-2.0": "source-available with a non-compete term",
    "CAL-1.0": "source-available with data-sharing obligations",
    # Non-commercial / share-alike content licenses. Common on medical
    # datasets and simulator assets, and fatal to a commercial catalog.
    "CC-BY-NC-3.0": "non-commercial use only",
    "CC-BY-NC-4.0": "non-commercial use only",
    "CC-BY-NC-SA-3.0": "non-commercial use only, share-alike",
    "CC-BY-NC-SA-4.0": "non-commercial use only, share-alike",
    "CC-BY-NC-ND-3.0": "non-commercial use only, no derivatives",
    "CC-BY-NC-ND-4.0": "non-commercial use only, no derivatives",
    "CC-BY-ND-4.0": "no derivatives",
    "CC-BY-SA-3.0": "share-alike",
    "CC-BY-SA-4.0": "share-alike",
    # Explicit non-answers. SPDX says "we did not look"; treating either as
    # permissive is the fabrication this module exists to prevent.
    "NONE": "no license declared",
    "NOASSERTION": "no license conclusion asserted",
}

#: Package names that must not appear in the runtime allowlist table; these are
#: not part of the shipped runtime and are handled separately.
REPO_PACKAGE = "surgeval"

_ALLOWED_BY_KEY: Mapping[str, str] = {spdx.lower(): spdx for spdx in ALLOWLIST}
_DENIED_BY_KEY: Mapping[str, tuple[str, str]] = {
    spdx.lower(): (spdx, reason) for spdx, reason in DENYLIST.items()
}


class LicenseStatus(StrEnum):
    """Verdict classes. ``unknown`` is a refusal, not a shrug."""

    #: Permissive and sublicensable: redistributable inside the artifact.
    ALLOWED = "allowed"
    #: Reviewed and refused: copyleft, non-commercial, or field-of-use limited.
    RESTRICTED = "restricted"
    #: Not reviewed. Never assumed permissive.
    UNKNOWN = "unknown"


class LicenseVerdict(BaseModel):
    """One classification of one declared SPDX expression."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The expression as declared, whitespace-normalized (empty when absent).
    spdx: str
    status: LicenseStatus
    #: Why, in terms the package author can act on.
    reason: str


class LicenseAudit(BaseModel):
    """Outcome of auditing a resolved runtime dependency closure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Sorted dependency closure that was audited (project package excluded).
    dependencies: tuple[str, ...] = ()
    #: Human-readable governance failures; empty means the gate passes.
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the closure cleared the gate."""
        return not self.errors


def _classify_term(term: str) -> LicenseVerdict:
    """Classify a single SPDX identifier (no operators)."""
    key = term.lower()
    allowed = _ALLOWED_BY_KEY.get(key)
    if allowed is not None:
        return LicenseVerdict(
            spdx=term,
            status=LicenseStatus.ALLOWED,
            reason=f"{allowed} is on the reviewed permissive allowlist",
        )
    denied = _DENIED_BY_KEY.get(key)
    if denied is not None:
        name, why = denied
        return LicenseVerdict(
            spdx=term,
            status=LicenseStatus.RESTRICTED,
            reason=(
                f"{name} is {why}; it cannot be redistributed inside the curated "
                "artifact. Relicense upstream, isolate the world behind an adapter "
                "package that redistributes none of its code, or drop the wrap"
            ),
        )
    return LicenseVerdict(
        spdx=term,
        status=LicenseStatus.UNKNOWN,
        reason=(
            f"{term!r} is not a reviewed SPDX identifier; an unreviewed license is "
            "never assumed permissive. Declare the package's real SPDX id, or add "
            "the identifier to or_audit.eval.licensing.ALLOWLIST / DENYLIST in review"
        ),
    )


class _ParseError(ValueError):
    """The declaration is not a well-formed SPDX expression. Refused, not guessed."""


#: Tokens that can never stand in for a license identifier.
_RESERVED: frozenset[str] = frozenset({"AND", "OR", "WITH", "(", ")"})

#: Nesting limit for parenthesised groups. A declaration is third-party data,
#: so the parser must refuse pathological input rather than exhaust the stack:
#: a classifier that dies on a hostile string is its own kind of fail-open. No
#: real SPDX expression nests anywhere near this deep.
_MAX_DEPTH = 32


def _tokenize(expression: str) -> list[str]:
    return expression.replace("(", " ( ").replace(")", " ) ").split()


def _render(tokens: Sequence[str]) -> str:
    """Re-join a token slice so a sub-verdict quotes the sub-expression as written."""
    text = ""
    for token in tokens:
        if not text or token == ")" or text.endswith("("):
            text += token
        else:
            text += f" {token}"
    return text


def _classify_id(token: str) -> LicenseVerdict:
    """Classify one license id, honouring the ``+`` (or-later) suffix.

    ``+`` widens the grant to every later version of the same license. A denied
    family stays denied — no later GPL stops being copyleft — but a permissive
    base cannot carry its clearance forward onto a version nobody has read.
    """
    if not token.endswith("+"):
        return _classify_term(token)
    base = _classify_term(token[:-1])
    if base.status is LicenseStatus.RESTRICTED:
        return base.model_copy(update={"spdx": token})
    return LicenseVerdict(
        spdx=token,
        status=LicenseStatus.UNKNOWN,
        reason=(
            f"{token!r} admits any later version of {token[:-1]!r}, and later versions are "
            "not reviewed here; declare the exact SPDX id the package ships under"
        ),
    )


def _classify_with(base: LicenseVerdict, exception: str, expression: str) -> LicenseVerdict:
    """Apply a ``WITH`` exception against the (empty) table of reviewed exceptions.

    An SPDX exception grants additional permission, but *which* permission is a
    property of its text and no exception has been reviewed here. So a refused
    base stays refused — naming the exception, since reviewing it is the way
    out — and a permissive base drops to ``unknown`` rather than inheriting a
    clearance that was granted to the bare license, not to this variant.
    """
    if base.status is LicenseStatus.RESTRICTED:
        return LicenseVerdict(
            spdx=expression,
            status=LicenseStatus.RESTRICTED,
            reason=(
                f"{base.reason}. The {exception!r} exception is not reviewed here, so it "
                "cannot lift that refusal"
            ),
        )
    return LicenseVerdict(
        spdx=expression,
        status=LicenseStatus.UNKNOWN,
        reason=(
            f"{exception!r} is not a reviewed SPDX license exception, and an exception "
            f"nobody has read may not be assumed harmless to {base.spdx!r}; declare the "
            "bare license, or add the exception in review"
        ),
    )


def _classify_any_of(expression: str, verdicts: list[LicenseVerdict]) -> LicenseVerdict:
    """A dual license: the licensee picks, so one permissive alternative suffices."""
    allowed = next((v for v in verdicts if v.status is LicenseStatus.ALLOWED), None)
    if allowed is not None:
        return LicenseVerdict(
            spdx=expression,
            status=LicenseStatus.ALLOWED,
            reason=f"dual-licensed; {allowed.reason}",
        )
    restricted = [v for v in verdicts if v.status is LicenseStatus.RESTRICTED]
    if len(restricted) == len(verdicts):
        return LicenseVerdict(
            spdx=expression,
            status=LicenseStatus.RESTRICTED,
            reason="every alternative is restricted: " + "; ".join(v.reason for v in restricted),
        )
    unreviewed = next(v for v in verdicts if v.status is LicenseStatus.UNKNOWN)
    return LicenseVerdict(
        spdx=expression,
        status=LicenseStatus.UNKNOWN,
        reason=f"no reviewed permissive alternative; {unreviewed.reason}",
    )


def _classify_all_of(expression: str, verdicts: list[LicenseVerdict]) -> LicenseVerdict:
    """Conjunctive terms: every obligation applies, so every term must clear."""
    restricted = [v for v in verdicts if v.status is LicenseStatus.RESTRICTED]
    if restricted:
        return LicenseVerdict(
            spdx=expression,
            status=LicenseStatus.RESTRICTED,
            reason="; ".join(v.reason for v in restricted),
        )
    unreviewed = [v for v in verdicts if v.status is LicenseStatus.UNKNOWN]
    if unreviewed:
        return LicenseVerdict(
            spdx=expression,
            status=LicenseStatus.UNKNOWN,
            reason="; ".join(v.reason for v in unreviewed),
        )
    return LicenseVerdict(
        spdx=expression,
        status=LicenseStatus.ALLOWED,
        reason=(
            "every term is on the reviewed permissive allowlist: "
            + ", ".join(v.spdx for v in verdicts)
        ),
    )


class _Parser:
    """Recursive descent over the SPDX expression grammar, classifying as it goes.

    Grammar (SPDX Annex D, restricted to the forms declarations actually use)::

        alternatives := conjunction ("OR" conjunction)*
        conjunction  := operand ("AND" operand)*
        operand      := "(" alternatives ")" | id ["WITH" exception]
        id           := <license identifier> ["+"]

    Grouping is the whole point: ``AND`` binds tighter than ``OR``, so
    ``A AND (B OR C)`` and ``A AND B OR C`` are different licenses, and
    flattening the parentheses away turns the first into the second — which is
    how a copyleft term that applies under *every* choice gets read as one
    optional branch among several.
    """

    def __init__(self, tokens: Sequence[str]) -> None:
        self._tokens = tokens
        self._pos = 0
        self._depth = 0

    def parse(self) -> LicenseVerdict:
        verdict = self._alternatives()
        if self._pos < len(self._tokens):
            raise _ParseError(f"unexpected {self._tokens[self._pos]!r}")
        return verdict

    def _peek(self) -> str | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _alternatives(self) -> LicenseVerdict:
        start = self._pos
        verdicts = [self._conjunction()]
        while self._peek() == "OR":
            self._pos += 1
            verdicts.append(self._conjunction())
        if len(verdicts) == 1:
            return verdicts[0]
        return _classify_any_of(_render(self._tokens[start : self._pos]), verdicts)

    def _conjunction(self) -> LicenseVerdict:
        start = self._pos
        verdicts = [self._operand()]
        while self._peek() == "AND":
            self._pos += 1
            verdicts.append(self._operand())
        if len(verdicts) == 1:
            return verdicts[0]
        return _classify_all_of(_render(self._tokens[start : self._pos]), verdicts)

    def _operand(self) -> LicenseVerdict:
        start = self._pos
        token = self._peek()
        if token is None:
            raise _ParseError("the expression ends where a license identifier is required")
        if token == "(":
            if self._depth >= _MAX_DEPTH:
                raise _ParseError(f"nested more than {_MAX_DEPTH} groups deep")
            self._pos += 1
            self._depth += 1
            inner = self._alternatives()
            self._depth -= 1
            if self._peek() != ")":
                raise _ParseError("unbalanced '('")
            self._pos += 1
            return inner.model_copy(update={"spdx": _render(self._tokens[start : self._pos])})
        if token in _RESERVED:
            raise _ParseError(f"expected a license identifier, found {token!r}")
        self._pos += 1
        verdict = _classify_id(token)
        if self._peek() != "WITH":
            return verdict
        self._pos += 1
        exception = self._peek()
        if exception is None or exception in _RESERVED:
            raise _ParseError("'WITH' is not followed by an exception identifier")
        self._pos += 1
        return _classify_with(verdict, exception, _render(self._tokens[start : self._pos]))


def classify_license(spdx: str) -> LicenseVerdict:
    """Classify a declared SPDX expression against the reviewed tables.

    The expression is parsed, not split: composition follows SPDX meaning with
    the declared structure intact. An ``AND`` node is allowed only when *every*
    operand is (all obligations apply at once), an ``OR`` node as soon as *one*
    operand is (the licensee chooses), and parentheses override the default
    precedence rather than being discarded. Anything undecided by a restricted
    or permissive operand resolves to ``unknown``; an expression that does not
    parse is ``unknown`` too, never allowed.
    """
    expression = " ".join(spdx.split())
    if not expression:
        return LicenseVerdict(
            spdx="",
            status=LicenseStatus.UNKNOWN,
            reason=(
                "no SPDX license identifier is declared; a curated package must state "
                "its license explicitly"
            ),
        )
    try:
        verdict = _Parser(_tokenize(expression)).parse()
    except _ParseError as exc:
        return LicenseVerdict(
            spdx=expression,
            status=LicenseStatus.UNKNOWN,
            reason=(
                f"{expression!r} is not a well-formed SPDX expression ({exc}); an expression "
                "nobody can evaluate is never assumed permissive. Declare the package's real "
                "SPDX expression"
            ),
        )
    return verdict.model_copy(update={"spdx": expression})


def is_allowed(spdx: str) -> bool:
    """Whether a declared SPDX expression clears the commercial allowlist."""
    return classify_license(spdx).status is LicenseStatus.ALLOWED


def runtime_dependency_names(pyproject: Path) -> tuple[str, ...]:
    """Top-level runtime dependency names from pyproject, extras stripped."""
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    names: list[str] = []
    for dep in data["project"]["dependencies"]:
        name = dep.split(";", 1)[0].strip()
        name = name.split("[", 1)[0].strip()
        # strip version specifier / extras / markers
        name = name.split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0].split("~=", 1)[0]
        names.append(name.strip())
    return tuple(names)


def resolve_lock_closure(lock: Path, roots: tuple[str, ...]) -> frozenset[str]:
    """BFS over the lock graph from the given roots (conservative superset)."""
    with lock.open("rb") as fh:
        data = tomllib.load(fh)
    graph: dict[str, list[str]] = {}
    for pkg in data["package"]:
        graph[pkg["name"]] = [dep["name"] for dep in pkg.get("dependencies", [])]
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(dep for dep in graph.get(name, []) if dep not in seen)
    return frozenset(seen)


def audit_package_licenses(
    *,
    pyproject: Path,
    lock: Path,
    table: Mapping[str, str] = THIRD_PARTY,
) -> LicenseAudit:
    """Audit a project's resolved runtime closure against the reviewed table.

    Three independent governance failures, reported together so one run names
    every problem: an unreviewed dependency, a reviewed dependency whose
    license is not redistributable, and a declared root missing from the table.
    """
    roots = runtime_dependency_names(pyproject)
    closure = resolve_lock_closure(lock, roots) - {REPO_PACKAGE}
    errors: list[str] = []

    missing_review = sorted(name for name in closure if name not in table)
    if missing_review:
        errors.append(
            "runtime dependencies with no reviewed license entry: " + ", ".join(missing_review)
        )

    not_allowed = sorted(
        name for name, lic in table.items() if name in closure and not is_allowed(lic)
    )
    if not_allowed:
        how = [f"{name} ({table[name]})" for name in not_allowed]
        errors.append("runtime dependencies outside the commercial allowlist: " + ", ".join(how))

    unmapped_roots = sorted(name for name in roots if name not in table)
    if unmapped_roots:
        errors.append(
            "top-level project dependencies missing from table: " + ", ".join(unmapped_roots)
        )

    return LicenseAudit(dependencies=tuple(sorted(closure)), errors=tuple(errors))


#: Where a package's SPDX declaration may live, in resolution order. See
#: :mod:`or_audit.eval.conformance` for the contract this enforces.
_SPDX_MARKER = "SPDX-License-Identifier:"

#: Names an upstream may use for its license text. The ``LICENCE`` spellings are
#: not pedantry: ORBIT-Surgical (BSD-3-Clause) and SonoGym (MIT) both ship
#: ``LICENCE``, and GitHub's own detector reports ORBIT-Surgical as
#: ``NOASSERTION`` because of it. Missing the file means a permissively licensed
#: world gets refused as "no license declared", which is a false refusal - the
#: opposite failure from the one this module exists to prevent, and just as bad.
_LICENSE_FILES: tuple[str, ...] = (
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "LICENCE",
    "LICENCE.txt",
    "LICENCE.md",
    "COPYING",
)

#: ``[metadata].tags`` entry a task package uses to declare its license, since
#: ``TaskMetadata`` forbids unknown keys and tags are the open search field.
LICENSE_TAG_PREFIX = "license:"


class DeclaredLicense(BaseModel):
    """A package's license declaration and where it was found."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spdx: str = ""
    #: Package-relative provenance of the declaration (``""`` when absent).
    source: str = ""


def _from_wrap_json(root: Path) -> DeclaredLicense | None:
    path = root / "wrap.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise TaskContractError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TaskContractError(f"{path.name} must contain a JSON object")
    value = payload.get("license")
    if value is None:
        return None
    if not isinstance(value, str):
        raise TaskContractError(f"{path.name} 'license' must be an SPDX id string")
    return DeclaredLicense(spdx=value, source="wrap.json:license")


def _from_license_toml(root: Path) -> DeclaredLicense | None:
    path = root / "license.toml"
    if not path.is_file():
        return None
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    value = data.get("spdx")
    if value is None:
        return None
    if not isinstance(value, str):
        raise TaskContractError("license.toml 'spdx' must be an SPDX id string")
    return DeclaredLicense(spdx=value, source="license.toml:spdx")


def _from_license_file(root: Path) -> DeclaredLicense | None:
    for name in _LICENSE_FILES:
        path = root / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            marker = line.find(_SPDX_MARKER)
            if marker == -1:
                continue
            value = line[marker + len(_SPDX_MARKER) :].strip()
            if value:
                return DeclaredLicense(spdx=value, source=f"{name}:{_SPDX_MARKER}")
        # A license *text* with no SPDX marker is deliberately not classified:
        # inferring an identifier from prose is how the wrong license ships.
        return DeclaredLicense(spdx="", source=f"{name} (no {_SPDX_MARKER} marker)")
    return None


def declared_package_license(root: Path, tags: tuple[str, ...] = ()) -> DeclaredLicense:
    """Resolve a package's SPDX declaration in contract order.

    ``wrap.json`` wins because ``surgeval wrap`` writes it from a required
    argument, then an explicit ``license.toml``, then a ``license:<spdx>`` tag,
    then an ``SPDX-License-Identifier`` marker inside a bundled ``LICENSE``.
    An empty result is an absent declaration, which the caller must treat as a
    refusal.
    """
    for resolver in (_from_wrap_json, _from_license_toml):
        found = resolver(root)
        if found is not None and found.spdx:
            return found
    for tag in tags:
        if tag.lower().startswith(LICENSE_TAG_PREFIX):
            value = tag[len(LICENSE_TAG_PREFIX) :].strip()
            if value:
                return DeclaredLicense(spdx=value, source=f"task.toml:[metadata].tags {tag!r}")
    from_file = _from_license_file(root)
    if from_file is not None:
        return from_file
    return DeclaredLicense()
