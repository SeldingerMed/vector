"""The task-package provenance boundary: read a package's claims, re-derive, refuse.

An agent-authored adaptation (see :mod:`or_audit.concierge.adapt`) writes a
``provenance.json`` naming its parent and its quarantine. This module is the
kernel side, and the side that makes the N11 invariants real rather than
documented: every path that scores a package or publishes a row reads that
record back and recomputes the claims in it from the package on disk.

Three claims are re-derived, never trusted:

* **The frozen content.** :func:`content_digest` hashes every durable file
  except ``provenance.json`` itself, so the pin can live inside the package it
  pins. An adaptation edited after freezing no longer matches and is refused.
* **The parent's verifier.** :func:`verifier_identity` hashes the verifier
  declaration, the projection, and the verifier file bytes. The *parent's*
  identity is recorded at freeze time, so a moved gate or a retuned projection
  is caught at scoring even when the parent package is long gone.
* **The quarantine.** A record cannot grant itself public-leaderboard
  eligibility: :class:`PackageProvenance` refuses to parse one that claims it.

A package with no ``provenance.json`` is a curated package, not a suspect one:
the readers return ``None`` and check nothing at scoring time, because there is
no adaptation claim to check. Public-leaderboard ingestion is stricter: see
:func:`adaptation_tells`.

**This is drift detection, not authentication.** Every digest stored here is
self-authored. ``content_digest`` lives inside the package it pins, so an
operator who edits a verifier can recompute it, and ``verifier_identity``
records what freeze observed rather than what any authority attested. Local
signing would not change that and is deliberately absent: a key sitting on the
same disk as the package authenticates nothing, and a scheme that reads as
authentication while remaining forgeable is worse than an honest checksum.

Limitation, stated because overstating it would be the defect this boundary
exists to prevent: the quarantine is **not** tamper-proof. The package sits on
the operator's disk, and deleting ``provenance.json`` makes the adaptation
indistinguishable from a hand-authored package to every check here. What is
enforced is narrower and still worth having: a package that *presents*
provenance must have provenance that checks out, and a package whose provenance
marks it quarantined is refused. The dishonest path therefore requires
deliberately destroying the evidence rather than quietly editing a gate.

:func:`adaptation_tells` narrows the deletion path further by reading the marks
freeze leaves *outside* ``provenance.json``, so removing the record is no
longer sufficient — it also takes renaming the task version, which changes the
package's identity and severs the parent lineage the deletion was trying to
keep. That converts a silent bypass into a loud one. It does not make it a
trust boundary.

**Where the real boundary is, as a deliberate design position.** Public trust
needs an anchor outside the operator's disk: a receipt issued by the hosted
control plane when a package is ingested, and checked server-side against the
control plane's own record rather than against anything shipped beside the
package. Everything in this module is the local half — the half that makes
drift visible to the operator running the harness, and that a hosted ingestion
path can re-derive cheaply. The remote half belongs to the hosted surface and
is not implemented here. A reader finding no signature in this module is
looking at that division, not at an oversight.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final, Self

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from or_audit.audit.canonical import digest
from or_audit.errors import TaskContractError
from or_audit.eval.integrity import file_sha256
from or_audit.eval.loader import load_task
from or_audit.eval.task import TaskSpec

PROVENANCE_FILENAME: Final = "provenance.json"

#: Records written before the self-pin existed cannot establish what they claim,
#: so they are refused rather than read leniently.
PROVENANCE_FORMAT_VERSION: Final = "2"

#: The authorship class of an agent-authored package. Not a label a caller
#: chooses: the quarantine that keeps these packages off public leaderboards
#: keys off it, so a package free to claim ``human`` would be a package free to
#: opt out of its own quarantine.
AUTHORED_BY_AGENT: Final = "agent"

# Mirrors the filter in or_audit.eval.integrity.tree_digest, minus
# provenance.json. Kept local rather than parameterised into tree_digest so the
# canonical package identity keeps exactly one definition.
_IGNORED_PARTS: Final = frozenset(
    {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".git"}
)
_IGNORED_SUFFIXES: Final = frozenset({".pyc", ".pyo"})

#: ``freeze_adapted_package`` bumps the parent's ``task_version`` into an
#: ``-adaptedN`` lineage. Written unconditionally, so it is the one mark every
#: adaptation carries whether or not it came from a search.
_ADAPTED_VERSION: Final = re.compile(r"-adapted\d+$")

#: ``_derive_scenario`` names a re-seeded scenario ``<parent-id>-seed<N>`` and
#: sets its ``seed`` to the same ``N``. Present only when a search-derived
#: candidate was frozen, and matched on the pair rather than the suffix alone.
_DERIVED_SCENARIO: Final = re.compile(r"-seed(\d+)$")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ParentPin(_Frozen):
    """The package an adaptation was derived from, and what it may not change."""

    task_id: str
    task_version: str
    #: ``tree_digest`` of the parent package at freeze time.
    digest: str
    #: :func:`verifier_identity` of the parent: gates, metrics, headline,
    #: abstention, projection, and the verifier file bytes.
    verifier_identity: str


class PackageProvenance(_Frozen):
    """``provenance.json``: what was frozen, from what, and under what quarantine."""

    format_version: str
    authored_by: str
    public_leaderboard_eligible: bool = False
    quarantine_reason: str
    parent: ParentPin
    task_id: str
    task_version: str
    #: :func:`content_digest` of this package as frozen.
    content_digest: str
    scenarios: tuple[str, ...] = ()
    perturbations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _quarantined(self) -> Self:
        if self.format_version != PROVENANCE_FORMAT_VERSION:
            raise TaskContractError(
                f"package provenance for {self.task_id} declares format_version "
                f"{self.format_version!r}, not {PROVENANCE_FORMAT_VERSION!r}; a record "
                "without a content pin cannot establish that the package was not "
                "edited after freezing, so it is refused rather than read. Fix: "
                "re-freeze the adaptation from its parent"
            )
        if self.authored_by != AUTHORED_BY_AGENT:
            raise TaskContractError(
                f"package provenance for {self.task_id} claims authored_by "
                f"{self.authored_by!r}; an adaptation is {AUTHORED_BY_AGENT!r}-authored "
                "by construction, and relabelling the author would exempt it from the "
                "quarantine that keeps agent-authored packages off public leaderboards"
            )
        if self.public_leaderboard_eligible:
            raise TaskContractError(
                f"refusing to read public-leaderboard eligibility for "
                f"{self.task_id}: an adaptation is eligible only after promotion "
                "through the same Tier-1 conformance any wrap faces; a package cannot "
                "grant it to itself in its own provenance"
            )
        if not self.quarantine_reason:
            raise TaskContractError(
                f"package provenance for {self.task_id} records no quarantine_reason; "
                "an exclusion whose grounds nobody can read is not an exclusion"
            )
        if not self.content_digest:
            raise TaskContractError(
                f"package provenance for {self.task_id} records no content_digest; "
                "without a self-pin the package cannot be shown to be the one that "
                "was frozen"
            )
        if not self.parent.verifier_identity:
            raise TaskContractError(
                f"package provenance for {self.task_id} records no parent verifier "
                "identity; without it a moved gate or a retuned projection cannot be "
                "detected once the parent package is gone"
            )
        if not self.scenarios:
            raise TaskContractError(
                f"package provenance for {self.task_id} names no frozen scenario; an "
                "adaptation is an initial-condition claim, and an empty one is not one"
            )
        return self


def verifier_files(task: TaskSpec) -> tuple[str, ...]:
    """Package-relative files carrying the verifier a scenario author may not touch."""
    entrypoint = task.verifier.entrypoint
    module = entrypoint.split(":", 1)[0] if entrypoint else ""
    return tuple(name for name in (module, "verifier.toml") if name)


def verifier_identity(task_dir: Path | str, task: TaskSpec | None = None) -> str:
    """Digest everything an adaptation is forbidden to change.

    Declaration *and* bytes: a gate can be moved by editing ``task.toml`` or by
    editing the module that realises it, and either one turns an "adaptation"
    into a different claim wearing the parent's name.
    """
    root = Path(task_dir)
    spec = load_task(root) if task is None else task
    payload: dict[str, Any] = {
        "verifier": spec.verifier.model_dump(mode="json"),
        "projection": spec.projection.model_dump(mode="json") if spec.projection else None,
        "files": {
            name: file_sha256(root / name)
            for name in verifier_files(spec)
            if (root / name).is_file()
        },
    }
    return digest(payload)


def content_digest(task_dir: Path | str) -> str:
    """Hash every durable file except ``provenance.json``.

    The pin has to live inside the package it pins, and a digest cannot cover
    the file carrying it. Excluding exactly that one file keeps everything a
    scored run actually reads — task.toml, verifier.toml, the verifier module,
    scenario records — inside the pin.
    """
    root = Path(task_dir)
    if not root.is_dir():
        raise TaskContractError(f"package root is not a directory: {root}")
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.as_posix() == PROVENANCE_FILENAME:
            continue
        if any(part in _IGNORED_PARTS for part in relative.parts):
            continue
        if path.suffix in _IGNORED_SUFFIXES:
            continue
        rows.append({"path": relative.as_posix(), "sha256": file_sha256(path)})
    if not rows:
        raise TaskContractError(
            f"package has no durable file besides {PROVENANCE_FILENAME}: {root}"
        )
    return digest(rows)


def provenance_record(
    *,
    parent: TaskSpec,
    parent_dir: Path | str,
    parent_digest: str,
    task_version: str,
    package_dir: Path | str,
    scenarios: tuple[str, ...],
    perturbations: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build the record an adaptation writes as its ``provenance.json``.

    Validated through :class:`PackageProvenance` before it is returned, so a
    record that could not be read back is never written to disk.
    """
    record: dict[str, Any] = {
        "format_version": PROVENANCE_FORMAT_VERSION,
        "authored_by": AUTHORED_BY_AGENT,
        "public_leaderboard_eligible": False,
        "quarantine_reason": (
            "agent-authored scenario package: excluded from public leaderboards "
            "until promoted through the same Tier-1 conformance any wrap faces"
        ),
        "parent": {
            "task_id": parent.id,
            "task_version": parent.task_version,
            "digest": parent_digest,
            "verifier_identity": verifier_identity(parent_dir, parent),
        },
        "task_id": parent.id,
        "task_version": task_version,
        "content_digest": content_digest(package_dir),
        "scenarios": list(scenarios),
        "perturbations": list(perturbations),
    }
    PackageProvenance.model_validate(record)
    return record


def read_provenance(task_dir: Path | str) -> PackageProvenance | None:
    """Parse a package's provenance record, or ``None`` if it presents none."""
    path = Path(task_dir) / PROVENANCE_FILENAME
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskContractError(
            f"{path} is not readable JSON, so the package's quarantine and content "
            f"pin cannot be established: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise TaskContractError(
            f"{path} is not a provenance object (got {type(raw).__name__}); a package "
            "presenting an unreadable provenance file is refused, not treated as "
            "having none"
        )
    try:
        return PackageProvenance.model_validate(raw)
    except TaskContractError:
        raise
    except ValidationError as exc:
        raise TaskContractError(
            f"{path} is not a readable package provenance record: {exc}"
        ) from exc


def assert_scoreable_package(task_dir: Path | str) -> PackageProvenance | None:
    """Re-derive a package's provenance claims from disk, or refuse to score it.

    Returns ``None`` for a package presenting no ``provenance.json``: a curated
    package makes no adaptation claim, so there is none to verify. For an
    adapted package every recorded claim is recomputed here — this is the only
    thing standing between "no in-place mutation, ever" and a comment.
    """
    root = Path(task_dir)
    provenance = read_provenance(root)
    if provenance is None:
        return None
    task = load_task(root)
    if (task.id, task.task_version) != (provenance.task_id, provenance.task_version):
        raise TaskContractError(
            f"refusing to score {root}: it loads as {task.id}@{task.task_version} but "
            f"its provenance records {provenance.task_id}@{provenance.task_version}; "
            "one package cannot answer to two identities"
        )
    # Verifier first: a verifier edit also breaks the content pin, and naming the
    # specific violation beats reporting that something, somewhere, changed.
    found_verifier = verifier_identity(root, task)
    if found_verifier != provenance.parent.verifier_identity:
        raise TaskContractError(
            f"refusing to score {task.id}@{task.task_version}: the verifier, its "
            f"gates, or the projection differs from parent "
            f"{provenance.parent.task_id}@{provenance.parent.task_version} that it was "
            f"adapted from ({provenance.parent.verifier_identity} -> {found_verifier}). "
            "The concierge can author scenarios, never verifiers"
        )
    found_content = content_digest(root)
    if found_content != provenance.content_digest:
        raise TaskContractError(
            f"refusing to score {task.id}@{task.task_version}: package at {root} was "
            f"edited after freezing (pinned {provenance.content_digest}, now "
            f"{found_content}). An adaptation is a new version, not an edit"
        )
    return provenance


def adaptation_tells(task_dir: Path | str, task: TaskSpec | None = None) -> tuple[str, ...]:
    """Marks ``freeze_adapted_package`` leaves *outside* ``provenance.json``.

    Deleting the provenance record is the one bypass no local check can close,
    so the next best thing is to make the deletion insufficient. These are the
    complete set of things freeze writes into a package besides the record
    itself — it rewrites ``task.toml``'s ``task_version``, ``scenarios`` and
    ``perturbations``, copies everything else byte-for-byte, and writes the
    record. Perturbation ids are copied unchanged and carry no mark, so the
    list below is exhaustive rather than a sample.

    Returns human-readable findings, empty when the package carries no mark.
    """
    root = Path(task_dir)
    spec = load_task(root) if task is None else task
    found: list[str] = []
    if _ADAPTED_VERSION.search(spec.task_version):
        found.append(
            f"task_version {spec.task_version!r} is in the '-adaptedN' lineage that "
            "freezing an adaptation writes"
        )
    for scenario in spec.scenarios:
        match = _DERIVED_SCENARIO.search(scenario.id)
        # The pair, not the suffix: a re-seeded scenario is named for the seed it
        # was re-seeded to, so an id and seed that disagree is somebody's own
        # naming rather than a derived scenario.
        if match and scenario.seed == int(match.group(1)):
            found.append(
                f"scenario {scenario.id!r} is named for its own seed "
                f"({scenario.seed}), the shape a search-derived scenario takes"
            )
    return tuple(found)


def assert_public_leaderboard_eligible(task_dir: Path | str) -> None:
    """Refuse an agent-authored adaptation at public-leaderboard ingestion.

    Integrity is re-derived first: a quarantine flag read off a package that was
    edited after freezing says nothing about what actually ran.

    A package presenting no record but bearing :func:`adaptation_tells` is
    refused too. Publishing is where the quarantine has to hold, and a package
    that looks adapted and cannot say what it was adapted from is exactly the
    case that must not be ranked silently. This is a naming-convention check,
    not a proof: see the module docstring.
    """
    root = Path(task_dir)
    provenance = assert_scoreable_package(root)
    if provenance is None:
        task = load_task(root)
        tells = adaptation_tells(root, task)
        if not tells:
            return
        listed = "; ".join(tells)
        raise TaskContractError(
            f"refusing to publish {task.id}@{task.task_version} on a public "
            f"leaderboard: it carries the marks of an agent-authored adaptation "
            f"({listed}) but presents no {PROVENANCE_FILENAME}, so it cannot say what "
            f"it was adapted from or whether its verifier still matches that parent. "
            f"Fix: restore the package's {PROVENANCE_FILENAME} if it is an adaptation "
            f"(it stays quarantined, and a quarantined row is refused here by name), "
            f"or rename it out of the adaptation conventions if it is not — a package "
            f"that keeps an adapted lineage in its identity while dropping the "
            f"evidence for it is claiming the parent's name without the parent's pin"
        )
    raise TaskContractError(
        f"refusing to publish {provenance.task_id}@{provenance.task_version} on a "
        f"public leaderboard: it is an agent-authored adaptation of "
        f"{provenance.parent.task_id}@{provenance.parent.task_version} "
        f"({provenance.quarantine_reason}). Fix: promote it through the same Tier-1 "
        f"conformance any wrap faces and publish the promoted package, or drop this "
        f"job from the leaderboard inputs deliberately rather than letting the row "
        f"disappear unstated"
    )


__all__ = [
    "AUTHORED_BY_AGENT",
    "PROVENANCE_FILENAME",
    "PROVENANCE_FORMAT_VERSION",
    "PackageProvenance",
    "ParentPin",
    "adaptation_tells",
    "assert_public_leaderboard_eligible",
    "assert_scoreable_package",
    "content_digest",
    "provenance_record",
    "read_provenance",
    "verifier_files",
    "verifier_identity",
]
