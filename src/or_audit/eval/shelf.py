"""Benchmark shelves: loadable -> legible -> comparable (§2.5, §2.6, N5).

A shelf is one policy class run across several worlds of one modality and
reported as **per-world leaderboards**, never as a single-env leaderboard and
never as a cross-world ranking. Two editorial rules are structural here rather
than aspirational:

* **A sim shelf ships with real-data bench results** (§2.5). Naming a bench is
  not enough: every declared bench must resolve to a verified job bundle among
  the jobs the shelf is built from, and the build is refused otherwise. A shelf
  whose external claim would ship from sim rows alone does not build.
* **Cross-world collapse is refused** (§2.6). Aggregating, ranking, or ordering
  across worlds requires a validated
  :class:`~or_audit.eval.equivalence.EquivalenceArtifact` for that exact shelf
  and task family. This is the same posture the kernel already applies to
  composite scalars: ``shelf.json`` carries no cross-world number and the
  renderer emits no cross-world ordering.

Rows come from :func:`or_audit.eval.leaderboard.leaderboard_data`, so every row
on a shelf — sim or bench — has already been head-verified and trajectory-
reconstituted from its own bundle: a shelf stays replayable, not merely
displayable.
"""

from __future__ import annotations

import html
import json
import os
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from or_audit.errors import ScoreContractError, TaskContractError
from or_audit.eval.equivalence import EquivalenceArtifact, validate_equivalence
from or_audit.eval.leaderboard import leaderboard_data
from or_audit.eval.task import NonEmpty, Slug
from or_audit.eval.worlds import WorldKindSlug, world_kind_spec

SHELF_FORMAT_VERSION = "1"

#: Sections whose rows are scoped to a single result; everything else in the
#: payload is shelf-level and must therefore carry no number at all.
_ROW_SECTIONS = ("worlds", "benches")

#: Printed wherever a cross-world number would otherwise go.
CROSS_WORLD_REFUSAL = (
    "cross-world aggregation, ranking, and ordering are refused until a validated "
    "§2.6 equivalence artifact is published for this shelf and task family"
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ShelfGateRef(_Frozen):
    """One hard gate a shelf world's task publishes, as id and unit.

    The pair is the whole content: a gate is only comparable across worlds if
    both worlds gate *the same quantity* in *the same unit*, so an equivalence
    artifact is checked against these two fields and nothing else.
    """

    gate_id: Slug
    unit: str = ""


class ShelfWorldEntry(_Frozen):
    """One world on the shelf and the task that is run in it."""

    world_id: Slug
    world_kind: WorldKindSlug
    world_pin: str = ""
    task_id: NonEmpty
    #: Shelf-level grouping: worlds sharing a family pose the same kind of task
    #: and are therefore *candidates* for an equivalence artifact. Sharing a
    #: family is not itself a comparability claim.
    task_family: Slug

    @model_validator(mode="after")
    def _pin_is_declared_when_the_kind_requires_one(self) -> Self:
        """A kind whose capabilities require a world pin may not be shelved unpinned.

        Not merely under-specified: :func:`build_shelf` matches jobs on the exact
        ``(task_id, world_pin)`` pair, so an unpinned entry for a pinned kind
        rejects every real job as undeclared. The published endovascular shelf
        shipped exactly that for its ``sofa``/stEVE row, which meant the shelf
        could never be completed by any run.

        Unregistered kinds are not judged: we have no capability record for a
        third-party kind and inventing one would refuse legitimate shelves.
        """
        spec = world_kind_spec(self.world_kind)
        if spec is not None and spec.capabilities.requires_world_pin and not self.world_pin:
            raise TaskContractError(
                f"shelf world '{self.world_id}' is kind '{self.world_kind}', whose capabilities "
                f"require a world pin, but declares none. build_shelf matches jobs on the exact "
                f"(task_id, world_pin) pair, so every real job for this world would be refused "
                f"as undeclared and the shelf could never be completed. Fix: set world_pin to "
                f"the revision the task runs at — the catalog records one for every curated "
                f"world of this kind."
            )
        return self


class ShelfBenchEntry(_Frozen):
    """A real-data bench the shelf's worlds are stress-tested against."""

    task_id: NonEmpty
    #: Fixed discriminator: a bench is a real-data frozen-model contract, not a
    #: world. Declaring a world here would re-import the defect §2.5 forbids.
    kind: Literal["bench"] = "bench"
    description: str = ""
    #: World ids this bench pairs with; empty means the whole shelf.
    pairs_worlds: tuple[Slug, ...] = ()


class ShelfSpec(_Frozen):
    """A ``shelf.toml``: which worlds are on the shelf and what pairs them."""

    id: Slug
    title: NonEmpty
    modality: NonEmpty
    worlds: tuple[ShelfWorldEntry, ...]
    benches: tuple[ShelfBenchEntry, ...] = ()
    #: Path to the shelf's published equivalence artifact, when one exists.
    #: Recorded for citation; comparability is still checked, never assumed.
    equivalence_artifact: str = ""
    #: The §2.5 pairing rule, kept visible in the artifact instead of implicit.
    #: It is not a waiver switch: see :meth:`_bench_pairing`.
    require_bench: bool = True

    @model_validator(mode="after")
    def _worlds_are_declared_once(self) -> Self:
        if not self.worlds:
            raise TaskContractError("a shelf must declare at least one world")
        seen: set[str] = set()
        for world in self.worlds:
            if world.world_id in seen:
                raise TaskContractError(f"shelf world id declared twice: {world.world_id}")
            seen.add(world.world_id)
        return self

    @model_validator(mode="after")
    def _bench_pairing(self) -> Self:
        """Refuse a shelf whose external claim would ship from sim rows alone.

        ``require_bench = false`` is not an escape hatch. A shelf that names
        worlds is exactly the case §2.5 legislates, so the flag may only be
        cleared by a shelf that names none — and such a shelf is refused above.
        The field exists so the rule is legible in the published spec. Whether
        the named bench was actually *run* is enforced in :func:`build_shelf`;
        this validator only refuses a shelf that does not name one.
        """
        if not self.require_bench:
            raise TaskContractError(
                f"shelf '{self.id}' sets require_bench = false; the §2.5 pairing rule is "
                f"not waivable for a shelf that names worlds — declare a real-data bench"
            )
        if not self.benches:
            raise TaskContractError(
                f"shelf '{self.id}' names {len(self.worlds)} world(s) and no bench; every "
                f"sim domain on a shelf pairs with a real-data bench (§2.5), otherwise the "
                f"shelf's external claim would ship from sim rows alone. Add a "
                f'[[benches]] entry with kind = "bench" naming the real-data task.'
            )
        known = {world.world_id for world in self.worlds}
        world_tasks = {world.task_id for world in self.worlds}
        seen: set[str] = set()
        for bench in self.benches:
            if bench.task_id in seen:
                raise TaskContractError(f"shelf bench declared twice: {bench.task_id}")
            seen.add(bench.task_id)
            if bench.task_id in world_tasks:
                raise TaskContractError(
                    f"task '{bench.task_id}' is declared both as a shelf world and as a "
                    f"real-data bench; a sim result cannot stress-test itself"
                )
            unknown = sorted(set(bench.pairs_worlds) - known)
            if unknown:
                raise TaskContractError(
                    f"bench '{bench.task_id}' pairs with undeclared world(s): {', '.join(unknown)}"
                )
        return self

    @model_validator(mode="after")
    def _named_worlds_are_shelf_items(self) -> Self:
        """A shelf may not name a world the catalog records as off-shelf.

        This closes a real drift: the endovascular shelf listed ``cathsim`` as a
        target world while the catalog moved that row to ``skip`` on its
        CC-BY-NC-SA-4.0 terms. Two files, one claim, no check between them - so a
        published shelf kept naming a world nobody may redistribute.

        Scope, stated precisely so this is not mistaken for a licence gate:

        * Restricted terms are excluded *transitively*, not here.
          ``WorldPackage`` already refuses ``shipped``/``wrap`` under a denied
          licence, so a row that reaches this check cannot carry restricted
          terms. That is the property worth knowing; duplicating the licence
          test here would drift from it.
        * An **unrecorded** licence is not checked, and deliberately so. It
          blocks *installation* (``worlds install`` will not direct a fetch under
          terms nobody read) without making the world unnameable - first-party
          ``lumen`` is exactly that case, and dropping our own world from its own
          shelf would misreport an unfilled catalog field as a legal barrier.
        * Only ids the catalog knows are checked. A third-party shelf naming
          worlds we do not curate stays legal; we have no basis to judge those,
          and inventing one would be the same error in the other direction.
        """
        from or_audit.install.catalog import Disposition, load_catalog

        rows = {pkg.id: pkg for pkg in load_catalog().worlds}
        for world in self.worlds:
            pkg = rows.get(world.world_id)
            if pkg is None or pkg.disposition in {Disposition.SHIPPED, Disposition.WRAP}:
                continue
            raise TaskContractError(
                f"shelf '{self.id}' names world '{world.world_id}', which the catalog records "
                f"as disposition '{pkg.disposition.value}': {pkg.risks.strip() or 'see catalog'} "
                f"Fix: drop the world from the shelf, or promote its catalog row first — a "
                f"shelf that names a survey row claims a world it cannot put on a shelf."
            )
        return self


@dataclass(frozen=True, slots=True)
class ShelfWorldRows:
    """One world's verified rows, in that world's own ordering."""

    entry: ShelfWorldEntry
    rows: tuple[dict[str, Any], ...]
    #: The hard gates this world's task publishes, captured at build time from
    #: the verified job bundles and persisted in ``shelf.json``. ``None`` means
    #: *never established* — no row was built for this world, so nothing is
    #: known about its gates. That is not the same claim as an empty tuple
    #: ("this task declares no hard gates"), and a cross-world unlock must
    #: refuse the first while it may reason about the second.
    gate_manifest: tuple[ShelfGateRef, ...] | None = None


@dataclass(frozen=True, slots=True)
class ShelfBenchRows:
    """One real-data bench's verified rows: the shelf's reality check."""

    entry: ShelfBenchEntry
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ShelfReport:
    """A built shelf: per-world rows, bench rows, and nothing that crosses worlds."""

    spec: ShelfSpec
    worlds: tuple[ShelfWorldRows, ...]
    benches: tuple[ShelfBenchRows, ...]
    #: The job bundles the rows were built from, recorded so a reader can
    #: re-verify them instead of trusting the file. Empty means the report was
    #: assembled without evidence and :func:`load_shelf_report` will refuse the
    #: ``shelf.json`` it writes.
    job_paths: tuple[Path, ...] = ()

    def world(self, world_id: str) -> ShelfWorldRows:
        for world in self.worlds:
            if world.entry.world_id == world_id:
                return world
        known = ", ".join(world.entry.world_id for world in self.worlds)
        raise TaskContractError(f"shelf '{self.spec.id}' has no world '{world_id}' (has: {known})")


def load_shelf_spec(path: Path | str) -> ShelfSpec:
    """Load a ``shelf.toml`` (or the directory containing one)."""
    source = Path(path)
    if source.is_dir():
        source = source / "shelf.toml"
    if not source.is_file():
        raise TaskContractError(f"missing shelf.toml: {source}")
    # A user's malformed file is a contract refusal with a file name in it, not
    # a traceback: the CLI handlers catch only our own error types, so anything
    # tomllib or pydantic raises has to be translated here or it escapes raw.
    try:
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise TaskContractError(f"shelf {source} is not valid TOML: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise TaskContractError(f"shelf {source} could not be read: {exc}") from exc
    try:
        return ShelfSpec.model_validate(raw)
    except ValidationError as exc:
        raise TaskContractError(f"shelf {source} failed validation: {exc}") from exc


def _row_provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    """Tier-0 provenance for one row, with absence read as unattested.

    ``JobResult.world_engine`` is head-covered, so its *absence* is a real
    statement: the bundle never recorded which backend produced the
    observations. Defaulting that to ``real`` would let an unattested row print
    exactly like an attested one, which is the asymmetry this block exists to
    keep visible.
    """
    engine = row.get("world_engine") or {}
    return {
        "attested": bool(engine),
        "engine": engine.get("engine", ""),
        "backend": engine.get("backend", "unknown"),
        "backend_version": engine.get("backend_version", ""),
        "adapter_id": engine.get("adapter_id", ""),
        # Tier-0 honesty label: this world's instrumentation cannot report the
        # safety state a hard gate would need, so the row is explicitly not
        # safety-attested however clean its metrics look.
        "metrics_only": bool(engine.get("metrics_only")),
    }


def _shelf_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Reshape a leaderboard row so gates, abstention, and metrics stay apart."""
    assessed = row["headline_true"] + row["headline_false"]
    provenance = _row_provenance(row)
    declared = list(row.get("gate_specs") or ())
    return {
        "agent_identity": row["agent_identity"],
        "task_id": row["task_id"],
        "task_version": row["task_version"],
        "world_pin": row["world_pin"],
        "n": row["n"],
        # Hard gates never fold into the metric vector; they are their own block.
        # ``declared`` travels with the verdict because an empty gate set is not
        # a passing one: without it every metrics-only row reads as clean safety.
        "gates": {
            "any_gate_failed": row["any_gate_failed"],
            "declared": declared,
            "safety_attested": bool(declared) and not provenance["metrics_only"],
        },
        # Which world actually produced the observations, and at what tier.
        "provenance": provenance,
        "headline": {
            "id": row["headline"],
            "kind": row["headline_kind"],
            "direction": row["headline_direction"],
            "value": row["headline_value"],
            "true": row["headline_true"],
            "false": row["headline_false"],
            "rate": row["headline_rate"],
        },
        # Abstention is a legal outcome and is shown, not silently dropped.
        "abstention": {"unassessable": row["headline_unassessable"], "assessed": assessed},
        "metrics": row["metrics"],
        # Every row stays replayable from its own bundle.
        "head": row["head"],
    }


def _world_capabilities(entry: ShelfWorldEntry) -> dict[str, Any] | None:
    spec = world_kind_spec(entry.world_kind)
    return None if spec is None else spec.capabilities.model_dump(mode="json")


def _gate_manifest(
    entry: ShelfWorldEntry, rows: Iterable[Mapping[str, Any]]
) -> tuple[ShelfGateRef, ...] | None:
    """The hard gates this world's rows agree their task publishes.

    ``None`` when no row was built: the manifest was never established, which a
    cross-world unlock must treat as unknown rather than as "no gates". Rows
    that disagree are refused outright — a manifest that is two things cannot be
    the thing an equivalence artifact is checked against.
    """
    seen: set[tuple[ShelfGateRef, ...]] = set()
    for row in rows:
        seen.add(
            tuple(
                ShelfGateRef(gate_id=gate["gate_id"], unit=gate.get("unit", ""))
                for gate in sorted(row["gates"]["declared"], key=lambda gate: gate["gate_id"])
            )
        )
    if not seen:
        return None
    if len(seen) > 1:
        variants = " | ".join(
            ", ".join(f"{gate.gate_id}({gate.unit or '-'})" for gate in manifest) or "(none)"
            for manifest in sorted(seen, key=lambda manifest: [gate.gate_id for gate in manifest])
        )
        raise TaskContractError(
            f"shelf world '{entry.world_id}' carries rows whose tasks declare different hard "
            f"gates ({variants}); the world has no single gate manifest, so no equivalence "
            f"artifact could be checked against it — rebuild the shelf from one task version"
        )
    return seen.pop()


def _verified_buckets(
    spec: ShelfSpec, job_paths: Iterable[Path]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Head-verify every supplied bundle and bucket its rows by declared world.

    Shared by :func:`build_shelf` and :func:`load_shelf_report` on purpose: a
    reader must reach its rows by the same evidence path the writer did, or the
    two disagree about what "verified" means.
    """
    rows = leaderboard_data(list(job_paths))["rows"]
    world_buckets: dict[str, list[dict[str, Any]]] = {world.world_id: [] for world in spec.worlds}
    bench_buckets: dict[str, list[dict[str, Any]]] = {bench.task_id: [] for bench in spec.benches}
    by_key: dict[tuple[str, str], str] = {
        (world.task_id, world.world_pin): world.world_id for world in spec.worlds
    }
    for row in rows:
        if row["task_id"] in bench_buckets:
            bench_buckets[row["task_id"]].append(_shelf_row(row))
            continue
        world_id = by_key.get((row["task_id"], row["world_pin"] or ""))
        if world_id is None:
            declared = ", ".join(f"{task}@{pin or '-'}" for task, pin in sorted(by_key))
            raise TaskContractError(
                f"job for task '{row['task_id']}' at world pin "
                f"'{row['world_pin'] or '-'}' is not declared on shelf '{spec.id}' "
                f"(declared worlds: {declared}); add the world to shelf.toml or drop the job"
            )
        world_buckets[world_id].append(_shelf_row(row))

    unrun = [task_id for task_id, bench_rows in bench_buckets.items() if not bench_rows]
    if unrun:
        raise TaskContractError(
            f"shelf '{spec.id}' declares real-data bench(es) {', '.join(sorted(unrun))} with "
            f"no verified job bundle among the supplied jobs; a shelf cannot ship an external "
            f"claim from sim rows alone (§2.5) — run the bench task and pass its job directory"
        )
    return world_buckets, bench_buckets


def _assemble(
    spec: ShelfSpec,
    world_buckets: Mapping[str, list[dict[str, Any]]],
    bench_buckets: Mapping[str, list[dict[str, Any]]],
    job_paths: tuple[Path, ...],
) -> ShelfReport:
    return ShelfReport(
        spec=spec,
        worlds=tuple(
            ShelfWorldRows(
                entry=world,
                rows=tuple(world_buckets[world.world_id]),
                gate_manifest=_gate_manifest(world, world_buckets[world.world_id]),
            )
            for world in spec.worlds
        ),
        benches=tuple(
            ShelfBenchRows(entry=bench, rows=tuple(bench_buckets[bench.task_id]))
            for bench in spec.benches
        ),
        job_paths=job_paths,
    )


def build_shelf(
    spec: ShelfSpec,
    job_paths: Iterable[Path | str],
    *,
    out: Path | str,
) -> ShelfReport:
    """Group verified job rows under the shelf's worlds and paired benches.

    Two refusals hold the shelf's claims to its evidence:

    * every row must belong to a declared world or bench — an undeclared job
      silently appearing on a shelf would make its coverage claim untrue;
    * every declared bench must resolve to a verified job bundle — otherwise
      the shelf publishes a pairing it never ran.
    """
    resolved = tuple(Path(path) for path in job_paths)
    world_buckets, bench_buckets = _verified_buckets(spec, resolved)
    report = _assemble(spec, world_buckets, bench_buckets, resolved)
    write_shelf(report, Path(out))
    return report


def _job_references(report: ShelfReport, relative_to: Path | None) -> list[str]:
    """The job bundles, recorded so a reader can re-verify rather than trust.

    Written relative to the shelf directory when the two share a tree, so a
    shelf that is moved or copied wholesale still points at its own evidence.
    """
    references: list[str] = []
    for job in report.job_paths:
        if relative_to is None:
            references.append(str(job))
            continue
        try:
            references.append(os.path.relpath(job, relative_to))
        except ValueError:
            # Different drive on Windows: no relative form exists.
            references.append(str(Path(job).resolve()))
    return sorted(references)


def shelf_data(report: ShelfReport, *, relative_to: Path | None = None) -> dict[str, Any]:
    """Deterministic ``shelf.json`` payload: per-world rows, no cross-world number."""
    data: dict[str, Any] = {
        "format_version": SHELF_FORMAT_VERSION,
        "shelf": report.spec.model_dump(mode="json"),
        # Not decoration: `load_shelf_report` re-runs head verification and
        # world membership against these bundles before any row is ranked, so a
        # shelf.json alone is a report and never the evidence for one.
        "jobs": _job_references(report, relative_to),
        "cross_world": {
            "permitted": False,
            "reason": CROSS_WORLD_REFUSAL,
            "equivalence_artifact": report.spec.equivalence_artifact or None,
        },
        "worlds": [
            {
                **world.entry.model_dump(mode="json"),
                "capabilities": _world_capabilities(world.entry),
                # ``null`` distinguishes "never established" (no rows) from
                # "[]" ("this task declares no hard gates"). §2.6 consumers
                # must refuse the first and may reason about the second.
                "gate_manifest": (
                    None
                    if world.gate_manifest is None
                    else [gate.model_dump(mode="json") for gate in world.gate_manifest]
                ),
                "row_count": len(world.rows),
                "rows": list(world.rows),
            }
            for world in report.worlds
        ],
        "benches": [
            {
                **bench.entry.model_dump(mode="json"),
                "row_count": len(bench.rows),
                "rows": list(bench.rows),
            }
            for bench in report.benches
        ],
    }
    _assert_no_cross_world_scalar(data)
    return data


def _assert_no_cross_world_scalar(data: Mapping[str, Any]) -> None:
    """Refuse to emit any number that lives outside a single result's rows.

    Enforced rather than merely intended: a shelf-level count, mean, or score
    is precisely the collapse §2.6 forbids, and it is easy to reintroduce by
    adding one "harmless" summary field.
    """

    def walk(value: Any, path: str) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, int | float):
            raise ScoreContractError(
                f"shelf payload carries a cross-world scalar at {path}: {value!r}; "
                f"{CROSS_WORLD_REFUSAL}"
            )
        if isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, f"{path}.{key}")
        elif isinstance(value, list | tuple):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    for key, value in data.items():
        if key not in _ROW_SECTIONS:
            walk(value, key)


def _metric_display(metric: Mapping[str, Any]) -> str:
    if metric["kind"] == "boolean" and metric.get("rate") is not None:
        return f"{metric['rate']:.4g}"
    if metric["kind"] == "continuous" and metric.get("mean") is not None:
        unit = f" {metric['unit']}" if metric.get("unit") else ""
        return f"{metric['mean']:.4g}{unit}"
    if metric["kind"] == "categorical":
        return ", ".join(f"{category}: {count}" for category, count in metric["counts"].items())
    return "—"


def _gate_cell(row: Mapping[str, Any]) -> str:
    """What the gate column may honestly claim about this row.

    A row whose task declares no hard gates — which every Tier-0 metrics-only
    wrap necessarily is — has verified nothing about safety. Printing "no" there
    gives it the same cell a genuinely passing row gets, so an empty gate set
    reads as a clean safety result. Say what was checked instead.
    """
    gates = row["gates"]
    if gates["safety_attested"]:
        return "yes" if gates["any_gate_failed"] else "no"
    if row["provenance"]["metrics_only"]:
        return "not attested — Tier-0 metrics-only world, no safety state to gate"
    return "not attested — task declares no hard gates"


def _backend_cell(row: Mapping[str, Any]) -> str:
    """Which world produced the observations, with absence read as unattested."""
    provenance = row["provenance"]
    if not provenance["attested"]:
        return "unattested — bundle recorded no engine provenance"
    backend = provenance["backend"]
    engine = provenance["engine"]
    if backend == "synthetic-stub":
        return f"SYNTHETIC STUB — not a real world ({engine})" if engine else "SYNTHETIC STUB"
    if backend == "real":
        return f"real ({engine})" if engine else "real"
    return "unknown backend"


def _row_table(rows: Iterable[Mapping[str, Any]]) -> str:
    body: list[str] = []
    for row in rows:
        headline = row["headline"]
        value = headline["value"]
        shown = "—" if value is None else f"{value:.4g}"
        metrics = ", ".join(
            f"{key}={_metric_display(metric)}" for key, metric in row["metrics"].items()
        )
        cells = (
            row["agent_identity"],
            row["task_id"],
            str(row["n"]),
            _backend_cell(row),
            _gate_cell(row),
            f"{headline['id']} {shown}",
            str(row["abstention"]["unassessable"]),
            metrics,
            row["head"],
        )
        body.append(
            "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells) + "</tr>"
        )
    if not body:
        body.append('<tr><td colspan="9">no verified rows</td></tr>')
    return (
        "<table><thead><tr>"
        "<th>Agent</th><th>Task</th><th>n</th><th>Backend</th><th>Gate failures</th>"
        "<th>Headline</th><th>Unassessable</th><th>Metrics</th><th>Artifact head</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def render_html(data: Mapping[str, Any]) -> str:
    """Render one dependency-free table per world and bench; no cross-world ordering."""
    sections: list[str] = []
    for world in data["worlds"]:
        pin = world["world_pin"] or "—"
        manifest = world["gate_manifest"]
        if manifest is None:
            gates = "gate manifest not established (no verified row)"
        elif manifest:
            gates = "gates " + ", ".join(
                f"{gate['gate_id']} ({gate['unit'] or 'unitless'})" for gate in manifest
            )
        else:
            gates = "no hard gates declared — Tier-0, not safety-attested"
        sections.append(
            f"<h2>World: {html.escape(world['world_id'])}</h2>"
            f"<p>kind <code>{html.escape(world['world_kind'])}</code> · pin "
            f"<code>{html.escape(pin)}</code> · family "
            f"<code>{html.escape(world['task_family'])}</code> · "
            f"{html.escape(gates)}</p>" + _row_table(world["rows"])
        )
    for bench in data["benches"]:
        paired = ", ".join(bench["pairs_worlds"]) or "whole shelf"
        sections.append(
            f"<h2>Real-data bench: {html.escape(bench['task_id'])}</h2>"
            f"<p>{html.escape(bench['description'] or 'real-data stress test')} · pairs "
            f"<code>{html.escape(paired)}</code></p>" + _row_table(bench["rows"])
        )
    shelf = data["shelf"]
    return f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(shelf["title"])}</title>
<style>
body{{font:15px system-ui,sans-serif;margin:2rem;color:#17202a}}
h1{{margin-bottom:.25rem}}
h2{{margin-top:2rem}}
p{{color:#52606d}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #d8dee4;padding:.55rem;text-align:left;vertical-align:top}}
th{{background:#f4f6f8}}
td:last-child{{font:12px ui-monospace,monospace;overflow-wrap:anywhere}}
tr:nth-child(even){{background:#fafbfc}}
</style>
<h1>{html.escape(shelf["title"])}</h1>
<p>Modality <code>{html.escape(shelf["modality"])}</code>. Rows are reported per world,
with hard gates separated from metrics, abstention shown, and each row replayable from
its artifact head. Every world is paired with a real-data bench whose own verified rows
are reported below (§2.5).</p>
<p><strong>No cross-world ranking is shown.</strong> {html.escape(CROSS_WORLD_REFUSAL)}.</p>
{"".join(sections)}
"""


def write_shelf(report: ShelfReport, out: Path) -> dict[str, Any]:
    """Write deterministic ``shelf.json`` and a static ``index.html``."""
    data = shelf_data(report, relative_to=out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "shelf.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (out / "index.html").write_text(render_html(data), encoding="utf-8")
    return data


def _canonical(value: Any) -> str:
    """Stable text for comparing a persisted section against a rebuilt one."""
    return json.dumps(value, sort_keys=True, default=str)


def _recorded_jobs(data: Mapping[str, Any], source: Path) -> tuple[Path, ...]:
    """Resolve the job bundles a ``shelf.json`` claims it was built from."""
    recorded = data.get("jobs")
    if not isinstance(recorded, list) or not recorded:
        raise TaskContractError(
            f"shelf {source} records no job bundles, so its rows cannot be re-verified. A "
            f"shelf.json is a report, not the evidence for one: ranking rows nobody re-read "
            f"would let a hand-edited file set a cross-world mean rank. Rebuild it with "
            f"`surgeval shelf build`."
        )
    jobs: list[Path] = []
    for entry in recorded:
        if not isinstance(entry, str) or not entry:
            raise TaskContractError(f"shelf {source} records a malformed job reference: {entry!r}")
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = source.parent / candidate
        if not candidate.is_dir():
            raise TaskContractError(
                f"shelf {source} references job bundle '{entry}', which is not present at "
                f"{candidate}. Its rows cannot be re-verified from here, and ranking them on "
                f"the file's own word is exactly the check this refuses to fake — rebuild the "
                f"shelf where its job bundles live."
            )
        jobs.append(candidate)
    return tuple(jobs)


def load_shelf_report(path: Path | str) -> ShelfReport:
    """Read back a written ``shelf.json``, re-verifying it against its job bundles.

    Reading is not trusting. ``surgeval shelf rank`` uses each world's stored
    row order as that world's within-world rank, so a report reconstructed from
    the file alone would let anyone move a line in a text editor and change a
    cross-world mean rank while a valid equivalence artifact still supplied the
    "licensed by" label. Every row is therefore re-read from the job bundles the
    file names — head, task/agent digests, and trajectory reconstitution via
    :func:`~or_audit.eval.leaderboard.leaderboard_data`, then world membership
    by the same ``(task_id, world_pin)`` match the build used — and the result
    is compared against what the file claims. Any divergence is a refusal.
    """
    source = Path(path)
    if source.is_dir():
        source = source / "shelf.json"
    if not source.is_file():
        raise TaskContractError(f"missing shelf.json: {source}")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TaskContractError(f"shelf {source} is not valid JSON: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise TaskContractError(f"shelf {source} could not be read: {exc}") from exc
    if not isinstance(data, Mapping):
        raise TaskContractError(f"shelf {source} is not a JSON object: got {type(data).__name__}")
    if data.get("format_version") != SHELF_FORMAT_VERSION:
        raise TaskContractError(
            f"unsupported shelf format_version {data.get('format_version')!r} in {source}"
        )
    try:
        spec = ShelfSpec.model_validate(data["shelf"])
    except KeyError as exc:
        raise TaskContractError(f"shelf {source} has no 'shelf' section") from exc
    except ValidationError as exc:
        raise TaskContractError(f"shelf {source} failed validation: {exc}") from exc

    jobs = _recorded_jobs(data, source)
    world_buckets, bench_buckets = _verified_buckets(spec, jobs)
    report = _assemble(spec, world_buckets, bench_buckets, jobs)
    rebuilt = shelf_data(report, relative_to=source.parent)
    for section in _ROW_SECTIONS:
        if _canonical(data.get(section)) != _canonical(rebuilt[section]):
            raise TaskContractError(
                f"shelf {source} section '{section}' does not match the rows re-verified from "
                f"its job bundles: the file has been edited since it was built. Row order is "
                f"the within-world rank, so stored order that cannot be reproduced is refused "
                f"rather than ranked — rebuild with `surgeval shelf build`."
            )
    return report


def refuse_cross_world_aggregate(
    report: ShelfReport,
    *,
    task_family: str,
    operation: str = "aggregate",
    equivalence: EquivalenceArtifact | None = None,
) -> EquivalenceArtifact:
    """Gate every cross-world collapse on a validated §2.6 artifact.

    Returns the artifact when it licenses the operation and raises
    :class:`~or_audit.errors.ScoreContractError` otherwise. The refusal is the
    point: without published equivalence, ordering across worlds asserts a
    comparison that nobody measured.
    """
    if equivalence is None:
        raise ScoreContractError(
            f"refusing to {operation} across worlds on shelf '{report.spec.id}' for task "
            f"family '{task_family}': {CROSS_WORLD_REFUSAL}. Publish an equivalence "
            f"artifact (or_audit.eval.equivalence) and pass it explicitly."
        )
    if equivalence.shelf_id != report.spec.id:
        raise ScoreContractError(
            f"refusing to {operation} across worlds: equivalence artifact "
            f"'{equivalence.published_as.artifact_id}' was published for shelf "
            f"'{equivalence.shelf_id}', not '{report.spec.id}'"
        )
    if equivalence.task_family != task_family:
        raise ScoreContractError(
            f"refusing to {operation} across worlds: equivalence artifact "
            f"'{equivalence.published_as.artifact_id}' covers task family "
            f"'{equivalence.task_family}', not '{task_family}'"
        )
    declared = {world.entry.world_id: world for world in report.worlds}
    for world_id in equivalence.world_pair:
        world = declared.get(world_id)
        if world is None:
            raise ScoreContractError(
                f"refusing to {operation} across worlds: equivalence artifact names world "
                f"'{world_id}', which is not on shelf '{report.spec.id}'"
            )
        if world.entry.task_family != task_family:
            raise ScoreContractError(
                f"refusing to {operation} across worlds: shelf world '{world_id}' is in task "
                f"family '{world.entry.task_family}', not '{task_family}'"
            )
        if world.gate_manifest is None:
            raise ScoreContractError(
                f"refusing to {operation} across worlds: shelf world '{world_id}' has no "
                f"verified row, so its gate manifest was never established. An artifact cannot "
                f"be checked against gates nobody read, and an unchecked artifact is not a "
                f"licence — run the world's task and rebuild the shelf."
            )
    verdict = validate_equivalence(
        equivalence,
        world_gates={
            world.entry.world_id: world.gate_manifest
            for world in report.worlds
            if world.gate_manifest is not None
        },
    )
    if not verdict.valid:
        reasons = "; ".join(verdict.failures)
        raise ScoreContractError(
            f"refusing to {operation} across worlds: equivalence artifact "
            f"'{verdict.artifact_id}' fails "
            f"{', '.join(verdict.failed_requirements)} — {reasons}"
        )
    # Belt and braces: a valid verdict that skipped the manifest check licenses
    # nothing here. Silence from a check that did not run is not a pass.
    if not verdict.shelf_gates_checked:
        raise ScoreContractError(
            f"refusing to {operation} across worlds: equivalence artifact "
            f"'{verdict.artifact_id}' was not checked against the shelf's gate manifest"
        )
    return equivalence


def _order(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rows in their leaderboard ordering, each carrying why it sits there."""
    return [
        {
            "rank": index + 1,
            "agent_identity": row["agent_identity"],
            "task_id": row["task_id"],
            "any_gate_failed": row["gates"]["any_gate_failed"],
            # Carried so a ranking consumer cannot read "no gate failed" off a
            # row whose task gates nothing; see :func:`_gate_cell`.
            "safety_attested": row["gates"]["safety_attested"],
            "metrics_only": row["provenance"]["metrics_only"],
            "backend": row["provenance"]["backend"] if row["provenance"]["attested"] else None,
            "headline": row["headline"]["id"],
            "headline_value": row["headline"]["value"],
            "headline_direction": row["headline"]["direction"],
            "unassessable": row["abstention"]["unassessable"],
            "head": row["head"],
        }
        for index, row in enumerate(rows)
    ]


def shelf_ranking(
    report: ShelfReport,
    *,
    equivalence: EquivalenceArtifact | None = None,
) -> dict[str, Any]:
    """Per-world orderings always; one cross-world ordering only if earned.

    The cross-world ordering aggregates **within-world ranks**, never scores:
    what a validated equivalence artifact licenses is the claim that the two
    worlds order policies comparably, not that their numbers share a scale.
    """
    ranking: dict[str, Any] = {
        "shelf_id": report.spec.id,
        "per_world": [
            {
                "world_id": world.entry.world_id,
                "world_kind": world.entry.world_kind,
                "world_pin": world.entry.world_pin,
                "task_family": world.entry.task_family,
                "order": _order(world.rows),
            }
            for world in report.worlds
        ],
        "benches": [
            {"task_id": bench.entry.task_id, "order": _order(bench.rows)}
            for bench in report.benches
        ],
        "cross_world": None,
        "cross_world_refusal": CROSS_WORLD_REFUSAL,
    }
    if equivalence is None:
        return ranking

    artifact = refuse_cross_world_aggregate(
        report,
        task_family=equivalence.task_family,
        operation="rank",
        equivalence=equivalence,
    )
    ranks: dict[str, dict[str, int]] = {}
    for world_id in artifact.world_pair:
        for entry in _order(report.world(world_id).rows):
            ranks.setdefault(entry["agent_identity"], {})[world_id] = entry["rank"]
    covered = len(set(artifact.world_pair))
    paired = {agent: per_world for agent, per_world in ranks.items() if len(per_world) == covered}
    if not paired:
        raise ScoreContractError(
            f"refusing to rank across worlds on shelf '{report.spec.id}': no agent is "
            f"ranked in both {artifact.world_pair[0]} and {artifact.world_pair[1]}, so "
            f"there is nothing the equivalence artifact makes comparable"
        )
    order = sorted(
        (
            {
                "agent_identity": agent,
                "world_ranks": dict(sorted(per_world.items())),
                "mean_world_rank": sum(per_world.values()) / len(per_world),
            }
            for agent, per_world in paired.items()
        ),
        key=lambda item: (item["mean_world_rank"], item["agent_identity"]),
    )
    ranking["cross_world"] = {
        "task_family": artifact.task_family,
        "world_pair": list(artifact.world_pair),
        "basis": "mean within-world rank; per-world scores are never pooled",
        "equivalence_artifact": {
            "artifact_id": artifact.published_as.artifact_id,
            "digest": artifact.published_as.digest,
        },
        "order": order,
        "excluded_partial_coverage": sorted(set(ranks) - set(paired)),
    }
    ranking["cross_world_refusal"] = None
    return ranking
