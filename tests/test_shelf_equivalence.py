"""Shelf building (§2.5) and cross-world equivalence (§2.6).

The contracts under test are refusals: a shelf that names a bench it never ran
does not build, a shelf.json carries no cross-world number, and no surface
orders across worlds without a validated, published equivalence artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from or_audit.commands.shelf import register
from or_audit.errors import ScoreContractError, TaskContractError
from or_audit.eval.equivalence import (
    DeclaredMatch,
    EquivalenceArtifact,
    ExternalReferent,
    GateCalibration,
    GateEquivalence,
    Publication,
    ScenarioAlignment,
    TaskEquivalence,
    equivalence_digest,
    load_equivalence_artifact,
    spearman_rank_correlation,
    validate_equivalence,
    write_equivalence_artifact,
)
from or_audit.eval.loader import load_agent, load_task
from or_audit.eval.runner import run_job
from or_audit.eval.shelf import (
    ShelfBenchEntry,
    ShelfReport,
    ShelfSpec,
    ShelfWorldEntry,
    build_shelf,
    load_shelf_report,
    load_shelf_spec,
    refuse_cross_world_aggregate,
    shelf_ranking,
)

ROOT = Path(__file__).resolve().parents[1]
VIDEO_TASK = ROOT / "docs/examples/tasks/video-nextstep"
VIDEO_AGENT = ROOT / "docs/examples/agents/example-video-predictor"
_NO_PYCACHE = shutil.ignore_patterns("__pycache__")

SHELF_ID = "endovascular"
FAMILY = "next-step-prediction"
BENCH_TASK = "angio-realdata-bench"


def _task_package(dst: Path, *, task_id: str, world_pin: str = "") -> Path:
    """A video-nextstep package repointed at one shelf world."""
    shutil.copytree(VIDEO_TASK, dst, ignore=_NO_PYCACHE)
    toml_path = dst / "task.toml"
    text = toml_path.read_text(encoding="utf-8").replace(
        'id = "video-nextstep"', f'id = "{task_id}"', 1
    )
    if world_pin:
        text = text.replace(
            'kind = "frame-source"',
            f'kind = "frame-source"\nworld_pin = "{world_pin}"',
            1,
        )
    toml_path.write_text(text, encoding="utf-8")
    return dst


def _agent_package(dst: Path, *, agent_id: str, wrong: frozenset[str] = frozenset()) -> Path:
    """A frozen predictor package that gets ``wrong`` clips wrong."""
    shutil.copytree(VIDEO_AGENT, dst, ignore=_NO_PYCACHE)
    predictions = dst / "predictions.json"
    payload = json.loads(predictions.read_text(encoding="utf-8"))
    for item in payload["items"]:
        if item["id"] in wrong and "next_step" in item:
            item["next_step"] = "hold" if item["next_step"] != "hold" else "advance"
    text = json.dumps(payload, indent=2) + "\n"
    predictions.write_text(text, encoding="utf-8")
    pin = hashlib.sha256(text.encode("utf-8")).hexdigest()
    toml_path = dst / "agent.toml"
    agent_text = re.sub(
        r'weights_pin = "[0-9a-f]+"',
        f'weights_pin = "{pin}"',
        toml_path.read_text(encoding="utf-8"),
    ).replace('id = "example/video-predictor"', f'id = "{agent_id}"', 1)
    toml_path.write_text(agent_text, encoding="utf-8")
    return dst


def _run(task_dir: Path, agent_dir: Path, out: Path) -> Path:
    run_job(
        task=load_task(task_dir),
        task_dir=task_dir,
        agent=load_agent(agent_dir),
        agent_dir=agent_dir,
        out=out,
        n=3,
    )
    return out


@pytest.fixture(scope="module")
def bundles(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Two sim worlds and one real-data bench, each an actually-run job bundle."""
    root = tmp_path_factory.mktemp("shelf-bundles")
    world_a = _task_package(root / "task-a", task_id="endo-nav-a", world_pin="corpus-a@v1")
    world_b = _task_package(root / "task-b", task_id="endo-nav-b", world_pin="corpus-b@v1")
    bench = _task_package(root / "task-bench", task_id=BENCH_TASK)
    strong = _agent_package(root / "agent-strong", agent_id="example/predictor-strong")
    weak = _agent_package(
        root / "agent-weak", agent_id="example/predictor-weak", wrong=frozenset({"clip-001"})
    )
    jobs = {
        "a-strong": _run(world_a, strong, root / "job-a-strong"),
        "a-weak": _run(world_a, weak, root / "job-a-weak"),
        "b-strong": _run(world_b, strong, root / "job-b-strong"),
        "b-weak": _run(world_b, weak, root / "job-b-weak"),
        "bench-strong": _run(bench, strong, root / "job-bench-strong"),
    }
    return {"root": root, "jobs": jobs}


def _spec(**overrides: Any) -> ShelfSpec:
    fields: dict[str, Any] = {
        "id": SHELF_ID,
        "title": "Endovascular next-step shelf",
        "modality": "procedural-video",
        "worlds": (
            ShelfWorldEntry(
                world_id="corpus-a",
                world_kind="frame-source",
                world_pin="corpus-a@v1",
                task_id="endo-nav-a",
                task_family=FAMILY,
            ),
            ShelfWorldEntry(
                world_id="corpus-b",
                world_kind="frame-source",
                world_pin="corpus-b@v1",
                task_id="endo-nav-b",
                task_family=FAMILY,
            ),
        ),
        "benches": (
            ShelfBenchEntry(
                task_id=BENCH_TASK,
                description="frozen-model perception bench on real angiography",
                pairs_worlds=("corpus-a", "corpus-b"),
            ),
        ),
    }
    fields.update(overrides)
    return ShelfSpec(**fields)


def _sim_jobs(bundles: dict[str, Any]) -> list[Path]:
    return [path for key, path in bundles["jobs"].items() if not key.startswith("bench")]


def _all_jobs(bundles: dict[str, Any]) -> list[Path]:
    return list(bundles["jobs"].values())


@pytest.fixture(scope="module")
def built(
    bundles: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> tuple[ShelfReport, Path]:
    out = tmp_path_factory.mktemp("shelf-out") / "shelf"
    return build_shelf(_spec(), _all_jobs(bundles), out=out), out


@pytest.fixture(scope="module")
def report(built: tuple[ShelfReport, Path]) -> ShelfReport:
    return built[0]


@pytest.fixture(scope="module")
def shelf_json(built: tuple[ShelfReport, Path]) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((built[1] / "shelf.json").read_text(encoding="utf-8"))
    return payload


def _agents(report: ShelfReport, world_id: str) -> list[str]:
    return [row["agent_identity"] for row in report.world(world_id).rows]


# --- shelf construction -----------------------------------------------------


def test_shelf_groups_verified_rows_per_world_with_heads(report: ShelfReport):
    assert [world.entry.world_id for world in report.worlds] == ["corpus-a", "corpus-b"]
    for world in report.worlds:
        assert len(world.rows) == 2
        for row in world.rows:
            assert len(row["head"]) == 64
            assert row["world_pin"] == world.entry.world_pin
            # Gates are a block of their own; the metric vector stays intact.
            assert set(row["gates"]) == {"any_gate_failed"}
            assert "next_step_correct" in row["metrics"]
            # Abstention is reported, not folded away: one clip abstains.
            assert row["abstention"]["unassessable"] == 1
            assert row["abstention"]["assessed"] == 2
    # The stronger predictor outranks the weaker one inside each world, on its
    # own headline value; nothing is pooled across worlds to get there.
    for world_id in ("corpus-a", "corpus-b"):
        rows = report.world(world_id).rows
        assert [row["headline"]["value"] for row in rows] == [1.0, 0.5]
        assert _agents(report, world_id)[0].startswith("example/predictor-strong@")


def test_shelf_carries_verified_bench_rows(report: ShelfReport, shelf_json: dict[str, Any]):
    assert [bench.entry.task_id for bench in report.benches] == [BENCH_TASK]
    bench_rows = report.benches[0].rows
    assert len(bench_rows) == 1
    assert len(bench_rows[0]["head"]) == 64
    payload_bench = shelf_json["benches"][0]
    assert payload_bench["kind"] == "bench"
    assert payload_bench["rows"][0]["head"] == bench_rows[0]["head"]


def test_shelf_json_has_no_cross_world_scalar(shelf_json: dict[str, Any]):
    """Every number in the payload must be scoped to one world's or bench's rows."""
    found: list[str] = []

    def scan(value: Any, path: str) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, int | float):
            found.append(f"{path}={value!r}")
        elif isinstance(value, dict):
            for key, item in value.items():
                scan(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                scan(item, f"{path}[{index}]")

    for key, value in shelf_json.items():
        if key not in {"worlds", "benches"}:
            scan(value, key)
    assert found == []
    assert shelf_json["cross_world"]["permitted"] is False
    assert "equivalence artifact" in shelf_json["cross_world"]["reason"]
    # Rows themselves keep their numbers: the refusal is about shelf-level scalars.
    assert shelf_json["worlds"][0]["rows"][0]["n"] == 3


def test_shelf_html_reports_per_world_and_refuses_an_overall_ranking(
    bundles: dict[str, Any], tmp_path: Path
):
    build_shelf(_spec(), _all_jobs(bundles), out=tmp_path / "html")
    page = (tmp_path / "html" / "index.html").read_text(encoding="utf-8")
    assert "World: corpus-a" in page
    assert "World: corpus-b" in page
    assert f"Real-data bench: {BENCH_TASK}" in page
    assert "No cross-world ranking is shown." in page


def test_shelf_refuses_a_declared_bench_with_no_verified_job(
    bundles: dict[str, Any], tmp_path: Path
):
    with pytest.raises(TaskContractError) as excinfo:
        build_shelf(_spec(), _sim_jobs(bundles), out=tmp_path / "no-bench")
    message = str(excinfo.value)
    assert BENCH_TASK in message
    assert "sim rows alone" in message
    assert not (tmp_path / "no-bench").exists()


def test_shelf_spec_without_a_paired_bench_is_refused():
    with pytest.raises(TaskContractError, match=r"ships from sim rows alone|real-data bench"):
        _spec(benches=())


def test_shelf_spec_cannot_waive_the_pairing_rule():
    with pytest.raises(TaskContractError, match="not waivable"):
        _spec(require_bench=False)


def test_shelf_spec_refuses_a_bench_that_is_its_own_world():
    with pytest.raises(TaskContractError, match="cannot stress-test itself"):
        _spec(benches=(ShelfBenchEntry(task_id="endo-nav-a"),))


def test_shelf_refuses_a_world_the_catalog_will_not_ship():
    """Two files, one claim: the shelf may not outlive its catalog row.

    Live drift, not hypothetical - the endovascular shelf named ``cathsim`` as a
    target world after the audit moved that row to ``skip`` on its
    CC-BY-NC-SA-4.0 terms, and nothing checked between the two files.
    """
    with pytest.raises(TaskContractError, match="disposition 'skip'"):
        _spec(
            worlds=(
                ShelfWorldEntry(
                    world_id="cathsim",
                    world_kind="gym",
                    task_id="cathsim-aorta-nav",
                    task_family=FAMILY,
                ),
            ),
            benches=(ShelfBenchEntry(task_id=BENCH_TASK, pairs_worlds=("cathsim",)),),
        )


def test_shelf_may_name_a_world_the_catalog_does_not_curate():
    """A third-party world is legal: we have no basis to judge its terms.

    Refusing every unknown id would push third-party shelves out of the format
    for no evidence, which is the same error as naming an unshippable world.
    """
    spec = _spec(
        worlds=(
            ShelfWorldEntry(
                world_id="some-third-party-world",
                world_kind="gym",
                task_id="tp-nav",
                task_family=FAMILY,
            ),
        ),
        benches=(ShelfBenchEntry(task_id=BENCH_TASK, pairs_worlds=("some-third-party-world",)),),
    )
    assert spec.worlds[0].world_id == "some-third-party-world"


def test_the_shipped_endovascular_shelf_names_only_shelf_items():
    """The published example is held to the rule, not just the model."""
    from or_audit.install.catalog import Disposition, world_package

    spec = load_shelf_spec(Path("docs/examples/shelves/endovascular.toml"))
    assert [world.world_id for world in spec.worlds] == ["lumen", "steve"]
    for world in spec.worlds:
        assert world_package(world.world_id).disposition in {
            Disposition.SHIPPED,
            Disposition.WRAP,
        }


def test_a_shelf_item_can_never_carry_restricted_terms():
    """The property the shelf's disposition check leans on, pinned here.

    `ShelfSpec` checks disposition only. That is sound *because* a catalog row
    cannot be `shipped`/`wrap` under denied terms - so if this ever stops
    holding, the shelf check silently weakens instead of failing.
    """
    from or_audit.eval.licensing import LicenseStatus
    from or_audit.install.catalog import Disposition, WorldPackage, load_catalog

    for pkg in load_catalog().worlds:
        if pkg.disposition in {Disposition.SHIPPED, Disposition.WRAP}:
            assert pkg.license_verdict.status is not LicenseStatus.RESTRICTED

    with pytest.raises(TaskContractError, match="separate written terms"):
        WorldPackage.model_validate(
            {
                "id": "restricted-shelf-item",
                "display_name": "X",
                "domain": "d",
                "engine": "e",
                "disposition": "wrap",
                "license": "CC-BY-NC-SA-4.0",
                "safety_evidence": "fixture",
                "install": {"strategy": "source-build"},
            }
        )


def test_shelf_refuses_a_job_for_an_undeclared_world(bundles: dict[str, Any], tmp_path: Path):
    narrowed = _spec(
        worlds=(_spec().worlds[0],),
        benches=(ShelfBenchEntry(task_id=BENCH_TASK, pairs_worlds=("corpus-a",)),),
    )
    with pytest.raises(TaskContractError, match="is not declared on shelf"):
        build_shelf(narrowed, _all_jobs(bundles), out=tmp_path / "undeclared")


def test_shelf_toml_round_trips(tmp_path: Path):
    (tmp_path / "shelf.toml").write_text(
        """
id = "endovascular"
title = "Endovascular next-step shelf"
modality = "procedural-video"

[[worlds]]
world_id = "corpus-a"
world_kind = "frame-source"
world_pin = "corpus-a@v1"
task_id = "endo-nav-a"
task_family = "next-step-prediction"

[[benches]]
task_id = "angio-realdata-bench"
kind = "bench"
description = "real angiography"
pairs_worlds = ["corpus-a"]
""",
        encoding="utf-8",
    )
    spec = load_shelf_spec(tmp_path)
    assert spec.worlds[0].task_family == FAMILY
    assert spec.benches[0].kind == "bench"


# --- equivalence artifact ---------------------------------------------------


def _matched(statement: str) -> DeclaredMatch:
    return DeclaredMatch(statement=statement, matched=True)


def _gate(*, unit_b: str = "newton", quantity_b: str = "wall contact force") -> GateEquivalence:
    return GateEquivalence(
        gate_id="unsafe_prediction",
        physical_quantity="wall contact force",
        unit="newton",
        physical_event="vessel wall perforation onset",
        calibration=(
            GateCalibration(
                world_id="corpus-a",
                physical_quantity="wall contact force",
                unit="newton",
                threshold=0.35,
                bites_at_declared_event=True,
                evidence="force sensor sweep on the silicone loop; onset at 0.34-0.36 N",
            ),
            GateCalibration(
                world_id="corpus-b",
                physical_quantity=quantity_b,
                unit=unit_b,
                threshold=0.35,
                bites_at_declared_event=True,
                evidence="matched sweep on the second engine; onset at 0.34-0.36 N",
            ),
        ),
    )


def _artifact(
    *,
    gate: GateEquivalence | None = None,
    world_rankings: dict[str, tuple[str, ...]] | None = None,
    referent_ranking: tuple[str, ...] = ("strong", "middle", "weak"),
    rank_correlation: float = 1.0,
    min_rank_correlation: float = 0.8,
    digest_value: str = "",
) -> EquivalenceArtifact:
    rankings = world_rankings or {
        "corpus-a": ("strong", "middle", "weak"),
        "corpus-b": ("strong", "middle", "weak"),
    }
    return EquivalenceArtifact(
        shelf_id=SHELF_ID,
        task_family=FAMILY,
        world_pair=("corpus-a", "corpus-b"),
        task_equivalence=TaskEquivalence(
            objective=_matched("both worlds score next-step prediction against held-out labels"),
            initial_state_distribution=_matched("both draw clips from the same held-out split"),
            termination=_matched("both terminate after a single scored prediction"),
        ),
        gate_equivalence=(gate or _gate(),),
        scenario_alignment=ScenarioAlignment(
            mode="stratified",
            statement="anatomy differs; comparison is made within difficulty strata",
            strata=("proximal", "distal"),
        ),
        external_referent=ExternalReferent(
            kind="phantom",
            description="silicone flow-loop phantom scored by two blinded raters",
            world_rankings=rankings,
            referent_ranking=referent_ranking,
            rank_correlation=rank_correlation,
            min_rank_correlation=min_rank_correlation,
        ),
        published_as=Publication(artifact_id="endovascular-equivalence-v1", digest=digest_value),
    )


def _published(artifact: EquivalenceArtifact) -> EquivalenceArtifact:
    return artifact.model_copy(
        update={
            "published_as": artifact.published_as.model_copy(
                update={"digest": equivalence_digest(artifact)}
            )
        }
    )


def test_spearman_handles_perfect_inverse_and_ties():
    assert spearman_rank_correlation([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)
    assert spearman_rank_correlation([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == pytest.approx(-1.0)
    # Average ranks for ties: the tied pair shares rank 1.5, so agreement is partial.
    tied = spearman_rank_correlation([1.0, 1.0, 3.0], [1.0, 2.0, 3.0])
    assert tied == pytest.approx(0.8660254, rel=1e-6)
    assert spearman_rank_correlation([1.0, 1.0, 2.0], [5.0, 5.0, 9.0]) == pytest.approx(1.0)


def test_spearman_refuses_undefined_correlations():
    with pytest.raises(ScoreContractError, match="equal-length"):
        spearman_rank_correlation([1.0, 2.0], [1.0])
    with pytest.raises(ScoreContractError, match="at least two"):
        spearman_rank_correlation([1.0], [2.0])
    with pytest.raises(ScoreContractError, match="ties every subject"):
        spearman_rank_correlation([1.0, 1.0], [1.0, 2.0])


def test_valid_artifact_validates_every_requirement():
    verdict = validate_equivalence(_published(_artifact()))
    assert verdict.valid
    assert verdict.failures == ()
    assert set(verdict.requirements) == {
        "task_equivalence",
        "gate_equivalence",
        "scenario_alignment",
        "external_referent",
        "publication",
    }
    assert verdict.computed_rank_correlation == pytest.approx(1.0)


def test_identically_named_gates_in_different_units_fail_gate_equivalence():
    verdict = validate_equivalence(_published(_artifact(gate=_gate(unit_b="millinewton"))))
    assert not verdict.valid
    assert verdict.failed_requirements == ("gate_equivalence",)
    assert any(
        "'millinewton'" in failure and "not the same gate" in failure
        for failure in verdict.failures
    )


def test_gate_measuring_a_different_quantity_fails():
    verdict = validate_equivalence(_published(_artifact(gate=_gate(quantity_b="tip displacement"))))
    assert verdict.failed_requirements == ("gate_equivalence",)
    assert any("tip displacement" in failure for failure in verdict.failures)


def test_referent_below_the_declared_minimum_fails():
    artifact = _published(
        _artifact(
            world_rankings={
                "corpus-a": ("strong", "middle", "weak"),
                "corpus-b": ("weak", "middle", "strong"),
            },
            rank_correlation=-1.0,
        )
    )
    verdict = validate_equivalence(artifact)
    assert verdict.failed_requirements == ("external_referent",)
    assert any("below the declared minimum" in failure for failure in verdict.failures)
    assert verdict.computed_rank_correlation == pytest.approx(-1.0)


def test_declared_correlation_must_match_the_declared_rankings():
    verdict = validate_equivalence(_published(_artifact(rank_correlation=0.9)))
    assert verdict.failed_requirements == ("external_referent",)
    assert any("does not match the rankings" in failure for failure in verdict.failures)


def test_unmatched_task_equivalence_and_bare_strata_are_listed_separately():
    base = _artifact()
    broken = base.model_copy(
        update={
            "task_equivalence": base.task_equivalence.model_copy(
                update={
                    "termination": DeclaredMatch(
                        statement="world B terminates on a timeout the other does not have",
                        matched=False,
                    )
                }
            ),
            "scenario_alignment": ScenarioAlignment(
                mode="stratified", statement="strata were never written down"
            ),
        }
    )
    verdict = validate_equivalence(_published(broken))
    assert verdict.failed_requirements == ("task_equivalence", "scenario_alignment")


def test_an_unpublished_artifact_is_not_a_citable_claim():
    verdict = validate_equivalence(_artifact())
    assert verdict.failed_requirements == ("publication",)
    assert any("unpublished" in failure for failure in verdict.failures)


def test_write_and_load_pin_the_artifact_by_its_own_digest(tmp_path: Path):
    path = tmp_path / "equivalence.json"
    published = write_equivalence_artifact(_artifact(), path)
    assert published.published_as.digest == equivalence_digest(_artifact())
    reloaded = load_equivalence_artifact(path)
    assert reloaded == published
    assert validate_equivalence(reloaded).valid
    # Determinism: republishing identical content reproduces the same bytes.
    again = tmp_path / "again.json"
    write_equivalence_artifact(_artifact(), again)
    assert again.read_bytes() == path.read_bytes()


def test_write_refuses_to_repin_edited_content(tmp_path: Path):
    stale = _artifact(digest_value="0" * 64)
    with pytest.raises(TaskContractError, match="clear the digest to republish"):
        write_equivalence_artifact(stale, tmp_path / "stale.json")


def test_load_refuses_an_unknown_artifact_format(tmp_path: Path):
    path = tmp_path / "equivalence.yaml"
    path.write_text("shelf_id: endovascular\n", encoding="utf-8")
    with pytest.raises(TaskContractError, match=r"\.toml or \.json"):
        load_equivalence_artifact(path)


# --- cross-world refusal ----------------------------------------------------


def test_refuse_cross_world_aggregate_without_an_artifact(report: ShelfReport):
    with pytest.raises(ScoreContractError) as excinfo:
        refuse_cross_world_aggregate(report, task_family=FAMILY)
    assert "refusing to aggregate across worlds" in str(excinfo.value)


def test_refuse_cross_world_aggregate_with_an_invalid_artifact(report: ShelfReport):
    with pytest.raises(ScoreContractError, match="fails gate_equivalence"):
        refuse_cross_world_aggregate(
            report,
            task_family=FAMILY,
            equivalence=_published(_artifact(gate=_gate(unit_b="millinewton"))),
        )


def test_refuse_cross_world_aggregate_for_another_shelf(report: ShelfReport):
    other = _published(_artifact()).model_copy(update={"shelf_id": "orthopedic"})
    with pytest.raises(ScoreContractError, match="published for shelf 'orthopedic'"):
        refuse_cross_world_aggregate(report, task_family=FAMILY, equivalence=other)


def test_ranking_is_per_world_until_equivalence_is_published(report: ShelfReport):
    ranking = shelf_ranking(report)
    assert ranking["cross_world"] is None
    assert [world["world_id"] for world in ranking["per_world"]] == ["corpus-a", "corpus-b"]
    for world in ranking["per_world"]:
        assert [entry["rank"] for entry in world["order"]] == [1, 2]
    assert ranking["benches"][0]["task_id"] == BENCH_TASK


def test_a_valid_artifact_unlocks_exactly_one_cross_world_ordering(report: ShelfReport):
    agents = _agents(report, "corpus-a")
    artifact = _published(
        _artifact(
            world_rankings={"corpus-a": tuple(agents), "corpus-b": tuple(agents)},
            referent_ranking=tuple(agents),
        )
    )
    ranking = shelf_ranking(report, equivalence=artifact)
    cross = ranking["cross_world"]
    assert cross is not None
    assert cross["world_pair"] == ["corpus-a", "corpus-b"]
    assert [entry["agent_identity"] for entry in cross["order"]] == agents
    assert cross["order"][0]["world_ranks"] == {"corpus-a": 1, "corpus-b": 1}
    assert cross["excluded_partial_coverage"] == []
    assert cross["equivalence_artifact"]["digest"] == artifact.published_as.digest
    assert ranking["cross_world_refusal"] is None
    # Exactly one ordering: the per-world orderings are unchanged beside it.
    assert len(ranking["per_world"]) == 2
    assert isinstance(cross["order"], list)


# --- CLI --------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="surgeval")
    register(parser.add_subparsers(dest="command"))
    return parser


def _invoke(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    exit_code: int = args.func(args)
    return exit_code


def test_cli_build_rank_and_refuse(
    bundles: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    spec_path = tmp_path / "shelf.toml"
    worlds = "\n".join(
        f"""
[[worlds]]
world_id = "{world.world_id}"
world_kind = "frame-source"
world_pin = "{world.world_pin}"
task_id = "{world.task_id}"
task_family = "{FAMILY}"
"""
        for world in _spec().worlds
    )
    spec_path.write_text(
        f'id = "{SHELF_ID}"\ntitle = "Endovascular"\nmodality = "procedural-video"\n'
        f'{worlds}\n[[benches]]\ntask_id = "{BENCH_TASK}"\nkind = "bench"\n',
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert (
        _invoke(
            [
                "shelf",
                "build",
                str(spec_path),
                "--jobs",
                *[str(path) for path in _all_jobs(bundles)],
                "--out",
                str(out),
            ]
        )
        == 0
    )
    built = capsys.readouterr().out
    assert "world corpus-a: 2 row(s)" in built
    assert f"bench {BENCH_TASK}: 1 row(s)" in built

    assert _invoke(["shelf", "rank", str(out)]) == 0
    ranked = capsys.readouterr().out
    assert "world corpus-a" in ranked
    assert "cross-world: refused" in ranked

    assert _invoke(["shelf", "rank", str(out), "--cross-world"]) == 1
    assert "REFUSED: cross-world ordering requested" in capsys.readouterr().err


def test_cli_rank_unlocks_cross_world_with_a_valid_artifact(
    report: ShelfReport, bundles: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    out = tmp_path / "out"
    build_shelf(_spec(), _all_jobs(bundles), out=out)
    agents = _agents(report, "corpus-a")
    artifact_path = tmp_path / "equivalence.json"
    write_equivalence_artifact(
        _artifact(
            world_rankings={"corpus-a": tuple(agents), "corpus-b": tuple(agents)},
            referent_ranking=tuple(agents),
        ),
        artifact_path,
    )
    assert _invoke(["shelf", "equivalence", "check", str(artifact_path)]) == 0
    assert "equivalence: VALID" in capsys.readouterr().out

    assert _invoke(["shelf", "rank", str(out), "--equivalence", str(artifact_path)]) == 0
    printed = capsys.readouterr().out
    assert "cross-world (corpus-a <-> corpus-b" in printed
    assert "licensed by:" in printed


def test_cli_equivalence_check_lists_failed_requirements(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    path = tmp_path / "broken.json"
    path.write_text(
        json.dumps(_published(_artifact(gate=_gate(unit_b="millinewton"))).model_dump(mode="json")),
        encoding="utf-8",
    )
    assert _invoke(["shelf", "equivalence", "check", str(path)]) == 1
    captured = capsys.readouterr()
    assert "[FAIL] gate_equivalence" in captured.out
    assert "REFUSED: equivalence artifact fails gate_equivalence" in captured.err


def test_cli_build_refuses_a_shelf_whose_bench_never_ran(
    bundles: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    spec_path = tmp_path / "shelf.toml"
    spec_path.write_text(
        f'id = "{SHELF_ID}"\ntitle = "Endovascular"\nmodality = "procedural-video"\n'
        f'\n[[worlds]]\nworld_id = "corpus-a"\nworld_kind = "frame-source"\n'
        f'world_pin = "corpus-a@v1"\ntask_id = "endo-nav-a"\ntask_family = "{FAMILY}"\n'
        f'\n[[benches]]\ntask_id = "{BENCH_TASK}"\nkind = "bench"\n',
        encoding="utf-8",
    )
    assert (
        _invoke(
            [
                "shelf",
                "build",
                str(spec_path),
                "--jobs",
                str(bundles["jobs"]["a-strong"]),
                "--out",
                str(tmp_path / "out"),
            ]
        )
        == 1
    )
    assert "sim rows alone" in capsys.readouterr().err


def test_load_shelf_report_round_trips(
    report: ShelfReport, bundles: dict[str, Any], tmp_path: Path
):
    build_shelf(_spec(), _all_jobs(bundles), out=tmp_path / "rt")
    reloaded = load_shelf_report(tmp_path / "rt")
    assert reloaded.spec == report.spec
    assert [world.rows for world in reloaded.worlds] == [world.rows for world in report.worlds]
    assert reloaded.benches[0].rows[0]["head"] == report.benches[0].rows[0]["head"]
