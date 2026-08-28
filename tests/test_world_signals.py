"""Audited signal surfaces: per-env gate eligibility and its cross-checks.

N4 asks each wrap for a *gate mapping*, and prose cannot be checked. These
tests defend the machine-checkable form of that mapping and, specifically, the
two real defects it was built to catch:

* a citation that points at code the pinned revision does not contain
  (``surrol`` pinned to a monorepo whose ``surrol/`` package does not exist,
  ``sonogym`` citing a filename that was really the directory name);
* a scene borrowing a sibling scene's signal, which would let a wrap publish a
  gate with a real-looking citation for a quantity its env never reports.

The upstream reads behind the catalog data are verified separately, against
fetched pinned trees, by ``scripts/check_world_signals.py`` - not here, because
these tests must stay hermetic.
"""

from __future__ import annotations

from typing import Any

import pytest

from or_audit.errors import TaskContractError
from or_audit.eval.wrap import GateMapping, WrapRequest
from or_audit.install.catalog import (
    AuditedEnv,
    Disposition,
    SignalKind,
    SourceBuildInstall,
    WorldPackage,
    WorldSignal,
    load_catalog,
    world_package,
)

LAPGYM_PIN = "85bf7e05dd088b824794dda0046679df13b13e6e"
SURROL_PIN = "aa430af5ca3ee62a69d677d2c8dfd031efe20204"


def _gate(signal: str, gate_id: str = "g", unit: str = "scaled-N") -> GateMapping:
    """A gate whose unit is the one the catalog actually recorded.

    ``scaled-N``, not ``N``: LapGym's force is a SOFA internal force times a
    scene scaling factor. The first version of this helper said ``N`` and was
    accepted, which is the hole the unit cross-check now closes - a fixture
    publishing a physically false unit is how that lie reaches a scorecard.
    """
    return GateMapping(
        id=gate_id,
        signal=signal,
        fail_when=f"{signal} > 1.0",
        threshold=1.0,
        unit=unit,
        citation="fixture",
    )


def _request(**overrides: Any) -> WrapRequest:
    fields: dict[str, Any] = {
        "env_id": "grasp_lift_touch",
        "task_id": "t",
        "world_pin": LAPGYM_PIN,
        "license": "MIT",
        "world_kind": "sofa",
        "gate_mappings": (_gate("dynamic_force_on_gallbladder"),),
    }
    fields.update(overrides)
    return WrapRequest(**fields)


def _package(**overrides: Any) -> WorldPackage:
    fields: dict[str, Any] = {
        "id": "fixture-world",
        "display_name": "Fixture",
        "domain": "test",
        "engine": "test",
        "disposition": Disposition.WATCH,
        "install": {"strategy": "source-build", "repo": "https://example.invalid/x"},
    }
    fields.update(overrides)
    return WorldPackage.model_validate(fields)


class TestSignalEligibility:
    def test_only_published_physical_signals_can_carry_a_gate(self):
        """The two halves are independent, and both are load-bearing."""
        published_physical = WorldSignal(key="force", kind=SignalKind.PHYSICAL)
        unpublished_physical = WorldSignal(key="force", kind=SignalKind.PHYSICAL, published=False)
        published_geometric = WorldSignal(key="near", kind=SignalKind.GEOMETRIC)
        published_diagnostic = WorldSignal(key="nan_guard", kind=SignalKind.DIAGNOSTIC)

        assert published_physical.gate_eligible
        assert not unpublished_physical.gate_eligible
        assert not published_geometric.gate_eligible
        assert not published_diagnostic.gate_eligible

    def test_a_diagnostic_is_not_a_safety_signal(self):
        """stEVE's real surface: a NaN guard means the run is invalid, not unsafe."""
        steve = world_package("steve")
        (env,) = steve.envs
        (signal,) = env.signals
        assert signal.key == "simulation_error"
        assert signal.kind is SignalKind.DIAGNOSTIC
        assert not env.safety_eligible

    def test_eligibility_is_per_env_not_per_world(self):
        """LapGym is the case that forces this: scenes differ inside one package."""
        lapgym = world_package("lapgym")
        surfaces = {env.env_id: {s.key for s in env.gate_signals} for env in lapgym.envs}
        assert "dynamic_force_on_gallbladder" in surfaces["grasp_lift_touch"]
        assert "dynamic_force_on_gallbladder" not in surfaces["tissue_dissection"]
        assert "dynamic_force_on_gallbladder" not in surfaces["pick_and_place"]


class TestAbsenceMarkers:
    """An empty signal surface has to be *stated*, and the statement checked.

    ``surgical-gym`` records one audited env that publishes nothing. Recording
    that by leaving ``signals`` empty made the citation checker fetch the tree,
    read not one line, and print "0 signal(s) resolved" - a zero-byte file was
    enough to earn it. ``absence_markers`` turns the emptiness into line-pinned
    citations that ``scripts/check_world_signals.py`` resolves at the pin.
    """

    def test_the_empty_surface_is_stated_with_pinned_lines(self):
        (env,) = world_package("surgical-gym").envs
        assert env.signals == ()
        assert {(m.key, m.line) for m in env.absence_markers} == {
            ("rew_buf", 266),
            ("progress_buf", 273),
        }

    def test_a_marker_does_not_make_the_env_gate_eligible(self):
        """A marker records what upstream computes and never publishes."""
        (env,) = world_package("surgical-gym").envs
        assert env.gate_signals == ()
        assert not env.safety_eligible
        assert all(not marker.published for marker in env.absence_markers)

    def test_a_marker_that_pins_no_line_is_refused(self):
        """An unchecked absence claim is the same defect in better clothes."""
        with pytest.raises(TaskContractError, match="pin no line"):
            AuditedEnv(
                env_id="e",
                path="e.py",
                absence_markers=(
                    WorldSignal(key="rew_buf", kind=SignalKind.BOOKKEEPING, published=False),
                ),
            )

    def test_a_published_marker_is_refused(self):
        with pytest.raises(TaskContractError, match="marked published"):
            AuditedEnv(
                env_id="e",
                path="e.py",
                absence_markers=(WorldSignal(key="rew_buf", kind=SignalKind.BOOKKEEPING, line=1),),
            )

    def test_markers_alongside_signals_are_refused(self):
        """An empty surface is not empty if the env records a published key."""
        with pytest.raises(TaskContractError, match="records both signals"):
            AuditedEnv(
                env_id="e",
                path="e.py",
                signals=(WorldSignal(key="is_success", kind=SignalKind.BOOKKEEPING, line=2),),
                absence_markers=(
                    WorldSignal(
                        key="rew_buf", kind=SignalKind.BOOKKEEPING, line=1, published=False
                    ),
                ),
            )

    def test_a_gate_cannot_bind_a_marker(self):
        """The field must not become a back door to the gate surface."""
        with pytest.raises(TaskContractError, match="not a signal this env publishes"):
            _request(
                env_id="surgicalgym.tasks.psm:PSM",
                world_pin="57fd88d296f63cc9bebc30c5b332ceba70a04ff6",
                world_kind="gym",
                gate_mappings=(_gate("rew_buf"),),
            )


class TestMetricsOnlyConsistency:
    def test_a_safety_claim_needs_a_gate_eligible_signal(self):
        with pytest.raises(TaskContractError, match="none of its audited"):
            _package(
                metrics_only=False,
                envs=(
                    AuditedEnv(
                        env_id="e",
                        path="e.py",
                        signals=(WorldSignal(key="reward", kind=SignalKind.BOOKKEEPING),),
                    ),
                ),
            )

    def test_under_claiming_stays_legal(self):
        """metrics_only=true over a physical signal is the safe error, and is
        the honest state of a world whose scenes are not all read yet."""
        pkg = _package(
            metrics_only=True,
            envs=(
                AuditedEnv(
                    env_id="e",
                    path="e.py",
                    signals=(WorldSignal(key="force", kind=SignalKind.PHYSICAL, unit="N"),),
                ),
            ),
        )
        assert pkg.metrics_only
        assert pkg.gate_eligible_envs == ("e",)

    def test_an_unaudited_world_constrains_nothing(self):
        pkg = _package(metrics_only=False)
        assert pkg.envs == ()
        assert pkg.gate_eligible_envs == ()


class TestDeterminismEvidence:
    def test_a_measured_class_must_name_its_measurement(self):
        with pytest.raises(TaskContractError, match="no determinism_evidence"):
            _package(determinism="bitwise")

    def test_unmeasured_carries_the_blocker_instead(self):
        """Every wrap target records why it is unmeasured, not just that it is."""
        for pkg in load_catalog().worlds:
            if pkg.disposition not in {Disposition.SHIPPED, Disposition.WRAP}:
                continue
            assert pkg.determinism.value == "unmeasured", pkg.id
            assert pkg.determinism_evidence.strip(), f"{pkg.id} records no blocker"
            assert "nmeasured" in pkg.determinism_evidence, pkg.id


class TestGateBinding:
    def test_a_gate_on_an_audited_signal_is_accepted(self):
        assert _request().gate_mappings[0].signal == "dynamic_force_on_gallbladder"

    def test_a_scene_cannot_borrow_a_sibling_scenes_signal(self):
        """The advisory case: tissue_dissection claiming the gallbladder force."""
        with pytest.raises(TaskContractError, match="publishes in env 'grasp_lift_touch'"):
            _request(env_id="tissue_dissection")

    def test_a_signal_absent_from_an_audited_env_is_refused_with_its_surface(self):
        with pytest.raises(TaskContractError, match="gate-eligible signals there"):
            _request(gate_mappings=(_gate("invented_force"),))

    def test_a_geometric_signal_cannot_carry_a_hard_gate(self):
        with pytest.raises(TaskContractError, match="audited as geometric"):
            _request(
                env_id="robot_US_guided_surgery",
                world_pin="e67be58334d1a5274f0913af36f56e4b0b7ffe5a",
                world_kind="isaac-lab",
                gate_mappings=(_gate("cost"),),
            )

    def test_an_env_that_publishes_nothing_refuses_every_gate(self):
        """surgical-gym: audited and empty, so no gate can bind at all."""
        with pytest.raises(TaskContractError, match="not a signal this env publishes"):
            _request(
                env_id="surgicalgym.tasks.psm:PSM",
                world_pin="57fd88d296f63cc9bebc30c5b332ceba70a04ff6",
                world_kind="gym",
                gate_mappings=(_gate("contact_force_n"),),
            )

    def test_an_uncatalogued_revision_is_self_service(self):
        """A third party wrapping their own world is not constrained by us."""
        request = _request(
            env_id="ThirdParty/Env-v0",
            world_pin="d" * 40,
            world_kind="gym",
            gate_mappings=(_gate("my_own_force"),),
        )
        assert request.gated

    def test_an_unaudited_scene_of_a_known_world_stays_self_service(self):
        """Only *borrowing* is refused; genuinely new signals are the author's."""
        request = _request(env_id="magnetic_continuum_robot", gate_mappings=(_gate("tip_force"),))
        assert request.gated

    def test_env_matching_is_component_wise_not_substring(self):
        """A lookalike name must not inherit a surface it may not have."""
        with pytest.raises(TaskContractError, match="not in 'grasp_lift_touch_hard'"):
            _request(env_id="grasp_lift_touch_hard")

    def test_a_namespaced_env_id_matches_its_audited_scene(self):
        assert _request(env_id="LapGym/grasp_lift_touch").gated

    def test_metrics_only_wraps_skip_the_check_entirely(self):
        request = _request(metrics_only=True, gate_mappings=())
        assert not request.gated


class TestConstructionConditions:
    """A signal whose kind depends on construction must have it pinned."""

    def test_the_condition_must_be_declared(self):
        with pytest.raises(TaskContractError, match="only a physical measurement when"):
            _request(
                env_id="tissue_dissection",
                gate_mappings=(_gate("collision_with_board"),),
            )

    def test_the_wrong_value_is_refused(self):
        with pytest.raises(TaskContractError, match="with_board_collision=True"):
            _request(
                env_id="tissue_dissection",
                gate_mappings=(_gate("collision_with_board"),),
                parameters={"with_board_collision": False},
            )

    def test_pinning_the_condition_accepts(self):
        request = _request(
            env_id="tissue_dissection",
            gate_mappings=(_gate("collision_with_board", unit="contacts"),),
            parameters={"with_board_collision": True},
        )
        assert request.parameters["with_board_collision"] is True

    def test_the_shipped_lapgym_condition_is_recorded(self):
        """LapGym line 394 falls back to a pose predicate under the same key."""
        env = world_package("lapgym").audited_env("tissue_dissection")
        assert env is not None
        (board,) = [s for s in env.signals if s.key == "collision_with_board"]
        assert board.requires_parameters == {"with_board_collision": True}


class TestAuditedUnitIsPublished:
    """The published unit must be the audited unit, exactly.

    A bound number with an unbound unit is the same defect as an unbound
    number: ``dynamic_force_on_gallbladder`` is a SOFA internal force times a
    scene scaling factor, so publishing a threshold on it as ``N`` states a
    quantity the engine never produced. §2.6 gate equivalence compares gates
    by unit, so a false unit also makes a gate falsely comparable across
    worlds - it would line up against a real newton reading somewhere else.
    """

    def test_a_false_unit_is_refused(self):
        with pytest.raises(TaskContractError, match="audited reading of that signal is in"):
            _request(gate_mappings=(_gate("dynamic_force_on_gallbladder", unit="N"),))

    def test_the_audited_unit_is_accepted(self):
        request = _request(gate_mappings=(_gate("dynamic_force_on_gallbladder", unit="scaled-N"),))
        assert request.gate_mappings[0].unit == "scaled-N"

    def test_a_count_may_not_be_published_as_a_force(self):
        with pytest.raises(TaskContractError, match="physically false quantity"):
            _request(
                env_id="pick_and_place",
                gate_mappings=(_gate("gripper_jaw_peg_collisions", unit="N"),),
            )

    def test_a_gate_eligible_signal_must_record_a_unit(self):
        """Otherwise the cross-check has nothing to compare and passes silently."""
        with pytest.raises(TaskContractError, match="record no unit"):
            AuditedEnv(
                env_id="e",
                path="e.py",
                signals=(WorldSignal(key="force", kind=SignalKind.PHYSICAL),),
            )

    def test_a_non_physical_signal_needs_no_unit(self):
        """Only gate-eligible signals carry the requirement."""
        env = AuditedEnv(
            env_id="e",
            path="e.py",
            signals=(
                WorldSignal(key="progress", kind=SignalKind.BOOKKEEPING),
                WorldSignal(key="nan_guard", kind=SignalKind.DIAGNOSTIC),
            ),
        )
        assert not env.safety_eligible

    def test_every_shipped_gate_eligible_signal_has_a_unit(self):
        for pkg in load_catalog().worlds:
            for env in pkg.envs:
                for signal in env.gate_signals:
                    assert signal.unit.strip(), f"{pkg.id}/{env.env_id}/{signal.key}"

    def test_a_task_gate_unit_must_match_its_basis_unit(self):
        """The same divergence inside a hand-written package."""
        from or_audit.eval.task import GateSpec

        with pytest.raises(TaskContractError, match="but its basis is stated in"):
            GateSpec.model_validate(
                {
                    "id": "f",
                    "kind": "threshold",
                    "realization": "scalar-dsl",
                    "inputs": {"f": "info.f"},
                    "maps_to": "unsafe",
                    "fail_when": "f > 1.5",
                    "threshold": 1.5,
                    "unit": "N",
                    "threshold_basis": {"value": 1.5, "unit": "mmHg", "citation": "c"},
                }
            )


class TestShippedCatalogAudit:
    def test_every_wrap_target_records_an_audited_env(self):
        """A wrap target with no read env is an unaudited gate surface."""
        unaudited = [
            pkg.id
            for pkg in load_catalog().worlds
            if pkg.disposition is Disposition.WRAP and not pkg.envs
        ]
        assert unaudited == [], f"wrap rows with no audited env: {unaudited}"

    def test_every_audited_signal_cites_a_path_and_a_line(self):
        for pkg in load_catalog().worlds:
            for env in pkg.envs:
                assert env.path, f"{pkg.id}/{env.env_id} has no path"
                for signal in env.signals:
                    assert signal.line > 0, f"{pkg.id}/{env.env_id}/{signal.key} has no line"

    def test_the_surrol_pin_is_the_revision_that_holds_the_world(self):
        """Regression for the pin that named a repo but not a world.

        The old pin was HEAD of the default branch, a monorepo with six
        divergent vendored copies of ``surrol/`` and no top-level package, so
        the cited paths resolved to nothing and the pin identified no world.
        """
        surrol = world_package("surrol")
        assert surrol.world_pin == SURROL_PIN
        install = surrol.install
        assert isinstance(install, SourceBuildInstall)
        assert install.commit == SURROL_PIN
        (env,) = surrol.envs
        assert env.path == "surrol/gym/surrol_env.py"
        assert "PIN CORRECTED" in surrol.notes

    def test_surrols_contact_call_is_recorded_as_unpublished(self):
        """The distinction that keeps it metrics-only: computed, not published."""
        (env,) = world_package("surrol").envs
        contact = next(s for s in env.signals if s.key == "getContactPoints")
        assert contact.kind is SignalKind.PHYSICAL
        assert not contact.published
        assert not env.safety_eligible

    def test_the_sonogym_citation_names_the_real_file(self):
        """Regression for a citation that transcribed the directory name."""
        (env,) = world_package("sonogym").envs
        assert env.path.endswith("robotic_US_guided_surgery.py")
        assert "robot_US_guided_surgery/robotic_US" in env.path

    def test_the_status_doc_counts_match_the_catalog(self):
        """Two files, one claim - the pattern the shelf rule already uses.

        ``NEXT_STATUS.md`` states the audit's size, and a document whose whole
        subject is numbers being true cannot carry a stale one. The first draft
        of that row said 12 envs / 16 signals against a real 10 / 17.
        """
        import re
        from pathlib import Path

        catalog = load_catalog()
        envs = sum(len(pkg.envs) for pkg in catalog.worlds)
        signals = sum(len(env.signals) for pkg in catalog.worlds for env in pkg.envs)
        status = Path("docs/NEXT_STATUS.md").read_text(encoding="utf-8")
        match = re.search(r"(\d+) audited envs, (\d+) signals", status)
        assert match is not None, "NEXT_STATUS.md no longer states the audit size"
        assert (int(match.group(1)), int(match.group(2))) == (envs, signals)

    def test_only_lapgym_can_host_a_hard_gate_today(self):
        """The audit's headline finding, pinned so a regression is visible.

        If another world becomes gate-eligible this test fails, which is the
        point: that is a claim change and it should be deliberate.
        """
        eligible = {
            pkg.id
            for pkg in load_catalog().worlds
            if pkg.disposition in {Disposition.SHIPPED, Disposition.WRAP} and pkg.gate_eligible_envs
        }
        assert eligible == {"lapgym"}


class TestCitedThresholdIsEnforced:
    """The cited number and the enforced number must be the same number.

    Before this check both authoring paths accepted a gate that cited a real
    normative source at 1.5 N and enforced ``> 999``: a gate that can never
    fire, wearing a citation, published as a safety claim. The scorecard, the
    ``wrap.json`` record, and the rendered verifier docstring all displayed
    the cited 1.5, so nothing downstream could catch it.
    """

    def test_wrap_refuses_a_predicate_that_enforces_another_number(self):
        with pytest.raises(TaskContractError, match=r"enforces \[999\.0\]"):
            GateMapping(
                id="g",
                signal="contact_force_n",
                fail_when="contact_force_n > 999",
                threshold=1.5,
                unit="N",
                citation="ORBIT needle handover envelope v1, Table 2",
            )

    def test_task_spec_refuses_the_same_divergence(self):
        from or_audit.eval.task import GateSpec

        with pytest.raises(TaskContractError, match=r"enforces \[999\.0\]"):
            GateSpec.model_validate(
                {
                    "id": "force",
                    "kind": "threshold",
                    "realization": "scalar-dsl",
                    "inputs": {"contact_force_n": "info.contact_force_n"},
                    "maps_to": "unsafe",
                    "fail_when": "contact_force_n > 999",
                    "threshold": 1.5,
                    # The unit is compared against threshold_basis unconditionally
                    # now, so an omitted one raises before the divergence under test.
                    "unit": "N",
                    "threshold_basis": {
                        "value": 1.5,
                        "unit": "N",
                        "citation": "ORBIT Table 2",
                    },
                }
            )

    def test_a_threshold_must_match_its_own_basis(self):
        from or_audit.eval.task import GateSpec

        with pytest.raises(TaskContractError, match="does not match its own basis"):
            GateSpec.model_validate(
                {
                    "id": "force",
                    "kind": "threshold",
                    "realization": "scalar-dsl",
                    "inputs": {"f": "info.f"},
                    "maps_to": "unsafe",
                    "fail_when": "f > 1.5",
                    "threshold": 1.5,
                    "threshold_basis": {"value": 99.0, "unit": "N", "citation": "c"},
                }
            )

    def test_an_uncited_inline_number_is_refused(self):
        with pytest.raises(TaskContractError, match="is uncited"):
            GateMapping(id="g", signal="contact_force_n", fail_when="contact_force_n > 2.0")

    def test_a_declared_threshold_must_be_used(self):
        with pytest.raises(TaskContractError, match="decoration"):
            GateMapping(
                id="g",
                signal="unsafe_flag",
                fail_when="unsafe_flag",
                threshold=1.5,
                unit="N",
                citation="c",
            )

    def test_a_boolean_gate_is_not_read_as_the_number_one(self):
        """``bool`` subclasses ``int``; a naive check refuses this honest gate."""
        mapping = GateMapping(id="g", signal="unsafe_flag", fail_when="unsafe_flag == true")
        assert mapping.threshold is None

    def test_a_negative_bound_is_sign_folded(self):
        """Isaac Lab's ``minimum_height=-0.05`` shape: UnaryOp over a positive."""
        mapping = GateMapping(
            id="g",
            signal="root_height",
            fail_when="root_height < -0.05",
            threshold=-0.05,
            unit="m",
            citation="ORBIT lift_env_cfg.py:179",
        )
        assert mapping.threshold == -0.05

    def test_a_two_sided_predicate_cannot_hide_a_second_bound(self):
        """The bug a naive de-duplication would have introduced."""
        with pytest.raises(TaskContractError, match=r"enforces \[0\.05\]"):
            GateMapping(
                id="g",
                signal="offset",
                fail_when="offset > 0.05 or offset < -0.05",
                threshold=-0.05,
                unit="m",
                citation="c",
            )

    def test_reversed_operand_order_still_binds(self):
        mapping = GateMapping(
            id="g",
            signal="force",
            fail_when="1.5 < force",
            threshold=1.5,
            unit="N",
            citation="c",
        )
        assert mapping.numeric
