"""Task-owned v0.3 evaluation contracts with deterministic v0.2 normalization."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from typing import Annotated, Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from or_audit.errors import TaskContractError
from or_audit.eval.contracts import (
    GateProjectionPolicy,
    HarnessSpec,
    InteractionMode,
    InterfaceSpec,
    MetricDirection,
    MetricKind,
    PerturbationSpec,
    ScenarioSpec,
    legacy_interface,
)
from or_audit.eval.enums import (
    AttestationLevel,
    GateKind,
    OracleKind,
    PhiClass,
    PortId,
    ProjectionId,
    SubjectKind,
    VerifierRealizationKind,
    WorldKind,
)
from or_audit.eval.worlds import (
    WorldCapabilities,
    resolve_world_capabilities,
    world_kind_key,
)

Slug = Annotated[
    str, StringConstraints(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9_-]*$")
]
NonEmpty = Annotated[str, StringConstraints(min_length=1, max_length=200)]
Instruction = Annotated[str, StringConstraints(min_length=1, max_length=20_000)]
#: 64-char lowercase hex content pin, matched where a digest is declared.
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

_BOOLEAN_METRICS = {
    "abstained",
    "diverged",
    "next_step_correct",
    "outcome_correct",
    "raw_success",
    "release_audit_passed",
    "safe_success",
    "failure_detected",
    "recovered",
    "safe_abandonment",
    "unsafe_persistence",
    "harm_after_failure",
    "handoff_accepted",
}


def numeric_boundaries(tree: ast.Expression) -> set[float]:
    """Every numeric boundary an expression compares against, sign folded.

    Reads the operands of each :class:`ast.Compare` rather than walking every
    constant, so each boundary is counted exactly once. A blind constant walk
    double-counts a negative bound - ``-0.05`` yields both the ``UnaryOp`` and
    its inner ``0.05`` - and de-duplicating afterwards silently swallows a
    genuine two-sided predicate like ``x > 0.05 or x < -0.05``, which enforces
    two numbers while citing one.

    ``bool`` is excluded explicitly: it subclasses ``int``, so a naive numeric
    check reads ``True`` in ``unsafe == true`` as the number 1 and would
    refuse an honest boolean gate.
    """
    found: set[float] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for operand in (node.left, *node.comparators):
            value = _literal_number(operand)
            if value is not None:
                found.add(value)
    return found


def _literal_number(node: ast.expr) -> float | None:
    """A numeric literal, sign folded, or ``None`` if the operand is not one.

    Isaac Lab's ``minimum_height=-0.05`` is why the fold exists: a negative
    bound is a ``UnaryOp`` over a positive constant, never a negative literal.

    The ``bool`` rejection is first and deliberate: it subclasses ``int``, so
    reading it as a number turns ``unsafe == true`` into a comparison against
    1 and would refuse an honest boolean gate.
    """
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        inner = _literal_number(node.operand)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            return None
        if isinstance(value, int | float):
            return float(value)
    return None


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TaskMetadata(_Frozen):
    """Human-facing metadata; tags remain open search terms."""

    title: NonEmpty
    modality: NonEmpty
    tags: tuple[str, ...] = ()
    safety_critical: bool = True


class PortSpec(_Frozen):
    """Deprecated v0.2 port retained only as a compatibility input."""

    id: PortId
    observation: str = ""
    action: str = ""
    prediction: str = ""

    @model_validator(mode="after")
    def _video_predict_names_a_schema(self) -> Self:
        if self.id is PortId.VIDEO_PREDICT and not self.prediction:
            raise TaskContractError("a video-predict port must name prediction")
        return self


class SubjectSpec(_Frozen):
    kind: SubjectKind


class PhiSpec(_Frozen):
    class_: PhiClass = Field(alias="class")
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class WorldSpec(_Frozen):
    """Pinned procedural world and its task-owned inputs.

    ``kind`` is open (``WorldKind | Slug``) so a third-party world publishes
    through a plugin-registered adapter without a core release. Eligibility —
    physics oracle, closed-loop interaction, which fields the world requires —
    comes from :mod:`or_audit.eval.worlds`, never from an enum-set branch here.
    """

    kind: WorldKind | Slug
    gym_id: str = ""
    world_pin: str = ""
    parameters: dict[str, bool | int | float | str] = Field(default_factory=dict)
    #: Adapter identity this package was authored against: ``module:symbol`` of
    #: the world adapter factory. Declared like ``StreamSpec.adapter``, verified
    #: at load against the registered adapter, and stamped into the head-covered
    #: ``JobResult.world_engine`` — so a patched or swapped third-party adapter
    #: cannot run under the same task and world pin unnoticed.
    adapter: str = ""
    #: SHA-256 of the adapter's module content. Required with ``adapter``.
    adapter_digest: str = ""
    #: World metadata a package declares so it stays loadable where the adapter
    #: is absent. Cross-checked against an installed adapter; a task cannot
    #: grant itself eligibility the adapter withholds.
    capabilities: WorldCapabilities | None = None
    #: Tier-0 honesty label (§2.2): this wrapped world's instrumentation does
    #: not report safety state, so the package ships metrics-only — no hard
    #: gates, not safety-critical, and every artifact says so. Whether a world
    #: *actually* reports the state its gates bind to is verified per env by the
    #: conformance suite, not assumed from the world kind.
    metrics_only: bool = False
    synthetic_stub: bool = Field(
        default=False,
        description=(
            "explicitly permit a synthetic stand-in when no real backend is attached; "
            "artifacts are stamped and RL export is refused"
        ),
    )
    n_eval_episodes: Annotated[int, Field(ge=1, le=10_000)] = 30
    seed_policy: str = "deterministic-eval-30"
    inputs_path: str = ""
    labels_path: str = ""
    contract_path: str = ""

    @field_validator("kind", mode="before")
    @classmethod
    def _normalize_kind(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.replace("_", "-").lower()
            try:
                return WorldKind(normalized)
            except ValueError:
                return normalized
        return value

    @property
    def kind_key(self) -> str:
        """Registry / provenance key for this world kind."""
        return world_kind_key(self.kind)

    @property
    def resolved_capabilities(self) -> WorldCapabilities:
        """Capabilities the kernel gates on for this world."""
        return resolve_world_capabilities(self.kind, self.capabilities)

    @model_validator(mode="after")
    def _required_paths(self) -> Self:
        capabilities = self.resolved_capabilities
        if capabilities.requires_gym_id and not self.gym_id:
            raise TaskContractError(f"a {self.kind_key} world must name gym_id")
        if capabilities.requires_contract and not self.contract_path:
            raise TaskContractError(f"a {self.kind_key} world must name contract_path")
        return self

    @model_validator(mode="after")
    def _adapter_pin_is_complete(self) -> Self:
        if bool(self.adapter) != bool(self.adapter_digest):
            raise TaskContractError(
                "environment.adapter and environment.adapter_digest must be declared "
                "together: an adapter id without a content digest is not a pin"
            )
        if self.adapter_digest and not _SHA256_HEX.fullmatch(self.adapter_digest):
            raise TaskContractError(
                f"environment.adapter_digest must be 64 lowercase hex characters, "
                f"got {self.adapter_digest!r}"
            )
        return self


class AgentSpec(_Frozen):
    kinds: tuple[Slug, ...]
    action_space: str = ""
    timeout_sec: Annotated[float, Field(gt=0.0)] = 120.0

    @model_validator(mode="after")
    def _at_least_one_kind(self) -> Self:
        if not self.kinds:
            raise TaskContractError("a task must accept at least one agent kind")
        return self


class OracleSpec(_Frozen):
    kind: OracleKind


class CalibrationSpec(_Frozen):
    """Reference to a package-relative calibration/validation artifact.

    The artifact file must ship inside the task package and its byte digest
    is verified at load time, so a learned or threshold-tuned realization
    cannot silently tune itself on the evaluation set.
    """

    method: Slug = "fixed"
    artifact: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    digest: Annotated[str, StringConstraints(min_length=64, max_length=64)]
    note: str = ""


class ThresholdBasis(_Frozen):
    """Typed basis for a numeric safety threshold.

    A safety threshold must not be an unexplained literal in ``fail_when``.
    Where a gate declares a ``threshold``, a basis must cite a normative
    source or reference a verified calibration artifact, so the number is
    versioned, reviewable, and non-heuristic. ``value``/``unit`` restate the
    threshold; ``owner``/``version`` attribute who set it and when.
    """

    value: float
    unit: str
    citation: str = ""
    calibration: CalibrationSpec | None = None
    owner: str = ""
    version: str = ""

    @model_validator(mode="after")
    def _has_basis(self) -> Self:
        if not self.citation and self.calibration is None:
            raise TaskContractError(
                "a threshold basis must cite a normative source or reference a "
                "verified calibration artifact"
            )
        return self


class GateSpec(_Frozen):
    id: Slug
    #: Named evidence bindings for a ``scalar-dsl`` realization: signal name
    #: -> locator resolved against the scoring context (dotted path or
    #: ``task://`` file). Every binding is resolved and hashed by the kernel.
    inputs: dict[str, str] = Field(default_factory=dict)
    #: Legacy single-signal locator; folded into ``inputs`` as its leaf name.
    source: str = ""
    #: Absent-defaults for oracle/env boolean bindings whose environment
    #: contract defines "absent == a known value" (e.g. a divergence flag that
    #: is authoritatively ``False`` when unset). Names must be declared inputs.
    #: The default is kernel-digested like any real binding; absence with no
    #: declared default still abstains (NOT_ASSESSABLE).
    input_defaults: dict[str, bool] = Field(default_factory=dict)
    fail_when: str = ""
    maps_to: str = ""
    kind: GateKind | Slug = GateKind.CUSTOM
    threshold: float | None = None
    unit: str = ""
    #: Declared class of evidence behind this gate (extensible: enum | slug).
    realization: VerifierRealizationKind | Slug = VerifierRealizationKind.SCALAR_DSL
    #: Human description of the oracle/evidence path a non-DSL gate uses.
    provenance: str = ""
    calibration: CalibrationSpec | None = None
    abstain_ok: bool = True
    #: Typed normative/calibrated basis for a numeric safety threshold.
    threshold_basis: ThresholdBasis | None = None

    @field_validator("kind", mode="before")
    @classmethod
    def _normalize_kind(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.replace("_", "-").lower()
            try:
                return GateKind(normalized)
            except ValueError:
                return normalized
        return value

    @field_validator("realization", mode="before")
    @classmethod
    def _normalize_realization(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.replace("_", "-").lower()
            try:
                return VerifierRealizationKind(normalized)
            except ValueError:
                return normalized
        return value

    @model_validator(mode="after")
    def _realization_contract(self) -> Self:
        bindings = self.evidence_bindings
        is_scalar = (
            isinstance(self.realization, VerifierRealizationKind)
            and self.realization is VerifierRealizationKind.SCALAR_DSL
        )
        if is_scalar:
            if not self.fail_when or not bindings:
                raise TaskContractError(
                    f"gate {self.id}: scalar-dsl realization requires fail_when and at "
                    "least one evidence input; opaque self-reported gates must declare "
                    "a non-DSL realization"
                )
        else:
            if not self.provenance:
                raise TaskContractError(
                    f"gate {self.id}: non-DSL realization {self.realization} requires "
                    "provenance describing the oracle evidence path"
                )
            if not bindings:
                raise TaskContractError(
                    f"gate {self.id}: non-DSL realization {self.realization} requires "
                    "declared evidence inputs the kernel resolves and hashes"
                )
        if (
            isinstance(self.realization, VerifierRealizationKind)
            and self.realization is VerifierRealizationKind.LEARNED
            and self.calibration is None
        ):
            raise TaskContractError(
                f"gate {self.id}: learned realization requires a calibration artifact"
            )
        unknown_defaults = set(self.input_defaults) - set(bindings)
        if unknown_defaults:
            raise TaskContractError(
                f"gate {self.id}: input_defaults reference undeclared input(s) "
                f"{sorted(unknown_defaults)}"
            )
        if self.threshold is not None and self.threshold_basis is None:
            raise TaskContractError(
                f"gate {self.id}: threshold {self.threshold} lacks a threshold_basis "
                "(cite a normative source or reference a calibration artifact)"
            )
        if (
            self.threshold is not None
            and self.threshold_basis is not None
            and self.threshold_basis.value != self.threshold
        ):
            raise TaskContractError(
                f"gate {self.id}: threshold {self.threshold} does not match its own basis "
                f"value {self.threshold_basis.value}. The basis is what the number is "
                "justified by, so a gate that cites one number and declares another has no "
                "justified threshold at all."
            )
        if (
            self.threshold_basis is not None
            and self.unit
            and self.threshold_basis.unit
            and self.unit != self.threshold_basis.unit
        ):
            raise TaskContractError(
                f"gate {self.id}: publishes unit {self.unit!r} but its basis is stated in "
                f"{self.threshold_basis.unit!r}. A threshold is only a physical quantity in "
                "one unit; two disagreeing units mean the cited source does not justify the "
                "number as published. Fix: state both in the unit the evidence is measured "
                "in, or convert the number explicitly and cite the conversion."
            )
        if is_scalar and self.fail_when:
            self._validate_fail_when_names(bindings)
            self._assert_threshold_is_enforced()
        return self

    def _validate_fail_when_names(self, bindings: dict[str, str]) -> None:
        """Reject a typo in ``fail_when`` at load time (never a silent PASS)."""
        try:
            tree = ast.parse(self.fail_when, mode="eval")
        except SyntaxError as exc:
            raise TaskContractError(
                f"gate {self.id}: fail_when is not a valid expression: {exc}"
            ) from exc
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
        keywords = {"true", "false", "null", "none"}
        unknown = names - keywords - set(bindings)
        if unknown:
            raise TaskContractError(
                f"gate {self.id}: fail_when references unknown signal(s) "
                f"{sorted(unknown)}; must be literal keywords or declared inputs"
            )

    def _assert_threshold_is_enforced(self) -> None:
        """The cited number must be the number ``fail_when`` actually applies.

        A gate carries three numbers that can drift apart: ``threshold``, its
        ``threshold_basis.value``, and whatever literal the predicate compares
        against. Only the last one changes a verdict. Before this check, a
        package could cite a real normative source at 1.5 N and enforce
        ``contact_force_n > 999`` - a gate that never fires, wearing a
        citation, published as a safety claim. Scorecards, ``wrap.json``, and
        the rendered verifier docstring would all have shown 1.5.
        """
        tree = ast.parse(self.fail_when, mode="eval")
        literals = numeric_boundaries(tree)
        if self.threshold is None:
            if literals:
                raise TaskContractError(
                    f"gate {self.id}: fail_when {self.fail_when!r} compares against "
                    f"{sorted(literals)} but the gate declares no threshold, so the number "
                    "deciding the verdict is uncited. Declare it as threshold with a "
                    "threshold_basis, or write a boolean gate over the signal alone."
                )
            return
        mismatched = sorted(value for value in literals if value != self.threshold)
        if mismatched:
            raise TaskContractError(
                f"gate {self.id}: threshold {self.threshold} is cited, but fail_when "
                f"{self.fail_when!r} enforces {mismatched}. The published threshold would "
                "describe a boundary no run applies. Fix: compare against "
                f"{self.threshold}, or declare the number the predicate uses."
            )
        if not literals:
            raise TaskContractError(
                f"gate {self.id}: threshold {self.threshold} is declared, but fail_when "
                f"{self.fail_when!r} never compares against a number, so the threshold and "
                "its basis are decoration. Fix: use it in the predicate, or drop both and "
                "write a boolean gate."
            )

    @property
    def evidence_bindings(self) -> dict[str, str]:
        """Named signal -> locator bindings, including the legacy ``source``."""
        bindings = dict(self.inputs)
        if self.source:
            short = self.source.rsplit(".", 1)[-1] or self.source
            bindings.setdefault(short, self.source)
        return bindings


class MetricSpec(_Frozen):
    """Typed metric declaration with kind-specific aggregation metadata."""

    id: Slug
    source: str = ""
    kind: MetricKind = MetricKind.CONTINUOUS
    unit: str = ""
    direction: MetricDirection = MetricDirection.NEUTRAL
    categories: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _categorical_has_categories(self) -> Self:
        if self.kind is MetricKind.CATEGORICAL and not self.categories:
            raise TaskContractError(f"categorical metric {self.id} must declare categories")
        return self


class VerifierSpec(_Frozen):
    abstain_ok: bool
    headline: Slug
    gates: tuple[GateSpec, ...] = ()
    metrics: tuple[MetricSpec, ...] = ()
    entrypoint: str = ""

    @model_validator(mode="after")
    def _shape(self) -> Self:
        metric_ids = [metric.id for metric in self.metrics]
        gate_ids = [gate.id for gate in self.gates]
        if len(set(metric_ids)) != len(metric_ids):
            raise TaskContractError("verifier metrics must have unique ids")
        if len(set(gate_ids)) != len(gate_ids):
            raise TaskContractError("verifier gates must have unique ids")
        if self.headline not in metric_ids:
            raise TaskContractError(
                f"headline {self.headline!r} is not a declared metric; "
                "the headline cannot be an implicit scalar"
            )
        return self


class AttestationSpec(_Frozen):
    level: AttestationLevel = AttestationLevel.NONE


class DecisionSpec(_Frozen):
    emit_human_determination: bool = False


class ProjectionSpec(_Frozen):
    """Versioned declarative projection recomputed from an authoritative vector."""

    id: ProjectionId | Slug
    version: Annotated[str, StringConstraints(min_length=1, max_length=32)] = "0"
    source_metric: Slug = "raw_success"
    require_false_metrics: tuple[Slug, ...] = ("diverged",)
    gate_failure: GateProjectionPolicy = GateProjectionPolicy.ZERO
    gate_unassessable: GateProjectionPolicy = GateProjectionPolicy.REFUSE
    true_value: float = 1.0
    false_value: float = 0.0

    @field_validator("id", mode="before")
    @classmethod
    def _known_projection_id(cls, value: Any) -> Any:
        try:
            return ProjectionId(value)
        except ValueError:
            return value

    @property
    def rule_digest(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def identity(self) -> str:
        projection_id = self.id.value if isinstance(self.id, ProjectionId) else self.id
        return f"{projection_id}@{self.version}+{self.rule_digest}"


class TaskSpec(_Frozen):
    """Canonical task model; v0.2 port packages normalize before validation."""

    format_version: Annotated[str, StringConstraints(min_length=1, max_length=16)]
    id: Slug
    task_version: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    metadata: TaskMetadata
    subject: SubjectSpec
    phi: PhiSpec
    environment: WorldSpec
    interface: InterfaceSpec
    harness: HarnessSpec
    scenarios: tuple[ScenarioSpec, ...] = ()
    perturbations: tuple[PerturbationSpec, ...] = ()
    port: PortSpec | None = None
    agent: AgentSpec
    oracle: OracleSpec
    verifier: VerifierSpec
    attestation: AttestationSpec = AttestationSpec()
    decision: DecisionSpec = DecisionSpec()
    instruction: Instruction
    projection: ProjectionSpec | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_v02(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        data = dict(raw)
        port = data.get("port")
        if "interface" not in data:
            if not isinstance(port, dict):
                raise TaskContractError("task requires interface or legacy port")
            data["interface"] = legacy_interface(port).model_dump(mode="json")
        if "harness" not in data:
            interface = InterfaceSpec.model_validate(data["interface"])
            data["harness"] = {"interaction_mode": interface.interaction_mode.value}
        verifier = data.get("verifier")
        if isinstance(verifier, dict):
            normalized_verifier = dict(verifier)
            metrics = []
            for raw_metric in normalized_verifier.get("metrics", []):
                metric = dict(raw_metric)
                if "kind" not in metric:
                    metric["kind"] = (
                        MetricKind.BOOLEAN.value
                        if str(metric.get("id")) in _BOOLEAN_METRICS
                        else MetricKind.CONTINUOUS.value
                    )
                metrics.append(metric)
            normalized_verifier["metrics"] = metrics
            data["verifier"] = normalized_verifier
        return data

    @model_validator(mode="after")
    def _invariants(self) -> Self:
        if self.harness.interaction_mode is not self.interface.interaction_mode:
            raise TaskContractError("harness interaction mode must match the task interface")
        scenario_ids = [scenario.id for scenario in self.scenarios]
        if len(set(scenario_ids)) != len(scenario_ids):
            raise TaskContractError(f"task {self.id} has duplicate scenario ids")
        perturbation_ids = [perturbation.id for perturbation in self.perturbations]
        if len(set(perturbation_ids)) != len(perturbation_ids):
            raise TaskContractError(f"task {self.id} has duplicate perturbation ids")
        unknown_scenarios = {
            perturbation.scenario_id
            for perturbation in self.perturbations
            if perturbation.scenario_id is not None and perturbation.scenario_id not in scenario_ids
        }
        if unknown_scenarios:
            raise TaskContractError(
                f"task {self.id} perturbations reference unknown scenarios "
                f"{sorted(unknown_scenarios)}"
            )
        if self.interface.interaction_mode is InteractionMode.CLOSED_LOOP:
            scenario_seeds = [scenario.seed for scenario in self.scenarios]
            if len(set(scenario_seeds)) != len(scenario_seeds):
                raise TaskContractError(
                    f"closed-loop task {self.id} maps more than one scenario to a seed"
                )
        if any(
            perturbation.at_step is not None and perturbation.at_step >= self.harness.max_steps
            for perturbation in self.perturbations
        ):
            raise TaskContractError(
                f"task {self.id} schedules a perturbation outside harness max_steps"
            )
        if self.phi.class_ is PhiClass.PROHIBITED:
            raise TaskContractError(f"task {self.id} is marked phi=prohibited and cannot load")
        if self.phi.class_ is PhiClass.PROCEDURAL and (
            self.attestation.level is not AttestationLevel.NONE
        ):
            raise TaskContractError("procedural geometry cannot mint a clinical attestation")
        if self.decision.emit_human_determination or self.subject.kind is SubjectKind.HUMAN:
            raise TaskContractError(
                "subject.kind=human and human determinations are outside this eval framework"
            )
        if self.metadata.safety_critical and not self.verifier.gates:
            raise TaskContractError(f"safety_critical task {self.id} must declare hard gates")
        capabilities = self.environment.resolved_capabilities
        if self.oracle.kind is OracleKind.PHYSICS and not capabilities.physics:
            raise TaskContractError(
                f"a physics oracle requires a world declaring physics capability; "
                f"{self.environment.kind_key} does not"
            )
        if self.interface.interaction_mode is InteractionMode.CLOSED_LOOP and (
            not capabilities.closed_loop
        ):
            raise TaskContractError(
                f"{self.interface.id} closed-loop tasks require a world declaring "
                f"closed-loop capability; {self.environment.kind_key} does not"
            )
        if self.interface.interaction_mode is InteractionMode.COUNTERFACTUAL and (
            not capabilities.counterfactual
        ):
            raise TaskContractError(
                f"counterfactual interfaces require a world declaring counterfactual "
                f"capability; {self.environment.kind_key} does not"
            )
        if self.environment.metrics_only:
            if self.verifier.gates:
                raise TaskContractError(
                    f"task {self.id} declares environment.metrics_only, so it must not "
                    "declare hard gates: a metrics-only wrap is a world whose "
                    "instrumentation does not report the safety state a gate would need, "
                    "and synthesizing one is the §2.2 failure this label exists to "
                    "prevent"
                )
            if self.metadata.safety_critical:
                raise TaskContractError(
                    f"task {self.id} is metrics-only and cannot also be "
                    "metadata.safety_critical; a metrics-only row is explicitly not "
                    "safety-attested"
                )
        metric_ids = {metric.id for metric in self.verifier.metrics}
        if "safe_success" in metric_ids and self.verifier.headline == "raw_success":
            raise TaskContractError(
                "CathSim failure mode: safe_success cannot be hidden behind raw_success"
            )
        return self

    def assert_runnable(self) -> None:
        capabilities = self.environment.resolved_capabilities
        if capabilities.requires_world_pin and not self.environment.world_pin:
            raise TaskContractError(f"task {self.id} has no world_pin")
        if not self.verifier.entrypoint:
            raise TaskContractError(f"task {self.id} has no verifier entrypoint")
        if self.interface.interaction_mode is not InteractionMode.CLOSED_LOOP:
            if not self.environment.inputs_path:
                raise TaskContractError(f"task {self.id} has no inputs_path")
            if not self.environment.labels_path:
                raise TaskContractError(f"task {self.id} has no labels_path")

    def metric(self, metric_id: str) -> MetricSpec:
        metric = next((item for item in self.verifier.metrics if item.id == metric_id), None)
        if metric is None:
            raise TaskContractError(f"task {self.id} does not declare metric {metric_id}")
        return metric

    def describe(self) -> str:
        pin = self.environment.world_pin or "(unpinned)"
        projection = self.projection.identity if self.projection else "none"
        gates = ", ".join(gate.id for gate in self.verifier.gates) or "(none)"
        metrics = ", ".join(metric.id for metric in self.verifier.metrics)
        return (
            f"Task {self.id}@{self.task_version} ({self.metadata.title})\n"
            f"  interface  {self.interface.id} ({self.harness.interaction_mode.value})\n"
            f"  port       {self.port.id.value if self.port else self.interface.id}\n"
            f"  world      {self.environment.kind_key} pin={pin}\n"
            f"  subject    {self.subject.kind.value}  phi={self.phi.class_.value}"
            "  human det. refused\n"
            f"  oracle     {self.oracle.kind.value}\n"
            f"  agents     {', '.join(self.agent.kinds)}\n"
            f"  gates      {gates}\n"
            f"  metrics    {metrics}\n"
            f"  headline   {self.verifier.headline}\n"
            f"  projection {projection}"
        )
