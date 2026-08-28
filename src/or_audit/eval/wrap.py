"""Wrap kit (N3): scaffold a task package around a third-party world.

A wrapped env earns a shelf slot by carrying what its upstream repo does not:
a pinned world, a mapped safety signal per hard gate, a cited basis for every
threshold, and a verifier that reads the engine's *reported* state instead of
inventing it. This module emits exactly that skeleton and refuses the two ways
a wrap goes wrong:

* a gate synthesized from state the env never reports (§2.2) — refused unless
  the author ships the honest label instead (``metrics_only``), and
* a threshold that is an unexplained literal — refused unless it cites a
  normative source or references a verified calibration artifact.

Generation is a pure function of the request: the same :class:`WrapRequest`
produces byte-identical files, with no timestamp, so a scaffold can be
regenerated and diffed as review evidence. What the wrap *claims* is still
unproven at this point; the generated package points at
``surgeval conformance`` (see :mod:`or_audit.eval.conformance`), which is what
measures determinism class and gate-state availability for the wrapped env.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from or_audit.errors import TaskContractError
from or_audit.eval.contracts import MetricDirection, MetricKind
from or_audit.eval.enums import GateKind, OracleKind, VerifierRealizationKind
from or_audit.eval.task import CalibrationSpec, NonEmpty, Slug, numeric_boundaries
from or_audit.eval.worlds import (
    WORLD_KIND_ENTRY_POINT_GROUP,
    WorldCapabilities,
    WorldKindSpec,
    resolve_world_capabilities,
    world_kind_key,
)
from or_audit.version import PACKAGE_VERSION

if TYPE_CHECKING:
    from or_audit.install.catalog import AuditedEnv, WorldPackage

#: Schema version of the emitted ``wrap.json`` provenance record.
WRAP_FORMAT_VERSION = "1"

#: An engine ``info`` key. Constrained to a lowercase identifier because the
#: same string becomes a gate evidence-binding name, a ``fail_when`` variable,
#: and a metric id — three places with narrower rules than an arbitrary key.
SignalName = Annotated[
    str, StringConstraints(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
]

#: Outcome names a safety signal must not claim: a success flag is the thing a
#: gate qualifies, never the evidence a gate is scored from.
_OUTCOME_NAMES = frozenset({"raw_success", "safe_success", "success", "is_success"})

#: Expression nodes a mapped ``fail_when`` may contain. Two consumers bound
#: this set: the kernel's gate DSL, which evaluates nothing else (an
#: unsupported node makes the gate abstain forever), and the generated
#: verifier, which embeds the expression as Python — so the allowlist is also
#: what keeps generated code free of calls, attributes, and subscripts.
_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Name,
    ast.Load,
    ast.Constant,
)

#: DSL literal keywords and the Python constants they mean. The gate DSL reads
#: ``true``/``false``/``null``; Python does not, so a generated verifier must
#: translate rather than emit a NameError at scoring time.
_DSL_KEYWORDS: dict[str, bool | None] = {
    "true": True,
    "false": False,
    "null": None,
    "none": None,
}


class _DslLiterals(ast.NodeTransformer):
    """Rewrite DSL literal names (``true``/``false``/``null``) into constants."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in _DSL_KEYWORDS:
            return ast.Constant(value=_DSL_KEYWORDS[node.id])
        return node


def _match_audited_env(pkg: WorldPackage, env_id: str) -> AuditedEnv | None:
    """Audited record for ``env_id``, or ``None`` when this env was never read.

    Exact match first. Scenes are the reason for the fallback: LapGym's audited
    ids are scene names (``grasp_lift_touch``), while a wrap may legitimately
    name the same scene as ``LapGym/grasp_lift_touch`` or
    ``sofa_env.scenes.grasp_lift_touch``. Matching is on whole separated
    components only - never a bare substring - so ``grasp_lift_touch_hard``
    stays unaudited instead of silently inheriting a surface it may not have.
    """
    exact = pkg.audited_env(env_id)
    if exact is not None:
        return exact
    parts = {piece for piece in re.split(r"[/:.\\-]+", env_id) if piece}
    for env in pkg.envs:
        tokens = {piece for piece in re.split(r"[/:.\\-]+", env.env_id) if piece}
        if tokens and tokens <= parts:
            return env
    return None


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GateMapping(_Frozen):
    """One hard gate mapped onto a signal the wrapped engine actually reports.

    ``signal`` is the engine ``info`` key the wrap read — not a name invented
    for the gate — and ``fail_when`` is the kernel-evaluated expression over
    it. A numeric ``threshold`` must carry a basis (``citation`` or
    ``calibration``): the number is what makes the gate a physical claim, and
    an uncited number is a heuristic wearing a unit.
    """

    id: Slug
    signal: SignalName
    fail_when: NonEmpty
    threshold: float | None = None
    unit: str = ""
    citation: str = ""
    calibration: CalibrationSpec | None = None

    @model_validator(mode="after")
    def _mapping_is_honest(self) -> Self:
        if self.signal in _OUTCOME_NAMES:
            raise TaskContractError(
                f"gate {self.id}: {self.signal!r} is an outcome flag, not a safety "
                "signal; a gate qualifies success, it is not scored from it"
            )
        self._assert_expression()
        has_basis = bool(self.citation) or self.calibration is not None
        if self.threshold is not None:
            if not self.unit:
                raise TaskContractError(
                    f"gate {self.id}: threshold {self.threshold} has no unit; an "
                    "unqualified number is not a physical quantity and cannot be "
                    "compared across worlds (§2.6 gate equivalence)"
                )
            if not has_basis:
                raise TaskContractError(
                    f"gate {self.id}: threshold {self.threshold} has no basis; cite a "
                    "normative source (--gate ...:UNIT:CITATION) or reference a "
                    "verified calibration artifact. A wrap does not get to pick a "
                    "safety number and leave it unexplained"
                )
        elif has_basis:
            raise TaskContractError(
                f"gate {self.id}: a threshold basis was declared without a threshold, "
                "so it has nowhere to land; declare the threshold it justifies"
            )
        return self

    def _assert_expression(self) -> None:
        """Refuse a ``fail_when`` the kernel or the generated verifier cannot host."""
        try:
            tree = ast.parse(self.fail_when, mode="eval")
        except SyntaxError as exc:
            raise TaskContractError(
                f"gate {self.id}: fail_when is not a valid expression: {exc}"
            ) from exc
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                raise TaskContractError(
                    f"gate {self.id}: fail_when may only compare the mapped signal "
                    f"against literals, got {type(node).__name__}"
                )
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        unknown = names - set(_DSL_KEYWORDS) - {self.signal}
        if unknown:
            raise TaskContractError(
                f"gate {self.id}: fail_when references {sorted(unknown)}, but this "
                f"mapping declares only the signal {self.signal!r} and the literals "
                f"{sorted(_DSL_KEYWORDS)}; map one signal per gate so the failing "
                "evidence is unambiguous"
            )
        if self.signal not in names:
            raise TaskContractError(
                f"gate {self.id}: fail_when {self.fail_when!r} never reads the signal "
                f"{self.signal!r}, so it is a constant verdict, not a gate. (A quoted "
                "expression becomes a string literal — quote the whole --gate value "
                "for the shell, not the expression inside it.)"
            )
        self._assert_threshold_is_the_enforced_number(tree)

    def _assert_threshold_is_the_enforced_number(self, tree: ast.Expression) -> None:
        """Every number ``fail_when`` enforces must be the number that was cited.

        Without this, a gate could declare ``threshold=1.5`` with a normative
        citation and enforce ``contact_force_n > 999``: the scorecard, the
        `wrap.json` record, and the rendered verifier docstring would all show
        the cited 1.5 while the run applied 999. That is worse than an uncited
        number, because the citation makes it look audited. The threshold is
        the *claim*; the predicate is the *enforcement*; a gate is only honest
        when they are the same number.
        """
        literals = numeric_boundaries(tree)
        if self.threshold is None:
            if literals:
                raise TaskContractError(
                    f"gate {self.id}: fail_when {self.fail_when!r} compares against "
                    f"{sorted(literals)} but declares no threshold, so the number it "
                    "enforces is uncited. Declare it as the threshold with a unit and a "
                    "basis (@THRESHOLD:UNIT:CITATION), or write a boolean gate over the "
                    "signal alone."
                )
            return
        mismatched = sorted(value for value in literals if value != self.threshold)
        if mismatched:
            raise TaskContractError(
                f"gate {self.id}: threshold {self.threshold} is cited, but fail_when "
                f"{self.fail_when!r} enforces {mismatched}. The citation would describe a "
                "boundary the run never applies. Fix: make the predicate compare against "
                f"{self.threshold}, or declare the number the predicate actually uses."
            )
        if not literals:
            raise TaskContractError(
                f"gate {self.id}: threshold {self.threshold} is declared with a basis, but "
                f"fail_when {self.fail_when!r} never compares against it, so the cited "
                "number is decoration. Fix: use the threshold in the predicate, or drop it "
                "and write a boolean gate."
            )

    @property
    def numeric(self) -> bool:
        """Whether this gate is scored from a numeric quantity."""
        return self.threshold is not None

    @property
    def python_expression(self) -> str:
        """``fail_when`` as Python, with DSL literals translated.

        The generated verifier evaluates the same condition the kernel gate
        does, so the two cannot drift on a typo; only the literal spelling
        differs, and that is mechanical.
        """
        tree = ast.parse(self.fail_when, mode="eval")
        translated = _DslLiterals().visit(tree)
        return ast.unparse(ast.fix_missing_locations(translated))

    @property
    def direction(self) -> MetricDirection:
        """Which way the mapped signal should move, read off the fail condition.

        Probing the condition is the only honest source: ``unsafe == true``
        makes the signal something to minimize, while
        ``safety_state_reported == false`` makes it something to maximize, and
        the kit must not label either backwards. A condition that fires on both
        sides (or neither) tells us nothing, so the metric stays neutral.
        """
        low: Any
        high: Any
        if self.threshold is None:
            low, high = False, True
        else:
            low, high = self.threshold - 1.0, self.threshold + 1.0
        try:
            code = compile(
                ast.Expression(body=ast.parse(self.python_expression, mode="eval").body),
                filename="<fail_when>",
                mode="eval",
            )
            fails_low = bool(eval(code, {"__builtins__": {}}, {self.signal: low}))
            fails_high = bool(eval(code, {"__builtins__": {}}, {self.signal: high}))
        except Exception:  # a probe is a convenience, never a load-time contract
            return MetricDirection.NEUTRAL
        if fails_high and not fails_low:
            return MetricDirection.MINIMIZE
        if fails_low and not fails_high:
            return MetricDirection.MAXIMIZE
        return MetricDirection.NEUTRAL

    @property
    def compares_to_boolean(self) -> bool:
        """Whether the translated expression compares against a boolean literal.

        The generated verifier keeps that comparison verbatim rather than
        collapsing it to a truth test: the kernel evaluates ``==`` on the raw
        reported value, and Python truthiness disagrees for anything that is
        not a bool. Keeping them identical costs one lint suppression.
        """
        expression = self.python_expression
        return any(
            f"{operator} {literal}" in expression
            for operator in ("==", "!=")
            for literal in ("True", "False", "None")
        )


class WrapRequest(_Frozen):
    """Everything the kit needs to emit a wrap, and nothing it can guess."""

    #: Upstream env identity, e.g. ``SurRoL/NeedleReach-v0``.
    env_id: NonEmpty
    task_id: Slug
    world_kind: Annotated[
        str, StringConstraints(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    ] = "gym"
    #: Immutable revision of the wrapped world. Required: an unpinned wrap
    #: cannot be replayed, so its rows are not evidence of anything.
    world_pin: str
    #: SPDX id. The Tier-1 license audit reads this out of ``wrap.json``.
    license: NonEmpty
    interface_id: Slug = "gym-policy"
    modality: NonEmpty = "robotic-kinematics"
    source_repo: str = ""
    gate_mappings: tuple[GateMapping, ...] = ()
    #: Ship the §2.2 honesty label instead of a synthesized gate.
    metrics_only: bool = False
    #: Permit a non-physical stand-in when the vendor runtime is absent;
    #: artifacts are stamped ``synthetic-stub`` and RL export refuses them.
    synthetic_stub: bool = False
    n_eval_episodes: Annotated[int, Field(ge=1, le=10_000)] = 5
    max_steps: Annotated[int, Field(ge=1, le=1_000_000)] = 100
    parameters: dict[str, bool | int | float | str] = Field(default_factory=dict)
    #: World eligibility, declared when no adapter for ``world_kind`` is installed.
    capabilities: WorldCapabilities | None = None

    @model_validator(mode="after")
    def _wrap_is_claimable(self) -> Self:
        if not self.world_pin.strip():
            raise TaskContractError(
                f"wrap {self.task_id}: --world-pin is required; a world that cannot be "
                "pinned cannot be replayed, and an unreplayable row is not evidence"
            )
        if self.metrics_only and self.gate_mappings:
            raise TaskContractError(
                f"wrap {self.task_id} is metrics-only and also maps "
                f"{len(self.gate_mappings)} gate(s); metrics-only means explicitly not "
                "safety-attested, so drop --metrics-only to publish the gates"
            )
        if not self.metrics_only and not self.gate_mappings:
            raise TaskContractError(
                f"wrap {self.task_id} maps no gates. Map the safety signals this env "
                "actually reports (--gate ID=SIGNAL:EXPR@THRESHOLD:UNIT:CITATION), or "
                "ship it honestly labelled with --metrics-only (§2.2: never synthesize "
                "a gate from state the env does not report)"
            )
        ids = [mapping.id for mapping in self.gate_mappings]
        if len(set(ids)) != len(ids):
            raise TaskContractError(f"wrap {self.task_id} declares duplicate gate ids")
        signals = [mapping.signal for mapping in self.gate_mappings]
        if len(set(signals)) != len(signals):
            raise TaskContractError(
                f"wrap {self.task_id} maps the same engine signal to more than one "
                "gate; one signal, one gate, one metric"
            )
        return self

    @model_validator(mode="after")
    def _gates_bind_to_audited_signals(self) -> Self:
        """A catalogued world's audited env decides which signals may carry gates.

        Keyed on ``world_pin``, not on a name: the pin is the immutable identity
        of the revision that was read, so this cannot be dodged by relabelling
        the world, and it simply does not apply to a revision nobody audited.

        Two cases are refused, and only two:

        * the named env *is* audited and a gate binds something that env does
          not publish as a physical value;
        * the named env is not audited, but the signal is audited in a
          *sibling* env of the same world - LapGym's ``tissue_dissection``
          claiming ``dynamic_force_on_gallbladder``, which exists only in
          ``grasp_lift_touch``.

        Everything else stays self-service: wrapping an unaudited scene with
        signals nobody has catalogued is how a third party onboards a world,
        and the ``:CITATION`` requirement already makes them say where their
        number came from.
        """
        if not self.gate_mappings:
            return self
        from or_audit.install.catalog import load_catalog

        pin = self.world_pin.strip()
        pkg = next((row for row in load_catalog().worlds if row.world_pin == pin), None)
        if pkg is None or not pkg.envs:
            return self
        env = _match_audited_env(pkg, self.env_id)

        for mapping in self.gate_mappings:
            if env is not None:
                bound = next(
                    (signal for signal in env.gate_signals if signal.key == mapping.signal), None
                )
                if bound is not None:
                    missing = {
                        name: want
                        for name, want in bound.requires_parameters.items()
                        if self.parameters.get(name) != want
                    }
                    if missing:
                        pinned = ", ".join(f"{name}={want!r}" for name, want in missing.items())
                        raise TaskContractError(
                            f"wrap {self.task_id}: gate {mapping.id!r} binds "
                            f"{mapping.signal!r}, which is only a physical measurement when "
                            f"constructed with {pinned}. Without that, upstream falls back to a "
                            "different quantity under the same key, so the gate would report a "
                            f"geometric predicate as a measurement. Fix: pass {pinned} via "
                            "--param, or bind a signal whose kind does not depend on "
                            "construction. Audited note: "
                            f"{bound.note.strip().splitlines()[0] if bound.note.strip() else '-'}"
                        )
                    if mapping.unit and mapping.unit != bound.unit:
                        raise TaskContractError(
                            f"wrap {self.task_id}: gate {mapping.id!r} publishes "
                            f"{mapping.signal!r} in {mapping.unit!r}, but the audited reading "
                            f"of that signal is in {bound.unit!r}. The threshold would be a "
                            "physically false quantity, and §2.6 gate equivalence compares "
                            "gates by unit, so this would also make it falsely comparable to "
                            f"other worlds. Fix: publish it as {bound.unit!r}, or convert the "
                            "number explicitly and cite the conversion. Audited note: "
                            f"{bound.note.strip().splitlines()[0] if bound.note.strip() else '-'}"
                        )
                    continue
                available = ", ".join(s.key for s in env.gate_signals) or "(none)"
                held = next((s for s in env.signals if s.key == mapping.signal), None)
                if held is not None and not held.published:
                    detail = (
                        f"{mapping.signal!r} is computed by this env but never published to "
                        "info/extras, so no verifier can read it"
                    )
                elif held is not None:
                    detail = (
                        f"{mapping.signal!r} is published but audited as {held.kind.value}, "
                        "not a measured physical quantity"
                    )
                else:
                    sibling = next(
                        (
                            sib.env_id
                            for sib in pkg.envs
                            if sib.env_id != env.env_id
                            and any(s.key == mapping.signal for s in sib.signals)
                        ),
                        "",
                    )
                    detail = (
                        f"{mapping.signal!r} is not a signal this env publishes; "
                        f"{pkg.id} publishes in env {sibling!r}, and signal surfaces are per "
                        "env, so a scene cannot borrow a sibling's gate"
                        if sibling
                        else f"{mapping.signal!r} is not a signal this env publishes at all"
                    )
                raise TaskContractError(
                    f"wrap {self.task_id}: gate {mapping.id!r} cannot bind {mapping.signal!r}. "
                    f"{detail}. Audited at {pkg.id}@{pin[:12]} env {env.env_id!r}; "
                    f"gate-eligible signals there: {available}. Fix: bind one of those, or "
                    "ship the wrap with --metrics-only."
                )
            owner = next(
                (sib for sib in pkg.envs if any(s.key == mapping.signal for s in sib.signals)),
                None,
            )
            if owner is not None:
                raise TaskContractError(
                    f"wrap {self.task_id}: gate {mapping.id!r} binds {mapping.signal!r}, which "
                    f"{pkg.id} publishes in env {owner.env_id!r}, not in {self.env_id!r}. "
                    "Signal surfaces are per env - eligibility is a property of (pin, env) - so "
                    f"a scene cannot borrow a sibling's gate. Fix: wrap {owner.env_id!r}, or "
                    f"audit {self.env_id!r} and record the signals it publishes itself."
                )
        return self

    @property
    def gated(self) -> bool:
        """Whether this wrap ships hard gates."""
        return bool(self.gate_mappings)


@dataclass(frozen=True)
class WrapResult:
    """What a scaffold produced, and the command that validates the claim."""

    root: Path
    files: tuple[Path, ...]
    task_id: str
    world_kind: str
    world_pin: str
    #: Adapter identity the package was pinned against, empty when none is installed.
    adapter_id: str
    adapter_digest: str
    capabilities: WorldCapabilities
    metrics_only: bool
    gate_ids: tuple[str, ...]
    conformance_command: str


@dataclass(frozen=True)
class _Metric:
    """A metric the generated task declares and its verifier emits."""

    id: str
    kind: MetricKind
    direction: MetricDirection
    unit: str = ""


def _installed_kind_spec(world_kind: str) -> WorldKindSpec | None:
    from or_audit.eval.sim import world_kind_spec  # registers the built-in adapters

    return world_kind_spec(world_kind)


def _resolve(spec: WrapRequest) -> tuple[WorldCapabilities, WorldKindSpec | None]:
    """Resolve the wrapped world's eligibility, or refuse with the plugin seam."""
    kind_spec = _installed_kind_spec(spec.world_kind)
    try:
        capabilities = resolve_world_capabilities(spec.world_kind, spec.capabilities)
    except TaskContractError as exc:
        raise TaskContractError(
            f"world kind {spec.world_kind!r} has no installed adapter and the wrap "
            "declares no capabilities, so the kit cannot say what this world is "
            "eligible for. Install the adapter distribution (it publishes a "
            f"WorldAdapter under the {WORLD_KIND_ENTRY_POINT_GROUP!r} entry-point "
            "group), or pass the world's declared capabilities"
        ) from exc
    if not capabilities.closed_loop:
        raise TaskContractError(
            f"world kind {spec.world_kind!r} does not declare closed-loop capability, "
            "so it cannot host a stepped wrap; the wrap kit scaffolds worlds a policy "
            "drives"
        )
    if capabilities.requires_contract:
        raise TaskContractError(
            f"world kind {spec.world_kind!r} is addressed by a frozen-data contract, "
            "not a stepped env; author that package directly rather than wrapping it"
        )
    return capabilities, kind_spec


def _metrics(spec: WrapRequest) -> tuple[_Metric, ...]:
    """The metric set both the task.toml and the generated verifier commit to."""
    metrics: list[_Metric] = []
    if spec.gated:
        metrics.append(_Metric("safe_success", MetricKind.BOOLEAN, MetricDirection.MAXIMIZE))
    metrics.append(_Metric("raw_success", MetricKind.BOOLEAN, MetricDirection.MAXIMIZE))
    for mapping in spec.gate_mappings:
        metrics.append(
            _Metric(
                mapping.signal,
                MetricKind.CONTINUOUS if mapping.numeric else MetricKind.BOOLEAN,
                mapping.direction,
                mapping.unit,
            )
        )
    if not any(metric.id == "diverged" for metric in metrics):
        metrics.append(_Metric("diverged", MetricKind.BOOLEAN, MetricDirection.MINIMIZE))
    return tuple(metrics)


def _headline(spec: WrapRequest) -> str:
    return "safe_success" if spec.gated else "raw_success"


def _toml_str(value: str) -> str:
    """Render a TOML basic string (JSON escaping is a compatible subset)."""
    return json.dumps(value, ensure_ascii=False)


def _toml_value(value: bool | int | float | str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _toml_str(value)
    return repr(value)


def _toml_inline(values: dict[str, bool | int | float | str]) -> str:
    body = ", ".join(f"{key} = {_toml_value(values[key])}" for key in sorted(values))
    return "{ " + body + " }"


def _render_task_toml(
    spec: WrapRequest,
    *,
    capabilities: WorldCapabilities,
    kind_spec: WorldKindSpec | None,
) -> str:
    """Render a v0.3 task.toml that is loadable the moment it is written."""
    # `environment.parameters` is forwarded verbatim to the env constructor, so
    # the harness step limit does not belong in it: `gymnasium.make("FrozenLake-v1",
    # max_steps=8)` is a TypeError, and a scaffold that cannot be run is not a
    # scaffold. The limit already lives in `[harness].max_steps`, which is where
    # the runner reads it; only caller-supplied parameters are passed through.
    parameters: dict[str, bool | int | float | str] = dict(spec.parameters)
    oracle = OracleKind.PHYSICS if capabilities.physics else OracleKind.SCRIPT
    lines = [
        "# Generated by `surgeval wrap`. Regenerate rather than hand-patch the pins:",
        f"# the wrapped world is {spec.env_id} at {spec.world_pin}.",
        'format_version = "2"',
        f"id = {_toml_str(spec.task_id)}",
        'task_version = "0"',
        "",
        "[metadata]",
        f"title = {_toml_str(f'Wrapped world: {spec.env_id}')}",
        f"modality = {_toml_str(spec.modality)}",
        f'tags = ["wrap", {_toml_str(spec.world_kind)}'
        + (', "metrics-only"' if spec.metrics_only else "")
        + "]",
        f"safety_critical = {'false' if spec.metrics_only else 'true'}",
        "",
        "[subject]",
        'kind = "policy"',
        "",
        "[phi]",
        'class = "procedural"',
        "",
        "[environment]",
        f"kind = {_toml_str(spec.world_kind)}",
        f"gym_id = {_toml_str(spec.env_id)}",
        f"world_pin = {_toml_str(spec.world_pin)}",
    ]
    if kind_spec is not None and kind_spec.adapter_id and kind_spec.adapter_digest:
        lines += [
            f"adapter = {_toml_str(kind_spec.adapter_id)}",
            f"adapter_digest = {_toml_str(kind_spec.adapter_digest)}",
        ]
    lines += [
        f"parameters = {_toml_inline(parameters)}",
        f"n_eval_episodes = {spec.n_eval_episodes}",
        f"seed_policy = {_toml_str(f'deterministic-eval-{spec.n_eval_episodes}')}",
    ]
    if spec.metrics_only:
        lines.append("metrics_only = true")
    if spec.synthetic_stub:
        lines.append("synthetic_stub = true")
    lines += [
        "",
        "# Declared so this package stays loadable where the adapter is absent;",
        "# cross-checked against the installed adapter at load time. determinism_class",
        "# stays `unmeasured` until `surgeval conformance` measures a seeded rerun.",
        "[environment.capabilities]",
        f"physics = {_toml_value(capabilities.physics)}",
        f"closed_loop = {_toml_value(capabilities.closed_loop)}",
        f"counterfactual = {_toml_value(capabilities.counterfactual)}",
        f"requires_gym_id = {_toml_value(capabilities.requires_gym_id)}",
        f"requires_world_pin = {_toml_value(capabilities.requires_world_pin)}",
        f"requires_contract = {_toml_value(capabilities.requires_contract)}",
        f"determinism_class = {_toml_str(capabilities.determinism_class.value)}",
        "",
        "[interface]",
        f"id = {_toml_str(spec.interface_id)}",
        'interaction_mode = "closed-loop"',
        'protocol_version = "1"',
        f"observations = [{_toml_str(f'{spec.task_id}-obs')}]",
        f"actions = [{_toml_str(f'{spec.task_id}-action')}]",
        "",
        "[harness]",
        'interaction_mode = "closed-loop"',
        'protocol_version = "1"',
        f"max_steps = {spec.max_steps}",
        "",
        "[agent]",
        'kinds = ["policy", "random"]',
        f"action_space = {_toml_str(f'{spec.task_id}-action')}",
        "timeout_sec = 60.0",
        "",
        "[oracle]",
        f"kind = {_toml_str(oracle.value)}",
        "",
        "[verifier]",
        "# A freshly wrapped world is not yet known to report every mapped signal, and",
        "# an abstained gate is the honest outcome when it does not.",
        "abstain_ok = true",
        f"headline = {_toml_str(_headline(spec))}",
        'entrypoint = "verifier.py:load_verifier"',
    ]
    for mapping in spec.gate_mappings:
        lines += [
            "",
            "[[verifier.gates]]",
            f"id = {_toml_str(mapping.id)}",
            f"inputs = {{ {mapping.signal} = {_toml_str(f'info.{mapping.signal}')} }}",
            f"fail_when = {_toml_str(mapping.fail_when)}",
            'maps_to = "unsafe"',
            f"kind = {_toml_str(GateKind.CUSTOM.value)}",
            f"realization = {_toml_str(VerifierRealizationKind.SCALAR_DSL.value)}",
        ]
        if mapping.threshold is not None:
            lines += [
                f"threshold = {mapping.threshold!r}",
                f"unit = {_toml_str(mapping.unit)}",
                "[verifier.gates.threshold_basis]",
                f"value = {mapping.threshold!r}",
                f"unit = {_toml_str(mapping.unit)}",
            ]
            if mapping.citation:
                lines.append(f"citation = {_toml_str(mapping.citation)}")
            if mapping.calibration is not None:
                calibration = mapping.calibration
                lines += [
                    "[verifier.gates.threshold_basis.calibration]",
                    f"method = {_toml_str(calibration.method)}",
                    f"artifact = {_toml_str(calibration.artifact)}",
                    f"digest = {_toml_str(calibration.digest)}",
                ]
                if calibration.note:
                    lines.append(f"note = {_toml_str(calibration.note)}")
    for metric in _metrics(spec):
        lines += [
            "",
            "[[verifier.metrics]]",
            f"id = {_toml_str(metric.id)}",
            f"kind = {_toml_str(metric.kind.value)}",
            f"direction = {_toml_str(metric.direction.value)}",
        ]
        if metric.unit:
            lines.append(f"unit = {_toml_str(metric.unit)}")
        lines.append(f"source = {_toml_str(f'info.{metric.id}')}")
    return "\n".join(lines) + "\n"


def _render_instruction(spec: WrapRequest, *, capabilities: WorldCapabilities) -> str:
    """Render instruction.md: what is claimed, and just as loudly what is not."""
    lines = [
        f"# {spec.env_id} (wrapped)",
        "",
        f"Drive the policy in `{spec.env_id}`, hosted through the "
        f"`{spec.world_kind}` world adapter and pinned at `{spec.world_pin}`.",
        f"Episodes run for at most {spec.max_steps} steps; "
        f"{spec.n_eval_episodes} seeded evaluation episode(s) per run.",
        "",
        "## Claim boundary",
        "",
        f"- This package makes one claim: *this policy ran in `{spec.env_id}` at "
        f"`{spec.world_pin}` under this harness*.",
        "- Results are per-world rows. No cross-world aggregate, ranking, or ordering "
        "is licensed until a published equivalence artifact covers this shelf.",
        f"- Execution determinism is `{capabilities.determinism_class.value}` until "
        "`surgeval conformance` measures a seeded rerun of this env.",
    ]
    if spec.synthetic_stub:
        lines.append(
            "- The vendor runtime is not redistributable here, so a non-physical "
            "stand-in may serve the world. Those artifacts are stamped "
            '`backend="synthetic-stub"` and RL export refuses them: they are '
            "plumbing evidence, never physical evidence."
        )
    if spec.metrics_only:
        lines += [
            "",
            "## Not safety-attested (metrics-only)",
            "",
            "This package is explicitly **not safety-attested**. The wrapped world does "
            "not report the safety state a hard gate would need, so it declares no gates "
            "and is not `safety_critical`. Nothing here attests that a run was safe — "
            "only that it happened and how the declared metrics came out. Synthesizing "
            "a gate from state the env never reports would be the exact failure this "
            "label exists to prevent; the honest fix is upstream instrumentation.",
        ]
    else:
        lines += [
            "",
            "## Hard gates",
            "",
            "Each gate below is scored by the kernel from a signal this env actually "
            "reports. A gate whose signal is absent abstains — it never reads as a "
            "pass, and it is never averaged into a metric.",
            "",
        ]
        for mapping in spec.gate_mappings:
            basis = mapping.citation or (
                f"calibration artifact `{mapping.calibration.artifact}`"
                if mapping.calibration is not None
                else ""
            )
            bound = (
                f" threshold {mapping.threshold} {mapping.unit} — basis: {basis}"
                if mapping.threshold is not None
                else ""
            )
            lines.append(
                f"- `{mapping.id}`: fails when `{mapping.fail_when}` over engine signal "
                f"`info.{mapping.signal}`.{bound}"
            )
    lines += [
        "",
        "## Next",
        "",
        "Run `surgeval conformance` on this package before publishing it: Tier-1 "
        "placement requires measured gate-state availability, a license check, "
        "evidence-replay round-trip, and a recorded determinism class.",
    ]
    return "\n".join(lines) + "\n"


def _verifier_function_name(gate_id: str) -> str:
    return "_gate_" + gate_id.replace("-", "_")


def _render_verifier(spec: WrapRequest) -> str:
    """Render a verifier that reads reported state and defaults nothing."""
    metrics = _metrics(spec)
    lines = [
        f'"""Generated verifier for the wrapped world `{spec.env_id}`.',
        "",
        "Every value here comes from what the engine reported in `info`. A signal the",
        "engine did not report is emitted as `None` (an unassessable metric), never as",
        "`0.0` or `False`: a defaulted safety number is a fabricated one, and the",
        "headline would inherit the fabrication.",
        "",
    ]
    if spec.gate_mappings:
        lines += [
            "Hard-gate verdicts are deliberately absent from the returned `gates` object.",
            "The declared gates are `scalar-dsl`, so the kernel resolves and hashes their",
            "evidence and decides pass/fail/abstain; a verifier that also self-reported",
            "those statuses would be an unhashed second opinion.",
        ]
    else:
        lines += [
            "This wrap maps no hard gate, so `GATES` is empty and every returned value is a",
            "measurement. That is the metrics-only posture of §2.2, not an omission: a gate",
            "here would need a cited threshold and a signal this world is known to report.",
        ]
    lines += [
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Any",
        "",
    ]
    for mapping in spec.gate_mappings:
        lines += [
            "",
            f"def {_verifier_function_name(mapping.id)}({mapping.signal}: Any) -> bool:",
            f'    """Gate `{mapping.id}` as declared in task.toml."""',
            f"    return bool({mapping.python_expression})"
            + (
                "  # noqa: E712 - kernel equality, not truthiness"
                if mapping.compares_to_boolean
                else ""
            ),
            "",
        ]
    # With gate functions above, the module needs the two-blank-line separator a
    # `def` block takes; with none, the comment follows the imports after one.
    if spec.gate_mappings:
        lines.append("")
    lines += [
        "#: (gate id, engine info key, predicate) for every declared hard gate.",
        "GATES: tuple[tuple[str, str, Any], ...] = ("
        if spec.gate_mappings
        else "GATES: tuple[tuple[str, str, Any], ...] = ()",
    ]
    for mapping in spec.gate_mappings:
        lines.append(
            f'    ("{mapping.id}", "{mapping.signal}", {_verifier_function_name(mapping.id)}),'
        )
    if spec.gate_mappings:
        lines.append(")")
    lines += [
        "",
        "",
        "def _reported(info: dict[str, Any], key: str, *aliases: str) -> Any:",
        '    """First reported alias, or ``None`` when the engine reported none."""',
        "    for name in (key, *aliases):",
        "        if name in info:",
        "            return info[name]",
        "    return None",
        "",
        "",
        "def _boolean(info: dict[str, Any], key: str, *aliases: str) -> bool | None:",
        "    value = _reported(info, key, *aliases)",
        "    return None if value is None else bool(value)",
        "",
        "",
        "def _numeric(info: dict[str, Any], key: str, *aliases: str) -> float | None:",
        "    value = _reported(info, key, *aliases)",
        "    return None if value is None else float(value)",
        "",
        "",
        "class WrapVerifier:",
        f'    """Verifier for the wrapped world `{spec.env_id}`."""',
        "",
        "    def score(self, context: dict[str, Any]) -> dict[str, Any]:",
        '        raw_info = context.get("info")',
        "        info: dict[str, Any] = raw_info if isinstance(raw_info, dict) else {}",
        '        raw_success = _boolean(info, "raw_success", "success", "is_success")',
        '        diverged = _boolean(info, "diverged")',
        "        breached: list[str] = []",
        "        unreported: list[str] = []",
        "        for gate_id, signal, predicate in GATES:",
        "            value = _reported(info, signal)",
        "            if value is None:",
        "                unreported.append(signal)",
        "            elif predicate(value):",
        "                breached.append(gate_id)",
    ]
    if spec.gated:
        lines += [
            "        # Unreported safety state, an unreported outcome, or unreported",
            "        # divergence all make safety unassessable, not satisfied.",
            "        safe_success: bool | None = None",
            "        if not unreported and raw_success is not None and diverged is not None:",
            "            safe_success = bool(raw_success and not breached and not diverged)",
        ]
    lines.append("        metrics: dict[str, Any] = {")
    for metric in metrics:
        if metric.id == "safe_success":
            lines.append('            "safe_success": safe_success,')
        elif metric.id == "raw_success":
            lines.append('            "raw_success": raw_success,')
        elif metric.id == "diverged":
            lines.append('            "diverged": diverged,')
        elif metric.kind is MetricKind.CONTINUOUS:
            lines.append(f'            "{metric.id}": _numeric(info, "{metric.id}"),')
        else:
            lines.append(f'            "{metric.id}": _boolean(info, "{metric.id}"),')
    lines += [
        "        }",
        "        return {"
        '"gates": {}, "metrics": metrics, '
        '"unreported_signals": sorted(unreported)}',
        "",
        "",
        "def load_verifier(*, root: Any = None) -> WrapVerifier:",
        "    del root",
        "    return WrapVerifier()",
    ]
    return "\n".join(lines) + "\n"


def _render_wrap_json(
    spec: WrapRequest,
    *,
    kind_spec: WorldKindSpec | None,
    conformance_command: str,
) -> str:
    """Render wrap provenance. No timestamp: the record must be reproducible."""
    payload: dict[str, Any] = {
        "format_version": WRAP_FORMAT_VERSION,
        "adapter_digest": kind_spec.adapter_digest if kind_spec else "",
        "adapter_id": kind_spec.adapter_id if kind_spec else "",
        "adapter_identity": kind_spec.adapter_identity if kind_spec else "unattached",
        "env_id": spec.env_id,
        "generator": f"surgeval {PACKAGE_VERSION}",
        "license": spec.license,
        "metrics_only": spec.metrics_only,
        "next_steps": [
            conformance_command,
            "surgeval tasks validate <package>",
            "publish only per-world rows until a shelf equivalence artifact exists",
        ],
        "source_repo": spec.source_repo,
        "synthetic_stub": spec.synthetic_stub,
        "task_id": spec.task_id,
        "world_kind": world_kind_key(spec.world_kind),
        "world_pin": spec.world_pin,
    }
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def scaffold_wrap(spec: WrapRequest, out: Path) -> WrapResult:
    """Write a wrap task package for ``spec`` into ``out``.

    Deterministic: identical requests write byte-identical files, so a
    regenerated scaffold diffs cleanly against a hand-edited one.
    """
    capabilities, kind_spec = _resolve(spec)
    root = Path(out)
    conformance_command = f"surgeval conformance {root}"
    contents = {
        "instruction.md": _render_instruction(spec, capabilities=capabilities),
        "task.toml": _render_task_toml(spec, capabilities=capabilities, kind_spec=kind_spec),
        "verifier.py": _render_verifier(spec),
        "wrap.json": _render_wrap_json(
            spec, kind_spec=kind_spec, conformance_command=conformance_command
        ),
    }
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in sorted(contents):
        path = root / name
        path.write_text(contents[name], encoding="utf-8")
        written.append(path)
    return WrapResult(
        root=root,
        files=tuple(written),
        task_id=spec.task_id,
        world_kind=world_kind_key(spec.world_kind),
        world_pin=spec.world_pin,
        adapter_id=kind_spec.adapter_id if kind_spec else "",
        adapter_digest=kind_spec.adapter_digest if kind_spec else "",
        capabilities=capabilities,
        metrics_only=spec.metrics_only,
        gate_ids=tuple(mapping.id for mapping in spec.gate_mappings),
        conformance_command=conformance_command,
    )
