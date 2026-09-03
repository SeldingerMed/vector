"""World catalog: typed, validated view of ``catalog.toml``.

The catalog is the curated artifact of §2.4 — what each world is, what its
Appendix B disposition is, and how it installs (N10 item 2). It is loaded into
frozen models rather than read as raw dicts so that a missing pin, an
unaudited license, or a nonsense strategy payload is a load-time refusal
instead of a surprise in front of a user's shell.

Nothing in this module invents data. Where the catalog records ``unverified``
or an empty pin, the installer and ``surgeval doctor`` refuse and name the
missing artifact; they never substitute a plausible-looking digest, license,
or version, because a fabricated pin is worse than a missing one — it looks
reproducible.
"""

from __future__ import annotations

import re
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from or_audit.errors import TaskContractError
from or_audit.eval.licensing import LicenseStatus, LicenseVerdict, classify_license
from or_audit.eval.worlds import DeterminismClass

#: Sentinel license value: we have not read this world's LICENSE ourselves.
UNVERIFIED = "unverified"

#: Packaged catalog resource, resolved relative to this module so the catalog
#: travels with the wheel rather than depending on a source checkout.
CATALOG_PATH = Path(__file__).resolve().parent / "catalog.toml"
PLANNER_CATALOG_PATH = Path(__file__).resolve().parent / "planner-catalog.json"

_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

WorldPackageId = Annotated[
    str, StringConstraints(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9_-]*$")
]


class Disposition(StrEnum):
    """Appendix B verdict for a surveyed environment.

    ``shipped`` is the fourth value Appendix B does not need: first-party
    content already on the shelf, which is not a *wrap* decision at all.
    """

    #: First-party content already on the shelf (§2.4: Lumen).
    SHIPPED = "shipped"
    #: v0.2 Tier-1 wrap target (N4).
    WRAP = "wrap"
    #: Revisit on demand or upstream maturity. Recorded, not shelved.
    WATCH = "watch"
    #: Reason given; not a shelf item.
    SKIP = "skip"


class InstallStrategy(StrEnum):
    """How a world reaches a user's machine (N10 item 2).

    Chosen by the *worst first-hour experience*, not by what is technically
    possible: source builds that eat an afternoon get a container, and a
    runtime we may not redistribute gets a vendor-driven pull.

    ``source-build`` is the case the original ladder missed. A world whose own
    code is permissive but whose *runtime closure* is copyleft (every SOFA
    world: SOFA core LGPL-2.1, SofaPython3 LGPL-2.1, BeamAdapter LGPL-2.1,
    SoftRobots/Cosserat LGPL-3.0, ModelOrderReduction GPL-2.0) cannot be
    shipped as an image *we* publish without redistributing that closure. The
    honest answer is not a nicer container; it is that the user builds it, and
    we say so with the pins and the build reference.
    """

    #: Ships with the harness or as a sibling first-party distribution.
    FIRST_PARTY = "first-party"
    #: Upstream is pip-friendly and actually publishes an installable artifact.
    PIP_EXTRA = "pip-extra"
    #: We publish a digest-pinned OCI image (only for redistributable runtimes).
    PREBUILT_CONTAINER = "prebuilt-container"
    #: The user compiles the runtime; we pin the revision and cite the build docs.
    SOURCE_BUILD = "source-build"
    #: The vendor's runtime, pulled under the vendor's EULA (Isaac Sim).
    VENDOR_RUNTIME = "vendor-runtime"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SignalKind(StrEnum):
    """What an audited upstream signal actually measures.

    The distinction decides what a task may bind a *hard* gate to. Only a
    measured physical quantity is evidence about a procedure; a geometric
    predicate is a statement about simulated poses, and task bookkeeping is
    not a safety signal at all no matter how it is named.
    """

    #: A measured contact, force, penetration, or stress quantity.
    PHYSICAL = "physical"
    #: A distance / pose / velocity predicate over simulated state. Real, but a
    #: threshold on geometry, not a measurement of what the tissue felt.
    GEOMETRIC = "geometric"
    #: Solver or numerical health: NaN guards, instability heuristics. A run
    #: that trips one of these is *invalid*, not unsafe - stEVE's
    #: ``simulation_error`` fires on NaN tracking, which says nothing about
    #: the vessel. Binding a safety gate to it would report solver trouble as
    #: patient harm.
    DIAGNOSTIC = "diagnostic"
    #: Task progress: success flags, step counters, reward components.
    BOOKKEEPING = "bookkeeping"


class WorldSignal(_Frozen):
    """One key an upstream env publishes, read first-hand at the world's pin."""

    #: The ``info`` / ``extras`` key when the value is published; the upstream
    #: symbol that computes it when it is not. Either way it must appear
    #: verbatim in the cited file, which ``scripts/check_world_signals.py``
    #: enforces against the pinned tree - the first thing that check caught
    #: was an invented key name in this very catalog.
    key: str
    kind: SignalKind
    #: File the key is assigned in, when that is not the env's own file.
    #: stEVE's ``simulation_error`` is set in the SOFA adapter, several
    #: modules away from the env it surfaces on.
    path: str = ""
    #: Line in the audited file where the key is assigned. 0 admits that
    #: the read was not pinned to a line.
    line: int = Field(default=0, ge=0)
    unit: str = ""
    #: Whether the value reaches ``info`` / ``extras`` where a verifier can
    #: read it. A quantity the engine computes but never publishes cannot
    #: carry a gate: SurRoL calls ``getContactPoints`` for grasp logic and
    #: discards it, which is exactly the case this field exists to record.
    published: bool = True
    #: Construction parameters that must hold for the recorded ``kind`` to be
    #: true. LapGym's ``collision_with_board`` is a counted contact only when
    #: ``with_board_collision=True``; otherwise the same key silently becomes a
    #: ``cauter_position[2] < 0.0`` pose predicate. A gate bound to a signal
    #: with this field set must pin these parameters, or it is claiming a
    #: measurement the run may not produce.
    requires_parameters: dict[str, bool | int | float | str] = Field(default_factory=dict)
    note: str = ""

    @property
    def gate_eligible(self) -> bool:
        """Whether a hard gate may bind to this signal."""
        return self.published and self.kind is SignalKind.PHYSICAL


class AuditedEnv(_Frozen):
    """One upstream env / scene, read first-hand at the world's pin.

    Keyed per env, never per world, because eligibility genuinely varies
    inside one package: LapGym's ``grasp_lift_touch`` publishes a gallbladder
    internal force, while its ``magnetic_continuum_robot`` scene publishes
    nothing physical. A package-wide signal set would let the second scene
    borrow the first's gate - a fabricated gate with a real-looking citation,
    which is the failure this whole record exists to prevent.
    """

    #: How a task names this env: a gym id, or the scene module for envs that
    #: are constructed directly rather than registered.
    env_id: str
    #: Exact repo-relative path at the world's pin. Checked against the
    #: fetched tree by ``scripts/check_world_signals.py``; a path that does
    #: not resolve at the pin is a fabricated citation, not a typo.
    path: str
    signals: tuple[WorldSignal, ...] = ()
    #: Line-pinned readings that state an *empty* signal surface, for an env
    #: that publishes nothing. Only meaningful because they are checked
    #: exactly like signals: Surgical Gym's task writes an L2 distance to
    #: ``rew_buf`` and terminates on ``progress_buf``, and recording those two
    #: sites is what makes "nothing is published here" a citation rather than
    #: a silence. Without it ``scripts/check_world_signals.py`` fetched the
    #: tree, read not one line, and printed "0 signal(s) resolved" - a pass
    #: earned by omission, which a zero-byte file was enough to obtain.
    absence_markers: tuple[WorldSignal, ...] = ()
    note: str = ""

    @property
    def gate_signals(self) -> tuple[WorldSignal, ...]:
        """Signals a hard gate may bind to for *this* env."""
        return tuple(signal for signal in self.signals if signal.gate_eligible)

    @property
    def safety_eligible(self) -> bool:
        """Whether any gate can honestly bind in this env."""
        return bool(self.gate_signals)

    @model_validator(mode="after")
    def _gate_eligible_signals_carry_a_unit(self) -> Self:
        """A signal a gate may bind must say what its numbers are measured in.

        Without this the unit cross-check has nothing to compare against, and
        an unrecorded unit becomes a silent pass: a wrap could publish
        ``1.5 N`` for LapGym's ``dynamic_force_on_gallbladder``, which is a
        SOFA internal force *multiplied by a scene scaling factor* and is
        therefore not newtons. The unit is part of the reading, not a label
        added afterwards.
        """
        missing = [signal.key for signal in self.gate_signals if not signal.unit.strip()]
        if missing:
            raise TaskContractError(
                f"audited env {self.env_id!r}: gate-eligible signal(s) {missing} record no "
                "unit. A physical signal a gate can bind must state what it is measured in, "
                "because the gate's threshold will be published in that unit. Record the "
                "unit as upstream actually produces it (e.g. 'scaled-N' when a scaling "
                "factor is applied, 'contacts' for a count), or record the signal as "
                "non-physical if the quantity is not a measurement."
            )
        return self

    @model_validator(mode="after")
    def _absence_markers_are_checkable_and_mean_absence(self) -> Self:
        """An absence record is a citation, so it has to be checkable.

        Unstated absence let a zero-signal env pass on an empty file. Replacing
        it with an *unchecked* assertion would be the same defect in better
        clothes, so every marker must pin a line for
        ``scripts/check_world_signals.py`` to resolve at the pin. And a marker
        claims nothing reaches ``info``/``extras``, which is false if the env
        also records published signals - a key that is published belongs in
        ``signals``, kind and all.
        """
        if not self.absence_markers:
            return self
        if self.signals:
            raise TaskContractError(
                f"audited env {self.env_id!r}: records both signals "
                f"{[s.key for s in self.signals]} and absence_markers "
                f"{[m.key for m in self.absence_markers]}. absence_markers state that this "
                "env's signal surface is empty; a key it does publish belongs in signals."
            )
        unpinned = [marker.key for marker in self.absence_markers if not marker.line]
        if unpinned:
            raise TaskContractError(
                f"audited env {self.env_id!r}: absence_marker(s) {unpinned} pin no line, so "
                "the citation checker cannot resolve them and the recorded absence stays "
                "unverified. Record the line each was read at, or drop the marker."
            )
        published = [marker.key for marker in self.absence_markers if marker.published]
        if published:
            raise TaskContractError(
                f"audited env {self.env_id!r}: absence_marker(s) {published} are marked "
                "published, which contradicts the absence they record. A marker is a site "
                "that computes something and does not publish it; set published = false, or "
                "record it as a signal."
            )
        return self


class PinnedPackage(_Frozen):
    """One lockfile-style pin. An empty ``version`` is an admitted gap."""

    name: Annotated[str, StringConstraints(min_length=1)]
    version: str = ""


class FirstPartyInstall(_Frozen):
    """First-party content: nothing third-party to fetch, but still pinned."""

    strategy: Literal[InstallStrategy.FIRST_PARTY] = InstallStrategy.FIRST_PARTY
    #: Distribution name, when the world's physics lives beside the harness.
    distribution: str = ""
    #: Module the doctor imports to prove the world is present.
    verify_import: str = ""
    pinned_version: str = ""

    @property
    def pinned(self) -> bool:
        return bool(self.pinned_version)


class PipExtraInstall(_Frozen):
    """Plain extras with lockfile pins."""

    strategy: Literal[InstallStrategy.PIP_EXTRA]
    #: Extras of *our* distribution that pull the world's dependency set.
    extras: tuple[str, ...] = ()
    #: Explicit upstream pins, which are what makes the install replayable.
    packages: tuple[PinnedPackage, ...] = ()
    verify_import: str = ""

    @property
    def pinned(self) -> bool:
        return bool(self.packages) and not self.unpinned

    @property
    def unpinned(self) -> tuple[str, ...]:
        """Names declared without a version, i.e. the gaps to close."""
        return tuple(pkg.name for pkg in self.packages if not pkg.version)

    @property
    def specs(self) -> tuple[str, ...]:
        return tuple(f"{pkg.name}=={pkg.version}" for pkg in self.packages)


class PrebuiltContainerInstall(_Frozen):
    """A digest-pinned OCI image we publish ourselves."""

    strategy: Literal[InstallStrategy.PREBUILT_CONTAINER]
    image: str = ""
    #: ``sha256:<64 hex>``. Empty means we have not published the image yet.
    image_digest: str = ""

    @property
    def pinned(self) -> bool:
        return bool(self.image) and bool(_SHA256_DIGEST.match(self.image_digest))

    @property
    def reference(self) -> str:
        """Digest-pinned reference, or the bare image when unpinned."""
        if not self.image_digest:
            return self.image
        return f"{self.image}@{self.image_digest}"


class SourceBuildInstall(_Frozen):
    """A runtime the user compiles; we pin the revision and cite the build docs.

    ``runtime_licenses`` is the field that makes this strategy honest rather
    than merely inconvenient: it records the copyleft closure that forbids us
    from publishing a prebuilt image, so the reason for the worse first-hour
    experience is auditable instead of folklore.
    """

    strategy: Literal[InstallStrategy.SOURCE_BUILD]
    #: Upstream repository the user clones.
    repo: str = ""
    #: Full-length commit SHA. A branch name is not a pin.
    commit: str = ""
    #: Upstream build instructions. Empty means we would be guessing.
    build_docs: str = ""
    #: Engine/plugin versions the env requires, as upstream states them.
    requires: tuple[str, ...] = ()
    #: SPDX ids in the runtime closure the user must build. Recorded because
    #: these are why we do not ship an image, not incidental trivia.
    runtime_licenses: tuple[str, ...] = ()
    verify_import: str = ""

    @property
    def pinned(self) -> bool:
        return bool(self.repo) and bool(_COMMIT_SHA.match(self.commit))

    @property
    def reference(self) -> str:
        return f"{self.repo}@{self.commit}" if self.commit else self.repo


class VendorRuntimeInstall(_Frozen):
    """A runtime we must not redistribute; the vendor ships it, we verify it."""

    strategy: Literal[InstallStrategy.VENDOR_RUNTIME]
    vendor: str = ""
    #: The vendor's own container reference. Must name ``pinned_version``.
    container: str = ""
    #: Registry the user authenticates against before the pull, when required.
    login_registry: str = ""
    #: Terms the user must accept. Empty means we have not recorded them, and
    #: informed acceptance of unrecorded terms is not a thing we will ask for.
    eula_url: str = ""
    pinned_version: str = ""

    @property
    def pinned(self) -> bool:
        return bool(self.pinned_version) and self.pinned_version in self.container


InstallSpec = Annotated[
    FirstPartyInstall
    | PipExtraInstall
    | PrebuiltContainerInstall
    | SourceBuildInstall
    | VendorRuntimeInstall,
    Field(discriminator="strategy"),
]


class WorldPackage(_Frozen):
    """One catalog row: identity, disposition, provenance, install strategy."""

    id: WorldPackageId
    display_name: str
    domain: str
    engine: str
    disposition: Disposition
    #: SPDX expression read from the upstream license text, ``unverified`` when
    #: we have not audited it, or ``NOASSERTION`` when upstream ships no license
    #: file at all. The last case is a finding, not a gap: no license means no
    #: permission, and it is classified by the denylist rather than excused.
    license: str = UNVERIFIED
    #: Repo / arXiv references the disposition was decided from.
    source: tuple[str, ...] = ()
    #: Kernel world kind this world publishes under; empty until decided.
    world_kind: str = ""
    #: Pinned world revision; empty until the wrap pins it.
    world_pin: str = ""
    install: InstallSpec
    #: Whether *no* audited env in this world exposes a gate-bindable signal.
    #: A package-level ceiling, not the gate rule: eligibility is decided per
    #: env by :class:`AuditedEnv`, because it genuinely varies per scene.
    metrics_only: bool = True
    #: Envs read first-hand at ``world_pin``, each with its own signal surface.
    #: Empty means unaudited, which is honest for a survey row and is *not*
    #: the same as "publishes nothing": an unaudited env constrains nothing.
    envs: tuple[AuditedEnv, ...] = ()
    #: Measured execution-determinism class for this world (N3). Never
    #: assumed: ``unmeasured`` stands until a seeded rerun is actually
    #: compared, and ``determinism_evidence`` says which command measured it
    #: or which concrete blocker stopped the measurement.
    determinism: DeterminismClass = DeterminismClass.UNMEASURED
    determinism_evidence: str = ""
    #: Where the safety verdict came from: the file and symbol we read, or the
    #: command we ran. Empty is only honest for undisposed survey rows.
    safety_evidence: str = ""
    notes: str = ""
    risks: str = ""

    @property
    def strategy(self) -> InstallStrategy:
        return self.install.strategy

    @property
    def license_verified(self) -> bool:
        """Whether we have read the upstream terms. Says nothing about permission."""
        return bool(self.license) and self.license != UNVERIFIED

    @property
    def license_verdict(self) -> LicenseVerdict:
        """Classification of the recorded SPDX expression against §2.4's tables."""
        return classify_license(self.license)

    @property
    def license_permitted(self) -> bool:
        """Whether the recorded terms actually allow a commercial wrap.

        Distinct from :attr:`license_verified` on purpose, and the distinction
        is load-bearing: CathSim's terms are now *read* (CC-BY-NC-SA-4.0), and
        reading them is exactly what proves we may not ship it. Treating
        "audited" as "cleared" would turn a completed audit into a green light.
        """
        return self.license_verified and self.license_verdict.status is LicenseStatus.ALLOWED

    @property
    def pin_state(self) -> str:
        """One word for the ``worlds list`` pin column.

        Both halves matter: an install pin without a world pin reproduces the
        software but not the world, and a world pin without an install pin
        names a revision nobody can fetch the same way twice.
        """
        world = bool(self.world_pin)
        install = self.install.pinned
        if world and install:
            return "pinned"
        if install:
            return "world-unpinned"
        if world:
            return "install-unpinned"
        return "unpinned"

    @property
    def installable(self) -> bool:
        """Whether *we* can install this world for you today.

        ``source-build`` is pinned data but not an install we can perform: the
        runtime is compiled by the user under upstream's own constraints, so
        counting it here would promise a command that does not exist.
        """
        return (
            self.disposition in {Disposition.SHIPPED, Disposition.WRAP}
            and self.license_permitted
            and self.install.pinned
            and self.strategy is not InstallStrategy.SOURCE_BUILD
        )

    def audited_env(self, env_id: str) -> AuditedEnv | None:
        """The audited record for one upstream env, or ``None`` if unaudited.

        ``None`` is deliberately not an error: a catalogued world may have
        scenes nobody has read yet, and a wrap of one of those is legitimate
        self-service. It constrains nothing, which is the honest state.
        """
        for env in self.envs:
            if env.env_id == env_id:
                return env
        return None

    @property
    def gate_eligible_envs(self) -> tuple[str, ...]:
        """Audited env ids where a hard gate can honestly bind."""
        return tuple(env.env_id for env in self.envs if env.safety_eligible)

    @model_validator(mode="after")
    def _metrics_only_matches_the_audited_envs(self) -> Self:
        """A safety claim must be backed by a signal somebody actually read.

        Only the unsafe direction is refused. Declaring ``metrics_only = false``
        with no gate-eligible audited env is a claim with no evidence under it.
        The reverse - ``true`` while some scene does publish a force - stays
        legal, because under-claiming is the safe error and is exactly the
        pre-audit state of a world whose scenes have not all been read.
        """
        if self.metrics_only or not self.envs:
            return self
        if not self.gate_eligible_envs:
            audited = ", ".join(env.env_id for env in self.envs)
            raise TaskContractError(
                f"world {self.id!r} declares metrics_only = false, but none of its audited "
                f"envs ({audited}) publishes a gate-eligible signal. A signal is gate-eligible "
                "only when it is physical *and* published to info/extras. Fix: record the "
                "signal and the line it is assigned on, or set metrics_only = true."
            )
        return self

    @model_validator(mode="after")
    def _determinism_claim_names_its_measurement(self) -> Self:
        """A determinism class stronger than ``unmeasured`` must say what measured it.

        Same rule the conformance suite already applies to a running task
        (§2.2: measured, never assumed), applied to catalog data so a row
        cannot inherit a reproducibility claim nobody produced. ``unmeasured``
        with an evidence string is fine and is the useful case: the string
        carries the concrete blocker that stopped the measurement.
        """
        if self.determinism is DeterminismClass.UNMEASURED:
            return self
        if not self.determinism_evidence.strip():
            raise TaskContractError(
                f"world {self.id!r} declares determinism {self.determinism.value!r} with no "
                "determinism_evidence. Name the command whose two runs were compared, because "
                "a reproducibility class is a measurement, not a property of the engine's "
                "reputation."
            )
        return self

    @model_validator(mode="after")
    def _shelf_rows_are_legally_eligible(self) -> Self:
        """A world under restricted terms cannot be a shelf item, audited or not.

        The gap this closes is specific and was live until the audit ran:
        ``cathsim`` sat at ``disposition = "wrap"`` while its license said
        CC-BY-NC-SA-4.0, so the moment anyone recorded the real SPDX the row
        would have started claiming a wrap target we may not ship. An empty or
        ``unverified`` license stays legal here - it is the honest pre-audit
        state - but a *classified refusal* is not something a disposition may
        override.
        """
        if self.disposition not in {Disposition.SHIPPED, Disposition.WRAP}:
            return self
        verdict = self.license_verdict
        if verdict.status is LicenseStatus.RESTRICTED:
            raise TaskContractError(
                f"world {self.id!r} is disposition {self.disposition.value!r} under license "
                f"{self.license!r} ({verdict.reason}). A wrap target must be redistributable: "
                "record the disposition as 'skip' with the license as the reason, or obtain "
                "separate written terms and record those instead."
            )
        if not self.safety_evidence.strip():
            raise TaskContractError(
                f"world {self.id!r} is disposition {self.disposition.value!r} but records no "
                "safety_evidence. Say where the gate verdict came from - the file and symbol "
                "read, or the command run - because 'metrics_only' is a claim about a specific "
                "codebase, not a default to inherit."
            )
        return self

    def describe(self) -> str:
        """Human detail for ``surgeval worlds info``."""
        lines = [
            f"{self.id}  {self.display_name}",
            f"  domain      {self.domain}",
            f"  engine      {self.engine}",
            f"  disposition {self.disposition.value}",
            f"  strategy    {self.strategy.value}",
            f"  license     {self.license} ({self.license_verdict.status.value})",
            f"  gates       {'metrics-only' if self.metrics_only else 'safety-eligible'}",
            f"  world kind  {self.world_kind or '(undecided)'}",
            f"  world pin   {self.world_pin or '(unpinned)'}",
            f"  pin state   {self.pin_state}",
            f"  determinism {self.determinism.value}",
        ]
        for line in _strategy_detail(self.install):
            lines.append(f"  {line}")
        if self.envs:
            lines.append(f"  audited     {len(self.envs)} env(s) read at the pin")
            for env in self.envs:
                eligible = ", ".join(s.key for s in env.gate_signals) or "(no hard gate can bind)"
                lines.append(f"    {env.env_id}  [{env.path}]")
                lines.append(f"      gate-eligible: {eligible}")
                for signal in env.signals:
                    where = signal.path or env.path
                    at = f"{where}:{signal.line}" if signal.line else where
                    unit = f" [{signal.unit}]" if signal.unit else ""
                    flags = "" if signal.published else " (computed, never published)"
                    pinned = (
                        " requires "
                        + ", ".join(f"{k}={v!r}" for k, v in signal.requires_parameters.items())
                        if signal.requires_parameters
                        else ""
                    )
                    lines.append(
                        f"      - {signal.key}: {signal.kind.value}{unit}{flags}{pinned}  {at}"
                    )
                for marker in env.absence_markers:
                    where = marker.path or env.path
                    lines.append(
                        f"      - (empty surface) {marker.key}: {marker.kind.value}, "
                        f"computed and never published  {where}:{marker.line}"
                    )
        else:
            lines.append("  audited     no env read at the pin; constrains nothing")
        if self.determinism_evidence.strip():
            lines.append(f"  determinism evidence: {self.determinism_evidence.strip()}")
        for name, text in (("notes", self.notes), ("risks", self.risks)):
            if text.strip():
                lines.append(f"  {name}       {text.strip()}")
        if self.safety_evidence.strip():
            lines.append(f"  safety      {self.safety_evidence.strip()}")
        if self.source:
            lines.append("  sources     " + ", ".join(self.source))
        else:
            lines.append("  sources     (none recorded)")
        return "\n".join(lines)


def _strategy_detail(spec: InstallSpec) -> tuple[str, ...]:
    """Strategy-specific lines for ``worlds info``, including admitted gaps."""
    if isinstance(spec, FirstPartyInstall):
        return (
            f"dist        {spec.distribution or '(in-tree)'}",
            f"import      {spec.verify_import or '(none declared)'}",
            f"version     {spec.pinned_version or '(unpinned)'}",
        )
    if isinstance(spec, PipExtraInstall):
        packages = ", ".join(
            f"{pkg.name}=={pkg.version}" if pkg.version else f"{pkg.name} (unpinned)"
            for pkg in spec.packages
        )
        return (
            f"extras      {', '.join(spec.extras) or '(none)'}",
            f"packages    {packages or '(none declared)'}",
            f"import      {spec.verify_import or '(none declared)'}",
        )
    if isinstance(spec, PrebuiltContainerInstall):
        return (
            f"image       {spec.image or '(not published)'}",
            f"digest      {spec.image_digest or '(not published)'}",
        )
    if isinstance(spec, SourceBuildInstall):
        return (
            f"repo        {spec.repo or '(unrecorded)'}",
            f"commit      {spec.commit or '(unpinned)'}",
            f"build docs  {spec.build_docs or '(not recorded)'}",
            f"requires    {', '.join(spec.requires) or '(unrecorded)'}",
            f"runtime lic {', '.join(spec.runtime_licenses) or '(unrecorded)'}",
            f"import      {spec.verify_import or '(none declared)'}",
        )
    return (
        f"vendor      {spec.vendor or '(unrecorded)'}",
        f"container   {spec.container or '(unrecorded)'}",
        f"registry    {spec.login_registry or '(none)'}",
        f"eula        {spec.eula_url or '(not recorded)'}",
        f"version     {spec.pinned_version or '(unpinned)'}",
    )


class WorldCatalog(_Frozen):
    """The whole catalog, with unique ids enforced at load."""

    catalog_version: str
    worlds: tuple[WorldPackage, ...]

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        seen: set[str] = set()
        for pkg in self.worlds:
            if pkg.id in seen:
                raise TaskContractError(f"duplicate world package id {pkg.id!r} in catalog")
            seen.add(pkg.id)
        return self

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(pkg.id for pkg in self.worlds)

    def get(self, world_id: str) -> WorldPackage | None:
        for pkg in self.worlds:
            if pkg.id == world_id:
                return pkg
        return None

    def require(self, world_id: str) -> WorldPackage:
        """Look up a world, or refuse with the known ids listed."""
        pkg = self.get(world_id)
        if pkg is None:
            known = ", ".join(self.ids)
            raise TaskContractError(f"unknown world package {world_id!r}; catalog has: {known}")
        return pkg

    def select(
        self,
        *,
        disposition: Disposition | None = None,
        strategy: InstallStrategy | None = None,
    ) -> tuple[WorldPackage, ...]:
        return tuple(
            pkg
            for pkg in self.worlds
            if (disposition is None or pkg.disposition is disposition)
            and (strategy is None or pkg.strategy is strategy)
        )


_DEFAULT: WorldCatalog | None = None


def load_catalog(path: Path | None = None) -> WorldCatalog:
    """Load and validate a catalog; the packaged default is cached.

    A malformed catalog is a :class:`TaskContractError` naming the file: the
    catalog is a published artifact, and shipping one nobody can parse must
    fail loudly rather than degrade to an empty world list.
    """
    global _DEFAULT
    if path is None and _DEFAULT is not None:
        return _DEFAULT
    target = CATALOG_PATH if path is None else path
    try:
        raw = tomllib.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TaskContractError(f"cannot read world catalog {target}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise TaskContractError(f"world catalog {target} is not valid TOML: {exc}") from exc
    try:
        catalog = WorldCatalog.model_validate(raw)
    except ValidationError as exc:
        raise TaskContractError(f"world catalog {target} is invalid: {exc}") from exc
    if path is None:
        _DEFAULT = catalog
    return catalog


def world_package(world_id: str, *, catalog: WorldCatalog | None = None) -> WorldPackage:
    """One world by id, or refuse with the known ids listed."""
    return (catalog or load_catalog()).require(world_id)


def planner_catalog_data(catalog: WorldCatalog | None = None) -> dict[str, object]:
    """Compact, machine-readable world evidence for evaluation planners."""
    resolved = catalog or load_catalog()
    return {
        "format_version": "1",
        "catalog_version": resolved.catalog_version,
        "worlds": [
            {
                "id": world.id,
                "display_name": world.display_name,
                "domain": world.domain,
                "engine": world.engine,
                "disposition": world.disposition.value,
                "license": world.license,
                "sources": list(world.source),
                "world_kind": world.world_kind,
                "world_pin": world.world_pin,
                "metrics_only": world.metrics_only,
                "determinism": world.determinism.value,
                "install": {
                    "strategy": world.strategy.value,
                    "pin_state": world.pin_state,
                    "installable": world.installable,
                },
                "environments": [
                    {
                        "env_id": environment.env_id,
                        "safety_eligible": environment.safety_eligible,
                        "signals": [
                            {
                                "key": signal.key,
                                "kind": signal.kind.value,
                                "unit": signal.unit,
                                "published": signal.published,
                                "gate_eligible": signal.gate_eligible,
                                "requires_parameters": signal.requires_parameters,
                            }
                            for signal in environment.signals
                        ],
                    }
                    for environment in world.envs
                ],
                "safety_evidence": world.safety_evidence.strip(),
                "risks": world.risks.strip(),
            }
            for world in resolved.worlds
        ],
    }


def iter_packages(
    *,
    disposition: Disposition | None = None,
    strategy: InstallStrategy | None = None,
    catalog: WorldCatalog | None = None,
) -> tuple[WorldPackage, ...]:
    """Catalog rows in file order, optionally filtered."""
    return (catalog or load_catalog()).select(disposition=disposition, strategy=strategy)
