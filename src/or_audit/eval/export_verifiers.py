"""Export a task package as a verifiers-style training environment.

The eval-time surface (``export-rl``) is post-hoc: it recomputes projections
from vectors that a scored job already produced. This is the train-time
surface. A training loop points at the generated package, rolls the task's own
pinned world, scores it with the task's own verifier, and receives the
task-declared versioned projection of the resulting vector as its reward.

Three properties make that reward honest, and each is enforced here rather
than documented:

* The reward is recomputed by :func:`or_audit.eval.vector.project` from a
  freshly scored :class:`~or_audit.eval.vector.TrialVector` on every rollout.
  The world's own reward channel is never read and no stored float is trusted.
* A failed hard gate projects to zero, because that is what the task's
  declarative projection rule says, not because the export decided so.
* No scalar leaves the export without its projection digest and the canonical
  digest of the parent vector (ASSESSMENT R3). :func:`emit_reward_record` is
  the only sanctioned constructor and :class:`RewardRecord` refuses a record
  missing either reference, so an export cannot ship a bare number. The
  generated rubric does not trust that record either: it re-derives the parent
  reference from the vector beside it and recomputes the scalar, so a forged
  record cannot become a training signal.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from or_audit.audit.canonical import digest
from or_audit.errors import ScoreContractError, TaskContractError
from or_audit.eval.contracts import GateProjectionPolicy, InteractionMode
from or_audit.eval.enums import ProjectionId
from or_audit.eval.gym_world import GymEnv, GymFactory, make_gym, run_gym_episode, sample_action
from or_audit.eval.integrity import tree_digest
from or_audit.eval.loader import load_task
from or_audit.eval.plugins import VerifierRuntime, load_verifier_runtime
from or_audit.eval.predict import index_items, load_items
from or_audit.eval.runner import SAFETY_MAX_PEN, preprocess_observation, stream_adapters
from or_audit.eval.sim import get_simulation_engine, world_kind_spec
from or_audit.eval.task import ProjectionSpec, TaskSpec
from or_audit.eval.vector import TrialVector
from or_audit.eval.verifier import score_context
from or_audit.version import PACKAGE_VERSION

#: Directory the task package is copied into, relative to the export root.
TASK_PACKAGE_DIR = "task"

#: Identity recorded for a policy that is being trained: a training-time
#: reward has no pinned agent package, and saying so is better than borrowing
#: an eval identity the record has not earned.
DEFAULT_AGENT_IDENTITY = "unpinned-training-policy"

#: One step of a policy under training: ``(observation, step) -> action``.
#: Closed-loop worlds consume the action; prediction-mode tasks consume the
#: returned object as the prediction payload.
Policy = Callable[[Any, int], Any]

# Mirrors the filter in or_audit.eval.integrity.tree_digest. write_export
# re-checks the copy's digest against the source's, so a divergence refuses
# the export instead of shipping a package whose pin no longer matches.
_IGNORED_PARTS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".git"})
_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})


class RewardRecord(BaseModel):
    """One reward with the provenance that makes it interpretable.

    A scalar without its projection rule and its parent vector is unusable
    evidence: nobody downstream can tell which collapse produced it, whether a
    hard gate zeroed it, or which vector it came from. ASSESSMENT R3 forbids
    emitting one, so this model refuses to exist without both references.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reward: float
    projection_id: str
    projection_version: str
    projection_digest: str
    #: Canonical digest of the authoritative vector this reward projects from.
    parent_vector_ref: str
    task_id: str
    task_version: str
    task_digest: str
    world_pin: str
    agent_identity: str
    seed: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _reward_carries_its_provenance(self) -> Self:
        if not self.projection_digest:
            raise ScoreContractError(
                "a reward record requires the digest of the projection rule that "
                "produced it; an undigested collapse is not reproducible and "
                "ASSESSMENT R3 refuses to emit the scalar"
            )
        if not self.parent_vector_ref:
            raise ScoreContractError(
                "a reward record requires the canonical digest of its parent trial "
                "vector; a scalar with no vector reference cannot be audited back to "
                "the gates that produced it (ASSESSMENT R3)"
            )
        if not self.projection_id or not self.task_id or not self.task_digest:
            raise ScoreContractError(
                "a reward record requires a projection id and a digest-pinned task; "
                "an unattributed reward is not evidence"
            )
        if not self.agent_identity:
            raise ScoreContractError(
                "a reward record requires an agent identity; declare the policy under "
                f"training or accept {DEFAULT_AGENT_IDENTITY!r}"
            )
        if not math.isfinite(self.reward):
            raise ScoreContractError(
                f"projection {self.projection_id} produced a non-finite reward "
                f"({self.reward}); a training signal is not a placeholder"
            )
        return self


def vector_reference(vector: TrialVector) -> str:
    """Canonical digest of the vector a reward projects from."""
    return digest(vector.model_dump(mode="json"))


def emit_reward_record(
    *,
    reward: float,
    projection_id: str,
    projection_version: str,
    projection_digest: str,
    parent_vector_ref: str,
    task_id: str,
    task_version: str,
    task_digest: str,
    world_pin: str,
    agent_identity: str,
    seed: int,
) -> RewardRecord:
    """Build the only kind of reward an export is allowed to return.

    Takes the projection identity as plain strings rather than a
    :class:`~or_audit.eval.task.ProjectionSpec` so the invariant is checked on
    the values that actually ship: a caller that loses the digest on the way
    here is refused, instead of having one silently regenerated for it.
    """
    return RewardRecord(
        reward=reward,
        projection_id=projection_id,
        projection_version=projection_version,
        projection_digest=projection_digest,
        parent_vector_ref=parent_vector_ref,
        task_id=task_id,
        task_version=task_version,
        task_digest=task_digest,
        world_pin=world_pin,
        agent_identity=agent_identity,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class VectorRollout:
    """One rollout: the vector, its reward record, and the evidence behind it."""

    seed: int
    vector: TrialVector
    record: RewardRecord
    info: dict[str, Any]
    steps: tuple[dict[str, Any], ...]

    @property
    def reward(self) -> float:
        """The projected reward. Only reachable through a provenance-bearing record."""
        return self.record.reward

    def to_state(self) -> dict[str, Any]:
        """Rollout state in the shape a verifiers rubric reads."""
        return {
            "seed": self.seed,
            "reward_record": self.record.model_dump(mode="json"),
            "vector": self.vector.model_dump(mode="json"),
            "info": self.info,
        }


def projection_for_task(task: TaskSpec, projection_id: ProjectionId | str) -> ProjectionSpec:
    """Resolve the requested projection or refuse.

    Message shapes mirror :func:`or_audit.eval.export_rl._spec_for_task`: the
    train-time and eval-time surfaces must refuse the same task the same way,
    or a training loop will read a disagreement as a bug in one of them.
    """
    requested = projection_id.value if isinstance(projection_id, ProjectionId) else projection_id
    if task.projection is None:
        raise TaskContractError(
            f"task {task.id} has no declared projection or diverged gating rule; "
            f"declare [projection] in the package's verifier.toml before exporting "
            f"a training environment"
        )
    declared = task.projection.id
    declared_text = declared.value if isinstance(declared, ProjectionId) else declared
    if declared_text != requested:
        raise TaskContractError(f"task {task.id} declares {declared_text!r}, not {requested!r}")
    if task.projection.gate_failure is not GateProjectionPolicy.ZERO:
        raise TaskContractError(
            f"task {task.id} projection {declared_text!r} declares "
            f"gate_failure={task.projection.gate_failure.value!r}; a train-time export "
            f"must project a hard-gate failure to zero, because a training loop needs a "
            f"reward for the unsafe episode rather than an exception mid-rollout. Fix: "
            f'declare gate_failure = "zero" in the package\'s [projection], or keep this '
            f"projection eval-only and export a gated one for training. (Note: "
            f'gate_unassessable may stay "refuse" — an abstained gate is a measurement '
            f"gap, and handing a training loop 0.0 for it would fabricate evidence of "
            f"safety.)"
        )
    return task.projection


def assert_exportable(task: TaskSpec) -> None:
    """Refuse tasks whose reward would train against something we do not measure.

    Each refusal here is a product boundary, not a lint: an RL loop that fits
    to a synthetic stand-in, or to a world with no safety instrumentation,
    produces exactly the "trained on physics nobody validated" result this
    harness exists to make impossible.
    """
    if task.environment.synthetic_stub:
        raise TaskContractError(
            f"task {task.id} declares environment.synthetic_stub, so its world is a "
            f"non-physical stand-in; refusing to export a training reward derived "
            f"from fabricated physics. Fix: attach a real simulation backend for "
            f"{task.environment.kind_key} and drop environment.synthetic_stub"
        )
    if task.environment.metrics_only:
        raise TaskContractError(
            f"task {task.id} is environment.metrics_only, so its world reports no "
            f"safety state and the task declares no hard gates; refusing to export a "
            f"training reward from a world with no safety instrumentation, because "
            f"the projection could never be gate-zeroed and the reward would look "
            f"safe by construction. Fix: upstream the force/collision/penetration "
            f"instrumentation this wrap is missing, declare the gates it enables, "
            f"and drop environment.metrics_only"
        )
    if task.metadata.safety_critical and not task.verifier.gates:
        # The loader already refuses this pairing; restated because export is a
        # separate trust boundary and a gate-free safety-critical reward is the
        # one shape that must never reach a training loop.
        raise TaskContractError(
            f"task {task.id} is metadata.safety_critical but declares no hard gates; "
            f"refusing to export a training reward that can never be zeroed by a "
            f"safety failure. Fix: declare the task's hard gates, or label the "
            f"package environment.metrics_only and accept that it is not exportable"
        )


class ExportFile(BaseModel):
    """One generated text file, relative to the export root."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    text: str


class VerifiersExport(BaseModel):
    """A generated verifiers-style environment package and its pins."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    environment_id: str
    task_id: str
    task_version: str
    task_digest: str
    projection_id: str
    projection_version: str
    projection_digest: str
    projection_identity: str
    world_kind: str
    world_pin: str
    adapter_id: str
    adapter_digest: str
    interaction_mode: str
    harness_version: str
    #: Package the export was generated from; its durable files are copied in.
    task_source: Path
    files: tuple[ExportFile, ...]
    task_files: tuple[str, ...]

    @property
    def paths(self) -> tuple[str, ...]:
        """Every path the export writes, relative to the export root."""
        return tuple(file.path for file in self.files) + tuple(
            f"{TASK_PACKAGE_DIR}/{name}" for name in self.task_files
        )


def _durable_files(root: Path) -> tuple[str, ...]:
    """Relative posix paths a copy must carry for the tree digest to hold."""
    names: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in _IGNORED_PARTS for part in relative.parts):
            continue
        if path.suffix in _IGNORED_SUFFIXES:
            continue
        names.append(relative.as_posix())
    return tuple(names)


def _toml_string(value: str) -> str:
    """Render a TOML basic string. JSON escaping is a subset TOML accepts."""
    return json.dumps(value)


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def _env_toml(export: VerifiersExport, task: TaskSpec) -> str:
    lines = [
        "# Generated by `surgeval export-verifiers`. Do not edit by hand.",
        "#",
        "# The scalar this environment returns is a task-declared, versioned",
        "# projection of a safety vector, not a score. Every pin below is part of",
        "# what makes the reward interpretable; changing one invalidates the",
        "# rewards produced under it.",
        "",
        "[environment]",
        f"id = {_toml_string(export.environment_id)}",
        'entrypoint = "load_environment.py:load_environment"',
        'harness = "surgeval"',
        f"harness_version = {_toml_string(export.harness_version)}",
        f"interaction_mode = {_toml_string(export.interaction_mode)}",
        "",
        "[task]",
        f"id = {_toml_string(export.task_id)}",
        f"version = {_toml_string(export.task_version)}",
        f"digest = {_toml_string(export.task_digest)}",
        f"package = {_toml_string(TASK_PACKAGE_DIR)}",
        f"headline = {_toml_string(task.verifier.headline)}",
        f"gates = [{', '.join(_toml_string(gate.id) for gate in task.verifier.gates)}]",
        "",
        "[projection]",
        f"id = {_toml_string(export.projection_id)}",
        f"version = {_toml_string(export.projection_version)}",
        f"digest = {_toml_string(export.projection_digest)}",
        f"identity = {_toml_string(export.projection_identity)}",
        "",
        "[world]",
        f"kind = {_toml_string(export.world_kind)}",
        f"pin = {_toml_string(export.world_pin)}",
        f"gym_id = {_toml_string(task.environment.gym_id)}",
        f"adapter_id = {_toml_string(export.adapter_id)}",
        f"adapter_digest = {_toml_string(export.adapter_digest)}",
        f"metrics_only = {_toml_bool(task.environment.metrics_only)}",
        f"synthetic_stub = {_toml_bool(task.environment.synthetic_stub)}",
        "",
    ]
    return "\n".join(lines)


def _pyproject_toml(export: VerifiersExport) -> str:
    dependency = _toml_string(f"surgeval>={export.harness_version}")
    lines = [
        "# Generated by `surgeval export-verifiers`. Do not edit by hand.",
        "[project]",
        f"name = {_toml_string(export.environment_id)}",
        f"version = {_toml_string(export.task_version)}",
        "description = "
        + _toml_string(
            f"surgeval task {export.task_id} as a verifiers-style environment; "
            f"reward is the {export.projection_id} projection of its safety vector"
        ),
        'requires-python = ">=3.11"',
        f"dependencies = [{dependency}]",
        "",
        "[project.optional-dependencies]",
        "# Only needed to attach a verifiers Rubric; the environment loads without it.",
        'verifiers = ["verifiers>=0.1"]',
        "",
        "[build-system]",
        'requires = ["hatchling"]',
        'build-backend = "hatchling.build"',
        "",
        "# Pins the rewards produced by this environment are only valid under.",
        "[tool.surgeval.export]",
        f"task_id = {_toml_string(export.task_id)}",
        f"task_version = {_toml_string(export.task_version)}",
        f"task_digest = {_toml_string(export.task_digest)}",
        f"projection_identity = {_toml_string(export.projection_identity)}",
        f"world_kind = {_toml_string(export.world_kind)}",
        f"world_pin = {_toml_string(export.world_pin)}",
        f"adapter_id = {_toml_string(export.adapter_id)}",
        f"adapter_digest = {_toml_string(export.adapter_digest)}",
        "",
    ]
    return "\n".join(lines)


def _readme_md(export: VerifiersExport, task: TaskSpec) -> str:
    gates = ", ".join(f"`{gate.id}`" for gate in task.verifier.gates) or "(none)"
    lines = [
        f"# {export.environment_id}",
        "",
        f"surgeval task `{export.task_id}@{export.task_version}` exported as a",
        "verifiers-style environment. Generated by `surgeval export-verifiers`;",
        "do not edit by hand.",
        "",
        "## What the reward is",
        "",
        "The reward is the task's own declared, versioned projection of a safety",
        "vector:",
        "",
        f"    {export.projection_identity}",
        "",
        "On every rollout the environment",
        "rolls the pinned world, scores it with the task's own verifier, and",
        "recomputes the projection from the resulting trial vector. It never reads",
        "the world's reward channel and never trusts a stored float.",
        "",
        f"- Hard gates: {gates}. **A failed hard gate projects to 0**, by the task's",
        "  declarative rule, not by this export's choice.",
        "- An unassessable gate refuses rather than guesses, per the same rule.",
        "- The full vector (every gate and metric) is logged on each rollout, and the",
        "  scalar is only reachable through a `RewardRecord` carrying the projection",
        "  digest and the canonical digest of its parent vector (ASSESSMENT R3).",
        "",
        "## What the reward is not",
        "",
        "It is not a score, not a leaderboard row, and not comparable across worlds.",
        "A single number cannot express a safety result; this one exists because a",
        "training loop needs a gradient, and it carries the rule that produced it so",
        "the collapse stays auditable.",
        "",
        "`reward.txt` is **not** an interface here. Nothing in this package reads or",
        "writes one, and a reward without its projection digest and parent vector",
        "reference is refused at construction rather than written to a file.",
        "",
        "## Pins",
        "",
        f"- task digest: `{export.task_digest}`",
        f"- projection: `{export.projection_identity}`",
        f"- world: `{export.world_kind}` pin `{export.world_pin or '(unpinned)'}`",
        f"- world adapter: `{export.adapter_id or '(unattached)'}`"
        f" digest `{export.adapter_digest or '(none)'}`",
        f"- harness: `surgeval {export.harness_version}`",
        "",
        f"The task package is vendored under `{TASK_PACKAGE_DIR}/`. `load_environment`",
        "re-digests it on load and refuses if the content no longer matches the pin,",
        "so a locally edited verifier cannot quietly change the reward.",
        "",
        "## Use",
        "",
        "```python",
        "from load_environment import load_environment",
        "",
        "env = load_environment()",
        "rollout = env.rollout(policy, seed=0)",
        "rollout.reward           # float, gated projection",
        "rollout.record           # RewardRecord: digest + parent vector reference",
        "rollout.vector           # the authoritative gate/metric vector",
        "```",
        "",
        "`verifiers` is imported lazily: when it is installed, `load_environment`",
        "attaches a single-function `Rubric`. That rubric does not read the recorded",
        "scalar on trust — it revalidates the record, checks its pins against this",
        "package's own, requires `parent_vector_ref` to be the canonical digest of the",
        "vector in the same state, and recomputes the projection. A hand-written state",
        "carrying an invented reward is refused, not trained on. Without `verifiers`",
        "installed, the environment and its reward are unchanged.",
        "",
    ]
    return "\n".join(lines)


_LOADER_HEADER = '''"""surgeval task {task_id} as a verifiers-style environment.

Generated by `surgeval export-verifiers`. Do not edit by hand.

The reward is the task's declared versioned projection

    {projection_identity}

recomputed from a freshly scored trial vector on every rollout. The world's own
reward channel is never read and no stored float is trusted. A failed hard gate
projects to 0. `reward.txt` is not an interface here.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

from or_audit.errors import ScoreContractError, TaskContractError
from or_audit.eval.export_verifiers import (
    DEFAULT_AGENT_IDENTITY,
    Policy,
    RewardRecord,
    VectorRollout,
    close_runtime,
    emit_reward_record,
    observation_for,
    open_verifier,
    open_world,
    prediction_items,
    prediction_vector,
    projection_for_task,
    rollout_vector,
    vector_reference,
)
from or_audit.eval.gym_world import GymEnv, GymFactory
from or_audit.eval.integrity import tree_digest
from or_audit.eval.loader import load_task
from or_audit.eval.plugins import VerifierRuntime
from or_audit.eval.task import ProjectionSpec, TaskSpec
from or_audit.eval.vector import TrialVector, project

TASK_PACKAGE = {task_package!r}
TASK_ID = {task_id!r}
TASK_VERSION = {task_version!r}
TASK_DIGEST = {task_digest!r}
PROJECTION_ID = {projection_id!r}
PROJECTION_VERSION = {projection_version!r}
PROJECTION_DIGEST = {projection_digest!r}
PROJECTION_IDENTITY = f"{{PROJECTION_ID}}@{{PROJECTION_VERSION}}+{{PROJECTION_DIGEST}}"
WORLD_KIND = {world_kind!r}
WORLD_PIN = {world_pin!r}
ADAPTER_ID = {adapter_id!r}
ADAPTER_DIGEST = {adapter_digest!r}
INTERACTION_MODE = {interaction_mode!r}
CLOSED_LOOP = {closed_loop!r}
'''

_LOADER_BODY = '''

def pinned_task() -> tuple[TaskSpec, Path, ProjectionSpec]:
    """Load the vendored task package, refusing any drift from its pins."""
    root = Path(__file__).resolve().parent / TASK_PACKAGE
    if not root.is_dir():
        raise TaskContractError(
            f"exported environment is missing its task package: {root}"
        )
    found = tree_digest(root)
    if found != TASK_DIGEST:
        raise TaskContractError(
            f"vendored task package digest {found} does not match the exported pin "
            f"{TASK_DIGEST}; the reward rule, verifier, or world pin has been edited "
            f"since export, so rewards from this package would not be the rewards "
            f"the export attested. Re-run surgeval export-verifiers"
        )
    task = load_task(root)
    projection = projection_for_task(task, PROJECTION_ID)
    if projection.identity != PROJECTION_IDENTITY:
        raise TaskContractError(
            f"task {task.id} now projects {projection.identity}, not the exported "
            f"{PROJECTION_IDENTITY}; refusing to train against a changed collapse"
        )
    return task, root, projection


def _verifiers_module() -> ModuleType | None:
    """Import verifiers if it is installed. The export works without it."""
    try:
        import verifiers
    except ImportError:
        return None
    return verifiers


class VectorProjectionEnv:
    """Environment whose reward is a gated projection of a safety vector.

    Reward emission is deliberately funnelled through one method: every scalar
    this environment returns carries the projection digest and the canonical
    digest of the vector it came from, so no bare number can escape.
    """

    def __init__(
        self,
        *,
        gym_factory: GymFactory | None = None,
        agent_identity: str = DEFAULT_AGENT_IDENTITY,
        n: int | None = None,
        **unknown: Any,
    ) -> None:
        if unknown:
            raise TaskContractError(
                f"load_environment does not accept {sorted(unknown)}; an exported "
                f"environment refuses options it would silently ignore"
            )
        self.task, self.task_dir, self.projection = pinned_task()
        self.agent_identity = agent_identity
        self.n = n if n is not None else self.task.environment.n_eval_episodes
        self.rubric: Any = None
        self.records: list[RewardRecord] = []
        self._gym_factory = gym_factory
        self._world: GymEnv | None = None
        self._verifier: VerifierRuntime | None = None

    # -- the only path a scalar leaves this environment -------------------

    def emit(self, vector: TrialVector, *, seed: int) -> RewardRecord:
        """Project a scored vector into a provenance-bearing reward record."""
        reward = project(vector, self.projection)
        record = emit_reward_record(
            reward=reward,
            projection_id=PROJECTION_ID,
            projection_version=PROJECTION_VERSION,
            projection_digest=self.projection.rule_digest,
            parent_vector_ref=vector_reference(vector),
            task_id=TASK_ID,
            task_version=TASK_VERSION,
            task_digest=TASK_DIGEST,
            world_pin=WORLD_PIN,
            agent_identity=self.agent_identity,
            seed=seed,
        )
        self.records.append(record)
        return record

    # -- rollouts ---------------------------------------------------------

    def world(self) -> GymEnv:
        """The pinned world, opened once and reused across rollouts."""
        if self._world is None:
            self._world = open_world(self.task, gym_factory=self._gym_factory)
        return self._world

    def verifier(self) -> VerifierRuntime:
        """The task's own verifier, opened once and reused across rollouts."""
        if self._verifier is None:
            self._verifier = open_verifier(self.task, self.task_dir)
        return self._verifier

    def rollout(self, policy: Policy | None = None, *, seed: int = 0) -> VectorRollout:
        """Run one episode (or score one item) and return its gated reward."""
        if CLOSED_LOOP:
            vector, info, steps = rollout_vector(
                task=self.task,
                task_dir=self.task_dir,
                world=self.world(),
                seed=seed,
                agent_identity=self.agent_identity,
                policy=policy,
                verifier=self.verifier(),
            )
        else:
            vector, info, steps = prediction_vector(
                task=self.task,
                task_dir=self.task_dir,
                seed=seed,
                agent_identity=self.agent_identity,
                policy=policy,
                verifier=self.verifier(),
            )
        record = self.emit(vector, seed=seed)
        return VectorRollout(
            seed=seed, vector=vector, record=record, info=info, steps=steps
        )

    def evaluate(
        self, policy: Policy | None = None, *, n: int | None = None
    ) -> list[VectorRollout]:
        """Roll the task's declared eval seeds. Returns per-rollout evidence."""
        count = self.n if n is None else n
        return [self.rollout(policy, seed=seed) for seed in range(count)]

    def items(self) -> tuple[dict[str, Any], ...]:
        """Dataset rows for a prediction-mode task; empty for closed-loop worlds."""
        if CLOSED_LOOP:
            return ()
        inputs, _labels = prediction_items(self.task, self.task_dir)
        return tuple(observation_for(self.task, item) for item in inputs)

    # -- verifiers rubric compatibility -----------------------------------

    def reward_func(self, *, state: dict[str, Any], **kwargs: Any) -> float:
        """Recompute the reward from the state's own vector, or refuse.

        This is the RL boundary: whatever this returns is the training signal.
        Reading `state["reward_record"]["reward"]` on trust would make any dict
        with a float and two plausible strings a reward, so nothing here is
        believed. The record is revalidated as a `RewardRecord`, its pins are
        checked against this export's own constants, `parent_vector_ref` must be
        the canonical digest of the vector actually supplied, and the scalar must
        equal what the task's declared projection computes from that vector here
        and now.
        """
        del kwargs
        raw_record = state.get("reward_record")
        if not isinstance(raw_record, dict):
            raise ScoreContractError(
                "rollout state carries no reward_record; this environment does not "
                "produce a reward outside a recorded projection"
            )
        try:
            record = RewardRecord.model_validate(raw_record)
        except ScoreContractError:
            raise
        except Exception as exc:
            raise ScoreContractError(
                f"rollout state carries something that is not a reward record, so "
                f"the scalar in it has no provenance to check (ASSESSMENT R3): {exc}"
            ) from exc
        for field, found, expected in (
            ("projection_id", record.projection_id, PROJECTION_ID),
            ("projection_version", record.projection_version, PROJECTION_VERSION),
            ("projection_digest", record.projection_digest, PROJECTION_DIGEST),
            ("task_id", record.task_id, TASK_ID),
            ("task_version", record.task_version, TASK_VERSION),
            ("task_digest", record.task_digest, TASK_DIGEST),
            ("world_pin", record.world_pin, WORLD_PIN),
        ):
            if found != expected:
                raise ScoreContractError(
                    f"reward record {field}={found!r} is not this export's "
                    f"{expected!r}; a reward earned under a different task, world, or "
                    f"collapse rule is not a training signal for this one"
                )
        raw_vector = state.get("vector")
        if not isinstance(raw_vector, dict):
            raise ScoreContractError(
                "rollout state carries a reward record but not the vector it claims to "
                "project; without the vector the reward cannot be recomputed and this "
                "environment does not return unrecomputed scalars"
            )
        try:
            vector = TrialVector.model_validate(raw_vector)
        except ScoreContractError:
            raise
        except Exception as exc:
            raise ScoreContractError(
                f"rollout state carries something that is not a trial vector, so the "
                f"reward has nothing to be recomputed from: {exc}"
            ) from exc
        found_ref = vector_reference(vector)
        if record.parent_vector_ref != found_ref:
            raise ScoreContractError(
                f"reward record names parent vector {record.parent_vector_ref} but the "
                f"vector in this state digests to {found_ref}; the scalar and the "
                f"evidence beside it did not come from the same trial (ASSESSMENT R3)"
            )
        recomputed = project(vector, self.projection)
        if recomputed != record.reward:
            raise ScoreContractError(
                f"reward record claims {record.reward!r} but {PROJECTION_IDENTITY} "
                f"projects {recomputed!r} from the vector it references; refusing to "
                f"train on a scalar the projection rule does not reproduce"
            )
        return recomputed

    def attach_rubric(self, verifiers: ModuleType) -> None:
        """Attach a single-function rubric that recomputes the recorded reward."""
        self.rubric = verifiers.Rubric(funcs=[self.reward_func], weights=[1.0])

    def close(self) -> None:
        close_runtime(self._world)
        close_runtime(self._verifier)
        self._world = None
        self._verifier = None


def load_environment(**kwargs: Any) -> VectorProjectionEnv:
    """Prime Environments Hub entry point."""
    env = VectorProjectionEnv(**kwargs)
    verifiers = _verifiers_module()
    if verifiers is not None:
        env.attach_rubric(verifiers)
    return env
'''


def _loader_source(export: VerifiersExport) -> str:
    header = _LOADER_HEADER.format(
        task_package=TASK_PACKAGE_DIR,
        task_id=export.task_id,
        task_version=export.task_version,
        task_digest=export.task_digest,
        projection_id=export.projection_id,
        projection_version=export.projection_version,
        projection_digest=export.projection_digest,
        projection_identity=export.projection_identity,
        world_kind=export.world_kind,
        world_pin=export.world_pin,
        adapter_id=export.adapter_id,
        adapter_digest=export.adapter_digest,
        interaction_mode=export.interaction_mode,
        closed_loop=export.interaction_mode == InteractionMode.CLOSED_LOOP.value,
    )
    return header + _LOADER_BODY


def build_export(task_dir: Path, *, projection_id: ProjectionId | str) -> VerifiersExport:
    """Resolve a task package into a complete, unwritten environment export."""
    root = Path(task_dir).resolve()
    task = load_task(root)
    spec = projection_for_task(task, projection_id)
    assert_exportable(task)
    task.assert_runnable()
    kind_spec = world_kind_spec(task.environment.kind)
    projection_id_text = spec.id.value if isinstance(spec.id, ProjectionId) else spec.id
    export = VerifiersExport(
        environment_id=f"surgeval-env-{task.id}",
        task_id=task.id,
        task_version=task.task_version,
        task_digest=tree_digest(root),
        projection_id=projection_id_text,
        projection_version=spec.version,
        projection_digest=spec.rule_digest,
        projection_identity=spec.identity,
        world_kind=task.environment.kind_key,
        world_pin=task.environment.world_pin,
        adapter_id=kind_spec.adapter_id if kind_spec else "",
        adapter_digest=kind_spec.adapter_digest if kind_spec else "",
        interaction_mode=task.interface.interaction_mode.value,
        harness_version=PACKAGE_VERSION,
        task_source=root,
        files=(),
        task_files=_durable_files(root),
    )
    files = (
        ExportFile(path="env.toml", text=_env_toml(export, task)),
        ExportFile(path="load_environment.py", text=_loader_source(export)),
        ExportFile(path="pyproject.toml", text=_pyproject_toml(export)),
        ExportFile(path="README.md", text=_readme_md(export, task)),
    )
    return export.model_copy(update={"files": tuple(sorted(files, key=lambda f: f.path))})


def write_export(export: VerifiersExport, out: Path) -> tuple[Path, ...]:
    """Write an export deterministically. Identical inputs, identical bytes."""
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for file in export.files:
        path = root / file.path
        path.write_text(file.text, encoding="utf-8", newline="\n")
        written.append(path)
    package = root / TASK_PACKAGE_DIR
    if package.exists():
        # A stale vendored package would break the digest pin the loader checks.
        shutil.rmtree(package)
    for name in export.task_files:
        source = export.task_source / name
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        written.append(target)
    copied = tree_digest(package)
    if copied != export.task_digest:
        raise TaskContractError(
            f"vendored task package digest {copied} does not match the source digest "
            f"{export.task_digest}; refusing to ship an environment whose task pin is "
            f"already wrong"
        )
    return tuple(sorted(written))


def export_verifiers_environment(
    task_dir: Path,
    *,
    out: Path,
    projection_id: ProjectionId | str,
) -> VerifiersExport:
    """Generate a verifiers-style environment package for one task."""
    export = build_export(Path(task_dir), projection_id=projection_id)
    write_export(export, Path(out))
    return export


# --- runtime the generated loader drives the pinned world through -----------


def open_world(task: TaskSpec, *, gym_factory: GymFactory | None = None) -> GymEnv:
    """Open the task's world through the kernel, never a private copy."""
    if gym_factory is not None:
        return gym_factory(task)
    engine = get_simulation_engine(task)
    if engine is not None:
        return cast(GymEnv, engine)
    return make_gym(task)


def open_verifier(task: TaskSpec, task_dir: Path) -> VerifierRuntime:
    """The task's own verifier, opened once and reused across rollouts.

    Training reuses the package's verifier rather than a training-side copy of
    its scoring rule, so a reward and an eval row disagree only if the world
    differs — never because the scorer does.
    """
    if not task.verifier.entrypoint:
        raise TaskContractError(f"task {task.id} has no verifier entrypoint")
    return load_verifier_runtime(Path(task_dir), task.verifier.entrypoint)


def close_runtime(runtime: object | None) -> None:
    """Close a world or verifier that owns resources; one without close() is fine."""
    close = getattr(runtime, "close", None)
    if callable(close):
        close()


def observation_for(task: TaskSpec, item: dict[str, Any]) -> dict[str, Any]:
    """Compose the observation the eval harness would hand an agent.

    Training and evaluation must show a policy the same channels, so this
    reuses the runner's stream composition rather than a second copy of it.
    """
    return preprocess_observation(task, stream_adapters(task), item)


def rollout_vector(
    *,
    task: TaskSpec,
    task_dir: Path,
    world: GymEnv,
    seed: int,
    agent_identity: str,
    policy: Policy | None = None,
    verifier: VerifierRuntime | None = None,
) -> tuple[TrialVector, dict[str, Any], tuple[dict[str, Any], ...]]:
    """Roll one closed-loop episode and score it with the task's own verifier."""
    if task.interface.interaction_mode is not InteractionMode.CLOSED_LOOP:
        raise TaskContractError(
            f"task {task.id} is {task.interface.interaction_mode.value}, not "
            f"closed-loop; use prediction_vector"
        )
    adapters = stream_adapters(task)
    scenario = next((candidate for candidate in task.scenarios if candidate.seed == seed), None)
    perturbations = tuple(
        perturbation
        for perturbation in task.perturbations
        if perturbation.scenario_id is None
        or (scenario is not None and perturbation.scenario_id == scenario.id)
    )
    reset_options = (
        {
            "or_audit": {
                "scenario": (scenario.model_dump(mode="json") if scenario is not None else None),
                "perturbations": [
                    perturbation.model_dump(mode="json") for perturbation in perturbations
                ],
            }
        }
        if scenario is not None or perturbations
        else None
    )

    def action_fn(env: GymEnv, observation: Any, step: int) -> Any:
        if policy is None:
            return sample_action(env, seed=seed, step=step)
        return policy(preprocess_observation(task, adapters, observation), step)

    info, steps = run_gym_episode(
        world,
        seed=seed,
        action_fn=action_fn,
        max_steps=task.harness.max_steps,
        reset_options=reset_options,
    )
    unwrapped = getattr(world, "unwrapped", world)
    nested = getattr(unwrapped, "_env", unwrapped)
    safety = float(getattr(nested, "safety_max_pen", SAFETY_MAX_PEN))
    vector = score_context(
        task=task,
        task_dir=task_dir,
        agent_identity=agent_identity,
        seed=seed,
        context={
            "kind": "gym-policy",
            "info": info,
            "trajectory": [dict(step) for step in steps],
            "safety_max_pen": safety,
        },
        runtime=verifier,
    )
    return vector, info, steps


def prediction_items(
    task: TaskSpec, task_dir: Path
) -> tuple[tuple[dict[str, Any], ...], dict[str, dict[str, Any]]]:
    """The task's own inputs and oracle labels for a prediction-mode export."""
    if not task.environment.inputs_path or not task.environment.labels_path:
        raise TaskContractError(
            f"task {task.id} declares no inputs_path/labels_path, so a prediction "
            f"reward has no oracle to score against"
        )
    inputs = load_items(Path(task_dir) / task.environment.inputs_path)
    labels = index_items(load_items(Path(task_dir) / task.environment.labels_path))
    return inputs, labels


def prediction_vector(
    *,
    task: TaskSpec,
    task_dir: Path,
    seed: int,
    agent_identity: str,
    policy: Policy | None = None,
    verifier: VerifierRuntime | None = None,
) -> tuple[TrialVector, dict[str, Any], tuple[dict[str, Any], ...]]:
    """Score one prediction against the task's oracle label."""
    if policy is None:
        raise TaskContractError(
            f"task {task.id} is {task.interface.interaction_mode.value}: it has no "
            f"world to sample actions from, so a rollout needs a policy that produces "
            f"a prediction"
        )
    inputs, labels = prediction_items(task, Path(task_dir))
    if seed >= len(inputs):
        raise TaskContractError(
            f"task {task.id} declares {len(inputs)} items; seed {seed} is out of range"
        )
    item = inputs[seed]
    item_id = str(item["id"])
    if item_id not in labels:
        raise TaskContractError(f"task {task.id} has no label for item {item_id!r}")
    observation = observation_for(task, item)
    prediction = policy(observation, 0)
    if not isinstance(prediction, dict):
        raise TaskContractError(
            f"task {task.id} scores a prediction object; policy returned "
            f"{type(prediction).__name__}"
        )
    kind = (
        "counterfactual"
        if task.interface.interaction_mode is InteractionMode.COUNTERFACTUAL
        else "video-predict"
    )
    context: dict[str, Any] = {
        "kind": kind,
        "input": item,
        "label": labels[item_id],
        "prediction": prediction,
    }
    vector = score_context(
        task=task,
        task_dir=Path(task_dir),
        agent_identity=agent_identity,
        seed=seed,
        context=context,
        runtime=verifier,
    )
    return vector, {"item_id": item_id}, ({**context, "obs": observation},)
