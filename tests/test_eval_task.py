"""P0 eval contract: Harbor-shaped tasks that cannot collapse to a scalar."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from or_audit.cli import main
from or_audit.domain.enums import GateStatus
from or_audit.errors import ScoreContractError, TaskContractError
from or_audit.eval.bind import assert_bind
from or_audit.eval.loader import load_agent, load_dataset, load_task
from or_audit.eval.task import ProjectionSpec, TaskSpec
from or_audit.eval.vector import TrialVector, project
from or_audit.eval.verifier import score_context

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_TASK = ROOT / "docs" / "examples" / "tasks" / "lumen-nav-safe"
VIDEO_TASK = ROOT / "docs" / "examples" / "tasks" / "video-nextstep"
EXAMPLE_DATASET = ROOT / "docs" / "examples" / "datasets" / "lumen-nav-v0"
VIDEO_DATASET = ROOT / "docs" / "examples" / "datasets" / "video-nextstep-v0"
CATH_AGENT = ROOT / "docs" / "examples" / "agents" / "seldingermed-lumen-linear"
VIDEO_AGENT = ROOT / "docs" / "examples" / "agents" / "example-video-predictor"


def _copy_task(tmp_path: Path) -> Path:
    dest = tmp_path / "task"
    shutil.copytree(EXAMPLE_TASK, dest)
    return dest


def _patch_toml(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} not in {path}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


class TestExampleTaskLoads:
    def test_lumen_nav_safe_is_valid(self):
        task = load_task(EXAMPLE_TASK)
        assert task.id == "lumen-nav-safe"
        assert task.port is None
        assert task.interface.id == "gym-policy"
        assert task.verifier.headline == "safe_success"
        assert task.projection is not None
        assert str(task.projection.id) == "gated_reach_v0"
        assert "safe success" in task.instruction.lower()

    @pytest.mark.parametrize(
        ("path", "interface"),
        [
            (EXAMPLE_TASK, "gym-policy"),
            (VIDEO_TASK, "video-predict"),
            (ROOT / "docs" / "examples" / "tasks" / "angiostress-dias", "video-predict"),
        ],
    )
    def test_reference_interfaces_have_a_loadable_seed(self, path: Path, interface: str):
        task = load_task(path)
        assert task.port is None
        assert task.interface.id == interface

    def test_interface_is_required(self, tmp_path):
        dest = _copy_task(tmp_path)
        interface_block = "\n".join(
            [
                "[interface]",
                'id = "gym-policy"',
                'interaction_mode = "closed-loop"',
                'protocol_version = "1"',
                'observations = ["gym-obs"]',
                'actions = ["insertion_twist"]',
                "",
                "",
            ]
        )
        _patch_toml(dest / "task.toml", interface_block, "")
        with pytest.raises(TaskContractError, match="interface or legacy port"):
            load_task(dest)

    def test_legacy_port_normalizes_to_interface(self, tmp_path):
        dest = _copy_task(tmp_path)
        interface_block = "\n".join(
            [
                "[interface]",
                'id = "gym-policy"',
                'interaction_mode = "closed-loop"',
                'protocol_version = "1"',
                'observations = ["gym-obs"]',
                'actions = ["insertion_twist"]',
                "",
            ]
        )
        port_block = "\n".join(
            [
                "[port]",
                'id = "gym-policy"',
                'observation = "gym-obs"',
                'action = "insertion_twist"',
                "",
            ]
        )
        harness_block = "\n".join(
            [
                "[harness]",
                'interaction_mode = "closed-loop"',
                'protocol_version = "1"',
                "max_steps = 90",
                "",
                "",
            ]
        )
        _patch_toml(dest / "task.toml", interface_block, port_block)
        _patch_toml(dest / "task.toml", harness_block, "")

        task = load_task(dest)
        assert task.port is not None
        assert task.port.id.value == "gym-policy"
        assert task.interface.id == "gym-policy"

    def test_procedure_name_is_just_a_tag(self, tmp_path):
        dest = _copy_task(tmp_path)
        _patch_toml(
            dest / "task.toml",
            'tags = ["lumen", "navigation", "safety"]',
            'tags = ["cabg", "next-step"]',
        )
        task = load_task(dest)
        assert "cabg" in task.metadata.tags
        assert task.port is None
        assert task.interface.id == "gym-policy"

    def test_task_toml_path_also_loads(self):
        assert load_task(EXAMPLE_TASK / "task.toml").id == "lumen-nav-safe"

    def test_missing_task_is_rejected(self, tmp_path):
        with pytest.raises(TaskContractError, match="missing"):
            load_task(tmp_path / "task.toml")

    def test_unpinned_task_is_valid_but_not_runnable(self, tmp_path):
        dest = _copy_task(tmp_path)
        pin = load_task(dest).environment.world_pin
        _patch_toml(dest / "task.toml", f'world_pin = "{pin}"', 'world_pin = ""')
        with pytest.raises(TaskContractError, match="world_pin"):
            load_task(dest).assert_runnable()

    def test_pinned_task_is_runnable(self):
        load_task(EXAMPLE_TASK).assert_runnable()

    def test_describe_names_the_headline_and_refuses_human_det(self):
        text = load_task(EXAMPLE_TASK).describe()
        assert "headline   safe_success" in text
        assert "port       gym-policy" in text
        assert "human det. refused" in text
        assert "3c6bb39" in text


class TestContractRefusals:
    def test_raw_success_cannot_headline_when_safe_success_exists(self, tmp_path):
        dest = _copy_task(tmp_path)
        _patch_toml(dest / "task.toml", 'headline = "safe_success"', 'headline = "raw_success"')
        with pytest.raises(TaskContractError, match="CathSim"):
            load_task(dest)

    def test_safety_critical_task_without_gates_is_rejected(self, tmp_path):
        dest = _copy_task(tmp_path)
        _patch_toml(
            dest / "task.toml",
            '[[verifier.gates]]\nid = "wall_penetration"\n'
            'inputs = { unsafe = "info.unsafe", max_pen = "info.max_pen", '
            'safety_max_pen = "safety_max_pen", diverged = "info.diverged" }\n'
            "input_defaults = { diverged = false }\n"
            'fail_when = "unsafe or max_pen > safety_max_pen or diverged"\n'
            'maps_to = "unsafe"\nrealization = "scalar-dsl"\n',
            "",
        )
        with pytest.raises(TaskContractError, match="safety_critical"):
            load_task(dest)

    def test_human_determination_is_refused(self, tmp_path):
        dest = _copy_task(tmp_path)
        _patch_toml(
            dest / "task.toml",
            "emit_human_determination = false",
            "emit_human_determination = true",
        )
        with pytest.raises(TaskContractError, match="human determination"):
            load_task(dest)

    def test_human_subject_is_refused(self, tmp_path):
        dest = _copy_task(tmp_path)
        _patch_toml(dest / "task.toml", 'kind = "policy"', 'kind = "human"')
        with pytest.raises(TaskContractError, match=r"subject\.kind=human"):
            load_task(dest)

    def test_procedural_phi_cannot_attest(self, tmp_path):
        dest = _copy_task(tmp_path)
        _patch_toml(dest / "task.toml", 'level = "none"', 'level = "attested"')
        with pytest.raises(TaskContractError, match="attestation"):
            load_task(dest)

    def test_prohibited_phi_cannot_load(self, tmp_path):
        dest = _copy_task(tmp_path)
        _patch_toml(dest / "task.toml", 'class = "procedural"', 'class = "prohibited"')
        with pytest.raises(TaskContractError, match="prohibited"):
            load_task(dest)

    def test_lumen_gym_requires_gym_id(self, tmp_path):
        dest = _copy_task(tmp_path)
        _patch_toml(dest / "task.toml", 'gym_id = "Lumen/NavTreeBranch-v0"', 'gym_id = ""')
        with pytest.raises(TaskContractError, match="gym_id"):
            load_task(dest)

    def test_missing_instruction_is_rejected(self, tmp_path):
        dest = _copy_task(tmp_path)
        (dest / "instruction.md").unlink()
        with pytest.raises(TaskContractError, match=r"instruction\.md"):
            load_task(dest)

    def test_single_turn_interface_requires_an_output_schema(self, tmp_path):
        dest = tmp_path / "video-task"
        shutil.copytree(VIDEO_TASK, dest)
        _patch_toml(dest / "task.toml", 'outputs = ["next-step"]', "outputs = []")
        with pytest.raises(TaskContractError, match="needs output"):
            load_task(dest)

    def test_gym_policy_cannot_use_a_frame_source(self, tmp_path):
        dest = _copy_task(tmp_path)
        _patch_toml(dest / "task.toml", 'kind = "lumen-gym"', 'kind = "frame-source"')
        _patch_toml(dest / "task.toml", 'kind = "physics"', 'kind = "contract"')
        with pytest.raises(TaskContractError, match="gym-policy"):
            load_task(dest)


class TestBind:
    def test_lumen_linear_binds_to_lumen(self):
        assert_bind(load_task(EXAMPLE_TASK), load_agent(CATH_AGENT))

    def test_video_predictor_binds_to_video_task(self):
        assert_bind(load_task(VIDEO_TASK), load_agent(VIDEO_AGENT))

    def test_video_model_does_not_bind_to_gym_task(self):
        with pytest.raises(TaskContractError, match="video-predict"):
            assert_bind(load_task(EXAMPLE_TASK), load_agent(VIDEO_AGENT))

    def test_wrong_kind_does_not_bind(self, tmp_path):
        dest = tmp_path / "agent"
        dest.mkdir()
        (dest / "weights.bin").write_bytes(b"weights")
        (dest / "agent.py").write_text("def load_predictor(**kwargs): pass\n", encoding="utf-8")
        weights_pin = hashlib.sha256(b"weights").hexdigest()
        (dest / "agent.toml").write_text(
            "\n".join(
                [
                    'format_version = "1"',
                    'id = "acme/vlm"',
                    'agent_version = "0"',
                    'port = "gym-policy"',
                    'kind = "vlm"',
                    f'weights_pin = "{weights_pin}"',
                    'weights_path = "weights.bin"',
                    'entrypoint = "agent.py:load_predictor"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        with pytest.raises(TaskContractError, match="kind="):
            assert_bind(load_task(EXAMPLE_TASK), load_agent(dest))

    def test_random_agent_cannot_pin_weights(self, tmp_path):
        dest = tmp_path / "agent"
        dest.mkdir()
        (dest / "agent.toml").write_text(
            "\n".join(
                [
                    'format_version = "1"',
                    'id = "seldingermed/random"',
                    'agent_version = "0"',
                    'port = "gym-policy"',
                    'kind = "random"',
                    'weights_pin = "abc"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        with pytest.raises(TaskContractError, match="random"):
            load_agent(dest)

    def test_agent_id_and_capability_are_canonical(self):
        agent = load_agent(CATH_AGENT)
        assert agent.id == "seldingermed/lumen-linear"
        assert agent.port is None
        assert agent.capabilities[0].interface == "gym-policy"

    def test_agent_without_slash_is_rejected(self, tmp_path):
        dest = tmp_path / "agent"
        dest.mkdir()
        (dest / "agent.toml").write_text(
            "\n".join(
                [
                    'format_version = "1"',
                    'id = "cathmodel"',
                    'agent_version = "0"',
                    'port = "gym-policy"',
                    'kind = "policy"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        with pytest.raises(TaskContractError, match="id"):
            load_agent(dest)


class TestDataset:
    def test_example_dataset_loads_the_task(self):
        dataset = load_dataset(EXAMPLE_DATASET)
        assert dataset.id == "seldingermed/lumen-nav"
        assert dataset.headline == "safe_success"
        assert len(dataset.tasks) == 1
        assert dataset.tasks[0].id == "lumen-nav-safe"

    def test_video_predict_dataset_loads(self):
        dataset = load_dataset(VIDEO_DATASET)
        assert dataset.id == "seldingermed/video-nextstep"
        assert dataset.headline == "next_step_correct"
        assert dataset.tasks[0].port is None
        assert dataset.tasks[0].interface.id == "video-predict"

    def test_headline_mismatch_is_rejected(self, tmp_path):
        dest = tmp_path / "ds"
        dest.mkdir()
        (dest / "dataset.toml").write_text(
            "\n".join(
                [
                    'format_version = "1"',
                    'id = "example/bad-headline"',
                    'dataset_version = "0"',
                    'headline = "max_pen"',
                    'phi_class = "procedural"',
                    "tasks = [",
                    f'    "{EXAMPLE_TASK.as_posix()}",',
                    "]",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        with pytest.raises(TaskContractError, match="headlines"):
            load_dataset(dest)


class TestTrialVector:
    def _vector(self, info: dict[str, object]) -> TrialVector:
        return score_context(
            task=load_task(EXAMPLE_TASK),
            task_dir=EXAMPLE_TASK,
            agent_identity="random@0",
            seed=0,
            context={"kind": "gym-policy", "info": info, "safety_max_pen": 0.3},
        )

    def test_safe_success_is_the_headline(self):
        vector = self._vector(
            {"success": True, "safe_success": True, "unsafe": False, "max_pen": 0.0}
        )
        assert vector.headline.id == "safe_success"
        assert vector.headline.value is True
        assert not vector.any_gate_failed

    def test_raw_reach_with_wall_injury_fails_the_gate(self):
        vector = self._vector(
            {"success": True, "safe_success": False, "unsafe": True, "max_pen": 0.9}
        )
        raw = vector.metric("raw_success")
        assert raw is not None
        assert raw.value is True
        assert vector.headline.value is False
        assert vector.any_gate_failed

    def test_diverged_fails_the_gate(self):
        vector = self._vector({"success": False, "diverged": True, "max_pen": 0.0})
        assert vector.any_gate_failed
        diverged = vector.metric("diverged")
        assert diverged is not None
        assert diverged.value is True

    def test_float_raises(self):
        vector = self._vector(
            {"success": True, "safe_success": True, "unsafe": False, "max_pen": 0.0}
        )
        with pytest.raises(ScoreContractError, match="no scalar"):
            float(vector)

    def test_int_raises(self):
        vector = self._vector(
            {"success": True, "safe_success": True, "unsafe": False, "max_pen": 0.0}
        )
        with pytest.raises(ScoreContractError, match="no scalar"):
            int(vector)

    def test_bool_raises(self):
        vector = self._vector(
            {"success": True, "safe_success": True, "unsafe": False, "max_pen": 0.0}
        )
        with pytest.raises(ScoreContractError, match="no truth value"):
            bool(vector)

    def test_gated_reach_is_zero_when_the_wall_is_injured(self):
        vector = self._vector(
            {"success": True, "safe_success": False, "unsafe": True, "max_pen": 0.9}
        )
        spec = ProjectionSpec(id="gated_reach_v0", version="0")
        assert project(vector, spec) == 0.0

    def test_gated_reach_is_one_only_on_clean_raw_success(self):
        vector = self._vector(
            {"success": True, "safe_success": True, "unsafe": False, "max_pen": 0.01}
        )
        spec = ProjectionSpec(id="gated_reach_v0", version="0")
        assert project(vector, spec) == 1.0

    def test_unassessable_gate_cannot_be_projected(self):
        vector = TrialVector(
            task_id="x",
            task_version="0",
            agent_identity="a",
            seed=0,
            gates=(
                {"id": "wall_penetration", "status": GateStatus.NOT_ASSESSABLE, "reason": "fog"},
            ),
            metrics=(
                {"id": "raw_success", "value": True},
                {"id": "safe_success", "value": None, "headline": True},
                {"id": "diverged", "value": False},
            ),
        )
        spec = ProjectionSpec(id="gated_reach_v0", version="0")
        with pytest.raises(ScoreContractError, match="unassessable"):
            project(vector, spec)


class TestCli:
    def test_tasks_validate_example(self, capsys):
        assert main(["tasks", "validate", str(EXAMPLE_TASK)]) == 0
        out = capsys.readouterr().out
        assert "valid: lumen-nav-safe@1" in out
        assert " runnable" in out

    def test_tasks_describe_example(self, capsys):
        assert main(["tasks", "describe", str(EXAMPLE_TASK)]) == 0
        out = capsys.readouterr().out
        assert "Task lumen-nav-safe@1" in out
        assert "Remain inside the lumen" in out

    def test_datasets_validate_example(self, capsys):
        assert main(["datasets", "validate", str(EXAMPLE_DATASET)]) == 0
        assert "seldingermed/lumen-nav@0" in capsys.readouterr().out

    def test_tasks_validate_rejects_a_broken_task(self, capsys, tmp_path):
        dest = _copy_task(tmp_path)
        _patch_toml(dest / "task.toml", 'headline = "safe_success"', 'headline = "raw_success"')
        assert main(["tasks", "validate", str(dest)]) == 1
        assert "INVALID" in capsys.readouterr().err

    def test_agents_validate_example(self, capsys):
        assert main(["agents", "validate", str(CATH_AGENT)]) == 0
        assert "seldingermed/lumen-linear" in capsys.readouterr().out

    def test_bind_matching_ports(self, capsys):
        assert main(["bind", str(EXAMPLE_TASK), str(CATH_AGENT)]) == 0
        out = capsys.readouterr().out
        assert "seldingermed/lumen-linear" in out
        assert "gym-policy" in out

    def test_bind_refuses_a_video_model_on_a_gym_task(self, capsys):
        assert main(["bind", str(EXAMPLE_TASK), str(VIDEO_AGENT)]) == 1
        assert "INCOMPATIBLE" in capsys.readouterr().err

    def test_bind_video_model_to_video_task(self, capsys):
        assert main(["bind", str(VIDEO_TASK), str(VIDEO_AGENT)]) == 0
        assert "video-predict" in capsys.readouterr().out


def test_taskspec_is_exported() -> None:
    assert TaskSpec.__name__ == "TaskSpec"
