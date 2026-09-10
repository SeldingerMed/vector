"""Unit tests for Phase D1 split manifests and independent-case counting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from or_audit.errors import TaskContractError
from or_audit.eval.split import SplitCaseEntry, SplitManifest, load_split_manifest


def test_split_manifest_patient_overlap_refused() -> None:
    entries = (
        SplitCaseEntry(
            case_id="case-001",
            episode_id="ep-001",
            split="train",
            patient_id="patient-A",
            item_ids=("frame-001", "frame-002"),
        ),
        SplitCaseEntry(
            case_id="case-002",
            episode_id="ep-002",
            split="test",
            patient_id="patient-A",  # Leaks into test split!
            item_ids=("frame-003",),
        ),
    )
    with pytest.raises(TaskContractError, match=r"patient overlap across splits.*patient-A"):
        SplitManifest(
            dataset_id="test-dataset",
            dataset_revision="1.0.0",
            entries=entries,
            disjoint_by=("patient",),
        )


def test_split_manifest_site_overlap_refused() -> None:
    entries = (
        SplitCaseEntry(
            case_id="case-001",
            episode_id="ep-001",
            split="train",
            site_id="center-1",
            item_ids=("f-1",),
        ),
        SplitCaseEntry(
            case_id="case-002",
            episode_id="ep-002",
            split="test",
            site_id="center-1",  # Overlaps into test!
            item_ids=("f-2",),
        ),
    )
    with pytest.raises(TaskContractError, match=r"site overlap across splits.*center-1"):
        SplitManifest(
            dataset_id="test-dataset",
            dataset_revision="1.0.0",
            entries=entries,
            disjoint_by=("site",),
        )


def test_split_manifest_case_crossing_splits_refused() -> None:
    entries = (
        SplitCaseEntry(
            case_id="case-001",
            episode_id="ep-001",
            split="train",
            item_ids=("f-1",),
        ),
        SplitCaseEntry(
            case_id="case-001",
            episode_id="ep-002",
            split="test",
            item_ids=("f-2",),
        ),
    )
    with pytest.raises(TaskContractError, match="case 'case-001' crosses multiple splits"):
        SplitManifest(
            dataset_id="test-dataset",
            dataset_revision="1.0.0",
            entries=entries,
        )


def test_splitting_adjacent_frames_does_not_increase_independent_count() -> None:
    # 5 frames from the same case in test split
    entries = (
        SplitCaseEntry(
            case_id="case-001",
            episode_id="ep-001",
            split="test",
            patient_id="patient-1",
            item_ids=("frame-1", "frame-2", "frame-3", "frame-4", "frame-5"),
        ),
    )
    manifest = SplitManifest(
        dataset_id="endo-frames",
        dataset_revision="1",
        entries=entries,
    )
    # 5 item frames, but independent count is strictly 1
    assert len(manifest.items_for_split("test")) == 5
    assert manifest.independent_case_count("test", unit="case") == 1
    assert manifest.independent_case_count("test", unit="patient") == 1


def test_multi_case_independent_counts() -> None:
    entries = (
        SplitCaseEntry(
            case_id="case-001",
            episode_id="ep-001",
            split="test",
            patient_id="pt-1",
            item_ids=("f-1",),
        ),
        SplitCaseEntry(
            case_id="case-002",
            episode_id="ep-002",
            split="test",
            patient_id="pt-1",  # Same patient, distinct case
            item_ids=("f-2",),
        ),
        SplitCaseEntry(
            case_id="case-003",
            episode_id="ep-003",
            split="test",
            patient_id="pt-2",
            item_ids=("f-3",),
        ),
    )
    manifest = SplitManifest(
        dataset_id="test-multi",
        dataset_revision="1",
        entries=entries,
    )
    assert manifest.independent_case_count("test", unit="case") == 3
    assert manifest.independent_case_count("test", unit="patient") == 2


def test_validate_against_input_items_matches() -> None:
    entries = (
        SplitCaseEntry(
            case_id="c1",
            episode_id="e1",
            split="train",
            item_ids=("item-1", "item-2"),
        ),
        SplitCaseEntry(
            case_id="c2",
            episode_id="e2",
            split="test",
            item_ids=("item-3",),
        ),
    )
    manifest = SplitManifest(
        dataset_id="items-data",
        dataset_revision="v1",
        entries=entries,
    )
    # Exact match passes
    manifest.validate_against_input_items(["item-1", "item-2", "item-3"])

    # Extra input item missing from manifest
    with pytest.raises(TaskContractError, match="missing items from inputs"):
        manifest.validate_against_input_items(["item-1", "item-2", "item-3", "item-4"])

    # Missing input item present in manifest
    with pytest.raises(TaskContractError, match="contains items not in inputs"):
        manifest.validate_against_input_items(["item-1", "item-2"], strict=True)


def test_unsupported_patient_disjoint_refused_on_count() -> None:
    entries = (
        SplitCaseEntry(
            case_id="c1",
            episode_id="e1",
            split="test",
            patient_id=None,
            item_ids=("item-1",),
        ),
    )
    manifest = SplitManifest(
        dataset_id="deidentified-data",
        dataset_revision="v1",
        entries=entries,
    )
    assert not manifest.supports_patient_disjoint
    with pytest.raises(TaskContractError, match="cannot count independent patients"):
        manifest.independent_case_count("test", unit="patient")


def test_load_split_manifest_from_file(tmp_path: Path) -> None:
    manifest_data = {
        "format_version": "1",
        "dataset_id": "test-dataset",
        "dataset_revision": "rev-1",
        "disjoint_by": ["case"],
        "entries": [
            {
                "case_id": "case-01",
                "episode_id": "ep-01",
                "split": "train",
                "item_ids": ["item-1"],
            },
            {
                "case_id": "case-02",
                "episode_id": "ep-02",
                "split": "test",
                "item_ids": ["item-2"],
            },
        ],
    }
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(manifest_data), encoding="utf-8")

    loaded = load_split_manifest(path)
    assert loaded.dataset_id == "test-dataset"
    assert loaded.dataset_revision == "rev-1"
    assert loaded.independent_case_count("train") == 1
    assert loaded.independent_case_count("test") == 1


def test_assert_disjoint_stage_manifests() -> None:
    from or_audit.eval.split import assert_disjoint_stage_manifests

    m_qual = SplitManifest(
        dataset_id="ds-1",
        dataset_revision="v1",
        disjoint_by=("patient",),
        entries=(
            SplitCaseEntry(
                case_id="c1",
                episode_id="e1",
                split="qualification",
                patient_id="p1",
                item_ids=("i1",),
            ),
        ),
    )
    m_val = SplitManifest(
        dataset_id="ds-1",
        dataset_revision="v1",
        disjoint_by=("patient",),
        entries=(
            SplitCaseEntry(
                case_id="c2",
                episode_id="e2",
                split="validation",
                patient_id="p1",  # Overlap with qual!
                item_ids=("i2",),
            ),
        ),
    )
    # Overlap refused
    with pytest.raises(TaskContractError, match="patient 'p1' appears across multiple stages"):
        assert_disjoint_stage_manifests(
            {"qualification": m_qual, "validation": m_val},
            require_patient_disjoint=True,
        )

    # Dataset revision mismatch refused
    m_val_rev2 = SplitManifest(
        dataset_id="ds-1",
        dataset_revision="v2",
        disjoint_by=("patient",),
        entries=(
            SplitCaseEntry(
                case_id="c2",
                episode_id="e2",
                split="validation",
                patient_id="p2",
                item_ids=("i2",),
            ),
        ),
    )
    with pytest.raises(TaskContractError, match="differing from stage 'qualification'"):
        assert_disjoint_stage_manifests(
            {"qualification": m_qual, "validation": m_val_rev2},
        )


def test_runner_validates_split_manifest_and_stamps_head(tmp_path: Path) -> None:
    import shutil

    from or_audit.eval.loader import load_agent, load_task
    from or_audit.eval.runner import run_job

    root = Path(__file__).resolve().parents[1]
    task_src = root / "docs/examples/tasks/video-nextstep"
    agent_src = root / "docs/examples/agents/example-video-predictor"

    task_dir = tmp_path / "task"
    shutil.copytree(task_src, task_dir)

    # 3 clips in inputs.json: clip-001, clip-002, clip-003
    # Group clip-001 and clip-002 into case-1, clip-003 into case-2
    splits_data = {
        "format_version": "1",
        "dataset_id": "video-nextstep-dataset",
        "dataset_revision": "1.0",
        "disjoint_by": ["case"],
        "entries": [
            {
                "case_id": "case-1",
                "episode_id": "ep-1",
                "split": "test",
                "item_ids": ["clip-001", "clip-002"],
            },
            {
                "case_id": "case-2",
                "episode_id": "ep-2",
                "split": "test",
                "item_ids": ["clip-003"],
            },
        ],
    }
    (task_dir / "splits.json").write_text(json.dumps(splits_data), encoding="utf-8")

    task = load_task(task_dir)
    agent = load_agent(agent_src)
    out = tmp_path / "job"

    result = run_job(
        task=task,
        task_dir=task_dir,
        agent=agent,
        agent_dir=agent_src,
        out=out,
        n=3,
    )
    # 3 trials, but only 2 independent cases
    assert result.n == 3
    assert result.independent_cases == 2
    assert len(result.split_manifest_digest) == 64
    assert result.head


def test_runner_refuses_input_items_missing_from_split_manifest(tmp_path: Path) -> None:
    import shutil

    from or_audit.eval.loader import load_agent, load_task
    from or_audit.eval.runner import run_job

    root = Path(__file__).resolve().parents[1]
    task_src = root / "docs/examples/tasks/video-nextstep"
    agent_src = root / "docs/examples/agents/example-video-predictor"

    task_dir = tmp_path / "task"
    shutil.copytree(task_src, task_dir)

    # Omit clip-003 from split manifest
    splits_data = {
        "format_version": "1",
        "dataset_id": "video-nextstep-dataset",
        "dataset_revision": "1.0",
        "disjoint_by": ["case"],
        "entries": [
            {
                "case_id": "case-1",
                "episode_id": "ep-1",
                "split": "test",
                "item_ids": ["clip-001", "clip-002"],
            },
        ],
    }
    (task_dir / "splits.json").write_text(json.dumps(splits_data), encoding="utf-8")

    task = load_task(task_dir)
    agent = load_agent(agent_src)
    out = tmp_path / "job"

    with pytest.raises(TaskContractError, match=r"missing items from inputs.*clip-003"):
        run_job(
            task=task,
            task_dir=task_dir,
            agent=agent,
            agent_dir=agent_src,
            out=out,
            n=3,
        )


def test_cartesian_refuses_missing_split_selection(tmp_path: Path) -> None:
    from or_audit.eval.cartesian import run_cartesian_job
    from or_audit.eval.job_config import resolve_job

    root = Path(__file__).resolve().parents[1]
    task_src = root / "docs/examples/tasks/video-nextstep"
    agent_src = root / "docs/examples/agents/example-video-predictor"

    job = tmp_path / "job-stage-missing-split"
    job.mkdir()
    body = f"""format_version = "1"
id = "video-stage-test"
n = 3
tasks = [{json.dumps(str(task_src))}]
agents = [{json.dumps(str(agent_src))}]
[stage]
name = "qualification"
evaluation_unit = "scored clip"
target_units = 3
independent_case_unit = "held-out clip"
independent_case_key = "id"
independent_cases = 3
scenarios = ["video-nextstep"]
operator_contexts = ["offline"]
stop_conditions = ["stop on any hard gate failure"]
prerequisites = ["integration-smoke", "pilot"]
"""
    (job / "job.toml").write_text(body, encoding="utf-8")
    with pytest.raises(
        TaskContractError, match=r"requests split 'qualification' but manifest.*only defines splits"
    ):
        run_cartesian_job(resolve_job(job), out=tmp_path / "out")


def test_cartesian_refuses_unsupported_case_unit(tmp_path: Path) -> None:
    from or_audit.eval.cartesian import run_cartesian_job
    from or_audit.eval.job_config import resolve_job

    root = Path(__file__).resolve().parents[1]
    task_src = root / "docs/examples/tasks/video-nextstep"
    agent_src = root / "docs/examples/agents/example-video-predictor"

    job = tmp_path / "job-stage-bad-unit"
    job.mkdir()
    body = f"""format_version = "1"
id = "video-stage-test"
n = 3
tasks = [{json.dumps(str(task_src))}]
agents = [{json.dumps(str(agent_src))}]
[stage]
name = "qualification"
split = "test"
evaluation_unit = "scored clip"
target_units = 3
independent_case_unit = "unsupported-free-prose"
independent_case_key = "id"
independent_cases = 3
scenarios = ["video-nextstep"]
operator_contexts = ["offline"]
stop_conditions = ["stop on any hard gate failure"]
prerequisites = ["integration-smoke", "pilot"]
"""
    (job / "job.toml").write_text(body, encoding="utf-8")
    with pytest.raises(
        TaskContractError,
        match=r"unsupported independent_case_unit 'unsupported-free-prose'",
    ):
        run_cartesian_job(resolve_job(job), out=tmp_path / "out")


def test_cartesian_refuses_patient_unit_without_patient_support(tmp_path: Path) -> None:
    import shutil

    from or_audit.eval.cartesian import run_cartesian_job
    from or_audit.eval.job_config import resolve_job

    root = Path(__file__).resolve().parents[1]
    task_src = root / "docs/examples/tasks/video-nextstep"
    agent_src = root / "docs/examples/agents/example-video-predictor"

    task_dir = tmp_path / "task-no-patient"
    shutil.copytree(task_src, task_dir)

    # Write manifest without patient_id
    splits_data = {
        "format_version": "1",
        "dataset_id": "video-nextstep-dataset",
        "dataset_revision": "1.0",
        "disjoint_by": ["case"],
        "entries": [
            {
                "case_id": "case-1",
                "episode_id": "ep-1",
                "split": "test",
                "item_ids": ["clip-001", "clip-002", "clip-003"],
            },
        ],
    }
    (task_dir / "splits.json").write_text(json.dumps(splits_data), encoding="utf-8")

    job = tmp_path / "job-stage-patient"
    job.mkdir()
    body = f"""format_version = "1"
id = "video-stage-test"
n = 3
tasks = [{json.dumps(str(task_dir))}]
agents = [{json.dumps(str(agent_src))}]
[stage]
name = "qualification"
split = "test"
evaluation_unit = "scored clip"
target_units = 3
independent_case_unit = "patient"
independent_case_key = "id"
independent_cases = 1
scenarios = ["video-nextstep"]
operator_contexts = ["offline"]
stop_conditions = ["stop on any hard gate failure"]
prerequisites = ["integration-smoke", "pilot"]
"""
    (job / "job.toml").write_text(body, encoding="utf-8")
    with pytest.raises(TaskContractError, match=r"does not support patient-disjoint"):
        run_cartesian_job(resolve_job(job), out=tmp_path / "out")


def test_runner_independent_cases_counts_evaluated_subset_only(tmp_path: Path) -> None:
    import shutil

    from or_audit.eval.loader import load_agent, load_task
    from or_audit.eval.runner import run_job

    root = Path(__file__).resolve().parents[1]
    task_src = root / "docs/examples/tasks/video-nextstep"
    agent_src = root / "docs/examples/agents/example-video-predictor"

    task_dir = tmp_path / "task-subset"
    shutil.copytree(task_src, task_dir)

    # Manifest has 3 entries in test split (one per clip)
    # Running with n=2 must report independent_cases=2, NOT 3!
    task = load_task(task_dir)
    agent = load_agent(agent_src)
    out = tmp_path / "job-subset"

    result = run_job(
        task=task,
        task_dir=task_dir,
        agent=agent,
        agent_dir=agent_src,
        out=out,
        n=2,
        split="test",
    )
    assert result.n == 2
    assert result.independent_cases == 2
    assert len(result.trials) == 2


def test_cartesian_coalesces_cases_with_independent_case_groups(tmp_path: Path) -> None:
    import shutil

    from or_audit.eval.cartesian import run_cartesian_job
    from or_audit.eval.job_config import resolve_job

    root = Path(__file__).resolve().parents[1]
    task_src = root / "docs/examples/tasks/video-nextstep"
    agent_src = root / "docs/examples/agents/example-video-predictor"

    task_a = tmp_path / "task-a"
    task_b = tmp_path / "task-b"
    shutil.copytree(task_src, task_a)
    shutil.copytree(task_src, task_b)

    # Give task-b a different task id
    task_b_toml = task_b / "task.toml"
    old_text = task_b_toml.read_text(encoding="utf-8")
    new_text = old_text.replace('id = "video-nextstep"', 'id = "video-nextstep-b"')
    task_b_toml.write_text(new_text, encoding="utf-8")
    job = tmp_path / "job-groups"
    job.mkdir()
    # Both tasks share the same patient cases, grouped under "shared-group"
    # n=1 on each task (1 trial each, evaluating case-001 on both)
    # With grouping, they coalesce to 1 independent case total!
    body = f"""format_version = "1"
id = "grouped-stage"
n = 1
tasks = [{json.dumps(str(task_a))}, {json.dumps(str(task_b))}]
agents = [{json.dumps(str(agent_src))}]
[stage]
name = "qualification"
split = "test"
evaluation_unit = "scored clip"
target_units = 2
independent_case_unit = "held-out clip"
independent_case_key = "id"
independent_cases = 1
independent_case_groups = {{"video-nextstep" = "shared-group", "video-nextstep-b" = "shared-group"}}
scenarios = ["video-nextstep", "video-nextstep-b"]
operator_contexts = ["offline"]
stop_conditions = ["stop on any hard gate failure"]
prerequisites = ["integration-smoke", "pilot"]
"""
    (job / "job.toml").write_text(body, encoding="utf-8")
    manifest = run_cartesian_job(resolve_job(job), out=tmp_path / "out-groups")
    assert manifest.stage is not None
    assert manifest.stage.independent_cases == 1


def test_runner_filters_interleaved_splits_without_leakage(tmp_path: Path) -> None:
    import shutil

    from or_audit.eval.loader import load_agent, load_task
    from or_audit.eval.runner import run_job

    root = Path(__file__).resolve().parents[1]
    task_src = root / "docs/examples/tasks/video-nextstep"
    agent_src = root / "docs/examples/agents/example-video-predictor"

    task_dir = tmp_path / "task-interleaved"
    shutil.copytree(task_src, task_dir)

    # Write interleaved inputs: train-1, test-1, train-2, test-2
    inputs_data = {
        "items": [
            {"id": "clip-train-1", "media": "public://train/1"},
            {"id": "clip-001", "media": "public://test/1"},
            {"id": "clip-train-2", "media": "public://train/2"},
            {"id": "clip-002", "media": "public://test/2"},
        ]
    }
    (task_dir / "inputs.json").write_text(json.dumps(inputs_data), encoding="utf-8")
    labels_data = {
        "items": [
            {"id": "clip-train-1", "next_step": "advance", "outcome": "continue", "unsafe": False},
            {"id": "clip-001", "next_step": "advance", "outcome": "continue", "unsafe": False},
            {"id": "clip-train-2", "next_step": "advance", "outcome": "continue", "unsafe": False},
            {"id": "clip-002", "next_step": "advance", "outcome": "continue", "unsafe": False},
        ]
    }
    (task_dir / "labels.json").write_text(json.dumps(labels_data), encoding="utf-8")

    splits_data = {
        "format_version": "1",
        "dataset_id": "interleaved-dataset",
        "dataset_revision": "1.0",
        "disjoint_by": ["case", "patient"],
        "entries": [
            {
                "case_id": "c-tr-1",
                "episode_id": "e-tr-1",
                "patient_id": "p-tr-1",
                "split": "train",
                "item_ids": ["clip-train-1"],
            },
            {
                "case_id": "c-tr-2",
                "episode_id": "e-tr-2",
                "patient_id": "p-tr-2",
                "split": "train",
                "item_ids": ["clip-train-2"],
            },
            {
                "case_id": "c-te-1",
                "episode_id": "e-te-1",
                "patient_id": "p-te-1",
                "split": "test",
                "item_ids": ["clip-001"],
            },
            {
                "case_id": "c-te-2",
                "episode_id": "e-te-2",
                "patient_id": "p-te-2",
                "split": "test",
                "item_ids": ["clip-002"],
            },
        ],
    }
    (task_dir / "splits.json").write_text(json.dumps(splits_data), encoding="utf-8")

    task = load_task(task_dir)
    agent = load_agent(agent_src)
    out = tmp_path / "job-interleaved"

    # Evaluate split="test" with n=2
    result = run_job(
        task=task,
        task_dir=task_dir,
        agent=agent,
        agent_dir=agent_src,
        out=out,
        n=2,
        split="test",
    )
    assert result.n == 2
    assert result.split == "test"
    executed_ids = [trial.trajectory[0].get("input", {}).get("id") for trial in result.trials]
    assert executed_ids == ["clip-001", "clip-002"]


def test_cross_split_resume_refused(tmp_path: Path) -> None:
    import shutil

    from or_audit.eval.loader import load_agent, load_task
    from or_audit.eval.runner import run_job

    root = Path(__file__).resolve().parents[1]
    task_src = root / "docs/examples/tasks/video-nextstep"
    agent_src = root / "docs/examples/agents/example-video-predictor"

    task_dir = tmp_path / "task-cross-resume"
    shutil.copytree(task_src, task_dir)

    # Write splits with both train and test
    splits_data = {
        "format_version": "1",
        "dataset_id": "cross-resume-dataset",
        "dataset_revision": "1.0",
        "disjoint_by": ["case"],
        "entries": [
            {"case_id": "c1", "episode_id": "e1", "split": "train", "item_ids": ["clip-001"]},
            {
                "case_id": "c2",
                "episode_id": "e2",
                "split": "test",
                "item_ids": ["clip-002", "clip-003"],
            },
        ],
    }
    (task_dir / "splits.json").write_text(json.dumps(splits_data), encoding="utf-8")

    task = load_task(task_dir)
    agent = load_agent(agent_src)
    out = tmp_path / "job-cross-resume"

    # Run 1 trial on split="test"
    run_job(
        task=task,
        task_dir=task_dir,
        agent=agent,
        agent_dir=agent_src,
        out=out,
        n=1,
        split="test",
    )

    # Attempt to resume with split="train" -> must be refused!
    with pytest.raises(TaskContractError, match="cross-split resume is refused"):
        run_job(
            task=task,
            task_dir=task_dir,
            agent=agent,
            agent_dir=agent_src,
            out=out,
            n=1,
            split="train",
            resume=True,
        )

    # Simulate killed run (unlink result.json) and attempt resume with split="train"
    (out / "result.json").unlink()
    with pytest.raises(TaskContractError, match="cross-split resume is refused"):
        run_job(
            task=task,
            task_dir=task_dir,
            agent=agent,
            agent_dir=agent_src,
            out=out,
            n=1,
            split="train",
            resume=True,
        )


def test_replay_job_preserves_split(tmp_path: Path) -> None:
    from or_audit.eval.loader import load_agent, load_task
    from or_audit.eval.runner import replay_job, run_job

    root = Path(__file__).resolve().parents[1]
    task_src = root / "docs/examples/tasks/video-nextstep"
    agent_src = root / "docs/examples/agents/example-video-predictor"

    out = tmp_path / "job-replay-split"
    result = run_job(
        task=load_task(task_src),
        task_dir=task_src,
        agent=load_agent(agent_src),
        agent_dir=agent_src,
        out=out,
        n=2,
        split="test",
    )
    assert result.split == "test"
    replayed = replay_job(out, load_task=load_task, load_agent=load_agent)
    assert replayed.split == "test"
    assert replayed.head == result.head


def test_cartesian_independent_cases_matches_input_subset(tmp_path: Path) -> None:
    import shutil

    from or_audit.eval.cartesian import run_cartesian_job
    from or_audit.eval.job_config import resolve_job

    root = Path(__file__).resolve().parents[1]
    task_src = root / "docs/examples/tasks/video-nextstep"
    agent_src = root / "docs/examples/agents/example-video-predictor"

    task_dir = tmp_path / "task-input-subset"
    shutil.copytree(task_src, task_dir)

    # Manifest defines 4 test items across 4 cases
    splits_data = {
        "format_version": "1",
        "dataset_id": "subset-test",
        "dataset_revision": "1.0",
        "disjoint_by": ["case"],
        "entries": [
            {"case_id": "c1", "episode_id": "e1", "split": "test", "item_ids": ["clip-001"]},
            {"case_id": "c2", "episode_id": "e2", "split": "test", "item_ids": ["clip-002"]},
            {"case_id": "c3", "episode_id": "e3", "split": "test", "item_ids": ["clip-003"]},
            {"case_id": "c4", "episode_id": "e4", "split": "test", "item_ids": ["clip-004"]},
        ],
    }
    (task_dir / "splits.json").write_text(json.dumps(splits_data), encoding="utf-8")

    # But inputs.json only has clip-001 and clip-002 (2 items)
    inputs_data = {
        "items": [
            {"id": "clip-001", "media": "public://clip-1"},
            {"id": "clip-002", "media": "public://clip-2"},
        ]
    }
    (task_dir / "inputs.json").write_text(json.dumps(inputs_data), encoding="utf-8")
    labels_data = {
        "items": [
            {"id": "clip-001", "next_step": "advance", "outcome": "continue", "unsafe": False},
            {"id": "clip-002", "next_step": "advance", "outcome": "continue", "unsafe": False},
        ]
    }
    (task_dir / "labels.json").write_text(json.dumps(labels_data), encoding="utf-8")

    job = tmp_path / "job-cartesian-subset"
    job.mkdir()
    body = f"""format_version = "1"
id = "subset-stage"
n = 2
tasks = [{json.dumps(str(task_dir))}]
agents = [{json.dumps(str(agent_src))}]
[stage]
name = "qualification"
split = "test"
evaluation_unit = "scored clip"
target_units = 2
independent_case_unit = "case"
independent_case_key = "id"
independent_cases = 2
scenarios = ["video-nextstep"]
operator_contexts = ["offline"]
stop_conditions = ["stop on any hard gate failure"]
prerequisites = ["integration-smoke", "pilot"]
"""
    (job / "job.toml").write_text(body, encoding="utf-8")
    manifest = run_cartesian_job(resolve_job(job), out=tmp_path / "out-subset")
    assert manifest.stage is not None
    assert manifest.stage.independent_cases == 2
    from or_audit.eval.job import read_job_result

    assert len(manifest.pairs) == 1
    pair_result = read_job_result(tmp_path / "out-subset" / manifest.pairs[0].dir)
    assert pair_result.split == "test"
    pair_ids = [trial.trajectory[0].get("input", {}).get("id") for trial in pair_result.trials]
    assert pair_ids == ["clip-001", "clip-002"]


def test_explicit_split_without_splits_path_refused(tmp_path: Path) -> None:
    import shutil

    from or_audit.eval.loader import load_agent, load_task
    from or_audit.eval.runner import run_job

    root = Path(__file__).resolve().parents[1]
    task_src = root / "docs/examples/tasks/video-nextstep"
    agent_src = root / "docs/examples/agents/example-video-predictor"

    task_dir = tmp_path / "task-no-split-decl"
    shutil.copytree(task_src, task_dir)

    # Remove splits_path from task.toml
    task_toml = task_dir / "task.toml"
    old_text = task_toml.read_text(encoding="utf-8")
    task_toml.write_text(old_text.replace('splits_path = "splits.json"', ""), encoding="utf-8")

    task = load_task(task_dir)
    agent = load_agent(agent_src)
    out = tmp_path / "out"

    with pytest.raises(
        TaskContractError,
        match="has no declared splits_path; cannot execute explicit split",
    ):
        run_job(
            task=task,
            task_dir=task_dir,
            agent=agent,
            agent_dir=agent_src,
            out=out,
            n=1,
            split="test",
        )


def test_independent_case_unit_counts_patient_cases_when_declared(tmp_path: Path) -> None:
    import shutil

    from or_audit.eval.loader import load_agent, load_task
    from or_audit.eval.runner import run_job

    root = Path(__file__).resolve().parents[1]
    task_src = root / "docs/examples/tasks/video-nextstep"
    agent_src = root / "docs/examples/agents/example-video-predictor"
    task_dir = tmp_path / "task-patient-split"
    shutil.copytree(task_src, task_dir)

    splits_file = task_dir / "splits.json"
    splits_data = json.loads(splits_file.read_text(encoding="utf-8"))
    splits_data["entries"][0]["patient_id"] = "pt-shared"
    splits_data["entries"][1]["patient_id"] = "pt-shared"
    splits_file.write_text(json.dumps(splits_data), encoding="utf-8")

    task = load_task(task_dir)
    agent = load_agent(agent_src)

    # 1. When independent_case_unit="patient" -> counts distinct patients (1)
    res_patient = run_job(
        task=task,
        task_dir=task_dir,
        agent=agent,
        agent_dir=agent_src,
        out=tmp_path / "out-patient",
        n=2,
        split="test",
        independent_case_unit="patient",
    )
    assert res_patient.independent_case_unit == "patient"
    assert res_patient.independent_cases == 1

    # 2. When default (unclustered / case unit) -> counts distinct cases (2)
    res_case = run_job(
        task=task,
        task_dir=task_dir,
        agent=agent,
        agent_dir=agent_src,
        out=tmp_path / "out-case",
        n=2,
        split="test",
    )
    assert res_case.independent_case_unit == ""
    assert res_case.independent_cases == 2
