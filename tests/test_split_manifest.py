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
