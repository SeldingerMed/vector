"""C3 label provenance: bidirectional input/label coverage and bench provenance."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from or_audit.errors import TaskContractError
from or_audit.eval.loader import load_agent, load_task
from or_audit.eval.runner import run_job

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "docs/examples/tasks"


def _ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["id"]) for item in payload["items"]}


def test_packaged_label_sets_match_inputs_bidirectionally() -> None:
    for task_dir in sorted(TASKS.iterdir()):
        labels_path = task_dir / "labels.json"
        inputs_path = task_dir / "inputs.json"
        if not labels_path.is_file() or not inputs_path.is_file():
            continue
        inputs = _ids(inputs_path)
        labels = _ids(labels_path)
        assert not labels - inputs, f"{task_dir.name} orphan labels: {sorted(labels - inputs)}"
        assert not inputs - labels, f"{task_dir.name} unlabeled: {sorted(inputs - labels)}"


def test_orphan_labels_refuse_at_run(tmp_path: Path) -> None:
    import shutil

    task_dir = tmp_path / "video-nextstep"
    shutil.copytree(
        TASKS / "video-nextstep", task_dir, ignore=shutil.ignore_patterns("__pycache__")
    )
    labels_path = task_dir / "labels.json"
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    payload["items"].append({"id": "clip-999", "next_step": "advance"})
    labels_path.write_text(json.dumps(payload), encoding="utf-8")
    agent_dir = ROOT / "docs" / "examples" / "agents" / "example-video-predictor"
    with pytest.raises(TaskContractError, match="match no inputs"):
        run_job(
            task=load_task(task_dir),
            task_dir=task_dir,
            agent=load_agent(agent_dir),
            agent_dir=agent_dir,
            out=tmp_path / "job",
            n=3,
        )


def test_bench_provenance_declares_source_and_limits() -> None:
    provenance = tomllib.loads(
        (TASKS / "angiostress-dias" / "labels-provenance.toml").read_text(encoding="utf-8")
    )
    assert provenance["source_pin"] == "d386bcc2f5bc534d18e9a866340adfc4f51e6463"
    assert "no panel" in provenance["adjudication"]
    assert _ids(TASKS / "angiostress-dias" / "inputs.json") == _ids(
        TASKS / "angiostress-dias" / "labels.json"
    )
