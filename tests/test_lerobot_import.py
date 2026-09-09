"""LeRobotDataset v3 read-only access: index and episode materialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from or_audit.errors import TaskContractError
from or_audit.eval.datasets.lerobot_v3 import materialize_episode, read_episode_index

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _fixture(root: Path) -> Path:
    import json

    ds = root / "tiny-v3"
    (ds / "meta" / "episodes").mkdir(parents=True)
    (ds / "data").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "fps": 30}), encoding="utf-8"
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1],
                "dataset_length": [3, 2],
                "tasks": ["reach", "grasp"],
            }
        ),
        ds / "meta" / "episodes" / "chunk-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 0, 0, 1, 1],
                "timestamp": [0.0, 0.1, 0.2, 0.0, 0.1],
                "observation.state": [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [1.0, 1.1], [1.2, 1.3]],
                "action": [[0.0], [0.1], [0.2], [0.5], [0.6]],
                "video_blob": [b"x" * 1000] * 5,
            }
        ),
        ds / "data" / "chunk-000.parquet",
    )
    return ds


def test_fat_columns_never_load(tmp_path: Path) -> None:
    rows = materialize_episode(_fixture(tmp_path), 0)
    assert len(rows) == 3
    assert all("video_blob" not in row.source for row in rows)


def test_missing_episode_index_refuses(tmp_path: Path) -> None:
    import pyarrow as pa_local
    import pyarrow.parquet as pq_local

    ds = _fixture(tmp_path)
    (ds / "data" / "chunk-000.parquet").unlink()
    pq_local.write_table(
        pa_local.table({"timestamp": [0.0], "observation.state": [[0.0]], "action": [[0.0]]}),
        ds / "data" / "chunk-000.parquet",
    )
    with pytest.raises(TaskContractError, match="no 'episode_index' column"):
        materialize_episode(ds, 0)


def test_episode_index_lists_tasks_and_lengths(tmp_path: Path) -> None:
    info, episodes = read_episode_index(_fixture(tmp_path))
    assert info["fps"] == 30
    assert [(ref.episode_index, ref.dataset_length, ref.tasks) for ref in episodes] == [
        (0, 3, ("reach",)),
        (1, 2, ("grasp",)),
    ]


def test_materialize_selects_episode_rows_only(tmp_path: Path) -> None:
    rows = materialize_episode(_fixture(tmp_path), 1)
    assert [row.index for row in rows] == [0, 1]
    assert [row.state for row in rows] == [[1.0, 1.1], [1.2, 1.3]]
    assert [row.action for row in rows] == [[0.5], [0.6]]
    assert [row.timestamp for row in rows] == [0.0, 0.1]
    assert rows[0].source["episode_index"] == 1
    assert rows[0].source["shard"] == "chunk-000.parquet"


def test_missing_layout_refuses(tmp_path: Path) -> None:
    with pytest.raises(TaskContractError, match="no meta/episodes"):
        read_episode_index(tmp_path)


def test_missing_state_column_refuses(tmp_path: Path) -> None:
    import pyarrow as pa_local
    import pyarrow.parquet as pq_local

    ds = _fixture(tmp_path)
    (ds / "data" / "chunk-000.parquet").unlink()
    pq_local.write_table(
        pa_local.table({"episode_index": [0], "action": [[0.0]]}),
        ds / "data" / "chunk-000.parquet",
    )
    with pytest.raises(TaskContractError, match=r"no 'observation\.state' column"):
        materialize_episode(ds, 0)
