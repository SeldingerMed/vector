"""Read-only LeRobotDataset v3 episode access for evaluation (Phase E3).

A v3 dataset stores many episodes per Parquet/MP4 shard; episode boundaries
come from metadata, not filenames. This module reads the tabular side only:
per-episode rows (states, actions, timestamps) for replay and scoring.
Video shards are never downloaded or decoded here — records carry the shard
reference so a verifier can resolve frames when a media adapter exists.

pyarrow is an optional dependency and is imported lazily; without it every
entry point raises a clear error naming the install.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ImportError("reading LeRobotDataset v3 needs pyarrow: pip install pyarrow") from exc
    return parquet


@dataclass(frozen=True)
class EpisodeRef:
    """One episode's identity and location inside a v3 dataset."""

    episode_index: int
    dataset_length: int
    tasks: tuple[str, ...] = ()


@dataclass(frozen=True)
class EpisodeRow:
    """One low-dimensional frame: states, action, timestamp, provenance."""

    index: int
    timestamp: float | None
    state: list[float] | None
    action: list[float] | None
    source: dict[str, Any]


def _read_parquet_dir(
    parquet: Any, directory: Path, *, columns: list[str] | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in sorted(directory.glob("*.parquet")):
        table = parquet.read_table(shard, columns=columns)
        rows.extend(table.to_pylist())
    return rows


def read_episode_index(root: Path | str) -> tuple[dict[str, Any], list[EpisodeRef]]:
    """Dataset metadata plus one ref per episode described in ``meta/episodes``.

    Requires an ``episode_index`` column; ``dataset_length`` defaults to 0
    when the layout omits it (length is then the row count on materialize).
    """
    from or_audit.errors import TaskContractError

    parquet = _require_pyarrow()
    base = Path(root)
    meta_dir = base / "meta" / "episodes"
    if not meta_dir.is_dir():
        raise TaskContractError(f"not a LeRobotDataset v3 directory: {base} (no meta/episodes)")
    info_path = base / "meta" / "info.json"
    info: dict[str, Any] = {}
    if info_path.is_file():
        import json

        raw = json.loads(info_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            info = raw
    episodes: list[EpisodeRef] = []
    for row in _read_parquet_dir(parquet, meta_dir):
        if "episode_index" not in row:
            raise TaskContractError("meta/episodes row has no episode_index column")
        tasks = row.get("tasks", row.get("task", []))
        if isinstance(tasks, str):
            tasks = [tasks]
        episodes.append(
            EpisodeRef(
                episode_index=int(row["episode_index"]),
                dataset_length=int(row.get("dataset_length", 0) or 0),
                tasks=tuple(str(task) for task in (tasks or [])),
            )
        )
    episodes.sort(key=lambda ref: ref.episode_index)
    return info, episodes


def materialize_episode(
    root: Path | str,
    episode_index: int,
    *,
    state_key: str = "observation.state",
    action_key: str = "action",
    timestamp_key: str = "timestamp",
) -> list[EpisodeRow]:
    """Low-dimensional rows for one episode, in file order.

    Selects data-shard rows whose ``episode_index`` column matches. Video
    columns are never requested; each row records its shard and offset so
    evidence stays traceable without embedding media.
    """
    from or_audit.errors import TaskContractError

    parquet = _require_pyarrow()
    base = Path(root)
    data_dir = base / "data"
    if not data_dir.is_dir():
        raise TaskContractError(f"not a LeRobotDataset v3 directory: {base} (no data/)")
    rows: list[EpisodeRow] = []
    for shard in sorted(data_dir.glob("*.parquet")):
        names = set(parquet.read_schema(shard).names)
        if "episode_index" not in names:
            raise TaskContractError(f"data shard {shard.name} has no 'episode_index' column")
        for required in (state_key, action_key):
            if required not in names:
                raise TaskContractError(f"data shard {shard.name} has no {required!r} column")
        wanted = ["episode_index", state_key, action_key]
        if timestamp_key in names:
            wanted.append(timestamp_key)
        table = parquet.read_table(shard, columns=wanted)
        for position, record in enumerate(table.to_pylist()):
            if int(record.get("episode_index", -1)) != episode_index:
                continue
            timestamp = record.get(timestamp_key)
            state = record.get(state_key)
            action = record.get(action_key)
            rows.append(
                EpisodeRow(
                    index=len(rows),
                    timestamp=float(timestamp) if timestamp is not None else None,
                    state=[float(v) for v in state] if state is not None else None,
                    action=[float(v) for v in action] if action is not None else None,
                    source={
                        "dataset": str(base),
                        "shard": shard.name,
                        "shard_offset": position,
                        "episode_index": episode_index,
                    },
                )
            )
    return rows
