"""Static, vector-preserving leaderboard generation."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from or_audit.errors import TaskContractError
from or_audit.eval.integrity import tree_digest
from or_audit.eval.job import (
    JobResult,
    compute_head,
    read_job_config,
    read_job_result,
    resolve_bundle_path,
)
from or_audit.eval.loader import load_task
from or_audit.eval.provenance import assert_public_leaderboard_eligible
from or_audit.eval.reconstitute import assert_trajectory_matches_vector
from or_audit.eval.scorecard import scorecard_data
from or_audit.eval.task import TaskSpec


def _result_paths(paths: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for path in paths:
        if (path / "result.json").is_file() and (path / "config.json").is_file():
            found.add(path)
        elif path.is_dir():
            found.update(
                candidate.parent
                for candidate in path.rglob("result.json")
                if (candidate.parent / "config.json").is_file()
            )
    if not found:
        raise TaskContractError("no job result.json files found")
    return sorted(found)


def _metric_summary(result: JobResult) -> dict[str, dict[str, Any]]:
    return {
        metric["id"]: {key: value for key, value in metric.items() if key != "id"}
        for metric in scorecard_data(result)["metrics"]
    }


def _metric_value(metric: Mapping[str, Any]) -> float | None:
    if metric["kind"] == "boolean":
        return metric.get("rate")
    if metric["kind"] == "continuous":
        return metric.get("mean")
    return None


def _metric_display(metric: Mapping[str, Any]) -> str:
    value = _metric_value(metric)
    if value is not None:
        unit = f" {metric['unit']}" if metric.get("unit") else ""
        return f"{value:.4g}{unit}"
    if metric["kind"] == "categorical":
        return ", ".join(f"{category}: {count}" for category, count in metric["counts"].items())
    return "—"


def _verified_result(job_dir: Path) -> tuple[JobResult, TaskSpec]:
    config = read_job_config(job_dir)
    result = read_job_result(job_dir)
    if compute_head(result) != result.head:
        raise TaskContractError(f"result head mismatch in {job_dir}")
    if config.get("task_digest") != result.task_digest:
        raise TaskContractError(f"task digest mismatch in {job_dir}")
    if config.get("agent_digest") != result.agent_digest:
        raise TaskContractError(f"agent digest mismatch in {job_dir}")
    task_dir = resolve_bundle_path(job_dir, config["task_dir"], label="task")
    if tree_digest(task_dir) != result.task_digest:
        raise TaskContractError(f"bundled task digest mismatch in {job_dir}")
    agent_dir_raw = config.get("agent_dir")
    if (
        agent_dir_raw
        and tree_digest(resolve_bundle_path(job_dir, agent_dir_raw, label="agent"))
        != result.agent_digest
    ):
        raise TaskContractError(f"bundled agent digest mismatch in {job_dir}")
    task = load_task(task_dir)
    assert_trajectory_matches_vector(
        job_dir,
        task=task,
        task_dir=task_dir,
        result=result,
        config=config,
    )
    # A public leaderboard row is an ingestion, so the quarantine is enforced at
    # the one choke point every row passes through, and by refusal: a silently
    # dropped adaptation would be an unstated absence.
    assert_public_leaderboard_eligible(task_dir)
    # The task travels back with the result: a row's gate *set* is a property of
    # the task that ran, not of the outcome, and downstream surfaces cannot tell
    # "no gate failed" from "nothing was gated" without it.
    return result, task


def leaderboard_data(paths: list[Path]) -> dict[str, Any]:
    """Load verified jobs and return deterministic task-scoped rows."""
    rows: list[dict[str, Any]] = []
    for job_dir in _result_paths(paths):
        result, task = _verified_result(job_dir)
        metrics = _metric_summary(result)
        headline = metrics[result.headline]
        assessed = result.headline_true + result.headline_false
        rows.append(
            {
                "task_id": result.task_id,
                "task_version": result.task_version,
                "agent_identity": result.agent_identity,
                "world_pin": result.world_pin,
                "n": result.n,
                "headline": result.headline,
                "headline_kind": headline["kind"],
                "headline_direction": headline["direction"],
                "headline_value": _metric_value(headline),
                "headline_true": result.headline_true,
                "headline_false": result.headline_false,
                "headline_unassessable": result.headline_unassessable,
                "headline_rate": (
                    result.headline_true / assessed
                    if headline["kind"] == "boolean" and assessed
                    else None
                ),
                "any_gate_failed": result.any_gate_failed,
                # The hard gates the task declares, so a consumer can tell a
                # passing gate set from an empty one. Sorted for determinism.
                "gate_specs": [
                    {"gate_id": gate.id, "unit": gate.unit}
                    for gate in sorted(task.verifier.gates, key=lambda gate: gate.id)
                ],
                # Head-covered engine provenance, or ``None`` when the bundle
                # recorded none. Absent is *unattested*, never "real".
                "world_engine": (
                    result.world_engine.model_dump(mode="json")
                    if result.world_engine is not None
                    else None
                ),
                "metrics": metrics,
                "head": result.head,
            }
        )

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        value = row["headline_value"]
        ordered = value is not None and row["headline_direction"] != "neutral"
        direction_value = -value if ordered and row["headline_direction"] == "maximize" else value
        return (
            row["task_id"],
            row["any_gate_failed"],
            not ordered,
            direction_value if direction_value is not None else 0.0,
            row["agent_identity"],
        )

    rows.sort(key=sort_key)
    return {"format_version": "2", "rows": rows}


def render_html(data: dict[str, Any]) -> str:
    """Render a dependency-free static table without collapsing the vector."""
    body: list[str] = []
    for row in data["rows"]:
        headline = _metric_display(row["metrics"][row["headline"]])
        metrics = ", ".join(
            f"{key}={_metric_display(value)}" for key, value in row["metrics"].items()
        )
        cells = (
            row["task_id"],
            row["agent_identity"],
            row["world_pin"] or "—",
            str(row["n"]),
            f"{row['headline']} {headline}",
            str(row["headline_unassessable"]),
            str(row["any_gate_failed"]),
            metrics,
            row["head"],
        )
        body.append(
            "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells) + "</tr>"
        )
    return f"""<!doctype html>
<meta charset="utf-8">
<title>OR-Audit safety-vector leaderboard</title>
<style>
body{{font:15px system-ui,sans-serif;margin:2rem;color:#17202a}}
h1{{margin-bottom:.25rem}}
p{{color:#52606d}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #d8dee4;padding:.55rem;text-align:left;vertical-align:top}}
th{{background:#f4f6f8}}
td:last-child{{font:12px ui-monospace,monospace;overflow-wrap:anywhere}}
tr:nth-child(even){{background:#fafbfc}}
</style>
<h1>OR-Audit safety-vector leaderboard</h1>
<p>Ranked within each task by fewer hard-gate failures, then by the declared direction
of an ordered headline. Categorical headlines remain unranked. The complete metric vector,
pins, and artifact heads remain visible; there is no cross-task overall score.</p>
<table><thead><tr>
<th>Task</th><th>Agent</th><th>World pin</th><th>n</th><th>Headline</th>
<th>Unassessable</th><th>Gate failures</th><th>Metrics</th><th>Artifact head</th>
</tr></thead><tbody>{"".join(body)}</tbody></table>
"""


def write_leaderboard(paths: list[Path], out: Path) -> dict[str, Any]:
    """Write deterministic ``leaderboard.json`` and ``index.html``."""
    data = leaderboard_data(paths)
    out.mkdir(parents=True, exist_ok=True)
    (out / "leaderboard.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (out / "index.html").write_text(render_html(data), encoding="utf-8")
    return data
