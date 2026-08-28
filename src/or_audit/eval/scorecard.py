"""Deterministic human-readable scorecards for vector-valued eval jobs."""

from __future__ import annotations

import html
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from or_audit.eval.job import JobResult
from or_audit.eval.sim.base import BACKEND_SYNTHETIC_STUB, BACKEND_UNKNOWN

STUB_HEADLINE = "NOT PHYSICAL EVIDENCE - SYNTHETIC STAND-IN"
METRICS_ONLY_HEADLINE = "METRICS-ONLY - NOT SAFETY-ATTESTED"


def _engine_labels(world_engine: dict[str, Any] | None) -> tuple[str, str]:
    engine = world_engine or {}
    return (
        str(engine.get("engine") or BACKEND_UNKNOWN),
        str(engine.get("backend") or BACKEND_UNKNOWN),
    )


def _metrics_only(result: JobResult, world_engine: dict[str, Any] | None) -> bool:
    """Whether this row carries the Tier-0 metrics-only label (§2.2)."""
    if result.world_engine is not None:
        return result.world_engine.metrics_only
    return bool((world_engine or {}).get("metrics_only"))


def scorecard_data(
    result: JobResult,
    *,
    world_engine: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate each gate and metric independently; never create a composite score."""
    gate_ids = [gate.id for gate in result.trials[0].vector.gates]
    metric_ids = [metric.id for metric in result.trials[0].vector.metrics]
    gates = []
    for gate_id in gate_ids:
        statuses = [trial.vector.gate(gate_id).status.value for trial in result.trials]  # type: ignore[union-attr]
        gates.append(
            {
                "id": gate_id,
                "pass": statuses.count("pass"),
                "fail": statuses.count("fail"),
                "not_assessable": statuses.count("not_assessable"),
                "not_applicable": statuses.count("not_applicable"),
            }
        )
    metrics = []
    for metric_id in metric_ids:
        outcomes = [trial.vector.metric(metric_id) for trial in result.trials]
        definition = outcomes[0]
        values = [outcome.value for outcome in outcomes if outcome is not None]
        assessed = [value for value in values if value is not None]
        row: dict[str, Any] = {
            "id": metric_id,
            "headline": metric_id == result.headline,
            "kind": definition.kind.value if definition and definition.kind else "boolean",
            "unit": definition.unit if definition else "",
            "direction": definition.direction.value if definition else "neutral",
            "assessed": len(assessed),
            "unassessable": len(values) - len(assessed),
        }
        if row["kind"] == "boolean":
            row.update(
                {
                    "true": assessed.count(True),
                    "false": assessed.count(False),
                    "rate": assessed.count(True) / len(assessed) if assessed else None,
                }
            )
        elif row["kind"] == "continuous":
            numeric = [float(value) for value in assessed]
            row.update(
                {
                    "mean": fmean(numeric) if numeric else None,
                    "min": min(numeric) if numeric else None,
                    "max": max(numeric) if numeric else None,
                }
            )
        else:
            row["counts"] = {
                category: assessed.count(category) for category in sorted(set(assessed))
            }
        metrics.append(row)
    return {
        "task_id": result.task_id,
        "task_version": result.task_version,
        "task_digest": result.task_digest,
        "agent_identity": result.agent_identity,
        "agent_digest": result.agent_digest,
        "world_pin": result.world_pin,
        "world_engine": dict(world_engine) if world_engine else None,
        "interface_id": result.interface_id,
        "interaction_mode": result.interaction_mode,
        "runtime_identity": result.runtime_identity,
        "projection_identity": result.projection_identity,
        "n": result.n,
        "headline": result.headline,
        "gates": gates,
        "metrics": metrics,
        "claim_footer": result.claim_footer,
        "metrics_only": _metrics_only(result, world_engine),
        "head": result.head,
    }


def render_markdown(
    result: JobResult,
    *,
    world_engine: dict[str, Any] | None = None,
) -> str:
    data = scorecard_data(result, world_engine=world_engine)
    engine_name, backend = _engine_labels(data["world_engine"])
    lines = [
        f"# OR-Audit scorecard: {data['task_id']}",
        "",
    ]
    if backend == BACKEND_SYNTHETIC_STUB:
        lines.extend(
            [
                f"> **{STUB_HEADLINE}.** This job ran against a synthetic",
                f"> stand-in for the `{engine_name}` world, not a physics backend. Every",
                "> observation, safety margin, gate outcome, and metric below was produced",
                "> by a placeholder and is not evidence about physical behaviour.",
                "> `export-rl` refuses this job.",
                "",
            ]
        )
    if data["metrics_only"]:
        lines.extend(
            [
                f"> **{METRICS_ONLY_HEADLINE}.** This world's instrumentation",
                "> does not report the safety state a hard gate would bind to, so this",
                "> package declares `environment.metrics_only` and ships no gates. The",
                "> metrics below describe task behaviour only; nothing here attests",
                "> safety (§2.2 Tier 0).",
                "",
            ]
        )
    lines.extend(
        [
            f"- Agent: `{data['agent_identity']}`",
            f"- Trials: `{data['n']}`",
            f"- World pin: `{data['world_pin'] or 'none'}`",
            f"- World engine: `{engine_name}` (backend `{backend}`)",
            f"- Interface: `{data['interface_id']}` (`{data['interaction_mode']}`)",
            f"- Runtime identity: `{data['runtime_identity'] or 'none'}`",
            f"- Projection identity: `{data['projection_identity'] or 'none'}`",
            f"- Task digest: `{data['task_digest']}`",
            f"- Agent digest: `{data['agent_digest']}`",
            f"- Artifact head: `{data['head']}`",
            "",
            "## Safety gates",
            "",
            "| Gate | Pass | Fail | Not assessable | Not applicable |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for gate in data["gates"]:
        lines.append(
            f"| {gate['id']} | {gate['pass']} | {gate['fail']} | "
            f"{gate['not_assessable']} | {gate['not_applicable']} |"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Metric | Headline | Result | Assessed | Unassessable |",
            "|---|:---:|---:|---:|---:|",
        ]
    )
    for metric in data["metrics"]:
        if metric["kind"] == "boolean":
            value = "n/a" if metric["rate"] is None else f"{metric['rate']:.6f}"
        elif metric["kind"] == "continuous":
            value = "n/a" if metric["mean"] is None else f"{metric['mean']:.6f}"
        else:
            value = (
                ", ".join(f"{category}: {count}" for category, count in metric["counts"].items())
                or "n/a"
            )
        lines.append(
            f"| {metric['id']} | {'yes' if metric['headline'] else 'no'} | {value} | "
            f"{metric['assessed']} | {metric['unassessable']} |"
        )
    if data["claim_footer"]:
        lines.extend(["", "## Claim boundary", "", data["claim_footer"]])
    lines.extend(
        [
            "",
            "> Safety gates and metrics are reported separately. "
            "This scorecard has no composite score.",
            "",
        ]
    )
    return "\n".join(lines)


def render_html(
    result: JobResult,
    *,
    world_engine: dict[str, Any] | None = None,
) -> str:
    data = scorecard_data(result, world_engine=world_engine)
    markdown = render_markdown(result, world_engine=world_engine)
    payload = html.escape(json.dumps(data, indent=2))
    engine_name, backend = _engine_labels(data["world_engine"])
    banners: list[str] = []
    if backend == BACKEND_SYNTHETIC_STUB:
        # The wording, not the border colour, has to carry the refusal (WCAG 2.2 AA 1.4.1).
        banners.append(
            f'<p class="stub" role="note"><strong>{STUB_HEADLINE}.</strong> This job ran '
            f"against a synthetic stand-in for the "
            f"<code>{html.escape(engine_name)}</code> world, not a physics backend. Its "
            "observations, safety margins, gates, and metrics are placeholders and are not "
            "evidence about physical behaviour. <code>export-rl</code> refuses this job.</p>"
        )
    if data["metrics_only"]:
        banners.append(
            f'<p class="stub" role="note"><strong>{METRICS_ONLY_HEADLINE}.</strong> The '
            f"<code>{html.escape(engine_name)}</code> world's instrumentation does not "
            "report the safety state a hard gate would bind to, so this package declares "
            "<code>environment.metrics_only</code> and ships no gates. The metrics below "
            "describe task behaviour only and attest nothing about safety.</p>"
        )
    banner = "".join(banners)
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>OR-Audit scorecard</title><style>body{font:15px system-ui;max-width:960px;"
        "margin:40px auto;padding:0 20px;color:#172033}pre{white-space:pre-wrap;background:#f4f6f8;"
        "padding:20px;border-radius:8px}details{margin-top:24px}"
        ".stub{border:3px solid #8c1d18;background:#fdf3f2;color:#4a0f0c;padding:16px 20px;"
        "border-radius:8px;font-size:16px}</style>"
        f"<body>{banner}<pre>{html.escape(markdown)}</pre><details>"
        "<summary>Machine-readable vector</summary>"
        f"<pre>{payload}</pre></details></body></html>\n"
    )


def write_scorecards(
    out: Path,
    result: JobResult,
    *,
    world_engine: dict[str, Any] | None = None,
) -> None:
    """Write stable Markdown, HTML, and JSON scorecard surfaces."""
    (out / "scorecard.md").write_text(
        render_markdown(result, world_engine=world_engine), encoding="utf-8"
    )
    (out / "scorecard.html").write_text(
        render_html(result, world_engine=world_engine), encoding="utf-8"
    )
    (out / "scorecard.json").write_text(
        json.dumps(scorecard_data(result, world_engine=world_engine), indent=2) + "\n",
        encoding="utf-8",
    )
