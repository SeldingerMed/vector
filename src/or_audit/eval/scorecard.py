"""Deterministic human-readable scorecards for vector-valued eval jobs."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from or_audit.eval.contracts import MetricKind
from or_audit.eval.job import JobResult, TrialRecord
from or_audit.eval.sim.base import BACKEND_SYNTHETIC_STUB, BACKEND_UNKNOWN
from or_audit.eval.uncertainty import (
    bootstrap_mean_ci,
    clustered_bootstrap_mean_ci,
    wilson_score_interval,
)

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


def _is_abstained(trial: TrialRecord) -> bool:
    if any(gate.abstained for gate in trial.vector.gates):
        return True
    ab_metric = trial.vector.metric("abstained")
    return bool(ab_metric is not None and ab_metric.value is True)


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
            true_count = assessed.count(True)
            rate = true_count / len(assessed) if assessed else None
            ci_95 = list(wilson_score_interval(true_count, len(assessed))) if assessed else None
            row.update(
                {
                    "true": true_count,
                    "false": assessed.count(False),
                    "rate": rate,
                    "ci_95": ci_95,
                    "ci_method": "wilson",
                    "confidence": 0.95,
                }
            )
        elif row["kind"] == "continuous":
            numeric = [float(value) for value in assessed]
            clusters: dict[str, list[float]] = defaultdict(list)
            for trial in result.trials:
                m = trial.vector.metric(metric_id)
                if m is not None and m.value is not None:
                    cluster_key = trial.patient_id or trial.case_id or f"seed-{trial.seed}"
                    clusters[cluster_key].append(float(m.value))
            ci_method = "bootstrap"
            if len(clusters) >= 2:
                ci_95 = list(clustered_bootstrap_mean_ci(clusters))
                ci_method = "clustered_bootstrap"
            else:
                ci_95 = list(bootstrap_mean_ci(numeric)) if numeric else None
            row.update(
                {
                    "mean": fmean(numeric) if numeric else None,
                    "min": min(numeric) if numeric else None,
                    "max": max(numeric) if numeric else None,
                    "ci_95": ci_95,
                    "ci_method": ci_method,
                    "confidence": 0.95,
                    "draws": 1000,
                }
            )
        else:
            row["counts"] = {
                category: assessed.count(category) for category in sorted(set(assessed))
            }
        metrics.append(row)
    headline_outcome = next((m for m in result.trials[0].vector.metrics if m.headline), None)
    headline_kind = headline_outcome.kind if headline_outcome else MetricKind.BOOLEAN
    headline_direction = (
        headline_outcome.direction.value
        if headline_outcome and hasattr(headline_outcome.direction, "value")
        else (str(headline_outcome.direction) if headline_outcome else "maximize")
    )
    # Coverage & Risk analysis for abstaining models (Phase D5)
    abstained_count = sum(1 for trial in result.trials if _is_abstained(trial))
    coverage = (result.n - abstained_count) / result.n if result.n > 0 else 0.0
    covered_trials = [t for t in result.trials if not _is_abstained(t)]
    risk_at_coverage: float | None = None
    if covered_trials:
        if headline_kind is MetricKind.BOOLEAN:
            failed_count = sum(
                1
                for t in covered_trials
                if t.vector.any_gate_failed or t.vector.headline.value is False
            )
        else:
            failed_count = sum(1 for t in covered_trials if t.vector.any_gate_failed)
        risk_at_coverage = round(failed_count / len(covered_trials), 4)
    coverage_report = {
        "abstained": abstained_count,
        "coverage": round(coverage, 4),
        "risk_at_coverage": risk_at_coverage,
    }

    # Subgroups & Worst-Case Scenario Analysis (Phase D5)
    # Subgroups come strictly from explicit cohort fields (trial.subgroups).
    # patient_id, site_id, and case_id are reserved SOLELY for clustering.
    cohort_keys: set[str] = set()
    for trial in result.trials:
        cohort_keys.update(trial.subgroups.keys())
    subgroups: list[dict[str, Any]] = []
    worst_case = None
    for axis in sorted(cohort_keys):
        axis_groups: dict[str, list[TrialRecord]] = defaultdict(list)
        for trial in result.trials:
            val = trial.subgroups.get(axis)
            if val:
                axis_groups[val].append(trial)

        if len(axis_groups) > 1:
            for val, s_trials in sorted(axis_groups.items()):
                s_count = len(s_trials)
                s_rate: float | None = None
                ci: list[float] | None = None
                if headline_kind is MetricKind.BOOLEAN:
                    assessed_trials = [
                        t
                        for t in s_trials
                        if t.vector.headline.value is not None and not _is_abstained(t)
                    ]
                    s_assessed = len(assessed_trials)
                    s_unassessable = s_count - s_assessed
                    if s_assessed > 0:
                        s_pass = sum(
                            1
                            for t in assessed_trials
                            if not t.vector.any_gate_failed and t.vector.headline.value is True
                        )
                        s_rate = round(s_pass / s_assessed, 4)
                        ci = list(wilson_score_interval(s_pass, s_assessed))
                    else:
                        s_pass = 0
                elif headline_kind is MetricKind.CONTINUOUS:
                    s_vals = [
                        float(t.vector.headline.value)
                        for t in s_trials
                        if t.vector.headline.value is not None and not _is_abstained(t)
                    ]
                    s_assessed = len(s_vals)
                    s_unassessable = s_count - s_assessed
                    if s_assessed > 0:
                        s_rate = round(fmean(s_vals), 4)
                        ci = list(bootstrap_mean_ci(s_vals))
                        s_pass = sum(1 for t in s_trials if not t.vector.any_gate_failed)
                    else:
                        s_pass = 0
                else:
                    assessed_trials = [
                        t
                        for t in s_trials
                        if t.vector.headline.value is not None and not _is_abstained(t)
                    ]
                    s_assessed = len(assessed_trials)
                    s_unassessable = s_count - s_assessed
                    if s_assessed > 0:
                        s_pass = sum(1 for t in assessed_trials if not t.vector.any_gate_failed)
                        s_rate = round(s_pass / s_assessed, 4)
                        ci = list(wilson_score_interval(s_pass, s_assessed))
                    else:
                        s_pass = 0

                entry = {
                    "axis": axis,
                    "subgroup": val,
                    "count": s_count,
                    "assessed": s_assessed,
                    "unassessable": s_unassessable,
                    "pass": s_pass,
                    "rate": s_rate,
                    "ci_95": ci,
                    "is_underpowered": s_assessed < 10,
                }
                subgroups.append(entry)

    worst_cases: dict[str, dict[str, Any]] = {}
    if headline_direction in ("maximize", "minimize"):
        for axis in sorted(cohort_keys):
            axis_entries = [sg for sg in subgroups if sg["axis"] == axis and sg["rate"] is not None]
            if axis_entries:
                if headline_direction == "minimize":
                    worst_cases[axis] = max(
                        axis_entries,
                        key=lambda sg: (float(str(sg["rate"])), -int(str(sg["assessed"]))),
                    )
                else:
                    worst_cases[axis] = min(
                        axis_entries,
                        key=lambda sg: (float(str(sg["rate"])), int(str(sg["assessed"]))),
                    )

    worst_case = next(iter(worst_cases.values())) if len(worst_cases) == 1 else None
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
        "independent_cases": result.independent_cases,
        "split_manifest_digest": result.split_manifest_digest,
        "worst_cases": worst_cases,
        "worst_case": worst_case,
        "coverage": coverage_report,
        "subgroups": subgroups,
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
        ]
    )
    if data.get("independent_cases") is not None:
        lines.append(f"- Independent cases: `{data['independent_cases']}`")
    if data.get("split_manifest_digest"):
        lines.append(f"- Split manifest digest: `{data['split_manifest_digest']}`")
    lines.extend(
        [
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
            "| Metric | Headline | Result | 95% CI | Assessed | Unassessable |",
            "|---|:---:|---:|:---:|---:|---:|",
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
        ci_str = (
            f"[{metric['ci_95'][0]:.4f}, {metric['ci_95'][1]:.4f}]"
            if metric.get("ci_95")
            else "n/a"
        )
        lines.append(
            f"| {metric['id']} | {'yes' if metric['headline'] else 'no'} | {value} | "
            f"{ci_str} | {metric['assessed']} | {metric['unassessable']} |"
        )
    if data["coverage"]["abstained"] > 0:
        cov = data["coverage"]
        lines.extend(
            [
                "",
                "## Risk vs coverage",
                "",
                (
                    f"- Model coverage: `{cov['coverage'] * 100:.1f}%` "
                    f"({data['n'] - cov['abstained']}/{data['n']} non-abstained)"
                ),
                (
                    f"- Risk at coverage: `{cov['risk_at_coverage'] * 100:.1f}%`"
                    if cov["risk_at_coverage"] is not None
                    else "- Risk at coverage: n/a"
                ),
            ]
        )
    if data["subgroups"]:
        lines.extend(
            [
                "",
                "## Subgroups and worst-case analysis",
                "",
                "| Cohort axis | Subgroup | Count | Assessed | Unassessable | "
                "Pass rate | 95% CI | Power |",
                "|---|---|---:|---:|---:|---:|:---:|:---:|",
            ]
        )
        for sg in data["subgroups"]:
            ci_str = f"[{sg['ci_95'][0]:.4f}, {sg['ci_95'][1]:.4f}]" if sg["ci_95"] else "n/a"
            pwr = "underpowered (<10)" if sg["is_underpowered"] else "adequate"
            rate_str = "n/a" if sg["rate"] is None else f"{sg['rate']:.4f}"
            lines.append(
                f"| {sg['axis']} | {sg['subgroup']} | {sg['count']} | "
                f"{sg['assessed']} | {sg['unassessable']} | "
                f"{rate_str} | {ci_str} | {pwr} |"
            )
        if data.get("worst_case") and data["worst_case"]["rate"] is not None:
            wc = data["worst_case"]
            lines.append(
                f"\n> **Worst-case subgroup:** `{wc['axis']}={wc['subgroup']}` "
                f"with pass rate `{wc['rate']:.4f}`."
            )
        elif data.get("worst_cases"):
            for axis, wc in sorted(data["worst_cases"].items()):
                lines.append(
                    f"\n> **Worst-case subgroup ({axis}):** `{wc['subgroup']}` "
                    f"with pass rate `{wc['rate']:.4f}`."
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
