"""Paired model comparison with uncertainty (Phase D, core).

Scorecards describe one job; this module compares two jobs run on the same
cases. Comparability is enforced, not assumed: same task, version, world
pin, and interface; unique seeds per job; identical seed sets (mismatched
cohorts are refused, never inner-joined). Unassessable values on either
side drop that pair and are counted, never zero-filled.

Seeds pair trials; they are not claimed to be independent cases. Clustered
resampling by declared independent unit is follow-up work for when trial
records carry case manifests (D1). Until then the bootstrap resamples
seeds, and small-n intervals stay honestly wide.

Uncertainty is a seeded percentile bootstrap over the paired differences.
It quantifies sampling noise of the comparison, not clinical significance,
and wide intervals on few trials are the honest answer, not a failure.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from or_audit.errors import TaskContractError
from or_audit.eval.contracts import MetricKind
from or_audit.eval.job import JobResult

#: Bootstrap draws per comparison. Fixed default so published intervals
#: reproduce exactly; callers needing tighter Monte Carlo error say so.
BOOTSTRAP_DRAWS = 2000


class PairedComparison(BaseModel):
    """Paired difference of one metric across two jobs on shared seeds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_id: str
    paired_seeds: tuple[int, ...]
    dropped_unassessable: int
    mean_diff: float
    ci_low: float
    ci_high: float
    confidence: float = 0.95
    draws: int = BOOTSTRAP_DRAWS
    bootstrap_seed: int = 0


def _metric_number(vector: Any, metric_id: str) -> float | None:
    outcome = vector.metric(metric_id)
    if outcome is None:
        raise TaskContractError(f"metric {metric_id!r} is not declared")
    value = outcome.value
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if outcome.kind is MetricKind.CATEGORICAL:
        raise TaskContractError(f"metric {metric_id!r} is categorical and has no paired difference")
    return None


def _assert_comparable(a: JobResult, b: JobResult) -> None:
    """Refuse comparisons across different tasks, worlds, or cohorts."""
    for field in ("task_id", "task_version", "world_pin", "interface_id"):
        first, second = getattr(a, field), getattr(b, field)
        if first != second:
            raise TaskContractError(
                f"paired comparison needs identical {field}: {first!r} != {second!r}"
            )
    for label, job in (("a", a), ("b", b)):
        seeds = [trial.seed for trial in job.trials]
        if len(set(seeds)) != len(seeds):
            raise TaskContractError(f"job {label} has duplicate trial seeds; cannot pair")
    only_a = sorted({t.seed for t in a.trials} - {t.seed for t in b.trials})
    only_b = sorted({t.seed for t in b.trials} - {t.seed for t in a.trials})
    if only_a or only_b:
        raise TaskContractError(
            f"paired comparison needs identical seed sets: only in a={only_a}, only in b={only_b}"
        )


def paired_differences(
    a: JobResult, b: JobResult, metric_id: str
) -> tuple[tuple[int, ...], list[float], int]:
    """Kept seeds, per-seed (b - a) differences, and dropped count."""
    _assert_comparable(a, b)
    a_by_seed = {trial.seed: trial.vector for trial in a.trials}
    b_by_seed = {trial.seed: trial.vector for trial in b.trials}
    shared = sorted(set(a_by_seed) & set(b_by_seed))
    seeds: list[int] = []
    diffs: list[float] = []
    dropped = 0
    for seed in shared:
        first = _metric_number(a_by_seed[seed], metric_id)
        second = _metric_number(b_by_seed[seed], metric_id)
        if first is None or second is None:
            dropped += 1
            continue
        seeds.append(seed)
        diffs.append(second - first)
    if not diffs:
        raise TaskContractError("paired comparison kept no assessable pairs")
    return tuple(seeds), diffs, dropped


def bootstrap_ci(
    diffs: list[float], *, confidence: float = 0.95, draws: int = BOOTSTRAP_DRAWS, seed: int = 0
) -> tuple[float, float]:
    """Seeded percentile bootstrap interval over paired differences."""
    if not diffs:
        raise TaskContractError("bootstrap needs at least one paired difference")
    if not 0.0 < confidence < 1.0:
        raise TaskContractError(f"confidence {confidence!r} must be in (0, 1)")
    if draws < 1:
        raise TaskContractError(f"draws {draws!r} must be >= 1")
    rng = np.random.default_rng(seed)
    sample = np.asarray(diffs, dtype=float)
    means = np.mean(rng.choice(sample, size=(draws, sample.size), replace=True), axis=1)
    lower = (1.0 - confidence) / 2.0 * 100.0
    return float(np.percentile(means, lower)), float(np.percentile(means, 100.0 - lower))


def compare_jobs(
    a: JobResult,
    b: JobResult,
    metric_id: str,
    *,
    confidence: float = 0.95,
    draws: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = 0,
) -> PairedComparison:
    """Paired (b - a) comparison of one metric with a bootstrap interval."""
    seeds, diffs, dropped = paired_differences(a, b, metric_id)
    low, high = bootstrap_ci(diffs, confidence=confidence, draws=draws, seed=bootstrap_seed)
    return PairedComparison(
        metric_id=metric_id,
        paired_seeds=seeds,
        dropped_unassessable=dropped,
        mean_diff=float(sum(diffs) / len(diffs)),
        ci_low=low,
        ci_high=high,
        confidence=confidence,
        draws=draws,
        bootstrap_seed=bootstrap_seed,
    )
