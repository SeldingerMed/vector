"""Statistical uncertainty intervals and clustered resampling (Phase D4/D5).

Provides Wilson score intervals for binary rates (preserving the invariant
that zero observed failures is not zero risk) and seeded clustered bootstrap
intervals over declared independent units.
"""

from __future__ import annotations

import math
import random

from or_audit.errors import TaskContractError

DEFAULT_CONFIDENCE = 0.95
BOOTSTRAP_DRAWS = 1000


def normal_z(confidence: float) -> float:
    """Two-sided normal critical value for a given confidence level in (0, 1)."""
    if not 0.0 < confidence < 1.0:
        raise TaskContractError(f"confidence {confidence!r} must be in (0, 1)")
    alpha = 1.0 - confidence
    p = 1.0 - alpha / 2.0
    if p >= 1.0:
        return 8.0
    q = 2.0 * p - 1.0
    a = 0.147
    term1 = 2.0 / (math.pi * a) + math.log(1.0 - q * q) / 2.0
    inner = term1 * term1 - math.log(1.0 - q * q) / a
    sign = 1.0 if q >= 0.0 else -1.0
    erf_inv = sign * math.sqrt(math.sqrt(max(0.0, inner)) - term1)
    return float(math.sqrt(2.0) * erf_inv)


def wilson_score_interval(
    successes: int,
    total: int,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Honest on small sample sizes: when successes=0, the upper bound remains
    strictly positive (e.g. n=3 yields ~56% upper bound at 95% confidence).
    """
    if total < 0 or successes < 0:
        raise TaskContractError("successes and total must be non-negative")
    if successes > total:
        raise TaskContractError(f"successes ({successes}) cannot exceed total ({total})")
    if total == 0:
        return (0.0, 1.0)

    z = normal_z(confidence)
    z2 = z * z
    n = float(total)
    p = float(successes) / n

    center = (p + z2 / (2.0 * n)) / (1.0 + z2 / n)
    margin = (z / (1.0 + z2 / n)) * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)

    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return round(low, 4), round(high, 4)


def bootstrap_mean_ci(
    values: list[float] | tuple[float, ...],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = 0,
) -> tuple[float, float]:
    """Seeded percentile bootstrap confidence interval for continuous sample mean."""
    if not values:
        raise TaskContractError("bootstrap needs at least one numeric value")
    if not 0.0 < confidence < 1.0:
        raise TaskContractError(f"confidence {confidence!r} must be in (0, 1)")
    if draws < 1:
        raise TaskContractError(f"draws {draws!r} must be >= 1")

    n = len(values)
    if n == 1:
        val = round(float(values[0]), 4)
        return val, val

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(draws):
        sample = rng.choices(values, k=n)
        means.append(sum(sample) / n)
    means.sort()

    lower_idx = int((1.0 - confidence) / 2.0 * draws)
    upper_idx = int((1.0 - (1.0 - confidence) / 2.0) * draws)
    upper_idx = min(upper_idx, draws - 1)
    return round(means[lower_idx], 4), round(means[upper_idx], 4)


def clustered_bootstrap_mean_ci(
    clusters: dict[str, list[float]],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = 0,
) -> tuple[float, float]:
    """Clustered percentile bootstrap resampling by independent cluster key (e.g. patient)."""
    if not clusters:
        raise TaskContractError("clustered bootstrap needs at least one cluster")
    cluster_keys = sorted(clusters.keys())
    k = len(cluster_keys)
    if k == 1:
        single_vals = clusters[cluster_keys[0]]
        return bootstrap_mean_ci(single_vals, confidence=confidence, draws=draws, seed=seed)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(draws):
        sampled_keys = rng.choices(cluster_keys, k=k)
        sampled_values: list[float] = []
        for key in sampled_keys:
            sampled_values.extend(clusters[key])
        if sampled_values:
            means.append(sum(sampled_values) / len(sampled_values))
    if not means:
        raise TaskContractError("all sampled bootstrap clusters were empty")
    means.sort()

    lower_idx = int((1.0 - confidence) / 2.0 * draws)
    upper_idx = int((1.0 - (1.0 - confidence) / 2.0) * draws)
    upper_idx = min(upper_idx, draws - 1)
    return round(means[lower_idx], 4), round(means[upper_idx], 4)
