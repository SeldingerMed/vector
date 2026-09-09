"""MONAI metric delegation for task verifiers (Phase E6).

MONAI is an optional dependency (``pip install monai``) and is imported
lazily, so the core harness never requires it. Task verifiers call these
helpers instead of reimplementing Dice/Hausdorff/IoU; TrialVector semantics
(no scalar headline without gates, unassessable stays ``None``) still apply
at the caller — these helpers return plain floats and never decide that.

Inputs are array-likes shaped ``(B, C, *spatial)`` (MONAI's cumulative
convention); outputs are Python floats. Empty foreground yields NaN, which
the caller maps to unassessable per kernel rules.
"""

import warnings
from typing import Any


def _require_monai() -> Any:
    try:
        import monai
    except ImportError as exc:
        raise ImportError(
            "task verifiers that delegate to MONAI metrics need it installed: pip install monai"
        ) from exc
    return monai


def _tensors(pred: Any, target: Any) -> tuple[Any, Any]:
    _require_monai()
    import numpy as np
    import torch

    def convert(value: Any) -> Any:
        array = np.asarray(value)
        if array.ndim < 3:
            raise ValueError(f"expected (B, C, *spatial), got shape {array.shape}")
        return torch.as_tensor(array)

    return convert(pred), convert(target)


def _evaluate(metric: Any, prediction: Any, truth: Any) -> float:
    """Run one cumulative metric, silencing MONAI's own deprecation noise
    (repo treats warnings as errors; this is upstream chatter, not signal)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="monai.*")
        value = metric(prediction, truth)
    metric.reset()
    return float(value.mean().item())


def dice_score(pred: Any, target: Any, *, include_background: bool = True) -> float:
    """Cumulative Dice over a batch (MONAI ``DiceMetric``)."""
    monai = _require_monai()
    metric = monai.metrics.DiceMetric(include_background=include_background, reduction="mean")
    return _evaluate(metric, *_tensors(pred, target))


def mean_iou(pred: Any, target: Any, *, include_background: bool = True) -> float:
    """Cumulative mean IoU over a batch (MONAI ``MeanIoU``)."""
    monai = _require_monai()
    metric = monai.metrics.MeanIoU(include_background=include_background, reduction="mean")
    prediction, truth = _tensors(pred, target)
    return _evaluate(metric, prediction, truth)


def hausdorff_distance(pred: Any, target: Any, *, percentile: float | None = None) -> float:
    """Hausdorff (or percentile variant) distance in voxel units."""
    monai = _require_monai()
    metric = monai.metrics.HausdorffDistanceMetric(include_background=False, percentile=percentile)
    prediction, truth = _tensors(pred, target)
    return _evaluate(metric, prediction, truth)
