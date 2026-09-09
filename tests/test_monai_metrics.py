"""MONAI delegation: established imaging metrics inside task verifiers."""

from __future__ import annotations

import numpy as np
import pytest

from or_audit.metrics.monai import dice_score, hausdorff_distance, mean_iou

pytest.importorskip("monai")


def _mask(offset: int = 0) -> np.ndarray:
    base = np.zeros((1, 2, 8, 8), dtype=np.float32)
    base[0, 1, 2 + offset : 6 + offset, 2:6] = 1.0
    base[0, 0] = 1.0 - base[0, 1]
    return base


def test_perfect_overlap_scores_one() -> None:
    mask = _mask()
    assert dice_score(mask, mask) == pytest.approx(1.0)
    assert mean_iou(mask, mask) == pytest.approx(1.0)
    assert hausdorff_distance(mask, mask) == pytest.approx(0.0)


def test_shifted_mask_degrades_monotonically() -> None:
    assert dice_score(_mask(0), _mask(0)) > dice_score(_mask(0), _mask(2))
    assert hausdorff_distance(_mask(0), _mask(2)) > 0.0


def test_rank_rejection() -> None:
    with pytest.raises(ValueError, match=r"\(B, C, \*spatial\)"):
        dice_score(np.zeros((8, 8), dtype=np.float32), np.zeros((8, 8), dtype=np.float32))


def test_empty_foreground_is_nan_for_caller_to_mark() -> None:
    empty = np.zeros((1, 2, 8, 8), dtype=np.float32)
    empty[0, 0] = 1.0
    assert np.isnan(dice_score(empty, empty, include_background=False))
