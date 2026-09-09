"""Null-model forecaster: predicts no penetration, ever.

The lower bound every dynamics model must beat. Real logic (a constant),
not a lookup table — but deliberately the weakest honest forecast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class NullForecaster:
    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "forecast": {
                "h1": {"max_pen": 0.0, "unsafe": False},
                "h5": {"max_pen": 0.0, "unsafe": False},
                "h20": {"max_pen": 0.0, "unsafe": False},
            }
        }


def load_predictor(*, root: Path, weights_path: Path) -> NullForecaster:
    del root, weights_path
    return NullForecaster()
