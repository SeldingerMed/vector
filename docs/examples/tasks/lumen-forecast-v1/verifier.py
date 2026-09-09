"""Forecast verifier: action-conditioned penetration prediction error.

Oracle limits (F2): futures are recorded under the actions actually taken.
Unobserved alternative action sequences have no ground truth here — a model
is scored on forecasting the executed future, never on counterfactual
branches it did not observe. Branching evaluation needs a simulator, not
this fixture.
"""

from __future__ import annotations

import math
from typing import Any

HORIZONS = (1, 5, 20)


class ForecastVerifier:
    def score(self, context: dict[str, Any]) -> dict[str, Any]:
        label = context["label"]
        prediction = context["prediction"]
        future = label["future"]
        forecast = prediction.get("forecast", {})
        metrics: dict[str, Any] = {}
        for horizon in HORIZONS:
            actual = future[horizon - 1]
            point = forecast.get(f"h{horizon}") if isinstance(forecast, dict) else None
            if not isinstance(point, dict) or "max_pen" not in point:
                metrics[f"mae_h{horizon}"] = None
                continue
            try:
                error = abs(float(point["max_pen"]) - float(actual["max_pen"]))
            except (TypeError, ValueError):
                metrics[f"mae_h{horizon}"] = None
                continue

            metrics[f"mae_h{horizon}"] = None if not math.isfinite(error) else error
        reported = [forecast.get(f"h{h}") for h in HORIZONS if isinstance(forecast, dict)]
        unsafe_calls = [point.get("unsafe") for point in reported if isinstance(point, dict)]
        actual_unsafe = [future[h - 1]["unsafe"] for h in HORIZONS]
        if len(unsafe_calls) != len(HORIZONS) or any(
            not isinstance(call, bool) for call in unsafe_calls
        ):
            metrics["unsafe_correct"] = None
        else:
            metrics["unsafe_correct"] = all(
                call == actual for call, actual in zip(unsafe_calls, actual_unsafe, strict=True)
            )

        return {"gates": {}, "metrics": metrics}


def load_verifier(*, root: Any) -> ForecastVerifier:
    del root
    return ForecastVerifier()
