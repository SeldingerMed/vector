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
        # Denominator policy: a horizon past the recorded future is
        # unassessable (None), never a zero error. Short episodes from early
        # termination therefore shrink coverage instead of faking accuracy.
        for horizon in HORIZONS:
            if horizon - 1 >= len(future):
                metrics[f"mae_h{horizon}"] = None
                metrics[f"unsafe_h{horizon}"] = None
                continue
            actual = future[horizon - 1]
            point = forecast.get(f"h{horizon}") if isinstance(forecast, dict) else None
            if not isinstance(point, dict) or "max_pen" not in point:
                metrics[f"mae_h{horizon}"] = None
            else:
                try:
                    error = abs(float(point["max_pen"]) - float(actual.get("max_pen")))
                except (TypeError, ValueError):
                    error = None
                metrics[f"mae_h{horizon}"] = (
                    None if error is None or not math.isfinite(error) else error
                )
            call = point.get("unsafe") if isinstance(point, dict) else None
            if not isinstance(call, bool) or not isinstance(actual.get("unsafe"), bool):
                metrics[f"unsafe_h{horizon}"] = None
            else:
                metrics[f"unsafe_h{horizon}"] = call == actual["unsafe"]

        return {"gates": {}, "metrics": metrics}


def load_verifier(*, root: Any) -> ForecastVerifier:
    del root
    return ForecastVerifier()
