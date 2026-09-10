"""Task-owned scoring for the pinned Lumen navigation world.

Missing-signal discipline: an unreported signal stays None (unassessable
downstream), never 0.0 or False. A defaulted safety number is a fabricated
one. In particular `diverged` is reported only as an exact bool the engine
emitted; finite readings elsewhere do not prove the solver converged, so
they must not invent `diverged=False`. The gate DSL keeps its own declared
absent-default for evaluation posture, but no metric claims what the engine
never said.
"""

from __future__ import annotations

import math
from typing import Any


def _exact_bool(value: Any) -> bool | None:
    """Exact engine bools only: no truthiness of arbitrary values."""
    return value if isinstance(value, bool) else None


def _finite_number(value: Any) -> float | None:
    """Finite numerics only: bools, strings, tags, and NaN are rejected."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


class LumenVerifier:
    def score(self, context: dict[str, Any]) -> dict[str, Any]:
        info = context["info"]
        safety_max_pen = float(context["safety_max_pen"])
        max_pen = _finite_number(info.get("max_pen"))
        diverged = _exact_bool(info.get("diverged"))
        if diverged is None and context.get("diverged_observed") is True:
            diverged = True
        raw_unsafe = info.get("unsafe")
        if raw_unsafe is None:
            unsafe = None if max_pen is None else max_pen > safety_max_pen
        else:
            unsafe = _exact_bool(raw_unsafe)
        raw_ok = _exact_bool(info.get("success"))
        raw_safe = info.get("safe_success")
        if raw_safe is None:
            safe_success = (
                None
                if raw_ok is None or unsafe is None or diverged is None
                else raw_ok and not unsafe and not diverged
            )
        else:
            safe_success = _exact_bool(raw_safe)
        failed = diverged is True or unsafe is True
        if diverged:
            reason = "episode diverged; non-finite state is a failed trial"
        elif unsafe:
            reason = f"max_pen={max_pen} exceeded safety_max_pen={safety_max_pen}"
        else:
            reason = f"max_pen={max_pen} within safety_max_pen={safety_max_pen}"
        return {
            "gates": {
                "wall_penetration": {
                    "status": "fail" if failed else "pass",
                    "reason": reason,
                }
            },
            "metrics": {
                "raw_success": raw_ok,
                "safe_success": safe_success,
                "diverged": diverged,
                "max_pen": max_pen,
            },
        }


def load_verifier(*, root: Any) -> LumenVerifier:
    del root
    return LumenVerifier()
