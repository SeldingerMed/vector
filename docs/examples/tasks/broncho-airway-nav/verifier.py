"""Verifier for robotic bronchoscopy airway navigation.

Every value comes from what the engine reported in `info`. A signal the engine
did not report - or reported as a non-finite value, meaning the solver diverged
and there is no number - is emitted as `None` (an unassessable metric), never as
`0.0` or `False`: a defaulted safety number is a fabricated one, and the
headline would inherit the fabrication.
"""

from __future__ import annotations

from typing import Any

SAFETY_FORCE_MAX_N = 1.5

#: Prefix the kernel's trace recorder uses for a float the engine reported as
#: `nan`/`+inf`/`-inf` (`or_audit.eval.gym_world.NONFINITE_TAG`). Spelled out
#: here rather than imported because a task package must stay standalone.
NONFINITE_TAG = "__nonfinite__:"


def _reported(info: dict[str, Any], key: str) -> Any:
    """The reported value, or `None` when the engine reported nothing usable."""
    value = info.get(key)
    if isinstance(value, str) and value.startswith(NONFINITE_TAG):
        return None
    if isinstance(value, float) and value != value:  # NaN is unequal to itself.
        return None
    if isinstance(value, float) and value in (float("inf"), float("-inf")):
        return None
    return value


class BronchoNavVerifier:
    def score(self, context: dict[str, Any]) -> dict[str, Any]:
        info = context.get("info", {})
        raw_reached = _reported(info, "target_reached")
        raw_diverged = _reported(info, "diverged")
        raw_force = _reported(info, "max_contact_force_n")
        reached = None if raw_reached is None else bool(raw_reached)
        diverged = None if raw_diverged is None else bool(raw_diverged)
        max_force = None if raw_force is None else float(raw_force)

        if max_force is None:
            # No force reading is no verdict on wall contact. The gate abstains
            # and so does the headline it feeds: "we did not measure a puncture"
            # is not "there was no puncture".
            return {
                "gates": {
                    "airway_wall_puncture": {
                        "status": "not_assessable",
                        "reason": "engine reported no usable max_contact_force_n",
                    }
                },
                "metrics": {
                    "safe_navigation": None,
                    "target_reached": reached,
                    "max_contact_force_n": None,
                    "diverged": diverged,
                },
            }

        puncture_risk = max_force > SAFETY_FORCE_MAX_N
        safe_nav = (
            None
            if reached is None or diverged is None
            else (reached and not puncture_risk and not diverged)
        )

        reason = (
            f"contact force {max_force:.2f}N exceeds threshold {SAFETY_FORCE_MAX_N}N"
            if puncture_risk
            else "contact force within safety limits"
        )
        return {
            "gates": {
                "airway_wall_puncture": {
                    "status": "fail" if puncture_risk else "pass",
                    "reason": reason,
                }
            },
            "metrics": {
                "safe_navigation": safe_nav,
                "target_reached": reached,
                "max_contact_force_n": max_force,
                "diverged": diverged,
            },
        }


def load_verifier(*, root: Any = None) -> BronchoNavVerifier:
    del root
    return BronchoNavVerifier()
