"""Generated verifier for the wrapped world `Isaac-Lift-Needle-PSM-v0`.

Every value here comes from what the engine reported in `info`. A signal the
engine did not report is emitted as `None` (an unassessable metric), never as
`0.0` or `False`: a defaulted safety number is a fabricated one, and the
headline would inherit the fabrication.

This wrap maps no hard gate, so `GATES` is empty and every returned value is a
measurement. That is the metrics-only posture of §2.2, not an omission: a gate
here would need a cited threshold and a signal this world is known to report.
"""

from __future__ import annotations

from typing import Any

#: (gate id, engine info key, predicate) for every declared hard gate.
GATES: tuple[tuple[str, str, Any], ...] = ()


def _reported(info: dict[str, Any], key: str, *aliases: str) -> Any:
    """First reported alias, or ``None`` when the engine reported none."""
    for name in (key, *aliases):
        if name in info:
            return info[name]
    return None


def _boolean(info: dict[str, Any], key: str, *aliases: str) -> bool | None:
    value = _reported(info, key, *aliases)
    return None if value is None else bool(value)


def _numeric(info: dict[str, Any], key: str, *aliases: str) -> float | None:
    value = _reported(info, key, *aliases)
    return None if value is None else float(value)


class WrapVerifier:
    """Verifier for the wrapped world `Isaac-Lift-Needle-PSM-v0`."""

    def score(self, context: dict[str, Any]) -> dict[str, Any]:
        raw_info = context.get("info")
        info: dict[str, Any] = raw_info if isinstance(raw_info, dict) else {}
        raw_success = _boolean(info, "raw_success", "success", "is_success")
        diverged = _boolean(info, "diverged")
        breached: list[str] = []
        unreported: list[str] = []
        for gate_id, signal, predicate in GATES:
            value = _reported(info, signal)
            if value is None:
                unreported.append(signal)
            elif predicate(value):
                breached.append(gate_id)
        metrics: dict[str, Any] = {
            "raw_success": raw_success,
            "diverged": diverged,
        }
        return {"gates": {}, "metrics": metrics, "unreported_signals": sorted(unreported)}


def load_verifier(*, root: Any = None) -> WrapVerifier:
    del root
    return WrapVerifier()
