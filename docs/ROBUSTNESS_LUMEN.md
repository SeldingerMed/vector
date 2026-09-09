# C4 — Lumen robustness note (Phase C)

Status: measured 2026-09-08 on CPU Warp, pinned stack
(`seldinger-lumen` @3c6bb39e, warp-lang 1.15.0, newton @6dfe730).
Random baseline only (`seldingermed/random`, n=2 per arm); this is a
harness-fault smoke comparison, not a policy robustness claim.

## Method

Harness-applied faults need no world support: the runner applies and records
them, so the comparison is valid on any backend. Fault arm attaches two
perturbations by in-test task copy (never shipped in the packaged task):

- `harness-observation-gaussian-noise` (std 0.01)
- `harness-action-hold` (repeat previous action)

Both arms run the production path (registered `LUMEN_GYM` engine).

## Results

| Arm | Head (prefix) | Gates (wall_penetration) | safe_success | max_pen |
|---|---|---|---|---|
| nominal | `32dc6ae4…` | pass, pass | false, false | 0.0, 0.0 |
| faults | `1d50682a…` | pass, pass | false, false | 0.0, 0.0 |

Recorded perturbations present in every fault trial trajectory
(`obs-noise`, `act-hold`); replay of fault arms against the packaged
(un-faulted) bundle is correctly refused as a head mismatch — replay
re-executes, it does not hallucinate the faults back.

## Limits

- Random actions neither reach the target nor touch the wall, so both arms
  agree trivially. A meaningful robustness comparison needs a competent
  policy (Phase E) and wider fault grids.
- World-native scenario distributions (anatomy, friction, latency) are not
  supported by the pinned env: it ignores reset options, so task-level
  scenario grids would be refused ornament. Harness faults are the
  supported robustness axis until a world revision applies scenarios.
