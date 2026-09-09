# F5 — Planning utility with a learned dynamics model (Phase F)

Status: measured 2026-09-08 on CPU. Reference-scale proof that planning
with a learned model selects better-executing actions than planning with
nothing — not evidence the model is good, and not closed-loop MPC.

## Model

MLP dynamics `(obs[5], act[2]) → next obs[5]`, 7-128-128-5, trained on 32
random-policy episodes (30 steps each), held out 8 episodes. Normalized
MSE: train 0.11, held out 0.13. Pinned stack: same as the training recipe
(`seldinger-lumen` @3c6bb39e, warp 1.15.0, torch 2.14.0 CPU).

## Protocol (fixed before running)

- 5 starting cases (seeds 0-4), replanned every 8 steps over a 30-step
  episode. Planner and baseline share the action budget and seeds.
- Planner: 32 random 8-step candidate sequences scored under the learned
  model by mean remaining distance + 2× mean penetration proxy; execute
  the best open-loop, then replan.
- Baseline: one random 8-step sequence per block (same real-step budget).
- Outcomes measured in the real pinned simulator only: progress
  (`dist` reduction) and any-unsafe. Model scores never leave the
  planner.

## Results

| Arm | Progress per seed | Any-unsafe seeds |
|---|---|---|
| planned | 11.31, 13.48, -0.56, 19.37, 17.54 (mean 12.2) | 2/5 |
| random | 0.51, -0.92, -8.07, -0.47, -1.42 (mean -2.1) | 3/5 |

The planner advances where random drifts. Safety is not shown to improve
(2/5 vs 3/5 unsafe on n=5 is noise), and one planned seed still goes
backward — plausible forecasts do not imply safe actions.

## Limits

- Toy model, toy horizon, n=5: no significance claim, no transfer claim.
- Open-loop blocks, not per-step MPC; no uncertainty-aware planning.
- Seeds do not vary initial conditions (see dossier), so this measures
  planner quality on fixed starts, not generalization.
- No checkpoint is committed; retrain from the recipe above. A stronger
  model or a real planner would change the numbers, not the protocol.
