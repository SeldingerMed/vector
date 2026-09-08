# A1 — Endovascular benchmark brief (Phase A flagship)

Status: brief frozen 2026-09-08. Implementation (A2+) follows; this document
is the decision record, not a completion claim.

Ledger correction: PR #34 (stale Lumen install-refusal note) was Phase 0
drift cleanup, not the A1 brief. This file is the actual A1 deliverable per
`docs/NEXT_STATUS.md` workstream A.

## Decision

One task family: `guidewire-branch-navigation`, reported per world. No
cross-world ranking without a validated `EquivalenceArtifact` — the existing
`shelf rank --cross-world` refusal stands
(`docs/examples/shelves/endovascular.toml`).

## Composition (pinned)

| Row | Task | World / revision | Score content |
|---|---|---|---|
| Sim 1 | `lumen-nav-safe` (exists) | Lumen `lumen-gym`, world pin `3c6bb39ecc916c73e81131b94353ad8deee62bd4`, dist `seldinger-lumen` 0.0.0 (pinned commit's own version; live HEAD reads higher — the pin governs) | Hard gates on `lumen.info.unsafe`, `lumen.info.max_pen`; `safe_success` vs `success` |
| Sim 2 | `steve-arch-variety-nav` (to author, metrics-only) | stEVE `sofa`, world pin `f909e7122d846a3c2fec91eedddaa0ba88b6391d` | `TargetReached` + `MaxSteps` + `simulation_error` NaN-guard diagnostic; no hard gates (upstream exports no wall-force channel) |
| Bench | `angiostress-dias` (exists) | Real-data frozen-model perception contract (DIAS angiography) | Claim-footer bound stress test; no sim-to-real proof by itself |

## Intended claim (and limits)

- Supported: reproducible per-world policy comparison on fixed scenarios with
  preserved safety vectors and explicit unassessable states.
- Not supported: clinical safety certification, tissue-injury thresholds
  (Lumen penetration is an engineering limit until C2 provenance exists),
  cross-engine ranking, or transfer to phantom/patients (needs C5 evidence).

## Exclusions

- CathSim: technically the best safety instrumentation surveyed, excluded on
  license (CC-BY-NC-SA-4.0 + field-of-use TERMS.md), not engineering.
- Human surgeon credentialing (refused at task load by design).
- Composite scores; hard gates stay separate from metrics.

## Evaluation discipline (D preview)

- Independent case unit: scenario/episode with realized initial state recorded;
  adjacent frames are not independent cases.
- Held-out scenario split frozen before model comparison; paired cases across
  policies; uncertainty clustered at the independent unit.
- Full statistical machinery lands in workstream D; no comparison ships without it.

## Compute envelope

- Lumen rollouts: `seldinger-lumen` at the pin plus its pinned solver stack
  (warp-lang 1.15.0, newton @6dfe730 — read from the pinned commit's own
  pyproject, not live HEAD). Core import path is numpy-only.
- Reference/dev host (this Mac, arm64, CPU, no SOFA) — tested 2026-09-08 in a
  throwaway venv, layer by layer:
  - core `import lumen` at the pin: OK;
  - `register_gym_envs()` then `gym.make("Lumen/NavTreeBranch-v0")`: constructs,
    `reset(seed=0)` + 5 random steps OK on CPU Warp (reset `info` keys: none —
    harness signal mapping verified separately, below);
  - without the solver stack, construction fails on `import warp`: the solver
    stack is required, GPU is not (on this host).
  Harness end-to-end passed 2026-09-08 on the exact pinned stack via the
  production path (registered `LUMEN_GYM` engine → `make_gym_bridge`):
  `python -m pytest tests/test_lumen_real_env.py` — 1-episode random run
  reports observed `backend=real` + world pin, an assessed `wall_penetration`
  gate, and a replay-matched head. stEVE/SOFA remains unbuildable on this
  host (x86_64 Linux required).

## Acceptance for A (reminder, not claimed here)

Clean-host install, real checkpoints, frozen held-out manifest, per-case
safety/success/abstention with failure evidence, external unchanged-package
reproduction plus a requested second use. A2–A5 PRs check off against this
brief; deviations amend this file first.
