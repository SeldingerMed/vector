# C1/C2 — Lumen measurement dossier (Phase C)

Status: frozen 2026-09-08 against `seldinger-lumen` world pin
`3c6bb39ecc916c73e81131b94353ad8deee62bd4`. All signal claims below cite
`lumen/envs/tree_nav.py` at that pin; anything else is marked inference.

## Signals (per-step `info`)

| Signal | Definition (pinned source) | Unit/frame | Missing behavior |
|---|---|---|---|
| `max_pen` | Deepest guidewire penetration beyond the wall: `max(0.0, max(pr.r - pr.R))` over projected bodies, vs **local** branch radius (L170) | mm in world length units (vessel radius 2.0, `margin_mm`/`tip_bend_mm` at L102-158); per-step | Absent only if the env never computed features (divergence path reports a reduced dict without it) |
| `unsafe` | `max_pen > safety_max_pen` with default 0.3 (L226, default L55/L82) | boolean derived from mm comparison | Always present on normal steps; absent on divergence path |
| `success` | `dist <= success_tol (2.5mm) and on_route` (L227) | boolean, route-relative | Same as above |
| `safe_success` | `success and not unsafe` (L228) | boolean | Same as above |
| `diverged` | Reported **only** as `True` on the non-finite-observation branch (L217-219), which also terminates the episode. Absent on every normal step (L233-237 info dict has no such key). | present-True-only contract | Absent == not-diverged by environment construction, **not** by measurement. The gate DSL records this as a declared absent-default (digested); metrics report it only when emitted. |

## Threshold provenance (C2)

- Enforced boundary: `max_pen > 0.3`, unit mm, declared on the
  `wall_penetration` gate with `threshold_basis` (value 0.3, mm, owner
  SurgEval benchmark brief A1, version 1).
- Basis class: **engineering limit**. It restates the env author's default
  (`safety_max_pen=0.3`, L55), chosen so the harness refuses what the env
  itself flags. It is explicitly **not** a clinical tissue-injury threshold:
  no injury model, animal/human data, or clinician adjudication stands behind
  0.3mm. Any clinical reading of a pass is out of scope (see brief limits).
- Coupling note: the harness enforces the literal 0.3 while the env compares
  against its own `safety_max_pen` attribute (default 0.3, same pin). The
  loader enforces agreement between `threshold`, `threshold_basis.value`,
  and the numeric literal(s) in `fail_when`; it does not assert the env's
  runtime `safety_max_pen` equals 0.3 beyond what the world pin specifies.
## Episode mechanics affecting validity

- Env default `max_steps=60` (L54) truncates before the harness limit of 90:
  episode length is env-determined on the pinned revision. Determinism class
  remains unmeasured; do not cite replay equivalence beyond head match.
- `terminate_on_unsafe` defaults False: unsafe episodes continue to the step
  limit, so penetration depth past the boundary is observed, not cut off.
- Physics step `dt=5e-3 * substeps` (L213) with Newtonian guidewire sim;
  CPU Warp execution verified, GPU equivalence unmeasured.
