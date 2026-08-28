# Isaac-Lift-Needle-PSM-v0 (wrapped)

Drive the policy in `Isaac-Lift-Needle-PSM-v0`, hosted through the `isaac-lab` world adapter and pinned at `6e47534f7d412e4be523116f250c992a63146883`.
Episodes run for at most 60 steps; 3 seeded evaluation episode(s) per run.

## Claim boundary

- This package makes one claim: *this policy ran in `Isaac-Lift-Needle-PSM-v0` at `6e47534f7d412e4be523116f250c992a63146883` under this harness*.
- Results are per-world rows. No cross-world aggregate, ranking, or ordering is licensed until a published equivalence artifact covers this shelf.
- Execution determinism is `unmeasured` until `surgeval conformance` measures a seeded rerun of this env.
- The vendor runtime is not redistributable here, so a non-physical stand-in may serve the world. Those artifacts are stamped `backend="synthetic-stub"` and RL export refuses them: they are plumbing evidence, never physical evidence.

## Not safety-attested (metrics-only)

This package is explicitly **not safety-attested**. The wrapped world does not report the safety state a hard gate would need, so it declares no gates and is not `safety_critical`. Nothing here attests that a run was safe — only that it happened and how the declared metrics came out. Synthesizing a gate from state the env never reports would be the exact failure this label exists to prevent; the honest fix is upstream instrumentation.

## Next

Run `surgeval conformance` on this package before publishing it: Tier-1 placement requires measured gate-state availability, a license check, evidence-replay round-trip, and a recorded determinism class.
