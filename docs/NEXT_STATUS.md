# next.md implementation status

`next.md` is a strategy document: some of its items are engineering, some are
commercial motion that no commit can perform. This file maps every numbered item
to what is in the tree, what is deliberately not, and what evidence exists —
so the plan cannot quietly drift from the repository.

Status vocabulary:

- **shipped** — implemented in-tree, exercised by tests and by a real command.
- **partial** — the mechanism ships; the item also requires content or evidence
  that does not exist yet, named explicitly below.
- **external** — cannot be satisfied by code in this repository (a design
  partner, a published study, a signed contract). Instrumentation for it, where
  applicable, is shipped.

| Item | Status | Where |
|---|---|---|
| N1 — land one external user | external (instrumented) | `surgeval quickstart` measures time-to-first-vector; `docs/ONRAMP.md` is the ≤15-minute path; `surgeval doctor` prints fixes instead of stack traces |
| N2 — Isaac Lab world bridge | shipped | `src/or_audit/eval/sim/isaac_bridge.py`, `docs/examples/tasks/orbit-needle-lift/` (metrics-only after the audit below) |
| N3 — wrap kit (open world kinds, conformance, `surgeval wrap`) | shipped | `src/or_audit/eval/worlds.py`, `sim/base.py` plugin discovery, `eval/conformance.py`, `eval/wrap.py`, `eval/licensing.py`. Gate mapping is now cross-checked against the wrapped world's audited signal surface rather than trusted: a gate can only bind a **published physical** signal of the **named env at the pinned revision**, so one LapGym scene cannot borrow another's force channel, and a signal whose kind depends on construction (`collision_with_board`) must have that construction pinned via `--param`. The cited threshold must also be the number `fail_when` enforces, on both authoring paths |
| N3 — execution-determinism measurement | partial (0 of 7 measurable here) | The measurement is implemented and exercised — `surgeval conformance` reports a measured class per run, and a `FrozenLake-v1` probe under real `gymnasium` earns Tier 1 — but no wrap target can be stepped on this machine class, so every row records `determinism = "unmeasured"` **plus its concrete blocker**, and a class stronger than the measurement is refused by the schema. Blockers: NVIDIA GPU absent (`orbit-surgical`, `sonogym`, `surgical-gym`); user-built SOFA (`steve`, and `lapgym`, whose `setup.py` hard-requires x86_64 + Python 3.10); pybullet has no macOS wheel and its sdist fails against clang 21 on arm64 (`surrol`, install-tested here); AMBF is ROS/Linux-only (`surgicai`); `lumen` is blocked by our own unrecorded SPDX. The outstanding measurement that matters most is `lapgym` — the only target whose audited env publishes a gate-eligible physical signal |
| N4 — catalog sprint: 8 curated wraps | partial | `src/or_audit/install/catalog.toml` carries every Appendix-B disposition. 11 of 18 rows have a license read first-hand from the upstream text; 7 remain `unverified` and are refused installs, including first-party `lumen`, which gets no exemption. By strategy: 3 vendor-runtime rows plan an install today (`orbit-surgical`, `sonogym`, `surgical-gym`), 4 WRAP source-build rows plan a pinned fetch and hand the build to the user (`steve`, `lapgym`, `surrol`, `surgicai`), and the 10 WATCH/SKIP rows are refusals, not installs — the eleventh refused install is `lumen`, which is SHIPPED but unverified. CathSim moved WRAP → SKIP on its license |
| N4 — per-wrap gate mapping | shipped (7 of 7 audited, machine-checked) | Every audited WRAP row carries `[[worlds.envs]]` records read first-hand at its pin: the `info`/`extras` key, its kind (`physical` / `geometric` / `diagnostic` / `bookkeeping`), the file and line it is assigned on, whether it is *published*, and any construction condition. `lumen` is the one SHIPPED row with no envs — it is unverified and refused installs, so there was no pinned tree to read. 10 audited envs, 17 signals, all verified to resolve at their pins by `scripts/check_world_signals.py` (scheduled CI), which now parses the cited line and requires an actual publication rather than an occurrence. Result: exactly **one** wrap target (`lapgym`, scenes `grasp_lift_touch` / `tissue_dissection` / `pick_and_place`) can host a hard gate today; the other six are metrics-only *for a stated reason* — SurRoL computes contacts and discards them, SonoGym publishes a geometric proximity flag, stEVE publishes a NaN guard, ORBIT/SurgicalGym/SurgicAI publish only bookkeeping. `surgical-gym`'s empty signal surface is now stated explicitly through `absence_markers` rather than by omission |
| N5 — endovascular benchmark family | partial | `src/or_audit/eval/shelf.py` + `docs/examples/shelves/endovascular.toml`. Per-world rows, bench pairing, and the cross-world refusal ship; the shelf's stEVE row waits on a user-built SOFA, and its CathSim row is refused outright on terms |
| N6 — 10-minute agent on-ramp | shipped | `src/surgeval/decorators.py` capability inference, `surgeval init-agent` / `describe-agent`, `docs/ONRAMP.md` |
| N7 — verifiers-compatible train-time export | shipped | `src/or_audit/eval/export_verifiers.py`, `surgeval export-verifiers` |
| N8 — sellable surface | external | The cloud control plane remains a loopback beta (`src/or_audit/cloud/`). Pricing, tenancy, and BAA posture are commercial decisions, gated behind N1 evidence per next.md's own ordering |
| N9 — sim-to-phantom correlation | partial (machinery only) | `src/or_audit/eval/equivalence.py` implements the artifact, its four requirements, and the rank-correlation check against an external referent. The phantom study itself is physical work: no artifact is published, and the code refuses cross-world claims until one is |
| N10 — Apache-2.0 open-core distribution | shipped | `src/or_audit/install/` (catalog, installer, doctor), `surgeval quickstart` / `worlds` / `doctor`, `.github/workflows/install-smoke.yml`, wheel-packaged reference examples |
| N11 — hosted agentic concierge | shipped (rails), external (hosting) | `src/or_audit/concierge/{intake,assess,select,adapt}.py` + `surgeval concierge`. The deterministic machinery and every invariant ship; running it as a hosted product depends on N8 |

## What the kernel change actually opened (N3)

`WorldSpec.kind` was a closed `WorldKind` enum and eligibility was enum-set
membership in two validators. A third-party non-Gym world therefore could not
publish without a core release. Now:

- `WorldSpec.kind` is `WorldKind | Slug`, normalized like `GateSpec.kind`.
- Physics-oracle, closed-loop, and counterfactual eligibility, plus which
  fields a world requires, come from `WorldCapabilities` in
  `or_audit.eval.worlds` — declared by an installed adapter, or by the task's
  own `[environment.capabilities]` block when the adapter is absent. A task
  declaration that disagrees with an installed adapter is refused.
- Adapters are discovered from the `or_audit.world_kinds` entry-point group and
  carry a digest-pinned identity (`module:symbol` + SHA-256 of the adapter
  module). A task may pin the adapter it was authored against; the loader
  verifies the pin, and the identity is recorded in the head-covered
  `JobResult.world_engine`, so a patched adapter cannot run under an unchanged
  task and world pin.
- A failed third-party entry point is recorded (`surgeval sim kinds`,
  `surgeval doctor`), never raised at import: one broken plugin must not brick
  the kernel.

## What first-hand verification changed (2026-08)

Every catalog row was re-derived by reading the upstream LICENSE text, resolving
the commit SHA against the repository, and attempting the install. Doing the work
rather than trusting the survey moved data *and* found code defects the survey
could not:

| Finding | Evidence | Consequence |
|---|---|---|
| CathSim is CC-BY-NC-SA-4.0 with a field-of-use `TERMS.md` | upstream `LICENSE`, `TERMS.md` | `disposition = "skip"`. It runs and reports real contact forces — the blocker is terms, not engineering, so no amount of work promotes it |
| `cathsim` and `surrol` are not on PyPI (404) | `pypi.org/pypi/<name>/json` | the two `pip-extra` rows named packages that could never resolve; the strategy now has no rows, and `source-build` was added for runtimes the user compiles |
| `pip install steve` fetches an unrelated static-site generator | PyPI metadata | a same-sounding distribution name is not the world; pins are cited to repositories, not names |
| ORBIT-Surgical and SonoGym ship `LICENCE`, not `LICENSE` | upstream trees | `licensing.py` resolved neither, so two permissive worlds read as "no license declared". Both spellings now resolve |
| Four of five surveyed SHAs were fabricated or short | `api.github.com/repos/.../commits` | every pin is now a verified 40-char SHA, and SurRoL's default branch is `SR-PVPV`, not `main` |
| The Isaac, SOFA, and Warp stand-ins invented `max_pen`, `wall_force_n`, `tissue_stress_kpa`, `haptic_overshoot_mm` | `sim/*_bridge.py` before this change | the ORBIT example's six cited gates were resolving against fabricated numbers. Stand-ins now report progress only, the example is metrics-only, and conformance tiers on the *observed* backend rather than the declared `synthetic_stub` flag |
| `environment.parameters` was carrying the harness step limit | `gymnasium.make("FrozenLake-v1", max_steps=8)` raises `TypeError` | that dict is forwarded verbatim to a real constructor; the limit now comes from `[harness].max_steps`, and `sample_action` resolves the action space through the bridge (a `Discrete` space previously produced a float vector) |
| The `surrol` pin named a repo but not a world | `codeload` tarball at the pin | The pin was HEAD of the *default* branch (`SR-VPPV`), a VPPV research monorepo carrying **six divergent vendored copies** of `surrol/` (`psm_env.py` ranges 559–1212 lines across them) and no top-level package, so the cited paths resolved to nothing. Repinned to the `SurRoL-v2` branch commit, where both cited files exist. A SHA that resolves in a repository is not a SHA that contains the world |
| The `sonogym` citation named a directory, not the file | pinned tree | Cited `robot_US_guided_surgery.py`; the file is `robotic_US_guided_surgery.py` inside the `robot_US_guided_surgery/` directory. Substance held (`extras['cost']` at line 974), but the citation pointed at nothing |
| Nothing checked citations against pins | absence of any such check | Both defects above survived because `safety_evidence` was prose. Gate mappings are now typed per-env records (key, kind, path, line, published, construction condition) and `scripts/check_world_signals.py` verifies each against the fetched pinned tree. It caught an invented key (`contact_points` for SurRoL's `getContactPoints`) in this very catalog minutes after being written |
| A gate could cite one number and enforce another | `GateMapping(threshold=1.5, fail_when="contact_force_n > 999")` was accepted, as was the same divergence in a hand-written `task.toml` | The citation, the `threshold_basis.value`, and the predicate literal are three numbers and only the last decides a verdict. All three must now agree, checked by AST on both authoring paths. A gate that can never fire, wearing a normative citation, is worse than an uncited number |
| A gate could publish a unit the engine never produced | `GateMapping(signal="dynamic_force_on_gallbladder", unit="N")` was accepted while the catalog records `scaled-N`; separately `GateSpec(unit="N", threshold_basis.unit="mmHg")` was accepted | Binding the number was not enough — a threshold in a false unit is a physically false claim, and §2.6 compares gates *by unit*, so it would have made the gate falsely comparable to a real newton reading elsewhere. The published unit must now equal the audited unit exactly, gate-eligible signals must record one, and a package's gate unit must match its own basis unit. Found in my own test helper, which mapped LapGym's scaled force as `N` and passed |
| The citation checker itself was fail-open | `scripts/check_world_signals.py` printed `SKIP` on a fetch failure and still exited 0 if any other world passed; missing `curl` and "checked nothing" also exited 0 | The one check that makes "audited" mean anything could have gone green while verifying none of the seven targets. Now fails closed: exit 1 unresolved citation, exit 2 could-not-check, 0 only when every audited world was fetched *and* verified. Being scheduled rather than PR-gating is what allows that strictness. A bounded retry absorbs a registry hiccup without ever passing on failure |
| `conformance --out` inside the task directory exploded | `shutil.copytree` walking its own output | Produced a ~100-level nested path wall instead of an error. Now refused with the fix named |

Reachability was proved, not asserted. A throwaway wrap of `FrozenLake-v1` under
real `gymnasium` passes all four checks and earns Tier 1, so the new backend rule
is satisfiable; `surgeval conformance --require-tier1` exits 0 on it. The same
command exits 1 on the in-tree ORBIT package and names both reasons — the
metrics-only declaration and the observed stand-in. Note what is *not* claimed:
that exact package cannot be re-run against a stub, because `gym_bridge` has no
synthetic path at all — a missing `gymnasium` is a refusal, not a stand-in.

## What three independent reviews found (2026-08)

The automated reviewers on the PR did not run — one hit a usage limit, the other
crashed before analysis — so three independent reviews were commissioned instead,
each required to reproduce a probe rather than assert a concern. All three
returned "incorrect". Every finding below was reproduced first and is now
regression-tested. The pattern is worth naming: **almost every defect was a check
that passed without establishing its claim**, which is the exact failure this
codebase exists to prevent, committed by the code that exists to prevent it.

| Finding | Evidence | Consequence |
|---|---|---|
| A gate could cite 1.5 N and enforce zero | `fail_when = "x > false or x > 1.5"` was accepted with `threshold=1.5`, and fired at 0.5 N — `false` compares equal to `0` | Ordering against a boolean, or anything boolean-producing, is refused. Equality against a boolean stays legal, since that is the ordinary boolean-gate pattern |
| A gate could publish no unit, or no number, over an audited quantity | LapGym's `gripper_jaw_peg_collisions` (audited in `contacts`) was accepted as a bare flag with neither | A bare signal name is a test against zero; omitting the unit relabels a measurement, omitting the threshold leaves the boundary uncited. Both refused. `GateSpec.unit` vs `threshold_basis.unit` is now compared unconditionally — an empty side is a mismatch |
| `unmeasured` determinism satisfied Tier 1 | four bare passing checks plus `determinism_class=UNMEASURED` validated as `tier=1` with no evidence at all | Tier 1 requires `bitwise` or `tolerance` *and* matching evidence at the report's own tolerance. `--tolerance inf` is refused before two jobs run: a comparison that excuses every difference has compared nothing |
| A stand-in could certify itself | `stepped_world` was derived from the `physics` capability, so a `closed_loop=True, physics=False` adapter waived its own provenance gate | Derived from the runner's actual route and cross-checked against `closed_loop`. The observed `world_pin` is now compared against the task's and the other run's; an unobserved pin is unverifiable, not a match |
| A diverged solver was recorded as a safe reading | the recorder normalised `NaN` to `0.0`, so `digest(jsonable(nan)) == digest(jsonable(inf))` and two divergent runs measured as bitwise-deterministic | Non-finite values are tagged distinctly and resolve to unassessable at every consumer. The generated verifier was one `bool()` from the worst case: `bool("__nonfinite__:nan")` is `True`, so a diverged run would have reported **success** |
| A batched Warp run scored corrupt numbers silently | `num_envs` was decoration; per-env flags `[False, False]` produced a one-step episode with `reward: [0.5, 1.5]` written to the trajectory, nothing raised | A completed, hashed, corrupt run is worse than a crash. All three bridges now refuse a batch they never reduced, and the "large-batch policy evaluation" claim is gone from the docstring that made it |
| A misconfigured task silently became a stub | `max_steps` reached `gymnasium.make`, which raised, and a bare `except Exception` read that as "no backend installed" | A silent substitution of a stand-in for a real backend is a fabricated measurement with extra steps. Registration errors are now resolved by name; anything else refuses, regardless of `synthetic_stub` — that flag permits a stand-in when the runtime is absent, it does not swallow configuration errors |
| Two adapters with different behaviour shared one "content pin" | a factory whose source could not be read hashed only its name, so two instances collided at `anonymous+2f183a…` | Refused: a digest derived from a constant string pins nothing. Registering an engine for a world kind with no declared capabilities is also refused — an undeclared adapter withholds every eligibility, and the task was left granting itself physics on its own word |
| An equivalence artifact could unlock a ranking it never covered | it was validated only against itself: no comparison to the shelf's real gates, and `("world-a", "world-a")` passed | The shelf now persists a per-world gate manifest and the artifact must cover it exactly, by id *and* unit. `None` (never established) and `()` (declares no gates) are kept distinct end to end |
| A hand-edited `shelf.json` changed a cross-world ranking | rows were reconstructed from arbitrary JSON with no head verification, and rank came from stored list order | A reader now reaches its rows by the same evidence path the writer did, and any divergence is refused |
| An empty gate set rendered as a clean safety result | metrics-only rows printed "Gate failures: no" in both HTML and CLI | Rendered as Tier-0 with the reason; a missing backend reads as unattested, never as real |
| Concierge quarantine was documented, not enforced | `assert_frozen_before_scoring`'s only callers were tests; a frozen package with an edited verifier scored and published | Enforced at `run_job` and at public-leaderboard ingestion. Stated limit: this is drift detection, not authentication — see the claim ledger |
| The exported RL rubric returned a forged reward verbatim | a three-key dict with two arbitrary strings returned `99.0` as the training reward | The record is revalidated, its pins compared, and the reward recomputed locally from the vector |
| `GPL-3.0-only AND (MIT OR Apache-2.0)` classified as permissive | parentheses were stripped before evaluating `AND`/`OR` | SPDX is parsed with grouping and precedence intact. Zero verdicts changed across all 18 catalog rows — the fix closes a hole without moving a single existing judgement |
| `doctor` reported healthy without probing | a required `unknown` check exited 0, and container worlds returned `ok` from tool presence alone | An unprobed requirement is not a satisfied one. The pinned image is inspected offline; unprobeable is `unknown` and a required `unknown` now fails |
| `--dry-run --execute` installed | `--dry-run` had a permanent `True` default and execution keyed only off `--execute` | Mutually exclusive. The dangerous direction was the one that worked |
| Generated packages could be injected through their own inputs | an `env_id` containing `"""` closed the module docstring; a `world_pin` newline opened a forged `[attestation]` table; a `--param` name escaped its inline TOML table — and `wrap` reported success each time | Every interpolation is escaped, control characters are refused at the boundary, and nothing is written until the rendered artifacts parse *and* round-trip the request. A scaffold the kit calls written must be a package that loads |

Two things did not need fixing and are recorded so nobody re-litigates them: the
gate DSL is not an evaluator and cannot reach arbitrary code from a task package,
and hostile-archive extraction was already refused. The one accidental mitigation
found — `canonical_digest` refusing non-finite floats, which two separate defects
were unknowingly relying on — is no longer load-bearing: every path that could
carry a non-finite value into a digest now refuses or abstains before reaching it.

## What is deliberately still refused

These are not gaps; they are the product.

- **No cross-world aggregate, ranking, or ordering** without a validated
  `EquivalenceArtifact` for that shelf and task family. `shelf.json` carries no
  cross-world number, and `shelf rank --cross-world` exits 1 without one.
- **No gate on a world that does not report the state it would score.** A wrap
  without a mapped safety signal must declare `environment.metrics_only`, which
  forbids hard gates, forbids `safety_critical`, and is stamped into the
  head-covered provenance and every scorecard surface.
- **No training reward from fabricated or unmeasured physics.**
  `export-verifiers` refuses a synthetic-stub task, a metrics-only task, a task
  with no declared projection, and a projection that does not zero a hard-gate
  failure. Every emitted reward carries its projection digest and parent vector
  reference.
- **No determinism claim stronger than the measurement.** The conformance suite
  measures a class from two identical runs and refuses a stronger declaration;
  `unmeasured` is the default, and `tolerance` is a legitimate Tier-1 outcome.
- **No unaudited fetch.** `surgeval worlds install` refuses a world whose
  license is `unverified`, an undigested container image, and a vendor runtime
  without explicit EULA acknowledgement. Isaac Sim is never redistributed.
- **No Tier 1 from a stand-in, and no self-classification out of that rule.**
  A world the harness *steps* earns Tier 1 only with an observed `real` backend
  read from both runs' head-covered provenance; a synthetic stand-in and a
  bridge with no `engine_provenance` reporter both drop to Tier 0. The
  `stepped_world` switch that scopes this rule is required (no default) and is
  cross-checked against the installed adapter's declared `physics` capability,
  so a report cannot reclassify its world to waive the requirement, and a kind
  with no installed adapter cannot reach Tier 1 at all.
- **No physical key from a world that has no physics.** The Isaac, SOFA, and
  Warp stand-ins synthesize no `max_pen`, `wall_force_n`, `tissue_stress_kpa`,
  or `haptic_overshoot_mm`. A gate bound to a fabricated force resolves *pass*,
  which is the most convincing available lie.
- **No deserialization of an untrusted upload.** Concierge intake requires a
  tenant-signed manifest (HMAC-SHA256 over every declared field), accepts only
  non-executing weight formats or a digest-pinned tenant container, and hashes
  bytes it never interprets. Endpoint intake refuses private, loopback, and
  link-local ranges, and probes only from the sandbox tier.
- **No agent-authored world scored in place.** An adapted scenario space is
  frozen into a new versioned, digest-pinned package marked
  `authored_by: agent` and excluded from public leaderboards before the first
  scored trial; the concierge can never edit a published verifier, gate, or
  projection.

## Kill criteria (next.md §6) are unchanged

None of this work satisfies N1. If after N1 plus the first three wraps no
external team has run a package unchanged **and** asked for a second run, the
commercial thesis fails and the honest move is Future A — a research harness
whose tasks are papers. The instrumentation to notice that (time-to-first-vector,
conformance reports, install-smoke) is now in the tree; the decision is not a
code change.
