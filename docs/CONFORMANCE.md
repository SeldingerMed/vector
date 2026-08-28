# Conformance and Claim Ledger

This document records what SurgEval verifies, what it does not verify, and what claims a conformant result may make. It is the authoritative reference for interpreting scorecard outputs.

## Conformance levels

### Task conformance

A task package conforms when:

1. `task.toml` validates against the schema: required fields present, closed kernel vocabularies respected, headline metric exists in the metric set.
2. `instruction.md` exists and states the task's purpose, data, and claim boundary.
3. The verifier (`verifier.py`) loads and returns a `VerifierRuntime` with declared metrics and gates.
4. Input and label files exist and are non-empty (for non-simulation tasks).
5. `WorldSpec` is valid: `kind` is a registered world kind or a slug backed by a declared `[environment.capabilities]` block; every field the world's capabilities require is present (`gym_id`, `world_pin`, `contract_path`); `synthetic_stub` is declared; a declared `adapter`/`adapter_digest` pin matches the installed adapter's content digest; and `metrics_only` packages declare no hard gates and are not `safety_critical`.
6. The interface spec declares `id`, `interaction_mode`, `protocol_versions`, and (when applicable) `modalities`.

Eligibility is read from declared world metadata, not from a closed enum: a physics oracle requires a world declaring `physics`, closed-loop interaction requires `closed_loop`, a counterfactual interface requires `counterfactual`. A third-party world publishes those declarations through a `or_audit.world_kinds` entry point (`WorldAdapter`), and the kernel refuses a task whose own `[environment.capabilities]` block disagrees with the installed adapter.

**What task conformance does not verify**: the clinical validity of the task design, the appropriateness of the gates for a real clinical workflow, or the correctness of the oracle labels. These are the task author's responsibility. In particular, task conformance does **not** establish that the wrapped world actually reports the safety state its gates bind to — that is Tier-1 conformance (`surgeval conformance`), below.

### Tier-1 world conformance

Package validity says a task is well-formed; Tier-1 says a *wrapped world* is curated. `surgeval conformance` runs four checks and records the result as an artifact:

1. **Gate-state availability** — every declared gate resolves its evidence from state the world actually reports. A gate that is never assessable is refused: the honest alternative is `environment.metrics_only = true` (Tier 0), or an upstream PR exposing force/collision/penetration in `info`.
2. **License audit** — the wrapped world's license is on the permissive allowlist; restricted and unknown licenses fail.
3. **Evidence replay** — the stored trajectory reconstitutes the stored vector through the bundled verifier.
4. **Execution determinism** — a seeded rerun is *measured*, not assumed, and recorded as a determinism class (`bitwise` / `tolerance` / `nondeterministic`). A declaration stronger than the measurement is refused. Many wrapped engines (SOFA, PhysX) are only best-effort deterministic, so `tolerance` is a legitimate Tier-1 outcome and `nondeterministic` is not. `unmeasured` — the honest default — does **not** satisfy Tier 1: the class must be `bitwise` or `tolerance` *and* the determinism check must carry evidence measured at that class and that tolerance, so a serialized report cannot mint a tier its own evidence does not support. The tolerance must be finite; `--tolerance inf` is refused before two jobs run, because a comparison that excuses every difference has compared nothing.

Tier 1 additionally requires a digest-pinned world adapter (`environment.adapter` + `adapter_digest`), so a patched or swapped adapter cannot run under an unchanged task and world pin.

Tier 1 also requires an **observed real backend** on any world the harness steps. The backend is read from both runs' head-covered `world_engine` provenance, never from the task's `synthetic_stub` flag — that flag is a *permission* to fall back, and a bridge that reaches a real engine anyway must not be demoted for having declared it. A synthetic stand-in is Tier 0 (a stub is deterministic because it is fake, which is the opposite of evidence), and so is a bridge exposing no `engine_provenance` reporter: there is nothing to attest. Two runs that disagree on the backend are refused rather than resolved.

The switch that scopes this rule (`stepped_world`) is derived from the runner's actual route — `interface.interaction_mode is closed-loop` — and cross-checked against the adapter's declared `closed_loop` capability. It is required in every report and refused when it disagrees with the registry: a report cannot reclassify the world it ran in order to waive its own provenance gate. It is deliberately **not** derived from the `physics` capability, which was a proxy that let a non-physics closed-loop adapter certify its own stand-in. Tier 1 also compares the head-covered `world_pin` observed in each run against the task's pin and against the other run's; a bridge that reports no pin yields no observed pin, and an unobserved pin is treated as unverifiable rather than as a match. A dataset-backed kind (`frame-source`, `counterfactual`, `angiostress-contract`) steps no engine and is judged on its pinned data instead. A world kind with **no installed adapter** cannot reach Tier 1 at all, because there is nothing to cross-check its declared capabilities against.

### What a gate may bind to

Two rules decide whether a hard gate is a physical claim or a fabricated one, and both are enforced at authoring time rather than discovered in a scorecard.

**1. The signal must be one the named env actually publishes.** For a world in `install/catalog.toml`, each audited env carries the signal surface read first-hand at the world's pin: the `info`/`extras` key, its kind (`physical` / `geometric` / `diagnostic` / `bookkeeping`), the file and line it is assigned on, and whether it is *published* at all. Only a **published physical** signal can carry a hard gate. Three distinctions do real work here:

- a quantity the engine computes and discards cannot be gated — SurRoL calls `getContactPoints` for grasp logic and never publishes it;
- a geometric predicate is not a measurement — SonoGym's `extras['cost']` is a proximity flag over drill-tip pose, not a force;
- a diagnostic is not a safety signal — stEVE's `simulation_error` fires on NaN tracking, so a run that trips it is *invalid*, not unsafe.

Eligibility is a property of **(pin, env)**, never of the world. LapGym's `grasp_lift_touch` publishes a gallbladder internal force; its other scenes do not. A wrap of `tissue_dissection` that binds `dynamic_force_on_gallbladder` is refused and told which env does publish it, because a package-wide signal list would let one scene borrow another's gate — a fabricated gate with a real citation. Where a signal's kind depends on construction, that condition is recorded and must be pinned: LapGym's `collision_with_board` is a counted contact only under `with_board_collision=True`, and otherwise silently becomes a `cauter_position[2] < 0.0` pose predicate under the same key.

Uncatalogued worlds are unconstrained, and so are unaudited scenes of catalogued ones: a third party wrapping their own world is self-service. Only *borrowing* an audited sibling's signal is refused. Citations are checked against the pinned tree by `scripts/check_world_signals.py`, which is what makes "audited" mean something. That check does three things a weaker one would not. It **authenticates the tree**: the source is fetched with `git fetch --depth 1 <sha>` and `HEAD` is verified to equal the requested revision, so git's content addressing — not the URL, and not a marker file — is the trust root; a cache is reused only when it is a clean git worktree at that exact sha. It **requires a publication, not an occurrence**: the cited line is parsed, and a `published` signal must appear there as an assignment, attribute store, subscript store, or dict key, so a key sitting in a comment or a one-word docstring no longer resolves. And it **fails closed**: exit 1 is an unresolved citation, exit 2 is a world that could not be checked at all — unreachable source, missing `git`, an empty cited file, or a world named on the command line that has no audited envs — and only a run in which *every* requested world was fetched and verified exits 0. It runs on a schedule rather than on pull requests, which is exactly what lets it be strict: an unreachable upstream turns the job red instead of quietly reporting success over an unchecked claim.

An env with a genuinely empty signal surface states that explicitly through `absence_markers` — line-pinned readings recording *where* the absence was established — rather than by publishing nothing and letting silence be read as evidence. A marker proves the cited line exists at the pin and the named symbol is used there; it does not prove nothing else is published, which remains the first-hand read recorded in `safety_evidence`.

**2. The cited number must be the number the predicate enforces.** A numeric gate carries three numbers — `threshold`, `threshold_basis.value`, and whatever literal `fail_when` compares against — and only the last one changes a verdict. All three must agree. A gate citing a normative source at 1.5 N while enforcing `contact_force_n > 999` is refused on both authoring paths (`surgeval wrap` and a hand-written `task.toml`); previously it was accepted, and the scorecard, `wrap.json`, and rendered verifier docstring would all have displayed 1.5. An inline number with no declared threshold is refused as uncited, and a declared threshold the predicate never uses is refused as decoration. Compound predicates remain fully supported when their bounds are **declared inputs** rather than inline literals — the pattern `lumen-nav-safe` ships (`unsafe or max_pen > safety_max_pen or diverged`) — because one gate carries one cited threshold and two inline bounds cannot both be cited.

**3. The published unit must be the audited unit, exactly, and the number must exist.** A bound number with an unbound unit is the same defect as an unbound number. LapGym's `dynamic_force_on_gallbladder` is a SOFA internal force *multiplied by a scene scaling factor*, so the catalog records it as `scaled-N`; a gate publishing a threshold on it as `N` states a quantity the engine never produced, and is refused. Because §2.6 gate equivalence compares gates *by unit*, a false unit is not merely cosmetic — it would make the gate falsely comparable to a genuine newton reading in another world. A gate-eligible signal must therefore record a unit at all (`contacts` for a count, `scaled-N` when a factor is applied), a gate binding an audited quantity must declare that same unit, and in a hand-written package `GateSpec.unit` must equal its own `threshold_basis.unit` — compared unconditionally, so an empty side is a mismatch and only two empty strings mean *dimensionless*. A gate over an audited quantity must also declare a numeric threshold: omitting it does not make the gate unitless, it applies an uncited boundary, because a bare signal name is a test against zero and zero was never cited by anyone.

**4. A boundary may not be smuggled in as a boolean.** `false` compares equal to `0`, so `fail_when = "x > false or x > 1.5"` enforces an uncited zero boundary while displaying a cited 1.5 — and it fires at 0.5 N. An ordering comparison against a boolean literal, or against anything boolean-producing (`not ...`, a nested comparison, an `and`/`or` result), is refused. Equality against a boolean is untouched: that is the ordinary boolean-gate pattern.

**5. A non-finite reading is never a measurement.** A diverged solver reports `NaN`, and the recorder used to normalise that to `0.0` — the safest possible force reading, invented, then hashed into the head as evidence. Non-finite values are now tagged distinctly in the trajectory and resolve to *unassessable* everywhere they are consumed: gate evidence abstains rather than comparing, a metric value becomes `None`, and a generated verifier returns `None` rather than `bool("__nonfinite__:nan")`, which is `True` and would have reported a diverged run as a success. `GateOutcome.confidence` and a projection's reward literals refuse non-finite values outright, since both are hashed into the head.

### Agent conformance

An agent package conforms when:

1. `agent.toml` validates against the schema: `id`, `agent_version`, `kind`, `weights_pin`, `weights_path`, at least one `CapabilitySpec`, and a `RuntimeDescriptor`.
2. The weights file at `weights_path` exists and its SHA-256 digest matches `weights_pin`.
3. The runtime entrypoint loads and returns the expected runtime type (`PolicyRuntime` or `PredictorRuntime`).
4. Each `CapabilitySpec` declares `interface`, `interaction_modes`, `protocol_versions`, and `schema_wildcard` (defaulting to `false`).

**What agent conformance does not verify**: the model's clinical performance, its generalization to populations not represented in the dataset, or its safety in a real clinical deployment. These require evaluation results, not package validation.

### Binding conformance

A binding (task-agent pair) conforms when:

1. The agent declares at least one `CapabilitySpec` whose `interface` matches the task's `InterfaceSpec.id`.
2. The capability's `interaction_modes` include the task's `HarnessSpec.interaction_mode`.
3. The capability's `protocol_versions` intersect with the interface's `protocol_versions`.
4. If `schema_wildcard = false`, the capability's observation and output schemas must satisfy the interface's declared schemas. If `schema_wildcard = true`, the binding is accepted but `binding_mode: "wildcard"` is stamped in the config and scorecard.

**What binding conformance does not verify**: semantic compatibility between the agent's outputs and the task's expectations beyond the declared schema. A wildcard binding is structurally valid but does not prove the agent produces meaningful outputs for this task.

### Result conformance

A job result conforms when:

1. Every trial has a `TrialVector` with gates and metrics matching the task's verifier declaration.
2. Every gate has a status of `pass`, `fail`, or `not-assessable` (never `null` for gates that the verifier assesses).
3. The `head` (SHA-256 of the canonical job payload) is present and matches recomputation.
4. The config records `task_dir` and `agent_dir` as relative paths within the bundle.
5. The bundle's task and agent tree digests match the config's `task_digest` and `agent_digest`.
6. If the task uses a simulation backend, the head-covered `world_engine` provenance records the engine, backend, backend version, world pin, and the world adapter's `adapter_id`/`adapter_digest` taken from the kernel registry (never from the bridge's own report).
7. If `synthetic_stub = true`, the scorecard displays a synthetic banner and the result is not exportable as RL training data.
8. If `metrics_only = true`, the head-covered provenance carries the label and every scorecard surface says the row is not safety-attested.

**What result conformance does not verify**: that the agent was the best possible model, that the gates caught every failure mode, or that the result generalizes beyond the evaluated data.

## Claim ledger

| Claim | Verified by | Limitation |
|---|---|---|
| "Agent X binds to task Y" | `surgeval bind` | Structural compatibility only; not semantic correctness. |
| "Agent X produced result Z on task Y" | Job config + result head + bundle digests | Proves the agent ran; does not prove the result is clinically meaningful. |
| "Result Z is reproducible" | `surgeval replay` | Proves the vector reconstructs from the trace; depends on the bundled task and agent packages being available. |
| "Gate G passed" | Verifier output in the trial vector | Proves the gate's condition was met for this trial; does not prove the gate is sufficient for clinical safety. |
| "Headline metric H = v" | Verifier-computed metric in the vector | Proves the metric value for this evaluation; does not prove the metric is the right measure of performance. |
| "Result Z is exportable as RL data" | `surgeval export-rl` with projection | Only for non-synthetic-stub runs; the projection rule is task-declared and versioned. |
| "World engine E was used" | `world_engine` provenance in config | Records which engine and whether it was a synthetic stub; does not validate the engine's physics fidelity. |
| "Agent identity is A@v+pin" | `agent.toml` weights_pin + bundle digest | Proves the agent package is pinned; for SDK-synthesized agents, the pin is derived from the model's serialized bytes. |
| "World adapter A@digest produced this row" | `world_engine.adapter_id` + `adapter_digest` in the head-covered result | Proves which adapter content ran; a patched adapter changes the digest and the head. Does not validate the adapter's physics. |
| "This row is metrics-only, not safety-attested" | `environment.metrics_only` → head-covered `world_engine.metrics_only` + scorecard banner | Proves the package declined to declare gates; does not prove the world lacks safety state, only that this wrap does not claim it. |
| "Gate G binds a signal world W really publishes" | `AuditedEnv` signal surface read first-hand at `world_pin`, path/line verified against the fetched tree by `scripts/check_world_signals.py` | Proves the key is assigned where the catalog says, at that pin, and is published to `info`/`extras`. Scoped to **(pin, env)**: says nothing about a sibling scene, an unaudited scene, or another revision. Does not validate that the engine's number is physically accurate. |
| "Gate G's threshold is the number G enforces" | AST comparison of `fail_when`'s numeric boundaries against `threshold` and `threshold_basis.value` | Proves the cited number is the applied number. Does not make the number *correct*: the citation is still only as good as the source it names. |
| "Gate G's threshold is in unit U" | `GateMapping.unit` matched exactly against the audited `WorldSignal.unit`, and `GateSpec.unit` against its own `threshold_basis.unit` | Proves the published unit is the one the reading was audited in. Does not prove the *engine's* unit is physically meaningful: `scaled-N` is recorded as scaled precisely because upstream's factor makes it not newtons. |
| "World W's determinism is D" | `surgeval conformance` two-run measurement recorded as a class | Never inherited from an engine's reputation. `unmeasured` is the default and carries the blocker that stopped the measurement; a declared class without a naming measurement is refused. |
| "World W is Tier-1 curated" | `surgeval conformance` report (four checks, adapter pin, measured determinism class, observed backend) | Scoped to the sampled trials and the measured determinism class; not a claim about every scenario the world can produce. |
| "A real engine produced this row" | `world_engine.backend == "real"` in **both** runs' head-covered provenance, cross-checked against the adapter's declared `closed_loop` capability and the observed `world_pin` | Proves the harness reached a real backend rather than a stand-in, at the revision the task names. Does not validate that backend's physics fidelity, and says nothing about a dataset-backed world, which reports no engine by construction. |
| "Policy A is safer than policy B across worlds" | **Nothing in the harness** — requires a validated `EquivalenceArtifact` whose `gate_equivalence` covers the shelf's persisted per-world gate manifest exactly, matched by gate id *and* unit | Refused by default: shelf surfaces report per-world rows only until equivalence is published (§2.6). An artifact naming a gate the shelf worlds do not declare, omitting one they do, or naming the same world twice is refused; a world whose manifest was never established cannot be aggregated at all. |
| "Reward r came from task T's projection" | The generated `reward_func` revalidates the record, requires its task/projection/world pins to equal the generated constants, requires `parent_vector_ref` to equal the canonical digest of the supplied vector, and **recomputes** `project(vector, projection)` locally, returning the recomputed value | A projection of a vector, never a score; a hard-gate failure projects to 0. The supplied scalar is not trusted — previously it was returned verbatim, so any dict with two non-empty strings became the training reward. |
| "This shelf row is safety-attested" | Row-level `gates.safety_attested`, derived from the task's declared gates and the head-covered `world_engine` | An empty gate set renders as "no hard gates declared", never as "no gate failures": a Tier-0 metrics-only row is not a clean safety result. A row whose bundle recorded no engine provenance renders as unattested, not as real. |
| "This scored run used the package that was frozen" | `run_job` re-derives `content_digest` (tree digest excluding `provenance.json`) and the parent's `verifier_identity` from the package on disk, and refuses on drift | Only for a package that still carries its `provenance.json`. Deleting that file removes the claim, and a package making no adaptation claim is checked for nothing. Does not prove the frozen package was ever *derived* from the named parent — only that its verifier still matches the identity recorded at freeze time. |
| "Agent-authored packages are excluded from public leaderboards" | `leaderboard.py::_verified_result` refuses the whole build, naming the package; structural adaptation tells refuse a tell-bearing package that presents no provenance | Same escape: an operator who deletes `provenance.json` *and* renames out of the adaptation conventions gets an ordinary row. The authorship label itself is not forgeable (`authored_by` must be `agent`, and a self-granted `public_leaderboard_eligible: true` refuses to parse), so the only route out is destroying the evidence, not editing it. **This is drift detection, not authentication** — every digest is self-authored and lives beside what it pins. A public trust boundary needs an anchor off the operator's disk; that belongs to the hosted surface. |
| "Signal S is published at line L of file F" | `scripts/check_world_signals.py` parses the cited line and requires an assignment, attribute store, subscript store, or dict key | Proves the key is *stored* under that name at that line in a git-verified tree at the pin. Does not prove that mapping reaches `info`, and does not prove nothing else is published. An `absence_marker` proves only that its cited line exists and uses the named symbol. |
| "The cached tree is the revision we asked for" | `git fetch --depth 1 <sha>` plus `rev-parse HEAD == sha` and a clean worktree | SHA-1 content addressing is the trust root, not the cache directory's permissions: an actor who can rewrite `.git` defeats any local check. |
| "This metric is unassessable" | A non-finite reading resolves to `None` at every consumer rather than to a number | Proves nothing was measured, which is the honest reading of a diverged solver. Distinguishes "not measured" from `0.0`; it does **not** diagnose *why* the world diverged. |

## What SurgEval does not claim

1. **Clinical validation**: SurgEval is evaluation infrastructure, not a clinical validation framework. A passing result does not certify a model for clinical use.
2. **Gate sufficiency**: The platform enforces that gates are evaluated and reported. It does not certify that the declared gates are sufficient for any clinical scenario.
3. **Dataset representativeness**: The platform pins datasets by digest. It does not assess whether a dataset is representative of any patient population.
4. **Model robustness**: A result on one task does not imply robustness on related tasks, out-of-distribution inputs, or adversarial conditions.
5. **Simulation fidelity**: The platform records the simulation engine and stamps synthetic runs. It does not validate that the simulation accurately models any real procedure.
6. **Cross-world comparability**: A shared `TrialVector` vocabulary makes results *legible* side by side, not comparable. Cross-world aggregation, ranking, or ordering is refused until a validated equivalence artifact exists for the shelf and task family (§2.6: matched task semantics, gate equivalence in physical units, scenario-distribution alignment, and an external referent).
7. **Stand-in physics**: A synthetic stand-in reports progress and its own step bookkeeping only. It synthesizes no penetration, force, stress, or haptic value, so a gate bound to one abstains instead of passing — but the absence of a fabricated number is not evidence about the real world, and a stub-backed run is Tier 0 by construction.
8. **Third-party adapter correctness**: The kernel pins a world adapter by content digest and records it in the head. It does not audit what that adapter does with the engine's state; a Tier-1 conformance report is the evidence that its gate mapping resolves and its determinism class was measured.

## Interpretation guide

A conformant SurgEval result supports the following chain of reasoning:

1. **What ran**: The config and bundle identify the exact task, agent, and runtime.
2. **What happened**: The trial vectors record gates, metrics, and typed traces.
3. **What it means**: The claim boundary in `instruction.md` scopes what the result asserts.
4. **Whether it holds**: Replay reconstructs the vectors from traces through the bundled verifier, and the result head catches any mismatch.

Everything beyond this chain — clinical interpretation, regulatory submission, deployment decisions — is the consumer's responsibility, not a SurgEval claim.
