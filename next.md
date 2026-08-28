# next.md — Vector as the environment index and scorekeeper for medical robotics

**Thesis (revised):** Vector is not an environment company and not a general agent-eval company. It is the **niche hub + neutral scorekeeper for procedural medical AI**: the place where every medical robotics / world-model environment — ours or anyone's — becomes loadable, pinned, safety-gated, and replayable, with results reported in one vector vocabulary and made cross-world comparable only where equivalence is validated (§2.6). Harbor and Prime Intellect define the *shape* (harness + hub + leaderboard); we do not compete in their market. NVIDIA Isaac for Healthcare is a **supplier of worlds**, not a competitor for this layer.

**Status:** written against `surgeval` 0.3.0a0, `docs/ASSESSMENT.md` (decision: Future B), `docs/BUILD.md`, and the public product surfaces of Harbor, Prime Intellect, and NVIDIA Isaac for Healthcare as of 2026-08 (see Appendix A for sources and audit caveats).

---

## 1. What we have

- **Kernel:** four harness modes (closed-loop, interactive, single-turn, counterfactual); immutable `org/name@version` task/agent packages; capability binding; subprocess isolation; typed `ProceduralTrace`; vector scorecards with hard gates that cannot be averaged away; abstention as a legal outcome; **evidence/verifier replay** (the vector is reconstructed from the stored trace through the bundled verifier — `reconstitute.py` explicitly does *not* step the world; a seeded world rerun is future work, see N3); digest-pinned RL projections.
- **World plumbing that already anticipates aggregation:** `SimulationEngine` protocol + registry (`src/or_audit/eval/sim/base.py`), a generic `GymnasiumBridge` for any Gymnasium/PyBullet env with `world_pin` and backend provenance (`real` / `synthetic-stub` / `unknown`), plus SOFA and Warp bridges.
- **First-party content:** Lumen (one sim world: continuum instrument in deformable lumen; endovascular navigation, CathSim/stEVE-class) and AngioStress (one real-data frozen-model perception bench). These are *content*, not the product.
- **Distribution scaffolding:** public registry repo, PyPI publishing with SBOM/attestation, beta cloud control plane (RunPod executors, loopback-default).

Based on their public product surfaces, none of Harbor, Prime, or Isaac exposes the medical-specific contract — `safe_success` vs `success`, gate/metric separation, abstention, PHI class, panel/ICC machinery, replayable attestation (not evident in [A1–A3]; see Appendix A). That is the moat *if* environments and users exist on the rails.

## 2. The coverage problem, stated honestly

Lumen covers one modality. We have no frameworks for robotic surgery, bronchoscopy, endoscopy, orthopedics, ultrasound guidance, or the dozens of other procedural domains — and we want to serve essentially all medical robotics and world models.

**We resolve this by aggregation, not authorship.** The hub pattern (Prime didn't author its environments; HF didn't train the models) plus one fact unique to this niche: the entire universe of credible open medical-robotics RL environments is small — on the order of 15–25 repos, not thousands. Exhaustive curation is impossible for LLM environments and *achievable here by 1–2 engineers in ~2 quarters*. Total catalog coverage of a field is a moat no horizontal player will bother to build.

### 2.1 Why wrapping is a product, not a chore

The existing open envs share four defects, and they are exactly our kernel's features:

1. **Scalar rewards, no raw/safe split.** Task success is the reward; wall force, tissue trauma, and collisions are buried in state or absent. Mapping each env's state to a `SafetyGateSet` is the curation craft.
2. **Bit-rot.** Academic envs pin to dead dependency versions and stop running after the paper. "The pinned, replayable, actually-runnable version of every medical env" is maintenance work startups will pay to avoid.
3. **No shared reporting vocabulary.** An endovascular team cannot today run one policy against CathSim-class, stEVE, and Lumen without three bespoke harnesses, and the results come back in three incompatible reward dialects. One trace/vector vocabulary makes those results **legible side-by-side** — per-world, gates separated, replayable — which is something **no env author can credibly self-issue**, including us for Lumen alone. Neutrality requires listing competitors' envs beside our own. Note the deliberate limit: shared vocabulary is *legibility*, not *comparability* — see §2.6.
4. **License landmines.** Some envs carry restrictive or contaminating licenses (the repo already firewalls CathSim-style contamination). A per-env license audit (`scripts/check_license_allowlist.py` exists) is itself part of the curated artifact.

### 2.2 Three-tier coverage model

| Tier | What | Guarantee | Status |
|---|---|---|---|
| **0 — Loadable** | Any Gymnasium/PyBullet/SOFA env through the existing bridges | Runs; provenance recorded; **metrics-only, explicitly not safety-attested** | Largely built (`gym_bridge`, `sofa_bridge`, `warp_bridge`) |
| **1 — Curated** | Pinned commit + gate mapping + determinism/replay validation + license audit, published as a task package | Full vector: hard gates, abstention, replay | The catalog. The main new work. |
| **2 — First-party** | Envs we build (Lumen; candidates below) | Same as Tier 1, plus we control physics fidelity and RL throughput | Lumen shipped; bronchoscopy candidate |

Tier-1 promotion rules where an env lacks safety instrumentation: (a) patch and upstream a PR exposing force/collision/penetration in `info` — goodwill plus marketing into that env's user base — or (b) ship honestly labeled metrics-only. Never synthesize a gate from state the env does not report.

### 2.3 Build-vs-wrap decision rule (governs Tier 2 forever)

Build first-party only when **all three** hold:

1. No credible open env exists for the modality (endovascular in 2024 ≈ only a license-encumbered option and a SOFA one — hence Lumen).
2. A design partner concretely needs it.
3. It fits our physics competence (continuum/deformable, differentiable, GPU-batched).

By this rule: **bronchoscopy/endoluminal navigation is the next first-party candidate** — Lumen's core names no anatomy and repurposing to airway is a profile swap, not a new engine. **We do not build a dVRK/laparoscopic sim** — SurRoL, LapGym, and ORBIT-Surgical/Isaac cover it; we wrap them.

### 2.4 The initial catalog target (survey in Appendix B; verify licenses/status per env before committing)

The full survey of candidate environments — including low-traffic academic ones — lives in **Appendix B**, with a wrap/watch/skip disposition per env. The v0.2 shelf targets:

- **Endovascular:** Lumen (first-party) · **stEVE + `stEVE_bench`** (SOFA BeamAdapter; the bench repo means gate mapping starts from published task definitions) · **CathSim** (MuJoCo; license audit first — the repo already firewalls CathSim contamination). Shelf goal: the same policy *run* across all three and reported per-world (§2.6); cross-world claims wait for the equivalence artifact. Overlap is deliberate: this shelf *is* the §2.6 equivalence program.
- **Robotic surgery / dVRK:** **ORBIT-Surgical via the Isaac Lab bridge** (highest-leverage single wrap — inherits NVIDIA's asset investment) · **SurRoL / SurRoL-v2** (PyBullet, 14 tasks, the most-adopted dVRK RL platform) · **LapGym / sofa_env** (SOFA, 12 envs; `sofa_bridge` seam exists) · **SurgicAI** (AMBF-based fine-grained suturing subtasks, NeurIPS 2024 D&B — arrives with metric definitions, the easiest gate mapping).
- **Ultrasound-guided:** **SonoGym** (Isaac-based robotic ultrasound / spine, NeurIPS 2025 — rides the same N2 Isaac bridge; the freshest codebase in the survey).
- **Bronchoscopy/endoscopy:** thin open landscape (watchlist: ROOM continuum sim, AI-Co-Pilot-Bronchoscope-Robot) → Tier-2 first-party candidate per §2.3; near-term, synthetic CT-airway (`broncho-airway-nav` matured).

### 2.5 Two registry axes: worlds and benches

Lumen and AngioStress are **separate categories and stay separate**:

- **Worlds** (sims, ours and wrapped): where policies act.
- **Benches** (real-data frozen-model contracts, AngioStress-class): where sim claims get stress-tested.

Editorial rule, already in ASSESSMENT §6.3 and now structural: every sim domain shelf pairs with a real-data bench, and no external claim ships from sim rows alone. "Train in any sim on the shelf, stress-test on real data, one replayable artifact" is a pairing no env author or horizontal hub offers.

### 2.6 Loadable → legible → comparable (comparability is earned, not schema'd)

A shared `TrialVector` schema does **not** make scores from different worlds comparable. CathSim-class, stEVE, and Lumen differ in observation/action interfaces, anatomy and scenario distributions, dynamics fidelity, termination conditions, and what their safety state physically means. Pretending otherwise would make the neutrality claim technically unsound. Three levels, each with its own bar:

| Level | Claim permitted | Requirement |
|---|---|---|
| **Loadable** | "This policy ran in world W under the harness" | Tier 0/1 bridge; provenance + pin recorded |
| **Legible** | "Here are per-world vectors, side by side, same vocabulary" | Tier 1: gate mapping, replay, declared units — **no cross-world aggregate, ranking, or ordering** |
| **Comparable** | "Policy A is safer than policy B across worlds" | Validated equivalence (below), published as its own artifact |

Cross-world comparability requires, per world pair and task family:

1. **Task equivalence:** matched objective, initial-state distribution, and termination semantics, declared in the task packages, not assumed.
2. **Gate equivalence:** each hard gate maps to the same physical quantity in the same unit (e.g. wall force in newtons), with per-engine calibration showing the thresholds bite at comparable physical events — not just identically named gates.
3. **Scenario-distribution alignment:** anatomy/difficulty distributions matched or explicitly stratified.
4. **An external referent:** agreement with a ground truth neither engine owns (the N9 phantom study is exactly this).

Until an equivalence artifact exists for a shelf, every public surface reports **per-world rows only**, and the runner/scorecard must refuse to emit a cross-world aggregate — the same refusal posture the kernel already applies to scalar collapse.

## 3. Competitive posture (revised)

- **Harbor / Prime:** shape templates, not competitors — they are LLM/terminal/agent-native. Borrow: harness-with-a-benchmark gravity, hub network effects, one-function on-ramp. Interoperate: a verifiers-compatible export puts our tasks in front of RL labs with the gated projection as the reward (hard-gate fail ⇒ 0) and the full vector logged. Never adopt: `reward.txt` as the primary interface.
- **NVIDIA Isaac for Healthcare:** supplier. They build and give away worlds; based on their public surfaces they do not offer neutral cross-vendor evaluation, vector safety semantics, or replayable attestation — and structurally are unlikely to (they sell compute to every vendor and grade none). One Isaac Lab bridge converts their spend into our Tier-1 catalog.
- **Our one-sentence position:** *the neutral, replayable, safety-gated scorekeeper and environment index for procedural medical AI* — the layer every env author needs and none can self-issue.

## 4. Remaining product gaps this strategy does not erase

1. **Zero external users.** The stated v0.3 milestone — one external team runs a published package unchanged — has not happened. Everything is downstream of it.
2. **On-ramp ceremony.** Package + `CapabilitySpec` + subprocess protocol is the publication path, not a trial path; the `@se.agent` decorator exists but isn't the headline and has no docs site.
3. **Eval-time only.** `export-rl` is post-hoc; a startup training a policy today would use a wrapped env's raw Gymnasium interface and bypass the safety kernel — the exact failure the product forbids. Needs a train-time (verifiers-style) surface.
4. **No scale-out.** CPU-only local runner; cloud is beta/loopback. Fine for N1; not for regression fleets.
5. **No sellable surface.** No pricing, tenancy, or BAA posture shipped.
6. **No sim-to-real evidence.** No published correlation between any sim vector and phantom/real outcomes; AngioStress documents transfer *failure*, which is honest but is the negative result.
7. **Content licensing wall for clinical video.** Public surgical video datasets are generally research-only; commercial tasks come from synthetic worlds, wrapped envs (license-audited), or an eventual own corpus (gated behind L4–L7).

## 5. The plan

### N1 — Land one external user (now; the only milestone that matters)
2–3 design partners from the natural funnel (Lumen users, AngioStress users, one endovascular/endoscopy autonomy team). Deliverable: they run a published taskset against their own model with zero harness changes and say whether the vector changes a decision. Instrument time-to-first-vector; must be under 15 minutes; fix whatever breaks that before anything else on this list.

### N2 — Isaac Lab world bridge (co-first; the coverage move)
`WorldSpec`/`SimulationEngine` adapter for Isaac Lab, proven on one ORBIT-Surgical subtask with a pinned Isaac commit and the raw/safe split enforced at the adapter. This single wrap makes robotic-surgery startups addressable and follows the existing `sofa_bridge`/`warp_bridge` seam.

### N3 — The wrap kit: open world kinds + adapter conformance + `surgeval wrap` (weeks, alongside N2)
- **Kernel change (prerequisite):** open the world-kind extension point. Today `register_simulation_engine` accepts `str` keys, but `WorldSpec.kind` is the closed `WorldKind` enum and validators gate on hardcoded enum sets (`task.py:106`, `task.py:495-533`), so a third-party non-Gym world cannot publish without a core release. Extend `WorldSpec.kind` to `WorldKind | Slug` (the `GateSpec.kind: GateKind | Slug` precedent at `task.py:205`), move physics-capability and closed-loop eligibility from enum-set membership to declared world metadata, and add plugin discovery (entry-point or container entrypoint) for adapter factories with digest-pinned adapter identity. Without this, `surgeval wrap` is self-serve for Gym IDs only.
- A conformance test suite defining Tier-1: gate-state availability, license check, evidence-replay round-trip, and **execution determinism** — a seeded world rerun given `(pin, seed)` reproduces the trace/vector. Execution determinism is *new work*: today's `replay` verb is evidence replay only (`reconstitute.py` does not step the world), and many wrapped engines (SOFA, PhysX) are only best-effort deterministic, so the check must measure and record a determinism class per world (`bitwise` / `tolerance` / `nondeterministic`) rather than assume it.
- `surgeval wrap <env>` scaffolding that emits a task-package skeleton from a Gymnasium env.
- This is what lets env authors and startups self-serve shelf placement; the registry only compounds if third parties can add to it.

### N4 — Catalog sprint: 8 curated wrapped envs (quarter)
Per §2.4 and Appendix B, in order: ORBIT-Surgical task (from N2) → stEVE + `stEVE_bench` → SurRoL → LapGym/sofa_env → SonoGym → SurgicAI → CathSim (license permitting) → Surgical Gym (capacity permitting; same Isaac bridge). Each wrap includes the gate mapping, the N3 determinism-class measurement, upstream PRs where safety instrumentation is missing, and an announcement into that env's user community — every wrap is marketing to an existing user base.

### N5 — Benchmark family: the endovascular shelf, per-world (after N2–N4 partial)
Not a single-env leaderboard and **not a cross-world ranking**: a benchmark *family* — **one policy class run across the endovascular shelf (Lumen, stEVE, CathSim-class), reported as per-world leaderboards** with gates separated from metrics, abstention shown, and every row replayable from its bundle. No cross-world aggregate, ranking, or ordering until the §2.6 equivalence artifact for this shelf is published; the scorecard renderer must refuse it, the same way it refuses composite scalars. Apply ASSESSMENT §6.2 B1/B2 gates verbatim. Seed it ourselves with public checkpoints; do not wait for submissions. Side-by-side per-world vectors are already the demo no env author or horizontal hub can produce; the validated comparison comes later and is worth more because it was withheld until earned.

### N6 — 10-minute agent on-ramp (weeks, parallel)
`@se.agent` + `surgeval run` as the front page; auto-generated `CapabilitySpec`; docs site with install → reference agent → your model → read vector → replay. Package authoring remains the publication path only.

### N7 — Train-time surface: verifiers-compatible export (month)
Any `surgeval` task exports as a verifiers-style environment whose reward is the versioned projection (zero on hard-gate fail) with the parent vector reference attached. Publish `lumen-nav-safe` + one wrapped env on the Prime Environments Hub as the medical vertical — their distribution, our contract. Invariant: no export path emits a scalar without the projection digest and parent vector reference (ASSESSMENT R3).

### N8 — Sellable surface (after N1 evidence)
Hosted parallel eval past loopback beta (auth, tenancy, GPU workers), private task registries, signed scorecards. Price per evaluated model-version. Sell runs and attestation; never the harness, never labels (ASSESSMENT R4). Write the BAA/PHI posture down before clinical tasks exist, because buyers ask first.

### N9 — Evidence moat: sim-to-phantom correlation + shelf equivalence (start now; publishes in 2–3 quarters)
Policies ranked by `safe_success` per world on the endovascular shelf vs the same policies on a silicone flow-loop phantom. Even N=5 policies with a rank correlation would be more published sim-to-real evidence than we have found for any medical gym. The phantom doubles as the **external referent for §2.6**: which worlds' rankings agree with the phantom (and with each other) *is* the shelf-equivalence artifact that unlocks cross-world comparison on N5. This converts "nice schema" into "predictive instrument" and gates the eventual regulatory-attestation business, which stays unbuilt until N1–N8 show pull.

### N10 — Apache-2.0 open-core distribution: idiot-proof install and launch (with N4; hardened by N6 feedback)
The harness is already Apache-2.0 (in-tree `LICENSE`; ASSESSMENT R4). This item makes the open distribution *real*: everything needed to run local evals — CLI, kernel, bridges, wrap kit, reference tasks, wrapped-world adapters — ships open; the hosted control plane and the N11 concierge stay commercial. The boundary is R4's: open the rails, sell runs and attestation.

The bounce risk is not the license, it is installation: SOFA needs specific builds, Isaac Sim cannot be redistributed, MuJoCo/PyBullet are pip-friendly, Unity worlds are their own runtime. Idiot-proofing ladder:

1. **`uv tool install surgeval` → `surgeval quickstart`**: runs a CPU-only reference task end-to-end, no GPU, no sim deps (promote `scripts/quickstart.sh` into the CLI). First vector in minutes on any machine.
2. **`surgeval worlds list` / `surgeval worlds install <id>`**: each world package declares its install strategy —
   - `pip-extra` (PyBullet/SurRoL, MuJoCo/CathSim): plain extras with lockfile pins.
   - `prebuilt-container` (SOFA worlds: stEVE, LapGym): we publish digest-pinned OCI images, because SOFA-from-source is the single worst first-hour experience in the survey.
   - `vendor-runtime` (Isaac worlds: ORBIT-Surgical, SonoGym, Surgical Gym): we cannot redistribute Isaac Sim; the installer drives NVIDIA's container pull + EULA acceptance, then verifies the pinned version.
3. **`surgeval doctor`**: per-installed-world checks (GPU/driver/container runtime/pins) that print the fix, not a stack trace.

Contract: happy path = fresh machine → first vector on a real sim world in **≤3 commands**, offline-after-fetch, digest-pinned, zero config-file editing; measured inside N1's time-to-first-vector metric. Containers wrap the runner and the world, never patient data (ASSESSMENT R1). CI: one install-smoke job per strategy, kept out of kernel CI (§6.5 pattern).

### N11 — Hosted agentic concierge: assess → select → adapt (after N6 + N8; adapt after N4)
The hosted differentiator is not a dashboard, it is an **evaluation agent** on top of the open rails. Four capabilities, shipped in order — and capability 0 is a security gate, not a feature:

0. **Intake (pre-Assess untrusted-model gate).** Uploaded weights and linked endpoints are hostile until proven otherwise: model formats and entrypoints can execute code on deserialization, and arbitrary endpoint probing is an SSRF primitive aimed at whatever the control plane can reach — including a customer's clinical network. B3 tenancy (protecting *their* IP from *us*) addresses neither. Requirements before the agent sees anything:
   - **Uploads:** signed manifest (declared format, digest, entrypoint); accept safe weight formats (e.g. safetensors) or a digest-pinned runtime container supplied by the tenant — never a bare pickle/checkpoint with our loader. All deserialization and probing happen in a disposable, no-egress sandbox with CPU/GPU/disk quotas and artifact scanning; `trust_remote_code` and equivalent dynamic-code paths disabled. **The control plane never deserializes an upload.**
   - **Endpoints:** tenant-registered, allowlisted hosts only (no redirects off-list, no private/link-local ranges); probes are synthetic-only inputs at bounded rate; probing runs from the sandbox tier, never from the control plane.
   - Only intake-passed identities become runtime descriptors the Assess step may touch; the intake result (format, digests, sandbox report) is recorded in the runtime descriptor so the eventual scorecard's execution identity includes it.
1. **Assess.** From inside the same sandbox boundary, the agent probes I/O schema, modality, and action space of the intake-passed model, then *drafts* the `CapabilitySpec` + runtime descriptor. Authority stays with the kernel: the agent proposes, `bind`'s capability satisfaction disposes, and the user confirms before the first scored run.
2. **Select.** The agent maps confirmed capabilities to the catalog — interface satisfaction first, then modality, difficulty ladder, `phi_class`, and budget — proposes an eval plan (shelves, tasksets, trial counts), executes it on the N8 fleet, and narrates the resulting vectors honestly: gates and abstention first, per-world rows, never an invented composite.
3. **Adapt.** The agent searches the task's declared `ScenarioSpec`/`PerturbationSpec` space — and later proposes new scenario packages — to find where the model degrades: adaptive stress testing toward the hardest honest test, not the flattering one.

Invariants that keep the concierge from destroying the product it sells:

- **No in-place mutation, ever.** Any agent-adapted gym is frozen into a new versioned, digest-pinned task/scenario package *before* a single scored trial. Results name the exact package; replay works identically for agent-authored and human-authored packages.
- **Provenance and quarantine.** Agent-generated packages carry `authored_by: agent` provenance and are excluded from public leaderboards unless promoted through the same Tier-1 conformance as any wrap.
- **Verifiers are off-limits.** The concierge can author new task/scenario packages; it can never edit the verifier, gates, or projection of an existing published task.
- **Tenant weights under B3 gates** (ASSESSMENT §6.2): never in our training set, one-time callback credentials, artifact head pinned outside tenant reach.

This completes the open-core logic: N10 makes local eval free and easy, so the paid product is judgment plus fleet — an agent that knows the whole catalog, your model's capabilities, and the perturbation space, with every conclusion still independently replayable on the open rails.

### Non-goals (restated)
- No general-purpose physics engine, no dVRK/laparoscopic sim of our own, no asset marketplace (NVIDIA's game; wrap it).
- No generic RL-infra or compute marketplace (Prime's game; publish on it).
- No `reward.txt`-primary interface, in any bridge or export.
- No named-human credentialing motion until Future C's Phase-0 gates clear.
- No Tier-2 build that fails the §2.3 three-condition rule.
- No agent-mutated world is ever scored in place: freezing into a versioned, digest-pinned package precedes the first trial (N11 invariant).

## 6. Kill criteria

- If after N1 + N4 (first three wraps) no external team has run a package unchanged **and** asked for a second run, the commercial thesis fails; revert to Future A (research harness, tasks as papers) deliberately.
- If design partners consistently ask for training throughput and never for attestation, the wedge is the wrap kit + verifiers/Isaac bridges + catalog, and attestation becomes the enterprise upsell — reprice, don't deny.
- If a wrapped env's community adopts the vector contract upstream (best case), double down on that shelf; if every upstream PR is rejected, the catalog is a fork-maintenance business — cost it explicitly before N4 completes.

---

## Appendix A — Sources and as-of caveats

Competitor claims in this document describe **public product surfaces as of 2026-08** and are strategic premises, not audited facts. Environment names, licenses, and maintenance status in §2.4 must be re-verified per env before catalog commitment.

- **[A1] Harbor:** container-based harness for evaluating/optimizing agents and LMs; ships with Terminal-Bench; parallel evaluation via providers (Daytona, Modal); rollout generation for RL/SFT. Sources: github.com/harbor-framework/harbor · tbench.ai/news/announcement-2-0 · pypi.org/project/harbor · harborframework.com/docs.
- **[A2] Prime Intellect Environments Hub:** community hub for RL environments on the `verifiers` library; "2,500+ community environments" figure from third-party coverage of the hub (zylos.ai RLaaS report, 2026-07; rl-list.com vendor profile) and Prime's own announcements (primeintellect.ai/blog/environments); treat the count as order-of-magnitude.
- **[A3] NVIDIA Isaac for Healthcare:** medical-robotics development platform on Isaac Sim/Omniverse/Holoscan; ORBIT-Surgical transitioned into it; workflows include telesurgery, surgical subtask automation, and ultrasound. Sources: developer.nvidia.com/isaac/healthcare · developer.nvidia.com/blog/introducing-nvidia-isaac-for-healthcare · orbit-surgical.github.io (arXiv:2404.16027).
- **[A4] Open medical RL environments (survey detail in Appendix B):** CathSim (arXiv:2208.01455; github.com/airvlab/cathsim) · stEVE / stEVE_bench / stEVE_training (github.com/lkarstensen/stEVE; arXiv:2410.01956) · SurRoL / SurRoL-v2 (arXiv:2108.13035; med-air.github.io/SurRoL) · LapGym / sofa_env (arXiv:2302.09606; github.com/ScheiklP/lap_gym) · SurgicAI (arXiv:2406.13865; github.com/surgical-robotics-ai/SurgicAI) · SonoGym (arXiv:2507.01152; sonogym.github.io) · Surgical Gym (arXiv:2310.04676; github.com/SamuelSchmidgall/SurgicalGym) · AMBF-RL / Surgical Robotics Challenge (surgical-robotics-ai.github.io) · SofaGym (github.com/SofaDefrost/SofaGym) · VR-Caps (github.com/CapsuleEndoscope/VirtualCapsuleEndoscopy) · dVRL (github.com/ucsdarclab/dVRL; arXiv:1903.02090) · ROOM continuum sim (arXiv:2509.13177) · AI-Co-Pilot-Bronchoscope-Robot (github.com/LiuLiluZJU) · spine pedicle-screw safe-RL (arXiv:2305.05354). Licenses and maintenance status intentionally unverified here; verification is step one of each N4 wrap.
- **Absence claims** ("does not offer neutral cross-vendor attestation," "no gate/metric separation") mean *not evident in the public surfaces above as of 2026-08*, not proven absence.
- **Repo-grounded claims** (kernel features, bridges, gaps N-list) cite `docs/ASSESSMENT.md`, `docs/BUILD.md`, `docs/V0.3.md`, `src/or_audit/eval/sim/`, and are auditable in-tree.

## Appendix B — Environment survey and wrap dispositions (as of 2026-08)

Disposition legend: **WRAP** = v0.2 Tier-1 target (N4). **WATCH** = revisit on demand or upstream maturity. **SKIP** = reason given; not a shelf item.

| Env | Domain | Engine | Disposition | Rationale / risks |
|---|---|---|---|---|
| **stEVE** (+`stEVE_bench`, `stEVE_training`) | Endovascular guidewire/catheter | SOFA + BeamAdapter | **WRAP** | Purpose-built modular env toolbox *with a published benchmark repo* — gate mapping starts from existing task definitions. Direct Lumen overlap is the point (§2.6 shelf). Risk: SOFA determinism class. |
| **CathSim** | Endovascular | MuJoCo | **WRAP** (license-gated) | Most-cited endovascular RL sim; completes the three-engine endovascular shelf. Hard gate: license audit first — this repo already firewalls CathSim contamination for Lumen; the wrap must not breach that firewall (isolated adapter package, no asset reuse). |
| **ORBIT-Surgical** (now in Isaac for Healthcare) | dVRK manipulation (needle handover, lift, reach, suturing subtasks) | Isaac Lab / PhysX | **WRAP** (= N2) | Highest-leverage single wrap: inherits NVIDIA's assets and active maintenance. Risk: Isaac version churn — pin aggressively. |
| **SurRoL / SurRoL-v2** | dVRK manipulation, 14 tasks | PyBullet | **WRAP** | Most-adopted dVRK RL platform; `GymnasiumBridge` handles PyBullet today. Risk: maintenance slowing; expect dependency-pin work (the bit-rot value proposition in practice). |
| **LapGym / sofa_env** | Laparoscopic (12 envs: rope, tissue retraction, ligating loop, …) | SOFA | **WRAP** | Parameterizable, well-documented, JMLR-published; `sofa_bridge` seam exists. Safety state (tissue force) partially exposed — expect upstream PRs. |
| **SurgicAI** | dVRK fine-grained suturing subtask hierarchy | AMBF | **WRAP** | NeurIPS 2024 D&B: arrives with metric definitions and subtask decomposition — the easiest honest gate mapping in the survey; also our AMBF-ecosystem entry point. |
| **SonoGym** | Robotic ultrasound (spine; US-guided surgery) | Isaac Lab | **WRAP** | NeurIPS 2025, freshest codebase, GPU-parallel, rides the N2 Isaac bridge for near-zero marginal adapter cost; opens the ultrasound modality shelf. |
| **Surgical Gym** | dVRK/STAR arm control | Isaac Gym/Sim | **WRAP** (capacity) | GPU-parallel; low traffic and older Isaac Gym API — cheap only because it shares the N2 bridge; drop first under schedule pressure. |
| AMBF-RL + Surgical Robotics Challenge | dVRK (debris removal, needle passing) | AMBF | **WATCH** | Active academic community (challenge recurs); ROS-heavy runtime raises wrap cost. SurgicAI covers the AMBF path first; revisit if the challenge community shows pull. |
| SofaGym | Generic SOFA→Gym wrapper (soft robotics) | SOFA | **WATCH** (integrate, don't shelve) | Infrastructure, not a medical env: overlaps our `sofa_bridge`. Consider consuming it inside the bridge rather than listing it as catalog content. |
| VR-Caps | Active capsule endoscopy | Unity + ML-Agents | **WATCH** | Only open capsule-endoscopy env; Unity/ML-Agents runtime is heavy and the repo is old (2020). Wrap on the first design partner in GI. |
| ROOM | Bronchoscopy continuum robot | Physics-based (2025) | **WATCH** | Young (2025); if it matures it may beat building first-party bronchoscopy — re-run the §2.3 build-vs-wrap test before starting Tier-2 airway work. |
| AI-Co-Pilot-Bronchoscope-Robot | Bronchoscopy | Bespoke sim | **WATCH** | Open training code but bespoke, thinly documented sim; evidence for the "thin bronchoscopy landscape" claim rather than a shelf item yet. |
| BronchoCopilot | Bronchoscopy | Bespoke sim | **WATCH** | Paper-first; public code availability unclear — verify before any commitment. |
| Spine pedicle-screw safe-RL (arXiv:2305.05354) | Orthopedic drilling | Bespoke | **WATCH** | The one orthopedic candidate found; code availability unverified. Orthopedics otherwise has no open env — a future Tier-2 conversation, but outside our §2.3 competence test (rigid, not continuum). |
| dVRL | dVRK reach/pick | V-REP/CoppeliaSim | **SKIP** | Historically first (2019) but superseded by SurRoL on the same tasks; V-REP runtime not worth a shelf slot. Credit it in the shelf's lineage notes. |
| RL_cataract, surgeon-in-the-loop ophthalmic apprentice | Ophthalmic | Unity/bespoke | **SKIP** (for now) | One-off research artifacts, unmaintained; no credible open ophthalmic env exists — recorded as a genuine coverage gap, not a wrap target. |

Survey method and caveat: web survey of published env/benchmark repos and papers as of 2026-08 (sources in [A4]); small-traffic academic envs may be missing — the N3 wrap kit plus a public "request a shelf" intake is the structural answer to long-tail discovery, not a bigger one-time survey. Every disposition above is re-testable; licenses, maintenance, and safety-state availability get verified as step one of each wrap.
