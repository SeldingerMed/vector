# Vector implementation roadmap and status

## Roadmap charter — 2026-09-08

Vector is independent evaluation and verifiable-reward infrastructure for
procedural medical imaging, robotics policies, and learned world models. It is
not a replacement for ManiSkill, Isaac Lab, LeRobot, MONAI, or an RL trainer.
Those stacks supply simulation, datasets, models, and learning; Vector owns
task binding, independent scoring, safety vectors, evidence, and scoped claims.

**Next objective:** one real, reproducible benchmark that an external team can
run unchanged and wants to run again. More catalog entries, schemas, and cloud
features do not satisfy that objective by themselves.

This document is the implementation plan for workstreams **A–H** below. It does
not report those workstreams as implemented. The historical N1–N11 ledger and
audit findings are retained after the roadmap. An old `shipped` designation
applies to its named mechanism, not to every new acceptance criterion here.

### Implementation status — 2026-09-09

Work is merged per phase as PRs; nothing below claims a workstream finished
unless every acceptance criterion is met. Partial means shipped mechanism
with named gaps; blocked means a non-code prerequisite (partner, hardware,
data, or product decision).

| ID | Status | Shipped (PRs) | Explicitly remaining |
|---|---|---|---|
| A | partial | Brief, runnable lumen task v1, strict verifier, real A3 comparison | A3 trained-checkpoint comparison; A5 external reproduction (blocked: partner) |
| B | partial | Threat model, env scrubbing, container backend + CI image job, bounded transfer, staging tests, attestation contract | B5 cloud mint/storage (blocked: cloud owner); registry image publication (commercial decision) |
| C | partial | Dossier, cited 0.3mm gate, orphan-label refusal, bench provenance, harness-fault robustness | C4 world-native distributions (need SOFA/GPU world revisions); C5 phantom (blocked: partner) |
| D | partial | Paired comparison + CIs, cohort enforcement, compare CLI | D1 case manifests with patient/site grouping; D5 uncertainty views in scorecards (no dataset carries the metadata yet) |
| E | partial | Obs/action contract test, interactive streams, LeRobot reader, MONAI delegation, honest video adapter | E4 media alignment (no test media; mp4 banned from VC); semantic output schemas (needs E contract design) |
| F | partial | Prefix-replay branching proof + seed caveat | F1/F2 trajectory-backed scoring (needs imported trajectory benchmark); F4/F5 planning utility (blocked: no learned dynamics model) |
| G | partial | SB3 PPO recipe with measured train→evaluate result | Verifier-derived training rewards (blocked: divergence unobservable, projection withdrawn); Prime interop |
| H | partial | Episode resume + crash-safe writes + partial recovery + CLI flag | Fleet queue/autoscaling (blocked: hosted capacity decision); vectorized stepping (not justified by measured need) |

### Status and evidence rules

- **Existing:** found in the reviewed source; runtime verification is stated
  separately. Existing source is not proof of production operation.
- **Planned:** specified below, not delivered by this documentation change.
- **Blocked:** requires a named artifact, supported host, permission, or external
  participant. Record the exact blocker; never replace it with a stand-in.
- **Accepted:** implementation plus the workstream's acceptance evidence exists.
  Record package versions, source revision, command, host/runtime, artifact head,
  reviewer, and claim limits. A passing unit test alone cannot accept a real-world
  integration or scientific claim.

Review evidence (2026-09-08): the CPU `video-nextstep` quickstart ran, and
`surgeval replay` matched the recorded job head (see `docs/ONRAMP.md` for the
exact commands, the recorded head digest, and what this run does and does not
exercise). It is a fixture-backed reference, not evidence of real video inference.
The world catalog command ran; catalog eligibility does not establish installed
simulator availability. Cloud orchestration source exists in a separate private
repository (mounted internally at `.projects/vector-cloud/`), but its deployment
was not exercised from this repo. Do not infer hosted deployment status or
adoption metrics from the public tree.

### Primary-source comparison map

Upstream documentation was reviewed on 2026-09-08. These are design references,
not compatibility pins. Each implementation must select and record an exact
supported dependency version/revision before claiming interoperability.

| Reference | What to reuse or match | What it does not establish for Vector |
|---|---|---|
| [ManiSkill GPU simulation](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/gpu_simulation.html) | Batched environment identity, reset/reconfiguration lifecycle, controller/physics stepping | Medical validity, safe cross-environment aggregation, or isolation between sub-scenes without checking spacing |
| [ManiSkill observations](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html) | Shapes, dtypes, units, RGB-D, proprioception, camera calibration, privileged-state distinction | That matching tensor shapes implies matching physical meaning |
| [LeRobotDataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) | Parquet/MP4 episodes, timestamps, feature schemas, normalization metadata, temporal windows | Unobserved counterfactual ground truth or patient-disjoint evaluation |
| [LeRobot](https://github.com/huggingface/lerobot) | Existing policy/data tooling and checkpoint workflows | A medical safety certificate or an independent held-out verifier |
| [robomimic datasets](https://robomimic.github.io/docs/datasets/overview.html) | HDF5 demonstrations and split/filter conventions where a partner already uses them | A reason to introduce another native dataset format |
| [Gymnasium environment API](https://gymnasium.farama.org/api/env/) and [VectorEnv](https://gymnasium.farama.org/api/vector/) | Observation/action spaces, reset/seed, terminated/truncated and vector lifecycle contracts | Statistical independence or bitwise GPU determinism |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | Optional simulator/controller integration and randomized scenarios | Tissue-valid dynamics merely because a backend is physically simulated |
| [MONAI metrics](https://monai.readthedocs.io/en/stable/metrics.html) | Established imaging metric implementations inside task-owned verifiers | Clinical usefulness of an arbitrary selected metric |
| [Harbor tasks](https://harborframework.com/docs/tasks) | Resource/network policies and separate verifier environments | Continuous-control, sensor timing, or medical physics semantics |
| [Prime verifiers](https://github.com/PrimeIntellect-ai/verifiers), [v1 environment contract](https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/docs/v1/env.md), [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) | Version-tested environment/trainer interoperability and isolated scoring patterns | A suitable trainer for every continuous-control policy |

### Workstream index and sequencing

| ID | Deliverable | Baseline → missing result | Dependencies |
|---|---|---|---|
| A | Flagship benchmark and external reproduction | Reference packages/catalog → real models on a useful fixed benchmark | A discovery starts now; acceptance consumes B–E and H scalar reliability |
| B | Adversarial agent/verifier isolation | Subprocess protocol separation → enforced oracle/evidence boundary | Can proceed with A/E; required before untrusted submissions |
| C | Benchmark and safety validity | Signal audit/equivalence machinery → supported thresholds and external evidence | A task choice and E signal semantics; physical study needs partner |
| D | Splits and statistical comparison | Descriptive scorecards → independent-case, paired, uncertainty-aware evaluation | A cases and E dataset identity |
| E | Multimodal data and real policy contracts | Generic adapters → tested ManiSkill/LeRobot-style integration | Starts with A; provides contracts used by F/G/H |
| F | Multi-horizon world-model evaluation | Fixture intervention ranking → trajectory/branch-backed forecasts | A/E data, C oracle limits, D splits |
| G | Training-to-independent-evaluation proof | Existing exports → pinned trainer, actual update, held-out checkpoint result | B boundary, A/E task-policy pair, D holdout |
| H | Reliable scalar execution, then measured batching | Cloud/bridges exist → resumable trials and optional per-env batching | Scalar reliability starts now; batching follows E and a measured need |

Delivery order is not alphabetical. Start **A selection + E real policy/data
path + H scalar reproducibility**, with B in parallel. Add D and C benchmark
protocols before interpreting comparisons. Complete an external reproduction,
then F/G expansion and H batching where justified. C's physical validation may
remain explicitly blocked without blocking a narrowly claimed simulator release.

Integration owner: the harness maintainer owns shared task/trace/job contracts.
Domain owners author task packages, adapters, and verifiers; runtime owners own
B/H; benchmark owners own A/C/D; trainer integration owners own G. These are
responsibilities, not claims that people have been assigned. Each implementation
change must name an actual owner and keep cross-cutting schema changes serialized.

## A — Flagship benchmark with real models and independent reproduction

**Status: planned; packaging, examples, registry and shelf machinery existing.**

### Comparison and scope

ManiSkill makes concrete tasks, observation modes and controllers executable;
LeRobot connects concrete datasets to policies. Vector must offer that same
practical completeness for one procedural-medical task, then add independent
scoring. A world catalog or lookup-table predictor is not the equivalent.

Default focus: the existing endovascular direction, subject to resolving actual
world license, install and signal blockers. Do not call Lumen turnkey until its
recorded blockers are closed. If no lawful supported runtime is available, record
the blocker and choose a real imaging benchmark explicitly; do not silently
substitute a toy task or claim an endovascular milestone was completed.

### Implementation

1. **A1 — Freeze the benchmark brief.** Name the user decision, task population,
   supported modality, simulator/dataset revision, intended claim, exclusion
   criteria, independent case unit and compute envelope. Select one task family,
   not a multi-modality launch. Record data/runtime terms and acquisition steps.
2. **A2 — Deliver a real runnable package.** Extend existing
   `docs/examples/tasks/`, agent packages, registry publication and world catalog
   paths. Pin model weights, preprocessing/controller configuration, task,
   verifier, world/assets and dependency environment. Use E's policy contract.
   Provide a clean supported Linux/GPU host path when macOS cannot run the world.
3. **A3 — Publish comparison content.** Include a transparent simple baseline
   and at least two meaningful model/policy checkpoints. Freeze D's held-out
   cases and report per-case safety, success, abstention and failure evidence.
   Random policies are sanity checks, not the only competing models.
4. **A4 — Make results inspectable.** Extend existing scorecard/artifact surfaces
   with links to the exact observation/action window behind a failure, within
   dataset access permissions. Distinguish re-scoring stored evidence from
   simulator re-execution; never imply that `replay` currently re-steps physics.
5. **A5 — External reproduction.** Publish versioned installation and run
   instructions, supported hardware, expected cost range measured from the run,
   and troubleshooting based on actual failures. Ask an independent team to run
   the immutable package without changing harness source and request a second
   useful comparison. Record their artifact and discrepancies, not a user-count
   assertion inferred from documentation.

### Acceptance and evidence

- Clean-host install and real checkpoint inference complete with no stand-in.
- All model comparisons use the same frozen held-out case manifest.
- Stored vectors reconstitute; re-execution matches the measured tolerance/class
  where supported, with environment differences disclosed.
- At least one external run is independently reproduced. A second-run request
  is product evidence; it cannot be supplied by adding tests or telemetry code.
- Publish benchmark version, supported claims, failures and limitations. A is
  not accepted by adding catalog rows, synthetic clips or checkpoint filenames.

## B — Adversarial isolation and independent evidence ownership

**Status: planned; subprocess/API separation and intake controls existing.**

### Comparison and scope

ManiSkill distinguishes privileged simulator state from sensor observations;
LeRobot stores labels, states and actions together for learning convenience.
Neither convention is an adversarial evaluation boundary. Vector must prevent
an evaluated agent from reading privileged state or held-out labels just because
they are present on the same host. Borrow Harbor/Prime's separate-verifier
execution pattern, not an assumption that a Gym wrapper provides security.

`src/or_audit/eval/plugins.py` currently uses ordinary `subprocess.Popen`.
Withholding labels from the request protects the protocol, not the filesystem.
A Machine0 VM per run isolates runs, not agent and verifier within a run.
Container/HF/API runtime descriptors are not all locally executable today.

### Implementation

1. **B1 — Freeze the threat model.** Identify untrusted agent code/weights,
   task-author trust, oracle data, simulator truth channels, runtime credentials,
   verifier code and publication authority. Document that malicious benchmark
   authors require governance/review; sandboxing an agent does not validate a
   dishonest task verifier.
2. **B2 — Implement one enforceable runtime.** Extend existing runtime contracts
   and plugin loading with one supported container/sandbox backend. Separate
   agent and verifier identities/filesystems; mount agent inputs read-only;
   exclude labels, private state, host sockets and service credentials. Keep
   local subprocess mode explicitly trusted-development-only for hostile-code
   claims. Reuse the cloud run lifecycle rather than creating another scheduler.
3. **B3 — Define evidence transfer.** Have the harness/world capture authoritative
   transitions and signals outside agent-writable storage. Transfer only
   manifest-declared outputs through bounded, validated artifacts into a fresh
   verifier environment. Recheck digests after transfer; reject symlink/path
   escape and undeclared inputs. Agent-produced predictions remain untrusted data.
4. **B4 — Enforce resources and networking.** Apply CPU/RAM/GPU/time limits and
   explicit egress policy. Give endpoint policies only the credentials and hosts
   needed for inference. Record enforcement capability and refuse an unsupported
   policy instead of treating it as applied. Preserve legitimate GPU access
   without exposing host control sockets.
5. **B5 — Cover replay and publication.** Execute verifier code during replay
   under the same declared trust model. Keep verifier-owned results outside agent
   write access. Distinguish a content digest from authenticated publisher or
   execution provenance; use existing release attestation mechanisms where they
   apply rather than inventing an in-band signing key.

Primary code seams: `src/or_audit/eval/plugins.py`, `src/or_audit/eval/plugin_host.py`, runtime contracts,
runner/reconstitution, concierge intake and the separate cloud executor.

### Acceptance and evidence

- Adversarial agents cannot read a private-label canary, alter world evidence or
  verifier code, access another run, exfiltrate through denied egress, or publish
  a forged authoritative vector. These probes must run in the actual backend.
- A legitimate GPU policy and verifier complete with declared artifact transfer.
- Timeouts/OOM/transfer failures become explicit failed or unassessable outcomes,
  never successful safety results. Cleanup leaves no active runtime credentials.
- Existing trusted local workflows remain available without being advertised as
  adversarial isolation. This work does not itself establish PHI compliance.

## C — Scientific validity, safety thresholds and sim-to-real protocols

**Status: planned evidence; signal audits and equivalence machinery existing.**

### Comparison and scope

ManiSkill's physical simulation and randomized scenes offer experimental inputs;
LeRobot's real trajectories offer external observations. Neither makes a generic
contact force a tissue-injury threshold or proves transfer. Vector's additional
responsibility is to bind each claim to evidence and preserve unsupported states.

Use `src/or_audit/eval/equivalence.py`, audited catalog signals, gate mapping and existing
shelf refusals. Do not replace them with a general cross-world leaderboard.

### Implementation

1. **C1 — Publish measurement dossiers.** For each flagship gate/metric record
   signal source at the pinned revision, unit, frame, sample rate, calibration,
   missing/non-finite behavior and applicability. Separate physical measurement,
   geometric proxy, diagnostic flag and clinical endpoint. Preserve the existing
   prohibition on fabricated physical signals.
2. **C2 — Establish threshold provenance.** Add benchmark-side evidence for
   threshold source, population/material, measurement setup, uncertainty and
   limitations. Classify empirical, literature-derived and engineering limits
   explicitly; an engineering limit must not be labelled clinically validated.
   Check threshold sensitivity on held-out cases without tuning on final test
   outcomes. Retain unit/predicate/citation consistency checks.
3. **C3 — Validate oracle labels.** Where human annotation is required, specify
   expertise, blinding, annotation protocol, adjudication, disagreement and
   provenance. Reuse applicable agreement utilities; do not treat the older
   credentialing path as automatically wired into model evaluation.
4. **C4 — Define robustness scenarios.** Freeze task-level distributions for
   anatomy/geometry, friction/compliance, camera conditions, latency and sensor
   noise only where supported by the selected world. Record realized values,
   not just a randomization label. Compare nominal and held-out regimes; do not
   rebuild ManiSkill/Isaac randomization engines in the kernel.
5. **C5 — Run paired external validation.** With a physical partner, predeclare
   simulator-to-phantom mapping, controllers/calibration, cases, outcome measures,
   sample-size rationale, uncertainty and failure criteria. Evaluate matched
   policies and compare rankings and absolute outcome discrepancies. Record
   domain gaps even if ranks agree. Use the existing EquivalenceArtifact only
   for precisely covered worlds, gates, units and task family.

### Acceptance and evidence

- Every asserted gate has a reviewed measurement/threshold dossier; unsupported
  channels remain metrics-only or unassessable under existing rules.
- A frozen robustness report includes realized parameters and failure examples.
- Scientific claims name their population and limits; threshold sensitivity and
  annotation uncertainty are visible rather than averaged into success.
- Sim-to-phantom claims require actual external paired data and a validated
  artifact. Until then C5 is blocked on the partner/data, and cross-world ranking
  remains refused. Simulation-only benchmark publication can proceed with that
  limitation prominently stated; no clinical certification claim follows.

## D — Independent-case splits, paired comparisons and uncertainty

**Status: planned; descriptive aggregates and case-count declarations existing.**

### Comparison and scope

ManiSkill parallel sub-scenes are execution units, not automatically independent
samples. LeRobot episode metadata gives boundaries, not patient/site-disjoint
splits. Vector must make the statistical unit explicit so adjacent frames and
repeated runs cannot inflate confidence.

Current `src/or_audit/eval/scorecard.py`, `src/or_audit/eval/job.py` and `src/or_audit/eval/leaderboard.py` provide descriptive
values; conformance pairing checks same-agent reproducibility, not model-vs-model
inference. `EvaluationStageSpec` already includes independent-case fields.
Extend those contracts rather than introducing a second case-count convention.

### Implementation

1. **D1 — Enforce split manifests.** Bind dataset/world revision, stable case ID,
   episode ID, pseudonymous patient/site grouping where relevant, split and
   scenario identity. Validate declared grouping against actual input rows.
   Reject prohibited group overlap; record when a dataset cannot support a
   patient/site-disjoint claim. Avoid publishing identifying metadata.
2. **D2 — Make scenario selection executable.** Replace reliance on descriptive
   `seed_policy` text with an explicit resolved case/seed schedule in the job.
   Persist realized initial states or their recoverable references and the
   randomization configuration. The schedule, not `range(n)` alone, defines a
   held-out scenario set. Preserve task-authored meaning and migrate callers.
3. **D3 — Add paired comparison.** Align models by frozen case/scenario IDs and
   document repeated policy seeds. Refuse mismatched cohorts or disclose a
   predeclared missing-case policy. Report task performance, safety outcomes and
   coverage separately; never discard failed trials to make pairing convenient.
4. **D4 — Implement uncertainty summaries.** Use an appropriate interval for
   binary rates and paired differences; cluster resampling at the declared
   independent unit for repeated frames/episodes. Pin method, confidence level,
   resampling seed and count. Report denominators and unassessable cases; zero
   observed failures is not a zero-risk claim. Prefer established statistical
   implementations where suitable, with benchmark-specific method review.
5. **D5 — Extend reporting.** Add uncertainty, coverage, subgroup and worst-case
   views to existing CLI/HTML/leaderboard artifacts. Report risk against coverage
   for abstaining models. Mark underpowered subgroups; freeze primary endpoints
   and distinguish exploratory comparisons to avoid selective reporting.

### Acceptance and evidence

- A patient/site overlap fixture is refused; splitting adjacent frames from one
  case does not increase the independent-case count.
- Paired identical models give zero paired differences; known small examples
  agree with a trusted statistical reference. Missingness is explicit.
- Job replay reproduces the comparison inputs and deterministic statistical
  output under its pinned method; non-comparable models are not silently ranked.
- A real A benchmark report contains independent-unit counts, intervals, gate
  outcomes, coverage and subgroup limitations. Old descriptive scorecards remain
  identifiable; no retrospective inference is fabricated from missing metadata.

## E — Multimodal observations, trajectory imports and real policy adapters

**Status: planned integration; generic contracts/modality adapters existing.**

### Comparison and scope

Match ManiSkill's concrete observation semantics: shapes/dtypes, sensor units,
camera transforms, proprioception and privileged state. Match LeRobot's temporal
episode access, timestamped actions/video and normalization metadata. Reuse their
formats; do not put large images into an ever-growing JSON protocol or invent a
new general-purpose dataset/training library.

### Implementation

1. **E1 — Specify the first domain profile.** Extend existing InterfaceSpec,
   StreamSpec, CapabilitySpec and modality adapters with the flagship's actual
   observation/action semantics. Record dimensions, dtypes, valid ranges, units,
   coordinate frames, invalid-depth encoding, camera calibration, joint ordering,
   privileged-vs-observable fields and controller identity. Equal shapes with
   different physical meanings must not bind as equivalent.
2. **E2 — Pin complete policies.** Bind weights plus preprocessing, normalization,
   history length, action scaling/clipping, absolute/delta command mode, control
   rate, action repeat and recurrent-state reset. Add one real robotics policy
   adapter and one real perception model adapter as their benchmarks are ready.
   Reuse installed model stacks; frozen prediction tables remain labelled demos.
3. **E3 — Implement read-only episode import.** Start with a pinned LeRobot
   dataset version; map Parquet/MP4 metadata to Vector case/episode/evidence
   references. Preserve revision, offsets, feature schema, statistics and license.
   Support robomimic HDF5 next only for a concrete partner dataset. Do not
   deserialize untrusted model objects as part of import.
4. **E4 — Resolve media deterministically.** Decode the needed windows through
   an optional media adapter; record decoder/version, sampling/interpolation,
   timestamps and alignment tolerance. Pin source media bytes and transformation
   identity. Keep bulk arrays/media in immutable artifacts or a backend-appropriate
   data plane; keep JSON for control and references. Missing frames and sensor
   dropout must be explicit, not silently forward-filled into confident evidence.
5. **E5 — Unify stream handling across modes.** Remove the interactive-stream
   restriction only after implementing the same pinned preprocessing and evidence
   behavior for interactive observations. Reuse the closed-loop adapter path;
   preserve terminal scoring context and keep oracle fields out of agent history.
6. **E6 — Supply verifier recipes.** Wrap existing MONAI metrics for the selected
   imaging task and add task-specific temporal/uncertainty scoring where needed.
   Bind spacing, label mapping and preprocessing so superficially equal metrics
   do not compare different definitions. DICOM/NIfTI support is conditional on
   an actual benchmark need, not a mandatory new subsystem.

Primary seams: `eval/adapters/`, contracts, runner stream preprocessing, agent
packages and task verifiers. Dataset-specific dependencies remain optional.

### Acceptance and evidence

- A real ManiSkill-style RGB-D/proprioception observation reaches a real policy
  with correct units/calibration; action interpretation matches a direct upstream
  policy run under the declared tolerance. No need to claim medical validity for
  this interoperability probe.
- A real LeRobot episode imports with verified boundaries and aligned temporal
  windows; no history window crosses episodes or exposes future frames.
- Wrong unit, controller mode, feature order or incompatible normalization is
  refused where the contract declares it; policy memory resets per episode.
- Interactive and closed-loop paths record the same transformation identities.
  Missing media gives a named failure, not fabricated observation data.
- At least one real A benchmark uses the path end to end; data handling and
  publication respect existing PHI/license restrictions.

## F — Multi-horizon, action-conditioned world-model evaluation

**Status: planned; single-decision counterfactual ranking existing.**

### Comparison and scope

LeRobot supplies observed histories/actions/futures; ManiSkill can supply
controlled simulator branches where reset/state restoration supports them.
These are different oracle classes. Recorded trajectories show only outcomes
under actions actually taken. They cannot label arbitrary unobserved alternatives.

Keep the existing counterfactual ranking mode valid; do not relabel it as a
multi-step dynamics evaluation. Extend task packages and reusable adapters before
adding another top-level harness mode or world-model trainer.

### Implementation

1. **F1 — Define a forecast task contract.** Specify history window, action
   sequence, forecast horizons in physical time, predicted state/media/event
   fields, uncertainty representation and oracle class. Pin time discretization,
   action rate and truncation policy using E contracts. Support a measured subset
   of outputs rather than claiming every latent representation is comparable.
2. **F2 — Implement trajectory-backed scoring.** Use E's LeRobot import on D's
   held-out episodes. Evaluate action-conditioned future predictions against
   observed future states/events; prevent future information in model inputs.
   Score by horizon, not only a final-step average. Use no-action/constant-state
   and simple dynamics baselines to expose trivial predictability.
3. **F3 — Implement simulator branching where valid.** Add an optional declared
   snapshot/restore capability to the world adapter only when it can restore all
   necessary simulator/controller/RNG state. Compare candidate sequences from a
   common initial state. If a backend cannot restore faithfully, use a validated
   deterministic reconstruction or refuse branch-equivalence claims. A seed alone
   must not be asserted to restore a full state.
4. **F4 — Score task-relevant prediction quality.** Report state/geometry error
   growth, safety-event precision/recall and calibrated probabilities, plus
   task-specific consistency checks. Include media similarity only as a secondary
   metric when relevant; visually plausible futures do not establish safe dynamics.
   Distinguish predicted safety from independently observed executed safety.
5. **F5 — Test planning utility.** Hold planner, action budget and starting cases
   fixed; swap learned dynamics models and evaluate chosen actions in the real
   pinned simulator. Compare a simple planner/model baseline and disclose oracle
   access. Report actual task/safety results separately from forecast metrics.

Primary seams: counterfactual task/verifier examples, runner prediction context,
world capability registration, simulation adapters and temporal evidence types.

### Acceptance and evidence

- One real checkpoint is scored at multiple declared horizons on held-out data.
- Missing futures/early termination have a declared denominator policy; windows
  neither cross episode boundaries nor leak future observations.
- Branch repeatability is measured for a supporting simulator; unsupported
  snapshot backends refuse rather than silently approximating equivalence.
- A planning-utility report compares measured executed outcomes, including
  failure cases where plausible forecasts lead to bad actions.
- Results explicitly distinguish observational forecasting, simulator
  counterfactuals and real-world causal evidence. None implies the others.

## G — Version-pinned training interoperability and held-out policy evaluation

**Status: planned proof; `export-rl` and `export-verifiers` existing.**

### Comparison and scope

LeRobot and ManiSkill already connect policies, trajectories and learning stacks.
Vector should supply an independently derived reward/evaluation contract, not
compete with their trainers. Prime is appropriate for supported LLM/VLM workflows;
an established robotics trainer is the default for continuous control.

The generated export currently uses `load_environment`, optional
`verifiers.Rubric`, and `verifiers>=0.1`. Current upstream also documents a v1
taskset/environment architecture. This is a compatibility question requiring a
real integration run, not evidence that the existing export is broken.

### Implementation

1. **G1 — Establish supported versions.** Select exact tested Vector, trainer,
   verifiers/LeRobot/robotics-library, model and simulator versions. Exercise
   loading before expanding the compatibility claim. Update generated dependency
   bounds/entry points based on observed support and upstream contracts; migrate
   every generated caller when changing an interface.
2. **G2 — Keep the safety projection authoritative.** Reuse existing export and
   projection code. Each reward must reference the freshly scored parent vector,
   task/world/verifier/projection identities and policy version. Preserve refusal
   of fabricated physics and existing gate-failure behavior. Expose richer costs
   only through an explicit supported contract, never by silently redefining the
   scalar reward. A zeroed reward is not a proof of constrained-policy safety.
3. **G3 — Publish one real training recipe.** Run a selected continuous-control
   policy through a supported trainer using the exported verifier-derived signal.
   Record initial and updated checkpoint digests, optimizer update evidence,
   training cases and resource usage. Add the Prime path separately for a real
   compatible language/vision task; do not force token-based RL onto robot actions.
4. **G4 — Evaluate the resulting checkpoint independently.** Freeze D's holdout
   and B's oracle boundary. Use E's complete policy package to evaluate before
   and after training on identical cases, reporting success, gates, coverage and
   uncertainty. Improvement is not required to prove the integration; an actual
   update and honest held-out result are required.
5. **G5 — Probe reward exploitation.** Include adversarial predictions/actions,
   abstention abuse, missing-signal behavior and attempts to forge reward records.
   Keep train-time shaping and held-out evaluation definitions distinct and
   versioned. Any asynchronous recipe records behavior-policy identity and uses
   the trainer's off-policy machinery rather than inventing importance correction.

Primary seams: `eval/export_rl.py`, `export_verifiers.py`, projection/vector
contracts, generated packages, real agent recipes and existing export tests.

### Acceptance and evidence

- A generated package loads in each advertised pinned trainer version.
- A real run changes policy parameters; rewards recompute from the authoritative
  vectors and carry their provenance. Mock rubric calls do not satisfy this.
- The saved checkpoint independently evaluates through Vector, with complete
  preprocessing/controller state and an untouched held-out split.
- Report performance regressions and reward exploits as results. Do not weaken
  gates or select a new holdout to manufacture a successful training narrative.

## H — Reliable execution, episode recovery and optional vectorized evaluation

**Status: planned extensions; scalar bridges and cloud orchestration existing.**

### Comparison and scope

ManiSkill's batched stepping and LeRobot's episode-indexed storage motivate
throughput without losing episode boundaries. Vector currently refuses batches
in `src/or_audit/eval/sim/base.py`; retain that correct refusal until a real batch contract exists.
The cloud already implements Machine0 provisioning, resource selection and some
reconciliation/cleanup retries. The missing work is reliable evaluation recovery
and measured batching, not a replacement cloud control plane.

### Implementation

1. **H1 — Reproduce the scalar path first.** Build a supported-host matrix for
   the selected real world/model. Pin runtime images/dependencies/assets and
   verify actual GPU/simulator/sensor access. Run conformance on that host, not
   only injected factories. Record runtime and install failures as blockers.
2. **H2 — Persist explicit trial lifecycle.** Extend job/runner artifacts with
   stable case, seed, attempt and policy identities plus pending/running/completed/
   failed states. Write completed evidence atomically and verify it before reuse.
   Resume at an episode boundary first; mid-episode continuation is supported only
   with validated F snapshot state including policy/controller/RNG state.
3. **H3 — Make recovery auditable.** Integrate with existing cloud reconciliation.
   Preserve failed attempts, distinguish infrastructure failure from task outcome,
   cap retries by declared policy, and never retry away an unsafe result. Resolve
   duplicate submissions/completions by stable identity; meter logical outcomes
   and real compute without double-counting a retry as another independent case.
   Verify VM cleanup after failures and interrupted publication.
4. **H4 — Measure the bottleneck.** Publish scalar throughput, startup time,
   policy inference/physics/rendering/scoring time, artifact I/O and memory on the
   actual benchmark. Prefer independent scalar workers if they meet the workload.
   Do not allocate a GPU batch subsystem merely because ManiSkill has one.
5. **H5 — Add a real vector contract if justified.** Introduce explicit per-env
   observations, actions, terminated/truncated flags, info, seeds, reset masks,
   terminal observations and trial identities. Integrate one real ManiSkill or
   supported vector backend first. Keep data on device where practical; transfer
   bounded evidence without serializing every image to JSON. Never select env 0,
   average gates, or broadcast one environment's failure to all environments.
6. **H6 — Validate lifecycle equivalence.** Compare batch-size-one and batched
   execution on the same resolved scenarios. Measure tolerances instead of
   promising bitwise equality. Check partial reset, recurrent policy reset,
   auto-reset semantics, final observation capture and truncation precedence.
   Exercise sub-scene spacing/collision isolation; ManiSkill explicitly warns
   that nearby sub-scenes can physically interact. Record any non-equivalence.

Primary seams: `eval/sim/base.py` and bridges, `runner.py`, job/cartesian
execution, conformance, artifact storage and `.projects/vector-cloud/` lifecycle.

### Acceptance and evidence

- A real scalar GPU/simulator run completes on a clean supported host with
  measured conformance and trace reconstitution.
- Killing a worker after a completed episode and resuming preserves completed
  evidence, records attempts and produces no lost/duplicated logical trials.
- An unsafe outcome is not erased by retry; corrupt partial artifacts cannot be
  accepted as completed results. Cloud teardown and billing behavior are exercised
  in an authorized test environment, not inferred from source.
- Batching is accepted only with per-env safety/termination parity, no reset-state
  leakage, and a published throughput/memory comparison. The scalar refusal stays
  for unsupported bridges. No mandatory speedup threshold is invented before H4.

## Milestone gates and completion ledger

All milestones below are **planned**. An implementation PR updates this ledger
with evidence and the exact A–H substeps it accepts; it does not mark an entire
workstream complete because one schema or demo merged.

| Milestone | Required outcome | Release boundary |
|---|---|---|
| M1 — Real executable path | A1–A2, E1–E4 for one policy/task, H1 | Trusted-development benchmark preview; no untrusted-code or clinical claims |
| M2 — Defensible comparison | B1–B5, C1–C4 as applicable, D1–D5, A3–A4 | Reproducible held-out benchmark with explicit validity limits |
| M3 — Independent usefulness | A5 and H2–H3 | External unchanged-package reproduction and evidence of a requested second use |
| M4 — Forecast and learning loops | F1–F5 where oracle capabilities exist, G1–G5 | Real world-model and training examples with independent held-out results |
| M5 — Scale and transfer | H4; H5–H6 only if justified; C5 with external data | Measured throughput and narrowly supported transfer/comparability claims |

M4's trajectory-only forecast release may precede simulator branching if it is
explicitly labelled observational; F as a whole is not then complete. E5/E6 ship
with the first interactive/imaging benchmark that consumes them, and remain open
until exercised. No milestone waives PHI, license, runtime or safety refusals.

### Verification and documentation policy

- Extend existing behavioral tests at changed contracts; avoid tests that only
  assert source text, mocked plumbing or current incidental defaults.
- Use real supported simulator/model/data smoke runs for integration acceptance.
  GPU/vendor-only checks run on their supported host, with artifacts attached.
- Update affected CLI help, task/agent examples, ONRAMP, CONFORMANCE and DATASETS
  documentation with each behavior change. Keep the wheel's CPU quickstart fast
  and honest; add real optional paths rather than making the core depend on CUDA.
- Maintain the historical refusal invariants below. Legacy scalar behavior must
  migrate through explicit contracts, never through silent reductions or aliases.
- Keep product names and execution claims consistent: Vector is the public name,
  `surgeval` the current distribution; source uses `or_audit`. Update stale claims
  as touched, including RunPod language where the actual cloud path is Machine0.
- Do not add a new simulator, trainer, universal dataset format, certification
  scheme, or scheduler to satisfy a comparison checklist. Reuse upstream systems
  and implement the minimum independently verifiable integration.

## Historical N1–N11 implementation ledger

The following records the prior implementation status and audit evidence. Read
historical counts and host blockers as dated observations, not live telemetry.
> Update 2026-09-08: N8/N11 below are split into “source exists” vs
> “deployment/compliance unverified.” The hosted orchestration source lives in
> the separate private `SeldingerMed/vector-cloud` repository (mounted locally
> at gitignored `.projects/vector-cloud/`, never in the public git tree). The
> older “outside this public harness” wording meant outside the public tree,
> not nonexistent — it is superseded by the split status in those rows.

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
| N8 — sellable surface | external (hosting source exists; deployment/compliance unverified) | Orchestration source exists in the separate private `SeldingerMed/vector-cloud` repo (gitignored `.projects/vector-cloud/` locally): Machine0 run lifecycle, CPU/GPU selection, R2 artifacts, auth, billing. Unverified from this repo: production deployment, tenancy/BAA posture, pricing. Pricing, tenancy, and BAA remain commercial decisions, gated behind N1 evidence per next.md's own ordering |
| N9 — sim-to-phantom correlation | partial (machinery only) | `src/or_audit/eval/equivalence.py` implements the artifact, its four requirements, and the rank-correlation check against an external referent. The phantom study itself is physical work: no artifact is published, and the code refuses cross-world claims until one is |
| N10 — Apache-2.0 open-core distribution | shipped | `src/or_audit/install/` (catalog, installer, doctor), `surgeval quickstart` / `worlds` / `doctor`, `.github/workflows/install-smoke.yml`, wheel-packaged reference examples |
| N11 — hosted agentic concierge | shipped (rails), external (hosting deployment unverified) | `src/or_audit/concierge/{intake,assess,select,adapt}.py` + `surgeval concierge`. The deterministic machinery and every invariant ship in the public tree; hosted execution lives in the private `SeldingerMed/vector-cloud` repo and its live deployment is unverified |

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
  `stepped_world` switch that scopes this rule is required (no default), is
  derived from the runner's actual route (`interaction_mode is closed-loop`)
  and cross-checked against the installed adapter's declared `closed_loop`
  capability, so a report cannot reclassify its world to waive the
  requirement, and a kind with no installed adapter cannot reach Tier 1 at
  all. It is deliberately not derived from `physics`, which was a proxy that
  let a non-physics closed-loop adapter certify its own stand-in.
- **No physical key from a world that has no physics.** The Isaac, SOFA, and
  Warp stand-ins synthesize no `max_pen`, `wall_force_n`, `tissue_stress_kpa`,
  or `haptic_overshoot_mm`. A gate bound to a *fabricated* force — a
  synthesized `0.0` — resolves **pass**, which is the most convincing
  available lie. A gate bound to a key the world simply never reports now
  abstains as unassessable, so the two cases are distinguishable: the danger
  was never the missing key, it was the invented number standing in for it.
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
