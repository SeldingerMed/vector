# The 10-minute on-ramp

Five steps from nothing to a replayable safety vector for your own model. Every command below is copy-pasteable, and every step says what you should see.

There are two paths through SurgEval, and only one of them belongs in your first ten minutes:

- **The trial path** — `@se.agent` on a class you already have, then `surgeval run`. No `task.toml`, no `agent.toml`, no digests to compute. This page.
- **The publication path** — a versioned, digest-pinned package directory that other people can pull, rerun, and hold you to. That comes later, and `surgeval init-agent` generates it for you when you get there.

---

## Step 1 — Install

```bash
uv tool install surgeval
surgeval --version
```

**You should see** a version string (`0.3.0a0` or later). Nothing else is installed: the kernel, the CLI, and the CPU-only reference tasks ship in the base package. Simulation worlds (SOFA, Isaac, MuJoCo) are separate extras you do not need yet.

Working from a clone instead:

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"
```

## Step 2 — Run the reference agent

Prove the harness works before you introduce your own code.

```bash
surgeval quickstart
```

**You should see** a task id, a head digest, an output directory, and the elapsed time to your first vector — followed by the exact `surgeval replay` line for that run. This runs a CPU-only reference task end to end: no GPU, no simulator, no network.

If you want to drive it by hand instead, the same run spelled out:

```bash
surgeval run \
  -t docs/examples/tasks/video-nextstep \
  -a docs/examples/agents/example-video-predictor \
  --out runs/reference
```

```
ran: video-nextstep n=3 head ceca73955d4c0e8f8166877f9656663140c2505ae9c5c656d424877c8b7805d4
```

The head is a digest over the whole job: task package, agent package, every trial vector. Two people who run the same pair get the same head, or one of them has changed something.

## Step 3 — Point it at your model

Decorate the class you already have. One import, one line.

```python
# mymodel.py
import surgeval as se


@se.agent(interface="video-predict", agent_id="myorg/next-step", version="1")
class MyModel:
    def predict(self, item):
        # item is the task's input record; return the task's declared output shape
        return {"next_step": "advance", "outcome": "continue", "unsafe": False}
```

```python
result = se.evaluate(MyModel(), "docs/examples/tasks/video-nextstep", out="runs/mine")
print(result.head)
```

**You should see** a head digest, and `runs/mine/` containing `config.json`, `result.json`, `scorecard.json`, and a replayable bundle.

The decorator infers your capability from the class rather than asking you to write one:

- `predict(item)` is the predictor protocol → single-turn, counterfactual, and interactive tasks.
- `act(observation, step=...)` is the policy protocol → closed-loop tasks.
- Implement both and both are declared.

Ask it what it inferred:

```bash
surgeval describe-agent mymodel:MyModel
```

```
agent: myorg/next-step@1
  class        MyModel
  interface    video-predict
  methods      predict(item)
  modes        single-turn, counterfactual, interactive
  kind         frozen-model
  entrypoint   runner.py:load_predictor
  binding      WILDCARD (unverified)
               ...
```

**WILDCARD is the one thing to fix.** A class that declares no schemas binds to *any* task on that interface, and the kernel never checks that your model speaks that task's data shapes. Every such job records `binding_mode: "wildcard"` in `config.json`, so the result is honest about it — but a wildcard binding is a convenience, not a verification. Declare your schemas and the binding becomes checked:

```python
@se.agent(interface="video-predict", agent_id="myorg/next-step", version="1")
class MyModel:
    observations = ["video-clip"]
    outputs = ["next-step"]
    features = ["reasoning", "abstention"]

    def predict(self, item): ...
```

To make an unverified binding an error rather than a footnote:

```python
se.evaluate(MyModel(), task, out="runs/mine", strict_schemas=True)
```

**You should see** a `TaskContractError` naming the interface and telling you which attributes to declare — or a normal run, if your declaration satisfies the task.

## Step 4 — Read the vector

Open `runs/mine/result.json`. Every trial carries a `TrialVector`:

```json
{
  "gates": [
    {"id": "unsafe_prediction", "status": "pass", "reason": "not(unsafe == true)", "evidence": [...]}
  ],
  "metrics": [
    {"id": "next_step_correct", "value": true, "kind": "boolean", "direction": "maximize", "headline": true},
    {"id": "abstained", "value": false, "kind": "boolean", "direction": "neutral", "headline": false}
  ]
}
```

### What the vector means

- **Gates are not metrics.** A hard gate is a safety predicate with a `pass` / `fail` / `not_assessable` status and the evidence digests it was computed from. It is never averaged into a quality number, because "mostly did not injure the patient" is not a score.
- **Abstention is legal.** `not_assessable` is a first-class outcome, and a task may declare `abstain_ok`. A model that declines to answer is recorded as declining, not as wrong. Metrics for an abstained trial are `null`, counted separately from false.
- **There is no composite score.** SurgEval never collapses gates and metrics into a single ranking number. A leaderboard row is the vector. If you need a scalar for RL, the *task* declares a `ProjectionSpec` (source metric, guard metrics, gate-failure behavior) and `surgeval export-rl` recomputes it from the authoritative vector and writes the rule plus its digest beside every reward.

Every run also writes the same vector as a human-readable scorecard:

```bash
cat runs/mine/scorecard.md
```

```
## Safety gates

| Gate | Pass | Fail | Not assessable | Not applicable |
|---|---:|---:|---:|---:|
| unsafe_prediction | 1 | 1 | 1 | 0 |

## Metrics

| Metric | Headline | Result | Assessed | Unassessable |
|---|:---:|---:|---:|---:|
| next_step_correct | yes | 1.000000 | 2 | 1 |
```

Gates counted, metrics counted, unassessable counted — three separate columns, never one number.

## Step 5 — Replay it

A result you cannot reproduce is a screenshot.

```bash
surgeval replay runs/mine
```

```
replay matched: video-nextstep head ceca73955d4c0e8f8166877f9656663140c2505ae9c5c656d424877c8b7805d4
```

**You should see** `replay matched:` and the same head as the original run. Replay reloads the pinned task and agent packages, re-executes, and requires the recomputed head to equal the stored one. Any drift — a changed weights file, an edited verifier, a mutated input — is a `REPLAY FAILED`, not a warning.

To assert a specific head in CI:

```bash
surgeval replay runs/mine --expect-head ceca7395...
```

---

## When you are ready to publish

Everything above ran from an in-memory class. Publishing means shipping a directory other people can pull and rerun. Generate it from the class you already decorated:

```bash
surgeval init-agent mymodel:MyModel --out packages/next-step --weights model.safetensors
```

**You should see** the package path, the id and version, the entrypoint, and the weights digest:

```
wrote agent package: /abs/path/packages/next-step
  id           myorg/next-step@1
  kind         frozen-model
  entrypoint   runner.py:load_predictor
  weights      model.safetensors  sha256:47c46a5fa409889c23e4d12bdc28a077a7ccc8a92c8dd2bfbe3d7d9c6c227e67
  vendored     mymodel.py (the package is self-contained)
  binding      verified against declared schemas
next: surgeval run -t <task> -a packages/next-step --out runs/first
```

`init-agent` writes `agent.toml` (the capability exactly as inferred), `runner.py` (the harness entrypoint), and the weights file, and pins `weights_pin` to the **real** SHA-256 of the bytes it wrote. Omit `--weights` and it writes a placeholder file and pins that placeholder honestly — the package loads, and nothing pretends to be weights it is not. Replace the file later and re-run with `--weights <file> --force`.

`init-agent` and `describe-agent` import `module:Class` from your working directory, which executes that module in-process. That is your own source tree, the same trust level as `python -c "import mymodel"`. It is not an upload path: untrusted models run through the isolated plugin-host runtime instead.

Package authoring — hand-written `task.toml` and `agent.toml`, digest pins, registry references — is the **publication** path, not the trial path. If you are still deciding whether SurgEval measures the thing you care about, stay on `@se.agent` and `surgeval run`; you lose nothing except the ability to hand your result to someone else and have them reproduce it byte for byte.
