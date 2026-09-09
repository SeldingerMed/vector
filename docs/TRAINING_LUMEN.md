# G — Training-to-evaluation recipe, Lumen PPO (Phase G)

# Status: measured 2026-09-08 on CPU. Train→same-environment evaluation
# plumbing only: a trained checkpoint runs through the harness on the same
# frozen cases with safety reporting preserved. Not held-out evaluation in
# any generalization sense (see Limits), and not a frontier result.

## Separation of concerns

- Training reward: the env's own shaped channel (progress, penalties).
- Evaluation: the task's safety vector, recomputed from stored trajectories.
  Never the same number. A zeroed training reward is not claimed; the
  packaged task offers no projection (see brief), so this loop proves
  train→evaluate plumbing, not reward integrity under training.

## Exact recipe

Pinned stack: `seldinger-lumen` @3c6bb39e, warp-lang 1.15.0,
newton @6dfe730, gymnasium 1.3.0, stable-baselines3 2.9.0, torch 2.14.0
(CPU), surgeval 0.3.0a11, numpy 2.5.3.

```bash
# 1. train (probe venv with the stack above)
/tmp/lumen-probe/.venv/bin/python - <<'EOF'
from lumen.envs.registration import register_gym_envs
register_gym_envs()
import gymnasium as gym
from stable_baselines3 import PPO
env = gym.make('Lumen/NavTreeBranch-v0')
model = PPO('MlpPolicy', env, verbose=1, seed=0, n_steps=512, batch_size=128, n_epochs=4)
model.learn(total_timesteps=20480)
model.save('/tmp/lumen-ppo-g1')
EOF

# 2. wrap the checkpoint as an agent package. stable-baselines3 is an agent
#    dependency: for runtime.kind="local" it is imported by the same Python
#    interpreter running `vector`, so run `vector` from the probe venv (or
#    install stable-baselines3 into your `vector` env). Never a harness dep.
mkdir -p /tmp/lumen-ppo-agent && cp /tmp/lumen-ppo-g1.zip /tmp/lumen-ppo-agent/ppo.zip
WEIGHTS_PIN=$(sha256sum /tmp/lumen-ppo-agent/ppo.zip | cut -d' ' -f1)
cat > /tmp/lumen-ppo-agent/agent.toml <<EOF
format_version = "2"
id = "local/ppo-nav"
agent_version = "0"
kind = "policy"
weights_pin = "$WEIGHTS_PIN"
weights_path = "ppo.zip"

[[capabilities]]
interface = "gym-policy"
interaction_modes = ["closed-loop"]
protocol_versions = ["1"]
observations = ["gym-obs"]
actions = ["insertion_twist"]

[runtime]
kind = "local"
protocol_version = "1"
entrypoint = "policy.py:load_policy"
timeout_sec = 120.0
EOF
cat > /tmp/lumen-ppo-agent/policy.py <<'EOF'
from pathlib import Path
from typing import Any
import numpy as np

class SB3Policy:
    def __init__(self, weights_path):
        from stable_baselines3 import PPO
        self._model = PPO.load(weights_path)
    def reset(self, *, seed):
        del seed
    def act(self, observation, *, step):
        del step
        action, _ = self._model.predict(np.asarray(observation, dtype=np.float32))
        return np.asarray(action, dtype=np.float32)

def load_policy(*, root, weights_path):
    del root
    return SB3Policy(weights_path)
EOF

# 3. baseline: the identical frozen cases (seeds 0-9) with the random agent
vector run -t docs/examples/tasks/lumen-nav-safe -a docs/examples/agents/seldingermed-random \
  -n 10 --out /tmp/vector-lumen-random10

# 4. evaluate the trained checkpoint on the same cases
vector run -t docs/examples/tasks/lumen-nav-safe -a /tmp/lumen-ppo-agent \
  -n 10 --out /tmp/vector-lumen-ppo10
vector replay /tmp/vector-lumen-ppo10

# 5. compare with uncertainty
vector compare /tmp/vector-lumen-random10 /tmp/vector-lumen-ppo10 \
  --metric safe_success
```

## Results (n=10 per arm, seeds 0-9)

| Arm | Head (prefix) | safe_success | Gates |
|---|---|---|---|
| random | `612e4910…` | 0.0 (0/10) | 10 pass |
| PPO 20k steps | `23e6b180…` | 0.6 (6/10) | 10 pass |

- `safe_success` diff: +0.6, 95% CI (0.3, 0.9) — improvement with uncertainty.
- `max_pen` diff: -0.019, 95% CI (-0.058, 0.0) — no measured safety regression,
  but the interval covers zero: not evidence of improvement either.
- `diverged` unassessable throughout (env reports no divergence channel).
- Checkpoint digest (sha256 of zip): `52894cfe…` (full value in run notes,
  `/tmp` only — binaries are never committed; retrain from the recipe).

## Limits

- 20k steps on a toy task; no claim about sample efficiency or transfer.
- No held-out *scenario* split exists (seeds don't vary initial conditions),
  so "held-out" here means fixed-case re-evaluation, not generalization.
- Reward hacking untested beyond gate outcomes; abstention abuse n/a
  (policy has no abstention channel).
- Prime/trainer interop beyond SB3 PPO is future work; token-based RL stays
  out of scope for continuous control.
