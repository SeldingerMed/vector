# G — Training-to-evaluation recipe, Lumen PPO (Phase G)

Status: measured 2026-09-08 on CPU. Reference-level proof that a trained
checkpoint runs through independent held-out evaluation with safety
reporting preserved — not a frontier result.

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

# 2. wrap as an agent package (policy.py loads the SB3 zip; weights pinned
#    by full sha256 in agent.toml; see /tmp/lumen-ppo-agent for the layout)

# 3. evaluate the identical frozen cases (seeds 0-9) through the harness
vector run -t docs/examples/tasks/lumen-nav-safe -a /tmp/lumen-ppo-agent -n 10 \
  --out /tmp/vector-lumen-ppo10
vector replay /tmp/vector-lumen-ppo10

# 4. compare against the random baseline on the same seeds
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
