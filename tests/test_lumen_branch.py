"""Lumen branching: deterministic prefix replay (Phase F3 support).

Skipped without the pinned stack. Establishes the only branching mechanism
the pinned env supports: there is no snapshot/restore API, but identical
seeds plus identical action prefixes reproduce identical trajectories, so a
counterfactual branch is a re-reset followed by the shared prefix and then
the candidate action sequence.

Caveat recorded here because it matters for evaluation design: seeds do not
vary initial conditions on the pinned revision (same actions from any seed
give identical trajectories), so repeated seeds measure policy
stochasticity only, not scenario coverage.
"""

from __future__ import annotations

import pytest


def _rollout(seed: int, actions: list[tuple[float, float]]) -> list[dict[str, object]]:
    import gymnasium as gym
    import numpy as np
    from lumen.envs.registration import register_gym_envs

    register_gym_envs()
    env = gym.make("Lumen/NavTreeBranch-v0")
    env.reset(seed=seed)
    try:
        infos: list[dict[str, object]] = []
        for pair in actions:
            _, _, terminated, truncated, info = env.step(np.array(pair, dtype=np.float32))
            infos.append(
                {
                    key: round(float(value), 6) if isinstance(value, (int, float)) else value
                    for key, value in info.items()
                }
            )
            if terminated or truncated:
                break
        return infos
    finally:
        env.close()


def _actions(n: int, seed: int) -> list[tuple[float, float]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    out: list[tuple[float, float]] = []
    for _ in range(n):
        pair = tuple(float(v) for v in rng.uniform(-1.0, 1.0, size=2))
        out.append((pair[0], pair[1]))
    return out


def test_branching_replays_prefix_exactly() -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("lumen.envs.registration")
    actions = _actions(10, 0)
    first = _rollout(0, actions)
    assert _rollout(0, actions) == first
    branched = list(actions)
    branched[5] = (0.9, -0.9)
    alternative = _rollout(0, branched)
    assert alternative[:5] == first[:5]
    assert alternative[5:] != first[5:]


def test_seeds_do_not_vary_initial_conditions() -> None:
    """Documents, not assumes: identical actions from different seeds give
    identical trajectories on the pinned revision."""
    pytest.importorskip("gymnasium")
    pytest.importorskip("lumen.envs.registration")
    actions = _actions(10, 0)
    assert _rollout(1, actions) == _rollout(0, actions)
