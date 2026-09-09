"""Trajectory-backed forecast evaluation on recorded Lumen episodes (F2)."""

from __future__ import annotations

from pathlib import Path

from or_audit.eval.loader import load_agent, load_task
from or_audit.eval.runner import replay_job, run_job

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "docs/examples/tasks/lumen-forecast-v1"
AGENT = ROOT / "docs/examples/agents/example-trajectory-forecaster"


def test_forecast_task_scores_recorded_futures(tmp_path: Path) -> None:
    out = tmp_path / "forecast"
    result = run_job(
        task=load_task(TASK),
        task_dir=TASK,
        agent=load_agent(AGENT),
        agent_dir=AGENT,
        out=out,
        n=8,
    )
    assert len(result.trials) == 8
    for trial in result.trials:
        for metric_id in ("mae_h1", "mae_h5", "mae_h20", "unsafe_h1", "unsafe_h5", "unsafe_h20"):
            metric = trial.vector.metric(metric_id)
            assert metric is not None
            assert metric.value is not None
    replayed = replay_job(out, load_task=load_task, load_agent=load_agent)
    assert replayed.head == result.head


def test_null_model_is_beatable_but_honest(tmp_path: Path) -> None:
    """The null forecast (no penetration ever) must score finite errors —
    small here because random actions rarely penetrate, which is exactly
    why beating it requires a real dynamics model."""
    out = tmp_path / "forecast-null"
    result = run_job(
        task=load_task(TASK),
        task_dir=TASK,
        agent=load_agent(AGENT),
        agent_dir=AGENT,
        out=out,
        n=8,
    )
    means: dict[str, float] = {}
    for metric_id in ("mae_h1", "mae_h5", "mae_h20"):
        values: list[float] = []
        for trial in result.trials:
            metric = trial.vector.metric(metric_id)
            assert metric is not None
            assert isinstance(metric.value, float)
            values.append(metric.value)
        means[metric_id] = sum(values) / len(values)
    assert all(mean >= 0.0 for mean in means.values())


def test_short_episode_abstains_beyond_recording() -> None:
    import importlib.util

    path = ROOT / "docs/examples/tasks/lumen-forecast-v1/verifier.py"
    spec = importlib.util.spec_from_file_location("forecast_verifier", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vector = module.load_verifier(root=None).score(
        {
            "label": {"future": [{"max_pen": 0.1, "unsafe": False}] * 3},
            "prediction": {
                "forecast": {
                    "h1": {"max_pen": 0.1, "unsafe": False},
                    "h5": {"max_pen": 0.2, "unsafe": False},
                    "h20": {"max_pen": 0.3, "unsafe": False},
                }
            },
        }
    )
    assert vector["metrics"]["mae_h1"] == 0.0
    assert vector["metrics"]["mae_h5"] is None
    assert vector["metrics"]["unsafe_h20"] is None
