"""Cover the subprocess plugin host that CI coverage cannot see via spawn."""

from __future__ import annotations

import io
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from or_audit.errors import TaskContractError
from or_audit.eval.plugin_host import (
    _build_runtime,
    _dispatch,
    _jsonable,
    main,
)
from or_audit.eval.plugins import _jsonable as plugin_jsonable
from or_audit.eval.plugins import load_entrypoint

_POLICY = """
from pathlib import Path
from typing import Any

class Policy:
    def __init__(self) -> None:
        self.closed = False
        self.seed = None

    def reset(self, *, seed: int) -> None:
        self.seed = seed

    def act(self, observation: Any, *, step: int) -> Any:
        return {"obs": observation, "step": step, "seed": self.seed}

    def close(self) -> None:
        self.closed = True

def load_policy(*, root: Path, weights_path: Path) -> Policy:
    del root, weights_path
    return Policy()
"""

_PREDICTOR = """
from pathlib import Path
from typing import Any

print("import log")

class Predictor:
    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        print("prediction log")
        return {"echo": item}

def load_predictor(*, root: Path, weights_path: Path) -> Predictor:
    print("load log")
    del root, weights_path
    return Predictor()
"""

_VERIFIER = """
from pathlib import Path
from typing import Any

class Verifier:
    def score(self, context: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "context": context}

def load_verifier(*, root: Path) -> Verifier:
    del root
    return Verifier()
"""


class _Array:
    def tolist(self) -> list[int]:
        return [1, 2]


class _Scalar:
    def item(self) -> float:
        return 3.5


def _write_plugin(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def test_jsonable_covers_containers_and_fallbacks() -> None:
    payload = {
        "nested": (_Array(),),
        "scalar": _Scalar(),
        "ok": True,
        "none": None,
        "obj": object(),
    }
    encoded = _jsonable(payload)
    assert encoded["nested"] == [[1, 2]]
    assert encoded["scalar"] == 3.5
    assert encoded["ok"] is True
    assert encoded["none"] is None
    assert isinstance(encoded["obj"], str)


def test_dispatch_policy_predictor_verifier_and_unknown() -> None:
    class Runtime:
        def reset(self, *, seed: int) -> None:
            self.seed = seed

        def act(self, observation: Any, *, step: int) -> list[int]:
            return [int(seed) if (seed := getattr(self, "seed", 0)) else 0, step]

        def predict(self, item: dict[str, Any]) -> dict[str, Any]:
            return item

        def score(self, context: dict[str, Any]) -> dict[str, Any]:
            return context

    runtime = Runtime()
    assert _dispatch(runtime, "policy", "close", {}) is None
    assert _dispatch(runtime, "policy", "reset", {"seed": 7}) is None
    assert _dispatch(runtime, "policy", "act", {"observation": [1], "step": 2}) == [
        7,
        2,
    ]
    assert _dispatch(runtime, "predictor", "predict", {"item": {"a": 1}}) == {"a": 1}
    assert _dispatch(runtime, "verifier", "score", {"context": {"b": 2}}) == {"b": 2}
    with pytest.raises(ValueError, match="does not implement"):
        _dispatch(runtime, "verifier", "act", {})


def test_build_runtime_loads_verifier_without_weights(tmp_path: Path) -> None:
    _write_plugin(tmp_path, "verifier.py", _VERIFIER)
    runtime = _build_runtime(
        Namespace(
            root=str(tmp_path),
            entrypoint="verifier.py:load_verifier",
            role="verifier",
            weights_path="",
        )
    )
    assert runtime.score({"x": 1}) == {"ok": True, "context": {"x": 1}}


def test_main_policy_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_plugin(tmp_path, "policy.py", _POLICY)
    (tmp_path / "weights.json").write_text("{}", encoding="utf-8")
    lines = [
        json.dumps({"request_id": "1", "op": "reset", "payload": {"seed": 4}}),
        json.dumps(
            {
                "request_id": "2",
                "op": "act",
                "payload": {"observation": [0.1], "step": 1},
            }
        ),
        json.dumps({"request_id": "3", "op": "not-json-payload", "payload": []}),
        "not-json",
        json.dumps({"request_id": "4", "op": "close", "payload": {}}),
    ]
    monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(lines) + "\n"))
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    assert (
        main(
            [
                "--role",
                "policy",
                "--root",
                str(tmp_path),
                "--entrypoint",
                "policy.py:load_policy",
                "--weights-path",
                "weights.json",
            ]
        )
        == 0
    )
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert responses[0] == {"request_id": "1", "ok": True, "result": None}
    assert responses[1]["ok"] is True
    assert responses[1]["result"]["seed"] == 4
    assert responses[2]["ok"] is False
    assert responses[3]["ok"] is False
    assert responses[4]["ok"] is True


def test_main_predictor_and_verifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_plugin(tmp_path, "predictor.py", _PREDICTOR)
    _write_plugin(tmp_path, "verifier.py", _VERIFIER)
    (tmp_path / "weights.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps({"request_id": "p", "op": "predict", "payload": {"item": {"k": 1}}})
            + "\n"
            + json.dumps({"request_id": "c", "op": "close", "payload": {}})
            + "\n"
        ),
    )
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)
    main(
        [
            "--role",
            "predictor",
            "--root",
            str(tmp_path),
            "--entrypoint",
            "predictor.py:load_predictor",
            "--weights-path",
            "weights.json",
        ]
    )
    assert json.loads(stdout.getvalue().splitlines()[0])["result"] == {"echo": {"k": 1}}
    assert stdout.getvalue().count("\n") == 2
    assert "import log\nload log\nprediction log\n" in stderr.getvalue()

    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps({"request_id": "v", "op": "score", "payload": {"context": {"n": 2}}}) + "\n"
        ),
    )
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    main(
        [
            "--role",
            "verifier",
            "--root",
            str(tmp_path),
            "--entrypoint",
            "verifier.py:load_verifier",
        ]
    )
    assert json.loads(stdout.getvalue().splitlines()[0])["result"]["ok"] is True


def test_plugin_loader_rejects_bad_entrypoints(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("value = 1\n", encoding="utf-8")
    with pytest.raises(TaskContractError, match="non-JSON"):
        plugin_jsonable(object())
    with pytest.raises(TaskContractError, match="entrypoint must be"):
        load_entrypoint(tmp_path, "mod.py", label="policy")
    with pytest.raises(TaskContractError, match="not callable"):
        load_entrypoint(tmp_path, "mod.py:value", label="policy")
