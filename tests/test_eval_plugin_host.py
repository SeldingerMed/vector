"""Cover the subprocess plugin host that CI coverage cannot see via spawn."""

from __future__ import annotations

import io
import json
import os
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
from or_audit.eval.plugins import (
    SubprocessPredictorRuntime,
    _scrubbed_plugin_env,
    load_entrypoint,
    load_predictor_runtime,
)
from or_audit.eval.plugins import _jsonable as plugin_jsonable

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


_ENV_SPY = """
import os
from pathlib import Path
from typing import Any

class Predictor:
    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        del item
        return {"env_keys": sorted(os.environ.keys()), "home": os.environ.get("HOME", "")}

def load_predictor(*, root: Path, weights_path: Path) -> Predictor:
    del root, weights_path
    return Predictor()
"""


def test_plugin_subprocess_sees_scrubbed_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SURG_EVAL_TEST_SECRET", "must-not-leak")
    monkeypatch.setenv("HF_TOKEN", "must-not-leak")
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "spy.py").write_text(_ENV_SPY, encoding="utf-8")
    (plugin / "weights.json").write_text("{}", encoding="utf-8")
    runtime = load_predictor_runtime(plugin, "spy.py:load_predictor", "weights.json")
    assert isinstance(runtime, SubprocessPredictorRuntime)
    result = runtime.predict({})
    keys = result.get("env_keys", [])
    assert "SURG_EVAL_TEST_SECRET" not in keys
    assert "HF_TOKEN" not in keys
    assert "PATH" in keys
    home = result.get("home", "")
    assert home not in (None, "", os.environ.get("HOME"))
    assert os.path.isdir(home)
    assert os.listdir(home) == []
    runtime.close()
    assert not os.path.exists(home)


def test_scrubbed_plugin_env_allowlists_runtime_vars() -> None:
    env = _scrubbed_plugin_env(
        {
            "PATH": "/bin",
            "HF_TOKEN": "x",
            "AWS_SECRET_ACCESS_KEY": "y",
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_FOO": "secret-or-config",
            "NVIDIA_FOO": "secret-or-config",
            "PYTHONPATH": "/tmp/evil",
            "HOME": "/root",
        }
    )
    assert env == {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0"}


def test_scrubbed_env_matches_windows_casing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os as os_module

    monkeypatch.setattr(os_module, "name", "nt")
    env = _scrubbed_plugin_env({"Path": "C:\\bin", "HF_TOKEN": "x"})
    assert env == {"Path": "C:\\bin"}


def _container_descriptor(image: str, digest: str) -> Any:
    from or_audit.eval.contracts import RuntimeDescriptor, RuntimeKind

    return RuntimeDescriptor(kind=RuntimeKind.CONTAINER, image=image, image_digest=digest)


def test_container_command_pins_isolates_and_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/docker")
    from or_audit.eval.plugins import _runtime_command

    command, _ = _runtime_command(
        _container_descriptor("registry.example/surgeval-plugin", "a" * 64),
        role="predictor",
        root=tmp_path,
        entrypoint="spy.py:load_predictor",
    )
    assert command[:6] == ("docker", "run", "--rm", "-i", "--pull", "never")
    assert "--network" in command
    assert "none" in command
    assert "--read-only" in command
    assert "--tmpfs" in command
    joined = " ".join(command)
    assert f"{tmp_path.resolve()}:/pkg:ro" in joined
    assert f"registry.example/surgeval-plugin@sha256:{'a' * 64}" in joined
    assert "or_audit.eval.plugin_host" in joined
    for flag in ("--memory", "--cpus", "--pids-limit"):
        assert flag in command
    assert "--workdir" in command
    assert "--user" in command
    assert "65534" in command
    assert "--cap-drop" in command
    assert "ALL" in command
    assert "--security-opt" in command
    assert "no-new-privileges" in command


def test_container_command_normalizes_and_refuses_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/docker")
    from or_audit.errors import TaskContractError
    from or_audit.eval.plugins import _runtime_command

    command, _ = _runtime_command(
        _container_descriptor("registry.example/p", "sha256:" + "b" * 64),
        role="predictor",
        root=tmp_path,
        entrypoint="spy.py:load_predictor",
    )
    assert f"registry.example/p@sha256:{'b' * 64}" in " ".join(command)
    with pytest.raises(TaskContractError, match="bare repository"):
        _runtime_command(
            _container_descriptor("registry.example/p@sha256:" + "c" * 64, "d" * 64),
            role="predictor",
            root=tmp_path,
            entrypoint="spy.py:load_predictor",
        )


def test_container_backend_runs_predictor_end_to_end(tmp_path: Path) -> None:
    import shutil

    image = os.environ.get("SURGEVAL_TEST_PLUGIN_IMAGE", "")
    if not image or shutil.which("docker") is None:
        pytest.skip("needs docker and SURGEVAL_TEST_PLUGIN_IMAGE=image@digest")
    from or_audit.eval.contracts import RuntimeDescriptor, RuntimeKind

    ref, _, digest = image.partition("@")
    assert digest, "SURGEVAL_TEST_PLUGIN_IMAGE must be digest-pinned"
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "spy.py").write_text(_ENV_SPY, encoding="utf-8")
    (plugin / "weights.json").write_text("{}", encoding="utf-8")
    runtime = load_predictor_runtime(
        plugin,
        "spy.py:load_predictor",
        "weights.json",
        runtime=RuntimeDescriptor(kind=RuntimeKind.CONTAINER, image=ref, image_digest=digest),
    )
    assert isinstance(runtime, SubprocessPredictorRuntime)
    try:
        result = runtime.predict({})
    finally:
        runtime.close()
    assert isinstance(result.get("env_keys"), list)
