"""Execute task and agent plugins through a line-delimited JSON subprocess protocol."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import selectors
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast, runtime_checkable

from or_audit.errors import TaskContractError
from or_audit.eval.contracts import RuntimeDescriptor, RuntimeKind
from or_audit.eval.integrity import package_file


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return _jsonable(to_list())
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    raise TaskContractError(f"plugin request contains non-JSON value {type(value).__name__}")


@runtime_checkable
class PolicyRuntime(Protocol):
    def reset(self, *, seed: int) -> None: ...
    def act(self, observation: Any, *, step: int) -> Any: ...


@runtime_checkable
class PredictorRuntime(Protocol):
    def predict(self, item: dict[str, Any]) -> dict[str, Any]: ...


@runtime_checkable
class VerifierRuntime(Protocol):
    def score(self, context: dict[str, Any]) -> dict[str, Any]: ...


def _module(path: Path) -> ModuleType:
    module_name = f"or_audit_plugin_{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise TaskContractError(f"cannot load plugin module {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise TaskContractError(f"plugin {path} failed to import: {exc}") from exc
    return module


def load_entrypoint(root: Path, entrypoint: str, *, label: str) -> Callable[..., Any]:
    """Load an entrypoint inside an already isolated plugin-host process."""
    module_path, separator, symbol = entrypoint.partition(":")
    if not separator or not module_path or not symbol:
        raise TaskContractError(
            f"{label} entrypoint must be relative.py:symbol, got {entrypoint!r}"
        )
    path = package_file(root, module_path, label=f"{label} module")
    target = getattr(_module(path), symbol, None)
    if not callable(target):
        raise TaskContractError(f"{label} entrypoint {entrypoint!r} is not callable")
    return cast(Callable[..., Any], target)


#: Environment variables a plugin child is allowed to inherit. Everything
#: else — ambient secrets, cloud credentials, model API keys, repo tokens —
#: stays in the parent. The set is enumerated, never prefix-matched: vendor
#: families like CUDA_/NVIDIA_ can carry secrets and config paths, so only
#: the device-selection variables needed for GPU discovery are named.
#: HOME is never inherited (it points at ~/.aws, ~/.huggingface, shell
#: history): each runtime gets a fresh empty directory instead (B1 tier T0).
_PLUGIN_ENV_ALLOW = frozenset(
    {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_CACHE_PATH",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "DOCKER_HOST",
    }
)

#: In-container mount point for the evaluated package (read-only).
_CONTAINER_PKG_DIR = "/pkg"
#: Default resource envelope for containerized plugin children (B4).
_CONTAINER_MEMORY = "2g"
_CONTAINER_CPUS = "2.0"
_CONTAINER_PIDS_LIMIT = "256"

#: Uppercase twin for Windows, where environment keys are case-insensitive
#: but stored mixed-case (`Path`, `SystemRoot`).
_PLUGIN_ENV_ALLOW_UPPER = frozenset(key.upper() for key in _PLUGIN_ENV_ALLOW)


def _scrubbed_plugin_env(source: dict[str, str] | None = None) -> dict[str, str]:
    """Allowlisted environment for plugin children (B1 §enforced-2)."""
    inherited = os.environ if source is None else source
    windows = os.name == "nt"
    return {
        key: value
        for key, value in inherited.items()
        if key in _PLUGIN_ENV_ALLOW or (windows and key.upper() in _PLUGIN_ENV_ALLOW_UPPER)
    }


def _private_plugin_home() -> Path:
    """Fresh empty HOME/TMP for one plugin child."""
    return Path(tempfile.mkdtemp(prefix="surgeval-plugin-"))


#: Largest single plugin response accepted (8 MiB). Evidence transfer is one
#: bounded JSON value per request, not a stream: an unbounded read lets a
#: compromised or buggy child exhaust host memory before the timeout fires.
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class JsonSubprocessRuntime:
    """Persistent JSON-lines child with bounded request latency."""

    def __init__(self, command: tuple[str, ...], *, cwd: Path, timeout_sec: float) -> None:
        self._timeout_sec = timeout_sec
        self._next_request = 0
        self._plugin_home = _private_plugin_home()
        env = _scrubbed_plugin_env()
        env["HOME"] = str(self._plugin_home)
        env["TMPDIR"] = str(self._plugin_home)
        env["TEMP"] = str(self._plugin_home)
        env["TMP"] = str(self._plugin_home)
        self._process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

    def request(self, op: str, payload: dict[str, Any]) -> Any:
        process = self._process
        if process.stdin is None or process.stdout is None:
            raise TaskContractError("plugin process has no JSON protocol streams")
        request_id = self._next_request
        self._next_request += 1
        message = json.dumps(
            {"request_id": request_id, "op": op, "payload": _jsonable(payload)},
            separators=(",", ":"),
        )
        try:
            process.stdin.write(message + "\n")
            process.stdin.flush()
        except BrokenPipeError as exc:
            raise TaskContractError(self._failure("plugin process exited before request")) from exc

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        ready = selector.select(self._timeout_sec)
        selector.close()
        if not ready:
            process.kill()
            process.wait()
            self._close_pipes()
            raise TaskContractError(f"plugin request {op!r} exceeded {self._timeout_sec}s")
        line = process.stdout.readline(_MAX_RESPONSE_BYTES + 2)
        if not line:
            raise TaskContractError(self._failure("plugin process returned no response"))
        if not line.endswith("\n") or len(line) - 1 > _MAX_RESPONSE_BYTES:
            process.kill()
            process.wait()
            self._close_pipes()
            raise TaskContractError(
                f"plugin {op!r} response exceeded {_MAX_RESPONSE_BYTES} bytes; "
                "evidence transfer is one bounded value per request"
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TaskContractError(f"plugin returned malformed JSON: {line[:200]!r}") from exc
        if response.get("request_id") != request_id:
            raise TaskContractError("plugin response request_id does not match request")
        if not response.get("ok"):
            error = response.get("error", "unknown plugin failure")
            raise TaskContractError(f"plugin {op} failed: {error}")
        return response.get("result")

    def _failure(self, message: str) -> str:
        stderr = ""
        if self._process.stderr is not None and self._process.poll() is not None:
            stderr = self._process.stderr.read().strip()
        return f"{message}: {stderr}" if stderr else message

    def _close_pipes(self) -> None:
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            if stream is not None and not stream.closed:
                stream.close()

    def close(self) -> None:
        try:
            if self._process.poll() is not None:
                self._close_pipes()
                return
            try:
                self.request("close", {})
            except TaskContractError:
                self._process.kill()
            finally:
                try:
                    self._process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
                self._close_pipes()
        finally:
            shutil.rmtree(self._plugin_home, ignore_errors=True)


class SubprocessPolicyRuntime(JsonSubprocessRuntime):
    def reset(self, *, seed: int) -> None:
        self.request("reset", {"seed": seed})

    def act(self, observation: Any, *, step: int) -> Any:
        return self.request("act", {"observation": observation, "step": step})


class SubprocessPredictorRuntime(JsonSubprocessRuntime):
    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        result = self.request("predict", {"item": item})
        if not isinstance(result, dict):
            raise TaskContractError("predictor subprocess must return an object")
        return result


class SubprocessVerifierRuntime(JsonSubprocessRuntime):
    def score(self, context: dict[str, Any]) -> dict[str, Any]:
        result = self.request("score", {"context": context})
        if not isinstance(result, dict):
            raise TaskContractError("verifier subprocess must return an object")
        return result


def _container_command(
    descriptor: RuntimeDescriptor,
    *,
    role: str,
    root: Path,
    entrypoint: str,
    weights_path: str = "",
) -> tuple[str, ...]:
    """Docker execution with no network, a read-only package, and caps.

    Separate containers per runtime object preserve the agent/verifier
    process boundary as filesystem/network boundaries (B2). The image
    reference is always digest-pinned and never pulled at run time: the
    image must be pre-provisioned, so a task package cannot turn
    evaluation into host-driven registry egress. Containers run as
    nobody with all capabilities dropped and no new privileges.
    """
    if shutil.which("docker") is None:
        raise TaskContractError(
            "container runtime needs the docker CLI on PATH; "
            "install Docker (or Colima) and ensure the daemon runs"
        )
    if "@" in descriptor.image:
        raise TaskContractError(
            f"container image {descriptor.image!r} must be a bare repository; "
            "pass the digest in image_digest, not in image"
        )
    digest = descriptor.image_digest.removeprefix("sha256:")
    inner = [
        "python",
        "-m",
        "or_audit.eval.plugin_host",
        "--role",
        role,
        "--root",
        _CONTAINER_PKG_DIR,
        "--entrypoint",
        entrypoint,
    ]
    if weights_path:
        inner.extend(["--weights-path", weights_path])
    return (
        "docker",
        "run",
        "--rm",
        "-i",
        # Never pull: the image must be pre-provisioned, so a task package
        # cannot turn evaluation into host-driven registry egress.
        "--pull",
        "never",
        "--network",
        "none",
        "--workdir",
        _CONTAINER_PKG_DIR,
        "--read-only",
        "--tmpfs",
        "/tmp",
        "--user",
        "65534",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--memory",
        _CONTAINER_MEMORY,
        "--cpus",
        _CONTAINER_CPUS,
        "--pids-limit",
        _CONTAINER_PIDS_LIMIT,
        "-v",
        f"{root.resolve()}:{_CONTAINER_PKG_DIR}:ro",
        f"{descriptor.image}@sha256:{digest}",
        *inner,
    )


def _host_command(
    *, role: str, root: Path, entrypoint: str, weights_path: str = ""
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "-m",
        "or_audit.eval.plugin_host",
        "--role",
        role,
        "--root",
        str(root.resolve()),
        "--entrypoint",
        entrypoint,
    ]
    if weights_path:
        command.extend(["--weights-path", weights_path])
    return tuple(command)


def _runtime_command(
    descriptor: RuntimeDescriptor | None,
    *,
    role: str,
    root: Path,
    entrypoint: str,
    weights_path: str = "",
) -> tuple[tuple[str, ...], float]:
    runtime = descriptor or RuntimeDescriptor(kind=RuntimeKind.LOCAL, entrypoint=entrypoint)
    if runtime.kind is RuntimeKind.CONTAINER:
        return _container_command(
            runtime, role=role, root=root, entrypoint=entrypoint, weights_path=weights_path
        ), runtime.timeout_sec
    if runtime.kind is not RuntimeKind.LOCAL:
        raise TaskContractError(
            f"runtime {runtime.kind.value} is represented but is not locally executable"
        )
    if runtime.command:
        replacements = {
            "{package}": str(root.resolve()),
            "{entrypoint}": entrypoint,
            "{weights}": weights_path,
            "{role}": role,
        }
        command = tuple(replacements.get(part, part) for part in runtime.command)
    else:
        command = _host_command(
            role=role,
            root=root,
            entrypoint=runtime.entrypoint or entrypoint,
            weights_path=weights_path,
        )
    return command, runtime.timeout_sec


def load_policy_runtime(
    root: Path,
    entrypoint: str,
    weights_path: str,
    runtime: RuntimeDescriptor | None = None,
) -> PolicyRuntime:
    if runtime is not None and runtime.kind is RuntimeKind.TRUSTED_IN_PROCESS:
        factory = load_entrypoint(root, runtime.entrypoint or entrypoint, label="trusted policy")
        loaded = factory(
            root=root,
            weights_path=package_file(root, weights_path, label="weights"),
        )
        if not isinstance(loaded, PolicyRuntime):
            raise TaskContractError("trusted policy factory returned an incompatible object")
        return loaded
    command, timeout = _runtime_command(
        runtime, role="policy", root=root, entrypoint=entrypoint, weights_path=weights_path
    )
    return SubprocessPolicyRuntime(command, cwd=root.resolve(), timeout_sec=timeout)


def load_predictor_runtime(
    root: Path,
    entrypoint: str,
    weights_path: str,
    runtime: RuntimeDescriptor | None = None,
) -> PredictorRuntime:
    if runtime is not None and runtime.kind is RuntimeKind.TRUSTED_IN_PROCESS:
        factory = load_entrypoint(root, runtime.entrypoint or entrypoint, label="trusted predictor")
        loaded = factory(
            root=root,
            weights_path=package_file(root, weights_path, label="weights"),
        )
        if not isinstance(loaded, PredictorRuntime):
            raise TaskContractError("trusted predictor factory returned an incompatible object")
        return loaded
    command, timeout = _runtime_command(
        runtime, role="predictor", root=root, entrypoint=entrypoint, weights_path=weights_path
    )
    return SubprocessPredictorRuntime(command, cwd=root.resolve(), timeout_sec=timeout)


def load_verifier_runtime(
    root: Path,
    entrypoint: str,
    runtime: RuntimeDescriptor | None = None,
) -> VerifierRuntime:
    if runtime is not None and runtime.kind is RuntimeKind.TRUSTED_IN_PROCESS:
        factory = load_entrypoint(root, runtime.entrypoint or entrypoint, label="trusted verifier")
        loaded = factory(root=root)
        if not isinstance(loaded, VerifierRuntime):
            raise TaskContractError("trusted verifier factory returned an incompatible object")
        return loaded
    command, timeout = _runtime_command(runtime, role="verifier", root=root, entrypoint=entrypoint)
    return SubprocessVerifierRuntime(command, cwd=root.resolve(), timeout_sec=timeout)
