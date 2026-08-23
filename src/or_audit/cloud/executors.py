"""Execution backends for the minimal Vector Cloud control plane."""

from __future__ import annotations

import json
import math
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from or_audit.errors import TaskContractError

from .models import ComputeClass, InputArtifact, JobRecord, JobRequest, JobStatus, MachineSize
from .store import JobStore


class Executor(Protocol):
    def submit(self, job: JobRecord) -> None: ...

    def cancel(self, job: JobRecord) -> None: ...
    def release(self, job: JobRecord) -> None: ...

    def reconcile(self, job: JobRecord) -> JobRecord: ...


class LocalExecutor:
    """Run the public SurgEval CLI in a background subprocess."""

    def __init__(self, store: JobStore, *, root: Path, package_root: Path) -> None:
        self.store = store
        self.root = root
        self.package_root = package_root
        self._processes: dict[str, subprocess.Popen[str] | None] = {}
        self._lock = threading.Lock()
        root.mkdir(parents=True, exist_ok=True)

    def submit(self, job: JobRecord) -> None:
        with self._lock:
            self._processes[job.id] = None
        threading.Thread(target=self._run, args=(job,), daemon=True).start()

    def cancel(self, job: JobRecord) -> None:
        with self._lock:
            self.store.transition(
                job.id,
                expected=(JobStatus.QUEUED, JobStatus.RUNNING),
                status=JobStatus.CANCELLED,
                error="cancelled by user",
            )
            process = self._processes.get(job.id)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def reconcile(self, job: JobRecord) -> JobRecord:
        return self.store.get(job.id) or job

    def release(self, job: JobRecord) -> None:
        del job

    def _run(self, job: JobRecord) -> None:
        process: subprocess.Popen[str] | None = None
        try:
            job_root = self.root / job.id
            artifact = job_root / "result"
            job_root.mkdir(parents=True, exist_ok=True)
            request = job.request
            command = [
                sys.executable,
                "-m",
                "or_audit.cli",
                "run",
                "-t",
                request.task,
                "-a",
                request.agent,
                "-n",
                str(request.n),
                "--out",
                str(artifact),
            ]
            if request.registry:
                command.extend(("--registry", request.registry))
            with self._lock:
                self.store.transition(
                    job.id,
                    expected=(JobStatus.QUEUED,),
                    status=JobStatus.RUNNING,
                )
                process = subprocess.Popen(
                    command,
                    cwd=self.package_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self._processes[job.id] = process
            stdout, stderr = process.communicate()
            (job_root / "stdout.log").write_text(stdout, encoding="utf-8")
            (job_root / "stderr.log").write_text(stderr, encoding="utf-8")
            current = self.store.get(job.id)
            if current is None or current.status is JobStatus.CANCELLED:
                return
            if process.returncode != 0:
                message = (
                    stderr.strip() or stdout.strip() or f"surgeval exited {process.returncode}"
                )
                self.store.transition(
                    job.id,
                    expected=(JobStatus.RUNNING,),
                    status=JobStatus.FAILED,
                    error=message[-4000:],
                )
                return
            result_path = artifact / "result.json"
            if not result_path.is_file():
                self.store.transition(
                    job.id,
                    expected=(JobStatus.RUNNING,),
                    status=JobStatus.FAILED,
                    error="surgeval completed without result.json",
                )
                return
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.store.transition(
                job.id,
                expected=(JobStatus.RUNNING,),
                status=JobStatus.SUCCEEDED,
                artifact_path=str(artifact),
                result_head=str(result.get("head", "")),
            )
        except Exception as exc:
            with suppress(TaskContractError):
                self.store.transition(
                    job.id,
                    expected=(JobStatus.QUEUED, JobStatus.RUNNING),
                    status=JobStatus.FAILED,
                    error=(f"{type(exc).__name__}: {exc}")[-4000:],
                )
        finally:
            with self._lock:
                self._processes.pop(job.id, None)


_MACHINE0_SIZE_CANDIDATES = {
    MachineSize.LARGE: ("large",),
    MachineSize.XL: ("xl",),
    MachineSize.XXL: ("xxl",),
    MachineSize.XXXL: ("xxxl",),
    MachineSize.GPU_L40S: ("gpu-l40s-1", "gpu-6000ada-1"),
    MachineSize.GPU_H100: ("gpu-h100-1", "gpu-h200-1"),
}

_MACHINE0_PRICE_PER_HOUR_MICROS = {
    "large": 52_000,
    "xl": 104_000,
    "xxl": 208_000,
    "xxxl": 825_000,
    "gpu-l40s-1": 1_727_000,
    "gpu-6000ada-1": 1_727_000,
    "gpu-h100-1": 4_851_000,
    "gpu-h200-1": 4_917_000,
}


class CommandRunner(Protocol):
    def __call__(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]: ...


def _run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
    )


class Machine0Executor:
    """Provision one isolated VM per run and remove it after evidence is copied back."""

    def __init__(
        self,
        store: JobStore,
        *,
        root: Path,
        package_root: Path,
        image: str = "ubuntu-24-04-loaded",
        gpu_image: str = "gpu-h100x1-base",
        binary: str = "machine0",
        allowed_input_host: str = "",
        key_name: str = "vector-service",
        keep_machines: bool = False,
        runner: CommandRunner = _run_command,
    ) -> None:
        self.store = store
        self.root = root
        self.package_root = package_root
        self.image = image
        self.gpu_image = gpu_image
        self.binary = binary
        self.allowed_input_host = allowed_input_host
        self.key_name = key_name
        self.keep_machines = keep_machines
        self.runner = runner
        self._machines: dict[str, str] = {}
        self._lock = threading.Lock()
        root.mkdir(parents=True, exist_ok=True)

    def submit(self, job: JobRecord) -> None:
        threading.Thread(target=self._run, args=(job,), daemon=True).start()

    def cancel(self, job: JobRecord) -> None:
        with self._lock:
            machine_name = self._machines.get(job.id) or job.machine_name
        self.store.transition(
            job.id,
            expected=(JobStatus.QUEUED, JobStatus.PROVISIONING, JobStatus.RUNNING),
            status=JobStatus.CANCELLED,
            error="cancelled by user",
            completed_at=datetime.now(UTC),
        )
        if machine_name:
            self._remove_machine(machine_name)

    def reconcile(self, job: JobRecord) -> JobRecord:
        return self.store.get(job.id) or job

    def release(self, job: JobRecord) -> None:
        if not self.keep_machines and job.machine_name:
            self._remove_machine(job.machine_name)

    def _run(self, job: JobRecord) -> None:
        machine_name = f"vector-{job.id[:12]}"
        started = datetime.now(UTC)
        request = job.request
        provider_size = request.machine_size.value
        with self._lock:
            self._machines[job.id] = machine_name
        try:
            self.store.transition(
                job.id,
                expected=(JobStatus.QUEUED,),
                status=JobStatus.PROVISIONING,
                machine_name=machine_name,
                provider_id=machine_name,
                started_at=started,
            )
            provider_size = self._create_machine(machine_name, request)
            self._wait_for_ssh(machine_name)
            self.store.transition(
                job.id,
                expected=(JobStatus.PROVISIONING,),
                status=JobStatus.RUNNING,
            )

            job_root = self.root / job.id
            inputs_root = job_root / "inputs"
            job_root.mkdir(parents=True, exist_ok=True)
            package_stage = self._stage_package(job_root)
            self._checked(
                [
                    self.binary,
                    "sync",
                    "push",
                    f"{package_stage}/",
                    f"{machine_name}:~/vector",
                ],
                timeout=1200,
            )
            if request.inputs:
                inputs_root.mkdir(parents=True, exist_ok=True)
                for input_artifact in request.inputs:
                    self._download_input(
                        str(input_artifact.url),
                        inputs_root / self._safe_name(input_artifact.name),
                    )
                self._checked(
                    [
                        self.binary,
                        "sync",
                        "push",
                        f"{inputs_root}/",
                        f"{machine_name}:~/vector/uploads",
                    ],
                    timeout=600,
                )

            task = self._remote_reference(request.task, request.inputs)
            agent = self._remote_reference(request.agent, request.inputs)
            remote_command = [
                "set -e",
                "sudo cloud-init status --wait >/dev/null",
                "sudo apt-get -o DPkg::Lock::Timeout=300 update -qq",
                "sudo DEBIAN_FRONTEND=noninteractive apt-get "
                "-o DPkg::Lock::Timeout=300 install -y -qq python3-venv",
                "cd ~/vector",
                "python3 -m venv .venv",
                ".venv/bin/pip install --disable-pip-version-check -e .",
                " ".join(
                    [
                        ".venv/bin/vector",
                        "run",
                        "-t",
                        shlex.quote(task),
                        "-a",
                        shlex.quote(agent),
                        "-n",
                        str(request.n),
                        "--out",
                        "~/vector-result",
                        *(
                            ["--registry", shlex.quote(request.registry)]
                            if request.registry
                            else []
                        ),
                    ]
                ),
            ]
            self._checked(
                [self.binary, "ssh", machine_name, " && ".join(remote_command)],
                timeout=request.estimated_minutes * 60 + 1800,
            )

            artifact = job_root / "result"
            self._checked(
                [
                    self.binary,
                    "sync",
                    "pull",
                    f"{machine_name}:~/vector-result",
                    str(artifact),
                ],
                timeout=900,
            )
            result_path = next(
                (
                    candidate
                    for candidate in (
                        artifact / "result.json",
                        artifact / "vector-result" / "result.json",
                    )
                    if candidate.is_file()
                ),
                None,
            )
            if result_path is None:
                raise TaskContractError("Machine0 run completed without result.json")
            artifact = result_path.parent
            result = json.loads(result_path.read_text(encoding="utf-8"))
            completed = datetime.now(UTC)
            runtime_seconds = max(1, math.ceil((completed - started).total_seconds()))
            provider_cost = self._provider_cost(provider_size, runtime_seconds)
            self.store.transition(
                job.id,
                expected=(JobStatus.RUNNING,),
                status=JobStatus.SUCCEEDED,
                artifact_path=str(artifact),
                result_head=str(result.get("head", "")),
                completed_at=completed,
                provider_cost_micros=provider_cost,
                runtime_seconds=runtime_seconds,
            )
        except Exception as exc:
            completed = datetime.now(UTC)
            runtime_seconds = max(1, math.ceil((completed - started).total_seconds()))
            provider_cost = self._provider_cost(provider_size, runtime_seconds)
            with suppress(TaskContractError):
                self.store.transition(
                    job.id,
                    expected=(JobStatus.QUEUED, JobStatus.PROVISIONING, JobStatus.RUNNING),
                    status=JobStatus.FAILED,
                    error=(f"{type(exc).__name__}: {exc}")[-4000:],
                    completed_at=completed,
                    provider_cost_micros=provider_cost,
                    runtime_seconds=runtime_seconds,
                )
        finally:
            if not self.keep_machines:
                self._remove_machine(machine_name)
            with self._lock:
                self._machines.pop(job.id, None)

    def _create_machine(self, machine_name: str, request: JobRequest) -> str:
        last_error = ""
        for provider_size in _MACHINE0_SIZE_CANDIDATES[request.machine_size]:
            result = self.runner(
                [
                    self.binary,
                    "new",
                    machine_name,
                    "--size",
                    provider_size,
                    "--region",
                    request.region,
                    "--image",
                    self._image_for_size(request.machine_size),
                    "--key",
                    self.key_name,
                ],
                timeout=900,
            )
            if result.returncode == 0:
                return provider_size
            last_error = result.stderr.strip() or result.stdout.strip()
            if "out of stock" not in last_error.lower():
                raise TaskContractError(last_error or "Machine0 VM creation failed")
        raise TaskContractError(last_error or "No requested hosted capacity is available")

    def _wait_for_ssh(self, machine_name: str) -> None:
        deadline = time.monotonic() + 600
        last_error = ""
        while time.monotonic() < deadline:
            try:
                result = self.runner(
                    [self.binary, "ssh", machine_name, "true"],
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                last_error = "SSH readiness check timed out"
                time.sleep(5)
                continue
            if result.returncode == 0:
                return
            last_error = result.stderr.strip() or result.stdout.strip()
            time.sleep(5)
        raise TaskContractError(f"Machine0 SSH did not become ready: {last_error}")

    def _checked(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        result = self.runner(command, cwd=cwd, timeout=timeout)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise TaskContractError(detail or f"{command[0]} exited {result.returncode}")
        return result

    def _download_input(self, url: str, destination: Path) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "https":
            raise TaskContractError("hosted inputs must use HTTPS")
        if self.allowed_input_host and parsed.hostname != self.allowed_input_host:
            raise TaskContractError("hosted input URL is not allowlisted")
        total = 0
        request = Request(url, headers={"User-Agent": "Vector-Cloud/1"})
        with urlopen(request, timeout=60) as response, destination.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 50 * 1024 * 1024:
                    raise TaskContractError("hosted input exceeds 50 MiB")
                output.write(chunk)

    def _stage_package(self, job_root: Path) -> Path:
        stage = job_root / "package"
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            source = self.package_root / name
            if source.is_file():
                shutil.copy2(source, stage / name)
        for relative in (Path("src"), Path("docs/examples")):
            source = self.package_root / relative
            if source.is_dir():
                shutil.copytree(
                    source,
                    stage / relative,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
        if not (stage / "pyproject.toml").is_file() or not (stage / "src").is_dir():
            raise TaskContractError("Vector package staging is incomplete")
        return stage

    @staticmethod
    def _safe_name(name: str) -> str:
        if name in {".", ".."} or "/" in name or "\\" in name:
            raise TaskContractError("invalid hosted input name")
        return name

    def _remote_reference(self, reference: str, inputs: tuple[InputArtifact, ...]) -> str:
        path = Path(reference)
        if path.is_absolute():
            try:
                return str(path.resolve().relative_to(self.package_root.resolve()))
            except ValueError:
                pass
        for artifact in inputs:
            if reference in {artifact.name, f"uploads/{artifact.name}"}:
                return f"uploads/{artifact.name}"
        return reference

    def _image_for_size(self, size: MachineSize) -> str:
        if size in {MachineSize.GPU_L40S, MachineSize.GPU_H100}:
            return self.gpu_image
        return self.image

    @staticmethod
    def _provider_cost(provider_size: str, runtime_seconds: int) -> int:
        return math.ceil(_MACHINE0_PRICE_PER_HOUR_MICROS[provider_size] * runtime_seconds / 3600)

    def _remove_machine(self, machine_name: str) -> None:
        with suppress(Exception):
            self.runner(
                [self.binary, "rm", machine_name, "--yes"],
                timeout=300,
            )


Transport = Callable[[Request, float], tuple[int, bytes]]

_RUNPOD_GPU = {
    ComputeClass.L4: "NVIDIA L4",
    ComputeClass.L40S: "NVIDIA L40S",
    ComputeClass.A100: "NVIDIA A100 80GB PCIe",
    ComputeClass.H100: "NVIDIA H100 PCIe",
}


class RunPodExecutor:
    """Provision the allowlisted Vector worker through RunPod's v2 Pods API."""

    def __init__(
        self,
        store: JobStore,
        *,
        api_key: str,
        callback_url: str,
        worker_image: str,
        registry_id: str = "",
        transport: Transport | None = None,
        base_url: str = "https://api.runpod.io",
    ) -> None:
        if not api_key:
            raise TaskContractError("RUNPOD_API_KEY is required for RunPod execution")
        if not callback_url.startswith("https://"):
            raise TaskContractError("RunPod callback URL must use HTTPS")
        if re.fullmatch(r".+@sha256:[0-9a-f]{64}", worker_image) is None:
            raise TaskContractError("Vector RunPod worker image must be pinned by sha256 digest")
        self.store = store
        self.api_key = api_key
        self.callback_url = callback_url.rstrip("/")
        self.worker_image = worker_image
        self.registry_id = registry_id
        self.transport = transport or _transport
        self.base_url = base_url.rstrip("/")

    def submit(self, job: JobRecord) -> None:
        request = job.request
        gpu_id = _RUNPOD_GPU.get(request.compute)
        if gpu_id is None:
            raise TaskContractError(f"unsupported RunPod compute class {request.compute.value!r}")
        callback_token = secrets.token_urlsafe(32)
        self.store.set_callback_token(job.id, callback_token)
        body: dict[str, object] = {
            "name": request.name or f"vector-{job.id[:12]}",
            "cloud": "SECURE",
            "image": self.worker_image,
            "args": "cloud worker",
            "disk": 20,
            "gpu": {"id": gpu_id, "count": 1},
            "env": {
                "VECTOR_CLOUD_CALLBACK_URL": self.callback_url,
                "VECTOR_CLOUD_CALLBACK_TOKEN": callback_token,
                "VECTOR_CLOUD_JOB_ID": job.id,
                "VECTOR_CLOUD_TASK": request.task,
                "VECTOR_CLOUD_AGENT": request.agent,
                "VECTOR_CLOUD_N": str(request.n),
                "VECTOR_CLOUD_REGISTRY": request.registry,
            },
        }
        if self.registry_id:
            body["registry"] = self.registry_id
        provider_id = ""
        try:
            response = self._request("POST", "/v2/pods", body)
            raw_provider_id = response.get("id")
            if not isinstance(raw_provider_id, str) or not raw_provider_id:
                raise TaskContractError("RunPod create response omitted pod id")
            provider_id = raw_provider_id
            try:
                self.store.transition(
                    job.id,
                    expected=(JobStatus.QUEUED,),
                    status=JobStatus.PROVISIONING,
                    provider_id=provider_id,
                )
            except TaskContractError:
                current = self.store.get(job.id)
                if current is None or current.status not in {
                    JobStatus.SUCCEEDED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                }:
                    raise
                with suppress(TaskContractError):
                    self._request("DELETE", f"/v2/pods/{provider_id}", None, expected=(204,))
                return
        except Exception:
            if provider_id:
                with suppress(TaskContractError):
                    self._request("DELETE", f"/v2/pods/{provider_id}", None, expected=(204,))
            self.store.clear_callback_token(job.id)
            raise

    def release(self, job: JobRecord) -> None:
        if job.provider_id:
            self._request("DELETE", f"/v2/pods/{job.provider_id}", None, expected=(204,))

    def cancel(self, job: JobRecord) -> None:
        self.release(job)
        self.store.transition(
            job.id,
            expected=(JobStatus.QUEUED, JobStatus.PROVISIONING, JobStatus.RUNNING),
            status=JobStatus.CANCELLED,
            error="cancelled by user",
            clear_callback=True,
        )

    def reconcile(self, job: JobRecord) -> JobRecord:
        current = self.store.get(job.id) or job
        if not current.provider_id or current.status in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            return current
        response = self._request("GET", f"/v2/pods/{current.provider_id}", None)
        provider_status = response.get("status")
        clear_callback = False
        if provider_status in {"PROVISIONING", "STARTING"}:
            status = JobStatus.PROVISIONING
            error = None
        elif provider_status == "RUNNING":
            status = JobStatus.RUNNING
            error = None
        elif provider_status in {"ERROR", "EXITED", "TERMINATED"}:
            status = JobStatus.FAILED
            error = f"RunPod worker ended with status {provider_status} before evidence callback"
            clear_callback = True
        else:
            return current
        try:
            return self.store.transition(
                job.id,
                expected=(JobStatus.PROVISIONING, JobStatus.RUNNING),
                status=status,
                error=error,
                clear_callback=clear_callback,
            )
        except TaskContractError:
            return self.store.get(job.id) or current

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None,
        *,
        expected: tuple[int, ...] = (200, 201),
    ) -> dict[str, object]:
        encoded = None if body is None else json.dumps(body).encode()
        request = Request(
            f"{self.base_url}{path}",
            data=encoded,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "VectorCloud/0.1",
            },
        )
        try:
            status, payload = self.transport(request, 30)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise TaskContractError(f"RunPod request failed: {exc}") from exc
        if status not in expected:
            detail = payload.decode(errors="replace")[:1000]
            raise TaskContractError(f"RunPod returned HTTP {status}: {detail}")
        if not payload:
            return {}
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise TaskContractError("RunPod returned a non-object response")
        return decoded


def _transport(request: Request, timeout: float) -> tuple[int, bytes]:
    with urlopen(request, timeout=timeout) as response:
        return response.status, response.read()
