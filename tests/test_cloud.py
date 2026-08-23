from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from or_audit.cloud import worker
from or_audit.cloud.api import create_app
from or_audit.cloud.executors import LocalExecutor, Machine0Executor, RunPodExecutor
from or_audit.cloud.models import (
    ComputeClass,
    DataClassification,
    ExecutorKind,
    JobRecord,
    JobRequest,
    JobStatus,
    MachineSize,
    UsageEvent,
    UsageSource,
)
from or_audit.cloud.store import JobStore
from or_audit.errors import TaskContractError

ROOT = Path(__file__).resolve().parents[1]
VIDEO_TASK = ROOT / "docs" / "examples" / "tasks" / "video-nextstep"
VIDEO_AGENT = ROOT / "docs" / "examples" / "agents" / "example-video-predictor"
PINNED_IMAGE = "registry.example/vector-worker@sha256:" + "a" * 64


class RecordingExecutor:
    def __init__(self) -> None:
        self.submitted: list[str] = []
        self.released: list[str] = []

    def submit(self, job: JobRecord) -> None:
        self.submitted.append(job.id)

    def cancel(self, job: JobRecord) -> None:
        del job

    def release(self, job: JobRecord) -> None:
        self.released.append(job.id)

    def reconcile(self, job: JobRecord) -> JobRecord:
        return job


def test_store_persists_job_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite"
    request = JobRequest(task="task", agent="agent")
    created = JobStore(path).create(request)

    loaded = JobStore(path).get(created.id)

    assert loaded is not None
    assert loaded.request == request
    assert loaded.status is JobStatus.QUEUED


def test_terminal_transition_is_compare_and_set(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    record = store.create(JobRequest(task="task", agent="agent"))
    store.transition(
        record.id,
        expected=(JobStatus.QUEUED,),
        status=JobStatus.CANCELLED,
    )

    with pytest.raises(TaskContractError, match="state changed"):
        store.transition(
            record.id,
            expected=(JobStatus.QUEUED, JobStatus.RUNNING),
            status=JobStatus.SUCCEEDED,
        )


def test_hosted_request_refuses_phi_confidential_and_unversioned_packages() -> None:
    with pytest.raises(ValidationError, match="data_classification"):
        JobRequest.model_validate({"task": "t", "agent": "a", "data_classification": "phi"})
    with pytest.raises(ValidationError, match="public or deidentified"):
        JobRequest(
            task="org/task@1",
            agent="org/agent@1",
            executor=ExecutorKind.RUNPOD,
            compute=ComputeClass.L4,
            data_classification=DataClassification.CONFIDENTIAL,
        )
    with pytest.raises(ValidationError, match="versioned registry"):
        JobRequest(
            task="local-task",
            agent="local-agent",
            executor=ExecutorKind.RUNPOD,
            compute=ComputeClass.L4,
        )
    with pytest.raises(ValidationError, match="public or deidentified"):
        JobRequest(
            task="task",
            agent="agent",
            executor=ExecutorKind.MACHINE0,
            data_classification=DataClassification.CONFIDENTIAL,
        )
    with pytest.raises(ValidationError, match="disagree"):
        JobRequest(
            task="task",
            agent="agent",
            executor=ExecutorKind.MACHINE0,
            compute=ComputeClass.L40S,
            machine_size=MachineSize.LARGE,
        )


def test_runpod_executor_refuses_mutable_worker_image(tmp_path: Path) -> None:
    with pytest.raises(TaskContractError, match="sha256"):
        RunPodExecutor(
            JobStore(tmp_path / "jobs.sqlite"),
            api_key="token",
            callback_url="https://vector.example",
            worker_image="registry.example/vector-worker:latest",
        )


def test_api_requires_bearer_token_and_persists_submission(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    executor = RecordingExecutor()
    app = create_app(
        store=store,
        executors={ExecutorKind.LOCAL: executor},
        artifact_root=tmp_path / "data",
        token="secret-token",
    )
    client = TestClient(app)
    payload = {"task": "task", "agent": "agent", "n": 2}

    assert client.post("/v1/jobs", json=payload).status_code == 401
    response = client.post(
        "/v1/jobs",
        json=payload,
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 202
    job = response.json()
    assert executor.submitted == [job["id"]]
    assert (
        client.get(
            f"/v1/jobs/{job['id']}",
            headers={"Authorization": "Bearer secret-token"},
        ).status_code
        == 200
    )
    assert len(JobStore(tmp_path / "jobs.sqlite").list()) == 1


def test_local_executor_runs_real_cli_and_writes_result(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    completed = _run_local_video(store, tmp_path)

    assert completed.status is JobStatus.SUCCEEDED, completed.error
    assert completed.result_head
    assert Path(completed.artifact_path, "result.json").is_file()


def test_local_executor_persists_unexpected_failure(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    executor = LocalExecutor(store, root=tmp_path / "data", package_root=ROOT)
    record = store.create(JobRequest(task="does-not-exist", agent="random"))

    executor.submit(record)
    completed = _wait_for_terminal(store, record.id)

    assert completed.status is JobStatus.FAILED
    assert completed.error


def test_machine0_executor_provisions_isolated_vm_and_records_cost(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    commands: list[list[str]] = []

    def runner(
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, timeout
        commands.append(command)
        if command[1:3] == ["sync", "push"] and command[-1].endswith(":~/vector"):
            stage = Path(command[3])
            assert (stage / "pyproject.toml").is_file()
            assert (stage / "src").is_dir()
            assert not (stage / ".env.production").exists()
            assert not (stage / ".git").exists()
        if command[1:3] == ["sync", "pull"]:
            destination = Path(command[-1]) / "vector-result"
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "result.json").write_text('{"head":"vector-head"}', encoding="utf-8")
            trial = destination / "trial-0"
            trial.mkdir()
            (trial / "result.json").write_text('{"head":"trial-head"}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    executor = Machine0Executor(
        store,
        root=tmp_path / "data",
        package_root=ROOT,
        runner=runner,
    )
    assert executor._image_for_size(MachineSize.GPU_L40S) == "gpu-h100x1-base"
    record = store.create(
        JobRequest(
            task=str(VIDEO_TASK),
            agent=str(VIDEO_AGENT),
            executor=ExecutorKind.MACHINE0,
            machine_size=MachineSize.LARGE,
        )
    )

    executor._run(record)
    completed = store.get(record.id)

    assert completed is not None
    assert completed.status is JobStatus.SUCCEEDED, completed.error
    assert all("--json" not in command for command in commands if command[1] == "new")
    assert completed.provider_cost_micros > 0
    assert completed.runtime_seconds > 0
    assert any(command[1] == "new" for command in commands)
    assert any(
        command[command.index("--key") + 1] == "vector-service"
        for command in commands
        if command[1] == "new"
    )
    assert any(command[1] == "rm" for command in commands)
    assert any("python3-venv" in command[-1] for command in commands if command[1] == "ssh")


def test_machine0_gpu_capacity_falls_back_to_compatible_size(tmp_path: Path) -> None:
    requested_sizes: list[str] = []

    def runner(
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, timeout
        requested_sizes.append(command[command.index("--size") + 1])
        if requested_sizes[-1] == "gpu-l40s-1":
            return subprocess.CompletedProcess(command, 1, "", "GPU is currently out of stock")
        return subprocess.CompletedProcess(command, 0, "", "")

    executor = Machine0Executor(
        JobStore(tmp_path / "jobs.sqlite"),
        root=tmp_path / "data",
        package_root=ROOT,
        runner=runner,
    )
    request = JobRequest(
        task="task",
        agent="agent",
        executor=ExecutorKind.MACHINE0,
        compute=ComputeClass.L40S,
        machine_size=MachineSize.GPU_L40S,
    )

    selected = executor._create_machine("vector-test", request)

    assert selected == "gpu-6000ada-1"
    assert requested_sizes == ["gpu-l40s-1", "gpu-6000ada-1"]


def test_machine0_ssh_readiness_retries_command_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = 0

    def runner(
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal attempts
        del cwd, timeout
        attempts += 1
        if attempts == 1:
            raise subprocess.TimeoutExpired(command, 30)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("or_audit.cloud.executors.time.sleep", lambda _seconds: None)
    executor = Machine0Executor(
        JobStore(tmp_path / "jobs.sqlite"),
        root=tmp_path / "data",
        package_root=ROOT,
        runner=runner,
    )

    executor._wait_for_ssh("vector-test")

    assert attempts == 2


def test_runpod_executor_sends_secure_allowlisted_worker_request(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    seen: list[tuple[str, str, dict[str, object]]] = []

    def transport(request: Request, timeout: float) -> tuple[int, bytes]:
        assert timeout == 30
        assert request.get_header("Authorization") == "Bearer runpod-token"
        assert request.get_header("User-agent") == "VectorCloud/0.1"
        assert isinstance(request.data, bytes)
        body = json.loads(request.data)
        assert isinstance(body, dict)
        seen.append((request.get_method(), request.full_url, body))
        return 201, b'{"id":"pod_123","status":"PROVISIONING"}'

    executor = RunPodExecutor(
        store,
        api_key="runpod-token",
        callback_url="https://vector.example",
        worker_image=PINNED_IMAGE,
        registry_id="reg_private",
        transport=transport,
    )
    record = store.create(
        JobRequest(
            task="seldingermed/video-nextstep@0",
            agent="example/video-predictor@0",
            executor=ExecutorKind.RUNPOD,
            compute=ComputeClass.L40S,
            data_classification=DataClassification.DEIDENTIFIED,
        )
    )

    executor.submit(record)
    updated = store.get(record.id)
    body = seen[0][2]
    env = body["env"]

    assert updated is not None
    assert updated.status is JobStatus.PROVISIONING
    assert updated.provider_id == "pod_123"
    assert seen[0][0:2] == ("POST", "https://api.runpod.io/v2/pods")
    assert body["cloud"] == "SECURE"
    assert body["image"] == PINNED_IMAGE
    assert body["args"] == "cloud worker"
    assert body["gpu"] == {"id": "NVIDIA L40S", "count": 1}
    assert body["registry"] == "reg_private"
    assert isinstance(env, dict)
    callback_token = env["VECTOR_CLOUD_CALLBACK_TOKEN"]
    assert isinstance(callback_token, str)
    assert store.verify_callback_token(record.id, callback_token)


def test_remote_callback_can_complete_before_provisioning_and_only_once(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    local = _run_local_video(store, tmp_path)
    remote = store.create(
        JobRequest(
            task="seldingermed/video-nextstep@0",
            agent="example/video-predictor@0",
            executor=ExecutorKind.RUNPOD,
            compute=ComputeClass.L4,
        )
    )
    store.set_callback_token(remote.id, "one-time-token")
    app = create_app(
        store=store,
        executors={},
        artifact_root=tmp_path / "remote-data",
        token="control-token",
    )
    client = TestClient(app)
    archive = _archive(Path(local.artifact_path))
    headers = {
        "Authorization": "Bearer one-time-token",
        "X-Vector-Result-Head": local.result_head,
        "Content-Type": "application/gzip",
    }

    response = client.post(
        f"/v1/internal/jobs/{remote.id}/complete",
        content=archive,
        headers=headers,
    )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "succeeded"
    assert not store.verify_callback_token(remote.id, "one-time-token")
    assert (
        client.post(
            f"/v1/internal/jobs/{remote.id}/complete",
            content=archive,
            headers=headers,
        ).status_code
        == 401
    )
    result = client.get(
        f"/v1/jobs/{remote.id}/result",
        headers={"Authorization": "Bearer control-token"},
    )
    assert result.status_code == 200


def test_remote_failure_callback_is_job_scoped_and_terminal(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    remote = store.create(
        JobRequest(
            task="seldingermed/video-nextstep@0",
            agent="example/video-predictor@0",
            executor=ExecutorKind.RUNPOD,
            compute=ComputeClass.L4,
        )
    )
    store.set_callback_token(remote.id, "failure-token")
    store.transition(
        remote.id,
        expected=(JobStatus.QUEUED,),
        status=JobStatus.PROVISIONING,
        provider_id="pod_123",
    )
    executor = RecordingExecutor()
    client = TestClient(
        create_app(
            store=store,
            executors={ExecutorKind.RUNPOD: executor},
            artifact_root=tmp_path / "data",
            token="control-token",
        )
    )

    response = client.post(
        f"/v1/internal/jobs/{remote.id}/fail",
        json={"error": "model process exited 7"},
        headers={"Authorization": "Bearer failure-token"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["error"] == "model process exited 7"
    assert not store.verify_callback_token(remote.id, "failure-token")
    assert executor.released == [remote.id]


def test_remote_callback_rejects_unsafe_archive_without_consuming_token(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    remote = store.create(
        JobRequest(
            task="seldingermed/video-nextstep@0",
            agent="example/video-predictor@0",
            executor=ExecutorKind.RUNPOD,
            compute=ComputeClass.L4,
        )
    )
    store.set_callback_token(remote.id, "archive-token")
    client = TestClient(
        create_app(
            store=store,
            executors={},
            artifact_root=tmp_path / "data",
            token="control-token",
        )
    )
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        member = tarfile.TarInfo("../escape")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))

    response = client.post(
        f"/v1/internal/jobs/{remote.id}/complete",
        content=output.getvalue(),
        headers={
            "Authorization": "Bearer archive-token",
            "X-Vector-Result-Head": "a" * 64,
            "Content-Type": "application/gzip",
        },
    )

    assert response.status_code == 422
    assert store.verify_callback_token(remote.id, "archive-token")
    assert not (tmp_path / "escape").exists()


def test_api_configuration_fails_closed_without_token(tmp_path: Path) -> None:
    with pytest.raises(TaskContractError, match="VECTOR_CLOUD_TOKEN"):
        create_app(
            store=JobStore(tmp_path / "jobs.sqlite"),
            executors={},
            artifact_root=tmp_path / "data",
        )


def test_runpod_reconcile_and_cancel_are_terminal_compare_and_set(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    calls: list[str] = []
    provider_status = "RUNNING"

    def transport(request: Request, timeout: float) -> tuple[int, bytes]:
        del timeout
        calls.append(request.get_method())
        if request.get_method() == "POST":
            return 201, b'{"id":"pod_123","status":"PROVISIONING"}'
        if request.get_method() == "GET":
            return 200, json.dumps({"id": "pod_123", "status": provider_status}).encode()
        return 204, b""

    executor = RunPodExecutor(
        store,
        api_key="token",
        callback_url="https://vector.example",
        worker_image=PINNED_IMAGE,
        transport=transport,
    )
    record = store.create(
        JobRequest(
            task="seldingermed/video-nextstep@0",
            agent="example/video-predictor@0",
            executor=ExecutorKind.RUNPOD,
            compute=ComputeClass.L4,
        )
    )
    executor.submit(record)
    running = executor.reconcile(store.get(record.id) or record)
    assert running.status is JobStatus.RUNNING

    executor.cancel(running)
    cancelled = store.get(record.id)
    assert cancelled is not None
    assert cancelled.status is JobStatus.CANCELLED
    assert calls == ["POST", "GET", "DELETE"]


def test_runpod_submit_failure_clears_job_callback_token(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    callback_token = ""

    def transport(request: Request, timeout: float) -> tuple[int, bytes]:
        nonlocal callback_token
        del timeout
        assert isinstance(request.data, bytes)
        body = json.loads(request.data)
        callback_token = body["env"]["VECTOR_CLOUD_CALLBACK_TOKEN"]
        return 500, b"provider unavailable"

    executor = RunPodExecutor(
        store,
        api_key="token",
        callback_url="https://vector.example",
        worker_image=PINNED_IMAGE,
        transport=transport,
    )
    record = store.create(
        JobRequest(
            task="seldingermed/video-nextstep@0",
            agent="example/video-predictor@0",
            executor=ExecutorKind.RUNPOD,
            compute=ComputeClass.L4,
        )
    )

    with pytest.raises(TaskContractError, match="HTTP 500"):
        executor.submit(record)
    assert callback_token
    assert not store.verify_callback_token(record.id, callback_token)


def test_worker_runs_job_and_posts_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    completed = _run_local_video(store, tmp_path)
    source = Path(completed.artifact_path)
    posted: dict[str, object] = {}
    commands: list[list[str]] = []
    _set_worker_env(monkeypatch)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        commands.append(command)
        destination = Path(command[command.index("--out") + 1]) / source.name
        shutil.copytree(source, destination)
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_post(url: str, token: str, payload: bytes, head: str) -> None:
        posted.update(url=url, token=token, payload=payload, head=head)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(worker, "_post_archive", fake_post)

    assert worker.run_from_env() == 0
    assert commands[0][commands[0].index("run") + 1 :][:2] == [
        "-s",
        "seldingermed/video-nextstep@0",
    ]
    assert posted["token"] == "callback-token"
    assert posted["head"] == completed.result_head
    assert str(posted["url"]).endswith("/v1/internal/jobs/job-123/complete")
    assert isinstance(posted["payload"], bytes)


def test_worker_refuses_multi_task_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    posted: dict[str, object] = {}
    _set_worker_env(monkeypatch)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        destination = Path(command[command.index("--out") + 1])
        for task_id in ("first", "second"):
            result = destination / task_id / "result.json"
            result.parent.mkdir(parents=True)
            result.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_post(url: str, token: str, body: dict[str, str]) -> None:
        posted.update(url=url, token=token, body=body)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(worker, "_post_json", fake_post)

    assert worker.run_from_env() == 1
    assert posted["body"] == {"error": "worker expected one task result, found 2"}


def test_worker_reports_cli_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posted: dict[str, object] = {}
    _set_worker_env(monkeypatch)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(command, 7, "", "model failed")

    def fake_post(url: str, token: str, body: dict[str, str]) -> None:
        posted.update(url=url, token=token, body=body)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(worker, "_post_json", fake_post)

    assert worker.run_from_env() == 7
    assert posted["body"] == {"error": "model failed"}
    assert str(posted["url"]).endswith("/v1/internal/jobs/job-123/fail")


def test_worker_refuses_missing_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "VECTOR_CLOUD_CALLBACK_URL",
        "VECTOR_CLOUD_CALLBACK_TOKEN",
        "VECTOR_CLOUD_JOB_ID",
        "VECTOR_CLOUD_TASK",
        "VECTOR_CLOUD_AGENT",
    ):
        monkeypatch.delenv(name, raising=False)
    assert worker.run_from_env() == 2


def _set_worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "VECTOR_CLOUD_CALLBACK_URL": "https://vector.example",
        "VECTOR_CLOUD_CALLBACK_TOKEN": "callback-token",
        "VECTOR_CLOUD_JOB_ID": "job-123",
        "VECTOR_CLOUD_TASK": "seldingermed/video-nextstep@0",
        "VECTOR_CLOUD_AGENT": "example/video-predictor@0",
        "VECTOR_CLOUD_N": "1",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _run_local_video(store: JobStore, tmp_path: Path) -> JobRecord:
    executor = LocalExecutor(store, root=tmp_path / "data", package_root=ROOT)
    record = store.create(JobRequest(task=str(VIDEO_TASK), agent=str(VIDEO_AGENT), n=1))
    executor.submit(record)
    return _wait_for_terminal(store, record.id)


def _archive(result_dir: Path) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        archive.add(result_dir, arcname="result")
    return output.getvalue()


def _wait_for_terminal(store: JobStore, job_id: str) -> JobRecord:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        record = store.get(job_id)
        assert record is not None
        if record.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
            return record
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not complete")


# --- append-only usage-event ledger ---------------------------------------


def test_store_appends_usage_events_on_terminal_transition(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    record = store.create(JobRequest(task="task", agent="agent"))
    completed = store.transition(
        record.id,
        expected=(JobStatus.QUEUED,),
        status=JobStatus.SUCCEEDED,
        completed_at=datetime.now(UTC),
        provider_cost_micros=12345,
        runtime_seconds=300,
    )
    events = {e.unit: e for e in completed.usage_events}
    assert set(events) == {"seconds", "cost_micros"}
    assert events["seconds"].quantity == 300
    assert events["seconds"].source is UsageSource.MEASURED
    assert events["cost_micros"].quantity == 12345
    assert events["cost_micros"].source is UsageSource.ESTIMATED
    assert len(completed.usage_events) == 2


def test_usage_events_are_append_only(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    record = store.create(JobRequest(task="task", agent="agent"))
    store.transition(
        record.id,
        expected=(JobStatus.QUEUED,),
        status=JobStatus.FAILED,
        completed_at=datetime.now(UTC),
        provider_cost_micros=10,
        runtime_seconds=120,
    )
    store.append_usage_event(
        record.id, unit="cost_micros", quantity=7, source=UsageSource.PROVIDER_REPORTED
    )
    events = store.usage_events(record.id)
    assert len(events) == 3  # prior events preserved, not overwritten
    assert events[-1].source is UsageSource.PROVIDER_REPORTED
    assert events[-1].quantity == 7


def test_summarize_usage_groups_by_job_unit_source(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    first = store.create(JobRequest(task="task", agent="agent"))
    second = store.create(JobRequest(task="task", agent="agent"))
    store.transition(
        first.id,
        expected=(JobStatus.QUEUED,),
        status=JobStatus.SUCCEEDED,
        completed_at=datetime.now(UTC),
        provider_cost_micros=1000,
        runtime_seconds=60,
    )
    store.transition(
        second.id,
        expected=(JobStatus.QUEUED,),
        status=JobStatus.SUCCEEDED,
        completed_at=datetime.now(UTC),
        provider_cost_micros=2500,
        runtime_seconds=180,
    )
    summary = store.summarize_usage()
    by_job = {(row["job_id"], row["unit"]): row for row in summary}
    assert by_job[(first.id, "cost_micros")]["quantity"] == 1000
    assert by_job[(first.id, "seconds")]["quantity"] == 60
    assert by_job[(second.id, "cost_micros")]["quantity"] == 2500
    assert by_job[(second.id, "seconds")]["quantity"] == 180
    one = store.summarize_usage(job_id=first.id)
    assert {row["job_id"] for row in one} == {first.id}


def test_append_usage_event_roundtrip(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    record = store.create(JobRequest(task="task", agent="agent"))
    event = store.append_usage_event(
        record.id, unit="tokens", quantity=1500, source=UsageSource.MEASURED
    )
    assert isinstance(event, UsageEvent)
    loaded = store.usage_events(record.id)
    assert loaded[0].id == event.id
    assert loaded[0].quantity == 1500
    assert loaded[0].source is UsageSource.MEASURED


class FailingSubmitExecutor:
    def submit(self, job: JobRecord) -> None:
        raise TaskContractError("provisioning sold out")

    def cancel(self, job: JobRecord) -> None:
        del job

    def release(self, job: JobRecord) -> None:
        del job

    def reconcile(self, job: JobRecord) -> JobRecord:
        raise TaskContractError("remote reconcile failed")


def _remote_job(store: JobStore) -> JobRecord:
    return store.create(
        JobRequest(
            task="seldingermed/video-nextstep@0",
            agent="example/video-predictor@0",
            executor=ExecutorKind.RUNPOD,
            compute=ComputeClass.L4,
        )
    )


def test_submit_job_surfaces_executor_failure_as_502(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    client = TestClient(
        create_app(
            store=store,
            executors={ExecutorKind.LOCAL: FailingSubmitExecutor()},
            artifact_root=tmp_path / "data",
            token="secret",
        )
    )

    response = client.post(
        "/v1/jobs",
        json={"task": "task", "agent": "agent"},
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 502


def test_submit_job_refuses_unconfigured_executor(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    client = TestClient(
        create_app(store=store, executors={}, artifact_root=tmp_path / "data", token="secret")
    )

    response = client.post(
        "/v1/jobs",
        json={"task": "task", "agent": "agent"},
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 400


def test_cancel_job_refuses_terminal_state(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    terminal = store.create(JobRequest(task="task", agent="agent"))
    store.transition(terminal.id, expected=(JobStatus.QUEUED,), status=JobStatus.CANCELLED)
    client = TestClient(
        create_app(
            store=store,
            executors={ExecutorKind.LOCAL: RecordingExecutor()},
            artifact_root=tmp_path / "data",
            token="secret",
        )
    )

    response = client.post(
        f"/v1/jobs/{terminal.id}/cancel", headers={"Authorization": "Bearer secret"}
    )

    assert response.status_code == 409


def test_cancel_job_refuses_unconfigured_executor(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    remote = _remote_job(store)
    client = TestClient(
        create_app(store=store, executors={}, artifact_root=tmp_path / "data", token="secret")
    )

    response = client.post(
        f"/v1/jobs/{remote.id}/cancel", headers={"Authorization": "Bearer secret"}
    )

    assert response.status_code == 400


def test_get_job_returns_404_and_502_on_reconcile_failure(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    client = TestClient(
        create_app(
            store=store,
            executors={ExecutorKind.LOCAL: FailingSubmitExecutor()},
            artifact_root=tmp_path / "data",
            token="secret",
        )
    )
    headers = {"Authorization": "Bearer secret"}

    assert client.get("/v1/jobs/missing", headers=headers).status_code == 404

    local = store.create(JobRequest(task="task", agent="agent"))
    assert client.get(f"/v1/jobs/{local.id}", headers=headers).status_code == 502


def test_list_jobs_returns_records(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    store.create(JobRequest(task="task", agent="agent"))
    store.create(JobRequest(task="task", agent="agent"))
    client = TestClient(
        create_app(store=store, executors={}, artifact_root=tmp_path / "data", token="secret")
    )

    response = client.get("/v1/jobs", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_get_result_refuses_non_terminal_and_missing_file(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    pending = store.create(JobRequest(task="task", agent="agent"))
    empty = store.create(JobRequest(task="task", agent="agent"))
    store.transition(
        empty.id,
        expected=(JobStatus.QUEUED,),
        status=JobStatus.SUCCEEDED,
        artifact_path=str(tmp_path / "no-result"),
    )
    (tmp_path / "no-result").mkdir(exist_ok=True)
    client = TestClient(
        create_app(store=store, executors={}, artifact_root=tmp_path / "data", token="secret")
    )
    headers = {"Authorization": "Bearer secret"}

    assert client.get(f"/v1/jobs/{pending.id}/result", headers=headers).status_code == 409
    assert client.get(f"/v1/jobs/{empty.id}/result", headers=headers).status_code == 404


def test_fail_callback_refuses_already_terminal_job(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    remote = _remote_job(store)
    store.set_callback_token(remote.id, "ok")
    store.transition(
        remote.id,
        expected=(JobStatus.QUEUED,),
        status=JobStatus.SUCCEEDED,
        artifact_path=str(tmp_path / "result"),
    )
    client = TestClient(
        create_app(store=store, executors={}, artifact_root=tmp_path / "data", token="secret")
    )

    response = client.post(
        f"/v1/internal/jobs/{remote.id}/fail",
        json={"error": "boom"},
        headers={"Authorization": "Bearer ok"},
    )

    assert response.status_code == 409


def test_complete_callback_refuses_non_remote_and_bad_credentials(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    local = store.create(JobRequest(task="task", agent="agent"))
    remote = _remote_job(store)
    store.set_callback_token(remote.id, "hello")
    client = TestClient(
        create_app(store=store, executors={}, artifact_root=tmp_path / "data", token="secret")
    )

    assert (
        client.post(
            f"/v1/internal/jobs/{local.id}/complete",
            content=b"",
            headers={"X-Vector-Result-Head": "0" * 64},
        ).status_code
        == 409
    )
    assert (
        client.post(
            f"/v1/internal/jobs/{remote.id}/complete",
            content=b"",
            headers={"X-Vector-Result-Head": "0" * 64},
        ).status_code
        == 401
    )
    assert (
        client.post(
            f"/v1/internal/jobs/{remote.id}/complete",
            content=b"",
            headers={
                "Authorization": "Bearer hello",
                "X-Vector-Result-Head": "not-a-hex-head",
            },
        ).status_code
        == 422
    )


def test_control_token_required_and_verified(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    client = TestClient(
        create_app(
            store=store,
            executors={ExecutorKind.LOCAL: RecordingExecutor()},
            artifact_root=tmp_path / "data",
            token="secret",
        )
    )

    assert client.get("/v1/jobs").status_code == 401
    assert client.get("/v1/jobs", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_healthz_and_anonymous_mode(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    client = TestClient(
        create_app(
            store=store,
            executors={ExecutorKind.LOCAL: RecordingExecutor()},
            artifact_root=tmp_path / "data",
            token="",
            allow_anonymous=True,
        )
    )

    assert client.get("/healthz").status_code == 200
    assert client.post("/v1/jobs", json={"task": "t", "agent": "a"}).status_code == 202


def test_complete_callback_rejects_archive_without_result(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    remote = _remote_job(store)
    store.set_callback_token(remote.id, "tok")
    store.transition(remote.id, expected=(JobStatus.QUEUED,), status=JobStatus.RUNNING)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        info = tarfile.TarInfo("result/")
        info.type = tarfile.DIRTYPE
        archive.addfile(info)
    client = TestClient(
        create_app(store=store, executors={}, artifact_root=tmp_path / "data", token="secret")
    )

    response = client.post(
        f"/v1/internal/jobs/{remote.id}/complete",
        content=buf.getvalue(),
        headers={
            "Authorization": "Bearer tok",
            "X-Vector-Result-Head": "0" * 64,
        },
    )

    assert response.status_code == 422
