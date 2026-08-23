"""FastAPI application for the minimal Vector Cloud control plane."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import tarfile
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from or_audit.errors import TaskContractError
from or_audit.eval.job import JobResult, verify_head

from .executors import Executor, LocalExecutor, Machine0Executor, RunPodExecutor
from .models import ExecutorKind, ImportRecord, ImportRequest, JobRecord, JobRequest, JobStatus
from .store import JobStore

_MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
_MAX_MEMBER_BYTES = 100 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
_MAX_MEMBERS = 10_000


_MAX_IMPORT_BYTES = 200 * 1024 * 1024


def resolve_import(request: ImportRequest, artifact_root: Path) -> tuple[str, Path]:
    """Fetch and verify an import snapshot, returning (snapshot_sha, resolved_path).

    GitHub resolves owner/repo@ref to a gzip tarball (validated as an archive).
    Hugging Face downloads the repo snapshot through huggingface_hub into a
    directory, then that tree is validated for a target-kind manifest. The
    snapshot_sha is a deterministic hash over the resolved fixture.
    """
    if request.kind.value == "github":
        return _resolve_github(request, artifact_root)
    return _resolve_huggingface(request, artifact_root)


def _resolve_github(request: ImportRequest, artifact_root: Path) -> tuple[str, Path]:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "snapshot.tar.gz"
        _download(_import_url(request), target, _MAX_IMPORT_BYTES)
        _validate_tar(target, request.target_kind.value)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        destination = _import_destination(request, artifact_root, "snapshot.tar.gz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        target.replace(destination)
        return digest, destination


def _resolve_huggingface(request: ImportRequest, artifact_root: Path) -> tuple[str, Path]:
    destination = _import_destination(request, artifact_root)
    destination.mkdir(parents=True, exist_ok=True)
    repo_type = "dataset" if request.target_kind.value == "dataset" else "model"
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise TaskContractError("huggingface_hub is not installed on this control plane") from exc
    _preflight_hf_total(request, repo_type)
    try:
        snapshot_download(
            repo_id=request.slug,
            revision=request.ref,
            repo_type=repo_type,
            local_dir=destination,
        )
    except TaskContractError:
        raise
    except Exception as exc:
        raise TaskContractError(f"failed to resolve Hugging Face snapshot: {exc}") from exc
    total = _downloaded_total(destination)
    if total > _MAX_IMPORT_BYTES:
        raise TaskContractError("Hugging Face snapshot exceeds import size limit")
    _validate_dir(destination, request.target_kind.value)
    return _dir_sha256(destination), destination


def _preflight_hf_total(request: ImportRequest, repo_type: str) -> None:
    from huggingface_hub import HfApi

    total = 0
    for entry in HfApi().list_repo_tree(
        request.slug, revision=request.ref, repo_type=repo_type, recursive=True
    ):
        size = getattr(entry, "size", None)
        if size is not None:
            total += size
            if total > _MAX_IMPORT_BYTES:
                raise TaskContractError("Hugging Face snapshot exceeds import size limit")


def _downloaded_total(path: Path) -> int:
    total = 0
    for candidate in path.rglob("*"):
        if candidate.is_file() and ".cache" not in candidate.relative_to(path).parts:
            total += candidate.stat().st_size
    return total


def _import_url(request: ImportRequest) -> str:
    owner, _, repo = request.slug.partition("/")
    if not owner or not repo:
        raise TaskContractError("github slug must be owner/repo")
    return f"https://codeload.github.com/{owner}/{repo}/tar.gz/{request.ref}"


def _download(url: str, target: Path, max_bytes: int) -> None:
    import urllib.request

    with urllib.request.urlopen(url, timeout=60) as response:
        size = 0
        with target.open("wb") as handle:
            while chunk := response.read(64 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise TaskContractError("snapshot exceeds import size limit")
                handle.write(chunk)


_KIND_REQUIRED: dict[str, set[str]] = {
    "task": {"task.toml", "instruction.md"},
    "agent": {"agent.toml"},
}
# dataset accepts either a v0.2 dataset dir or a canonical taskset dir
_KIND_ANY: dict[str, set[str]] = {
    "dataset": {"taskset.toml", "dataset.toml"},
}


def _check_manifests(basenames: set[str], target_kind: str) -> None:
    required = _KIND_REQUIRED.get(target_kind, set())
    missing = required - basenames
    if missing:
        raise TaskContractError(
            f"import snapshot misses required {target_kind} files: {sorted(missing)}"
        )
    any_of = _KIND_ANY.get(target_kind, set())
    if any_of and not (basenames & any_of):
        raise TaskContractError(f"import snapshot contains no {target_kind} manifest")


def _validate_tar(path: Path, target_kind: str) -> None:
    import tarfile

    try:
        with tarfile.open(name=path, mode="r:gz") as archive:
            members = archive.getmembers()
            if not members:
                raise TaskContractError("import snapshot archive is empty")
            basenames = {m.name.rsplit("/", 1)[-1].lower() for m in members if m.isfile()}
            _check_manifests(basenames, target_kind)
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise TaskContractError("import ref is not a valid GitHub archive snapshot") from exc


def _validate_dir(path: Path, target_kind: str) -> None:
    files = [p for p in path.rglob("*") if p.is_file()]
    if not files:
        raise TaskContractError("import snapshot directory is empty")
    basenames = {p.name.lower() for p in files}
    _check_manifests(basenames, target_kind)


def _safe_component(value: str, label: str) -> str:
    cleaned = value.strip()
    if (
        not cleaned
        or cleaned in {".", ".."}
        or "/" in cleaned
        or "\\" in cleaned
        or "\x00" in cleaned
    ):
        raise TaskContractError(f"{label} is not a safe path component")
    return cleaned


def _import_destination(request: ImportRequest, artifact_root: Path, *parts: str) -> Path:
    user = _safe_component(request.user_id, "user_id")
    import_id = _safe_component(request.id, "import id")
    destination = artifact_root / "imports" / user / import_id
    destination = destination.joinpath(*parts) if parts else destination
    root = artifact_root.resolve()
    dest = destination.resolve()
    if not dest.is_relative_to(root):
        raise TaskContractError("resolved import path escapes artifact root")
    return destination


def _dir_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for rel, file_sha in _walk_digest(path):
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(file_sha.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _walk_digest(path: Path) -> Iterator[tuple[str, str]]:
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        rel_parts = candidate.relative_to(path).parts
        if ".cache" in rel_parts:
            continue
        file_sha = hashlib.sha256()
        with candidate.open("rb") as handle:
            while chunk := handle.read(64 * 1024):
                file_sha.update(chunk)
        yield "/".join(rel_parts), file_sha.hexdigest()


@dataclass(frozen=True)
class ResolveCallback:
    supabase_url: str
    publishable_key: str
    control_token: str


def _run_import(
    request: ImportRequest,
    record: ImportRecord,
    artifact_root: Path,
    store: JobStore,
    callback: ResolveCallback,
) -> None:
    snapshot_sha = ""
    resolved_path = ""
    error = ""
    try:
        snapshot_sha, resolved = resolve_import(request, artifact_root)
        resolved_path = str(resolved)
    except TaskContractError as exc:
        error = str(exc)
    except Exception as exc:
        error = str(exc)
    callback_error = ""
    try:
        _post_resolve(request, snapshot_sha, resolved_path, error, callback)
    except TaskContractError as exc:
        callback_error = str(exc)
    store.record_import_resolution(
        record.id,
        snapshot_sha=snapshot_sha,
        resolved_path=resolved_path,
        error=error or callback_error,
    )


def _post_resolve(
    request: ImportRequest,
    snapshot_sha: str,
    resolved_path: str,
    error: str,
    callback: ResolveCallback,
) -> None:
    if not callback.supabase_url or not callback.publishable_key:
        raise TaskContractError("resolve callback is not configured")
    import urllib.request

    payload = {
        "p_control_token": callback.control_token,
        "p_import_id": request.id,
        "p_snapshot_sha": snapshot_sha,
        "p_resolved_path": resolved_path,
        "p_error": error,
    }
    body = json.dumps(payload).encode()
    url = f"{callback.supabase_url.rstrip('/')}/rest/v1/rpc/resolve_vector_import"
    headers = {"apikey": callback.publishable_key, "content-type": "application/json"}
    last: OSError | None = None
    for _attempt in range(3):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, data=body, headers=headers, method="POST"),
                timeout=30,
            ) as resp:
                if 200 <= resp.status < 300:
                    return
                last = OSError(f"resolve callback returned HTTP {resp.status}")
        except OSError as exc:
            last = exc
    raise TaskContractError(f"resolve callback failed: {last}")


class WorkerFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error: str = Field(min_length=1, max_length=4000)


def create_app(
    *,
    store: JobStore,
    executors: Mapping[ExecutorKind, Executor],
    artifact_root: Path,
    token: str = "",
    allow_anonymous: bool = False,
    supabase_url: str = "",
    supabase_publishable_key: str = "",
) -> FastAPI:
    if not token and not allow_anonymous:
        raise TaskContractError(
            "VECTOR_CLOUD_TOKEN is required unless anonymous local development is explicit"
        )
    artifact_root.mkdir(parents=True, exist_ok=True)
    callback = ResolveCallback(
        supabase_url=supabase_url or os.environ.get("SUPABASE_URL", ""),
        publishable_key=supabase_publishable_key or os.environ.get("SUPABASE_PUBLISHABLE_KEY", ""),
        control_token=token,
    )
    app = FastAPI(
        title="Vector Cloud",
        version="0.1.0",
        description="Managed execution for replayable SurgEval evidence bundles.",
    )

    def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        if allow_anonymous:
            return
        _require_control_token(authorization, token)

    auth = Depends(authorize)

    def schedule_release(record: JobRecord, background_tasks: BackgroundTasks) -> None:
        executor = executors.get(record.request.executor)
        if executor is not None and record.provider_id:
            background_tasks.add_task(executor.release, record)

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/v1/jobs",
        response_model=JobRecord,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[auth],
    )
    def submit_job(request: JobRequest) -> JobRecord:
        executor = executors.get(request.executor)
        if executor is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"executor {request.executor.value!r} is not configured",
            )
        record = store.create(request)
        try:
            executor.submit(record)
        except TaskContractError as exc:
            store.transition(
                record.id,
                expected=(JobStatus.QUEUED,),
                status=JobStatus.FAILED,
                error=str(exc),
                clear_callback=True,
            )
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        return store.get(record.id) or record

    @app.post(
        "/v1/imports",
        response_model=ImportRecord,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[auth],
    )
    async def submit_import(
        request: ImportRequest, background_tasks: BackgroundTasks
    ) -> ImportRecord:
        record = store.create_import(request)
        background_tasks.add_task(_run_import, request, record, artifact_root, store, callback)
        return record

    @app.get("/v1/jobs", response_model=list[JobRecord], dependencies=[auth])
    def list_jobs(limit: Annotated[int, Query(ge=1, le=500)] = 100) -> list[JobRecord]:
        return list(store.list(limit=limit))

    @app.get("/v1/jobs/{job_id}", response_model=JobRecord, dependencies=[auth])
    def get_job(job_id: str) -> JobRecord:
        record = _require_job(store, job_id)
        executor = executors.get(record.request.executor)
        if executor is not None:
            try:
                record = executor.reconcile(record)
            except TaskContractError as exc:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
                ) from exc
        return record

    @app.post("/v1/jobs/{job_id}/cancel", response_model=JobRecord, dependencies=[auth])
    def cancel_job(job_id: str) -> JobRecord:
        record = _require_job(store, job_id)
        if record.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"job is already {record.status.value}",
            )
        executor = executors.get(record.request.executor)
        if executor is None:
            raise HTTPException(status_code=400, detail="job executor is not configured")
        try:
            executor.cancel(record)
        except TaskContractError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        return _require_job(store, job_id)

    @app.get("/v1/jobs/{job_id}/result", dependencies=[auth], response_class=FileResponse)
    def get_result(job_id: str) -> FileResponse:
        record = _require_job(store, job_id)
        if record.status is not JobStatus.SUCCEEDED or not record.artifact_path:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="job has no result")
        result = Path(record.artifact_path) / "result.json"
        if not result.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="result artifact missing",
            )
        return FileResponse(
            result,
            media_type="application/json",
            filename=f"{job_id}-result.json",
        )

    @app.post("/v1/internal/jobs/{job_id}/complete", response_model=JobRecord)
    async def complete_job(
        job_id: str,
        request: Request,
        background_tasks: BackgroundTasks,
        authorization: Annotated[str | None, Header()] = None,
        x_vector_result_head: Annotated[str | None, Header()] = None,
    ) -> JobRecord:
        record = _require_remote_job(store, job_id)
        callback_token = _require_callback_token(store, job_id, authorization)
        if record.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
            raise HTTPException(status_code=409, detail=f"job is already {record.status.value}")
        if (
            x_vector_result_head is None
            or re.fullmatch(r"[0-9a-f]{64}", x_vector_result_head) is None
        ):
            raise HTTPException(status_code=422, detail="invalid result head")
        job_root = artifact_root / job_id
        job_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=job_root, suffix=".tar.gz") as upload:
            total = 0
            async for chunk in request.stream():
                total += len(chunk)
                if total > _MAX_ARCHIVE_BYTES:
                    raise HTTPException(status_code=413, detail="evidence archive exceeds 100 MiB")
                upload.write(chunk)
            upload.flush()
            extracted = _extract_evidence(Path(upload.name), job_root)
        try:
            result = JobResult.model_validate_json(
                (extracted / "result.json").read_text(encoding="utf-8")
            )
            if not verify_head(result):
                raise TaskContractError("result head does not verify")
        except (ValidationError, TaskContractError) as exc:
            shutil.rmtree(extracted, ignore_errors=True)
            raise HTTPException(status_code=422, detail=f"invalid evidence result: {exc}") from exc
        if result.head != x_vector_result_head:
            shutil.rmtree(extracted, ignore_errors=True)
            raise HTTPException(status_code=422, detail="callback head does not match result.json")
        try:
            completed = store.transition(
                job_id,
                expected=(JobStatus.QUEUED, JobStatus.PROVISIONING, JobStatus.RUNNING),
                status=JobStatus.SUCCEEDED,
                artifact_path=str(extracted),
                result_head=x_vector_result_head,
                error="",
                callback_token=callback_token,
            )
            schedule_release(completed, background_tasks)
            return completed
        except TaskContractError as exc:
            shutil.rmtree(extracted, ignore_errors=True)
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/internal/jobs/{job_id}/fail", response_model=JobRecord)
    def fail_job(
        job_id: str,
        failure: WorkerFailure,
        background_tasks: BackgroundTasks,
        authorization: Annotated[str | None, Header()] = None,
    ) -> JobRecord:
        record = _require_remote_job(store, job_id)
        callback_token = _require_callback_token(store, job_id, authorization)
        if record.status in {JobStatus.SUCCEEDED, JobStatus.CANCELLED}:
            raise HTTPException(status_code=409, detail=f"job is already {record.status.value}")
        try:
            failed = store.transition(
                job_id,
                expected=(JobStatus.QUEUED, JobStatus.PROVISIONING, JobStatus.RUNNING),
                status=JobStatus.FAILED,
                error=failure.error,
                callback_token=callback_token,
            )
            schedule_release(failed, background_tasks)
            return failed
        except TaskContractError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app


def app_from_env() -> FastAPI:
    db = Path(os.environ.get("VECTOR_CLOUD_DB", ".vector-cloud/jobs.sqlite"))
    data = Path(os.environ.get("VECTOR_CLOUD_DATA", ".vector-cloud/jobs"))
    package_root = Path(os.environ.get("VECTOR_CLOUD_PACKAGE_ROOT", ".")).resolve()
    token = os.environ.get("VECTOR_CLOUD_TOKEN", "")
    allow_anonymous = os.environ.get("VECTOR_CLOUD_ALLOW_ANONYMOUS") == "1"
    enable_local = os.environ.get("VECTOR_CLOUD_ENABLE_LOCAL") == "1"
    if allow_anonymous and not enable_local:
        raise TaskContractError("anonymous mode is only available with local development execution")
    store = JobStore(db)
    executors: dict[ExecutorKind, Executor] = {}
    if enable_local:
        executors[ExecutorKind.LOCAL] = LocalExecutor(store, root=data, package_root=package_root)
    if os.environ.get("VECTOR_CLOUD_ENABLE_MACHINE0") == "1":
        machine0_binary = os.environ.get("VECTOR_CLOUD_MACHINE0_BINARY", "machine0")
        if shutil.which(machine0_binary) is None:
            raise TaskContractError(f"Machine0 CLI not found: {machine0_binary}")
        executors[ExecutorKind.MACHINE0] = Machine0Executor(
            store,
            root=data,
            package_root=package_root,
            image=os.environ.get("VECTOR_CLOUD_MACHINE0_IMAGE", "ubuntu-24-04-loaded"),
            gpu_image=os.environ.get("VECTOR_CLOUD_MACHINE0_GPU_IMAGE", "gpu-h100x1-base"),
            binary=machine0_binary,
            allowed_input_host=os.environ.get("VECTOR_CLOUD_INPUT_HOST", ""),
            key_name=os.environ.get("VECTOR_CLOUD_MACHINE0_KEY", "vector-service"),
            keep_machines=os.environ.get("VECTOR_CLOUD_MACHINE0_KEEP") == "1",
        )
    runpod_key = os.environ.get("RUNPOD_API_KEY", "")
    callback_url = os.environ.get("VECTOR_CLOUD_PUBLIC_URL", "")
    worker_image = os.environ.get("VECTOR_CLOUD_RUNPOD_IMAGE", "")
    registry_id = os.environ.get("VECTOR_CLOUD_RUNPOD_REGISTRY", "")
    if runpod_key:
        if not callback_url or not worker_image:
            raise TaskContractError(
                "RunPod requires VECTOR_CLOUD_PUBLIC_URL and VECTOR_CLOUD_RUNPOD_IMAGE"
            )
        executors[ExecutorKind.RUNPOD] = RunPodExecutor(
            store,
            api_key=runpod_key,
            callback_url=callback_url,
            worker_image=worker_image,
            registry_id=registry_id,
        )
    return create_app(
        store=store,
        executors=executors,
        artifact_root=data,
        token=token,
        allow_anonymous=allow_anonymous,
    )


def _require_control_token(authorization: str | None, token: str) -> None:
    expected = f"Bearer {token}"
    if not token or authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _require_callback_token(store: JobStore, job_id: str, authorization: str | None) -> str:
    prefix = "Bearer "
    token = (
        authorization[len(prefix) :] if authorization and authorization.startswith(prefix) else ""
    )
    if not token or not store.verify_callback_token(job_id, token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid job callback token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


def _require_job(store: JobStore, job_id: str) -> JobRecord:
    record = store.get(job_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return record


def _require_remote_job(store: JobStore, job_id: str) -> JobRecord:
    record = _require_job(store, job_id)
    if record.request.executor is not ExecutorKind.RUNPOD:
        raise HTTPException(status_code=409, detail="job is not a remote execution")
    return record


def _extract_evidence(archive_path: Path, job_root: Path) -> Path:
    destination = job_root / "result"
    lock = job_root / ".callback.lock"
    try:
        lock.touch(exist_ok=False)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="evidence callback already active") from exc
    staging = Path(tempfile.mkdtemp(dir=job_root, prefix=".staging-"))
    try:
        if destination.exists():
            raise HTTPException(status_code=409, detail="result evidence already exists")
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > _MAX_MEMBERS:
                raise HTTPException(status_code=422, detail="evidence archive has too many members")
            total = 0
            for member in members:
                path = PurePosixPath(member.name)
                total += member.size
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not path.parts
                    or path.parts[0] != "result"
                    or not (member.isdir() or member.isfile())
                    or member.size > _MAX_MEMBER_BYTES
                    or total > _MAX_UNCOMPRESSED_BYTES
                ):
                    raise HTTPException(status_code=422, detail="unsafe evidence archive")
            for member in members:
                target = staging.joinpath(*PurePosixPath(member.name).parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise HTTPException(status_code=422, detail="unreadable evidence member")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
        staged_result = staging / "result"
        if not (staged_result / "result.json").is_file():
            raise HTTPException(status_code=422, detail="evidence archive omitted result.json")
        os.rename(staged_result, destination)
        return destination
    except (tarfile.TarError, OSError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid evidence archive: {exc}") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        lock.unlink(missing_ok=True)
