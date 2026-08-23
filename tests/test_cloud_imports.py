from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Literal

import huggingface_hub as hh
import pytest

from or_audit.cloud import api
from or_audit.cloud.models import ImportKind, ImportRequest, ImportTarget
from or_audit.cloud.store import JobStore
from or_audit.errors import TaskContractError


def _request(*, kind: ImportKind, slug: str, ref: str, target: ImportTarget) -> ImportRequest:
    return ImportRequest(
        id="import-1234",
        user_id="user-a",
        source_id="source-1",
        kind=kind,
        slug=slug,
        ref=ref,
        target_kind=target,
    )


def _task_tar(root: Path, *, include_instruction: bool = True) -> Path:
    src = root / "repo"
    src.mkdir(parents=True, exist_ok=True)
    (src / "task.toml").write_text("name = 'demo'\n", encoding="utf-8")
    if include_instruction:
        (src / "instruction.md").write_text("Complete the task.", encoding="utf-8")
    tar = root / "snapshot.tar.gz"
    with tarfile.open(tar, "w:gz") as archive:
        archive.add(src, arcname="repo")
    return tar


# --- store round-trip -------------------------------------------------------


def test_store_import_resolution_round_trip_success(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    request = _request(kind=ImportKind.GITHUB, slug="org/repo", ref="v1", target=ImportTarget.TASK)
    record = store.create_import(request)
    assert record.status == "pending"

    store.record_import_resolution(
        record.id,
        snapshot_sha="a" * 64,
        resolved_path="/data/imports/user-a/import-1234/snapshot.tar.gz",
    )
    got = store.get_import(record.id)
    assert got is not None
    assert got.status == "resolved"
    assert got.snapshot_sha == "a" * 64
    assert got.resolved_path.endswith("snapshot.tar.gz")
    assert got.error == ""


def test_store_import_resolution_round_trip_failure(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    record = store.create_import(
        _request(kind=ImportKind.GITHUB, slug="org/repo", ref="v1", target=ImportTarget.AGENT)
    )

    store.record_import_resolution(record.id, error="boom")
    got = store.get_import(record.id)
    assert got is not None
    assert got.status == "failed"
    assert got.error == "boom"
    assert got.snapshot_sha == ""


def test_store_import_resolution_unknown_id_raises(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    with pytest.raises(TaskContractError):
        store.record_import_resolution("nope", snapshot_sha="a" * 64)


# --- GitHub tar validation --------------------------------------------------


def test_validate_tar_accepts_task_manifest(tmp_path: Path) -> None:
    tar = _task_tar(tmp_path)
    api._validate_tar(tar, "task")  # must not raise


def test_validate_tar_rejects_missing_instruction(tmp_path: Path) -> None:
    tar = _task_tar(tmp_path, include_instruction=False)
    with pytest.raises(TaskContractError, match="instruction"):
        api._validate_tar(tar, "task")


def test_validate_tar_rejects_non_archive(tmp_path: Path) -> None:
    junk = tmp_path / "not-a-tar.gz"
    junk.write_bytes(b"<html>error page</html>")
    with pytest.raises(TaskContractError, match="archive"):
        api._validate_tar(junk, "task")


def test_resolve_import_github_pins_sha_and_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tar = _task_tar(tmp_path)
    artifact_root = tmp_path / "artifacts"

    def fake_download(url: str, target: Path, max_bytes: int) -> None:
        shutil.copyfile(tar, target)

    monkeypatch.setattr(api, "_download", fake_download)
    request = _request(kind=ImportKind.GITHUB, slug="org/repo", ref="v1", target=ImportTarget.TASK)

    digest, resolved = api.resolve_import(request, artifact_root)

    assert resolved.is_relative_to(artifact_root.resolve())
    assert resolved.name == "snapshot.tar.gz"
    assert digest == hashlib.sha256(tar.read_bytes()).hexdigest()
    assert resolved.read_bytes() == tar.read_bytes()


def test_resolve_import_rejects_unsafe_id(tmp_path: Path) -> None:
    request = ImportRequest(
        id="import-1234",
        user_id="../../escape",
        source_id="source-1",
        kind=ImportKind.GITHUB,
        slug="org/repo",
        ref="v1",
        target_kind=ImportTarget.TASK,
    )
    with pytest.raises(TaskContractError, match="safe path component"):
        api._import_destination(request, tmp_path)


# --- Hugging Face resolution (client mocked) -------------------------------


class _TreeFile:
    def __init__(self, size: int) -> None:
        self.size = size


def _hf_request(target: ImportTarget = ImportTarget.TASK) -> ImportRequest:
    return _request(kind=ImportKind.HUGGINGFACE, slug="org/model", ref="main", target=target)


def _monkeypatch_hf(monkeypatch: pytest.MonkeyPatch, *, sizes: list[int]) -> None:
    class FakeHfApi:
        def __init__(self) -> None:
            self.size = 0

        def list_repo_tree(
            self, repo_id, revision=None, repo_type=None, recursive=False
        ) -> list[object]:
            return [_TreeFile(size) for size in sizes]

    def fake_snapshot_download(**kwargs: object) -> None:
        local = Path(str(kwargs["local_dir"]))
        local.mkdir(parents=True, exist_ok=True)
        (local / "task.toml").write_text("name = 'demo'\n", encoding="utf-8")
        (local / "instruction.md").write_text("Do it.", encoding="utf-8")

    monkeypatch.setattr(hh, "HfApi", FakeHfApi)
    monkeypatch.setattr(hh, "snapshot_download", fake_snapshot_download)


def test_resolve_huggingface_downloads_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _monkeypatch_hf(monkeypatch, sizes=[10, 20])
    artifact_root = tmp_path / "artifacts"

    digest, resolved = api.resolve_import(_hf_request(), artifact_root)

    assert resolved.is_dir()
    assert resolved.is_relative_to(artifact_root.resolve())
    assert len(digest) == 64
    assert len(digest.encode()) == 64


def test_resolve_huggingface_rejects_over_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _monkeypatch_hf(monkeypatch, sizes=[api._MAX_IMPORT_BYTES + 1])
    with pytest.raises(TaskContractError, match="size limit"):
        api.resolve_import(_hf_request(), tmp_path / "artifacts")


# --- resolve callback (Supabase RPC) ---------------------------------------


class _FakeResp:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        return False


def test_post_resolve_sends_rpc_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout=30) -> _FakeResp:
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["apikey"] = req.headers.get("apikey") or req.headers.get("Apikey")
        captured["body"] = json.loads(req.data)
        return _FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    callback = api.ResolveCallback("https://db.supabase.co", "pub-key", "ctl-tok")
    request = _request(kind=ImportKind.GITHUB, slug="org/repo", ref="v1", target=ImportTarget.TASK)

    api._post_resolve(request, "a" * 64, "/data/snap", "", callback)

    assert captured["url"] == "https://db.supabase.co/rest/v1/rpc/resolve_vector_import"
    assert captured["method"] == "POST"
    assert captured["apikey"] == "pub-key"
    assert captured["body"] == {
        "p_control_token": "ctl-tok",
        "p_import_id": "import-1234",
        "p_snapshot_sha": "a" * 64,
        "p_resolved_path": "/data/snap",
        "p_error": "",
    }


def test_post_resolve_raises_after_retries_on_non2xx(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_urlopen(req, timeout=30) -> _FakeResp:
        calls.append(1)
        return _FakeResp(status=500)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    callback = api.ResolveCallback("https://db.supabase.co", "pub-key", "ctl-tok")
    request = _request(kind=ImportKind.GITHUB, slug="org/repo", ref="v1", target=ImportTarget.TASK)

    with pytest.raises(TaskContractError, match="resolve callback failed"):
        api._post_resolve(request, "a" * 64, "/data/snap", "", callback)
    assert len(calls) == 3


def test_post_resolve_requires_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=30: _FakeResp())
    request = _request(kind=ImportKind.GITHUB, slug="org/repo", ref="v1", target=ImportTarget.TASK)
    with pytest.raises(TaskContractError, match="not configured"):
        api._post_resolve(request, "a" * 64, "/data/snap", "", api.ResolveCallback("", "", ""))


def test_run_import_callbacks_error_on_resolution_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    junk = tmp_path / "bad.tar.gz"
    junk.write_bytes(b"<html>not an archive</html>")

    def fake_download(url: str, target: Path, max_bytes: int) -> None:
        shutil.copyfile(junk, target)

    monkeypatch.setattr(api, "_download", fake_download)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=30: captured.update(body=json.loads(req.data)) or _FakeResp(),
    )

    store = JobStore(tmp_path / "jobs.sqlite")
    request = _request(kind=ImportKind.GITHUB, slug="org/repo", ref="v1", target=ImportTarget.TASK)
    record = store.create_import(request)
    callback = api.ResolveCallback("https://db.supabase.co", "pub-key", "ctl-tok")

    api._run_import(request, record, tmp_path / "artifacts", store, callback)

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["p_error"] != ""
    got = store.get_import(record.id)
    assert got is not None
    assert got.status == "failed"
    assert got.error != ""


def _empty_tar(path: Path) -> Path:
    with tarfile.open(path, "w:gz"):
        pass
    return path


def test_import_url_requires_owner_repo(tmp_path: Path) -> None:
    request = _request(kind=ImportKind.GITHUB, slug="norepo", ref="v1", target=ImportTarget.TASK)
    with pytest.raises(TaskContractError, match="owner/repo"):
        api.resolve_import(request, tmp_path / "artifacts")


def test_validate_tar_rejects_missing_dataset_manifest(tmp_path: Path) -> None:
    tar = _task_tar(tmp_path)
    with pytest.raises(TaskContractError, match="manifest"):
        api._validate_tar(tar, "dataset")


def test_validate_tar_rejects_empty_archive(tmp_path: Path) -> None:
    tar = _empty_tar(tmp_path / "empty.tar.gz")
    with pytest.raises(TaskContractError, match="empty"):
        api._validate_tar(tar, "task")


def test_validate_dir_rejects_empty_directory(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(TaskContractError, match="empty"):
        api._validate_dir(empty, "task")


def test_resolve_huggingface_propagates_and_wraps_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _monkeypatch_hf(monkeypatch, sizes=[10])

    def boom_task_contract(**kwargs: object) -> None:
        raise TaskContractError("snapshot refused")

    monkeypatch.setattr(hh, "snapshot_download", boom_task_contract)
    with pytest.raises(TaskContractError, match="refused"):
        api.resolve_import(_hf_request(), tmp_path / "artifacts")

    def boom_generic(**kwargs: object) -> None:
        raise RuntimeError("socket error")

    monkeypatch.setattr(hh, "snapshot_download", boom_generic)
    with pytest.raises(TaskContractError, match="failed to resolve"):
        api.resolve_import(_hf_request(), tmp_path / "artifacts")


def test_resolve_huggingface_rejects_downloaded_over_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _monkeypatch_hf(monkeypatch, sizes=[10])
    monkeypatch.setattr(api, "_downloaded_total", lambda _path: api._MAX_IMPORT_BYTES + 1)
    with pytest.raises(TaskContractError, match="exceeds import size limit"):
        api.resolve_import(_hf_request(), tmp_path / "artifacts")


def test_submit_import_background_resolution_and_failed_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from or_audit.cloud.api import create_app

    store = JobStore(tmp_path / "jobs.sqlite")

    def fake_resolve(request: ImportRequest, root: Path) -> tuple[str, Path]:
        resolved = root / "resolved"
        resolved.mkdir(parents=True, exist_ok=True)
        (resolved / "task.toml").write_text("name = 'demo'\n", encoding="utf-8")
        return "a" * 64, resolved

    monkeypatch.setattr(api, "resolve_import", fake_resolve)
    client = TestClient(
        create_app(
            store=store,
            executors={},
            artifact_root=tmp_path / "data",
            token="secret",
        )
    )
    request = _request(kind=ImportKind.GITHUB, slug="org/repo", ref="v1", target=ImportTarget.TASK)

    response = client.post(
        "/v1/imports",
        json=request.model_dump(),
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 202
    record = store.get_import(request.id)
    assert record is not None
    assert record.status == "failed"
    assert "not configured" in record.error


def test_submit_import_records_generic_resolve_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from or_audit.cloud.api import create_app

    store = JobStore(tmp_path / "jobs.sqlite")

    def boom(request: ImportRequest, root: Path) -> tuple[str, Path]:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(api, "resolve_import", boom)
    client = TestClient(
        create_app(
            store=store,
            executors={},
            artifact_root=tmp_path / "data",
            token="secret",
        )
    )
    request = _request(kind=ImportKind.GITHUB, slug="org/repo", ref="v1", target=ImportTarget.TASK)

    client.post(
        "/v1/imports",
        json=request.model_dump(),
        headers={"Authorization": "Bearer secret"},
    )

    record = store.get_import(request.id)
    assert record is not None
    assert record.status == "failed"
    assert "disk on fire" in record.error


def test_post_resolve_retries_oserror_then_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urllib.request

    def always_oserror(*args: object, **kwargs: object) -> None:
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", always_oserror)
    request = _request(kind=ImportKind.GITHUB, slug="org/repo", ref="v1", target=ImportTarget.TASK)
    callback = api.ResolveCallback("https://db.example", "pub-key", "ctl")

    with pytest.raises(TaskContractError, match="resolve callback failed"):
        api._post_resolve(request, "abc", "/resolved", "", callback)
