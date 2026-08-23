"""SQLite persistence for Vector Cloud job records."""

from __future__ import annotations

import builtins
import hashlib
import hmac
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from or_audit.errors import TaskContractError

from .models import (
    ImportRecord,
    ImportRequest,
    JobRecord,
    JobRequest,
    JobStatus,
    UsageEvent,
    UsageSource,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL,
    request_json TEXT NOT NULL,
    provider_id TEXT NOT NULL DEFAULT '',
    artifact_path TEXT NOT NULL DEFAULT '',
    result_head TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    machine_name TEXT NOT NULL DEFAULT '',
    started_at TEXT,
    completed_at TEXT,
    provider_cost_micros INTEGER NOT NULL DEFAULT 0,
    runtime_seconds INTEGER NOT NULL DEFAULT 0,
    callback_token_hash TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS vector_imports (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL,
    kind TEXT NOT NULL,
    slug TEXT NOT NULL,
    ref TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    user_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    snapshot_sha TEXT NOT NULL DEFAULT '',
    resolved_path TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS vector_imports_created_at ON vector_imports(created_at DESC);
CREATE TABLE IF NOT EXISTS usage_events (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    unit TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_events_job_occurred ON usage_events(job_id, occurred_at);
CREATE INDEX IF NOT EXISTS jobs_created_at ON jobs(created_at DESC);
"""


class JobStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)")}
            additions = {
                "callback_token_hash": "TEXT NOT NULL DEFAULT ''",
                "machine_name": "TEXT NOT NULL DEFAULT ''",
                "started_at": "TEXT",
                "completed_at": "TEXT",
                "provider_cost_micros": "INTEGER NOT NULL DEFAULT 0",
                "runtime_seconds": "INTEGER NOT NULL DEFAULT 0",
            }
            for name, declaration in additions.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} {declaration}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create(self, request: JobRequest) -> JobRecord:
        record = JobRecord.new(request)
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO jobs
                   (id, created_at, updated_at, status, request_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    record.id,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.status.value,
                    request.model_dump_json(),
                ),
            )
        return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            record = _record(row)
            return record.model_copy(
                update={"usage_events": self._usage_events(connection, job_id)}
            )

    def list(self, *, limit: int = 100) -> tuple[JobRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return tuple(_record(row) for row in rows)

    def set_callback_token(self, job_id: str, token: str) -> None:
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET callback_token_hash = ?
                   WHERE id = ? AND status = ? AND callback_token_hash = ''""",
                (digest, job_id, JobStatus.QUEUED.value),
            )
            if cursor.rowcount != 1:
                raise TaskContractError(f"job {job_id!r} is not queued for callback setup")

    def verify_callback_token(self, job_id: str, token: str) -> bool:
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT callback_token_hash FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return (
            row is not None
            and bool(row["callback_token_hash"])
            and hmac.compare_digest(row["callback_token_hash"], digest)
        )

    def clear_callback_token(self, job_id: str) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE jobs SET callback_token_hash = '' WHERE id = ?", (job_id,))

    def transition(
        self,
        job_id: str,
        *,
        expected: tuple[JobStatus, ...],
        status: JobStatus,
        provider_id: str | None = None,
        artifact_path: str | None = None,
        result_head: str | None = None,
        error: str | None = None,
        machine_name: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        provider_cost_micros: int | None = None,
        runtime_seconds: int | None = None,
        callback_token: str | None = None,
        clear_callback: bool = False,
    ) -> JobRecord:
        if not expected:
            raise ValueError("transition requires at least one expected status")
        values: dict[str, object] = {
            "updated_at": datetime.now(UTC).isoformat(),
            "status": status.value,
        }
        if provider_id is not None:
            values["provider_id"] = provider_id
        if artifact_path is not None:
            values["artifact_path"] = artifact_path
        if result_head is not None:
            values["result_head"] = result_head
        if error is not None:
            values["error"] = error
        if machine_name is not None:
            values["machine_name"] = machine_name
        if started_at is not None:
            values["started_at"] = started_at.isoformat()
        if completed_at is not None:
            values["completed_at"] = completed_at.isoformat()
        if provider_cost_micros is not None:
            values["provider_cost_micros"] = provider_cost_micros
        if runtime_seconds is not None:
            values["runtime_seconds"] = runtime_seconds
        conditions = ["id = ?", f"status IN ({','.join('?' for _ in expected)})"]
        condition_values = [job_id, *(item.value for item in expected)]
        if callback_token is not None or clear_callback:
            values["callback_token_hash"] = ""
        if callback_token is not None:
            conditions.append("callback_token_hash = ?")
            condition_values.append(hashlib.sha256(callback_token.encode()).hexdigest())
        assignments = ", ".join(f"{field} = ?" for field in values)
        occurred = completed_at.isoformat() if completed_at else datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE jobs SET {assignments} WHERE {' AND '.join(conditions)}",
                (*values.values(), *condition_values),
            )
            if cursor.rowcount != 1:
                raise TaskContractError(f"job {job_id!r} state changed before transition")
            if status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED) and (
                runtime_seconds is not None or provider_cost_micros is not None
            ):
                if runtime_seconds is not None:
                    connection.execute(
                        "INSERT INTO usage_events "
                        "(id, job_id, occurred_at, unit, quantity, source, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            uuid4().hex,
                            job_id,
                            occurred,
                            "seconds",
                            runtime_seconds,
                            UsageSource.MEASURED.value,
                            occurred,
                        ),
                    )
                if provider_cost_micros is not None:
                    connection.execute(
                        "INSERT INTO usage_events "
                        "(id, job_id, occurred_at, unit, quantity, source, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            uuid4().hex,
                            job_id,
                            occurred,
                            "cost_micros",
                            provider_cost_micros,
                            UsageSource.ESTIMATED.value,
                            occurred,
                        ),
                    )
        record = self.get(job_id)
        if record is None:  # pragma: no cover - transition retains the primary key
            raise TaskContractError(f"unknown cloud job {job_id!r}")
        return record

    def append_usage_event(
        self,
        job_id: str,
        *,
        unit: str,
        quantity: int,
        source: UsageSource,
        occurred_at: datetime | None = None,
    ) -> UsageEvent:
        """Append one immutable usage fact (unit, quantity, provenance) to a job.

        This records evidence only. It does not change the authoritative
        `jobs.provider_cost_micros`/`runtime_seconds` scalars that billing
        reads; a provider-reported correction here is audited, not billed.
        """
        event = UsageEvent(
            id=uuid4().hex,
            job_id=job_id,
            occurred_at=occurred_at or datetime.now(UTC),
            unit=unit,
            quantity=quantity,
            source=source,
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO usage_events "
                "(id, job_id, occurred_at, unit, quantity, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    event.id,
                    event.job_id,
                    event.occurred_at.isoformat(),
                    event.unit,
                    event.quantity,
                    event.source.value,
                    event.occurred_at.isoformat(),
                ),
            )
        return event

    @staticmethod
    def _usage_events(connection: sqlite3.Connection, job_id: str) -> tuple[UsageEvent, ...]:
        rows = connection.execute(
            "SELECT * FROM usage_events WHERE job_id = ? ORDER BY occurred_at ASC, id ASC",
            (job_id,),
        ).fetchall()
        return tuple(_usage_event(row) for row in rows)

    def usage_events(self, job_id: str) -> tuple[UsageEvent, ...]:
        with self._connect() as connection:
            return self._usage_events(connection, job_id)

    def summarize_usage(self, *, job_id: str | None = None) -> builtins.list[dict[str, object]]:
        """Roll up the append-only evidence into per-job, per-unit, per-source totals.

        Audit/observability view only. Billing authority remains the scalar
        `jobs` columns (consumed by the worker's `reportUsage`); do not drive
        billing from this roll-up."""
        sql = (
            "SELECT job_id, unit, source, SUM(quantity) AS quantity, COUNT(*) AS events "
            "FROM usage_events"
        )
        params: list[object] = []
        if job_id is not None:
            sql += " WHERE job_id = ?"
            params.append(job_id)
        sql += " GROUP BY job_id, unit, source ORDER BY job_id, unit, source"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [
            {
                "job_id": row["job_id"],
                "unit": row["unit"],
                "source": row["source"],
                "quantity": row["quantity"],
                "events": row["events"],
            }
            for row in rows
        ]

    def create_import(self, request: ImportRequest) -> ImportRecord:
        now = datetime.now(UTC)
        record = ImportRecord(
            id=request.id,
            created_at=now,
            updated_at=now,
            status="pending",
        )
        with self._connect() as connection:
            connection.execute(
                (
                    "INSERT INTO vector_imports "
                    "(id, created_at, updated_at, status, kind, slug, ref, "
                    "target_kind, user_id, source_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                (
                    record.id,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.status,
                    request.kind.value,
                    request.slug,
                    request.ref,
                    request.target_kind.value,
                    request.user_id,
                    request.source_id,
                ),
            )
        return record

    def record_import_resolution(
        self,
        import_id: str,
        *,
        snapshot_sha: str = "",
        resolved_path: str = "",
        error: str = "",
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                (
                    "UPDATE vector_imports SET updated_at = ?, snapshot_sha = ?, "
                    "resolved_path = ?, status = CASE WHEN ? = '' THEN 'resolved' "
                    "ELSE 'failed' END, error = ? WHERE id = ?"
                ),
                (
                    datetime.now(UTC).isoformat(),
                    snapshot_sha,
                    resolved_path,
                    error,
                    error,
                    import_id,
                ),
            )
            if cursor.rowcount != 1:
                raise TaskContractError(f"unknown vector import {import_id!r}")

    def get_import(self, import_id: str) -> ImportRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM vector_imports WHERE id = ?", (import_id,)
            ).fetchone()
        return _import_record(row) if row is not None else None


def _record(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        id=row["id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        status=JobStatus(row["status"]),
        request=JobRequest.model_validate_json(row["request_json"]),
        provider_id=row["provider_id"],
        artifact_path=row["artifact_path"],
        result_head=row["result_head"],
        error=row["error"],
        machine_name=row["machine_name"],
        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        provider_cost_micros=row["provider_cost_micros"],
        runtime_seconds=row["runtime_seconds"],
    )


def _usage_event(row: sqlite3.Row) -> UsageEvent:
    return UsageEvent(
        id=row["id"],
        job_id=row["job_id"],
        occurred_at=datetime.fromisoformat(row["occurred_at"]),
        unit=row["unit"],
        quantity=row["quantity"],
        source=UsageSource(row["source"]),
    )


def _import_record(row: sqlite3.Row) -> ImportRecord:
    return ImportRecord(
        id=row["id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        status=row["status"],
        snapshot_sha=row["snapshot_sha"],
        error=row["error"],
        resolved_path=row["resolved_path"],
    )
