"""Public contracts for the minimal Vector Cloud control plane."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Self
from uuid import uuid4

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator


class ExecutorKind(StrEnum):
    LOCAL = "local"
    RUNPOD = "runpod"
    MACHINE0 = "machine0"


class ComputeClass(StrEnum):
    CPU = "cpu"
    L4 = "l4"
    L40S = "l40s"
    A100 = "a100-80gb"
    H100 = "h100-pcie"


class MachineSize(StrEnum):
    LARGE = "large"
    XL = "xl"
    XXL = "xxl"
    XXXL = "xxxl"
    GPU_L40S = "gpu-l40s-1"
    GPU_H100 = "gpu-h100-1"


class InputArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: AnyHttpUrl
    name: str = Field(min_length=1, max_length=200, pattern=r"^[^/\\]+$")
    type: str = Field(default="application/octet-stream", max_length=120)
    size: int = Field(default=0, ge=0, le=50 * 1024 * 1024)


class DataClassification(StrEnum):
    PUBLIC = "public"
    DEIDENTIFIED = "deidentified"
    CONFIDENTIAL = "confidential"


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobRequest(BaseModel):
    """One task-agent evaluation request.

    PHI is intentionally not a valid classification for the hosted beta.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: str = Field(min_length=1)
    agent: str = Field(min_length=1)
    name: str = Field(default="", max_length=120)
    n: int = Field(default=1, ge=1, le=10_000)
    executor: ExecutorKind = ExecutorKind.LOCAL
    compute: ComputeClass = ComputeClass.CPU
    data_classification: DataClassification = DataClassification.DEIDENTIFIED
    registry: str = ""
    machine_size: MachineSize = MachineSize.LARGE
    region: str = Field(default="us-east", pattern=r"^(us-east|us-west|uk|eu|asia)$")
    estimated_minutes: int = Field(default=10, ge=1, le=1440)
    inputs: tuple[InputArtifact, ...] = Field(default=(), max_length=20)

    @model_validator(mode="after")
    def validate_executor(self) -> Self:
        if self.executor is ExecutorKind.RUNPOD:
            if self.data_classification is DataClassification.CONFIDENTIAL:
                raise ValueError("hosted RunPod beta accepts public or deidentified data only")
            if self.compute is ComputeClass.CPU:
                raise ValueError("RunPod jobs require a GPU compute class")
            if "@" not in self.task or "@" not in self.agent:
                raise ValueError("RunPod jobs require versioned registry task and agent references")
        if self.executor is ExecutorKind.MACHINE0:
            if self.data_classification is DataClassification.CONFIDENTIAL:
                raise ValueError("hosted Vector accepts public or deidentified data only")
            gpu_size = self.machine_size in {MachineSize.GPU_L40S, MachineSize.GPU_H100}
            if gpu_size != (self.compute is not ComputeClass.CPU):
                raise ValueError("machine size and compute class disagree")
        return self


class JobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    created_at: datetime
    updated_at: datetime
    status: JobStatus
    request: JobRequest
    provider_id: str = ""
    artifact_path: str = ""
    result_head: str = ""
    error: str = ""
    machine_name: str = ""
    started_at: datetime | None = None
    completed_at: datetime | None = None
    provider_cost_micros: int = Field(default=0, ge=0)
    runtime_seconds: int = Field(default=0, ge=0)
    # Append-only provenance evidence; billing reads the scalar columns above.
    usage_events: tuple[UsageEvent, ...] = ()

    @classmethod
    def new(cls, request: JobRequest) -> JobRecord:
        now = datetime.now(UTC)
        return cls(
            id=uuid4().hex,
            created_at=now,
            updated_at=now,
            status=JobStatus.QUEUED,
            request=request,
        )


class UsageSource(StrEnum):
    MEASURED = "measured"
    PROVIDER_REPORTED = "provider_reported"
    ESTIMATED = "estimated"


class UsageEvent(BaseModel):
    """One immutable usage fact (unit, quantity, provenance) for a job.

    This table is PROVENANCE EVIDENCE, not the billing source of truth: it is
    evidence associated with the job's usage/cost calculation. `source` says
    how each quantity was obtained (wall-clock measured, directly reported by
    the provider, or estimated from a price table) so provider-reported
    figures are never conflated with derived estimates. Events never update
    the authoritative scalar columns on `jobs` (`provider_cost_micros`,
    `runtime_seconds`) and never rewrite history; a provider-reported
    correction here is audited, not billed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    job_id: str
    occurred_at: datetime
    unit: str = Field(min_length=1, max_length=64)
    quantity: int = Field(ge=0)
    source: UsageSource


class ImportKind(StrEnum):
    GITHUB = "github"
    HUGGINGFACE = "huggingface"


class ImportTarget(StrEnum):
    TASK = "task"
    AGENT = "agent"
    DATASET = "dataset"


class ImportRequest(BaseModel):
    """A pending snapshot import to fetch, verify, and pin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=64)
    user_id: str = Field(min_length=1, max_length=128)
    source_id: str = Field(min_length=1, max_length=64)
    kind: ImportKind
    slug: str = Field(min_length=1, max_length=500)
    ref: str = Field(min_length=1, max_length=500)
    target_kind: ImportTarget
    callback_url: AnyHttpUrl | None = None


class ImportRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    created_at: datetime
    updated_at: datetime
    status: str
    snapshot_sha: str = ""
    error: str = ""
    resolved_path: str = ""
