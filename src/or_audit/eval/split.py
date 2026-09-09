"""Independent-case split manifests and grouping validation (Phase D1).

Enforces patient/site-disjoint splits, stable case and episode bindings,
and explicit independent-unit counting so adjacent frames or repeated
episodes cannot inflate statistical degrees of freedom.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from or_audit.errors import TaskContractError

SplitName = Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=r"^[a-z0-9_-]+$")]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=128)]
DisjointUnit = Literal["case", "patient", "site"]


class SplitCaseEntry(BaseModel):
    """Binding for one coherent case/episode in a split."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: Identifier
    episode_id: Identifier
    split: SplitName
    patient_id: Identifier | None = None
    site_id: Identifier | None = None
    scenario_id: Identifier | None = None
    item_ids: tuple[Identifier, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _validate_items(self) -> Self:
        if len(set(self.item_ids)) != len(self.item_ids):
            raise TaskContractError(
                f"case entry {self.case_id!r} has duplicate item_ids: {self.item_ids}"
            )
        return self


class SplitManifest(BaseModel):
    """Authoritative split specification with patient/site disjoint enforcement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format_version: str = "1"
    dataset_id: Identifier
    dataset_revision: str
    entries: tuple[SplitCaseEntry, ...]
    disjoint_by: tuple[DisjointUnit, ...] = ("case",)

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        if not self.entries:
            raise TaskContractError(f"split manifest for {self.dataset_id} contains no entries")

        # 1. No item may belong to multiple case entries
        seen_items: dict[str, str] = {}
        for entry in self.entries:
            for item in entry.item_ids:
                if item in seen_items:
                    raise TaskContractError(
                        f"item {item!r} is assigned to multiple cases: "
                        f"{seen_items[item]!r} and {entry.case_id!r}"
                    )
                seen_items[item] = entry.case_id

        # 2. Case IDs cannot cross split boundaries
        case_to_splits: dict[str, set[str]] = defaultdict(set)
        for entry in self.entries:
            case_to_splits[entry.case_id].add(entry.split)
        for case_id, splits in case_to_splits.items():
            if len(splits) > 1:
                raise TaskContractError(
                    f"case {case_id!r} crosses multiple splits: {sorted(splits)}"
                )

        # 3. Disjoint unit enforcement across splits
        if "patient" in self.disjoint_by:
            patient_to_splits: dict[str, set[str]] = defaultdict(set)
            for entry in self.entries:
                if not entry.patient_id:
                    raise TaskContractError(
                        f"split manifest declares disjoint_by='patient' but case "
                        f"{entry.case_id!r} is missing patient_id"
                    )
                patient_to_splits[entry.patient_id].add(entry.split)
            for patient_id, splits in patient_to_splits.items():
                if len(splits) > 1:
                    raise TaskContractError(
                        f"patient overlap across splits: patient {patient_id!r} "
                        f"appears in {sorted(splits)}"
                    )

        if "site" in self.disjoint_by:
            site_to_splits: dict[str, set[str]] = defaultdict(set)
            for entry in self.entries:
                if not entry.site_id:
                    raise TaskContractError(
                        f"split manifest declares disjoint_by='site' but case "
                        f"{entry.case_id!r} is missing site_id"
                    )
                site_to_splits[entry.site_id].add(entry.split)
            for site_id, splits in site_to_splits.items():
                if len(splits) > 1:
                    raise TaskContractError(
                        f"site overlap across splits: site {site_id!r} appears in {sorted(splits)}"
                    )

        return self

    @property
    def supports_patient_disjoint(self) -> bool:
        """Whether all entries carry pseudonymous patient identifiers."""
        return all(bool(entry.patient_id) for entry in self.entries)

    @property
    def supports_site_disjoint(self) -> bool:
        """Whether all entries carry pseudonymous site identifiers."""
        return all(bool(entry.site_id) for entry in self.entries)

    def validate_against_input_items(
        self, input_ids: Iterable[str], *, strict: bool = False
    ) -> None:
        """Verify that input items are mapped in the manifest."""
        expected_set = set(input_ids)
        manifest_items: set[str] = set()
        for entry in self.entries:
            manifest_items.update(entry.item_ids)

        missing_from_manifest = expected_set - manifest_items
        if missing_from_manifest:
            raise TaskContractError(
                f"split manifest for {self.dataset_id} missing items from inputs: "
                f"{sorted(missing_from_manifest)}"
            )
        if strict:
            extra_in_manifest = manifest_items - expected_set
            if extra_in_manifest:
                raise TaskContractError(
                    f"split manifest for {self.dataset_id} contains items not in inputs: "
                    f"{sorted(extra_in_manifest)}"
                )

    def independent_case_count(self, split: SplitName, unit: DisjointUnit = "case") -> int:
        """Count distinct independent statistical units in a declared split."""
        matching = [entry for entry in self.entries if entry.split == split]
        if not matching:
            return 0
        if unit == "patient":
            if not self.supports_patient_disjoint:
                raise TaskContractError(
                    f"dataset {self.dataset_id} cannot count independent patients: "
                    "one or more entries lack patient_id"
                )
            return len({entry.patient_id for entry in matching if entry.patient_id})
        if unit == "site":
            if not self.supports_site_disjoint:
                raise TaskContractError(
                    f"dataset {self.dataset_id} cannot count independent sites: "
                    "one or more entries lack site_id"
                )
            return len({entry.site_id for entry in matching if entry.site_id})
        return len({entry.case_id for entry in matching})

    def items_for_split(self, split: SplitName) -> tuple[str, ...]:
        """Return all item IDs belonging to the given split."""
        items: list[str] = []
        for entry in self.entries:
            if entry.split == split:
                items.extend(entry.item_ids)
        return tuple(items)


def load_split_manifest(path: Path) -> SplitManifest:
    """Load a split manifest from a JSON file."""
    if not path.is_file():
        raise TaskContractError(f"split manifest file not found: {path}")
    content = path.read_text(encoding="utf-8")
    try:
        data = json.loads(content)
    except Exception as exc:
        raise TaskContractError(f"failed to parse split manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TaskContractError(f"split manifest at {path} is not an object")
    return SplitManifest.model_validate(data)


def assert_disjoint_stage_manifests(
    manifests: dict[str, SplitManifest],
    *,
    require_patient_disjoint: bool = False,
    require_site_disjoint: bool = False,
) -> None:
    """Validate that named stage manifests do not leak cases, patients, or sites across stages."""
    if len(manifests) <= 1:
        return

    # Check source-data identity alignment
    first_name, first_m = next(iter(manifests.items()))
    for name, m in manifests.items():
        if m.dataset_id != first_m.dataset_id:
            raise TaskContractError(
                f"stage manifest {name!r} has dataset_id {m.dataset_id!r}, "
                f"differing from stage {first_name!r} ({first_m.dataset_id!r})"
            )
        if m.dataset_revision != first_m.dataset_revision:
            raise TaskContractError(
                f"stage manifest {name!r} has dataset_revision {m.dataset_revision!r}, "
                f"differing from stage {first_name!r} ({first_m.dataset_revision!r})"
            )

    # Check case, patient, site disjointness across stage manifests
    case_to_stages: dict[str, set[str]] = defaultdict(set)
    patient_to_stages: dict[str, set[str]] = defaultdict(set)
    site_to_stages: dict[str, set[str]] = defaultdict(set)

    for stage_name, manifest in manifests.items():
        for entry in manifest.entries:
            case_to_stages[entry.case_id].add(stage_name)
            if entry.patient_id:
                patient_to_stages[entry.patient_id].add(stage_name)
            elif require_patient_disjoint:
                raise TaskContractError(
                    f"stage {stage_name!r} case {entry.case_id!r} is missing patient_id "
                    "required for patient-disjoint stage validation"
                )
            if entry.site_id:
                site_to_stages[entry.site_id].add(stage_name)
            elif require_site_disjoint:
                raise TaskContractError(
                    f"stage {stage_name!r} case {entry.case_id!r} is missing site_id "
                    "required for site-disjoint stage validation"
                )

    for case_id, stages in case_to_stages.items():
        if len(stages) > 1:
            raise TaskContractError(
                f"case {case_id!r} appears across multiple stages: {sorted(stages)}"
            )

    if require_patient_disjoint:
        for patient_id, stages in patient_to_stages.items():
            if len(stages) > 1:
                raise TaskContractError(
                    f"patient {patient_id!r} appears across multiple stages: {sorted(stages)}"
                )

    if require_site_disjoint:
        for site_id, stages in site_to_stages.items():
            if len(stages) > 1:
                raise TaskContractError(
                    f"site {site_id!r} appears across multiple stages: {sorted(stages)}"
                )
