"""Install-and-launch ladder (next.md N10).

The bounce risk for an open harness is not the license, it is installation:
SOFA needs specific builds, Isaac Sim cannot be redistributed, MuJoCo and
PyBullet are pip-friendly, Unity worlds are their own runtime. This package is
the ladder that makes those differences the harness's problem instead of the
user's:

* :mod:`or_audit.install.catalog` — what each world is and how it installs,
  loaded from a packaged TOML into frozen models.
* :mod:`or_audit.install.installer` — the explicit argv a given strategy would
  run, planned separately from running it.
* :mod:`or_audit.install.doctor` — per-check diagnosis that prints the fix.

Nothing here fabricates a pin, a digest, or a license. Where the catalog
records a gap, the planner refuses and names it.
"""

from __future__ import annotations

from or_audit.install.catalog import (
    UNVERIFIED,
    Disposition,
    InstallSpec,
    InstallStrategy,
    PinnedPackage,
    WorldCatalog,
    WorldPackage,
    iter_packages,
    load_catalog,
    world_package,
)
from or_audit.install.doctor import (
    CheckStatus,
    DoctorCheck,
    DoctorReport,
    find_reference_paths,
    require_reference_paths,
    run_doctor,
)
from or_audit.install.installer import (
    InstallOutcome,
    InstallPlan,
    InstallStep,
    Runner,
    execute_install,
    plan_install,
)

__all__ = [
    "UNVERIFIED",
    "CheckStatus",
    "Disposition",
    "DoctorCheck",
    "DoctorReport",
    "InstallOutcome",
    "InstallPlan",
    "InstallSpec",
    "InstallStep",
    "InstallStrategy",
    "PinnedPackage",
    "Runner",
    "WorldCatalog",
    "WorldPackage",
    "execute_install",
    "find_reference_paths",
    "iter_packages",
    "load_catalog",
    "plan_install",
    "require_reference_paths",
    "run_doctor",
    "world_package",
]
