"""Commercial license-allowlist gate for SurgEval runtime dependencies.

SurgEval ships as open-core (Apache-2.0) with commercial/SaaS and
regulatory-attestation tiers, so every *runtime* dependency must carry a
commercial-friendly license. This script gates the resolved runtime closure
from ``uv.lock`` against a curated, reviewed table:

1. Every runtime dependency reachable from ``[project].dependencies`` must be
   present in :data:`THIRD_PARTY` (an explicit maintainer review — a new
   dependency simply failing here is the intended governance mechanism).
2. Every mapped license must be in :data:`ALLOWLIST` (permissive, sublicensable,
   no copyleft/commodity restrictions). MPL-2.0 is *not* in the runtime
   allowlist (file-level copyleft); it may appear only as a dev/build
   dependency, which this script does not gate.

The tables and the classification live in :mod:`or_audit.eval.licensing`, which
the Tier-1 conformance suite audits *wrapped worlds* with. Same data, same
verdicts: a license we would not accept in our own wheel is not one we can
redistribute a curated world under.

Run from the repo root: ``uv run python scripts/check_license_allowlist.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

from or_audit.eval.licensing import (
    ALLOWLIST,
    REPO_PACKAGE,
    THIRD_PARTY,
    audit_package_licenses,
)

__all__ = ["ALLOWLIST", "REPO_PACKAGE", "THIRD_PARTY", "main"]


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    audit = audit_package_licenses(
        pyproject=root / "pyproject.toml",
        lock=root / "uv.lock",
    )
    if not audit.ok:
        print("FAIL: SurgEval licensing gate", file=sys.stderr)
        for error in audit.errors:
            print("  -", error, file=sys.stderr)
        return 1

    print(f"OK: {len(audit.dependencies)} runtime dependencies within commercial allowlist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
