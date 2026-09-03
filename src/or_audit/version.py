"""Component version identifiers.

Every attestation artifact records the versions that produced it, because
PLAN.md section 7.3 requires an immutable audit trail of score version,
model version, and decision-rule version. A score is not interpretable
without knowing what produced it.
"""

from __future__ import annotations

from typing import Final

#: Version of the SurgEval package as a whole.
PACKAGE_VERSION: Final = "0.3.0a11"

#: Version of the domain and record schema. Carried on every persisted audit
#: entry and covered by the entry hash, so a record is self-describing and a
#: later shape change cannot be applied retroactively without detection.
SCHEMA_VERSION: Final = "1"

#: Version of the audit-chain construction (hash inputs and ordering).
#: Bump invalidates chain verification of older logs, so treat as frozen.
AUDIT_CHAIN_VERSION: Final = "1"
