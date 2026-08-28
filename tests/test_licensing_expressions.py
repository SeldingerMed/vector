"""SPDX expression evaluation: the structure of a declaration is load-bearing.

``GPL-3.0-only AND (MIT OR Apache-2.0)`` and ``GPL-3.0-only AND MIT OR
Apache-2.0`` are different licenses. The first imposes the GPL under every
choice the licensee has; the second offers Apache-2.0 as a way out. A
classifier that discards the parentheses reads the first as the second and
clears copyleft for redistribution, so these tests pin the grouping, the
precedence, and the refusal of anything that does not parse.
"""

from __future__ import annotations

import pytest

from or_audit.errors import TaskContractError
from or_audit.eval.licensing import LicenseStatus, classify_license, is_allowed
from or_audit.install.catalog import (
    Disposition,
    InstallStrategy,
    PinnedPackage,
    PipExtraInstall,
    WorldPackage,
)

#: The parenthesized probe: copyleft applies under either alternative.
GROUPED_COPYLEFT = "GPL-3.0-only AND (MIT OR Apache-2.0)"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        # Bare identifiers, case-insensitive, from both reviewed tables.
        ("MIT", LicenseStatus.ALLOWED),
        ("mit", LicenseStatus.ALLOWED),
        ("GPL-3.0-only", LicenseStatus.RESTRICTED),
        # Disjunction: the licensee picks, so one permissive branch suffices.
        ("MIT OR GPL-3.0-only", LicenseStatus.ALLOWED),
        ("GPL-3.0-only OR MIT", LicenseStatus.ALLOWED),
        ("AGPL-3.0-only OR SSPL-1.0", LicenseStatus.RESTRICTED),
        ("GPL-3.0-only OR Nonexistent-1.0", LicenseStatus.UNKNOWN),
        # Conjunction: every obligation applies at once.
        ("MIT AND Apache-2.0", LicenseStatus.ALLOWED),
        ("MIT AND GPL-3.0-only", LicenseStatus.RESTRICTED),
        ("MIT AND Nonexistent-1.0", LicenseStatus.UNKNOWN),
        ("BSD-3-Clause AND MIT AND 0BSD AND Zlib AND CC0-1.0", LicenseStatus.ALLOWED),
        # Grouping vs. precedence: AND binds tighter, so the parentheses in the
        # first expression change which obligations survive the licensee's choice.
        (GROUPED_COPYLEFT, LicenseStatus.RESTRICTED),
        ("GPL-3.0-only AND MIT OR Apache-2.0", LicenseStatus.ALLOWED),
        ("(GPL-3.0-only OR MIT) AND (Apache-2.0 OR ISC)", LicenseStatus.ALLOWED),
        ("(MIT OR Apache-2.0) AND GPL-3.0-only", LicenseStatus.RESTRICTED),
        ("((MIT))", LicenseStatus.ALLOWED),
        ("(Nonexistent-1.0 OR GPL-3.0-only) AND MIT", LicenseStatus.UNKNOWN),
        # ``WITH``: no exception has been reviewed, so none clears a base and
        # none rescues one. Both directions are refusals, never a pass.
        ("MIT WITH Classpath-exception-2.0", LicenseStatus.UNKNOWN),
        ("GPL-3.0-only WITH Classpath-exception-2.0", LicenseStatus.RESTRICTED),
        ("MIT OR GPL-2.0-only WITH Classpath-exception-2.0", LicenseStatus.ALLOWED),
        # ``+`` widens to unreviewed later versions; a denied family stays denied.
        ("GPL-2.0+", LicenseStatus.RESTRICTED),
        ("Apache-2.0+", LicenseStatus.UNKNOWN),
        # Explicit non-answers and unreviewed text are refusals, not shrugs.
        ("NOASSERTION", LicenseStatus.RESTRICTED),
        ("NONE", LicenseStatus.RESTRICTED),
        ("CathSim-Research-Only-1.0", LicenseStatus.UNKNOWN),
        ("Copyright 2026 Someone. All rights reserved.", LicenseStatus.UNKNOWN),
        ("", LicenseStatus.UNKNOWN),
        ("   ", LicenseStatus.UNKNOWN),
        # Malformed expressions: refused, and refused without raising.
        ("(MIT OR GPL-3.0-only", LicenseStatus.UNKNOWN),
        ("MIT OR Apache-2.0)", LicenseStatus.UNKNOWN),
        ("(MIT))(", LicenseStatus.UNKNOWN),
        ("()", LicenseStatus.UNKNOWN),
        ("MIT AND", LicenseStatus.UNKNOWN),
        ("OR MIT", LicenseStatus.UNKNOWN),
        ("MIT WITH", LicenseStatus.UNKNOWN),
        ("MIT WITH AND Apache-2.0", LicenseStatus.UNKNOWN),
    ],
)
def test_spdx_expressions_are_evaluated_with_their_structure_intact(
    expression: str, expected: LicenseStatus
) -> None:
    verdict = classify_license(expression)
    assert verdict.status is expected, verdict.reason
    assert verdict.spdx == " ".join(expression.split())


def test_grouping_is_not_flattened_away() -> None:
    """The probe from review: stripping the parentheses cleared copyleft."""
    grouped = classify_license(GROUPED_COPYLEFT)
    assert grouped.status is LicenseStatus.RESTRICTED
    assert "reciprocal copyleft" in grouped.reason
    assert is_allowed(GROUPED_COPYLEFT) is False
    # Same tokens, no parentheses: Apache-2.0 really is an escape hatch here,
    # which is exactly why the two must not classify the same.
    assert is_allowed("GPL-3.0-only AND MIT OR Apache-2.0") is True


def _grouped_copyleft_world(
    license_expression: str, *, disposition: Disposition = Disposition.WRAP
) -> WorldPackage:
    """A world with pinned pip installs and a declared license: install-ready but for terms."""
    return WorldPackage(
        id="grouped-license-world",
        display_name="Grouped License World",
        domain="Endovascular",
        engine="MuJoCo",
        disposition=disposition,
        license=license_expression,
        world_kind="gym",
        world_pin="world-rev-1",
        safety_evidence="fixture: env reports info.max_pen",
        metrics_only=False,
        install=PipExtraInstall(
            strategy=InstallStrategy.PIP_EXTRA,
            extras=("grouped",),
            packages=(PinnedPackage(name="grouped-world", version="1.2.3"),),
            verify_import="grouped_world",
        ),
    )


def test_a_grouped_copyleft_world_is_refused_by_the_shelf_and_installer_gates() -> None:
    """The reviewer's probe: this row validated as a wrap target with installable=True."""
    with pytest.raises(TaskContractError, match="must be redistributable"):
        _grouped_copyleft_world(GROUPED_COPYLEFT)

    # Recorded honestly as a skip, the same terms still bar an install: the
    # disposition is what a row may change, not the verdict on its license.
    skipped = _grouped_copyleft_world(GROUPED_COPYLEFT, disposition=Disposition.SKIP)
    assert skipped.license_verified is True
    assert skipped.license_verdict.status is LicenseStatus.RESTRICTED
    assert skipped.license_permitted is False
    assert skipped.installable is False
    # Pins, disposition and audit are otherwise install-ready: only the terms stop it.
    assert _grouped_copyleft_world("MIT OR Apache-2.0").installable is True


def test_pathological_nesting_is_refused_rather_than_exhausting_the_stack() -> None:
    """A declaration is third-party data; a parser that dies on it decides nothing."""
    bomb = "(" * 5_000 + "MIT" + ")" * 5_000
    assert classify_license(bomb).status is LicenseStatus.UNKNOWN
    # The cap is on nesting, not on length: a wide expression still evaluates.
    assert classify_license(" OR ".join(["GPL-3.0-only"] * 500 + ["MIT"])).status is (
        LicenseStatus.ALLOWED
    )
