"""CLI entry point branding."""

from __future__ import annotations

import pytest

from or_audit.cli import build_parser, main


def test_build_parser_default_prog_is_surgeval() -> None:
    parser = build_parser(prog="surgeval")
    assert parser.prog == "surgeval"


def test_build_parser_vector_alias_prog() -> None:
    parser = build_parser(prog="vector")
    assert parser.prog == "vector"


def test_build_parser_or_audit_alias_prog() -> None:
    parser = build_parser(prog="or-audit")
    assert parser.prog == "or-audit"


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "0.3.0a0" in capsys.readouterr().out
