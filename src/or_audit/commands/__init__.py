"""Feature-owned CLI subcommands, registered into ``or_audit.cli``'s parser.

Each module exposes ``register(sub)`` and adds its own subparsers with
``set_defaults(func=handler)``, where a handler is ``(argparse.Namespace) ->
int``. Keeping one module per surface means a new surface (a wrap kit, an
install ladder, a concierge) lands without touching the shared parser, and the
CLI entry point stays a wiring file rather than a 2000-line switch.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from or_audit.commands import (
    concierge,
    conformance,
    doctor,
    export_verifiers,
    onramp,
    quickstart,
    shelf,
    worlds,
    wrap,
)

#: Registration order is the order these appear in ``surgeval --help``:
#: first-vector path first, then the wrap/curation kit, then publication and
#: hosted surfaces.
COMMAND_MODULES: tuple[Callable[[argparse._SubParsersAction[argparse.ArgumentParser]], None], ...]
COMMAND_MODULES = (
    quickstart.register,
    doctor.register,
    worlds.register,
    wrap.register,
    conformance.register,
    onramp.register,
    export_verifiers.register,
    shelf.register,
    concierge.register,
)


def register_all(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register every feature subcommand on ``sub``."""
    for register in COMMAND_MODULES:
        register(sub)


__all__ = ["COMMAND_MODULES", "register_all"]
