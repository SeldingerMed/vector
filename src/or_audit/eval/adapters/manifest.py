"""Manifest-driven bootstrap for bundled first-party adapter plugins.

Adapters are externalized into plugin modules that never self-register:
adding an adapter is one entry in :data:`BUNDLED_ADAPTER_PLUGINS` (plugin id
-> module/attribute + optional constructor params). Bootstrap resolves each
plugin module *without executing it* (``find_spec``), computes a binding
digest covering the canonical ``{id, module, attr, params}`` binding AND the
module's file bytes, verifies it against the manifest's required ``sha256``
pin *before importing the module*, and only then imports and registers it.

Because the digest covers the binding plus content, tampering with the plugin
code, its params, its attr, or retargeting the plugin id all change the pin.
Task streams pin to a plugin id via ``adapter_digest``, checked at task load
against :func:`~or_audit.eval.adapters.base.adapter_revision`.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
from typing import Any

from or_audit.audit.canonical import digest as canonical_digest
from or_audit.errors import TaskContractError
from or_audit.eval.adapters.base import register_adapter

#: Each entry carries an immutable SHA-256 pin of the plugin binding:
#: ``sha256 = plugin_binding_digest({id, module, attr, params})``. Bootstrap
#: refuses to load (or import) a plugin whose on-disk binding differs.
BUNDLED_ADAPTER_PLUGINS: tuple[dict[str, Any], ...] = (
    {
        "id": "video-laparoscopic",
        "module": "or_audit.eval.adapters.video",
        "attr": "VideoAdapter",
        "sha256": "b1b06a17ff0da168c8f6d5a60442a0279e56b7d2cfaf234f681f8195ca9fe12d",
    },
    {
        "id": "video-endoscopic",
        "module": "or_audit.eval.adapters.video",
        "attr": "VideoAdapter",
        "params": {"modality": "video-endoscopic"},
        "sha256": "48991e39f5afb997f60d402c44be5502e35b6c82e155a807007a0bfac3a7c24c",
    },
    {
        "id": "airway-bronchoscopy",
        "module": "or_audit.eval.adapters.endoluminal",
        "attr": "EndoluminalAdapter",
        "sha256": "8d621713470cae4f88fa90f0a8e583d76d9241458e1a680c02b2aea5e548733f",
    },
    {
        "id": "fluoroscopy-dsa",
        "module": "or_audit.eval.adapters.fluoroscopy",
        "attr": "FluoroscopyAdapter",
        "sha256": "1622d3f19a54f19233b9c74e97024d66e6d59932e74ef14fd8e77ca8ac8abe5d",
    },
    {
        "id": "endovascular-sim",
        "module": "or_audit.eval.adapters.fluoroscopy",
        "attr": "FluoroscopyAdapter",
        "params": {"modality": "endovascular-sim"},
        "sha256": "e083aa5f5614d602d7860c8431adebc2b06e4dc75f696c5dc0341282c3cd5c79",
    },
    {
        "id": "robotic-kinematics",
        "module": "or_audit.eval.adapters.kinematics",
        "attr": "KinematicsAdapter",
        "sha256": "d6db8147239ca7e73213fe80fb9130fc9a88f44510962ab59a02ae5255cddc24",
    },
    {
        "id": "orthopedic-pointcloud",
        "module": "or_audit.eval.adapters.kinematics",
        "attr": "KinematicsAdapter",
        "params": {"modality": "orthopedic-pointcloud"},
        "sha256": "df171dde5b3969c17262aaecaf640621a02837582c3fade40be60f99154c044e",
    },
)


def _resolve_module_file(module_name: str) -> Path:
    """Resolve a module's source file without importing/executing it."""
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise TaskContractError(f"cannot resolve adapter plugin module {module_name!r}") from exc
    if spec is None or not spec.origin or spec.origin in ("built-in", "frozen"):
        raise TaskContractError(f"adapter plugin module {module_name!r} has no source file")
    return Path(spec.origin)


def plugin_binding_digest(
    *,
    plugin_id: str,
    module: str,
    attr: str,
    params: dict[str, Any] | None,
) -> str:
    """SHA-256 of the canonical binding plus the plugin module's bytes."""
    binding = canonical_digest(
        {"id": plugin_id, "module": module, "attr": attr, "params": params or {}}
    )
    content = hashlib.sha256(_resolve_module_file(module).read_bytes()).hexdigest()
    return hashlib.sha256(f"{binding}\0{content}".encode()).hexdigest()


def plugin_content_digest(entry: dict[str, Any]) -> str:
    """SHA-256 of a single manifest entry's binding (module + metadata)."""
    return plugin_binding_digest(
        plugin_id=entry["id"],
        module=entry["module"],
        attr=entry["attr"],
        params=entry.get("params"),
    )


def _instantiate(cls: type[Any], params: dict[str, Any], kwargs: dict[str, Any]) -> Any:
    merged = dict(params)
    merged.update(kwargs)
    return cls(**merged)


def bootstrap_adapter_plugins(
    entries: tuple[dict[str, Any], ...] = BUNDLED_ADAPTER_PLUGINS,
) -> None:
    """Register every bundled adapter plugin, verifying pins before import.

    Two phases: first verify every binding digest against its manifest pin
    (resolving module files via ``find_spec``, executing no plugin code), then
    import each module and register a closure bound to the *verified* class —
    so no tampered code ever runs and no factory re-imports by name later.
    """
    for entry in entries:
        plugin_id = entry["id"]
        declared = entry.get("sha256")
        if not declared:
            raise TaskContractError(
                f"adapter plugin {plugin_id!r} has no sha256 pin in the manifest"
            )
        actual = plugin_content_digest(entry)
        if actual != declared:
            raise TaskContractError(
                f"adapter plugin {plugin_id!r} binding digest mismatch "
                f"(manifest {declared}, actual {actual})"
            )
    for entry in entries:
        plugin_id = entry["id"]
        module = importlib.import_module(entry["module"])
        cls = getattr(module, entry["attr"])
        digest = plugin_content_digest(entry)
        frozen_params = dict(entry.get("params", {}))
        register_adapter(
            plugin_id,
            lambda _c=cls, _p=frozen_params, **kw: _instantiate(_c, _p, kw),
            digest=digest,
            override=True,
        )
