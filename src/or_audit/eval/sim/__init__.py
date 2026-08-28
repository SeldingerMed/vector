"""Simulation bridges and physics connectors for procedural evaluation."""

from __future__ import annotations

from or_audit.eval.sim.base import (
    BACKEND_REAL,
    BACKEND_SYNTHETIC_STUB,
    BACKEND_UNKNOWN,
    AdapterDiscovery,
    BaseSimulationBridge,
    SimFactory,
    SimulationEngine,
    WorldAdapter,
    clear_simulation_registry,
    discover_world_adapters,
    get_simulation_engine,
    list_simulation_engines,
    module_distribution_version,
    register_simulation_engine,
    register_world_adapter,
    require_simulation_engine,
    reset_default_simulation_engines,
    world_adapter_discovery,
)
from or_audit.eval.sim.gym_bridge import GymnasiumBridge, make_gym_bridge
from or_audit.eval.sim.isaac_bridge import IsaacBridge, make_isaac_bridge
from or_audit.eval.sim.sofa_bridge import SofaBridge, make_sofa_bridge
from or_audit.eval.sim.warp_bridge import WarpBridge, make_warp_bridge
from or_audit.eval.worlds import (
    DeterminismClass,
    WorldCapabilities,
    WorldKindSpec,
    list_world_kinds,
    require_world_kind,
    world_kind_key,
    world_kind_spec,
)

reset_default_simulation_engines()
discover_world_adapters()

__all__ = [
    "BACKEND_REAL",
    "BACKEND_SYNTHETIC_STUB",
    "BACKEND_UNKNOWN",
    "AdapterDiscovery",
    "BaseSimulationBridge",
    "DeterminismClass",
    "GymnasiumBridge",
    "IsaacBridge",
    "SimFactory",
    "SimulationEngine",
    "SofaBridge",
    "WarpBridge",
    "WorldAdapter",
    "WorldCapabilities",
    "WorldKindSpec",
    "clear_simulation_registry",
    "discover_world_adapters",
    "get_simulation_engine",
    "list_simulation_engines",
    "list_world_kinds",
    "make_gym_bridge",
    "make_isaac_bridge",
    "make_sofa_bridge",
    "make_warp_bridge",
    "module_distribution_version",
    "register_simulation_engine",
    "register_world_adapter",
    "require_simulation_engine",
    "require_world_kind",
    "reset_default_simulation_engines",
    "world_adapter_discovery",
    "world_kind_key",
    "world_kind_spec",
]
