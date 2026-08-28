"""SurgEval: Universal Evaluation, Benchmarking & Safety Verification for Physical Healthcare AI."""

from __future__ import annotations

from or_audit.errors import AuditChainError, ScoreContractError, TaskContractError
from or_audit.eval.adapters import (
    BaseModalityAdapter,
    EndoluminalAdapter,
    FluoroscopyAdapter,
    KinematicsAdapter,
    ModalityAdapter,
    VideoAdapter,
    get_adapter,
    list_adapters,
    register_adapter,
    require_adapter,
)
from or_audit.eval.contracts import (
    CapabilitySpec,
    GateProjectionPolicy,
    HarnessSpec,
    InteractionMode,
    InterfaceSpec,
    MetricDirection,
    MetricKind,
    RuntimeDescriptor,
    RuntimeKind,
)
from or_audit.eval.enums import (
    AgentKind,
    GateKind,
    ModalityKind,
    OracleKind,
    PhiClass,
    WorldKind,
)
from or_audit.eval.job import JobResult, TrialRecord
from or_audit.eval.sim import (
    BaseSimulationBridge,
    GymnasiumBridge,
    SimulationEngine,
    SofaBridge,
    WarpBridge,
    get_simulation_engine,
    list_simulation_engines,
    register_simulation_engine,
    require_simulation_engine,
)
from or_audit.eval.task import GateSpec, MetricSpec, TaskSpec
from or_audit.eval.vector import GateOutcome, MetricOutcome, TrialVector
from or_audit.version import PACKAGE_VERSION
from surgeval.client import evaluate, load_agent, load_task, load_taskset
from surgeval.decorators import (
    AgentBinding,
    agent,
    capability_for,
    describe_agent,
    is_agent,
)
from surgeval.integrations import (
    GymnasiumPolicyWrapper,
    HuggingFacePredictorWrapper,
    PyTorchPolicyWrapper,
    wrap_gym_policy,
    wrap_hf,
    wrap_pytorch,
)

__version__ = PACKAGE_VERSION

__all__ = [
    "PACKAGE_VERSION",
    "AgentBinding",
    "AgentKind",
    "AuditChainError",
    "BaseModalityAdapter",
    "BaseSimulationBridge",
    "CapabilitySpec",
    "EndoluminalAdapter",
    "FluoroscopyAdapter",
    "GateKind",
    "GateOutcome",
    "GateProjectionPolicy",
    "GateSpec",
    "GymnasiumBridge",
    "GymnasiumPolicyWrapper",
    "HarnessSpec",
    "HuggingFacePredictorWrapper",
    "InteractionMode",
    "InterfaceSpec",
    "JobResult",
    "KinematicsAdapter",
    "MetricDirection",
    "MetricKind",
    "MetricOutcome",
    "MetricSpec",
    "ModalityAdapter",
    "ModalityKind",
    "OracleKind",
    "PhiClass",
    "PyTorchPolicyWrapper",
    "RuntimeDescriptor",
    "RuntimeKind",
    "ScoreContractError",
    "SimulationEngine",
    "SofaBridge",
    "TaskContractError",
    "TaskSpec",
    "TrialRecord",
    "TrialVector",
    "VideoAdapter",
    "WarpBridge",
    "WorldKind",
    "__version__",
    "agent",
    "capability_for",
    "describe_agent",
    "evaluate",
    "get_adapter",
    "get_simulation_engine",
    "is_agent",
    "list_adapters",
    "list_simulation_engines",
    "load_agent",
    "load_task",
    "load_taskset",
    "register_adapter",
    "register_simulation_engine",
    "require_adapter",
    "require_simulation_engine",
    "wrap_gym_policy",
    "wrap_hf",
    "wrap_pytorch",
]
