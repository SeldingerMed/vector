"""Surgical Video Modality Adapter (Laparoscopy and Endoscopy).

Handles video frame streaming, temporal chunking, tool action normalization,
and laparoscopic/endoscopic safety telemetry (critical view of safety, out-of-field
tool motion, unmonitored electrocautery).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from or_audit.eval.adapters.base import ModalityAdapter
from or_audit.eval.enums import ModalityKind


@dataclass(frozen=True)
class VideoFrameObservation:
    """Standard observation payload for procedural video tasks.

    Only ``frame_index`` is ever positioned; every other field is ``None``
    when the source did not report it. A defaulted timestamp or resolution
    is a fabricated measurement.
    """

    frame_index: int
    timestamp_ms: float | None = None
    image_uri: str | None = None
    width: int | None = None
    height: int | None = None
    optical_flow: tuple[float, ...] | None = None
    active_tools: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoToolAction:
    """Standard action payload for robotic/video tool commands."""

    tool_id: str
    action_kind: str
    target_pose: tuple[float, float, float] | None = None  # (x, y, z)
    electrocautery_active: bool = False
    grasp_force: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class VideoAdapter(ModalityAdapter):
    """Adapter for laparoscopic and endoscopic surgical video AI."""

    modality: ModalityKind | str = ModalityKind.VIDEO_LAPAROSCOPIC

    def __init__(self, modality: ModalityKind | str = ModalityKind.VIDEO_LAPAROSCOPIC) -> None:
        self.modality = modality

    def validate_observation(self, observation: Any) -> bool:
        """Validate observation has required video frame metadata."""
        if isinstance(observation, VideoFrameObservation):
            timestamp_ok = observation.timestamp_ms is None or observation.timestamp_ms >= 0.0
            return observation.frame_index >= 0 and timestamp_ok
        if isinstance(observation, dict):
            return "frame_index" in observation or "image" in observation or "frame" in observation
        return hasattr(observation, "__array__") or isinstance(observation, (list, tuple))

    def validate_action(self, action: Any) -> bool:
        """Validate action conforms to tool action format or discrete action."""
        if isinstance(action, VideoToolAction):
            return bool(action.tool_id and action.action_kind)
        if isinstance(action, dict):
            return "tool_id" in action or "action" in action or "prediction" in action
        return isinstance(action, (int, float, str, list, tuple)) or hasattr(action, "__array__")

    def preprocess_observation(self, observation: Any) -> Any:
        if isinstance(observation, dict) and "frame_index" in observation:
            flow = observation.get("optical_flow")
            flow_tuple = tuple(flow) if isinstance(flow, (list, tuple)) else None
            extra_dict = observation.get("extra")
            tools = observation.get("active_tools")
            ts = observation.get("timestamp_ms")
            w = observation.get("width")
            h = observation.get("height")
            return VideoFrameObservation(
                frame_index=int(observation["frame_index"]),
                timestamp_ms=float(ts) if ts is not None else None,
                image_uri=str(uri) if (uri := observation.get("image_uri")) is not None else None,
                width=int(w) if w is not None else None,
                height=int(h) if h is not None else None,
                optical_flow=flow_tuple,
                active_tools=tuple(tools) if isinstance(tools, (list, tuple)) else (),
                extra=extra_dict if isinstance(extra_dict, dict) else {},
            )
        if isinstance(observation, dict) and "video_uri" in observation:
            return {
                "frame_index": 0,
                "timestamp_ms": None,
                "image_uri": str(observation["video_uri"]),
                "clip_id": str(observation.get("id", "")),
                "frame_count": int(observation.get("frame_count", 0)),
                "modality": (
                    self.modality.value
                    if isinstance(self.modality, ModalityKind)
                    else str(self.modality)
                ),
            }
        return observation

    def postprocess_action(self, action: Any) -> Any:
        """Normalize action into VideoToolAction if structured."""
        if isinstance(action, dict) and "tool_id" in action:
            force = action.get("grasp_force")
            extra_dict = action.get("extra")
            raw_pose = action.get("target_pose")
            pose_tuple = tuple(raw_pose) if isinstance(raw_pose, (list, tuple)) else None
            return VideoToolAction(
                tool_id=str(action["tool_id"]),
                action_kind=str(action.get("action_kind") or "move"),
                target_pose=pose_tuple,
                electrocautery_active=bool(action.get("electrocautery_active", False)),
                grasp_force=float(force) if force is not None else 0.0,
                extra=extra_dict if isinstance(extra_dict, dict) else {},
            )
        return action

    def extract_safety_state(self, step_context: dict[str, Any] | None) -> dict[str, Any]:
        """Extract laparoscopic-specific safety telemetry."""
        safety = super().extract_safety_state(step_context)
        if not isinstance(step_context, dict):
            return safety
        info = step_context.get("info")
        if isinstance(info, dict):
            for key in (
                "distance_to_critical_structure",
                "out_of_field_tool_motion",
                "unmonitored_cautery",
                "critical_view_achieved",
                "bleeding_detected",
            ):
                if key in info:
                    safety.setdefault(key, info[key])
        return safety

    def get_schema_spec(self) -> dict[str, Any]:
        """Return video modality schema metadata."""
        spec = super().get_schema_spec()
        spec.update(
            {
                "observation_type": "VideoFrameObservation",
                "action_type": "VideoToolAction",
                "frame_format": "RGB/RGB-D",
            }
        )
        return spec
