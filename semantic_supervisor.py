"""Deterministic safety supervision for semantic driving suggestions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass


ALLOWED_ACTIONS = {
    "STOP",
    "HOLD",
    "FORWARD",
    "SLOW_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "SLOW_LEFT",
    "SLOW_RIGHT",
}


@dataclass(frozen=True)
class LidarSectors:
    nearest: float = math.nan
    front: float = math.nan
    left: float = math.nan
    right: float = math.nan


@dataclass(frozen=True)
class VLAResult:
    scene: str
    hazards: tuple[str, ...]
    action: str
    target_heading_deg: float
    max_speed_mps: float
    confidence: float
    reason: str
    raw_text: str = ""


@dataclass(frozen=True)
class SafetyDecision:
    action: str
    linear: float
    angular: float
    allowed: bool
    reason: str


def _finite_min(values: list[float]) -> float:
    return min((value for value in values if math.isfinite(value)), default=math.nan)


def compute_lidar_sectors(
    ranges: list[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
) -> LidarSectors:
    """Summarize a ROS LaserScan into forward/left/right safety distances."""
    buckets: dict[str, list[float]] = {"all": [], "front": [], "left": [], "right": []}
    for index, raw_value in enumerate(ranges):
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value) or value < range_min or value > range_max:
            continue
        angle = math.atan2(
            math.sin(angle_min + index * angle_increment),
            math.cos(angle_min + index * angle_increment),
        )
        degrees = math.degrees(angle)
        buckets["all"].append(value)
        if abs(degrees) <= 30.0:
            buckets["front"].append(value)
        elif 30.0 < degrees <= 110.0:
            buckets["left"].append(value)
        elif -110.0 <= degrees < -30.0:
            buckets["right"].append(value)
    return LidarSectors(
        nearest=_finite_min(buckets["all"]),
        front=_finite_min(buckets["front"]),
        left=_finite_min(buckets["left"]),
        right=_finite_min(buckets["right"]),
    )


def parse_vla_json(text: str) -> VLAResult:
    """Extract and validate the first JSON object produced by the VLM."""
    start = text.find("{")
    if start < 0:
        raise ValueError("model output does not contain a JSON object")
    data, _ = json.JSONDecoder().raw_decode(text[start:])
    action = str(data.get("action", "")).upper().strip()
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"unsupported action: {action or '<empty>'}")
    hazards = data.get("hazards", [])
    if isinstance(hazards, str):
        hazards = [hazards]
    if not isinstance(hazards, list):
        raise ValueError("hazards must be a list")
    return VLAResult(
        scene=str(data.get("scene", ""))[:240],
        hazards=tuple(str(item)[:100] for item in hazards[:8]),
        action=action,
        target_heading_deg=float(data.get("target_heading_deg", 0.0)),
        max_speed_mps=max(0.0, float(data.get("max_speed_mps", 0.0))),
        confidence=min(1.0, max(0.0, float(data.get("confidence", 0.0)))),
        reason=str(data.get("reason", ""))[:240],
        raw_text=text,
    )


class SafetyPolicy:
    """Turn an untrusted semantic suggestion into a bounded Twist command."""

    def __init__(self, config: dict) -> None:
        self.stop_distance = float(config.get("stop_distance_m", 0.45))
        self.turn_clearance = float(config.get("turn_clearance_m", 0.30))
        self.minimum_confidence = float(config.get("minimum_confidence", 0.65))
        self.max_result_age = float(config.get("max_result_age_s", 5.0))
        self.max_command_duration = float(config.get("max_command_duration_s", 0.6))
        self.require_lidar = bool(config.get("require_lidar", True))
        self.max_linear = abs(float(config.get("max_linear_mps", 0.10)))
        self.max_angular = abs(float(config.get("max_angular_radps", 0.35)))
        self.slow_linear = abs(float(config.get("slow_linear_mps", 0.05)))
        self.turn_linear = abs(float(config.get("turn_linear_mps", 0.03)))

    def stop(self, reason: str) -> SafetyDecision:
        return SafetyDecision("STOP", 0.0, 0.0, False, reason)

    def evaluate(
        self,
        result: VLAResult | None,
        sectors: LidarSectors,
        result_age_s: float,
    ) -> SafetyDecision:
        if result is None:
            return self.stop("waiting for a valid VLM result")
        if result_age_s > self.max_result_age:
            return self.stop(f"VLM result stale ({result_age_s:.1f}s)")
        if result.confidence < self.minimum_confidence:
            return self.stop(f"low confidence ({result.confidence:.2f})")
        if result.action in {"STOP", "HOLD"}:
            return SafetyDecision(result.action, 0.0, 0.0, True, result.reason or result.action)
        if result_age_s > self.max_command_duration:
            return self.stop(f"motion pulse expired ({result_age_s:.1f}s)")
        if self.require_lidar and not math.isfinite(sectors.front):
            return self.stop("front LiDAR sector unavailable")

        front_blocked = math.isfinite(sectors.front) and sectors.front < self.stop_distance
        if front_blocked and result.action in {"FORWARD", "SLOW_FORWARD", "SLOW_LEFT", "SLOW_RIGHT"}:
            return self.stop(f"front obstacle at {sectors.front:.2f}m")
        if result.action in {"TURN_LEFT", "SLOW_LEFT"}:
            if self.require_lidar and not math.isfinite(sectors.left):
                return self.stop("left LiDAR sector unavailable")
            if math.isfinite(sectors.left) and sectors.left < self.turn_clearance:
                return self.stop(f"left obstacle at {sectors.left:.2f}m")
            angular = self.max_angular
        elif result.action in {"TURN_RIGHT", "SLOW_RIGHT"}:
            if self.require_lidar and not math.isfinite(sectors.right):
                return self.stop("right LiDAR sector unavailable")
            if math.isfinite(sectors.right) and sectors.right < self.turn_clearance:
                return self.stop(f"right obstacle at {sectors.right:.2f}m")
            angular = -self.max_angular
        else:
            angular = 0.0

        requested = result.max_speed_mps or self.slow_linear
        if result.action == "FORWARD":
            linear = min(requested, self.max_linear)
        elif result.action == "SLOW_FORWARD":
            linear = min(requested, self.slow_linear)
        elif result.action.startswith("SLOW_"):
            linear = min(requested, self.turn_linear)
        else:
            linear = 0.0
        return SafetyDecision(result.action, linear, angular, True, result.reason or result.action)
