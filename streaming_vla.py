"""Stateful, low-rate Qwen3-VL scene reasoning for the autonomy console."""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


SYSTEM_PROMPT = """你是封闭室内场地的小型移动机器人任务规划器。你接收按时间排序的连续相机帧、二维地图、激光雷达摘要、定位与导航状态，以及操作者持续更新的指令。
你的职责是理解连续场景、记住最近走过的区域和用户要求，并给 Nav2 一个低频高层建议。不要直接控制电机，不要绕过 0.16 米安全边界。
只输出一个 JSON 对象，不要 Markdown。字段必须为：
scene: 当前场景简述；observations: 最多4条可从输入核验的观察；memory: 一句话更新后的场景记忆；risk: LOW/MEDIUM/HIGH；behavior: HOLD/NAVIGATE_RELATIVE/PAUSE/STOP；relative_goal_m: {forward: -0.30到0.60, left: -0.40到0.40}；target_yaw_deg: -90到90；confidence: 0到1；reason: 简短决策理由。
不确定、导航正在恢复、定位不可用或路径被完全堵塞时选择 HOLD/PAUSE/STOP。只有明确可通行并符合操作者任务时才选择 NAVIGATE_RELATIVE。"""


@dataclass(frozen=True)
class SceneDecision:
    scene: str
    observations: tuple[str, ...]
    memory: str
    risk: str
    behavior: str
    forward_m: float
    left_m: float
    target_yaw_deg: float
    confidence: float
    reason: str
    raw_text: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["observations"] = list(self.observations)
        data["relative_goal_m"] = {
            "forward": data.pop("forward_m"),
            "left": data.pop("left_m"),
        }
        return data


@dataclass(frozen=True)
class SceneSample:
    frame: np.ndarray
    instruction: str
    prompt_history: tuple[str, ...]
    lidar: dict
    nav_status: dict
    map_image: np.ndarray | None = None
    sampled_at: float = field(default_factory=time.time)


def parse_scene_decision(text: str) -> SceneDecision:
    start = text.find("{")
    if start < 0:
        raise ValueError("模型输出中没有 JSON 对象")
    data, _ = json.JSONDecoder().raw_decode(text[start:])
    behavior = str(data.get("behavior", "HOLD")).upper().strip()
    if behavior not in {"HOLD", "NAVIGATE_RELATIVE", "PAUSE", "STOP"}:
        raise ValueError(f"不支持的 behavior: {behavior}")
    risk = str(data.get("risk", "MEDIUM")).upper().strip()
    if risk not in {"LOW", "MEDIUM", "HIGH"}:
        risk = "MEDIUM"
    relative = data.get("relative_goal_m") or {}
    observations = data.get("observations") or []
    if isinstance(observations, str):
        observations = [observations]
    return SceneDecision(
        scene=str(data.get("scene", ""))[:300],
        observations=tuple(str(item)[:160] for item in observations[:4]),
        memory=str(data.get("memory", ""))[:400],
        risk=risk,
        behavior=behavior,
        forward_m=min(0.60, max(-0.30, float(relative.get("forward", 0.0)))),
        left_m=min(0.40, max(-0.40, float(relative.get("left", 0.0)))),
        target_yaw_deg=min(90.0, max(-90.0, float(data.get("target_yaw_deg", 0.0)))),
        confidence=min(1.0, max(0.0, float(data.get("confidence", 0.0)))),
        reason=str(data.get("reason", ""))[:300],
        raw_text=text,
    )


def relative_goal_to_map(
    x: float, y: float, yaw_rad: float, forward_m: float, left_m: float
) -> tuple[float, float, float]:
    """Convert a robot-relative goal to a map-frame pose."""
    goal_x = x + math.cos(yaw_rad) * forward_m - math.sin(yaw_rad) * left_m
    goal_y = y + math.sin(yaw_rad) * forward_m + math.cos(yaw_rad) * left_m
    return goal_x, goal_y, yaw_rad


def resize_image(image: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0 or max(image.shape[:2]) <= max_side:
        return image.copy()
    scale = max_side / max(image.shape[:2])
    width = max(28, int(image.shape[1] * scale) // 28 * 28)
    height = max(28, int(image.shape[0] * scale) // 28 * 28)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


class OpenVINOStatefulBackend:
    def __init__(
        self,
        model_dir: str,
        device: str = "GPU",
        max_new_tokens: int = 180,
        cache_dir: str = "models/qwen3-vl-cache",
    ) -> None:
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"找不到 OpenVINO 模型: {model_path}")
        import openvino_genai as ov_genai
        from openvino import Tensor

        properties: dict[str, object] = {}
        if device.upper().startswith("GPU") and cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            properties["CACHE_DIR"] = cache_dir
        self.Tensor = Tensor
        self.pipe = ov_genai.VLMPipeline(str(model_path), device, **properties)
        self.config = ov_genai.GenerationConfig()
        self.config.max_new_tokens = max_new_tokens
        self.config.do_sample = False
        self.pipe.start_chat(SYSTEM_PROMPT)

    def reset(self, memory: str) -> None:
        self.pipe.finish_chat()
        self.pipe.start_chat(SYSTEM_PROMPT + f"\n已有压缩记忆：{memory or '无'}")

    def generate(self, images_bgr: list[np.ndarray], prompt: str) -> str:
        tensors = []
        for image in images_bgr:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensors.append(self.Tensor(np.ascontiguousarray(rgb)))
        output = self.pipe.generate(prompt, images=tensors, generation_config=self.config)
        return output.texts[0]


class StatefulVLAWorker:
    """Consume 1 Hz samples while keeping the inference loop non-blocking."""

    def __init__(
        self,
        config: dict,
        on_decision: Callable[[SceneDecision, int], None] | None = None,
        backend_factory=OpenVINOStatefulBackend,
    ) -> None:
        self.config = config
        self.on_decision = on_decision
        self.backend_factory = backend_factory
        self.queue: queue.Queue[tuple[int, SceneSample] | None] = queue.Queue(maxsize=8)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="stateful-vla", daemon=True)
        self.status = "starting"
        self.error = ""
        self.raw_text = ""
        self.decision: SceneDecision | None = None
        self.latency_s = math.nan
        self.sequence = 0
        self.memory = str(config.get("initial_memory", ""))
        self.turns = 0
        self.generation = 0
        self.reset_requested = False

    def start(self) -> None:
        self.thread.start()

    def submit(self, sample: SceneSample) -> bool:
        with self.lock:
            item = (self.generation, sample)
        try:
            self.queue.put_nowait(item)
            return True
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(item)
                return True
            except (queue.Empty, queue.Full):
                return False

    def invalidate(self, memory: str | None = None) -> int:
        """Invalidate queued/in-flight work so it can never dispatch a control action."""
        with self.lock:
            self.generation += 1
            generation = self.generation
            self.decision = None
            if memory is not None:
                self.memory = memory
            self.reset_requested = True
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        return generation

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "status": self.status,
                "error": self.error,
                "raw_text": self.raw_text,
                "latency_s": self.latency_s,
                "sequence": self.sequence,
                "memory": self.memory,
                "decision": self.decision.to_dict() if self.decision else None,
                "queued_frames": self.queue.qsize(),
                "generation": self.generation,
            }

    def _set(self, **values: object) -> None:
        with self.lock:
            for key, value in values.items():
                setattr(self, key, value)

    def _run(self) -> None:
        try:
            self._set(status="loading")
            backend = self.backend_factory(
                self.config.get("model_dir", "models/qwen3-vl-4b-int4-ov"),
                self.config.get("device", "GPU"),
                int(self.config.get("max_new_tokens", 180)),
                self.config.get("cache_dir", "models/qwen3-vl-cache"),
            )
            self._set(status="ready")
        except Exception as exc:
            self._set(status="error", error=str(exc))
            return

        max_frames = max(1, int(self.config.get("frames_per_inference", 4)))
        max_side = int(self.config.get("image_max_side", 448))
        reset_turns = max(2, int(self.config.get("chat_reset_turns", 12)))
        while not self.stop_event.is_set():
            with self.lock:
                reset_requested = self.reset_requested
                reset_memory = self.memory
                if reset_requested:
                    self.reset_requested = False
            if reset_requested:
                backend.reset(reset_memory)
                self.turns = 0
            try:
                first_item = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if first_item is None:
                break
            generation, first = first_item
            pending = [first]
            while True:
                try:
                    item = self.queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    break
                item_generation, sample = item
                if item_generation == generation:
                    pending.append(sample)
            # Inference must see the newest continuous frames, never an old backlog.
            batch = pending[-max_frames:]
            latest = batch[-1]
            images = [resize_image(item.frame, max_side) for item in batch]
            if latest.map_image is not None:
                images.append(resize_image(latest.map_image, max_side))
            prompt = self._build_prompt(batch, latest)
            started = time.monotonic()
            raw_text = ""
            try:
                self._set(status="inferencing")
                raw_text = backend.generate(images, prompt)
                decision = parse_scene_decision(raw_text)
                with self.lock:
                    stale = generation != self.generation
                    if stale:
                        self.status = "ready"
                        self.latency_s = time.monotonic() - started
                    else:
                        self.turns += 1
                        self.memory = decision.memory or self.memory
                        self.status = "ready"
                        self.decision = decision
                        self.error = ""
                        self.raw_text = raw_text[:4000]
                        self.latency_s = time.monotonic() - started
                        self.sequence += 1
                        memory = self.memory
                if stale:
                    continue
                if self.turns >= reset_turns:
                    backend.reset(memory)
                    self.turns = 0
                if self.on_decision:
                    self.on_decision(decision, generation)
            except Exception as exc:
                with self.lock:
                    if generation == self.generation:
                        self.status = "ready"
                        self.error = str(exc)
                        self.raw_text = raw_text[:4000]
                        self.latency_s = time.monotonic() - started
                    else:
                        self.status = "ready"

    def _build_prompt(self, batch: list[SceneSample], latest: SceneSample) -> str:
        nav = json.dumps(latest.nav_status, ensure_ascii=False, separators=(",", ":"))
        lidar = json.dumps(latest.lidar, ensure_ascii=False, separators=(",", ":"))
        history = "；".join(latest.prompt_history[-6:]) or "无"
        times = [round(item.sampled_at - batch[-1].sampled_at, 1) for item in batch]
        return (
            f"当前任务：{latest.instruction or '保持等待'}\n"
            f"近期用户指令（旧到新）：{history}\n"
            f"本轮连续相机帧数={len(batch)}，相对最新帧时间={times}秒；最后一张若存在是二维地图。\n"
            f"激光雷达摘要：{lidar}\n导航与定位状态：{nav}\n"
            f"已有场景记忆：{self.memory or '无'}\n"
            "请基于连续变化而不是孤立单帧给出下一轮高层决策。"
            "特别注意：导航 JSON 中 map_available=true 表示地图已经加载，禁止再回答等待地图。"
            "当前任务字段中的探索要求本身就是有效的操作者导航指令，不需要先存在 Nav2 goal。"
            "如果任务要求探索、状态为 IDLE/SUCCEEDED/FAILED/CANCELED、定位在 map 且雷达至少一个方向明确畅通，"
            "应给出不超过 0.45 米的 NAVIGATE_RELATIVE 短程目标。"
            "输出必须紧凑：最多3条观察，每个中文字符串不超过40字，务必闭合完整 JSON。"
        )

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=5.0)
