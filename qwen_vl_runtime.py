"""Asynchronous OpenVINO Qwen3-VL runtime for ROSOrin."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from semantic_supervisor import LidarSectors, VLAResult, parse_vla_json


SYSTEM_PROMPT = """你是低速室内移动机器人的语义驾驶监督器。你只提供高级动作建议，不能绕过激光雷达安全规则。
只输出一个 JSON 对象，不要 Markdown，不要额外文字。字段必须为：
scene: 简短中文场景描述；hazards: 字符串数组；action: STOP/HOLD/FORWARD/SLOW_FORWARD/TURN_LEFT/TURN_RIGHT/SLOW_LEFT/SLOW_RIGHT 之一；
target_heading_deg: -45 到 45；max_speed_mps: 0 到 0.10；confidence: 0 到 1；reason: 简短中文理由。
不确定、画面模糊、目标不清楚或前方危险时选择 STOP。"""


@dataclass(frozen=True)
class VLARequest:
    frame: np.ndarray
    instruction: str
    sectors: LidarSectors
    linear: float
    angular: float


@dataclass
class VLAState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    status: str = "disabled"
    result: VLAResult | None = None
    result_time: float = 0.0
    latency_ms: float = float("nan")
    error: str = ""
    raw_text: str = ""
    sequence: int = 0

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "status": self.status,
                "result": self.result,
                "result_time": self.result_time,
                "latency_ms": self.latency_ms,
                "error": self.error,
                "raw_text": self.raw_text,
                "sequence": self.sequence,
            }


def build_prompt(request: VLARequest) -> str:
    def distance(value: float) -> str:
        return "unknown" if not np.isfinite(value) else f"{value:.2f}"

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"用户任务：{request.instruction}\n"
        "激光雷达距离（米）："
        f"front={distance(request.sectors.front)}, left={distance(request.sectors.left)}, "
        f"right={distance(request.sectors.right)}, nearest={distance(request.sectors.nearest)}\n"
        f"当前运动：linear={request.linear:.2f}m/s, angular={request.angular:.2f}rad/s\n"
        "根据图像和以上可靠测距给出下一步高级动作。"
    )


class OpenVINOVLMBackend:
    def __init__(self, model_dir: str, device: str, max_new_tokens: int, cache_dir: str) -> None:
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Qwen3-VL OpenVINO model not found: {model_path}")
        import openvino_genai as ov_genai
        from openvino import Tensor

        self.ov_genai = ov_genai
        self.Tensor = Tensor
        properties: dict[str, object] = {}
        if device.upper().startswith("GPU") and cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            properties["CACHE_DIR"] = cache_dir
        self.pipe = ov_genai.VLMPipeline(str(model_path), device, **properties)
        self.generation_config = ov_genai.GenerationConfig()
        self.generation_config.max_new_tokens = max_new_tokens
        self.generation_config.do_sample = False

    def generate(self, frame_bgr: np.ndarray, prompt: str) -> str:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.Tensor(np.ascontiguousarray(rgb))
        output = self.pipe.generate(
            prompt,
            images=[image],
            generation_config=self.generation_config,
        )
        return output.texts[0]


class VLAWorker:
    """Keep heavyweight VLM work off the camera and safety-control loops."""

    def __init__(self, config: dict, backend_factory=OpenVINOVLMBackend) -> None:
        self.config = config
        self.backend_factory = backend_factory
        self.state = VLAState(status="starting")
        self.requests: queue.Queue[VLARequest | None] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="qwen-vl", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def submit(self, request: VLARequest) -> bool:
        if self.stop_event.is_set():
            return False
        try:
            self.requests.put_nowait(request)
            return True
        except queue.Full:
            # Replace an unprocessed frame so inference always sees the newest scene.
            try:
                self.requests.get_nowait()
                self.requests.put_nowait(request)
                return True
            except (queue.Empty, queue.Full):
                return False

    def _set(self, **changes: object) -> None:
        with self.state.lock:
            for key, value in changes.items():
                setattr(self.state, key, value)

    def _run(self) -> None:
        try:
            self._set(status="loading")
            backend = self.backend_factory(
                self.config["model_dir"],
                self.config.get("device", "GPU"),
                int(self.config.get("max_new_tokens", 96)),
                self.config.get("cache_dir", "models/qwen3-vl-cache"),
            )
            self._set(status="ready")
        except Exception as exc:
            self._set(status="error", error=str(exc))
            return

        while not self.stop_event.is_set():
            try:
                request = self.requests.get(timeout=0.2)
            except queue.Empty:
                continue
            if request is None:
                break
            started = time.monotonic()
            try:
                self._set(status="inferencing", error="")
                frame = resize_for_vlm(request.frame, int(self.config.get("image_max_side", 672)))
                text = backend.generate(frame, build_prompt(request))
                result = parse_vla_json(text)
                finished = time.monotonic()
                with self.state.lock:
                    self.state.status = "ready"
                    self.state.result = result
                    self.state.result_time = finished
                    self.state.latency_ms = (finished - started) * 1000.0
                    self.state.raw_text = text
                    self.state.error = ""
                    self.state.sequence += 1
            except Exception as exc:
                self._set(
                    status="ready",
                    latency_ms=(time.monotonic() - started) * 1000.0,
                    error=str(exc),
                )

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=5.0)


def resize_for_vlm(frame: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0 or max(frame.shape[:2]) <= max_side:
        return frame.copy()
    scale = max_side / max(frame.shape[:2])
    width = max(32, int(round(frame.shape[1] * scale / 28.0)) * 28)
    height = max(32, int(round(frame.shape[0] * scale / 28.0)) * 28)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
