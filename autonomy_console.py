"""Browser control center for mapping, Nav2 goals and stateful Qwen3-VL reasoning."""

from __future__ import annotations

import argparse
import atexit
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import roslibpy
import yaml

from semantic_supervisor import compute_lidar_sectors
from streaming_vla import SceneDecision, SceneSample, StatefulVLAWorker, relative_goal_to_map


@dataclass
class MapSnapshot:
    data: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    scale: int


def map_pixel_to_world(snapshot: MapSnapshot, pixel_x: float, pixel_y: float) -> tuple[float, float]:
    col = min(snapshot.data.shape[1] - 1, max(0, int(pixel_x / snapshot.scale)))
    display_row = min(snapshot.data.shape[0] - 1, max(0, int(pixel_y / snapshot.scale)))
    grid_row = snapshot.data.shape[0] - 1 - display_row
    x = snapshot.origin_x + (col + 0.5) * snapshot.resolution
    y = snapshot.origin_y + (grid_row + 0.5) * snapshot.resolution
    return x, y


def world_to_map_pixel(snapshot: MapSnapshot, x: float, y: float) -> tuple[int, int]:
    col = int((x - snapshot.origin_x) / snapshot.resolution)
    grid_row = int((y - snapshot.origin_y) / snapshot.resolution)
    display_row = snapshot.data.shape[0] - 1 - grid_row
    return col * snapshot.scale, display_row * snapshot.scale


class CameraReceiver:
    def __init__(self, url: str) -> None:
        self.url = url
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="mjpeg-camera", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            capture = cv2.VideoCapture(self.url)
            while capture.isOpened() and not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                with self.lock:
                    self.frame = frame
            capture.release()
            self.stop_event.wait(1.0)

    def latest(self) -> np.ndarray | None:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3.0)


class RobotBridge:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.ros: roslibpy.Ros | None = None
        self.lock = threading.Lock()
        self.nav_status: dict = {"state": "CONNECTING", "message": "等待 ROS 2"}
        self.lidar: dict = {"front": None, "left": None, "right": None, "nearest": None}
        self.map_snapshot: MapSnapshot | None = None
        self.path: list[tuple[float, float]] = []
        self.goal: tuple[float, float, float] | None = None
        self.goal_topic: roslibpy.Topic | None = None
        self.command_topic: roslibpy.Topic | None = None
        self.subscriptions: list[roslibpy.Topic] = []

    def connect(self, timeout: float = 8.0) -> None:
        if self.is_connected:
            return
        if self.ros is not None:
            self.ros.terminate()
        self.subscriptions.clear()
        self.goal_topic = None
        self.command_topic = None
        # The asyncio transport avoids Twisted/zope mutating Gradio component classes.
        self.ros = roslibpy.Ros(host=self.host, port=self.port, transport="asyncio")
        try:
            self.ros.run(timeout=timeout)
        except Exception:
            self.ros.terminate()
            self.ros = None
            raise
        if not self.ros.is_connected:
            raise ConnectionError("无法连接机器人 rosbridge")
        self.goal_topic = roslibpy.Topic(self.ros, "/vla/nav_goal", "geometry_msgs/PoseStamped")
        self.command_topic = roslibpy.Topic(self.ros, "/vla/nav_command", "std_msgs/String")
        self._subscribe("/map", "nav_msgs/OccupancyGrid", self._on_map)
        self._subscribe("/scan_raw", "sensor_msgs/LaserScan", self._on_scan)
        self._subscribe("/vla/navigation_status", "std_msgs/String", self._on_status)
        self._subscribe("/plan", "nav_msgs/Path", self._on_path)

    def _subscribe(self, name: str, message_type: str, callback) -> None:
        if self.ros is None:
            raise RuntimeError("rosbridge 尚未连接")
        topic = roslibpy.Topic(self.ros, name, message_type)
        topic.subscribe(callback)
        self.subscriptions.append(topic)

    def _on_status(self, message: dict) -> None:
        try:
            value = json.loads(message.get("data", "{}"))
        except json.JSONDecodeError:
            return
        with self.lock:
            self.nav_status = value

    def _on_scan(self, message: dict) -> None:
        sectors = compute_lidar_sectors(
            message.get("ranges", []),
            float(message.get("angle_min", 0.0)),
            float(message.get("angle_increment", 0.0)),
            float(message.get("range_min", 0.02)),
            float(message.get("range_max", 12.0)),
        )
        with self.lock:
            self.lidar = {
                name: round(value, 3) if math.isfinite(value) else None
                for name, value in {
                    "front": sectors.front,
                    "left": sectors.left,
                    "right": sectors.right,
                    "nearest": sectors.nearest,
                }.items()
            }

    def _on_map(self, message: dict) -> None:
        info = message.get("info", {})
        height, width = int(info.get("height", 0)), int(info.get("width", 0))
        values = np.asarray(message.get("data", []), dtype=np.int16)
        if height <= 0 or width <= 0 or values.size != height * width:
            return
        origin = info.get("origin", {}).get("position", {})
        scale = max(1, min(4, 900 // max(height, width, 1)))
        snapshot = MapSnapshot(
            values.reshape(height, width),
            float(info.get("resolution", 0.05)),
            float(origin.get("x", 0.0)),
            float(origin.get("y", 0.0)),
            scale,
        )
        with self.lock:
            self.map_snapshot = snapshot

    def _on_path(self, message: dict) -> None:
        points = []
        for pose in message.get("poses", []):
            position = pose.get("pose", {}).get("position", {})
            points.append((float(position.get("x", 0.0)), float(position.get("y", 0.0))))
        with self.lock:
            self.path = points

    def publish_goal(self, x: float, y: float, yaw_deg: float = 0.0) -> str:
        if self.goal_topic is None:
            return "目标未发送：rosbridge 尚未连接"
        yaw = math.radians(yaw_deg)
        message = {
            "header": {"frame_id": "map", "stamp": {"sec": 0, "nanosec": 0}},
            "pose": {
                "position": {"x": float(x), "y": float(y), "z": 0.0},
                "orientation": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": math.sin(yaw / 2.0),
                    "w": math.cos(yaw / 2.0),
                },
            },
        }
        self.goal_topic.publish(roslibpy.Message(message))
        with self.lock:
            self.goal = (x, y, yaw_deg)
        return f"已发送地图目标：x={x:.2f}, y={y:.2f}, yaw={yaw_deg:.0f}°"

    def command(self, command: str, **parameters: object) -> str:
        if self.command_topic is None:
            return "命令未发送：rosbridge 尚未连接"
        payload = {"command": command.upper(), **parameters}
        self.command_topic.publish(roslibpy.Message({"data": json.dumps(payload, ensure_ascii=False)}))
        return f"已发送命令：{payload['command']}"

    def snapshot(self) -> tuple[dict, dict, MapSnapshot | None, list[tuple[float, float]]]:
        with self.lock:
            return self.nav_status.copy(), self.lidar.copy(), self.map_snapshot, self.path.copy()

    def render_map(self) -> np.ndarray:
        nav, _, snapshot, path = self.snapshot()
        if snapshot is None:
            image = np.full((480, 640, 3), 235, dtype=np.uint8)
            cv2.putText(image, "Waiting for /map", (170, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = np.flipud(snapshot.data)
        gray = np.full(data.shape, 127, dtype=np.uint8)
        gray[data == 0] = 245
        gray[data >= 65] = 20
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if snapshot.scale != 1:
            image = cv2.resize(
                image,
                (image.shape[1] * snapshot.scale, image.shape[0] * snapshot.scale),
                interpolation=cv2.INTER_NEAREST,
            )
        if path:
            pixels = [world_to_map_pixel(snapshot, x, y) for x, y in path]
            for start, end in zip(pixels, pixels[1:]):
                cv2.line(image, start, end, (220, 120, 20), 2)
        goal = nav.get("goal")
        if goal:
            gx, gy = world_to_map_pixel(snapshot, float(goal["x"]), float(goal["y"]))
            cv2.drawMarker(image, (gx, gy), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        pose = nav.get("pose") or {}
        if "x" in pose and "y" in pose:
            px, py = world_to_map_pixel(snapshot, float(pose["x"]), float(pose["y"]))
            yaw = math.radians(float(pose.get("yaw_deg", 0.0)))
            tip = (int(px + 18 * math.cos(yaw)), int(py - 18 * math.sin(yaw)))
            cv2.circle(image, (px, py), 7, (40, 190, 40), -1)
            cv2.arrowedLine(image, (px, py), tip, (20, 120, 20), 2, tipLength=0.35)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        for topic in self.subscriptions:
            topic.unsubscribe()
        if self.goal_topic:
            self.goal_topic.unadvertise()
        if self.command_topic:
            self.command_topic.unadvertise()
        if self.ros:
            self.ros.terminate()

    @property
    def is_connected(self) -> bool:
        return bool(self.ros and self.ros.is_connected)


class AutonomyConsole:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.robot = RobotBridge(args.host, args.rosbridge_port)
        video_url = (
            f"http://{args.host}:{args.video_port}/stream"
            "?topic=/ascamera/camera_publisher/rgb0/image&type=mjpeg&quality=70"
        )
        self.camera = CameraReceiver(video_url)
        self.worker: StatefulVLAWorker | None = None
        self.lock = threading.Lock()
        self.state_path = Path(getattr(args, "state_file", "state/autonomy_memory.json"))
        saved = self._load_memory()
        self.instruction = str(saved.get("instruction") or args.instruction)
        self.prompt_history = [str(item) for item in saved.get("prompt_history", [])][-12:]
        if self.instruction and (not self.prompt_history or self.prompt_history[-1] != self.instruction):
            self.prompt_history.append(self.instruction)
        self.saved_scene_memory = str(saved.get("scene_memory", ""))
        self.auto_dispatch = False
        self.last_action = ""
        self.last_sample = 0.0
        self.closed = False

    def start(self) -> None:
        self.camera.start()
        self.connect_robot()
        if not self.args.no_vlm:
            with open(self.args.config, "r", encoding="utf-8") as stream:
                config = yaml.safe_load(stream) or {}
            config.setdefault("max_new_tokens", 180)
            config.setdefault("frames_per_inference", 4)
            config.setdefault("chat_reset_turns", 12)
            config["initial_memory"] = self.saved_scene_memory
            self.worker = StatefulVLAWorker(config, on_decision=self._on_decision)
            self.worker.start()

    def connect_robot(self) -> str:
        try:
            self.robot.connect()
            return f"机器人已连接：ws://{self.args.host}:{self.args.rosbridge_port}"
        except Exception as exc:
            with self.robot.lock:
                self.robot.nav_status = {"state": "OFFLINE", "message": str(exc)}
            return f"机器人连接失败：{exc}"

    def set_instruction(self, instruction: str) -> str:
        instruction = instruction.strip()
        if not instruction:
            return "提示词不能为空"
        with self.lock:
            self.instruction = instruction
            self.prompt_history.append(instruction)
            self.prompt_history = self.prompt_history[-12:]
            self._save_memory()
        return f"实时提示词已生效：{instruction}"

    def set_auto_dispatch(self, enabled: bool) -> str:
        with self.lock:
            self.auto_dispatch = bool(enabled)
        return "AI 高层目标自动下发：已开启" if enabled else "AI 高层目标自动下发：已关闭（仅观察）"

    def _on_decision(self, decision: SceneDecision) -> None:
        with self.lock:
            self.saved_scene_memory = decision.memory or self.saved_scene_memory
            self._save_memory()
            enabled = self.auto_dispatch
        if not enabled or decision.confidence < 0.72:
            return
        nav, _, _, _ = self.robot.snapshot()
        if decision.behavior == "STOP":
            self.last_action = self.robot.command("STOP")
            return
        if decision.behavior == "PAUSE":
            self.last_action = self.robot.command("PAUSE")
            return
        if decision.behavior != "NAVIGATE_RELATIVE":
            return
        if nav.get("state") not in {"IDLE", "SUCCEEDED", "FAILED", "CANCELED"}:
            return
        pose = nav.get("pose") or {}
        if pose.get("frame") != "map":
            return
        yaw = math.radians(float(pose.get("yaw_deg", 0.0)))
        gx, gy, _ = relative_goal_to_map(
            float(pose["x"]), float(pose["y"]), yaw, decision.forward_m, decision.left_m
        )
        target_yaw = math.degrees(yaw) + decision.target_yaw_deg
        self.last_action = self.robot.publish_goal(gx, gy, target_yaw)

    def _load_memory(self) -> dict:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_memory(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(
                    {
                        "instruction": self.instruction,
                        "prompt_history": self.prompt_history[-12:],
                        "scene_memory": self.saved_scene_memory,
                        "updated_at": time.time(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    def tick(self):
        frame = self.camera.latest()
        nav, lidar, _, _ = self.robot.snapshot()
        map_rgb = self.robot.render_map()
        now = time.monotonic()
        if self.worker and frame is not None and now - self.last_sample >= 1.0:
            with self.lock:
                instruction = self.instruction
                history = tuple(self.prompt_history)
            map_bgr = cv2.cvtColor(map_rgb, cv2.COLOR_RGB2BGR)
            self.worker.submit(SceneSample(frame, instruction, history, lidar, nav, map_bgr))
            self.last_sample = now
        camera_rgb = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame is not None
            else np.full((360, 640, 3), 30, dtype=np.uint8)
        )
        model = self.worker.snapshot() if self.worker else {"status": "disabled", "decision": None}
        decision = model.get("decision") or {}
        observations = "\n".join(f"- {item}" for item in decision.get("observations", []))
        reasoning = (
            f"场景：{decision.get('scene', '等待模型')}\n\n"
            f"可核验观察：\n{observations or '- 暂无'}\n\n"
            f"决策：{decision.get('behavior', 'HOLD')} / 风险：{decision.get('risk', '-')}\n"
            f"理由：{decision.get('reason', '-')}\n"
            f"记忆：{model.get('memory', '-') or '-'}"
        )
        status = {
            "ros_connected": self.robot.is_connected,
            "navigation": nav,
            "lidar_m": lidar,
            "model": {key: value for key, value in model.items() if key != "decision"},
            "last_action": self.last_action,
        }
        return camera_rgb, map_rgb, reasoning, json.dumps(status, ensure_ascii=False, indent=2)

    def goal_from_numbers(self, x: float, y: float, yaw: float) -> str:
        return self.robot.publish_goal(float(x), float(y), float(yaw))

    def goal_from_map(self, event) -> tuple[float, float, str]:
        _, _, snapshot, _ = self.robot.snapshot()
        if snapshot is None:
            return 0.0, 0.0, "地图尚不可用"
        x, y = map_pixel_to_world(snapshot, event.index[0], event.index[1])
        return round(x, 3), round(y, 3), "已选择地图坐标；点击“发送目标”开始导航"

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.worker:
            self.worker.close()
        self.camera.close()
        self.robot.close()


def build_ui(console: AutonomyConsole):
    def select_map_goal(event: gr.SelectData):
        return console.goal_from_map(event)

    with gr.Blocks(title="ROSOrin 连续自主导航") as app:
        gr.Markdown("# ROSOrin 连续自主导航\n二维建图、Nav2 目标导航与 Qwen3-VL 连续场景记忆。红色按钮始终直接请求停车。")
        with gr.Row():
            camera = gr.Image(label="连续相机", interactive=False)
            map_image = gr.Image(label="二维地图（点击选点）", interactive=False)
        with gr.Row():
            instruction = gr.Textbox(value=console.instruction, label="实时提示词", scale=5)
            apply_prompt = gr.Button("应用提示词", variant="primary")
            reconnect = gr.Button("连接机器人")
            auto = gr.Checkbox(value=False, label="允许 AI 自动下发高层目标")
        prompt_result = gr.Textbox(label="操作结果", interactive=False)
        with gr.Row():
            x = gr.Number(value=0.0, label="目标 X / m")
            y = gr.Number(value=0.0, label="目标 Y / m")
            yaw = gr.Number(value=0.0, label="朝向 / deg")
            send_goal = gr.Button("发送目标", variant="primary")
        with gr.Row():
            stop = gr.Button("紧急停止", variant="stop")
            pause = gr.Button("暂停")
            resume = gr.Button("继续")
            cancel = gr.Button("取消目标")
            backup = gr.Button("受控倒车 0.15m")
            spin = gr.Button("原地旋转 45°")
            save = gr.Button("保存地图")
        with gr.Row():
            reasoning = gr.Textbox(label="每轮可核验观察与决策理由", lines=11, interactive=False)
            status = gr.Code(label="实时状态", language="json", lines=16)

        apply_prompt.click(console.set_instruction, instruction, prompt_result)
        reconnect.click(console.connect_robot, None, prompt_result)
        auto.change(console.set_auto_dispatch, auto, prompt_result)
        send_goal.click(console.goal_from_numbers, [x, y, yaw], prompt_result)
        map_image.select(select_map_goal, None, [x, y, prompt_result])
        stop.click(lambda: console.robot.command("STOP"), None, prompt_result)
        pause.click(lambda: console.robot.command("PAUSE"), None, prompt_result)
        resume.click(lambda: console.robot.command("RESUME"), None, prompt_result)
        cancel.click(lambda: console.robot.command("CANCEL"), None, prompt_result)
        backup.click(lambda: console.robot.command("BACKUP", distance_m=0.15), None, prompt_result)
        spin.click(lambda: console.robot.command("SPIN", angle_deg=45), None, prompt_result)
        save.click(lambda: console.robot.command("SAVE_MAP", name="rosorin_map"), None, prompt_result)
        timer = gr.Timer(1.0)
        timer.tick(console.tick, None, [camera, map_image, reasoning, status])
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROSOrin mapping and stateful VLM console")
    parser.add_argument("--host", default="192.168.3.17")
    parser.add_argument("--rosbridge-port", type=int, default=9090)
    parser.add_argument("--video-port", type=int, default=8080)
    parser.add_argument("--config", default="vla_config.yaml")
    parser.add_argument("--instruction", default="在封闭场地连续探索可通行区域，优先选择开阔路线")
    parser.add_argument("--no-vlm", action="store_true", help="只运行地图和 Nav2 控制台")
    parser.add_argument("--ui-port", type=int, default=7860)
    parser.add_argument("--state-file", default="state/autonomy_memory.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = AutonomyConsole(args)
    app = build_ui(console)
    console.start()
    atexit.register(console.close)
    app.launch(server_name="127.0.0.1", server_port=args.ui_port, inbrowser=True)


if __name__ == "__main__":
    main()
