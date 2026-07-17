"""Small ROS 2 API surface between rosbridge clients and Nav2 actions."""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import BackUp, NavigateToPose, Spin
from nav2_msgs.srv import SaveMap
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


def yaw_from_quaternion(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def roll_pitch_from_quaternion(q) -> tuple[float, float]:
    roll = math.atan2(
        2.0 * (q.w * q.x + q.y * q.z),
        1.0 - 2.0 * (q.x * q.x + q.y * q.y),
    )
    sin_pitch = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    return roll, pitch


class AutonomyBridge(Node):
    def __init__(self) -> None:
        super().__init__("autonomy_bridge")
        self.declare_parameter("status_rate_hz", 2.0)
        self.declare_parameter("stuck_timeout_s", 2.5)
        self.declare_parameter("movement_epsilon_m", 0.025)
        self.declare_parameter("rotation_epsilon_rad", 0.08)
        self.declare_parameter("map_directory", "/home/ubuntu/shared/maps")

        self.status_pub = self.create_publisher(String, "/vla/navigation_status", 10)
        self.emergency_pub = self.create_publisher(Twist, "/controller/cmd_vel", 10)
        self.create_subscription(PoseStamped, "/vla/nav_goal", self.on_goal, 10)
        self.create_subscription(String, "/vla/nav_command", self.on_command, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 20)
        self.create_subscription(Imu, "/imu", self.on_imu, 40)
        self.create_subscription(Twist, "/autonomy/cmd_vel_smoothed", self.on_cmd, 20)
        self.create_subscription(Twist, "/controller/cmd_vel", self.on_safe_cmd, 20)

        self.navigate = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.backup = ActionClient(self, BackUp, "/backup")
        self.spin = ActionClient(self, Spin, "/spin")
        self.save_map = self.create_client(SaveMap, "/map_saver/save_map")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.lock = threading.Lock()
        self.nav_state = "IDLE"
        self.nav_message = "waiting for a goal"
        self.active_goal: PoseStamped | None = None
        self.paused_goal: PoseStamped | None = None
        self.goal_handle = None
        self.distance_remaining = math.nan
        self.last_cmd = (0.0, 0.0)
        self.last_safe_cmd = (0.0, 0.0)
        self.odom_pose = (0.0, 0.0, 0.0)
        self.actual_velocity = (0.0, 0.0)
        self.imu_attitude = (0.0, 0.0)
        self.accel_samples: deque[float] = deque(maxlen=30)
        self.vibration_rms = 0.0
        self.map_pose: tuple[float, float, float] | None = None
        self.last_moving_pose: tuple[float, float, float] | None = None
        self.last_progress_time = time.monotonic()
        self.stuck = False
        self.last_map_path = ""
        self.emergency_until = 0.0

        rate = max(0.5, float(self.get_parameter("status_rate_hz").value))
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info("ROSOrin autonomy bridge ready")

    def on_odom(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        with self.lock:
            self.odom_pose = (
                float(pose.position.x),
                float(pose.position.y),
                yaw_from_quaternion(pose.orientation),
            )
            self.actual_velocity = (
                float(msg.twist.twist.linear.x),
                float(msg.twist.twist.angular.z),
            )

    def on_imu(self, msg: Imu) -> None:
        roll, pitch = roll_pitch_from_quaternion(msg.orientation)
        acceleration = msg.linear_acceleration
        magnitude = math.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2)
        with self.lock:
            self.imu_attitude = (roll, pitch)
            self.accel_samples.append(magnitude)
            if len(self.accel_samples) >= 4:
                mean = sum(self.accel_samples) / len(self.accel_samples)
                self.vibration_rms = math.sqrt(
                    sum((value - mean) ** 2 for value in self.accel_samples)
                    / len(self.accel_samples)
                )

    def on_cmd(self, msg: Twist) -> None:
        with self.lock:
            self.last_cmd = (float(msg.linear.x), float(msg.angular.z))

    def on_safe_cmd(self, msg: Twist) -> None:
        with self.lock:
            self.last_safe_cmd = (float(msg.linear.x), float(msg.angular.z))

    def on_goal(self, msg: PoseStamped) -> None:
        if not msg.header.frame_id:
            msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        self.send_navigation_goal(msg)

    def send_navigation_goal(self, goal_pose: PoseStamped) -> None:
        if not self.navigate.wait_for_server(timeout_sec=2.0):
            self.set_state("ERROR", "navigate_to_pose action is unavailable")
            return
        goal = NavigateToPose.Goal()
        goal.pose = goal_pose
        with self.lock:
            self.active_goal = goal_pose
            self.paused_goal = None
            self.nav_state = "STARTING"
            self.nav_message = "sending navigation goal"
            self.distance_remaining = math.nan
            self.stuck = False
        future = self.navigate.send_goal_async(goal, feedback_callback=self.on_feedback)
        future.add_done_callback(self.on_goal_response)

    def on_goal_response(self, future) -> None:
        try:
            handle = future.result()
        except Exception as exc:
            self.set_state("ERROR", f"goal request failed: {exc}")
            return
        if not handle.accepted:
            self.set_state("REJECTED", "navigation goal rejected")
            return
        with self.lock:
            self.goal_handle = handle
            self.nav_state = "NAVIGATING"
            self.nav_message = "goal accepted"
        result = handle.get_result_async()
        result.add_done_callback(self.on_navigation_result)

    def on_feedback(self, feedback_msg) -> None:
        feedback = feedback_msg.feedback
        with self.lock:
            self.distance_remaining = float(feedback.distance_remaining)
            self.nav_state = "NAVIGATING"
            self.nav_message = "following path"

    def on_navigation_result(self, future) -> None:
        try:
            wrapped = future.result()
            status = int(wrapped.status)
        except Exception as exc:
            self.set_state("ERROR", f"navigation result failed: {exc}")
            return
        states = {
            4: ("SUCCEEDED", "goal reached"),
            5: ("CANCELED", "navigation canceled"),
            6: ("FAILED", "navigation aborted after recovery attempts"),
        }
        state, message = states.get(status, ("FINISHED", f"action status={status}"))
        with self.lock:
            self.nav_state = state
            self.nav_message = message
            self.goal_handle = None
            if status == 4:
                self.active_goal = None
                self.distance_remaining = 0.0

    def on_command(self, msg: String) -> None:
        try:
            data = json.loads(msg.data) if msg.data.strip().startswith("{") else {"command": msg.data}
        except json.JSONDecodeError as exc:
            self.set_state("ERROR", f"invalid command JSON: {exc}")
            return
        command = str(data.get("command", "")).upper().strip()
        if command in {"STOP", "CANCEL"}:
            self.cancel(emergency=command == "STOP")
        elif command == "PAUSE":
            with self.lock:
                self.paused_goal = self.active_goal
            self.cancel(emergency=False, state="PAUSED")
        elif command == "RESUME":
            with self.lock:
                goal = self.paused_goal
            if goal is None:
                self.set_state("IDLE", "no paused goal")
            else:
                self.send_navigation_goal(goal)
        elif command == "BACKUP":
            self.run_backup(float(data.get("distance_m", 0.15)))
        elif command == "SPIN":
            self.run_spin(math.radians(float(data.get("angle_deg", 45.0))))
        elif command == "SAVE_MAP":
            self.run_save_map(str(data.get("name", "rosorin_map")))
        else:
            self.set_state("ERROR", f"unsupported command: {command or '<empty>'}")

    def cancel(self, emergency: bool, state: str = "CANCELED") -> None:
        with self.lock:
            handle = self.goal_handle
            self.nav_state = state
            self.nav_message = "operator stop" if emergency else state.lower()
            if state != "PAUSED":
                self.active_goal = None
            if emergency:
                self.emergency_until = time.monotonic() + 1.0
        if handle is not None:
            handle.cancel_goal_async()
        if emergency:
            self.emergency_pub.publish(Twist())

    def run_backup(self, distance_m: float) -> None:
        distance_m = min(0.30, max(0.05, abs(distance_m)))
        if not self.backup.wait_for_server(timeout_sec=2.0):
            self.set_state("ERROR", "backup action unavailable")
            return
        goal = BackUp.Goal()
        goal.target.x = -distance_m
        goal.speed = 0.04
        goal.time_allowance.sec = 8
        self.set_state("RECOVERING", f"backing up {distance_m:.2f}m")
        self.backup.send_goal_async(goal)

    def run_spin(self, angle_rad: float) -> None:
        angle_rad = min(math.pi, max(-math.pi, angle_rad))
        if not self.spin.wait_for_server(timeout_sec=2.0):
            self.set_state("ERROR", "spin action unavailable")
            return
        goal = Spin.Goal()
        goal.target_yaw = angle_rad
        goal.time_allowance.sec = 10
        self.set_state("RECOVERING", f"spinning {math.degrees(angle_rad):.0f}deg")
        self.spin.send_goal_async(goal)

    def run_save_map(self, name: str) -> None:
        safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_") or "rosorin_map"
        directory = Path(str(self.get_parameter("map_directory").value))
        directory.mkdir(parents=True, exist_ok=True)
        path = str(directory / safe)
        if not self.save_map.wait_for_service(timeout_sec=2.0):
            self.set_state("ERROR", "map saver service unavailable")
            return
        request = SaveMap.Request()
        request.map_topic = "/map"
        request.map_url = path
        request.image_format = "pgm"
        request.map_mode = "trinary"
        request.free_thresh = 0.25
        request.occupied_thresh = 0.65
        future = self.save_map.call_async(request)
        future.add_done_callback(lambda done: self.on_map_saved(done, path))
        self.set_state("SAVING_MAP", path)

    def on_map_saved(self, future, path: str) -> None:
        try:
            success = bool(future.result().result)
        except Exception as exc:
            self.set_state("ERROR", f"map save failed: {exc}")
            return
        if success:
            with self.lock:
                self.last_map_path = path
            self.set_state("IDLE", f"map saved: {path}")
        else:
            self.set_state("ERROR", "map saver returned failure")

    def set_state(self, state: str, message: str) -> None:
        with self.lock:
            self.nav_state = state
            self.nav_message = message

    def update_map_pose(self) -> None:
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "base_footprint", Time()
            )
        except TransformException:
            return
        t = transform.transform.translation
        q = transform.transform.rotation
        with self.lock:
            self.map_pose = (float(t.x), float(t.y), yaw_from_quaternion(q))

    def update_stuck(self) -> None:
        now = time.monotonic()
        timeout = float(self.get_parameter("stuck_timeout_s").value)
        epsilon = float(self.get_parameter("movement_epsilon_m").value)
        rotation_epsilon = float(self.get_parameter("rotation_epsilon_rad").value)
        with self.lock:
            pose = self.map_pose or self.odom_pose
            cmd = self.last_safe_cmd
            active = self.nav_state == "NAVIGATING" and (
                abs(cmd[0]) > 0.015 or abs(cmd[1]) > 0.06
            )
            current = (pose[0], pose[1], pose[2])
            if self.last_moving_pose is None:
                self.last_moving_pose = current
                self.last_progress_time = now
            else:
                translated = math.dist(current[:2], self.last_moving_pose[:2])
                yaw_delta = abs(
                    math.atan2(
                        math.sin(current[2] - self.last_moving_pose[2]),
                        math.cos(current[2] - self.last_moving_pose[2]),
                    )
                )
                progressed = translated >= epsilon or yaw_delta >= rotation_epsilon
                if progressed:
                    self.last_moving_pose = current
                    self.last_progress_time = now
                    self.stuck = False
                elif active and now - self.last_progress_time >= timeout:
                    self.stuck = True
                elif not active:
                    self.last_progress_time = now
                    self.last_moving_pose = current
                    self.stuck = False

    def tick(self) -> None:
        self.update_map_pose()
        self.update_stuck()
        now = time.monotonic()
        with self.lock:
            if now < self.emergency_until:
                self.emergency_pub.publish(Twist())
            pose = self.map_pose or self.odom_pose
            goal = self.active_goal
            payload = {
                "state": self.nav_state,
                "message": self.nav_message,
                "pose": {
                    "x": round(pose[0], 4),
                    "y": round(pose[1], 4),
                    "yaw_deg": round(math.degrees(pose[2]), 2),
                    "frame": "map" if self.map_pose is not None else "odom",
                },
                "goal": (
                    {
                        "x": round(float(goal.pose.position.x), 4),
                        "y": round(float(goal.pose.position.y), 4),
                        "frame": goal.header.frame_id,
                    }
                    if goal is not None
                    else None
                ),
                "distance_remaining_m": (
                    round(self.distance_remaining, 3)
                    if math.isfinite(self.distance_remaining)
                    else None
                ),
                "cmd": {
                    "requested_linear": round(self.last_cmd[0], 3),
                    "requested_angular": round(self.last_cmd[1], 3),
                    "approved_linear": round(self.last_safe_cmd[0], 3),
                    "approved_angular": round(self.last_safe_cmd[1], 3),
                },
                "stuck": self.stuck,
                "slip_suspected": self.stuck and abs(self.last_safe_cmd[0]) > 0.015,
                "motion": {
                    "actual_linear_mps": round(self.actual_velocity[0], 3),
                    "actual_angular_radps": round(self.actual_velocity[1], 3),
                    "roll_deg": round(math.degrees(self.imu_attitude[0]), 2),
                    "pitch_deg": round(math.degrees(self.imu_attitude[1]), 2),
                    "vibration_rms_mps2": round(self.vibration_rms, 3),
                },
                "last_map_path": self.last_map_path or None,
                "timestamp": time.time(),
            }
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AutonomyBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
