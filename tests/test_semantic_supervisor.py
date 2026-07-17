import json
import math
import time
import unittest

import numpy as np

from qwen_vl_runtime import VLARequest, VLAWorker, resize_for_vlm
from semantic_supervisor import (
    LidarSectors,
    SafetyPolicy,
    VLAResult,
    compute_lidar_sectors,
    parse_vla_json,
)


class SemanticSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.policy = SafetyPolicy(
            {
                "stop_distance_m": 0.16,
                "turn_clearance_m": 0.16,
                "minimum_confidence": 0.65,
                "max_result_age_s": 5.0,
                "max_command_duration_s": 0.6,
                "require_lidar": True,
                "max_linear_mps": 0.10,
                "max_angular_radps": 0.35,
            }
        )

    def result(self, action="FORWARD", confidence=0.9):
        return VLAResult("corridor", (), action, 0.0, 0.08, confidence, "clear")

    def test_parse_json_with_markdown_wrapper(self):
        payload = {
            "scene": "走廊",
            "hazards": ["纸箱"],
            "action": "slow_right",
            "target_heading_deg": -10,
            "max_speed_mps": 0.05,
            "confidence": 0.8,
            "reason": "右侧更宽",
        }
        result = parse_vla_json(f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```")
        self.assertEqual(result.action, "SLOW_RIGHT")
        self.assertEqual(result.hazards, ("纸箱",))

    def test_front_obstacle_vetoes_forward(self):
        decision = self.policy.evaluate(
            self.result(), LidarSectors(front=0.12, left=1.0, right=1.0), 0.1
        )
        self.assertEqual(decision.action, "STOP")
        self.assertEqual(decision.linear, 0.0)

    def test_low_confidence_and_stale_results_stop(self):
        sectors = LidarSectors(front=2.0, left=2.0, right=2.0)
        self.assertEqual(self.policy.evaluate(self.result(confidence=0.2), sectors, 0.1).action, "STOP")
        self.assertEqual(self.policy.evaluate(self.result(), sectors, 10.0).action, "STOP")

    def test_safe_turn_is_bounded(self):
        decision = self.policy.evaluate(
            self.result("TURN_LEFT"), LidarSectors(front=1.0, left=2.0, right=2.0), 0.1
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.linear, 0.0)
        self.assertLessEqual(decision.angular, 0.35)

    def test_motion_command_is_a_short_pulse(self):
        decision = self.policy.evaluate(
            self.result(), LidarSectors(front=2.0, left=2.0, right=2.0), 0.8
        )
        self.assertEqual(decision.action, "STOP")
        self.assertIn("pulse", decision.reason)

    def test_missing_lidar_vetoes_motion(self):
        decision = self.policy.evaluate(self.result(), LidarSectors(), 0.1)
        self.assertEqual(decision.action, "STOP")
        self.assertIn("LiDAR", decision.reason)

    def test_lidar_sector_summary(self):
        ranges = [2.0] * 360
        ranges[180] = 0.4  # angle 0: front
        ranges[270] = 0.7  # +90 degrees: left
        ranges[90] = 0.8   # -90 degrees: right
        sectors = compute_lidar_sectors(ranges, -math.pi, math.pi / 180, 0.02, 10.0)
        self.assertAlmostEqual(sectors.front, 0.4)
        self.assertAlmostEqual(sectors.left, 0.7)
        self.assertAlmostEqual(sectors.right, 0.8)

    def test_resize_preserves_reasonable_dimensions(self):
        resized = resize_for_vlm(np.zeros((1080, 1920, 3), np.uint8), 672)
        self.assertLessEqual(max(resized.shape[:2]), 672)
        self.assertEqual(resized.shape[0] % 28, 0)
        self.assertEqual(resized.shape[1] % 28, 0)


class FakeBackend:
    def __init__(self, *_args):
        pass

    def generate(self, _frame, _prompt):
        return json.dumps(
            {
                "scene": "test",
                "hazards": [],
                "action": "STOP",
                "target_heading_deg": 0,
                "max_speed_mps": 0,
                "confidence": 1,
                "reason": "test",
            }
        )


class WorkerTests(unittest.TestCase):
    def test_worker_runs_backend_off_thread(self):
        worker = VLAWorker(
            {"model_dir": ".", "device": "CPU", "max_new_tokens": 8, "image_max_side": 224},
            backend_factory=FakeBackend,
        )
        worker.start()
        deadline = time.monotonic() + 2
        while worker.state.snapshot()["status"] != "ready" and time.monotonic() < deadline:
            time.sleep(0.01)
        accepted = worker.submit(
            VLARequest(np.zeros((240, 320, 3), np.uint8), "stop", LidarSectors(), 0.0, 0.0)
        )
        self.assertTrue(accepted)
        while worker.state.snapshot()["sequence"] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        snapshot = worker.state.snapshot()
        worker.close()
        self.assertEqual(snapshot["result"].action, "STOP")
        self.assertEqual(snapshot["error"], "")


if __name__ == "__main__":
    unittest.main()
