import math
import threading
import time
import unittest

import numpy as np

from autonomy_console import AutonomyConsole, MapSnapshot, map_pixel_to_world, world_to_map_pixel
from streaming_vla import (
    SceneSample,
    StatefulVLAWorker,
    parse_scene_decision,
    relative_goal_to_map,
)


class StreamingAutonomyTests(unittest.TestCase):
    def test_parse_and_bound_relative_decision(self):
        decision = parse_scene_decision(
            'prefix {"scene":"走廊","observations":["前方开阔"],"memory":"刚从门口进入",'
            '"risk":"low","behavior":"navigate_relative",'
            '"relative_goal_m":{"forward":2,"left":-2},"target_yaw_deg":120,'
            '"confidence":1.4,"reason":"继续探索"}'
        )
        self.assertEqual(decision.behavior, "NAVIGATE_RELATIVE")
        self.assertEqual(decision.risk, "LOW")
        self.assertEqual(decision.forward_m, 0.60)
        self.assertEqual(decision.left_m, -0.40)
        self.assertEqual(decision.target_yaw_deg, 90.0)
        self.assertEqual(decision.confidence, 1.0)

    def test_relative_goal_rotates_into_map_frame(self):
        x, y, _ = relative_goal_to_map(1.0, 2.0, math.pi / 2.0, 0.5, 0.2)
        self.assertAlmostEqual(x, 0.8)
        self.assertAlmostEqual(y, 2.5)

    def test_map_pixel_round_trip(self):
        snapshot = MapSnapshot(np.zeros((20, 30)), 0.05, -1.0, -0.5, 3)
        pixel = world_to_map_pixel(snapshot, -0.5, 0.0)
        x, y = map_pixel_to_world(snapshot, *pixel)
        self.assertLess(abs(x + 0.5), 0.051)
        self.assertLess(abs(y - 0.0), 0.051)

    def test_invalidated_inference_cannot_dispatch(self):
        started = threading.Event()
        release = threading.Event()
        callbacks = []

        class Backend:
            def __init__(self, *_args):
                pass

            def generate(self, _images, _prompt):
                started.set()
                release.wait(2.0)
                return (
                    '{"scene":"clear","observations":[],"memory":"new",'
                    '"risk":"LOW","behavior":"NAVIGATE_RELATIVE",'
                    '"relative_goal_m":{"forward":0.3,"left":0},'
                    '"target_yaw_deg":0,"confidence":0.9,"reason":"clear"}'
                )

            def reset(self, _memory):
                pass

        worker = StatefulVLAWorker(
            {"frames_per_inference": 1, "image_max_side": 64},
            on_decision=lambda decision, generation: callbacks.append(
                (decision, generation)
            ),
            backend_factory=Backend,
        )
        worker.start()
        sample = SceneSample(
            np.zeros((64, 64, 3), np.uint8), "explore", (), {}, {"state": "IDLE"}
        )
        worker.submit(sample)
        self.assertTrue(started.wait(1.0))
        generation = worker.invalidate(memory="")
        release.set()
        deadline = time.monotonic() + 2.0
        while worker.snapshot()["status"] == "inferencing" and time.monotonic() < deadline:
            time.sleep(0.01)
        snapshot = worker.snapshot()
        worker.close()
        self.assertEqual(generation, 1)
        self.assertEqual(callbacks, [])
        self.assertIsNone(snapshot["decision"])

    def test_prompt_change_advances_enabled_auto_epoch(self):
        class Worker:
            def invalidate(self, memory=None):
                self.memory = memory
                return 9

        console = object.__new__(AutonomyConsole)
        console.lock = threading.Lock()
        console.instruction = "old"
        console.prompt_history = []
        console.saved_scene_memory = "stale"
        console.auto_dispatch = True
        console.auto_epoch = 3
        console.worker = Worker()
        console._save_memory = lambda: None
        result, enabled = console.set_instruction("continue exploring")
        self.assertTrue(enabled)
        self.assertEqual(console.auto_epoch, 9)
        self.assertEqual(console.saved_scene_memory, "")
        self.assertIn("continue exploring", result)


if __name__ == "__main__":
    unittest.main()
