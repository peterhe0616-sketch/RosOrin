import math
import unittest

import numpy as np

from autonomy_console import MapSnapshot, map_pixel_to_world, world_to_map_pixel
from streaming_vla import parse_scene_decision, relative_goal_to_map


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


if __name__ == "__main__":
    unittest.main()

