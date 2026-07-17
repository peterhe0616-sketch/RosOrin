"""Publish a fixed-size LaserScan for consumers that require stable geometry."""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def nearest_sample(values, index: int, default: float):
    return values[index] if index < len(values) else default


class ScanNormalizer(Node):
    """Resample LD19 revolutions whose raw point count varies by a few samples."""

    def __init__(self) -> None:
        super().__init__("scan_normalizer")
        self.declare_parameter("input_topic", "/scan_raw")
        self.declare_parameter("output_topic", "/scan_slam")
        self.declare_parameter("bins", 504)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.bins = max(32, int(self.get_parameter("bins").value))
        self.publisher = self.create_publisher(
            LaserScan, output_topic, qos_profile_sensor_data
        )
        self.subscription = self.create_subscription(
            LaserScan, input_topic, self.on_scan, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"normalizing {input_topic} to {self.bins} bins on {output_topic}"
        )

    def on_scan(self, source: LaserScan) -> None:
        count = len(source.ranges)
        if count < 2 or not math.isfinite(source.angle_increment) or source.angle_increment <= 0:
            return

        result = LaserScan()
        result.header = source.header
        result.angle_min = source.angle_min
        result.angle_max = source.angle_max
        result.angle_increment = (
            (source.angle_max - source.angle_min) / float(self.bins - 1)
        )
        result.time_increment = source.scan_time / float(self.bins)
        result.scan_time = source.scan_time
        result.range_min = source.range_min
        result.range_max = source.range_max

        ranges = []
        intensities = []
        for output_index in range(self.bins):
            angle = result.angle_min + output_index * result.angle_increment
            input_index = round((angle - source.angle_min) / source.angle_increment)
            input_index = min(count - 1, max(0, input_index))
            ranges.append(float(source.ranges[input_index]))
            if source.intensities:
                intensities.append(
                    float(nearest_sample(source.intensities, input_index, math.nan))
                )

        result.ranges = ranges
        result.intensities = intensities
        self.publisher.publish(result)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ScanNormalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
