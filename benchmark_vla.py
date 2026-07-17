"""Benchmark the converted Qwen3-VL model on this computer."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from qwen_vl_runtime import OpenVINOVLMBackend, VLARequest, build_prompt, resize_for_vlm
from semantic_supervisor import LidarSectors, parse_vla_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OpenVINO Qwen3-VL")
    parser.add_argument("--config", default="vla_config.yaml")
    parser.add_argument("--image", help="Input image; synthetic scene is used when omitted")
    parser.add_argument("--device", help="Override CPU/GPU")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, help="Override generated token limit")
    parser.add_argument("--image-max-side", type=int, help="Override image size")
    parser.add_argument("--decision", action="store_true", help="Use and validate the driving JSON prompt")
    return parser.parse_args()


def synthetic_scene() -> np.ndarray:
    image = np.full((448, 672, 3), (175, 175, 175), dtype=np.uint8)
    cv2.rectangle(image, (0, 310), (671, 447), (70, 70, 70), -1)
    cv2.line(image, (220, 447), (300, 220), (245, 245, 245), 5)
    cv2.line(image, (452, 447), (372, 220), (245, 245, 245), 5)
    cv2.rectangle(image, (280, 250), (360, 360), (30, 80, 210), -1)
    cv2.putText(image, "OBSTACLE", (250, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    return image


def main() -> int:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    device = args.device or config.get("device", "GPU")
    frame = cv2.imread(args.image) if args.image else synthetic_scene()
    if frame is None:
        raise FileNotFoundError(args.image)
    frame = resize_for_vlm(
        frame, args.image_max_side or int(config.get("image_max_side", 672))
    )
    started = time.monotonic()
    backend = OpenVINOVLMBackend(
        config["model_dir"],
        device,
        args.max_new_tokens or int(config.get("max_new_tokens", 96)),
        config.get("cache_dir", "models/qwen3-vl-cache"),
    )
    load_seconds = time.monotonic() - started
    if args.decision:
        prompt = build_prompt(
            VLARequest(
                frame,
                "绕过前方障碍物，无法安全通过时停车",
                LidarSectors(nearest=0.8, front=0.8, left=1.5, right=1.2),
                0.0,
                0.0,
            )
        )
    else:
        prompt = "只输出简短中文：描述画面中的道路、障碍物和最安全的通行方向。"
    for _ in range(max(0, args.warmup)):
        backend.generate(frame, prompt)
    latencies: list[float] = []
    output = ""
    for _ in range(max(1, args.iterations)):
        started = time.monotonic()
        output = backend.generate(frame, prompt)
        latencies.append(time.monotonic() - started)
    print(f"device={device} image={frame.shape[1]}x{frame.shape[0]}")
    print(f"load={load_seconds:.2f}s")
    print(
        f"inference mean={statistics.mean(latencies):.2f}s "
        f"min={min(latencies):.2f}s max={max(latencies):.2f}s"
    )
    print(f"output={output}")
    if args.decision:
        parsed = parse_vla_json(output)
        print(f"parsed action={parsed.action} confidence={parsed.confidence:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
