#!/usr/bin/env python3
"""Validate the live D435i stream and capture one FP++ initialization frame.

The RealSense driver runs where the camera is attached; this script only needs
the topics, so it can run on the host or inside the container with
`--network host`.
"""
from pathlib import Path
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import message_filters
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from perception_plus_plus_core.validation.camera import (
    CameraObservation,
    check_camera_contract,
)
from perception_plus_plus_core.validation.depth import depth_to_meters
from perception_plus_plus_core.validation.readiness import CheckResult, write_report

ROOT = Path(__file__).resolve().parents[1]


class CameraProbe(Node):
    def __init__(self, args) -> None:
        super().__init__("camera_check")
        self.bridge = CvBridge()
        self.frames = 0
        self.latest = None
        self.started = time.monotonic()
        subscribers = [
            message_filters.Subscriber(self, Image, args.rgb_topic,
                                       qos_profile=qos_profile_sensor_data),
            message_filters.Subscriber(self, Image, args.depth_topic,
                                       qos_profile=qos_profile_sensor_data),
            message_filters.Subscriber(self, CameraInfo, args.camera_info_topic,
                                       qos_profile=qos_profile_sensor_data),
        ]
        self.sync = message_filters.ApproximateTimeSynchronizer(
            subscribers, args.sync_queue_size, args.sync_slop_seconds)
        self.sync.registerCallback(self._callback)

    def _callback(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        if self.frames == 0:
            self.started = time.monotonic()
        self.frames += 1
        self.latest = (rgb_msg, depth_msg, info_msg)

    @property
    def elapsed(self) -> float:
        return max(time.monotonic() - self.started, 1e-9)


def collect(node: CameraProbe, frames: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while rclpy.ok() and node.frames < frames and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)


def segment_cup(rgb: np.ndarray, args) -> tuple[np.ndarray | None, str]:
    from perception_plus_plus_core.detection.yolo import YoloCupDetector

    weights = Path(args.yolo) if Path(args.yolo).is_absolute() else ROOT / args.yolo
    detections = YoloCupDetector(weights, args.class_id, args.confidence).detect(rgb)
    if not detections:
        return None, (f"no COCO class {args.class_id} instance above "
                      f"confidence {args.confidence}")
    best = detections[0]
    return best.mask, f"confidence {best.confidence:.3f}, {int(best.mask.sum())} px"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--depth-topic",
                        default="/camera/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-info-topic",
                        default="/camera/camera/color/camera_info")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--sync-slop-seconds", type=float, default=0.04)
    parser.add_argument("--sync-queue-size", type=int, default=10)
    parser.add_argument("--save", type=Path,
                        help="write an FP++ initialization NPZ to this path")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "reports/camera_readiness.json")
    parser.add_argument("--preview", type=Path,
                        help="write the captured RGB frame as a PNG for inspection")
    parser.add_argument("--yolo", default="models/yolo/yolo11n-seg.pt")
    parser.add_argument("--class-id", type=int, default=41)
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()

    rclpy.init()
    node = CameraProbe(args)
    try:
        collect(node, args.frames, args.timeout)
        frames, elapsed, latest = node.frames, node.elapsed, node.latest
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if latest is None:
        checks = [CheckResult("frame_delivery", "FAIL", True,
                              f"no synchronized frames on {args.rgb_topic}, "
                              f"{args.depth_topic}, {args.camera_info_topic} "
                              f"within {args.timeout}s")]
        status = write_report(args.report, checks)
        print(f"FAIL frame_delivery: {checks[0].detail}")
        print(status)
        return 1

    rgb_msg, depth_msg, info_msg = latest
    bridge = CvBridge()
    rgb = bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
    raw_depth = bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
    depth = depth_to_meters(raw_depth, depth_msg.encoding)
    k = np.asarray(info_msg.k, dtype=np.float64).reshape(3, 3)
    checks = check_camera_contract(CameraObservation(
        color_encoding=rgb_msg.encoding,
        depth_encoding=depth_msg.encoding,
        color_shape=(rgb_msg.height, rgb_msg.width),
        depth_shape=(depth_msg.height, depth_msg.width),
        color_frame_id=rgb_msg.header.frame_id,
        depth_frame_id=depth_msg.header.frame_id,
        info_frame_id=info_msg.header.frame_id,
        k=k,
        frames=frames,
        duration_s=elapsed,
    ))

    if args.preview is not None:
        import imageio.v3 as iio

        args.preview.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(args.preview, rgb)
        checks.append(CheckResult("preview_frame", "PASS", False, str(args.preview)))

    if args.save is not None:
        mask, detail = segment_cup(rgb, args)
        checks.append(CheckResult("cup_segmentation",
                                  "PASS" if mask is not None else "FAIL", True, detail))
        if mask is not None:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.save.with_suffix(args.save.suffix + ".part")
            np.savez(temporary, rgb=rgb, depth=depth.astype(np.float32),
                     mask=mask.astype(bool), K=k,
                     timestamps_ns=np.array([
                         rgb_msg.header.stamp.sec * 1_000_000_000
                         + rgb_msg.header.stamp.nanosec], dtype=np.int64))
            temporary.replace(args.save)
            checks.append(CheckResult("initialization_frame", "PASS", True,
                                      str(args.save)))

    status = write_report(args.report, checks)
    for check in checks:
        print(f"{check.status:4} {check.name}: {check.detail}")
    print(status)
    return 0 if status == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
