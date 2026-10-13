#!/usr/bin/env python3
"""vision-3090 에서 RealSense 컬러 영상 한 장 + 내부 파라미터를 npz 로 저장한다(읽기 전용).

    ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=126 python3 scripts/save_color_frame.py --out /tmp/frame.npz

카메라·FP++ 와 같은 localhost DDS 안에서 구독한다. 저장: rgb(uint8 HxWx3 RGB), K(3x3), stamp_s, frame_id.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

COLOR_TOPIC = "/camera/camera/color/image_raw"
INFO_TOPIC = "/camera/camera/color/camera_info"


def to_rgb(encoding: str, height: int, width: int, data: bytes) -> np.ndarray:
    buf = np.frombuffer(data, np.uint8)
    if encoding == "rgb8":
        return buf.reshape(height, width, 3).copy()
    if encoding == "bgr8":
        return buf.reshape(height, width, 3)[:, :, ::-1].copy()
    raise ValueError(f"영상 encoding {encoding} 미지원(rgb8 · bgr8)")


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    rclpy.init()
    node = Node("save_color_frame")
    got: dict = {}
    node.create_subscription(Image, COLOR_TOPIC, lambda m: got.__setitem__("img", m), qos_profile_sensor_data)
    node.create_subscription(CameraInfo, INFO_TOPIC, lambda m: got.__setitem__("info", m), qos_profile_sensor_data)
    t0 = time.time()
    while time.time() - t0 < args.timeout and not {"img", "info"} <= set(got):
        rclpy.spin_once(node, timeout_sec=0.2)
    if not {"img", "info"} <= set(got):
        print(f"❌ {args.timeout}s 안에 영상·camera_info 를 못 받았다", file=sys.stderr)
        return 1
    img, info = got["img"], got["info"]
    np.savez(args.out, rgb=to_rgb(img.encoding, img.height, img.width, bytes(img.data)),
             K=np.array(info.k, dtype=float).reshape(3, 3),
             stamp_s=img.header.stamp.sec + img.header.stamp.nanosec * 1e-9, frame_id=img.header.frame_id)
    print(f"saved {args.out} {img.width}x{img.height} stamp {img.header.stamp.sec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
