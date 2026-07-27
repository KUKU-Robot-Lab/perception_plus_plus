"""Detection-anchored FP++ cup 추종 ROS 노드 (신규).

live_fp_demo.py의 검증된 루프를 ROS로 승격: 매 프레임 YOLO cup bbox(bbox-only) +
FP++ track, 추정 pose가 bbox+margin 밖으로 patience 프레임 벗어나면 in-place
재-앵커(기존 engine 재사용, Hydra 재init 회피). FP는 프레임당 C/CUDA 프린트가
있어 fd 1을 /dev/null로 억제(상태는 stderr).

기존 CupTrackingNode/TrackingManager/어댑터는 수정하지 않는다.
"""
from __future__ import annotations

import os
import sys

import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster

from perception_plus_plus_core.tracking.anchor_geometry import anchor_valid, bbox_depth_mask
from perception_plus_plus_core.types import CameraIntrinsics, FrameBundle, MeshSpec
from perception_plus_plus_core.validation.depth import depth_to_meters
from perception_plus_plus_msgs.msg import TrackingStatus
from .conversion import fill_transform


def cup_bbox(model, rgb, conf, class_id):
    """bbox-only YOLO에서 최고 conf cup의 (conf, xyxy). 없으면 None."""
    best = None
    for res in model(rgb, verbose=False):
        if res.boxes is None:
            continue
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cfd = res.boxes.conf.cpu().numpy()
        box = res.boxes.xyxy.cpu().numpy()
        for i, (c, p) in enumerate(zip(cls, cfd)):
            if c == class_id and p >= conf and (best is None or p > best[0]):
                best = (float(p), box[i])
    return best


def _as_mat(pose):
    p = pose.detach().cpu().numpy() if hasattr(pose, "detach") else np.asarray(pose)
    return p.reshape(4, 4)


def reanchor(adapter, rgb, depth, K, mask_bool, mask_u8):
    """기존 engine에서 pose 재추정 + Cutie 마스크 재시드 + Kalman 재초기화."""
    eng = adapter.engine
    pose = eng.estimator.register(K=K, rgb=rgb, depth=depth,
                                  ob_mask=mask_u8 * 255, iteration=eng.est_iter)
    eng.cutie.initialize(rgb, {"mask": mask_u8})
    eng.mask = mask_bool
    eng.kf_mean, eng.kf_covariance = eng.kalman.initiate(eng.get_pose_array(pose))
    return _as_mat(pose), mask_bool


def _silence_stdout():
    sys.stdout.flush()
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)


class AnchoredCupNode(Node):
    def __init__(self) -> None:
        super().__init__("anchored_cup_tracking")
        defaults = {
            "rgb_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera/color/camera_info",
            "pose_topic": "/perception_plus_plus/cup/pose",
            "status_topic": "/perception_plus_plus/cup/tracking_status",
            "child_frame_id": "cup",
            "mesh_path": "assets/meshes/cup.obj",
            "mesh_scale_to_meters": 1.0,
            "yolo_weights": "models/yolo/yolo11n.pt",
            "cup_class_id": 41,
            "yolo_confidence": 0.25,
            "margin": 0.35,
            "z_min": 0.15,
            "z_max": 1.6,
            "patience": 3,
            "sync_slop_seconds": 0.04,
            "sync_queue_size": 10,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        g = lambda n: self.get_parameter(n).value

        from ultralytics import YOLO
        self.yolo = YOLO(g("yolo_weights"))
        self.class_id = int(g("cup_class_id"))
        self.conf = float(g("yolo_confidence"))
        self.margin = float(g("margin"))
        self.z_min, self.z_max = float(g("z_min")), float(g("z_max"))
        self.patience = int(g("patience"))
        from perception_plus_plus_core.fp_adapter.foundationpose_plus_plus import (
            FoundationPosePlusPlusAdapter,
        )
        self.adapter = FoundationPosePlusPlusAdapter()
        self.mesh = MeshSpec(g("mesh_path"), float(g("mesh_scale_to_meters")))
        self.bridge = CvBridge()
        self.child_frame_id = g("child_frame_id")
        self.pose_pub = self.create_publisher(PoseStamped, g("pose_topic"), 10)
        self.status_pub = self.create_publisher(TrackingStatus, g("status_topic"), 10)
        self.tf = TransformBroadcaster(self)

        self._started = False
        self._bad = 0
        self._valid = self._invalid = 0

        rgb = message_filters.Subscriber(self, Image, g("rgb_topic"),
                                         qos_profile=qos_profile_sensor_data)
        depth = message_filters.Subscriber(self, Image, g("depth_topic"),
                                           qos_profile=qos_profile_sensor_data)
        info = message_filters.Subscriber(self, CameraInfo, g("camera_info_topic"),
                                          qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb, depth, info], int(g("sync_queue_size")), float(g("sync_slop_seconds")))
        self.sync.registerCallback(self._callback)
        self.get_logger().info("anchored_cup_tracking 준비 (FP 로그 억제 시작)")
        _silence_stdout()

    def _callback(self, rgb_msg, depth_msg, info_msg):
        rgb = np.asarray(self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8"))
        depth = depth_to_meters(
            np.asarray(self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")),
            depth_msg.encoding)
        k = info_msg.k
        K = np.array([[k[0], 0, k[2]], [0, k[4], k[5]], [0, 0, 1]])
        intr = CameraIntrinsics(k[0], k[4], k[2], k[5], info_msg.width, info_msg.height)
        frame = FrameBundle(rgb, depth, intr,
                            rclpy.time.Time.from_msg(rgb_msg.header.stamp).nanoseconds,
                            rgb_msg.header.frame_id)

        det = cup_bbox(self.yolo, rgb, self.conf, self.class_id)
        pose = None
        if not self._started:
            if det is not None:
                mask = bbox_depth_mask(depth, det[1])
                try:
                    r = self.adapter.initialize(frame, mask, self.mesh)
                    pose = np.asarray(r.object_to_camera).reshape(4, 4)
                    self._started, self._bad = True, 0
                    self._log_stderr("initialized")
                except BaseException as e:                       # noqa: BLE001
                    self._log_stderr(f"init failed: {e}")
        else:
            r = self.adapter.track(frame)
            pose = np.asarray(r.object_to_camera).reshape(4, 4)
            det_xyxy = tuple(det[1]) if det is not None else None
            if anchor_valid(pose, K, det_xyxy, self.z_min, self.z_max, self.margin):
                self._bad, self._valid, self._invalid = 0, self._valid + 1, 0
            else:
                self._bad += 1
                self._valid, self._invalid = 0, self._invalid + 1
                if det is not None and self._bad >= self.patience:
                    mask = bbox_depth_mask(depth, det[1])
                    try:
                        pose, _ = reanchor(self.adapter, rgb, depth, K,
                                           mask, mask.astype(np.uint8))
                        self._bad = 0
                        self._log_stderr("re-anchored")
                    except BaseException as e:                   # noqa: BLE001
                        self._log_stderr(f"reanchor failed: {e}")

        if pose is not None:
            self._publish(rgb_msg, pose)
        self._publish_status(rgb_msg)

    def _publish(self, image, matrix):
        pose = PoseStamped()
        pose.header = image.header
        fill_transform(pose.pose, matrix)
        self.pose_pub.publish(pose)
        tfm = TransformStamped()
        tfm.header = image.header
        tfm.child_frame_id = self.child_frame_id
        fill_transform(tfm.transform, matrix)
        self.tf.sendTransform(tfm)

    def _publish_status(self, image):
        s = TrackingStatus()
        s.header = image.header
        s.state = 1 if self._started else 0
        s.failure_reason = "" if self._bad == 0 else "OFF_CUP_OR_Z"
        s.failure_detail = ""
        s.consecutive_valid = int(self._valid)
        s.consecutive_invalid = int(self._invalid)
        s.fatal = False
        self.status_pub.publish(s)

    @staticmethod
    def _log_stderr(msg):
        print(f"[anchored_cup] {msg}", file=sys.stderr, flush=True)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AnchoredCupNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
