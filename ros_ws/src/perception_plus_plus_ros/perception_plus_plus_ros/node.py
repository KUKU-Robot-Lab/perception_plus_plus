from pathlib import Path
import numpy as np

import message_filters
import rclpy
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster

from perception_plus_plus_msgs.msg import TrackingStatus
from perception_plus_plus_core.config import TrackingConfig
from perception_plus_plus_core.detection.yolo import YoloCupDetector
from perception_plus_plus_core.fp_adapter.foundationpose_plus_plus import FoundationPosePlusPlusAdapter
from perception_plus_plus_core.tracking.manager import TrackingManager
from perception_plus_plus_core.types import CameraIntrinsics, FrameBundle, MeshSpec
from perception_plus_plus_core.validation.depth import depth_to_meters
from .conversion import fill_transform


class CupTrackingNode(Node):
    def __init__(self) -> None:
        super().__init__("cup_tracking")
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
            "yolo_confidence": 0.5,
            "detection_pick": "confidence",
            "tracking_config": "config/cup_tracking.yaml",
            "sync_slop_seconds": 0.04,
            "sync_queue_size": 10,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        get = lambda name: self.get_parameter(name).value
        config = TrackingConfig.from_yaml(get("tracking_config"))
        self.manager = TrackingManager(
            FoundationPosePlusPlusAdapter(),
            YoloCupDetector(get("yolo_weights"), get("cup_class_id"),
                            get("yolo_confidence"), pick=get("detection_pick")),
            MeshSpec(get("mesh_path"), get("mesh_scale_to_meters")), config)
        self.bridge = CvBridge()
        self.child_frame_id = get("child_frame_id")
        self.pose_publisher = self.create_publisher(PoseStamped, get("pose_topic"), 10)
        self.status_publisher = self.create_publisher(
            TrackingStatus, get("status_topic"), 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.last_valid_stamp = Time()
        rgb = message_filters.Subscriber(
            self, Image, get("rgb_topic"), qos_profile=qos_profile_sensor_data)
        depth = message_filters.Subscriber(
            self, Image, get("depth_topic"), qos_profile=qos_profile_sensor_data)
        info = message_filters.Subscriber(
            self, CameraInfo, get("camera_info_topic"), qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb, depth, info], int(get("sync_queue_size")),
            float(get("sync_slop_seconds")))
        self.sync.registerCallback(self._callback)

    def _callback(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        if rgb_msg.header.frame_id != info_msg.header.frame_id:
            self.get_logger().warning("RGB and camera_info frame IDs differ")
            return
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
        depth = depth_to_meters(
            np.asarray(self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")),
            depth_msg.encoding)
        k = info_msg.k
        frame = FrameBundle(
            np.asarray(rgb), depth,
            CameraIntrinsics(k[0], k[4], k[2], k[5], info_msg.width, info_msg.height),
            rclpy.time.Time.from_msg(rgb_msg.header.stamp).nanoseconds,
            rgb_msg.header.frame_id)
        output = self.manager.process(frame)
        if output.pose is not None:
            self._publish_pose_and_tf(rgb_msg, output.pose.object_to_camera)
            self.last_valid_stamp = rgb_msg.header.stamp
        self._publish_status(rgb_msg, output)

    def _publish_pose_and_tf(self, image: Image, matrix: np.ndarray) -> None:
        pose = PoseStamped()
        pose.header = image.header
        fill_transform(pose.pose, matrix)
        self.pose_publisher.publish(pose)
        transform = TransformStamped()
        transform.header = image.header
        transform.child_frame_id = self.child_frame_id
        fill_transform(transform.transform, matrix)
        self.tf_broadcaster.sendTransform(transform)

    def _publish_status(self, image: Image, output) -> None:
        status = TrackingStatus()
        status.header = image.header
        status.state = int(output.state)
        status.last_valid_pose_stamp = self.last_valid_stamp
        status.failure_reason = output.reason
        status.failure_detail = ""
        status.consecutive_valid = output.consecutive_valid
        status.consecutive_invalid = output.consecutive_invalid
        status.fatal = output.fatal
        self.status_publisher.publish(status)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CupTrackingNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

