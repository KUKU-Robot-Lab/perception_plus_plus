"""LocateObject 액션 서버 — 목을 돌려 테이블 위 물체를 찾고 base_link 좌표를 돌려준다.

계산·자세 계획·복귀는 `perception_plus_plus_core.head_scan.locate` 가 한다. 이 노드는 ROS 연결(액션, 피드백, 취소)만 맡는다.
`/objects/<name>/pose` 는 발행하지 않는다(object_pose_node 가 낸다). 한 번에 하나의 목표만 받고, 목을 명령하는 다른 프로세스가
있으면 거절한다. 목 이동은 로봇을 움직이므로 이 노드를 띄우는 일은 사용자 허락 뒤에 한다.
"""
from __future__ import annotations

import threading
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from perception_plus_plus_core.head_scan import adapters, calib_profile, locate, objects
from perception_plus_plus_msgs.action import LocateObject

BASE_FRAME = "base_link"
SIDES = ("", "right", "left")


def _stamp_fields(stamp_s: float) -> tuple[int, int]:
    sec = int(stamp_s)
    return sec, int(round((stamp_s - sec) * 1e9))


class LocateObjectServer(Node):
    def __init__(self, head: locate.HeadPort | None = None, camera: locate.CameraPort | None = None) -> None:
        super().__init__("locate_object_server")
        self.declare_parameter("sim2real", str(adapters.SIM2REAL))
        self.declare_parameter("vision_host", "vision-3090")
        self.declare_parameter("action_name", "locate_object")
        self.declare_parameter("calib_profile", "")  # 보정 프로필 이름·경로. 주면 hand-eye 와 각도별 보정, 계획도 프로필 각도로
        self.declare_parameter("fixed_tilt_deg", -999.0)  # -999 이면 고정 안 함. 값을 주면 그 tilt 로 pan 만 돌린다
        self.declare_parameter("extrinsics", "")  # 비우면 저장소 기본 hand-eye. 후보 yaml 경로를 주면 그 값으로 계산한다
        sim2real = Path(self.get_parameter("sim2real").value)
        self._head = head or adapters.Sim2RealHead(sim2real)
        self._camera = camera or adapters.SshCamera(self.get_parameter("vision_host").value)
        extrinsics = str(self.get_parameter("extrinsics").value)
        self._extrinsics = {"extrinsics_yaml": Path(extrinsics)} if extrinsics else {}
        tilt = float(self.get_parameter("fixed_tilt_deg").value)
        self._fixed_tilt = None if tilt == -999.0 else tilt
        name = str(self.get_parameter("calib_profile").value)
        self._profile = calib_profile.load_profile(name) if name else None
        self._lock = threading.Lock()
        self._server = ActionServer(self, LocateObject, self.get_parameter("action_name").value, execute_callback=self._execute,
                                    goal_callback=self._on_goal, cancel_callback=lambda _: CancelResponse.ACCEPT,
                                    callback_group=ReentrantCallbackGroup())
        self.get_logger().info("locate_object 준비됨")

    def _on_goal(self, goal) -> GoalResponse:
        try:
            objects.resolve_name(goal.object_name)
        except ValueError as exc:
            self.get_logger().warning(f"거절: {exc}")
            return GoalResponse.REJECT
        if goal.side not in SIDES:
            self.get_logger().warning(f"거절: side {goal.side!r}")
            return GoalResponse.REJECT
        if len(goal.prior_xy) not in (0, 2):
            self.get_logger().warning("거절: prior_xy 는 비우거나 [x, y]")
            return GoalResponse.REJECT
        busy = adapters.busy_head_commanders()
        if busy:
            self.get_logger().warning(f"거절: 목을 명령하는 다른 프로세스 {busy}")
            return GoalResponse.REJECT
        if self._lock.locked():
            self.get_logger().warning("거절: 이미 처리 중")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute(self, goal_handle):
        goal = goal_handle.request
        result = LocateObject.Result()
        with self._lock:
            spec = objects.load_spec(goal.object_name)
            request = locate.LocateRequest(
                object_name=spec.name, side=goal.side or None, prior_xy=tuple(goal.prior_xy) if goal.prior_xy else None,
                max_poses=goal.max_poses or locate.DEFAULT_MAX_POSES,
                **({"tilt_deg": self._fixed_tilt, "fixed_tilt": True} if self._fixed_tilt is not None else {}))

            def on_stage(stage: str, pan: float, tilt: float) -> None:
                feedback = LocateObject.Feedback()
                feedback.stage, feedback.pan_deg, feedback.tilt_deg = stage, float(pan), float(tilt)
                goal_handle.publish_feedback(feedback)

            outcome = locate.locate_object(request, spec, self._head, self._camera, on_stage=on_stage,
                                           should_stop=lambda: goal_handle.is_cancel_requested, profile=self._profile, **self._extrinsics)
        self._fill(result, outcome)
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.succeed()
        return result

    @staticmethod
    def _fill(result, outcome: locate.LocateResult) -> None:
        result.found, result.home_ok, result.reason = outcome.found, outcome.home_ok, outcome.reason
        result.n_used = int(outcome.n_used)
        result.spread_m, result.mean_iou = (float(outcome.spread_m) if outcome.found else 0.0), float(outcome.mean_iou)
        if not outcome.found:
            return
        shot = next((s for s in reversed(outcome.shots) if s.n_blobs), outcome.shots[-1])
        pose = PoseStamped()
        pose.header.frame_id = BASE_FRAME
        pose.header.stamp.sec, pose.header.stamp.nanosec = _stamp_fields(shot.stamp_s)
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = outcome.x, outcome.y, outcome.z
        pose.pose.orientation.w = 1.0
        result.pose = pose
        result.shot_pan_deg, result.shot_tilt_deg = shot.pan_deg, shot.tilt_deg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LocateObjectServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
