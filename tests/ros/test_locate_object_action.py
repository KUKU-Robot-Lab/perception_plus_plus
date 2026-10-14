"""LocateObject 액션 서버를 가짜 목·카메라로 돌린다 — ROS 가 있고 msgs 가 빌드돼 있을 때만(없으면 건너뜀). 로봇은 움직이지 않는다."""
import sys
import threading
from pathlib import Path

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("perception_plus_plus_msgs.action")

from rclpy.action import ActionClient  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402

from perception_plus_plus_core.head_scan import fakes, objects  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ros_ws/src/perception_plus_plus_ros"))
from perception_plus_plus_msgs.action import LocateObject  # noqa: E402
from perception_plus_plus_ros.locate_object_node import LocateObjectServer  # noqa: E402

TRUTH = (0.30, -0.15)


@pytest.fixture
def rig():
    if not rclpy.ok():
        rclpy.init()
    head = fakes.FakeHead()
    spec = objects.load_spec("blue_cup")
    # 실제 색은 hue 100 부근의 파랑(OpenCV). 같은 hue 가 나오는 RGB 를 쓴다.
    camera = fakes.FakeCamera(head, TRUTH, spec, (0, 170, 255))
    server = LocateObjectServer(head=head, camera=camera)
    client_node = rclpy.create_node("test_client")
    executor = MultiThreadedExecutor()
    executor.add_node(server)
    executor.add_node(client_node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    yield head, ActionClient(client_node, LocateObject, "locate_object"), client_node
    executor.shutdown()
    server.destroy_node()
    client_node.destroy_node()


def _send(client, **fields):
    assert client.wait_for_server(timeout_sec=5.0)
    goal = LocateObject.Goal(**fields)
    stages = []
    send = client.send_goal_async(goal, feedback_callback=lambda fb: stages.append(fb.feedback.stage))
    done = threading.Event()
    box = {}
    send.add_done_callback(lambda f: (box.__setitem__("handle", f.result()), done.set()))
    assert done.wait(10.0)
    handle = box["handle"]
    if not handle.accepted:
        return None, stages
    res_done = threading.Event()
    res = {}
    handle.get_result_async().add_done_callback(lambda f: (res.__setitem__("r", f.result().result), res_done.set()))
    assert res_done.wait(120.0)
    return res["r"], stages


def test_action_returns_pose_stamped_in_base_link_and_goes_home(rig):
    head, client, _ = rig
    result, stages = _send(client, object_name="blue_cup", side="right")
    assert result.found and result.home_ok
    assert result.pose.header.frame_id == "base_link"
    assert (result.pose.pose.position.x, result.pose.pose.position.y) == pytest.approx(TRUTH, abs=3e-3)
    assert result.n_used >= 1
    assert head.homed == 1
    assert stages[-1] == "returning"


def test_action_rejects_unknown_object(rig):
    _, client, _ = rig
    result, _ = _send(client, object_name="nonexistent_thing", side="right")
    assert result is None
