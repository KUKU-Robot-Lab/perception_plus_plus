# robot camera extrinsics (copy, 2026-09-05)

원본(진실원천)은 `rl_ws/sim2real/config/` 이다. FP++ 노드는 카메라 optical 프레임의 자세만 내고,
base 프레임 변환은 sim2real `object_pose_node`/`cup_pose_relay` 가 이 값으로 한다. 여기 사본은
vision PC 에서 같은 값을 참조·검증하기 위한 것.

- `head_extrinsics.yaml` — v2: head_v1 CAD 사전 모델 hand-eye. T_neck_cam = Rot_y(+90.5145°)∘T_cad.
  RGB 렌즈 = head_v1 `camera_link` 의 y+0.0326 개구부(Fusion 라벨 ir_projector_frame). tilt 인코더 영점 +90.51°, pan 0.
  테이블 상면 = base z 0.204 (줄자 0.205).
- `global_camera_extrinsics.yaml` — 홈(pan 0/tilt −20) 정적 T_base_cam 스냅샷 + cad_to_body.
- `head_camera_sim.json` — sim 에 같은 카메라를 붙이는 사양(intrinsics 606.6/605.7, 320.0/240.6).
- `head_extrinsics_pre0905_handeye_v1.yaml` — 09-01 v1(카메라 높이 +21 mm 오류) 보관.

캡처(`scripts/capture_frame.py`)는 depth 를 color 에 정렬하므로 K 는 color 내부행렬이다.
