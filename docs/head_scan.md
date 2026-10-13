# head_scan — 목 스캔으로 알려진 물체의 테이블 좌표 얻기

목표: 카메라 1대를 목(pan/tilt)으로 움직여 테이블 전체를 보고, CAD 가 있는 물체의 base 좌표를 낸다.
잡은 뒤 추적은 하지 않는다(핸드 FK, sim2real PalmAttach).

## 좌표 체인
`base ← 목 FK(인코더 pan/tilt) ← hand-eye T_neck_cam ← 카메라 optical ← FP++ 물체 포즈`

- `head_scan/head_fk.py` — FK. 체인 값은 `config/robot_camera_extrinsics/head_chain.yaml`
  (원본 `urdf/generated/rl/openarm_rh56f1_bi_rl.urdf` 사본), hand-eye 는 `head_extrinsics.yaml`(v2).
- `head_scan/scan_poses.py` — 테이블 격자점 가시성으로 최소 스캔 자세 세트 계산.
  `python3 scripts/plan_head_scan.py --pan MIN MAX --tilt MIN MAX`
- `head_scan/merge_poses.py` — 관측마다 **촬영 순간의 목 각도**로 base 변환 후 이상치 제거·평균.

## 기존 파이프라인과의 차이
`sim2real/object_pose_node` 는 포즈를 **수신한 순간의 목 각도**로 변환한다(`--head-joint-topic` 을 켰을 때).
UDP 패킷(`fpp_pose_tx`)에는 목 각도가 없다. 그래서 스캔은 "정지 확인 → 목 각도 기록 → 촬영 → 관측과
각도를 함께 저장" 순서로 하고 `merge_poses` 에 넘긴다.

## 가정과 한계 (실기 확인 전)
- 스캔 범위 기본값(pan ±75, tilt -40~-10)은 가정이다. 실기 목 제한은 사용자가 의도적으로 둔 값을 따른다.
- 카메라 내부 파라미터는 `head_camera_sim.json`(sim 사양), depth 최소 거리 0.30 m 는 가정값.
- 팔·몸통·물체 가림은 가시성에 반영하지 않는다.
- 기준 자세(pan 0, tilt -20) 밖의 FK 정확도는 실측 전이다. 자세 간 일관성(`pos_spread_m`)으로 검증한다.
- 실기 단계(목을 움직이는 촬영·일관성 측정)는 매번 사용자 허락 후 진행한다.

## 실측 결과 (2026-10-13, 5090 실기, 컵 2개 정지)
같은 컵을 목 자세 3개에서 찍어 base 좌표의 자세 간 흩어짐(평균에서 최대 거리)을 비교했다. 물체는 정지.
목 각도는 `/head/joint_states` 의 **실측** 인코더 값(명령값 아님)을 썼다.

| 물체 | 관측 | (A) 정적 외부 파라미터 | (B) 촬영 시 목 각도 FK |
|---|---|---|---|
| shaker_c_orange | 3 (pan +15 · -15 · 0) | 84.8 mm | 5.2 mm |
| source240b_blue | 2 (pan -15 · 0) | 43.6 mm | 2.6 mm |

- 기준 자세(pan 0.13, tilt -20)에서는 A 와 B 가 1 mm 미만으로 같다(캘리브 자세).
- pan ±15° 에서 A 는 최대 약 8.7 cm 어긋난다. B 는 2.6~5.2 mm 로 일관하다 → 목을 움직일 때 FK 변환이 필요하다.
- 명령 대비 실제 각도 오차: pan +0.10~0.14°, tilt -0.11° (head_home 검증 평균). 좌표 변환에는 실측 각도를 쓴다.
- 한계: 컵 2개 · 자세 3개 · 컵이 측정 중 안 움직였다는 가정. 정답 좌표 없이 자세 간 일관성만 봤다.
  pan 0 / tilt -30 에서는 두 컵 모두 색 후보가 없어 관측이 없다(가시성 조건은 미해결).
- 같은 자세의 head_home 검증 평균(tilt -30.110)과 퍼블리셔 샘플(-29.934)이 0.18° 달랐다 — 샘플링 시점·노이즈 확인 필요.
