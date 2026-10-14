#!/usr/bin/env python3
"""기준 camera extrinsics yaml 의 `camera:` 블록만 지정한 목 자세(pan, tilt)의 T_base_cam 으로 바꾼 새 파일을 만든다. 목은 안 움직인다.

    python3 scripts/make_camera_extrinsics_at.py --pan 0 --tilt -30 --out ~/rl_ws/sim2real/config/global_camera_extrinsics_tilt30.yaml

왜: FP++ 스냅샷 노드는 후보를 거를 때(작업 영역, 서 있는 모양, 3D 분리) 이 파일의 정적 camera 블록을 쓴다. 블록은 기준 자세(tilt -20) 값이라
tilt 가 다르면 점군이 기운 것으로 보여 후보가 버려진다. tilt 를 고정해 운용하면 그 tilt 의 파일을 주면 된다(pan 은 수직축 회전이라 기울기에 영향이 작다).
나머지 키(cad_to_body 등)는 그대로 복사한다. 기존 파일은 덮어쓰지 않는다.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_plus_plus_core.head_scan import head_fk  # noqa: E402

DEFAULT_BASE = Path.home() / "rl_ws/sim2real/config/global_camera_extrinsics.yaml"
DEFAULT_HAND_EYE = head_fk.CONFIG_DIR / "head_extrinsics_cad_v6_candidate.yaml"


def render(base_text: str, pan: float, tilt: float, hand_eye: Path) -> str:
    pos, quat = (lambda t: (t[:3, 3], head_fk.quat_wxyz_from_matrix(t[:3, :3])))(head_fk.t_base_cam(pan, tilt, hand_eye))
    if quat[0] < 0:
        quat = -quat
    text, n1 = re.subn(r"(?m)^(  position: )\[.*\]$", rf"\1[{', '.join(f'{v:.9f}' for v in pos)}]", base_text, count=1)
    text, n2 = re.subn(r"(?m)^(  orientation_wxyz: )\[.*\]$", rf"\1[{', '.join(f'{v:.9f}' for v in quat)}]", text, count=1)
    if n1 != 1 or n2 != 1:
        raise ValueError("camera 블록의 position·orientation_wxyz 줄을 못 찾았다")
    header = (f"# ★이 파일은 camera 블록만 목 pan {pan:+g} / tilt {tilt:+g} 의 T_base_cam 으로 바꾼 사본이다(hand-eye {hand_eye.name}).\n"
              f"#   만든 도구: perception_plus_plus scripts/make_camera_extrinsics_at.py. 원본은 config/global_camera_extrinsics.yaml.\n")
    return header + text


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--pan", type=float, default=0.0)
    ap.add_argument("--tilt", type=float, required=True)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--hand-eye", type=Path, default=DEFAULT_HAND_EYE)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        print(f"❌ {args.out} 가 이미 있다 — 덮어쓰지 않는다", file=sys.stderr)
        return 2
    args.out.write_text(render(args.base.read_text(encoding="utf-8"), args.pan, args.tilt, args.hand_eye), encoding="utf-8")
    print("기록:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
