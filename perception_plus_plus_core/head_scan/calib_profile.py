"""보정 프로필 읽기·쓰기 — 구현은 sim2real `scripts/calib_profile.py` 하나다(노드와 같은 코드·같은 스키마).

두 저장소가 같은 파일 형식을 따로 구현하면 갈라지므로 여기서는 그 모듈을 경로로 불러 되돌려 준다.
sim2real 이 없는 호스트(vision-3090 등)에서는 이 모듈을 쓰지 않는다 — 프로필은 로봇 PC 에서만 읽는다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path.home() / "rl_ws/sim2real/scripts/calib_profile.py"


def _load():
    if "sim2real_calib_profile" in sys.modules:
        return sys.modules["sim2real_calib_profile"]
    if not _PATH.is_file():
        raise ImportError(f"{_PATH} 가 없다 — 보정 프로필은 sim2real 이 있는 로봇 PC 에서만 쓴다")
    spec = importlib.util.spec_from_file_location("sim2real_calib_profile", _PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sim2real_calib_profile"] = module
    spec.loader.exec_module(module)
    return module


_m = _load()
AngleEntry, CalibProfile, UncalibratedAngle, FrameAnchor = _m.AngleEntry, _m.CalibProfile, _m.UncalibratedAngle, _m.FrameAnchor
PROFILE_DIR, load_profile, save_profile, bias_for, lookup, anchor_position = (_m.PROFILE_DIR, _m.load_profile, _m.save_profile, _m.bias_for,
                                                                              _m.lookup, _m.anchor_position)


def write_extrinsics_yaml(profile, path) -> Path:
    """프로필의 hand-eye 를 `head_extrinsics.yaml` 형식 파일로 쓴다(`head_fk.t_base_cam(..., extrinsics_yaml)` 에 넘기려고)."""
    import yaml

    path = Path(path)
    path.write_text(yaml.safe_dump({"neck_to_camera": {"matrix": [[float(v) for v in row] for row in profile.neck_to_camera]}}), encoding="utf-8")
    return path
