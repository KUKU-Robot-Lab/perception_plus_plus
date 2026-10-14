"""capture_poses.py 가 저장한 폴더에서 ChArUco 보드 검출을 읽는다(보드 A·B 구분 포함). 영상 검출은 느려서(자세당 수 초) 결과를 폴더에 캐시한다."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import board_consistency as bc, head_fk

#: 같은 보드가 둘이다(2026-10-14). 마커 id 가 겹치므로 영상 속 덩어리를 대략적인 base y 로 가른다: 가운데 보드 A, -y 모서리 보드 B.
BOARD_B_MAX_Y = -0.20
ROUGH_Z = 0.200
B_ID_OFFSET = 100


def rough_board_label(t_base_cam: np.ndarray, k: np.ndarray, uv: np.ndarray) -> str:
    """덩어리 중심 픽셀을 z=보드 높이 평면에 얹은 base y 로 A/B 를 가른다(코너 id 구분용이라 대략이면 된다)."""
    obs = bc.PoseObs(t_base_cam, k, np.array([0]), uv.mean(axis=0, keepdims=True))
    return "B" if bc._plane_xy(obs, ROUGH_Z)[0, 1] < BOARD_B_MAX_Y else "A"


def _raw_detections(folder: Path, rec: dict) -> list[tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]]:
    """한 자세의 프레임별 (보드 검출 목록, K). 검출은 느려서 폴더의 board_detections.json 에 캐시한다."""
    cache_path = folder / "board_detections.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    out = []
    for k in range(rec["frames"]):
        key = f"{rec['index']}_{k}"
        data = np.load(folder / f"pose_{rec['index']}_{k}.npz")
        if key not in cache:
            cache[key] = [{"ids": ids.tolist(), "uv": uv.tolist()} for ids, uv in bc.detect_boards(data["rgb"], min_corners=4)]
        out.append(([(np.array(b["ids"]), np.array(b["uv"])) for b in cache[key]], data["K"]))
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return out


def load_detections(folder: Path, min_corners: int, label_yaml: Path) -> list[dict]:
    """자세마다 프레임들의 코너 픽셀을 (보드, id) 별로 평균한다. 코너가 min_corners 미만인 보드는 그 자세에서 뺀다."""
    out = []
    for rec in json.loads((folder / "poses.json").read_text(encoding="utf-8"))["records"]:
        acc: dict[tuple[str, int], list[np.ndarray]] = {}
        k_mat = None
        t_label = head_fk.t_base_cam(*rec["measured"], label_yaml)
        for boards, k_mat in _raw_detections(folder, rec):
            for ids, uv in boards:
                label = rough_board_label(t_label, k_mat, uv)
                for i, point in zip(ids, uv):
                    acc.setdefault((label, int(i)), []).append(point)
        for label in ("A", "B"):
            keys = sorted(key for key in acc if key[0] == label)
            if len(keys) < min_corners:
                if keys:
                    print(f"  자세 {rec['index']} {rec['commanded']} 보드 {label}: 코너 {len(keys)}개 — 제외")
                continue
            out.append({"rec": rec, "k": k_mat, "board": label, "ids": np.array([i for _, i in keys]),
                        "uv": np.array([np.mean(acc[key], axis=0) for key in keys])})
            print(f"  자세 {rec['index']} {rec['commanded']} 실측 {rec['measured']} 보드 {label}: 코너 {len(keys)}개")
    return out


def pose_obs(det: dict, yaml_path: Path, offset_b: bool = True) -> bc.PoseObs:
    ids = det["ids"] + (B_ID_OFFSET if offset_b and det["board"] == "B" else 0)
    return bc.PoseObs(head_fk.t_base_cam(*det["rec"]["measured"], yaml_path), det["k"], ids, det["uv"])


