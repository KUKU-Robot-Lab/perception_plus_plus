"""찾을 수 있는 물체 목록(메시·원점 높이·색) — `locate.ObjectSpec` 으로 읽는다.

값은 sim2real `config/objects.d` 와 `assets/meshes` 에서 2026-10-13 에 확인한 것이다. 물체가 늘면 여기에 한 줄을 더한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import trimesh

from .locate import ObjectSpec

MESH_DIR = Path.home() / "rl_ws/sim2real/assets/meshes"


@dataclass(frozen=True)
class _Entry:
    mesh_file: str
    origin_above_bottom: float  # 원점이 바닥에서 떠 있는 높이(m)
    hue: float  # OpenCV HSV hue


OBJECTS = {
    "source240b_blue": _Entry("source240b.obj", 0.085, 100.0),
    "shaker_c_orange": _Entry("shaker_c.obj", 0.065, 12.0),
}
#: 사람이 쓰는 이름 → 물체 이름
ALIASES = {"blue_cup": "source240b_blue", "orange_cup": "shaker_c_orange", "shaker": "shaker_c_orange"}


def resolve_name(name: str) -> str:
    key = ALIASES.get(name, name)
    if key not in OBJECTS:
        raise ValueError(f"모르는 물체 {name!r}. 가능: {sorted(OBJECTS) + sorted(ALIASES)}")
    return key


def load_spec(name: str, mesh_dir: Path = MESH_DIR) -> ObjectSpec:
    key = resolve_name(name)
    entry = OBJECTS[key]
    mesh = trimesh.load(mesh_dir / entry.mesh_file, force="mesh")
    return ObjectSpec(name=key, vertices=mesh.vertices, faces=mesh.faces, origin_above_bottom=entry.origin_above_bottom, hue=entry.hue)
