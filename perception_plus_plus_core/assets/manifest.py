from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class ModelEntry:
    name: str
    filename: str
    sha256: str
    required: bool
    source: str
    license_url: str
    url: str | None = None


@dataclass(frozen=True)
class ModelManifest:
    version: int
    models: tuple[ModelEntry, ...]


@dataclass(frozen=True)
class Verification:
    status: str
    name: str
    detail: str


def load_manifest(path: Path) -> ModelManifest:
    raw = json.loads(Path(path).read_text())
    if raw.get("version") != 1 or not isinstance(raw.get("models"), list):
        raise ValueError("model manifest must have version 1 and a models list")
    entries = []
    names: set[str] = set()
    for item in raw["models"]:
        digest = item.get("sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise ValueError(f"{item.get('name', 'model')} has invalid sha256")
        entry = ModelEntry(**item)
        if entry.name in names:
            raise ValueError(f"duplicate model name: {entry.name}")
        names.add(entry.name)
        entries.append(entry)
    return ModelManifest(version=1, models=tuple(entries))


def verify_file(entry: ModelEntry, root: Path) -> Verification:
    path = Path(root) / entry.filename
    if not path.is_file():
        status = "FAIL" if entry.required else "SKIP"
        return Verification(status, entry.name, f"missing file: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest.lower() != entry.sha256.lower():
        return Verification("FAIL", entry.name, f"sha256 mismatch for {path}")
    return Verification("PASS", entry.name, f"verified {path}")
