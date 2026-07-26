from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath


@dataclass(frozen=True)
class ModelEntry:
    name: str
    filename: str
    required: bool
    source: str
    license_url: str
    sha256: str | None = None
    provider: str = "http"
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


@dataclass(frozen=True)
class LockedModel:
    name: str
    filename: str
    size: int
    sha256: str
    source: str
    license_url: str


@dataclass(frozen=True)
class ModelLock:
    version: int
    models: tuple[LockedModel, ...]


def _validate_relative(filename: str) -> None:
    path = PurePosixPath(filename)
    if not filename or path.is_absolute() or ".." in path.parts or str(path) != filename:
        raise ValueError(f"model filename must be a normalized relative path: {filename}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> ModelManifest:
    raw = json.loads(Path(path).read_text())
    version = raw.get("version")
    if version not in {1, 2} or not isinstance(raw.get("models"), list):
        raise ValueError("model manifest must have version 1 or 2 and a models list")
    entries = []
    names: set[str] = set()
    filenames: set[str] = set()
    for item in raw["models"]:
        digest = item.get("sha256")
        if version == 1 and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in digest)
        ):
            raise ValueError(f"{item.get('name', 'model')} has invalid sha256")
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in digest)
        ):
            raise ValueError(f"{item.get('name', 'model')} has invalid sha256")
        entry = ModelEntry(**item)
        _validate_relative(entry.filename)
        if entry.name in names:
            raise ValueError(f"duplicate model name: {entry.name}")
        if entry.filename in filenames:
            raise ValueError(f"duplicate model filename: {entry.filename}")
        names.add(entry.name)
        filenames.add(entry.filename)
        entries.append(entry)
    return ModelManifest(version=version, models=tuple(entries))


def verify_file(entry: ModelEntry, root: Path) -> Verification:
    path = Path(root) / entry.filename
    return verify_path(entry, path)


def verify_path(entry: ModelEntry, path: Path) -> Verification:
    path = Path(path)
    if not path.is_file():
        status = "FAIL" if entry.required else "SKIP"
        return Verification(status, entry.name, f"missing file: {path}")
    if entry.sha256 is None:
        return Verification("FAIL", entry.name, f"no locked sha256 for {path}")
    digest = _sha256(path)
    if digest.lower() != entry.sha256.lower():
        return Verification("FAIL", entry.name, f"sha256 mismatch for {path}")
    return Verification("PASS", entry.name, f"verified {path}")


def create_lock(manifest: ModelManifest, root: Path) -> ModelLock:
    locked = []
    for entry in manifest.models:
        path = Path(root) / entry.filename
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing regular model file: {path}")
        locked.append(LockedModel(
            name=entry.name,
            filename=entry.filename,
            size=path.stat().st_size,
            sha256=_sha256(path),
            source=entry.source,
            license_url=entry.license_url,
        ))
    return ModelLock(version=1, models=tuple(locked))


def write_lock(lock: ModelLock, path: Path) -> None:
    payload = {
        "version": lock.version,
        "models": [
            {
                "name": model.name,
                "filename": model.filename,
                "size": model.size,
                "sha256": model.sha256,
                "source": model.source,
                "license_url": model.license_url,
            }
            for model in lock.models
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_lock(path: Path) -> ModelLock:
    raw = json.loads(Path(path).read_text())
    if raw.get("version") != 1 or not isinstance(raw.get("models"), list):
        raise ValueError("model lock must have version 1 and a models list")
    models = []
    names: set[str] = set()
    filenames: set[str] = set()
    for item in raw["models"]:
        model = LockedModel(**item)
        _validate_relative(model.filename)
        if model.name in names or model.filename in filenames:
            raise ValueError("model lock contains duplicate name or filename")
        if model.size < 0:
            raise ValueError(f"{model.name} has invalid size")
        if len(model.sha256) != 64 or any(c not in "0123456789abcdef" for c in model.sha256):
            raise ValueError(f"{model.name} has invalid sha256")
        names.add(model.name)
        filenames.add(model.filename)
        models.append(model)
    return ModelLock(version=1, models=tuple(models))


def verify_lock(lock_path: Path, root: Path) -> Verification:
    try:
        lock = load_lock(lock_path)
        for model in lock.models:
            path = Path(root) / model.filename
            if not path.is_file() or path.is_symlink():
                return Verification("FAIL", "model_lock", f"missing regular file: {path}")
            if path.stat().st_size != model.size:
                return Verification("FAIL", "model_lock", f"size mismatch for {path}")
            if _sha256(path) != model.sha256:
                return Verification("FAIL", "model_lock", f"sha256 mismatch for {path}")
        return Verification("PASS", "model_lock", f"verified {len(lock.models)} models")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return Verification("FAIL", "model_lock", str(error))
