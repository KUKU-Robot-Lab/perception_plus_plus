from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from pathlib import PurePosixPath
import shutil

from .manifest import ModelLock, Verification, load_lock, verify_lock, write_lock


def _safe(filename: str) -> bool:
    path = PurePosixPath(filename)
    return bool(filename) and not path.is_absolute() and ".." not in path.parts \
        and str(path) == filename


def export_bundle(model_root: Path, destination: Path) -> Path:
    model_root, destination = Path(model_root), Path(destination)
    result = verify_lock(model_root / "models.lock.json", model_root)
    if result.status != "PASS":
        raise ValueError(result.detail)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"bundle destination is not empty: {destination}")
    lock = load_lock(model_root / "models.lock.json")
    destination.mkdir(parents=True, exist_ok=True)
    for model in lock.models:
        source = model_root / model.filename
        target = destination / "models" / model.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target, follow_symlinks=False)
    payload = {"version": 1, "models": [asdict(model) for model in lock.models]}
    (destination / "bundle.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    verified = verify_bundle(destination)
    if verified.status != "PASS":
        raise ValueError(verified.detail)
    return destination


def _bundle_lock(bundle_root: Path) -> ModelLock:
    raw = json.loads((bundle_root / "bundle.json").read_text())
    if raw.get("version") != 1 or not isinstance(raw.get("models"), list):
        raise ValueError("bundle must have version 1 and a models list")
    temporary = bundle_root / ".bundle.lock.tmp"
    try:
        temporary.write_text(json.dumps(raw, indent=2) + "\n")
        return load_lock(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def verify_bundle(bundle_root: Path) -> Verification:
    bundle_root = Path(bundle_root)
    try:
        lock = _bundle_lock(bundle_root)
        if any(not _safe(model.filename) for model in lock.models):
            raise ValueError("bundle contains unsafe model path")
        declared = {model.filename for model in lock.models}
        files = {
            path.relative_to(bundle_root / "models").as_posix()
            for path in (bundle_root / "models").rglob("*") if path.is_file()
        }
        if files != declared:
            raise ValueError("bundle model files differ from manifest")
        lock_path = bundle_root / ".verify.lock.tmp"
        try:
            write_lock(lock, lock_path)
            return verify_lock(lock_path, bundle_root / "models")
        finally:
            lock_path.unlink(missing_ok=True)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return Verification("FAIL", "model_bundle", str(error))


def import_bundle(bundle_root: Path, model_root: Path, replace: bool = False) -> None:
    bundle_root, model_root = Path(bundle_root), Path(model_root)
    result = verify_bundle(bundle_root)
    if result.status != "PASS":
        raise ValueError(result.detail)
    lock = _bundle_lock(bundle_root)
    for model in lock.models:
        source = bundle_root / "models" / model.filename
        target = model_root / model.filename
        if target.exists() and not replace:
            raise ValueError(f"model already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        shutil.copyfile(source, partial, follow_symlinks=False)
        partial.replace(target)
    write_lock(lock, model_root / "models.lock.json")
    result = verify_lock(model_root / "models.lock.json", model_root)
    if result.status != "PASS":
        (model_root / "models.lock.json").unlink(missing_ok=True)
        raise ValueError(result.detail)
