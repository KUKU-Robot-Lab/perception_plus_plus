#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception_plus_plus_core.assets.download import download_http
from perception_plus_plus_core.assets.manifest import (
    create_lock,
    load_manifest,
    verify_lock,
    verify_path,
    write_lock,
)


def _install(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    shutil.copyfile(source, partial, follow_symlinks=False)
    partial.replace(target)


def _find_drive_file(staging: Path, filename: str) -> Path:
    expected = Path(filename)
    matches = [
        path for path in staging.rglob(expected.name)
        if expected.parent.name in path.parts and path.is_file() and not path.is_symlink()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one Drive match for {filename}, found {len(matches)}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Install and lock official FP++ models")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "assets/model_manifests/models.json")
    parser.add_argument("--model-root", type=Path, default=ROOT / "models")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    lock_path = args.model_root / "models.lock.json"
    if args.verify_only:
        result = verify_lock(lock_path, args.model_root)
        print(f"{result.status}: {result.detail}")
        return 0 if result.status == "PASS" else 1

    manifest = load_manifest(args.manifest)
    existing = [args.model_root / item.filename for item in manifest.models
                if (args.model_root / item.filename).exists()]
    if existing and not args.replace:
        result = verify_lock(lock_path, args.model_root)
        if result.status == "PASS":
            print(result.detail)
            return 0
        raise SystemExit(
            "unlocked or invalid model files exist; verify their origin and rerun with --replace")

    with tempfile.TemporaryDirectory(prefix="ppp-models-", dir=args.model_root.parent) as raw:
        staging = Path(raw)
        drive_sources = {item.source for item in manifest.models
                         if item.provider == "gdrive_folder"}
        for source in drive_sources:
            command = [sys.executable, "-m", "gdown", "--folder", source,
                       "--output", str(staging / "gdrive")]
            subprocess.run(command, check=True)
        for item in manifest.models:
            if item.provider == "http":
                source = staging / item.name
                download_http(item.source, source)
            elif item.provider == "gdrive_folder":
                source = _find_drive_file(staging / "gdrive", item.filename)
            else:
                raise RuntimeError(f"unsupported model provider: {item.provider}")
            verification = verify_path(item, source)
            if verification.status != "PASS":
                raise RuntimeError(verification.detail)
            _install(source, args.model_root / item.filename)
            print(f"INSTALLED: {item.name}: {item.filename}")
    lock = create_lock(manifest, args.model_root)
    write_lock(lock, lock_path)
    result = verify_lock(lock_path, args.model_root)
    print(f"{result.status}: {result.detail}")
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
