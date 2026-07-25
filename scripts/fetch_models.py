#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception_plus_plus_core.assets.manifest import load_manifest, verify_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and verify external model files")
    parser.add_argument("--manifest", type=Path, default=Path("assets/model_manifests/models.json"))
    parser.add_argument("--root", type=Path, default=Path("models"))
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    failed = False
    for entry in manifest.models:
        target = args.root / entry.filename
        if not target.exists() and args.source_root:
            source = args.source_root / entry.filename
            if source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        if not target.exists() and entry.url:
            target.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(entry.url, target)
        result = verify_file(entry, args.root)
        print(f"{result.status}: {entry.name}: {result.detail}")
        failed |= result.status == "FAIL"
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
