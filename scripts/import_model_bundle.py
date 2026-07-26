#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception_plus_plus_core.assets.bundle import import_bundle, verify_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a verified external-drive model bundle")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=ROOT / "models")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    result = verify_bundle(args.bundle)
    print(f"{result.status}: {result.detail}")
    if result.status != "PASS":
        return 1
    import_bundle(args.bundle, args.model_root, replace=args.replace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
