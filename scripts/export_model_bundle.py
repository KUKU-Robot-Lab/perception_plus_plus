#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception_plus_plus_core.assets.bundle import export_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Export verified models to an external drive")
    parser.add_argument("--model-root", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export_bundle(args.model_root, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
