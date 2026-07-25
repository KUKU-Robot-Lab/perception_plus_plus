#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception_plus_plus_core.assets.upstream import check_upstream_revision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("external/foundationpose_plus_plus"))
    parser.add_argument("--revision", default="58aa715")
    args = parser.parse_args()
    result = check_upstream_revision(args.path, args.revision)
    print(f"{result.status}: {result.detail}")
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
