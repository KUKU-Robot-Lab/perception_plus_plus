#!/usr/bin/env python3
from pathlib import Path
import argparse
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception_plus_plus_core.assets.manifest import verify_lock
from perception_plus_plus_core.assets.upstream import check_upstream_revision
from perception_plus_plus_core.validation.readiness import CheckResult, write_report


ROOT = Path(__file__).resolve().parents[1]


def command_check(name, command, required=True):
    if shutil.which(command[0]) is None:
        return CheckResult(name, "SKIP", required, f"{command[0]} is unavailable")
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    detail = (result.stdout + result.stderr).strip()[-2000:]
    return CheckResult(name, "PASS" if result.returncode == 0 else "FAIL",
                       required, detail)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-npz", type=Path)
    parser.add_argument("--mesh", type=Path,
                        default=ROOT / "assets/meshes/cup.obj")
    args = parser.parse_args()
    checks = []
    upstream = check_upstream_revision(ROOT / "external/foundationpose_plus_plus", "58aa715")
    checks.append(CheckResult("upstream_revision", upstream.status, True, upstream.detail))
    result = verify_lock(ROOT / "models/models.lock.json", ROOT / "models")
    checks.append(CheckResult("model_lock", result.status, True, result.detail))
    checks.append(command_check("cpu_tests", ["bash", "scripts/run_tests.sh"]))
    checks.append(command_check("ros_colcon", [
        "colcon", "build", "--base-paths", "ros_ws/src",
        "--build-base", "/tmp/ppp_readiness_build",
        "--install-base", "/tmp/ppp_readiness_install"], required=True))
    checks.append(command_check("cuda_torch", [
        "python3", "-c",
        "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
    ]))
    if not args.mesh.is_file():
        checks.append(CheckResult("cup_mesh", "SKIP", True, f"missing mesh: {args.mesh}"))
    else:
        checks.append(CheckResult("cup_mesh", "PASS", True, str(args.mesh)))
    if args.smoke_npz is None:
        checks.append(CheckResult("fpplusplus_smoke", "SKIP", True,
                                  "pass --smoke-npz with an initialization frame"))
    else:
        checks.append(command_check("fpplusplus_smoke", [
            "python3", "scripts/fpplusplus_smoke.py", "--npz", str(args.smoke_npz),
            "--mesh", str(args.mesh)]))
    checks.append(command_check(
        "docker_daemon", ["docker", "info", "--format", "{{.ServerVersion}}"],
        required=False))
    status = write_report(ROOT / "reports/pre_camera_readiness.json", checks)
    for check in checks:
        print(f"{check.status:4} {check.name}: {check.detail.splitlines()[-1] if check.detail else ''}")
    print(status)
    return 0 if status == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
