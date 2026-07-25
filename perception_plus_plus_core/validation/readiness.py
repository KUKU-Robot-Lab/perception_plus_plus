from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    required: bool
    detail: str

    def __post_init__(self):
        if self.status not in {"PASS", "FAIL", "SKIP"}:
            raise ValueError("status must be PASS, FAIL, or SKIP")


def readiness_status(checks: list[CheckResult]) -> str:
    return "READY" if all(c.status == "PASS" or not c.required for c in checks) else "NOT_READY"


def write_report(path: Path, checks: list[CheckResult]) -> str:
    status = readiness_status(checks)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": status, "checks": [asdict(c) for c in checks]},
                               indent=2) + "\n")
    return status

