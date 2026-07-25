from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class UpstreamCheck:
    status: str
    detail: str


def check_upstream_revision(path: Path, expected: str) -> UpstreamCheck:
    path = Path(path)
    if not (path / ".git").exists():
        return UpstreamCheck("FAIL", f"missing upstream checkout: {path}")
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    if result.returncode:
        return UpstreamCheck("FAIL", result.stderr.strip())
    actual = result.stdout.strip()
    if not actual.startswith(expected):
        return UpstreamCheck("FAIL", f"expected {expected}, found {actual}")
    return UpstreamCheck("PASS", actual)

