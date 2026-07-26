from __future__ import annotations

from pathlib import Path
from typing import Callable
import urllib.request


class DownloadError(RuntimeError):
    pass


def download_http(url: str, destination: Path,
                  opener: Callable = urllib.request.urlopen) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    try:
        with opener(url) as response, partial.open("wb") as stream:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                stream.write(block)
        partial.replace(destination)
        return destination
    except Exception as error:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"download failed for {url}: {error}") from error
