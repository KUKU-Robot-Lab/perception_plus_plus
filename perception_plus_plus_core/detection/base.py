from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class Detection:
    mask: np.ndarray
    confidence: float = 1.0
    class_id: int = 0
    xyxy: tuple[float, float, float, float] | None = None


class CupDetector(Protocol):
    def detect(self, rgb: np.ndarray) -> list[Detection]: ...

