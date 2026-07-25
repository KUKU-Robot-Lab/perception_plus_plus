from collections import deque
import numpy as np

from ..detection.base import Detection


class FakeDetector:
    def __init__(self, masks):
        self.masks = deque(masks)

    def detect(self, rgb):
        if not self.masks:
            return []
        mask = self.masks.popleft()
        return [] if mask is None else [Detection(mask)]


class FakeFpAdapter:
    def __init__(self, outputs):
        self.outputs = deque(outputs)
        self.reset_count = 0

    def _next(self):
        value = self.outputs.popleft()
        if isinstance(value, BaseException):
            raise value
        return value

    def initialize(self, frame, mask, mesh):
        return self._next()

    def track(self, frame):
        return self._next()

    def reset(self):
        self.reset_count += 1

