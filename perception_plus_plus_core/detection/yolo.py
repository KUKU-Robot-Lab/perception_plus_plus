from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from ..errors import DependencyUnavailable, ModelLoadError
from .base import Detection


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


class YoloCupDetector:
    def __init__(self, weights: str | Path, class_id: int, confidence: float,
                 model: Any | None = None, pick: str = "confidence") -> None:
        self.weights, self.class_id, self.confidence = str(weights), class_id, confidence
        if pick not in ("confidence", "dark", "bright", "red", "blue"):
            raise ValueError(f"unknown pick strategy: {pick}")
        self.pick = pick
        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise DependencyUnavailable("install perception-plus-plus[yolo]") from error
            try:
                model = YOLO(self.weights)
            except Exception as error:
                raise ModelLoadError(f"cannot load YOLO weights {self.weights}: {error}") from error
        self.model = model

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        results = self.model(rgb)
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None or result.masks is None:
                continue
            classes = _numpy(result.boxes.cls).astype(int)
            confidences = _numpy(result.boxes.conf)
            boxes = _numpy(result.boxes.xyxy)
            masks = _numpy(result.masks.data)
            for index, (class_id, confidence) in enumerate(zip(classes, confidences)):
                if class_id != self.class_id or confidence < self.confidence:
                    continue
                mask = masks[index].astype(bool)
                if mask.shape != rgb.shape[:2]:
                    try:
                        import cv2
                    except ImportError as error:
                        raise DependencyUnavailable("opencv is required to resize YOLO masks") from error
                    mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
                detections.append(Detection(mask, float(confidence), int(class_id),
                                            tuple(float(v) for v in boxes[index])))
        detections.sort(key=lambda item: item.confidence, reverse=True)
        if self.pick != "confidence" and len(detections) > 1:
            if self.pick in ("red", "blue"):
                ch = 0 if self.pick == "red" else 2
                def _dom(det):
                    if not det.mask.any():
                        return -255.0
                    m = rgb[det.mask].astype(float)
                    other = np.delete(m, ch, axis=1).max(axis=1)
                    return float((m[:, ch] - other).mean())
                detections.sort(key=_dom, reverse=True)
            else:
                gray = rgb.mean(axis=2)
                detections.sort(
                    key=lambda det: float(gray[det.mask].mean()) if det.mask.any() else 255.0,
                    reverse=self.pick == "bright")
        return detections

