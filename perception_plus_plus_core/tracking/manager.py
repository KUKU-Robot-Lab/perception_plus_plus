from __future__ import annotations

from dataclasses import dataclass

from ..config import TrackingConfig
from ..detection.base import CupDetector
from ..errors import CudaOutOfMemory, DependencyUnavailable, ModelLoadError, classify_exception
from ..fp_adapter.base import FpAdapter
from ..types import FrameBundle, MeshSpec, PoseResult
from ..validation.quality import evaluate_quality
from .state import TrackingState


@dataclass(frozen=True)
class TrackingOutput:
    state: TrackingState
    pose: PoseResult | None
    reason: str
    consecutive_valid: int
    consecutive_invalid: int
    fatal: bool = False


class TrackingManager:
    def __init__(self, adapter: FpAdapter, detector: CupDetector, mesh: MeshSpec,
                 config: TrackingConfig) -> None:
        self.adapter, self.detector, self.mesh, self.config = adapter, detector, mesh, config
        self.state = TrackingState.INITIALIZING
        self._previous: PoseResult | None = None
        self._valid = self._invalid = self._lost_frames = 0
        self._fatal = False
        self._reason = "STARTING"

    def _output(self, pose: PoseResult | None = None) -> TrackingOutput:
        return TrackingOutput(self.state, pose, self._reason, self._valid,
                              self._invalid, self._fatal)

    def _fatal_output(self, error: BaseException) -> TrackingOutput:
        classified = classify_exception(error)
        self._fatal = isinstance(
            classified, (CudaOutOfMemory, DependencyUnavailable, ModelLoadError))
        self.state = TrackingState.LOST
        self._reason = type(classified).__name__.upper()
        return self._output()

    def _initialize(self, frame: FrameBundle) -> TrackingOutput:
        detections = self.detector.detect(frame.rgb)
        if not detections:
            self._reason = "DETECTION_NOT_FOUND"
            return self._output()
        try:
            result = self.adapter.initialize(frame, detections[0].mask, self.mesh)
        except BaseException as error:
            return self._fatal_output(error)
        decision = evaluate_quality(frame, result, None, self.config)
        if not decision.valid:
            self._reason = decision.reason
            return self._output()
        self._previous, self._valid, self._invalid = result, 1, 0
        if self.config.reinitialize_valid_frames <= 1:
            self.state = TrackingState.TRACKING
        else:
            self.state = TrackingState.REINITIALIZING
        self._reason = "OK"
        return self._output(result if self.state is TrackingState.TRACKING else None)

    def process(self, frame: FrameBundle) -> TrackingOutput:
        if self._fatal:
            return self._output()
        if self.state is TrackingState.INITIALIZING:
            return self._initialize(frame)
        if self.state is TrackingState.LOST:
            self._lost_frames += 1
            if self._lost_frames < self.config.recovery_interval_frames:
                return self._output()
            self._lost_frames = 0
            output = self._initialize(frame)
            if output.reason == "DETECTION_NOT_FOUND":
                self.state = TrackingState.LOST
                return self._output()
            return output
        try:
            result = self.adapter.track(frame)
        except BaseException as error:
            classified = classify_exception(error)
            if isinstance(classified, (CudaOutOfMemory, DependencyUnavailable, ModelLoadError)):
                return self._fatal_output(error)
            decision_valid, reason = False, "FP_TRACKING_EXCEPTION"
        else:
            decision = evaluate_quality(frame, result, self._previous, self.config)
            decision_valid, reason = decision.valid, decision.reason
        if decision_valid:
            self._valid += 1
            self._invalid = 0
            self._previous = result
            self._reason = "OK"
            if self.state is TrackingState.REINITIALIZING:
                if self._valid >= self.config.reinitialize_valid_frames:
                    self.state = TrackingState.TRACKING
                else:
                    return self._output()
            return self._output(result)
        self._valid = 0
        self._invalid += 1
        self._reason = reason
        if self.state is TrackingState.REINITIALIZING or self._invalid >= self.config.max_invalid_frames:
            self.adapter.reset()
            self.state = TrackingState.LOST
            self._previous = None
            self._lost_frames = 0
        return self._output()
