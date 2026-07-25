# FoundationPose++ Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the complete Humble/Jazzy cup-tracking framework and automated pre-camera readiness checks.

**Architecture:** A ROS-independent Python core owns typed frame data, quality decisions, FP++/YOLO protocols, and the tracking state machine. Production adapters lazily load upstream GPU code; a thin ROS 2 node synchronizes inputs and publishes only manager-approved pose/TF/status outputs. Separate containers share source while pinning distribution-specific dependencies.

**Tech Stack:** Python 3.10+, NumPy, pytest, ROS 2 Humble/Jazzy, rclpy, message_filters, tf2_ros, Ultralytics YOLO, FoundationPose++, Docker/NVIDIA Container Toolkit.

## Global Constraints

- Ubuntu 22.04/ROS 2 Humble is the reference runtime.
- Ubuntu 24.04/ROS 2 Jazzy uses identical core and ROS interfaces.
- NVIDIA RTX 4070 Laptop GPU 8GB is the target GPU.
- Only RGB-aligned depth is accepted.
- One cup class and one CAD mesh are supported.
- Qwen2-VL, SAM-HQ, multi-object tracking, and Isaac ROS runtime coupling are excluded.
- Invalid pose and TF data must never be published.
- Upstream is a submodule pinned to commit `58aa715`.
- No Git commits are created; the user will commit separately.

---

### Task 1: Repository, Packaging, and Upstream Contract

**Files:**
- Create: `.gitignore`, `.gitmodules`, `pyproject.toml`, `README.md`
- Create: `external/foundationpose_plus_plus` submodule
- Create: `assets/model_manifests/models.json`
- Create: `scripts/fetch_models.py`, `scripts/check_upstream.py`
- Test: `tests/test_model_manifest.py`, `tests/test_upstream.py`

**Interfaces:**
- Produces: `load_manifest(path: Path) -> ModelManifest`
- Produces: `verify_file(entry: ModelEntry, root: Path) -> Verification`
- Produces: a pinned and inspectable upstream checkout

- [ ] Write tests that reject malformed digests, report missing optional/required models distinctly, accept a matching digest, and reject an unexpected submodule revision.
- [ ] Run `python -m pytest tests/test_model_manifest.py tests/test_upstream.py -q`; verify failure because package modules do not exist.
- [ ] Add Python package metadata, manifest dataclasses/verification, CLI scripts, ignore policy, and pinned submodule metadata.
- [ ] Re-run the two test files and verify they pass.

### Task 2: Frame, Geometry, Depth, and Configuration Core

**Files:**
- Create: `perception_plus_plus_core/types.py`
- Create: `perception_plus_plus_core/config.py`
- Create: `perception_plus_plus_core/validation/{depth.py,geometry.py,quality.py}`
- Create: `config/cup_tracking.yaml`
- Test: `tests/core/test_types.py`, `tests/core/test_depth.py`, `tests/core/test_geometry.py`, `tests/core/test_quality.py`

**Interfaces:**
- Produces: `CameraIntrinsics`, `FrameBundle`, `MeshSpec`, `PoseResult`
- Produces: `TrackingConfig.from_mapping(mapping) -> TrackingConfig`
- Produces: `evaluate_quality(frame, result, previous, config) -> QualityDecision`

- [ ] Write focused failing tests for array shape/type validation, millimetre-to-metre depth conversion, finite SE(3), translation/rotation deltas, mask/depth/workspace checks, and configuration invariants.
- [ ] Run `python -m pytest tests/core -q`; verify missing-module failures.
- [ ] Implement only the typed data, conversion, geometry, configuration, and structured reason codes required by the tests.
- [ ] Re-run `python -m pytest tests/core -q`; verify all tests pass.

### Task 3: Tracking State Machine and Fake Backends

**Files:**
- Create: `perception_plus_plus_core/fp_adapter/base.py`
- Create: `perception_plus_plus_core/detection/base.py`
- Create: `perception_plus_plus_core/tracking/{state.py,manager.py}`
- Create: `perception_plus_plus_core/testing/fakes.py`
- Test: `tests/tracking/test_manager.py`

**Interfaces:**
- Consumes: `FrameBundle`, `MeshSpec`, `PoseResult`, `QualityDecision`
- Produces: `FpAdapter`, `CupDetector` protocols
- Produces: `TrackingManager.process(frame) -> TrackingOutput`
- Produces: `TrackingOutput` with state, optional publishable pose, reason, counters, and fatal flag

- [ ] Write failing tests for initialization, healthy tracking, invalid-frame hysteresis, LOST cadence, reset, stable reinitialization, detector misses, and fatal OOM/model-load behavior.
- [ ] Run `python -m pytest tests/tracking/test_manager.py -q`; verify failure for missing manager.
- [ ] Implement deterministic protocols, fakes, four-state manager, and fatal classification.
- [ ] Re-run the manager tests and full CPU suite; verify all pass.

### Task 4: Production YOLO and FoundationPose++ Adapters

**Files:**
- Create: `perception_plus_plus_core/detection/yolo.py`
- Create: `perception_plus_plus_core/fp_adapter/foundationpose_plus_plus.py`
- Create: `perception_plus_plus_core/errors.py`
- Create: `scripts/fpplusplus_smoke.py`
- Test: `tests/adapters/test_yolo_adapter.py`, `tests/adapters/test_fp_adapter.py`

**Interfaces:**
- Produces: `YoloCupDetector(weights, class_id, confidence)`
- Produces: `FoundationPosePlusPlusAdapter(upstream_root, model_paths)`
- Produces: `DependencyUnavailable`, `ModelLoadError`, `CudaOutOfMemory`

- [ ] Write failing tests around injected model objects for YOLO result normalization, lazy imports, FP++ call conversion, reset, and exception classification.
- [ ] Run adapter tests and verify missing implementations fail.
- [ ] Implement adapters without importing GPU dependencies at package import time; isolate upstream API differences behind small compatibility functions.
- [ ] Re-run adapter tests and the CPU suite.

### Task 5: Deterministic Replay, Metrics, and Readiness

**Files:**
- Create: `perception_plus_plus_core/validation/replay.py`
- Create: `perception_plus_plus_core/validation/readiness.py`
- Create: `scripts/replay_rgbd.py`, `scripts/pre_camera_check.py`
- Create: `docs/validation.md`
- Test: `tests/validation/test_replay.py`, `tests/validation/test_readiness.py`

**Interfaces:**
- Produces: `ReplayReader` for an indexed NPZ/JSON recording contract
- Produces: `run_replay(manager, reader) -> ReplayReport`
- Produces: `CheckResult(PASS|FAIL|SKIP)` and JSON readiness report

- [ ] Write failing tests for deterministic ordering, malformed recordings, output metrics, required-vs-unavailable checks, and READY calculation.
- [ ] Run validation tests and verify missing implementations fail.
- [ ] Implement replay, timing/state metrics, machine-readable checks, CLI entry points, and recording documentation.
- [ ] Re-run validation and full CPU tests.

### Task 6: ROS 2 Messages, Node, Launch, and Tests

**Files:**
- Create: `ros_ws/src/perception_plus_plus_msgs/{CMakeLists.txt,package.xml,msg/TrackingStatus.msg}`
- Create: `ros_ws/src/perception_plus_plus_ros/{package.xml,setup.py,setup.cfg,resource/perception_plus_plus_ros}`
- Create: `ros_ws/src/perception_plus_plus_ros/perception_plus_plus_ros/{node.py,conversion.py,publisher.py}`
- Create: `ros_ws/src/perception_plus_plus_ros/{launch/cup_tracking.launch.py,config/cup_tracking.yaml}`
- Test: `tests/ros/test_contract.py`, `ros_ws/src/perception_plus_plus_ros/test/test_conversion.py`

**Interfaces:**
- Consumes: `TrackingManager.process`
- Produces: synchronized RGB/depth/camera-info input and PoseStamped/TF/TrackingStatus output

- [ ] Write CPU-discoverable contract tests for package metadata, message fields, parameters, and publication gating; add ROS-installed conversion tests.
- [ ] Run contract tests and verify failure because packages are absent.
- [ ] Add message package and ROS adapter with lazy ROS imports where CPU contract inspection requires it.
- [ ] Run CPU suite; when ROS exists also run `colcon build --symlink-install && colcon test`.

### Task 7: Humble/Jazzy Containers and Final Verification

**Files:**
- Create: `docker/{humble,jazzy}/Dockerfile`
- Create: `docker/{humble,jazzy}/requirements.lock`
- Create: `docker/compose.yaml`
- Create: `scripts/container_smoke.sh`, `scripts/run_tests.sh`
- Create: `docs/{setup.md,camera-validation.md}`

**Interfaces:**
- Produces: distribution-specific GPU/ROS environments with the same source contract
- Produces: one command for CPU tests and one command for pre-camera readiness

- [ ] Add static tests validating base OS/ROS distribution, shared workspace copy, non-root runtime, NVIDIA expectations, and distinct lock files.
- [ ] Run the static tests and verify they fail before container files exist.
- [ ] Add container definitions, compose profiles, smoke/test scripts, setup, and camera handoff documentation.
- [ ] Run `bash scripts/run_tests.sh`, Python compilation, manifest/upstream checks, and the pre-camera report.
- [ ] If Docker is available, build and smoke both images; otherwise record explicit SKIP reasons.
- [ ] Inspect `git diff --check` and `git status --short`; leave all changes uncommitted for the user.
