# FoundationPose++ Real-Time Cup Tracking System Design

## Goal

Build an independent perception repository that tracks one cup from aligned
RealSense D435i RGB-D input, publishes a valid 6D pose and TF, stops publishing
invalid poses, and automatically recovers tracking. The repository must use the
same core and ROS interfaces on Ubuntu 22.04/ROS 2 Humble and Ubuntu 24.04/ROS 2
Jazzy.

The first delivery ends immediately before physical camera validation. It
includes all source, containers, model preparation, offline replay, ROS
interfaces, and smoke checks needed to determine whether camera testing can
begin. Physical D435i tests and measured performance baselines remain follow-up
work.

## Supported Scope

The implementation includes FoundationPose++, FoundationPose, Cutie,
KalmanFilter6D, a YOLO cup initializer, aligned RGB-D input, CAD-based
initialization, pose/TF/status output, recovery, separate Humble and Jazzy
containers, model integrity checking, deterministic replay, and launch/config
files.

The first version supports one configured cup class and one configured mesh. It
does not include Qwen2-VL, SAM-HQ, multi-object tracking, Isaac ROS runtime
coupling, automatic CAD selection, or unaligned depth.

## Repository and Upstream Strategy

`perception_plus_plus` is an independent Git repository. The upstream
`teal024/FoundationPose-plus-plus` repository is stored at
`external/foundationpose_plus_plus` as a Git submodule pinned to commit
`58aa715`. The complete commit SHA is recorded by Git and repeated in an
upstream manifest. Local compatibility changes are stored as ordered patch
files rather than untracked edits to the submodule.

Weights, TensorRT engines, recorded RGB-D data, and large meshes are excluded
from Git. Model manifests record the logical component, expected filename,
download URL or user-supplied-path policy, SHA-256, license source, and whether
the file is required. Preparation scripts fail closed on a digest mismatch.

## Source Layout

```text
perception_plus_plus/
├── external/foundationpose_plus_plus/
├── patches/foundationpose_plus_plus/
├── perception_plus_plus_core/
│   ├── fp_adapter/
│   ├── detection/
│   ├── tracking/
│   └── validation/
├── ros_ws/src/
│   ├── perception_plus_plus_msgs/
│   └── perception_plus_plus_ros/
├── docker/{humble,jazzy}/
├── config/
├── assets/{meshes,model_manifests}/
├── scripts/
├── tests/
└── docs/
```

## Core Boundaries

### Frame Types and Adapter

The ROS-independent core uses NumPy arrays and immutable typed results.
`FrameBundle` contains RGB, metric depth, pinhole intrinsics, timestamp, and
camera frame. `PoseResult` contains a 4x4 object-to-camera transform, quality
measurements, timestamp, and optional diagnostic reason.

`FpAdapter` exposes:

```python
initialize(frame: FrameBundle, mask: np.ndarray, mesh: MeshSpec) -> PoseResult
track(frame: FrameBundle) -> PoseResult
reset() -> None
```

The production implementation adapts FP++ FoundationPose, Cutie, and
KalmanFilter6D. Imports are lazy so configuration checks and CPU unit tests do
not require CUDA. A protocol-compatible fake supports deterministic tests.

### Detection

`CupDetector.detect(rgb)` returns zero or more masks with class, confidence, and
bounding box metadata. The manager selects the highest-confidence configured
cup class that passes mask validation. YOLO runs only in `INITIALIZING`,
periodically in `LOST`, and once when entering `REINITIALIZING`; it does not run
for each healthy tracking frame.

### Quality Gate

The quality gate validates synchronized input, finite rigid transforms, mask
area, valid masked-depth ratio, depth range, workspace bounds, and maximum
frame-to-frame translation and rotation. It returns a structured decision with
a stable reason code. Thresholds come from one versioned YAML configuration
schema shared by replay and ROS.

### Tracking Manager

The manager owns exactly four public states:

- `INITIALIZING`: search for a YOLO mask and initialize FP++.
- `TRACKING`: publish only quality-approved poses.
- `LOST`: publish no pose or TF; periodically search for the cup.
- `REINITIALIZING`: validate consecutive stable initialization/update results.

Consecutive invalid frames are required to leave `TRACKING`. Consecutive valid
frames are required to leave `REINITIALIZING`. Entering recovery resets
FoundationPose, Cutie, and Kalman state. CUDA out-of-memory and unrecoverable
model/configuration failures are fatal and are not retried. A fatal condition
continues to publish status diagnostics but never pose or TF.

## ROS 2 Contract

One Python ROS node uses message filters to synchronize RGB, aligned depth, and
camera info. It converts depth to meters, verifies that camera-info and image
frames agree, calls the core manager, and publishes:

- `geometry_msgs/msg/PoseStamped` on
  `/perception_plus_plus/cup/pose`
- `perception_plus_plus_msgs/msg/TrackingStatus` on
  `/perception_plus_plus/cup/tracking_status`
- TF from the configured camera frame to `cup`

`TrackingStatus` contains state constants, last valid pose time, stable failure
reason code and text, consecutive valid/invalid counts, and a fatal flag.
Pose/TF publication occurs only when the manager returns an explicitly valid
pose. Node source and message definitions are identical for Humble and Jazzy;
only dependency/container layers differ.

## Runtime and Configuration

Humble uses Ubuntu 22.04 and ROS 2 Humble. Jazzy uses Ubuntu 24.04 and ROS 2
Jazzy. Each image pins its own Python, PyTorch, CUDA-extension, ROS, and system
dependency compatibility layer while copying the same workspace source.

Configuration covers input/output topics, frame IDs, mesh path and source-unit
scale, model paths, cup class/confidence, synchronization tolerance, depth
range, mask/depth thresholds, workspace, pose-change limits, invalid-frame
hysteresis, recovery cadence, and reinitialization stabilization.

The node requires RGB-aligned depth. The default launch file assumes the
RealSense ROS aligned-depth topic contract but does not start or configure the
physical camera driver.

## Error Handling

Startup validation reports all missing files, invalid hashes, incompatible
configuration fields, unavailable CUDA extensions, and unsupported encodings
before processing frames where possible. Recoverable per-frame faults produce
stable reason codes and feed hysteresis. Unexpected FP++ exceptions become
recoverable tracking faults unless classified as CUDA OOM, model-load failure,
or invalid static configuration.

No last-known pose is republished after loss. Replay and ROS paths use the same
manager decisions so status behavior cannot diverge between validation and
runtime.

## Verification

CPU tests cover intrinsics, depth conversion, mesh scaling, rigid-transform
validation, pose deltas, quality decisions, state hysteresis, reset/recovery,
fatal errors, model manifests, and deterministic replay.

ROS package tests cover message/parameter structure, synchronization rejection,
frame consistency, and the rule that invalid decisions cannot publish pose or
TF. Where ROS is installed, `colcon test` exercises the packages.

Environment checks cover submodule revision, model digests, CUDA-extension
imports, weight loading, upstream sample inference, replay inference, and
container smoke commands. Checks unavailable on the current host must return a
machine-readable `SKIP` with the missing prerequisite; a required prerequisite
that is present but broken returns `FAIL`.

The pre-camera readiness report records:

- source and upstream revisions;
- host GPU/CUDA/container/ROS capability;
- model and mesh integrity;
- CPU and ROS test results;
- Humble and Jazzy image build/smoke results;
- FP++ sample and replay results;
- remaining physical-camera-only checks.

Camera readiness is `READY` only if every non-camera required check passes.
Physical FPS, latency, VRAM stability, occlusion, departure, and reappearance
baselines are explicitly not claimed until D435i validation is performed.

## Delivery Sequence

1. Create the independent repository, pinned submodule, manifests, and
   Humble/Jazzy container skeletons.
2. Implement core types, geometry/depth validation, and quality gate with CPU
   tests.
3. Implement the tracking manager and deterministic fake backends with complete
   state-transition tests.
4. Implement production YOLO and FP++ adapters plus preparation and smoke
   scripts.
5. Implement replay input, metrics, and readiness reporting.
6. Implement messages, ROS node, launch, configuration, TF, and ROS tests.
7. Run every check supported by this host and produce the pre-camera readiness
   report.

Each stage must have a reproducible automated check before the next stage is
considered complete.
