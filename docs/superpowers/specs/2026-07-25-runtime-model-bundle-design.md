# FoundationPose++ Runtime and Portable Model Bundle Design

## Goal

Turn the existing ROS-independent FoundationPose++ framework into a reproducible
runtime for an Intel RealSense D435i and an RTX 4070. Model installation is
online by default, while the complete model set can be moved to another Humble
or Jazzy machine on an external drive and imported without network access.

The target cup remains model-based: the operator supplies `cup.obj`; no
instance-specific neural-network training is required.

## Confirmed hardware and camera contract

The D435i is connected over USB 3 at 5000 Mbit/s. Color and depth metadata are
stable at approximately 30 Hz. The runtime consumes:

- `/camera/camera/color/image_raw`
- `/camera/camera/aligned_depth_to_color/image_raw`
- `/camera/camera/color/camera_info`

RGB and aligned depth must use `camera_color_optical_frame`. The default camera
profile is synchronized 640x480 at 30 Hz with aligned depth. The tracker uses
`SENSOR_DATA` subscriber QoS. Operators may select a higher profile after
measuring end-to-end tracking latency.

## Model-based cup initialization

FoundationPose receives the metric RGB-D frame, camera intrinsics, an
instance mask, and the cup CAD mesh. The mesh, rather than a cup-specific
learned checkpoint, defines the tracked object geometry.

The initial mask comes from the COCO `cup` category (class ID 41) of
`yolo11n-seg.pt`. The existing `yolo11n.pt` default is invalid because that
detection checkpoint does not produce masks. YOLO runs only during
initialization and recovery. Cutie propagates the accepted mask between those
events.

If the generic COCO model cannot recognize a visually unusual cup, the runtime
reports a detector miss and remains in `INITIALIZING`; it must never publish an
unvalidated pose. Manual ROI/mask input and cup-specific detector training are
outside this delivery.

## Official model sources

The online bootstrap accepts only these declared sources:

- FoundationPose refiner `2023-10-28-18-33-37` and scorer
  `2024-01-11-20-02-45` from the official NVlabs FoundationPose Google Drive
  weights folder.
- Cutie `cutie-base-mega.pth` and RITM
  `coco_lvis_h18_itermask.pth` from the official Cutie v1.0 GitHub release.
- `yolo11n-seg.pt` resolved by the pinned `ultralytics==8.3.161` release
  downloader.

Downloads go to a temporary file in the target filesystem. A completed file is
hashed and atomically moved into `models/`. An existing valid file is reused.
An existing invalid file is rejected unless the operator explicitly requests
replacement. Network errors leave no file that can be mistaken for a valid
model.

The source manifest is version-controlled. A generated lock manifest records
the exact byte size and SHA-256 of every installed file. Readiness checks use
the lock manifest and fail closed when it is absent, incomplete, or does not
match the model tree.

## Portable external-drive bundle

The exporter creates a directory, not a compressed archive, so multi-gigabyte
checkpoint files are not needlessly recompressed and individual files can be
inspected:

```text
perception-plus-plus-models-v1/
├── bundle.json
├── LICENSES/
└── models/
    ├── foundationpose/
    ├── cutie/
    └── yolo/
```

`bundle.json` includes schema version, creation time, source URLs, relative
paths, sizes, and SHA-256 values. Export verifies the installed lock before
copying and verifies the destination after copying. Import first verifies the
entire external bundle, then copies through temporary files and writes the
local lock manifest last. It never follows symlinks and rejects absolute paths,
parent traversal, duplicate paths, missing files, extra model files, and hash
mismatches.

The same bundle is valid for Humble and Jazzy because model bytes are
distribution-independent. The two containers mount the host `models/`
directory read-only.

## Runtime integration

The ROS tracker defaults to project/package-share-resolved paths rather than
the caller's current working directory. A RealSense launch file starts the
camera and tracker together with the validated profile, topic names, aligned
depth, and QoS settings. A tracker-only launch remains available when the
camera driver runs on the host.

The runtime validates before constructing GPU models:

1. model lock and file hashes;
2. mesh and scale;
3. CUDA/PyTorch availability;
4. FoundationPose, Cutie, and YOLO imports;
5. ROS camera topic contract;
6. one captured initialization frame;
7. pose initialization followed by a short tracking sequence.

A failure identifies its layer and exits without publishing pose or TF.

## Testing and acceptance

CPU tests cover:

- source manifest validation and download resume/failure behavior;
- SHA-256 lock generation and verification;
- bundle export/import, path safety, corruption, missing and extra files;
- enforcement of a segmentation-capable YOLO result;
- installed path resolution;
- camera launch parameters and ROS topic contracts.

Hardware acceptance on the RTX 4070 requires:

- both target container images build;
- CUDA and compiled extensions import;
- all official model files pass their locked hashes;
- `yolo11n-seg.pt` produces a cup mask on a live or recorded frame;
- FoundationPose initializes from that mask and `cup.obj`;
- at least a short live sequence publishes valid pose/status without a fatal
  state;
- no pose or TF is published in `LOST`.

The repository does not commit checkpoint bytes. It commits the downloader,
source declarations, the exact installed lock after bootstrap, bundle tools,
tests, and documentation. The user remains responsible for reviewing and
accepting upstream model licenses before redistribution.
