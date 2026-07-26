# FoundationPose++ Runtime and Portable Model Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing FoundationPose++ ROS framework reproducibly install official models, move them through an external-drive bundle, start the validated D435i profile, and reject non-segmentation YOLO inputs.

**Architecture:** Extend the existing manifest layer with a source manifest and an installed lock. Pure core functions own hashing, validation, safe path handling, and bundle copying; thin scripts own network/model-provider calls. ROS launch and path resolution remain separate from model acquisition so Humble and Jazzy share identical model bytes.

**Tech Stack:** Python 3.10+, pathlib, hashlib, json, urllib/gdown, Ultralytics 8.3.161, pytest, ROS 2 Humble/Jazzy, RealSense ROS, Docker/NVIDIA Container Toolkit.

## Global Constraints

- Official public weights are the default source.
- Checkpoint bytes remain excluded from Git.
- The same portable model bundle must work for Humble and Jazzy.
- The target cup uses `cup.obj` and does not require instance-specific training.
- YOLO initialization requires an instance-segmentation result for COCO class ID 41.
- The D435i default is synchronized, aligned 640x480 RGB-D at 30 Hz.
- All writes use temporary files followed by atomic replacement.
- Import rejects symlinks, absolute paths, parent traversal, duplicates, missing files, extra model files, and digest mismatches.
- The user creates commits; implementation steps must not commit.

---

### Task 1: Model source and installed-lock contracts

**Files:**
- Modify: `perception_plus_plus_core/assets/manifest.py`
- Modify: `assets/model_manifests/models.json`
- Test: `tests/test_model_manifest.py`

**Interfaces:**
- Consumes: existing `ModelManifest`, `load_manifest()`, and `verify_file()`.
- Produces: `ModelSource`, `LockedModel`, `load_source_manifest(path)`, `create_lock(source_manifest, model_root)`, and `verify_lock(lock_path, model_root)`.

- [ ] **Step 1: Write failing tests for source fields and installed locks**

```python
def test_lock_records_size_and_sha256(tmp_path):
    model = tmp_path / "models/yolo/yolo11n-seg.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"segmentation")
    lock = create_lock(source_manifest_with("yolo/yolo11n-seg.pt"), tmp_path / "models")
    assert lock.models[0].size == 12
    assert len(lock.models[0].sha256) == 64


def test_verify_lock_rejects_changed_model(tmp_path):
    lock_path, model_root = write_valid_lock(tmp_path)
    (model_root / "yolo/yolo11n-seg.pt").write_bytes(b"changed")
    assert verify_lock(lock_path, model_root).status == "FAIL"
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python3 -m pytest tests/test_model_manifest.py -q`

Expected: collection/import failure because the lock interfaces do not exist.

- [ ] **Step 3: Implement the typed source/lock parser and verifier**

Implement strict relative POSIX paths, unique names/paths, 64-character lowercase SHA-256 values for installed locks, byte-size validation, deterministic JSON serialization, and streaming SHA-256 reads.

- [ ] **Step 4: Replace the placeholder manifest with official sources**

Declare the two official FoundationPose Google Drive folders/files, the two official Cutie v1.0 release URLs, and `yolo11n-seg.pt` from pinned Ultralytics 8.3.161. Remove `yolo11n.pt`.

- [ ] **Step 5: Run focused and complete CPU tests**

Run: `python3 -m pytest tests/test_model_manifest.py -q && bash scripts/run_tests.sh`

Expected: all tests pass.

### Task 2: Safe online bootstrap

**Files:**
- Create: `perception_plus_plus_core/assets/download.py`
- Create: `scripts/bootstrap_models.py`
- Modify: `scripts/fetch_models.py`
- Test: `tests/test_model_download.py`

**Interfaces:**
- Consumes: `ModelSource` and lock functions from Task 1.
- Produces: `download_http(source, destination, opener)`, `install_downloaded_file(temp_path, destination)`, `bootstrap_models(manifest, model_root, replace=False)`.

- [ ] **Step 1: Write failing download safety tests**

```python
def test_failed_download_leaves_no_destination(tmp_path):
    destination = tmp_path / "model.pth"
    with pytest.raises(DownloadError):
        download_http(source(), destination, opener=failing_opener)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_existing_file_requires_replace_when_digest_differs(tmp_path):
    destination = tmp_path / "model.pth"
    destination.write_bytes(b"old")
    with pytest.raises(ModelIntegrityError):
        install_model(fake_download, destination, replace=False)
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python3 -m pytest tests/test_model_download.py -q`

Expected: import failure because `assets.download` does not exist.

- [ ] **Step 3: Implement provider-isolated downloads**

Use standard HTTP streaming for Cutie, the pinned Ultralytics asset resolver for YOLO, and `gdown` for the official FoundationPose Drive folder. Provider calls write only to a temporary staging directory. Select only the declared relative files and reject ambiguous/missing results.

- [ ] **Step 4: Generate the installed lock only after all models validate**

`bootstrap_models.py` must accept `--model-root`, `--replace`, and `--verify-only`. A normal successful run writes `models/models.lock.json` last. A partial run returns non-zero without writing a new lock.

- [ ] **Step 5: Run focused and complete tests**

Run: `python3 -m pytest tests/test_model_download.py tests/test_model_manifest.py -q && bash scripts/run_tests.sh`

Expected: all tests pass without network access.

### Task 3: External-drive export and import

**Files:**
- Create: `perception_plus_plus_core/assets/bundle.py`
- Create: `scripts/export_model_bundle.py`
- Create: `scripts/import_model_bundle.py`
- Test: `tests/test_model_bundle.py`

**Interfaces:**
- Consumes: a verified `models/models.lock.json`.
- Produces: `export_bundle(model_root, destination) -> Path`, `verify_bundle(bundle_root) -> CheckResult`, and `import_bundle(bundle_root, model_root, replace=False)`.

- [ ] **Step 1: Write failing round-trip and attack tests**

```python
def test_bundle_round_trip_preserves_locked_bytes(tmp_path):
    source = make_locked_models(tmp_path / "source")
    bundle = export_bundle(source, tmp_path / "drive")
    imported = tmp_path / "target"
    import_bundle(bundle, imported)
    assert verify_lock(imported / "models.lock.json", imported).status == "PASS"


@pytest.mark.parametrize("path", ["../escape.pth", "/absolute.pth"])
def test_bundle_rejects_unsafe_paths(tmp_path, path):
    bundle = make_bundle(tmp_path, model_path=path)
    assert verify_bundle(bundle).status == "FAIL"
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python3 -m pytest tests/test_model_bundle.py -q`

Expected: import failure because `assets.bundle` does not exist.

- [ ] **Step 3: Implement verified directory bundles**

Copy regular files without following symlinks. Write `bundle.json` last, verify every copied destination, reject undeclared files below `models/`, and keep the target lock absent until a complete import succeeds.

- [ ] **Step 4: Add CLI wrappers and usage documentation**

Both scripts accept explicit source/destination paths suitable for `/media/<user>/<drive>/...`, print each verified model, and exit non-zero on any validation error.

- [ ] **Step 5: Run focused and complete tests**

Run: `python3 -m pytest tests/test_model_bundle.py -q && bash scripts/run_tests.sh`

Expected: all tests pass.

### Task 4: Segmentation enforcement and installed runtime paths

**Files:**
- Modify: `perception_plus_plus_core/detection/yolo.py`
- Modify: `perception_plus_plus_core/config.py`
- Modify: `ros_ws/src/perception_plus_plus_ros/perception_plus_plus_ros/node.py`
- Modify: `ros_ws/src/perception_plus_plus_ros/config/cup_tracking.yaml`
- Modify: `tests/adapters/test_adapters.py`
- Modify: `tests/ros/test_contract.py`

**Interfaces:**
- Consumes: installed `models/yolo/yolo11n-seg.pt`, `cup.obj`, and package-share paths.
- Produces: a fatal model error when the checkpoint/result cannot supply instance masks, and deterministic absolute runtime paths.

- [ ] **Step 1: Write failing tests for detection-only output and defaults**

```python
def test_yolo_rejects_detection_checkpoint_without_masks():
    detector = YoloCupDetector("unused", 41, 0.5, model=DetectionOnlyModel())
    with pytest.raises(ModelLoadError, match="segmentation"):
        detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))


def test_ros_defaults_use_yolo_segmentation_weights():
    text = Path(ROS_NODE).read_text()
    assert "models/yolo/yolo11n-seg.pt" in text
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `python3 -m pytest tests/adapters/test_adapters.py tests/ros/test_contract.py -q`

Expected: failure because detection-only output is silently ignored and the old filename remains.

- [ ] **Step 3: Enforce segmentation and resolve runtime roots**

Reject a result set that has boxes but no masks. Resolve mesh, tracking config,
model root, and upstream root from explicit ROS parameters; defaults supplied by
launch files are absolute package/project paths.

- [ ] **Step 4: Run focused and complete tests**

Run: `python3 -m pytest tests/adapters/test_adapters.py tests/ros/test_contract.py -q && bash scripts/run_tests.sh`

Expected: all tests pass.

### Task 5: Validated D435i launch and live readiness

**Files:**
- Create: `ros_ws/src/perception_plus_plus_ros/launch/realsense_cup_tracking.launch.py`
- Modify: `ros_ws/src/perception_plus_plus_ros/launch/cup_tracking.launch.py`
- Create: `scripts/camera_check.py`
- Modify: `scripts/pre_camera_check.py`
- Modify: `docker/compose.yaml`
- Modify: `docker/humble/Dockerfile`
- Modify: `docker/jazzy/Dockerfile`
- Modify: `tests/ros/test_contract.py`
- Modify: `tests/test_containers.py`

**Interfaces:**
- Consumes: D435i ROS topics, installed model lock, Docker GPU runtime.
- Produces: combined camera/tracker launch and a machine-readable camera readiness result.

- [ ] **Step 1: Write failing launch/container contract tests**

```python
def test_combined_launch_uses_validated_realsense_profile():
    text = Path(COMBINED_LAUNCH).read_text()
    for token in ("640x480x30", "align_depth.enable", "enable_sync"):
        assert token in text


def test_compose_mounts_models_read_only_and_uses_gpu():
    compose = yaml.safe_load(Path(COMPOSE).read_text())
    jazzy = compose["services"]["jazzy"]
    assert any("models:" in item and item.endswith(":ro") for item in resolved_volumes(jazzy))
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `python3 -m pytest tests/ros/test_contract.py tests/test_containers.py -q`

Expected: failure because the combined launch and camera check do not exist.

- [ ] **Step 3: Implement combined launch and camera validation**

Include `realsense2_camera/rs_launch.py` with synchronized aligned depth and
matching 640x480x30 profiles. `camera_check.py` validates required topic types,
matching optical frame IDs, dimensions, encodings, positive intrinsics, and
observed metadata/image delivery.

- [ ] **Step 4: Integrate lock/import checks into readiness**

Replace placeholder digest checks with installed-lock verification. Report camera, CUDA, model, FP++ import, smoke-frame, Docker, and ROS build results separately.

- [ ] **Step 5: Run CPU and ROS verification**

Run:

```bash
bash scripts/run_tests.sh
colcon build --base-paths ros_ws/src --build-base /tmp/ppp_runtime_build --install-base /tmp/ppp_runtime_install
colcon test --base-paths ros_ws/src --build-base /tmp/ppp_runtime_build --install-base /tmp/ppp_runtime_install
colcon test-result --test-result-base /tmp/ppp_runtime_build
```

Expected: CPU tests pass, two ROS packages build, and all ROS tests pass.

### Task 6: Official model bootstrap and GPU/live smoke

**Files:**
- Modify: `assets/model_manifests/models.json` with exact installed SHA-256 values
- Create: `models/models.lock.json` (ignored model-runtime artifact)
- Modify: `docs/setup.md`
- Modify: `docs/camera-validation.md`
- Modify: `docs/validation.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: network access, Docker daemon access, RTX 4070, D435i, and the Task 1–5 tools.
- Produces: verified official models, portable bundle instructions, built Humble/Jazzy images, and measured live readiness.

- [ ] **Step 1: Bootstrap and lock official models**

Run: `python3 scripts/bootstrap_models.py`

Expected: every declared file downloads from its official source, validates, and `models/models.lock.json` is written.

- [ ] **Step 2: Verify and export a portable bundle**

Run:

```bash
python3 scripts/bootstrap_models.py --verify-only
python3 scripts/export_model_bundle.py --output /path/to/external-drive/perception-plus-plus-models-v1
```

Expected: all model hashes pass and the destination bundle verifies.

- [ ] **Step 3: Build and smoke both containers**

Run:

```bash
docker build -f docker/humble/Dockerfile -t perception-plus-plus:humble .
docker build -f docker/jazzy/Dockerfile -t perception-plus-plus:jazzy .
bash scripts/container_smoke.sh humble
bash scripts/container_smoke.sh jazzy
```

Expected: CUDA, Torch, FoundationPose, Cutie, YOLO, and compiled extension imports pass in both images.

- [ ] **Step 4: Capture an initialization frame and run FP++ smoke**

Run the combined camera launch, capture synchronized RGB/depth/intrinsics to NPZ, then run:

```bash
python3 scripts/fpplusplus_smoke.py --npz reports/cup_initialization.npz --mesh assets/meshes/cup.obj
```

Expected: a finite 4x4 pose and non-empty mask without fatal state.

- [ ] **Step 5: Run final verification and document measured limits**

Run `python3 scripts/pre_camera_check.py --smoke-npz reports/cup_initialization.npz`.

Expected: `READY`; record initialization latency, tracking FPS, callback latency, peak/steady VRAM, and any remaining hardware-specific limitation without inventing target values.
