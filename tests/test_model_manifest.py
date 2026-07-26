import hashlib
import json

import pytest

from perception_plus_plus_core.assets.manifest import (
    create_lock,
    load_lock,
    load_manifest,
    verify_file,
    verify_path,
    verify_lock,
    write_lock,
)


def test_manifest_rejects_non_sha256_digest(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"version": 1, "models": [{
        "name": "cutie", "filename": "cutie.pth", "sha256": "bad",
        "required": True, "source": "user", "license_url": "https://example.test"
    }]}))
    with pytest.raises(ValueError, match="sha256"):
        load_manifest(path)


def test_verify_file_distinguishes_missing_required_and_optional(tmp_path):
    path = tmp_path / "models.json"
    models = []
    for name, required in (("required", True), ("optional", False)):
        models.append({"name": name, "filename": f"{name}.bin", "sha256": "0" * 64,
                       "required": required, "source": "user",
                       "license_url": "https://example.test"})
    path.write_text(json.dumps({"version": 1, "models": models}))
    manifest = load_manifest(path)
    assert verify_file(manifest.models[0], tmp_path).status == "FAIL"
    assert verify_file(manifest.models[1], tmp_path).status == "SKIP"


def test_verify_file_accepts_matching_digest(tmp_path):
    payload = b"weights"
    (tmp_path / "model.bin").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"version": 1, "models": [{
        "name": "model", "filename": "model.bin", "sha256": digest,
        "required": True, "source": "user", "license_url": "https://example.test"
    }]}))
    entry = load_manifest(path).models[0]
    assert verify_file(entry, tmp_path).status == "PASS"


def test_verify_path_rejects_downloaded_bytes_that_differ_from_source_digest(tmp_path):
    manifest_path = tmp_path / "models.json"
    manifest_path.write_text(json.dumps({"version": 2, "models": [{
        "name": "model", "filename": "model.bin", "sha256": hashlib.sha256(b"good").hexdigest(),
        "required": True, "provider": "http", "source": "https://example.test/model",
        "license_url": "https://example.test/license",
    }]}))
    downloaded = tmp_path / "staged.bin"
    downloaded.write_bytes(b"bad")
    assert verify_path(load_manifest(manifest_path).models[0], downloaded).status == "FAIL"


def _source_manifest(tmp_path, filename="yolo/yolo11n-seg.pt"):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"version": 2, "models": [{
        "name": "yolo_cup",
        "filename": filename,
        "required": True,
        "provider": "ultralytics",
        "source": "yolo11n-seg.pt",
        "license_url": "https://www.ultralytics.com/license",
    }]}))
    return load_manifest(path)


def test_source_manifest_accepts_official_provider_without_placeholder_digest(tmp_path):
    manifest = _source_manifest(tmp_path)
    assert manifest.models[0].provider == "ultralytics"
    assert manifest.models[0].sha256 is None


def test_manifest_rejects_unsafe_and_duplicate_paths(tmp_path):
    for filename in ("../escape.pth", "/absolute.pth"):
        with pytest.raises(ValueError, match="relative"):
            _source_manifest(tmp_path, filename)


def test_lock_records_size_and_sha256(tmp_path):
    manifest = _source_manifest(tmp_path)
    model = tmp_path / "models/yolo/yolo11n-seg.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"segmentation")
    lock = create_lock(manifest, tmp_path / "models")
    assert lock.models[0].size == len(b"segmentation")
    assert lock.models[0].sha256 == hashlib.sha256(b"segmentation").hexdigest()


def test_write_and_verify_lock_rejects_changed_model(tmp_path):
    manifest = _source_manifest(tmp_path)
    root = tmp_path / "models"
    model = root / "yolo/yolo11n-seg.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"segmentation")
    lock_path = root / "models.lock.json"
    write_lock(create_lock(manifest, root), lock_path)
    assert load_lock(lock_path).models[0].size == len(b"segmentation")
    assert verify_lock(lock_path, root).status == "PASS"
    model.write_bytes(b"changed")
    assert verify_lock(lock_path, root).status == "FAIL"
