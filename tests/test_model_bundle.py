import json

import pytest

from perception_plus_plus_core.assets.bundle import (
    export_bundle,
    import_bundle,
    verify_bundle,
)
from perception_plus_plus_core.assets.manifest import (
    create_lock,
    load_manifest,
    verify_lock,
    write_lock,
)


def _models(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"version": 2, "models": [{
        "name": "cup",
        "filename": "yolo/yolo11n-seg.pt",
        "required": True,
        "provider": "http",
        "source": "https://example.test/yolo11n-seg.pt",
        "license_url": "https://example.test/license",
    }]}))
    root = tmp_path / "models"
    model = root / "yolo/yolo11n-seg.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"weights")
    write_lock(create_lock(load_manifest(manifest_path), root), root / "models.lock.json")
    return root


def test_bundle_round_trip_preserves_locked_bytes(tmp_path):
    source = _models(tmp_path / "source")
    bundle = export_bundle(source, tmp_path / "drive")
    target = tmp_path / "target"
    import_bundle(bundle, target)
    assert verify_lock(target / "models.lock.json", target).status == "PASS"


@pytest.mark.parametrize("filename", ["../escape.pth", "/absolute.pth"])
def test_bundle_rejects_unsafe_paths(tmp_path, filename):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "bundle.json").write_text(json.dumps({
        "version": 1,
        "models": [{
            "name": "bad", "filename": filename, "size": 1, "sha256": "0" * 64,
            "source": "x", "license_url": "x",
        }],
    }))
    assert verify_bundle(bundle).status == "FAIL"


def test_bundle_rejects_symlinked_model(tmp_path):
    source = _models(tmp_path / "source")
    bundle = export_bundle(source, tmp_path / "drive")
    model = bundle / "models/yolo/yolo11n-seg.pt"
    model.unlink()
    model.symlink_to("/etc/hosts")
    assert verify_bundle(bundle).status == "FAIL"
