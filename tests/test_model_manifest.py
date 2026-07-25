import hashlib
import json

import pytest

from perception_plus_plus_core.assets.manifest import load_manifest, verify_file


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

