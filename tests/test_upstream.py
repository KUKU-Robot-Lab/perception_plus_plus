from pathlib import Path

from perception_plus_plus_core.assets.upstream import check_upstream_revision


def test_upstream_revision_rejects_unexpected_commit(tmp_path):
    result = check_upstream_revision(tmp_path, expected="58aa715")
    assert result.status == "FAIL"
    assert "missing" in result.detail.lower()


def test_upstream_manifest_pins_expected_revision():
    text = Path("assets/model_manifests/upstream.json").read_text()
    assert "58aa715" in text

