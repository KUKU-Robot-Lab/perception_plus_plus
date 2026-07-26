import io

import pytest

from perception_plus_plus_core.assets.download import DownloadError, download_http


class _Response:
    def __init__(self, chunks):
        self.chunks = iter(chunks)

    def read(self, _size):
        chunk = next(self.chunks)
        if isinstance(chunk, Exception):
            raise chunk
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_failed_download_leaves_no_destination_or_partial(tmp_path):
    destination = tmp_path / "model.pth"
    opener = lambda _url: _Response([b"partial", OSError("network down")])
    with pytest.raises(DownloadError, match="network down"):
        download_http("https://example.test/model", destination, opener=opener)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_download_atomically_installs_complete_file(tmp_path):
    destination = tmp_path / "nested/model.pth"
    opener = lambda _url: _Response([b"abc", b"def", b""])
    download_http("https://example.test/model", destination, opener=opener)
    assert destination.read_bytes() == b"abcdef"
