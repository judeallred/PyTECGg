from pathlib import Path
from unittest.mock import MagicMock, patch
from requests.exceptions import RequestException

import pytest

from pytecgg.utils.download_rinex import download_obs_ring, _download_file


def test_obs_ring_url_construction():
    """
    Verify that the RINEX observation file URL and path are constructed correctly.
    """
    # Mocking _batch_download to avoid downloads
    with patch("pytecgg.utils.download_rinex._batch_download") as mock_batch:
        station = "GROT"
        year = 2023
        doys = [1]
        out = Path("/tmp/gnss")

        download_obs_ring(station, year, doys, out)

        assert mock_batch.called

        tasks = mock_batch.call_args[0][0]
        url, path = tasks[0]

        assert "GROT00ITA" in url
        assert "2023001" in url
        assert path.name == "GROT00ITA_R_20230010000_01D_30S_MO.crx.gz"
        assert path.parent.name == "GROT"


def test_download_file_cleanup_on_failure(tmp_path):
    """
    Verify that temporary files (.tmp) are removed in case of an error.
    """
    mock_session = MagicMock()
    # Emulating a network error (RequestException)
    mock_session.get.side_effect = RequestException("Connection refused")

    dest = tmp_path / "test_file.crx.gz"
    tmp_file = dest.with_suffix(".tmp")
    tmp_file.write_text("dati parziali")

    with pytest.raises(RequestException):
        _download_file(mock_session, "http://fake-url.com", dest)

    assert not tmp_file.exists(), "Temporary file not removed after download failure"
    assert not dest.exists(), "The destination file should not exist"


def test_download_file_success(tmp_path):
    """
    Verify that a file is downloaded successfully and temporary files are cleaned up.
    """
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
    mock_response.status_code = 200
    mock_session.get.return_value = mock_response

    dest = tmp_path / "success_file.crx.gz"

    _download_file(mock_session, "http://fake-url.com", dest)

    assert dest.exists()
    assert dest.read_bytes() == b"chunk1chunk2"
    assert not dest.with_suffix(".tmp").exists()
