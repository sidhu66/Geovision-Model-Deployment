"""Flask route tests for GeoTIFF comparison."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tests.conftest import write_synthetic_tiff


def _upload_pair(client, path1: Path, path2: Path):
    with path1.open("rb") as file1, path2.open("rb") as file2:
        return client.post(
            "/compare",
            data={
                "file1": (file1, path1.name),
                "file2": (file2, path2.name),
            },
            content_type="multipart/form-data",
        )


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"GeoTIFF A" in response.data
    assert b"file1" in response.data
    assert b"file2" in response.data


def test_missing_upload(client, tmp_path):
    tiff_path = tmp_path / "only_one.tif"
    tiff_path.write_bytes(b"placeholder")

    with tiff_path.open("rb") as file1:
        response = client.post(
            "/compare",
            data={"file1": (file1, tiff_path.name)},
            content_type="multipart/form-data",
            follow_redirects=True,
        )

    assert response.status_code == 200
    assert b"Missing upload field" in response.data


def test_wrong_file_extension(client, tmp_path):
    tiff_path = tmp_path / "valid.tif"
    tiff_path.write_bytes(b"placeholder")
    wrong_path = tmp_path / "image.png"
    wrong_path.write_bytes(b"fake png")

    response = _upload_pair(client, tiff_path, wrong_path)
    assert response.status_code == 302


@patch("app.compare_patches", return_value={"similarity_score": 0.91, "interpretation": "high_similarity"})
def test_two_valid_tiffs_reach_model(
    mock_compare,
    client,
    valid_72x72_tiff,
    different_72x72_tiff,
):
    response = _upload_pair(client, valid_72x72_tiff, different_72x72_tiff)

    assert response.status_code == 200
    assert b"0.9100" in response.data
    assert b"High similarity" in response.data
    assert mock_compare.call_count == 1


@patch("app.compare_patches", return_value={"similarity_score": 0.91, "interpretation": "high_similarity"})
def test_identical_tiff_uploaded_twice(mock_compare, client, valid_72x72_tiff):
    response = _upload_pair(client, valid_72x72_tiff, valid_72x72_tiff)

    assert response.status_code == 200
    assert mock_compare.call_count == 1
    args, _kwargs = mock_compare.call_args
    assert args[0].shape == (72, 72, 1)
    assert args[1].shape == (72, 72, 1)


def test_two_different_tiffs_with_real_model(
    client,
    valid_72x72_tiff,
    different_72x72_tiff,
    model_available,
):
    if not model_available:
        pytest.skip("SavedModel not available")

    response = _upload_pair(client, valid_72x72_tiff, different_72x72_tiff)
    assert response.status_code == 200
    assert b"Similarity score" in response.data


def test_invalid_tiff_upload(client, tmp_path):
    invalid_path = tmp_path / "invalid.tif"
    invalid_path.write_bytes(b"not a geotiff")
    valid_path = tmp_path / "valid.tif"
    write_synthetic_tiff(valid_path, __import__("numpy").full((72, 72), 10.0, dtype="float32"))

    response = _upload_pair(client, valid_path, invalid_path)
    assert response.status_code == 302


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
