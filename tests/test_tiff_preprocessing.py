"""Unit tests for GeoTIFF preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.tiff_preprocessing import (
    BLANK_TILE_WARNING,
    EmptyRasterError,
    InvalidTiffError,
    NODATA_VALUE,
    preprocess_tiff,
)


def test_valid_72x72_tiff(valid_72x72_tiff):
    patch, metadata = preprocess_tiff(valid_72x72_tiff)

    assert patch.shape == (72, 72, 1)
    assert patch.dtype == np.float32
    assert metadata["resized"] is False
    assert metadata["original_width"] == 72
    assert metadata["original_height"] == 72
    assert metadata["warnings"] == []


def test_non_72x72_tiff_is_resized(resize_100x100_tiff):
    patch, metadata = preprocess_tiff(resize_100x100_tiff)

    assert patch.shape == (72, 72, 1)
    assert metadata["resized"] is True
    assert metadata["original_width"] == 100
    assert metadata["original_height"] == 100


def test_nodata_values_are_replaced(nodata_tiff):
    patch, metadata = preprocess_tiff(nodata_tiff)

    assert patch.shape == (72, 72, 1)
    assert metadata["nodata_pixel_count"] >= 100
    assert metadata["nan_count"] == 0
    assert metadata["inf_count"] == 0


def test_mostly_zero_tiff_emits_warning(mostly_zero_tiff):
    patch, metadata = preprocess_tiff(mostly_zero_tiff)

    assert patch.shape == (72, 72, 1)
    assert metadata["zero_percentage"] > 0.50
    assert BLANK_TILE_WARNING in metadata["warnings"]


def test_invalid_tiff_raises(tmp_path):
    invalid_path = tmp_path / "invalid.tif"
    invalid_path.write_bytes(b"not a geotiff")

    with pytest.raises(InvalidTiffError):
        preprocess_tiff(invalid_path)


def test_all_invalid_raster_raises(tmp_path, synthetic_tiff_factory):
    array = np.zeros((72, 72), dtype=np.float32)
    empty_path = synthetic_tiff_factory("empty.tif", array)

    with pytest.raises(EmptyRasterError):
        preprocess_tiff(empty_path)


def test_nan_and_inf_are_replaced(tmp_path, synthetic_tiff_factory):
    array = np.array([[1.0, np.nan], [np.inf, 4.0]], dtype=np.float32)
    path = synthetic_tiff_factory("bad_values.tif", array)

    patch, metadata = preprocess_tiff(path)

    assert patch.shape == (72, 72, 1)
    assert metadata["nan_count"] == 1
    assert metadata["inf_count"] == 1
    assert np.isfinite(patch).all()
