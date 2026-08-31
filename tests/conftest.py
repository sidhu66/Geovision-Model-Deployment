"""Shared pytest fixtures for GeoTIFF preprocessing and Flask tests."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from src.tiff_preprocessing import NODATA_VALUE

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "siamese_unet_model"


def _require_gdal():
    pytest.importorskip("osgeo.gdal")
    from osgeo import gdal

    return gdal


def write_synthetic_tiff(path: Path, array: np.ndarray, *, nodata: float | None = None) -> Path:
    gdal = _require_gdal()
    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        pytest.skip("GTiff driver unavailable")

    height, width = array.shape
    dataset = driver.Create(str(path), width, height, 1, gdal.GDT_Float32)
    if dataset is None:
        raise RuntimeError(f"Could not create synthetic TIFF at {path}")

    band = dataset.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(float(nodata))
    band.WriteArray(array.astype(np.float32))
    band.FlushCache()
    dataset.FlushCache()
    dataset = None
    return path


@pytest.fixture
def synthetic_tiff_factory(tmp_path):
    def _factory(name: str, array: np.ndarray, *, nodata: float | None = None) -> Path:
        return write_synthetic_tiff(tmp_path / name, array, nodata=nodata)

    return _factory


@pytest.fixture
def valid_72x72_array():
    rng = np.random.default_rng(7)
    return rng.integers(1, 200, size=(72, 72)).astype(np.float32)


@pytest.fixture
def valid_72x72_tiff(synthetic_tiff_factory, valid_72x72_array):
    return synthetic_tiff_factory("valid_72x72.tif", valid_72x72_array)


@pytest.fixture
def resize_100x100_tiff(synthetic_tiff_factory):
    rng = np.random.default_rng(11)
    array = rng.integers(1, 200, size=(100, 100)).astype(np.float32)
    return synthetic_tiff_factory("resize_100x100.tif", array)


@pytest.fixture
def nodata_tiff(synthetic_tiff_factory):
    array = np.full((72, 72), 50.0, dtype=np.float32)
    array[0:10, 0:10] = NODATA_VALUE
    return synthetic_tiff_factory("nodata.tif", array, nodata=float(NODATA_VALUE))


@pytest.fixture
def mostly_zero_tiff(synthetic_tiff_factory):
    array = np.zeros((72, 72), dtype=np.float32)
    array[0:10, 0:10] = 25.0
    return synthetic_tiff_factory("mostly_zero.tif", array)


@pytest.fixture
def different_72x72_tiff(synthetic_tiff_factory):
    rng = np.random.default_rng(99)
    array = rng.integers(1, 200, size=(72, 72)).astype(np.float32)
    return synthetic_tiff_factory("different_72x72.tif", array)


@pytest.fixture
def flask_app():
    from app import app as flask_application

    flask_application.config.update(
        {
            "TESTING": True,
            "WTF_CSRF_ENABLED": False,
        }
    )
    return flask_application


@pytest.fixture
def client(flask_app):
    return flask_app.test_client()


@pytest.fixture
def model_available():
    return MODEL_PATH.exists() and (MODEL_PATH / "saved_model.pb").exists()
