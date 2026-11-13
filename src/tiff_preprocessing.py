
"""GeoTIFF preprocessing matching the professor's training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

PATCH_HEIGHT = 72
PATCH_WIDTH = 72
PATCH_CHANNELS = 1

NODATA_VALUE = np.float32(-3.4028234663852886e38)
BLANK_TILE_ZERO_THRESHOLD = 0.50

BLANK_TILE_WARNING = (
    "This raster contains more than 50% zero/NoData pixels and may not "
    "resemble the training data."
)


class TiffPreprocessingError(Exception):
    """Base exception for TIFF preprocessing errors."""


class InvalidTiffError(TiffPreprocessingError):
    """Raised when a file cannot be read as a valid GeoTIFF raster."""


class EmptyRasterError(TiffPreprocessingError):
    """Raised when a raster contains no usable data after cleaning."""


def _require_gdal():
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise InvalidTiffError(
            "GDAL is not installed. Install system GDAL and run: "
            "pip install -r requirements.txt"
        ) from exc

    return gdal


def _open_dataset(path: Path, gdal: Any):
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)

    if dataset is None:
        raise InvalidTiffError(
            f"GDAL could not open file: {path.name}"
        )

    return dataset


def preprocess_tiff(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load and preprocess a GeoTIFF for Siamese model inference.

    Returns a normalized patch shaped (72, 72, 1) float32.
    Batch dimension is not added here; inference adds (1, 72, 72, 1).
    """

    gdal = _require_gdal()
    tiff_path = Path(path)

    if not tiff_path.exists():
        raise InvalidTiffError(f"File not found: {tiff_path.name}")

    dataset = _open_dataset(tiff_path, gdal)

    try:
        width = dataset.RasterXSize
        height = dataset.RasterYSize

        if width <= 0 or height <= 0:
            raise InvalidTiffError(
                f"Raster has invalid dimensions ({width} x {height}): "
                f"{tiff_path.name}"
            )

        band_count = dataset.RasterCount

        if band_count < 1:
            raise InvalidTiffError(
                f"Raster has no bands: {tiff_path.name}"
            )

        band = dataset.GetRasterBand(1)

        if band is None:
            raise InvalidTiffError(
                f"Could not read band 1: {tiff_path.name}"
            )

        array = band.ReadAsArray()

        if array is None:
            raise InvalidTiffError(
                f"Could not read raster data: {tiff_path.name}"
            )

        array = np.asarray(array, dtype=np.float32)

        if array.ndim != 2:
            raise InvalidTiffError(
                f"Expected a single-band 2D raster, got shape "
                f"{array.shape}: {tiff_path.name}"
            )

        original_min = float(np.min(array))
        original_max = float(np.max(array))

        # Identify NoData pixels before replacing them with zero.
        nodata_mask = array == NODATA_VALUE

        gdal_nodata = band.GetNoDataValue()

        if gdal_nodata is not None:
            nodata_mask |= array == np.float32(gdal_nodata)

        nodata_pixel_count = int(np.sum(nodata_mask))

        # Count special values separately.
        special_minus_one_count = int(np.sum(array == -1.0))
        special_minus_two_count = int(np.sum(array == -2.0))

        # Count genuine zero-valued pixels separately from NoData.
        valid_zero_pixel_count = int(
            np.sum((array == 0.0) & ~nodata_mask)
        )

        # Count positive SWE measurements.
        positive_pixel_count = int(np.sum(array > 0.0))

        # Preserve the existing model input behavior.
        array[nodata_mask] = 0.0

        nan_count = int(np.isnan(array).sum())
        inf_count = int(np.isinf(array).sum())

        if nan_count:
            array[np.isnan(array)] = 0.0

        if inf_count:
            array[np.isinf(array)] = 0.0

        if float(array.max()) <= 0.0:
            raise EmptyRasterError(
                f"Raster contains only invalid or NoData values: "
                f"{tiff_path.name}"
            )

        zero_pixel_count = int(np.sum(array == 0.0))
        zero_percentage = zero_pixel_count / array.size

        warnings: list[str] = []

        # Warn when more than 50% of the raster consists of zero or NoData pixels.
        if zero_percentage > BLANK_TILE_ZERO_THRESHOLD:
            warnings.append(BLANK_TILE_WARNING)

        resized = array.shape != (PATCH_HEIGHT, PATCH_WIDTH)

        if resized:
            array = cv2.resize(
                array,
                (PATCH_WIDTH, PATCH_HEIGHT),
                interpolation=cv2.INTER_NEAREST,
            )

        array = array[..., np.newaxis]

        normalized = (array / 255.0).astype(
            np.float32,
            copy=False,
        )

        metadata: dict[str, Any] = {
            "original_width": width,
            "original_height": height,
            "output_shape": tuple(normalized.shape),
            "original_min": original_min,
            "original_max": original_max,
            "normalized_min": float(np.min(normalized)),
            "normalized_max": float(np.max(normalized)),
            "nodata_pixel_count": nodata_pixel_count,
            "special_minus_one_count": special_minus_one_count,
            "special_minus_two_count": special_minus_two_count,
            "positive_pixel_count": positive_pixel_count,
            "valid_zero_pixel_count": valid_zero_pixel_count,
            "valid_zero_percentage": (
            valid_zero_pixel_count / (width * height)
            ),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "resized": resized,
            "zero_percentage": zero_percentage,
            "warnings": warnings,
        }

        return normalized, metadata

    finally:
        dataset = None