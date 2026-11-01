"""GeoTIFF preprocessing matching the professor's training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

PATCH_HEIGHT = 72
PATCH_WIDTH = 72
PATCH_CHANNELS = 1
PATCH_SIZE = 72

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


class PatchGridMismatchError(TiffPreprocessingError):
    """Raised when two tiled rasters do not share the same patch grid."""


def _require_gdal():
    try: from osgeo import gdal
    except ImportError as exc:
        raise InvalidTiffError(
            "GDAL is not installed. Install system GDAL and run: "
            "pip install -r requirements.txt" ) from exc
    return gdal


def _open_dataset(path: Path, gdal: Any):
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None: raise InvalidTiffError(f"GDAL could not open file: {path.name}")
    return dataset


def _zero_warnings(array: np.ndarray) -> tuple[float, list[str]]:
    zero_percentage = float(np.sum(array == 0.0) / array.size)
    warnings: list[str] = []
    if zero_percentage > BLANK_TILE_ZERO_THRESHOLD:
        warnings.append(BLANK_TILE_WARNING)
    return zero_percentage, warnings


def _normalize_patch(array: np.ndarray) -> np.ndarray:
    if array.shape != (PATCH_HEIGHT, PATCH_WIDTH):
        raise InvalidTiffError(
            f"Expected a {PATCH_HEIGHT} x {PATCH_WIDTH} patch, got {array.shape}"
        )
    channel = array[..., np.newaxis]
    return (channel / 255.0).astype(np.float32, copy=False)


def _load_clean_raster(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    gdal = _require_gdal()
    tiff_path = Path(path)
    if not tiff_path.exists(): raise InvalidTiffError(f"File not found: {tiff_path.name}")

    dataset = _open_dataset(tiff_path, gdal)
    try:
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        if width <= 0 or height <= 0: raise InvalidTiffError( f"Raster has invalid dimensions ({width} x {height}): {tiff_path.name}")

        band_count = dataset.RasterCount
        if band_count < 1: raise InvalidTiffError(f"Raster has no bands: {tiff_path.name}")

        band = dataset.GetRasterBand(1)
        if band is None: raise InvalidTiffError(f"Could not read band 1: {tiff_path.name}")

        array = band.ReadAsArray()
        if array is None: raise InvalidTiffError(f"Could not read raster data: {tiff_path.name}")

        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 2: raise InvalidTiffError( f"Expected a single-band 2D raster, got shape {array.shape}: {tiff_path.name}" )

        original_min = float(np.min(array))
        original_max = float(np.max(array))

        nodata_pixel_count = int(np.sum(array == NODATA_VALUE))
        gdal_nodata = band.GetNoDataValue()
        if gdal_nodata is not None:
            nodata_pixel_count = max( nodata_pixel_count, int(np.sum(array == np.float32(gdal_nodata))), )
            array[array == np.float32(gdal_nodata)] = 0.0

        array[array == NODATA_VALUE] = 0.0

        nan_count = int(np.isnan(array).sum())
        inf_count = int(np.isinf(array).sum())
        if nan_count: array[np.isnan(array)] = 0.0
        if inf_count: array[np.isinf(array)] = 0.0

        if float(array.max()) <= 0.0: raise EmptyRasterError( f"Raster contains only invalid or NoData values: {tiff_path.name}" )

        zero_percentage, warnings = _zero_warnings(array)

        metadata: dict[str, Any] = {
            "original_width": width,
            "original_height": height,
            "original_min": original_min,
            "original_max": original_max,
            "nodata_pixel_count": nodata_pixel_count,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "zero_percentage": zero_percentage,
            "warnings": warnings,
        }
        return array, metadata
    finally: dataset = None


def _extract_patches(array: np.ndarray, *, patch_size: int = PATCH_SIZE) -> list[dict[str, Any]]:
    height, width = array.shape
    if height < patch_size or width < patch_size:
        raise InvalidTiffError(
            f"Raster is {width} x {height}, which is smaller than the "
            f"{patch_size} x {patch_size} tiling window."
        )

    patches: list[dict[str, Any]] = []
    patch_index = 0
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            window = array[y:y + patch_size, x:x + patch_size]
            zero_percentage, warnings = _zero_warnings(window)
            patches.append({
                "array": window,
                "patch_index": patch_index,
                "row_offset": y,
                "col_offset": x,
                "zero_percentage": zero_percentage,
                "warnings": warnings,
            })
            patch_index += 1
    return patches


def preprocess_resize(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Stage 4b: load a GeoTIFF and resize the whole raster to 72x72.

    Returns a normalized patch shaped (72, 72, 1) float32 and metadata.
    Batch dimension is not added here; inference adds (1, 72, 72, 1).
    """
    array, metadata = _load_clean_raster(path)

    resized = array.shape != (PATCH_HEIGHT, PATCH_WIDTH)
    if resized:
        array = cv2.resize( array, (PATCH_WIDTH, PATCH_HEIGHT), interpolation=cv2.INTER_NEAREST, )

    normalized = _normalize_patch(array)
    metadata.update({
        "pipeline": "resize",
        "output_shape": tuple(normalized.shape),
        "normalized_min": float(np.min(normalized)),
        "normalized_max": float(np.max(normalized)),
        "resized": resized,
    })
    return normalized, metadata


def preprocess_tile(path: str | Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    """
    Stage 4a: slice a GeoTIFF into non-overlapping 72x72 patches.

    Remainder pixels on the right and bottom edges are discarded.
    Returns normalized patches and aggregate metadata including per-patch stats.
    """
    array, metadata = _load_clean_raster(path)
    extracted = _extract_patches(array, patch_size=PATCH_SIZE)

    patches: list[np.ndarray] = []
    patch_records: list[dict[str, Any]] = []
    for item in extracted:
        normalized = _normalize_patch(item["array"])
        patches.append(normalized)
        patch_records.append({
            "patch_index": item["patch_index"],
            "row_offset": item["row_offset"],
            "col_offset": item["col_offset"],
            "zero_percentage": item["zero_percentage"],
            "warnings": item["warnings"],
            "normalized_min": float(np.min(normalized)),
            "normalized_max": float(np.max(normalized)),
            "output_shape": tuple(normalized.shape),
        })

    metadata.update({
        "pipeline": "tile",
        "resized": False,
        "patch_count": len(patches),
        "output_shape": (PATCH_HEIGHT, PATCH_WIDTH, PATCH_CHANNELS),
        "patches": patch_records,
    })
    return patches, metadata


def preprocess_tiff(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Backward-compatible alias for preprocess_resize (Stage 4b)."""
    return preprocess_resize(path)
