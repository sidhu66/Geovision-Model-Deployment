"""Application-level comparison pipelines for resize (4b) and tile (4a)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.core.inference import compare_patches
from src.core.preview import patch_to_base64_png
from src.core.tiff_preprocessing import (
    PatchGridMismatchError,
    preprocess_resize,
    preprocess_tile,
)

VALID_PIPELINES = {"resize", "tile"}


def _summary_from_scores(scores: list[float]) -> dict[str, float | int]:
    array = np.asarray(scores, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _validate_tile_grids(meta1: dict[str, Any], meta2: dict[str, Any]) -> None:
    if (
        meta1["original_width"] != meta2["original_width"]
        or meta1["original_height"] != meta2["original_height"]
        or meta1["patch_count"] != meta2["patch_count"]
    ):
        raise PatchGridMismatchError(
            "Tile mode requires both GeoTIFFs to have the same pixel dimensions "
            f"({meta1['original_width']} x {meta1['original_height']} vs "
            f"{meta2['original_width']} x {meta2['original_height']})."
        )


def _run_resize(path1: Path, path2: Path) -> dict[str, Any]:
    patch1, meta1 = preprocess_resize(path1)
    patch2, meta2 = preprocess_resize(path2)
    result = compare_patches(patch1, patch2)
    return {
        "mode": "resize",
        "similarity_score": result["similarity_score"],
        "interpretation": result["interpretation"],
        "patch1": patch1,
        "patch2": patch2,
        "preview1": patch_to_base64_png(patch1),
        "preview2": patch_to_base64_png(patch2),
        "meta1": meta1,
        "meta2": meta2,
        "warnings": meta1.get("warnings", []) + meta2.get("warnings", []),
    }


def _run_tile(path1: Path, path2: Path) -> dict[str, Any]:
    patches1, meta1 = preprocess_tile(path1)
    patches2, meta2 = preprocess_tile(path2)
    _validate_tile_grids(meta1, meta2)

    patch_results: list[dict[str, Any]] = []
    for record1, record2, patch1, patch2 in zip(
        meta1["patches"], meta2["patches"], patches1, patches2, strict=True
    ):
        result = compare_patches(patch1, patch2)
        warnings = list(record1.get("warnings", [])) + list(record2.get("warnings", []))
        patch_results.append({
            "patch_index": record1["patch_index"],
            "row_offset": record1["row_offset"],
            "col_offset": record1["col_offset"],
            "similarity_score": result["similarity_score"],
            "interpretation": result["interpretation"],
            "warnings": warnings,
            "meta1": record1,
            "meta2": record2,
            "patch1": patch1,
            "patch2": patch2,
        })

    scores = [item["similarity_score"] for item in patch_results]
    summary = _summary_from_scores(scores)

    min_index = int(np.argmin(scores))
    max_index = int(np.argmax(scores))

    def _preview_pair(index: int) -> dict[str, Any]:
        item = patch_results[index]
        return {
            "patch_index": item["patch_index"],
            "row_offset": item["row_offset"],
            "col_offset": item["col_offset"],
            "similarity_score": item["similarity_score"],
            "interpretation": item["interpretation"],
            "warnings": item["warnings"],
            "preview1": patch_to_base64_png(item["patch1"]),
            "preview2": patch_to_base64_png(item["patch2"]),
            "meta1": item["meta1"],
            "meta2": item["meta2"],
        }

    worst_pair = _preview_pair(min_index)
    best_pair = _preview_pair(max_index)

    file_warnings = list(meta1.get("warnings", [])) + list(meta2.get("warnings", []))
    patch_warnings = [warning for item in patch_results for warning in item["warnings"]]

    return {
        "mode": "tile",
        "patch_results": [
            {
                "patch_index": item["patch_index"],
                "row_offset": item["row_offset"],
                "col_offset": item["col_offset"],
                "similarity_score": item["similarity_score"],
                "interpretation": item["interpretation"],
                "warnings": item["warnings"],
                "meta1": item["meta1"],
                "meta2": item["meta2"],
            }
            for item in patch_results
        ],
        "summary": summary,
        "best_pair": best_pair,
        "worst_pair": worst_pair,
        "meta1": meta1,
        "meta2": meta2,
        "warnings": file_warnings + patch_warnings,
    }


def run_comparison(
    path1: str | Path,
    path2: str | Path,
    *,
    pipeline: str,
) -> dict[str, Any]:
    """Compare two GeoTIFFs using the resize (4b) or tile (4a) pipeline."""
    if pipeline not in VALID_PIPELINES:
        raise ValueError(f"Unknown pipeline {pipeline!r}. Expected one of {sorted(VALID_PIPELINES)}.")

    file1 = Path(path1)
    file2 = Path(path2)
    if pipeline == "tile":
        return _run_tile(file1, file2)
    return _run_resize(file1, file2)
