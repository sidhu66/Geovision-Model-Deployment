"""Render normalized raster patches for web display."""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def patch_to_base64_png(normalized_patch: np.ndarray) -> str:
    """
    Render a normalized (72, 72, 1) patch with a scientific colormap.

    Display-only conversion; model input values are not modified.
    """
    array = np.asarray(normalized_patch, dtype=np.float32)
    if array.ndim == 3 and array.shape[-1] == 1: array = array[..., 0]
    elif array.ndim != 2: raise ValueError( f"Expected patch shape (72, 72, 1) or (72, 72), got {array.shape}" )

    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    image = ax.imshow(array, cmap="viridis", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Processed patch", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)

    return base64.b64encode(buffer.getvalue()).decode("ascii")
