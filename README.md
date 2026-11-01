# Geovision Model Deployment — SWE Change Detection

Phase 1 inference pipeline and Phase 2 Flask demo for comparing two 72×72 single-channel patches using a pre-trained Siamese Attention U-Net SavedModel.

## Phase 1: Inference

### What the model does

The model at `models/siamese_unet_model` is a **pre-trained Siamese Attention U-Net** that accepts **two** single-channel 72×72 patches and returns **one** similarity score in the range **0–1** (sigmoid output):

- **1** → the two inputs are similar
- **0** → the two inputs are not similar

The model is used **as-is**. It is **not** retrained or modified.

### Expected input format

- **Type:** NumPy `ndarray`
- **Shape:** `(72, 72)`, `(72, 72, 1)`, or `(1, 72, 72, 1)`
- **Channels:** single channel only
- **Dtype:** cast to `float32` at inference time

### Preprocessing (Phase 2)

Phase 2 offers two application-level pipelines. Choose one on the upload form.

| | **Tile (4a)** | **Resize (4b, default)** |
|--|----------------|--------------------------|
| **Operation** | Non-overlapping `arr[y:y+72, x:x+72]` slices | `cv2.resize(arr, (72, 72), INTER_NEAREST)` |
| **Output** | Many 72×72 patches per file | One 72×72 patch per file |
| **Spatial meaning** | Each patch is a native-resolution sub-region | The whole raster is compressed into 72×72 |
| **Aligned with** | Training patch generation | Trend / inference scripts |

Shared cleanup before either path:

```
GeoTIFF
  → first raster band
  → NoData / NaN / Inf → 0
  → Tile (4a) or Resize (4b)
  → add channel dimension
  → divide by 255
  → Siamese model
```

`preprocess_tiff()` remains a backward-compatible alias for resize mode.

**Tile mode constraints:** both uploads must have identical pixel dimensions. Remainder pixels on the right and bottom edges are discarded. Empty patches are kept and warned about if more than 50% of pixels are zero. Each patch pair is one model inference, so N patches means N forward passes.

No additional normalization, clipping, standardization, or scaling is applied.

### Output

```python
{
    "similarity_score": 0.87,
    "interpretation": "high_similarity"
}
```

**Application-level interpretation thresholds** (heuristic, not model-learned, not scientifically validated):

| Score | Interpretation |
|---|---|
| ≥ 0.80 | `high_similarity` |
| 0.50 – 0.79 | `moderate_similarity` |
| < 0.50 | `low_similarity` |

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
brew install gdal   # macOS system dependency for Python GDAL bindings
pip install -r requirements.txt
```

Verified loader: **TensorFlow 2.20.0** with `tf.saved_model.load()`.

Do **not** use `tf.keras.models.load_model()` — Keras 3 does not support this legacy SavedModel format.

Optional: override model location with `SIAMESE_MODEL_PATH`.

### Run inference (CLI smoke test)

From the project root:

```bash
python scripts/test_inference.py
```

Load patches from `.npy` files:

```bash
python scripts/test_inference.py \
    --patch1 data/patch1.npy \
    --patch2 data/patch2.npy
```

### Python API

```python
import numpy as np
from src.core.inference import compare_patches

patch1 = np.random.rand(72, 72)
patch2 = patch1.copy()

result = compare_patches(patch1, patch2)
print(result["similarity_score"])
print(result["interpretation"])
```

### SavedModel signature (discovered programmatically)

| Property | Value |
|---|---|
| Signature | `serving_default` |
| Inputs | `input_1`, `input_2` — shape `(None, 72, 72, 1)`, dtype `float32` |
| Output | `dense` — shape `(None, 1)`, dtype `float32` |

## Phase 2: Flask GeoTIFF Comparison Demo

### Start the app

From the project root:

```bash
python -m src
```

Open: **http://127.0.0.1:5000/**

Production-style launch (optional):

```bash
gunicorn -w 1 -b 127.0.0.1:5000 "src.web.app:app"
```

### Upload workflow

1. Upload **GeoTIFF A** and **GeoTIFF B** on the home page.
2. Choose **Resize (4b)** or **Tile (4a)**. Resize is the default.
3. Form field names: **`file1`**, **`file2`**, and **`pipeline`** (`resize` or `tile`)
4. Allowed extensions: `.tif`, `.tiff`
5. Maximum upload size: 50 MB per request
6. The backend preprocesses both rasters and shows:
   - **Resize:** one similarity score, heuristic interpretation, and 72×72 previews
   - **Tile:** per-patch similarity scores and previews for the highest- and lowest-scoring pairs
   - preprocessing metadata for reproducibility

Uploaded files are stored only in a temporary directory for the duration of the request and are deleted afterward.

### Example result

- **File 1:** patch_a.tif
- **File 2:** patch_b.tif
- **Similarity score:** `0.8234`
- **Interpretation:** High similarity
- **Preprocessing details:** original dimensions, resize status, NoData counts, zero percentage

## Scientific caveat

The current Phase 2 pipeline reproduces the known preprocessing from the professor's training and trend code:

- **Tile (4a)** matches training-time patch slicing.
- **Resize (4b)** matches the trend/inference whole-raster resize path.
- Both then apply NoData cleanup, `/255` normalization, and a `(72, 72, 1)` channel dimension.

However, we have **not** yet established:

- the exact source SWE dataset used during training
- the geographic meaning of each TIFF
- the temporal relationship between paired TIFFs
- whether resizing entire rasters to 72×72 is scientifically appropriate for our future SNODAS/CEDA/ERA5-Land datasets
- whether the model's similarity score is calibrated for SWE change magnitude

Therefore, this Flask application is an **inference/demo system**, **not** yet the final scientific SWE change-analysis pipeline.

## Scope

**Included now:**

- SavedModel inference via `src/core/inference.py`
- Professor-aligned GeoTIFF preprocessing via `src/core/tiff_preprocessing.py`
- Resize (4b) and tile (4a) comparison pipelines via `src/core/pipeline.py`
- Flask upload/compare UI with pipeline selection via `src/web/`

**Not included yet:**

- SNODAS/CEDA/ERA5 downloading
- geographic reprojection
- writing tiled GeoTIFF patches to disk
- regression, Mann-Kendall, changepoint analysis
- ecosystem analysis
- model retraining or fine-tuning
- database, authentication, or cloud deployment

## Project structure

```
Geovision-Model-Deployment/
├── src/
│   ├── __main__.py
│   ├── paths.py
│   ├── core/
│   │   ├── inference.py
│   │   ├── tiff_preprocessing.py
│   │   ├── pipeline.py
│   │   └── preview.py
│   └── web/
│       ├── app.py
│       ├── templates/
│       │   ├── index.html
│       │   └── result.html
│       └── static/
│           └── style.css
├── models/
│   └── siamese_unet_model/
├── scripts/
│   └── test_inference.py
├── data/
├── results/
├── requirements.txt
└── README.md
```
