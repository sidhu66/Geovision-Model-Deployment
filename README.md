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

### Preprocessing (Phase 2 / training-aligned)

Phase 2 reproduces the professor's known training preprocessing:

```
GeoTIFF
  → first raster band
  → NoData → 0
  → resize to 72×72 using INTER_NEAREST (if needed)
  → add channel dimension
  → divide by 255
  → Siamese model
```

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
from src.inference import compare_patches

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
python app.py
```

Open: **http://127.0.0.1:5000/**

Production-style launch (optional):

```bash
gunicorn -w 1 -b 127.0.0.1:5000 app:app
```

### Upload workflow

1. Upload **GeoTIFF A** and **GeoTIFF B** on the home page.
2. Form field names: **`file1`** and **`file2`**
3. Allowed extensions: `.tif`, `.tiff`
4. Maximum upload size: 50 MB per request
5. The backend preprocesses both rasters, runs the Siamese model once, and shows:
   - similarity score
   - heuristic interpretation
   - 72×72 preview images
   - preprocessing metadata for reproducibility

Uploaded files are stored only in a temporary directory for the duration of the request and are deleted afterward.

### Example result

- **File 1:** patch_a.tif
- **File 2:** patch_b.tif
- **Similarity score:** `0.8234`
- **Interpretation:** High similarity
- **Preprocessing details:** original dimensions, resize status, NoData counts, zero percentage

### Run tests

```bash
pytest tests/ -v
```

Tests use synthetic GeoTIFF fixtures and do not require real SNODAS data.

## Scientific caveat

The current Phase 2 pipeline reproduces the known preprocessing from the professor's training code:

```
GeoTIFF
  → first raster band
  → NoData → 0
  → resize to 72×72 using INTER_NEAREST
  → add channel dimension
  → divide by 255
  → Siamese model
```

However, we have **not** yet established:

- the exact source SWE dataset used during training
- the geographic meaning of each TIFF
- the temporal relationship between paired TIFFs
- whether resizing entire rasters to 72×72 is scientifically appropriate for our future SNODAS/CEDA/ERA5-Land datasets
- whether the model's similarity score is calibrated for SWE change magnitude

Therefore, this Flask application is an **inference/demo system**, **not** yet the final scientific SWE change-analysis pipeline.

## Scope

**Included now:**

- SavedModel inference via `src/inference.py`
- Professor-aligned GeoTIFF preprocessing via `src/tiff_preprocessing.py`
- Flask upload/compare UI
- Synthetic test coverage

**Not included yet:**

- SNODAS/CEDA/ERA5 downloading
- geographic reprojection
- spatial tiling across large rasters
- regression, Mann-Kendall, changepoint analysis
- ecosystem analysis
- model retraining or fine-tuning
- database, authentication, or cloud deployment

## Project structure

```
Geovision-Model-Deployment/
├── app.py
├── src/
│   ├── inference.py
│   ├── tiff_preprocessing.py
│   └── preview.py
├── models/
│   └── siamese_unet_model/
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── style.css
├── scripts/
│   └── test_inference.py
├── tests/
├── data/
├── results/
├── requirements.txt
└── README.md
```
