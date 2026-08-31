"""Geovision Phase 2 Flask application for GeoTIFF similarity comparison."""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from src.inference import InferenceError, compare_patches, get_engine
from src.preview import patch_to_base64_png
from src.tiff_preprocessing import (
    EmptyRasterError,
    InvalidTiffError,
    TiffPreprocessingError,
    preprocess_tiff,
)

ALLOWED_EXTENSIONS = {".tif", ".tiff"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024

INTERPRETATION_LABELS = {
    "high_similarity": "High similarity",
    "moderate_similarity": "Moderate similarity",
    "low_similarity": "Low similarity",
}

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_change_me")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

logger = logging.getLogger(__name__)


def _allowed_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _interpretation_label(interpretation: str) -> str:
    return INTERPRETATION_LABELS.get(
        interpretation,
        interpretation.replace("_", " ").capitalize(),
    )


def _format_metadata(metadata: dict) -> dict:
    return {
        **metadata,
        "zero_percentage_display": f"{metadata['zero_percentage'] * 100:.1f}%",
        "processed_dimensions": (
            f"{metadata['output_shape'][0]} x {metadata['output_shape'][1]} "
            f"x {metadata['output_shape'][2]}"
        ),
        "original_dimensions": (
            f"{metadata['original_width']} x {metadata['original_height']}"
        ),
    }


def _save_upload(file_storage, temp_dir: Path) -> tuple[Path, str]:
    original_name = secure_filename(file_storage.filename or "upload.tif")
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    save_path = temp_dir / unique_name
    file_storage.save(save_path)
    return save_path, original_name


def _validate_upload(field_name: str):
    if field_name not in request.files:
        raise InvalidTiffError(f"Missing upload field: {field_name}")

    file_storage = request.files[field_name]
    if not file_storage or file_storage.filename == "":
        raise InvalidTiffError(f"No file selected for {field_name}.")

    if not _allowed_extension(file_storage.filename):
        raise InvalidTiffError(
            f"Unsupported file extension for {field_name}. "
            "Only .tif and .tiff files are allowed."
        )
    return file_storage


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/compare", methods=["POST"])
def compare():
    temp_dir = Path(tempfile.mkdtemp(prefix="geovision_uploads_"))
    saved_paths: list[Path] = []

    try:
        file1 = _validate_upload("file1")
        file2 = _validate_upload("file2")

        path1, name1 = _save_upload(file1, temp_dir)
        path2, name2 = _save_upload(file2, temp_dir)
        saved_paths.extend([path1, path2])

        patch1, meta1 = preprocess_tiff(path1)
        patch2, meta2 = preprocess_tiff(path2)

        result = compare_patches(patch1, patch2)

        return render_template(
            "result.html",
            file1_name=name1,
            file2_name=name2,
            similarity_score=result["similarity_score"],
            interpretation=_interpretation_label(str(result["interpretation"])),
            interpretation_key=result["interpretation"],
            preview1=patch_to_base64_png(patch1),
            preview2=patch_to_base64_png(patch2),
            meta1=_format_metadata(meta1),
            meta2=_format_metadata(meta2),
            warnings=meta1.get("warnings", []) + meta2.get("warnings", []),
        )
    except RequestEntityTooLarge:
        flash("Uploaded files exceed the 50 MB limit.")
        return redirect(url_for("index"))
    except EmptyRasterError as exc:
        logger.exception("Empty raster uploaded")
        flash(str(exc))
        return redirect(url_for("index"))
    except InvalidTiffError as exc:
        logger.exception("Invalid TIFF upload")
        flash(str(exc))
        return redirect(url_for("index"))
    except TiffPreprocessingError as exc:
        logger.exception("TIFF preprocessing failed")
        flash(f"Preprocessing failed: {exc}")
        return redirect(url_for("index"))
    except InferenceError as exc:
        logger.exception("Model inference failed")
        flash(f"Inference failed: {exc}")
        return redirect(url_for("index"))
    except Exception:
        logger.exception("Unexpected error during comparison")
        flash("An unexpected error occurred. Please try again.")
        return redirect(url_for("index"))
    finally:
        for path in saved_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not delete temporary upload: %s", path)
        try:
            temp_dir.rmdir()
        except OSError:
            logger.warning("Could not remove temporary upload directory: %s", temp_dir)


@app.route("/health")
def health():
    return {"status": "ok"}


def _warm_up_model() -> None:
    try:
        engine = get_engine()
        engine._load()
        logger.info("Model loaded successfully from %s", engine.model_path)
    except InferenceError:
        logger.exception("Model failed to load at startup")


_warm_up_model()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="127.0.0.1", port=5000, debug=True)
