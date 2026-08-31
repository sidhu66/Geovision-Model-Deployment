"""Siamese Attention U-Net inference for 72x72 single-channel patch comparison."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

PATCH_HEIGHT = 72
PATCH_WIDTH = 72
PATCH_CHANNELS = 1
DEFAULT_SIGNATURE = "serving_default"

# Application-level interpretation thresholds (NOT model-learned).
HIGH_SIMILARITY_THRESHOLD = 0.80
MODERATE_SIMILARITY_THRESHOLD = 0.50

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_PATH = _REPO_ROOT / "models" / "siamese_unet_model"


class InferenceError(Exception):
    """Base exception for inference pipeline errors."""


class ModelNotFoundError(InferenceError):
    """Raised when the SavedModel directory or required files are missing."""


class ModelLoadError(InferenceError):
    """Raised when the SavedModel cannot be loaded."""


class InvalidPatchError(InferenceError):
    """Raised when an input patch fails validation."""


def _default_model_path() -> Path:
    env_path = os.getenv("SIAMESE_MODEL_PATH")
    if env_path: return Path(env_path)
    return _DEFAULT_MODEL_PATH


def interpret_similarity(score: float) -> str:
    """
    Map a raw similarity score to an application-level label.

    Thresholds are heuristic defaults for Phase 1 reporting only.
    They are NOT scientifically validated and were NOT learned by the model.
    """
    if score >= HIGH_SIMILARITY_THRESHOLD: return "high_similarity"
    if score >= MODERATE_SIMILARITY_THRESHOLD: return "moderate_similarity"
    return "low_similarity"


def _require_tensorflow():
    try: import tensorflow as tf
    except ImportError as exc:
        raise ModelLoadError(
            "TensorFlow is not installed. Install project dependencies with: "
            "pip install -r requirements.txt"
        ) from exc
    return tf


def _validate_model_directory(model_path: Path) -> None:
    if not model_path.exists(): raise ModelNotFoundError(f"Model directory not found: {model_path}")
    if not model_path.is_dir(): raise ModelNotFoundError(f"Model path is not a directory: {model_path}")

    required_files = ("saved_model.pb",)
    missing = [name for name in required_files if not (model_path / name).exists()]
    if missing: raise ModelNotFoundError( f"Model directory is missing required file(s) {missing}: {model_path}" )


def _normalize_patch(patch: Any, *, label: str) -> np.ndarray:
    if not isinstance(patch, np.ndarray): raise InvalidPatchError(f"{label} must be a NumPy ndarray, got {type(patch)!r}")

    array = np.asarray(patch)

    if array.size == 0: raise InvalidPatchError(f"{label} is empty")

    if not np.issubdtype(array.dtype, np.number): raise InvalidPatchError( f"{label} must have a numeric dtype, got {array.dtype}" )

    if np.isnan(array).any(): raise InvalidPatchError(f"{label} contains NaN values")

    if np.isinf(array).any(): raise InvalidPatchError(f"{label} contains infinite values")

    if array.ndim == 2:
        if array.shape != (PATCH_HEIGHT, PATCH_WIDTH):
            raise InvalidPatchError(
                f"{label} must have shape ({PATCH_HEIGHT}, {PATCH_WIDTH}) or "
                f"({PATCH_HEIGHT}, {PATCH_WIDTH}, 1), got {array.shape}"
            )
        array = array[..., np.newaxis]
    elif array.ndim == 3:
        if array.shape == (PATCH_HEIGHT, PATCH_WIDTH, PATCH_CHANNELS): pass
        elif array.shape == (PATCH_CHANNELS, PATCH_HEIGHT, PATCH_WIDTH): array = np.transpose(array, (1, 2, 0))
        else:
            raise InvalidPatchError(
                f"{label} must have shape ({PATCH_HEIGHT}, {PATCH_WIDTH}, 1), "
                f"got {array.shape}"
            )
    elif array.ndim == 4:
        if array.shape[0] != 1: raise InvalidPatchError( f"{label} batch dimension must be 1 when provided, got shape {array.shape}" )
        
        array = array[0]
        return _normalize_patch(array, label=label)
    
    else: raise InvalidPatchError( f"{label} must have 2, 3, or 4 dimensions, got {array.ndim}" )

    if array.shape != (PATCH_HEIGHT, PATCH_WIDTH, PATCH_CHANNELS):
        raise InvalidPatchError(
            f"{label} must have shape ({PATCH_HEIGHT}, {PATCH_WIDTH}, 1), "
            f"got {array.shape}" )

    return array


def _to_numpy_dtype(dtype: Any) -> np.dtype:
    as_numpy = getattr(dtype, "as_numpy_dtype", None)
    if as_numpy is not None: return np.dtype(as_numpy)
    return np.dtype(dtype)


def _to_model_batch(patch: np.ndarray, *, dtype: Any) -> np.ndarray:
    batch = patch[np.newaxis, ...]
    np_dtype = _to_numpy_dtype(dtype)
    try: return batch.astype(np_dtype, copy=False)
    except (TypeError, ValueError) as exc:
        raise InvalidPatchError( f"Unable to cast patch to model dtype {dtype}: {exc}" ) from exc


class SiameseInferenceEngine:
    """Loads the SavedModel once and runs pairwise patch comparison."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path is not None else _default_model_path()
        self._model = None
        self._signature = None
        self._input_names: list[str] = []
        self._patch1_input: str | None = None
        self._patch2_input: str | None = None
        self._output_name: str | None = None
        self._input_dtype = None

    def _load(self) -> None:
        if self._signature is not None: return

        _validate_model_directory(self.model_path)
        tf = _require_tensorflow()

        try: self._model = tf.saved_model.load(str(self.model_path))
        except Exception as exc:
            raise ModelLoadError(
                "Failed to load SavedModel. "
                f"TensorFlow={tf.__version__}, Keras={tf.keras.__version__}. "
                f"Path={self.model_path}. Error: {exc}" ) from exc

        signatures = self._model.signatures
        if DEFAULT_SIGNATURE not in signatures:
            available = sorted(signatures.keys())
            raise ModelLoadError( f"Signature '{DEFAULT_SIGNATURE}' not found. Available: {available}" )

        signature = signatures[DEFAULT_SIGNATURE]
        self._signature = signature

        if len(signature.structured_input_signature) < 2: raise ModelLoadError( "Unexpected SavedModel input signature structure; expected keyword inputs." )

        input_specs = signature.structured_input_signature[1]
        if not input_specs: raise ModelLoadError("SavedModel signature defines no input tensors.")

        self._input_names = list(input_specs.keys())
        if len(self._input_names) != 2:
            raise ModelLoadError(
                f"Expected exactly 2 model inputs, found {len(self._input_names)}: "
                f"{self._input_names}" )

        if {"input_1", "input_2"}.issubset(input_specs):
            self._patch1_input = "input_1"
            self._patch2_input = "input_2"
        else:
            self._patch1_input = self._input_names[0]
            self._patch2_input = self._input_names[1]

        output_specs = signature.structured_outputs
        if not output_specs: raise ModelLoadError("SavedModel signature defines no output tensors.")

        self._output_name = next(iter(output_specs.keys()))
        self._input_dtype = input_specs[self._input_names[0]].dtype

        for name in self._input_names:
            spec = input_specs[name]
            shape = tuple(spec.shape.as_list()) if spec.shape is not None else None
            if shape is not None:
                expected = (None, PATCH_HEIGHT, PATCH_WIDTH, PATCH_CHANNELS)
                spatial = shape[-3:]
                if spatial != expected[-3:]:
                    raise ModelLoadError(
                        f"Input '{name}' has unexpected shape {shape}; "
                        f"expected (*, {PATCH_HEIGHT}, {PATCH_WIDTH}, {PATCH_CHANNELS})."
                    )

    @property
    def input_names(self) -> list[str]:
        self._load()
        return list(self._input_names)

    @property
    def output_name(self) -> str:
        self._load()
        assert self._output_name is not None
        return self._output_name

    def compare(self, patch1: Any, patch2: Any) -> dict[str, float | str]:
        self._load()
        assert self._signature is not None
        assert self._output_name is not None
        assert self._input_dtype is not None

        normalized_1 = _normalize_patch(patch1, label="patch1")
        normalized_2 = _normalize_patch(patch2, label="patch2")

        batch_1 = _to_model_batch(normalized_1, dtype=self._input_dtype)
        batch_2 = _to_model_batch(normalized_2, dtype=self._input_dtype)

        assert self._patch1_input is not None
        assert self._patch2_input is not None

        feed = { self._patch1_input: batch_1, self._patch2_input: batch_2, }

        try: outputs = self._signature(**feed)
        except Exception as exc: raise InferenceError(f"Model inference failed: {exc}") from exc

        if self._output_name not in outputs: raise InferenceError( f"Expected output '{self._output_name}', got keys: {list(outputs.keys())}" )

        raw = outputs[self._output_name].numpy()
        score = float(np.asarray(raw).reshape(-1)[0])

        return { "similarity_score": score, "interpretation": interpret_similarity(score), }


_engine: SiameseInferenceEngine | None = None


def get_engine(model_path: str | Path | None = None) -> SiameseInferenceEngine:
    """Return a process-wide inference engine, loading the model on first use."""
    global _engine
    if model_path is not None: return SiameseInferenceEngine(model_path=model_path)
    if _engine is None: _engine = SiameseInferenceEngine()
    return _engine


def compare_patches( patch1: Any, patch2: Any, *, model_path: str | Path | None = None, ) -> dict[str, float | str]:
    """
    Compare two 72x72 single-channel patches and return similarity metadata.

    Inputs may be shaped (72, 72), (72, 72, 1), or (1, 72, 72, 1).
    Values are passed through unchanged aside from dtype casting to float32.
    """
    engine = get_engine(model_path=model_path)
    return engine.compare(patch1, patch2)
