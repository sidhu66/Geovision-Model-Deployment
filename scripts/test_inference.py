#!/usr/bin/env python3
"""Smoke-test script for Siamese SWE patch comparison inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.core.inference import (  # noqa: E402
    InferenceError,
    compare_patches,
    get_engine,
)


def _print_result(title: str, patch1: np.ndarray, patch2: np.ndarray) -> None:
    print(f"\n=== {title} ===")
    print(f"Input 1 shape: {patch1.shape}")
    print(f"Input 2 shape: {patch2.shape}")

    result = compare_patches(patch1, patch2)
    print(f"Similarity score: {result['similarity_score']:.4f}")
    print(f"Interpretation: {result['interpretation']}")


def _load_patch(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Patch file not found: {path}")
    patch = np.load(path)
    return np.asarray(patch)


def run_synthetic_tests() -> None:
    rng = np.random.default_rng(42)

    patch_a = rng.random((72, 72))
    patch_b = patch_a.copy()
    _print_result("Identical patches", patch_a, patch_b)

    patch_c = rng.random((72, 72))
    patch_d = rng.random((72, 72))
    _print_result("Different random patches", patch_c, patch_d)

    rng2 = np.random.default_rng(99)
    patch_e = rng2.random((72, 72))
    patch_f = rng2.random((72, 72))
    _print_result("Different random patches (seed 99)", patch_e, patch_f)


def run_file_tests(patch1_path: Path, patch2_path: Path) -> None:
    patch1 = _load_patch(patch1_path)
    patch2 = _load_patch(patch2_path)
    _print_result(f"Loaded patches: {patch1_path.name} vs {patch2_path.name}", patch1, patch2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Siamese model inference on 72x72 single-channel patches."
    )
    parser.add_argument(
        "--patch1",
        type=Path,
        help="Path to first .npy patch file (72x72 or 72x72x1).",
    )
    parser.add_argument(
        "--patch2",
        type=Path,
        help="Path to second .npy patch file (72x72 or 72x72x1).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if (args.patch1 is None) ^ (args.patch2 is None):
        print("Error: provide both --patch1 and --patch2, or neither.", file=sys.stderr)
        return 2

    try:
        engine = get_engine()
        engine._load()
        print("Model loaded successfully.")
        print(f"Model path: {engine.model_path}")
        print(f"Signature inputs: {engine.input_names}")
        print(f"Signature output: {engine.output_name}")

        if args.patch1 is not None and args.patch2 is not None:
            run_file_tests(args.patch1, args.patch2)
        else:
            run_synthetic_tests()

        return 0
    except InferenceError as exc:
        print(f"Inference error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
