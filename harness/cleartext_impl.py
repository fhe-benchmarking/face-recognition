#!/usr/bin/env python3
"""
cleartext_impl.py - Cleartext reference for the face verification workload
using ArcFace (InsightFace).

Reads the indexed face-pair store, runs face detection and feature extraction,
computes cosine similarity scores using ArcFace, and writes one score per line.
Used as the plaintext baseline in quality comparison.

Usage:  python3 cleartext_impl.py <test_pairs_h5> <output_scores_path>

Input:  test_pairs.h5 -- encoded image0/image1 arrays in input order
Output: one cosine similarity float per line
"""
# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import warnings
from contextlib import contextmanager
import h5py
import numpy as np
from pathlib import Path
from numpy.linalg import norm
from face_dataset_store import decode_image


ARCFACE_DET_SIZE = (640, 640)


@contextmanager
def suppress_native_output():
    """Suppress native library output while preserving harness logging."""
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.close(devnull_fd)


def load_arcface():
    """Load ArcFace via InsightFace FaceAnalysis (detection + alignment + recognition)."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    # Keep the quality reference aligned with the submission's explicit
    # single-scale face detector. InsightFace 1.0 changed the implicit default
    # to a 128x128 + 640x640 multi-scale pass.
    app.prepare(ctx_id=-1, det_size=ARCFACE_DET_SIZE)
    return app


def get_embedding(app, img_chw_rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Detect, align, and embed one face image using ArcFace.

    Args:
        app:                 InsightFace FaceAnalysis app
        img_chw_rgb_uint8:   (3, H, W) uint8 RGB, full-resolution

    Returns:
        1-D float32 embedding vector, or zero vector if no face detected.
        A zero vector produces cosine_similarity=0.0, which is treated as a
        real score and will degrade EER/TAR metrics if face detection fails.
    """
    img = img_chw_rgb_uint8.transpose(1, 2, 0)[:, :, ::-1]  # CHW RGB → HWC BGR
    faces = app.get(img)
    if not faces:
        print("[harness] Warning: no face detected, returning zero embedding", file=sys.stderr)
        return np.zeros(512, dtype=np.float32)
    return faces[0].embedding.flatten()


def cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    denom = norm(e1) * norm(e2)
    if denom == 0:
        return 0.0
    return float(np.dot(e1, e2) / denom)


def score_indexed_pairs(model, pairs, output) -> int:
    """Score an indexed pair store incrementally and return its pair count."""
    if not {"image0", "image1"}.issubset(pairs):
        raise ValueError("Indexed cleartext input is missing image datasets")
    count = len(pairs["image0"])
    if len(pairs["image1"]) != count:
        raise ValueError("Indexed cleartext image counts do not match")
    for index in range(count):
        embedding0 = get_embedding(model, decode_image(pairs["image0"][index]))
        embedding1 = get_embedding(model, decode_image(pairs["image1"][index]))
        output.write(f"{cosine_similarity(embedding0, embedding1):.6f}\n")
    return count


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: cleartext_impl.py <test_pairs_h5> <output_scores_path>")

    pairs_path  = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not pairs_path.exists():
        sys.exit(f"[harness] Error: test pairs not found: {pairs_path}")

    warnings.filterwarnings(
        "ignore", category=FutureWarning, module=r"insightface\..*"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".txt.tmp")
    with suppress_native_output():
        model = load_arcface()
        with h5py.File(pairs_path, "r") as pairs, temporary.open("w") as output:
            score_indexed_pairs(model, pairs, output)
    temporary.replace(output_path)
    print(f"[harness] ArcFace scores written -> {output_path}")


if __name__ == "__main__":
    main()
