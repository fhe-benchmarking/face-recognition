#!/usr/bin/env python3
"""
cleartext_impl.py - Cleartext reference for the face verification workload
using ArcFace (InsightFace).

Reads face pairs, runs face detection, alignment, feature extraction and
computes cosine similarity scores using ArcFace, and writes one score per line.
Used as the plaintext baseline in quality comparison.

Usage:  python3 cleartext_impl.py <test_pairs_npz> <output_scores_path>

Input:  test_pairs.npz -- keys pair_NNNNN_img0 / pair_NNNNN_img1, each (3, H, W) uint8 RGB
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
import numpy as np
from pathlib import Path
from numpy.linalg import norm


def load_arcface():
    """Load ArcFace via InsightFace FaceAnalysis (detection + alignment + recognition)."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1)
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


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: cleartext_impl.py <test_pairs_npz> <output_scores_path>")

    pairs_path  = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not pairs_path.exists():
        sys.exit(f"[harness] Error: test pairs not found: {pairs_path}")

    # Keep the benchmark console owned by run_submission.py, consistent with
    # ml-inference. This also suppresses InsightFace/onnxruntime diagnostics.
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)

    npz = np.load(pairs_path)
    n   = len(npz.files) // 2

    print(f"[harness] ArcFace cleartext: {n} pairs, loading model...")
    rec = load_arcface()
    print("[harness] Model ready. Computing embeddings...")

    scores = []
    for i in range(n):
        emb1 = get_embedding(rec, npz[f'pair_{i:05d}_img0'])
        emb2 = get_embedding(rec, npz[f'pair_{i:05d}_img1'])
        scores.append(cosine_similarity(emb1, emb2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(f"{s:.6f}" for s in scores) + "\n")
    print(f"[harness] ArcFace scores written → {output_path}")


if __name__ == "__main__":
    main()
