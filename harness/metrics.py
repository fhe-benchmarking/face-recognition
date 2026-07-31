#!/usr/bin/env python3
"""
metrics.py - Face verification quality metrics.

calculate_face_metrics() reads cosine similarity scores and ground-truth labels
from files, sweeps similarity thresholds over the full dataset, and returns
EER and TAR@FAR=1%/0.1%.
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

import numpy as np
from pathlib import Path

# EER is stable on every batched variant and does not require selecting an
# operating threshold from the test set. The encrypted model may trail the
# included ArcFace baseline by at most fifteen percentage points of absolute EER.
ACCEPTANCE_METRIC = "eer_gap_to_arcface"
MAX_EER_GAP_TO_ARCFACE = 0.15

def calculate_face_metrics(gt_labels_file: Path, scores_file: Path, tag: str) -> dict:
    """
    Compute EER and TAR@FAR from cosine similarity scores and ground-truth labels.

    Uses a global threshold sweep over the full dataset (no cross-validation),
    which is appropriate for small datasets where KFold calibration is noisy.

    Args:
        gt_labels_file: path to test_labels.txt (one int per line, 0 or 1)
        scores_file:    path to similarity scores file (one float per line)
        tag:            label for printed output

    Returns:
        dict with keys: eer, tar_far_1_percent, tar_far_01_percent
        For a single pair, returns its score and ground-truth label instead.
    """
    labels = [int(l.strip()) for l in Path(gt_labels_file).read_text().strip().splitlines() if l.strip()]
    scores = [float(s.strip()) for s in Path(scores_file).read_text().strip().splitlines() if s.strip()]

    invalid_labels = sorted(set(labels) - {0, 1})
    if invalid_labels:
        raise ValueError(f"[harness] {tag}: labels must be 0 or 1, got {invalid_labels}")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"[harness] {tag}: scores contain non-finite values")
    if len(labels) != len(scores):
        raise ValueError(
            f"[harness] {tag}: label/score count mismatch — "
            f"{len(labels)} labels vs {len(scores)} scores"
        )
    n = len(labels)
    if n == 0:
        print(f"[harness] {tag}: no label/score pairs found")
        return {}
    if n < 2:
        print(f"[harness] {tag}: score={scores[0]:.6f}  label={labels[0]}")
        return {"score": scores[0], "label": labels[0]}

    labels = np.array(labels, dtype=bool)
    scores = np.array(scores, dtype=float)

    n_pos = int(np.sum(labels))
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"[harness] {tag}: EER/TAR require both classes, got "
            f"{n_pos} genuine and {n_neg} impostor pairs"
        )

    # Evaluate the exact ROC points at every distinct score. Uniformly spaced
    # thresholds can skip narrow score intervals and bias EER on small sets.
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(int)
    threshold_indices = np.r_[
        np.flatnonzero(np.diff(sorted_scores)),
        n - 1,
    ]
    true_positives = np.cumsum(sorted_labels)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives
    tprs = np.r_[0.0, true_positives / n_pos]
    fprs = np.r_[0.0, false_positives / n_neg]
    fnrs = 1.0 - tprs

    # Linearly interpolate the point where FPR == FNR. The ROC is discrete,
    # so choosing only the nearest observed point makes EER sample-dependent.
    difference = fprs - fnrs
    upper = int(np.searchsorted(difference, 0.0))
    if upper == 0:
        eer = float(fprs[0])
    elif upper == len(difference):
        eer = float(fprs[-1])
    else:
        lower = upper - 1
        weight = -difference[lower] / (difference[upper] - difference[lower])
        eer = float(fprs[lower] + weight * (fprs[upper] - fprs[lower]))

    # TAR@FAR=1% and TAR@FAR=0.1%: largest TAR where FPR <= target.
    tar_1pct = float(np.max(tprs[fprs <= 0.01]))
    tar_01pct = float(np.max(tprs[fprs <= 0.001]))

    result = {
        "eer":              eer,
        "tar_far_1_percent":  tar_1pct,
        "tar_far_01_percent": tar_01pct,
    }
    print(f"[harness] {tag}: "
          f"EER={eer:.4f}  "
          f"TAR@FAR=1%={tar_1pct:.4f}  "
          f"TAR@FAR=0.1%={tar_01pct:.4f}")
    return result


def compare_to_arcface(encrypted: dict, arcface: dict) -> dict:
    """Return paired quality deltas and the benchmark acceptance verdict."""
    if not encrypted or not arcface:
        return {}
    eer_gap = encrypted["eer"] - arcface["eer"]
    return {
        "eer_gap": eer_gap,
        "tar_at_far_1pct_gap": (
            encrypted["tar_far_1_percent"] - arcface["tar_far_1_percent"]
        ),
        "tar_at_far_01pct_gap": (
            encrypted["tar_far_01_percent"] - arcface["tar_far_01_percent"]
        ),
        "acceptance_metric": ACCEPTANCE_METRIC,
        "maximum_eer_gap": MAX_EER_GAP_TO_ARCFACE,
        "passed": bool(eer_gap <= MAX_EER_GAP_TO_ARCFACE),
    }
