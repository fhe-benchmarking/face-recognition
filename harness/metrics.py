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

import math
import sqlite3
import tempfile
from itertools import zip_longest
from pathlib import Path

# EER is stable on every batched variant and does not require selecting an
# operating threshold from the test set. The encrypted model may trail the
# included ArcFace baseline by at most fifteen percentage points of absolute EER.
ACCEPTANCE_METRIC = "eer_gap_to_arcface"
MAX_EER_GAP_TO_ARCFACE = 0.15

def calculate_face_metrics(
    gt_labels_file: Path,
    scores_file: Path,
    tag: str,
    expected_count: int | None = None,
) -> dict:
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
    def values(path, conversion):
        with Path(path).open() as stream:
            for line in stream:
                if line.strip():
                    yield conversion(line.strip())

    marker = object()
    temporary = tempfile.NamedTemporaryFile(
        prefix="face-metrics-", suffix=".sqlite3", delete=False,
    )
    database_path = Path(temporary.name)
    temporary.close()
    connection = sqlite3.connect(database_path)
    count = n_pos = 0
    first = None
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("CREATE TABLE pairs (score REAL NOT NULL, label INTEGER NOT NULL)")
        batch = []
        for label, score in zip_longest(
            values(gt_labels_file, int), values(scores_file, float),
            fillvalue=marker,
        ):
            if label is marker or score is marker:
                raise ValueError(f"[harness] {tag}: label/score count mismatch")
            if label not in (0, 1):
                raise ValueError(
                    f"[harness] {tag}: labels must be 0 or 1, got {label}"
                )
            if not math.isfinite(score):
                raise ValueError(f"[harness] {tag}: scores contain non-finite values")
            pair = (score, label)
            first = first or pair
            batch.append(pair)
            count += 1
            n_pos += label
            if len(batch) == 10_000:
                connection.executemany("INSERT INTO pairs VALUES (?, ?)", batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO pairs VALUES (?, ?)", batch)
        connection.commit()

        if expected_count is not None and count != expected_count:
            raise ValueError(
                f"[harness] {tag}: expected {expected_count} scores, got {count}"
            )
        if count == 0:
            print(f"[harness] {tag}: no label/score pairs found")
            return {}
        if count == 1:
            score, label = first
            print(f"[harness] {tag}: score={score:.6f}  label={label}")
            return {"score": score, "label": label}

        n_neg = count - n_pos
        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                f"[harness] {tag}: EER/TAR require both classes, got "
                f"{n_pos} genuine and {n_neg} impostor pairs"
            )

        true_positives = false_positives = 0
        previous_fpr, previous_difference = 0.0, -1.0
        eer = None
        tar_1pct = tar_01pct = 0.0
        query = """
            SELECT score, SUM(label), COUNT(*)
            FROM pairs
            GROUP BY score
            ORDER BY score DESC
        """
        for _score, positives, group_count in connection.execute(query):
            true_positives += positives
            false_positives += group_count - positives
            tpr = true_positives / n_pos
            fpr = false_positives / n_neg
            difference = fpr - (1.0 - tpr)
            if fpr <= 0.01:
                tar_1pct = max(tar_1pct, tpr)
            if fpr <= 0.001:
                tar_01pct = max(tar_01pct, tpr)
            if eer is None and difference >= 0.0:
                weight = -previous_difference / (
                    difference - previous_difference
                )
                eer = previous_fpr + weight * (fpr - previous_fpr)
            previous_fpr = fpr
            previous_difference = difference
        if eer is None:
            eer = previous_fpr
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)

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
