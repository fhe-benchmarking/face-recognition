#!/usr/bin/env python3
"""Verify face-similarity scores against the ground-truth labels.

Usage: python3 verify_result.py <gt_labels_file> <scores_file> [tag]
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
from pathlib import Path

from metrics import calculate_face_metrics


def verify_result(
    gt_file: Path,
    scores_file: Path,
    tag: str,
    expected_count: int | None = None,
) -> dict:
    """Validate score output and return its face-verification metrics."""
    return calculate_face_metrics(
        gt_file,
        scores_file,
        tag,
        expected_count=expected_count,
    )


def main():
    if len(sys.argv) < 3:
        sys.exit("Usage: verify_result.py <gt_labels_file> <scores_file> [tag]")

    gt_file = Path(sys.argv[1])
    scores_file = Path(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else scores_file.stem
    verify_result(gt_file, scores_file, tag)


if __name__ == "__main__":
    main()
