#!/usr/bin/env python3
"""
generate_input.py - Sample face pairs for one benchmark run.

Samples batch_size row indices from the indexed face dataset. Input-store
materialization is separate so selection never decodes the source images.

Output format: test_selection.npz plus test_labels.txt.
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
import numpy as np
import h5py
from face_dataset_store import STORE_NAME
from utils import parse_submission_arguments


def main():
    _size, params, seed, _ = parse_submission_arguments(
        'Generate face pairs for FHE benchmark.'
    )

    store_path = params.rootdir / "datasets" / STORE_NAME
    if not store_path.exists():
        sys.exit(f"[harness] Error: indexed dataset not found: {store_path}")
    with h5py.File(store_path, "r") as store:
        n_total = len(store["labels"])
    batch_size = params.get_batch_size()

    if batch_size > n_total:
        sys.exit(f"[harness] Error: batch_size={batch_size} exceeds dataset size={n_total}")

    rng     = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=batch_size, replace=False)

    # HDF5 fancy indexing requires increasing indices. Read only selected
    # labels, then restore the randomized benchmark order.
    order = np.argsort(indices)
    with h5py.File(store_path, "r") as store:
        sorted_labels = store["labels"][indices[order]]
    labels = np.empty(batch_size, dtype=np.int8)
    labels[order] = sorted_labels

    out_dir = params.dataset_intermediate_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(out_dir / "test_selection.npz", indices=indices)
    (out_dir / "test_labels.txt").write_text(
        "\n".join(str(int(label)) for label in labels) + "\n"
    )

    seed_str = str(seed) if seed is not None else "random"
    print(
        f"[harness] Selected {batch_size} face pairs "
        f"(seed={seed_str}) → {out_dir / 'test_selection.npz'}"
    )


if __name__ == "__main__":
    main()
