#!/usr/bin/env python3
"""
generate_dataset.py - Provision and validate the face pair dataset.

The current benchmark dataset (face_dataset.npy + face_dataset_labels.txt) is
hosted on Hugging Face (halmsu/celeba-1024-pairs). It is migrated once to an
indexed face_dataset.h5 store. A pre-indexed store can instead be supplied
directly for datasets too large to load as a legacy object array.

To run fully offline, place the two files next to the given path beforehand.

Usage:  python3 generate_dataset.py <dataset_npy_path>
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
import shutil
from pathlib import Path
from face_dataset_store import (
    STORE_NAME, decode_image, ensure_pair_store, validate_pair_store,
)

# Hugging Face dataset repo hosting the benchmark face pairs.
HF_DATASET_REPO = "halmsu/celeba-1024-pairs"
DATASET_NPY     = "face_dataset.npy"
DATASET_LABELS  = "face_dataset_labels.txt"


def _download_from_hf(dest_dir: Path):
    """Download face_dataset.npy + labels from Hugging Face into dest_dir."""
    from huggingface_hub import hf_hub_download
    dest_dir.mkdir(parents=True, exist_ok=True)
    for fname in (DATASET_NPY, DATASET_LABELS):
        target = dest_dir / fname
        if target.exists():
            continue
        print(f"[harness] Downloading {fname} from HF dataset {HF_DATASET_REPO} ...",
              flush=True)
        cached = hf_hub_download(repo_id=HF_DATASET_REPO, filename=fname,
                                 repo_type="dataset")
        shutil.copyfile(cached, target)


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: generate_dataset.py <dataset_npy_path>")

    npy_path    = Path(sys.argv[1])
    labels_path = npy_path.parent / DATASET_LABELS
    store_path = npy_path.parent / STORE_NAME

    # Provision the legacy hosted files only when no indexed store is present.
    if not store_path.exists() and (not npy_path.exists() or not labels_path.exists()):
        try:
            _download_from_hf(npy_path.parent)
        except Exception as e:
            sys.exit(f"[harness] Error: dataset not found locally and Hugging Face "
                     f"download failed: {e}")

    try:
        if not store_path.exists():
            store_path = ensure_pair_store(npy_path, labels_path)
        pair_count, n_same = validate_pair_store(store_path)
        import h5py
        with h5py.File(store_path, "r") as store:
            example_shape = decode_image(store["image0"][0]).shape
    except (OSError, ValueError, KeyError) as exc:
        sys.exit(f"[harness] Error: invalid face dataset: {exc}")

    n_diff = pair_count - n_same
    print(f"[harness] Face dataset: {pair_count} pairs  example_img_shape={example_shape}  "
          f"same={n_same}  diff={n_diff}")


if __name__ == "__main__":
    main()
