#!/usr/bin/env python3
"""
client_preprocess_input.py — Face detection, alignment, and patch extraction.

Reads test_pairs.npz, runs InsightFace detection+alignment on each image,
extracts 32×32 patches, saves as .npy files to io/<size>/intermediate/.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    parse_stage_args,
    load_detector, preprocess_one_image
)


def main():
    size, cfg, params = parse_stage_args(resolve_checkpoint=False)

    pairs_path = params.get_test_input_file()  # datasets/<size>/intermediate/test_pairs.npz
    if not pairs_path.exists():
        print(f"[client_preprocess_input] ERROR: {pairs_path} not found", flush=True)
        sys.exit(1)

    npz = np.load(pairs_path)
    n_pairs = len(npz.files) // 2  # each pair has img0 + img1
    if n_pairs != params.get_batch_size():
        raise ValueError(
            f"Expected {params.get_batch_size()} input pairs, found {n_pairs}"
        )

    out_dir = params.io_intermediate_dir()  # io/<size>/intermediate/
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[client_preprocess_input] {n_pairs} pairs → {out_dir}", flush=True)
    print("[client_preprocess_input] Loading face detector...", flush=True)
    detector = load_detector()
    print("[client_preprocess_input] Processing pairs...", flush=True)

    n_patches = None
    for i in range(n_pairs):
        for j in range(2):
            img = npz[f"pair_{i:05d}_img{j}"]  # (3, H, W) uint8 RGB
            patches = preprocess_one_image(detector, img, cfg["input_size"])
            if n_patches is None:
                n_patches = len(patches)
            for k, patch in enumerate(patches):
                np.save(out_dir / f"p{i:04d}_i{j}_b{k}.npy", patch.numpy())

    if n_pairs == 0:
        print("[client_preprocess_input] No pairs found — nothing to preprocess", flush=True)
    else:
        print(f"[client_preprocess_input] Saved {n_pairs * 2 * n_patches} patch files  "
              f"(n_patches={n_patches})", flush=True)


if __name__ == "__main__":
    main()
