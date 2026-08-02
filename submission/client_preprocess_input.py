#!/usr/bin/env python3
"""
client_preprocess_input.py — Face detection, alignment, and patch extraction.

Reads the indexed test-pair store, runs InsightFace detection and alignment,
and writes patches incrementally to one indexed HDF5 artifact.
"""
import sys
import h5py
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    decode_image, load_circuit_manifest, load_detector, parse_stage_args,
    preprocess_one_image, release_file_cache,
)


def main():
    _size, cfg, params = parse_stage_args(resolve_checkpoint=False)

    pairs_path = params.get_test_input_file()
    if not pairs_path.exists():
        print(f"[client_preprocess_input] ERROR: {pairs_path} not found", flush=True)
        sys.exit(1)

    output_path = params.io_intermediate_dir() / "preprocessed_patches.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".h5.tmp")
    n_pairs = params.get_batch_size()
    branch_shapes = load_circuit_manifest(cfg)["circuit"]["input_shapes"]
    if not branch_shapes or any(shape != branch_shapes[0] for shape in branch_shapes):
        raise ValueError("All CryptoFace branch input shapes must match")
    branch_count = len(branch_shapes)
    patch_shape = tuple(branch_shapes[0])
    print(f"[client_preprocess_input] {n_pairs} pairs -> {output_path}", flush=True)
    detector = load_detector()

    with h5py.File(pairs_path, "r") as pairs, h5py.File(temporary, "w") as output:
        if not {"image0", "image1"}.issubset(pairs):
            raise ValueError(f"Input store is missing image datasets: {pairs_path}")
        if len(pairs["image0"]) != n_pairs or len(pairs["image1"]) != n_pairs:
            raise ValueError(
                f"Expected {n_pairs} input pairs, found "
                f"{len(pairs['image0'])}/{len(pairs['image1'])}"
            )
        patches_store = output.create_dataset(
            "patches",
            shape=(n_pairs, 2, branch_count, *patch_shape),
            dtype="float32",
            chunks=(1, 2, branch_count, *patch_shape),
        )
        for pair_index in range(n_pairs):
            for image_index in range(2):
                patches = preprocess_one_image(
                    detector,
                    decode_image(pairs[f"image{image_index}"][pair_index]),
                    cfg["input_size"],
                )
                if len(patches) != branch_count:
                    raise ValueError(
                        f"Expected {branch_count} patches, got {len(patches)}"
                    )
                for branch_index, patch in enumerate(patches):
                    patches_store[pair_index, image_index, branch_index] = patch.numpy()
        output.attrs["pair_count"] = n_pairs
        output.attrs["branch_count"] = branch_count
    temporary.replace(output_path)
    release_file_cache(pairs_path)
    release_file_cache(output_path)

    print(
        f"[client_preprocess_input] Saved {n_pairs * 2 * branch_count} patches",
        flush=True,
    )


if __name__ == "__main__":
    main()
