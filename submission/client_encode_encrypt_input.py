#!/usr/bin/env python3
"""
client_encode_encrypt_input.py — Encode and encrypt patches.

Initializes orion with io_mode=load (loads PK from keys.h5 for encryption),
reads input_level from public_keys/input_level.txt, then incrementally encodes
and encrypts patches from the indexed preprocessing artifact.

Does NOT need fit/compile/preload_all — encryption only requires the public key.
"""
import sys
import h5py
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    init_orion_scheme, load_circuit_manifest, pair_stem, parse_stage_args,
    release_file_cache, validate_key_cache,
)


def main():
    _size, cfg, params = parse_stage_args(resolve_checkpoint=False)

    import orion

    # Encryption uses the public key only — load it (not the secret key).
    validate_key_cache(cfg, params)
    init_orion_scheme(
        cfg, params, "load", "none", load_secret_key=False
    )
    input_level = int((params.iodir() / "public_keys" / "input_level.txt").read_text().strip())

    patch_store_path = params.io_intermediate_dir() / "preprocessed_patches.h5"
    if not patch_store_path.is_file():
        raise FileNotFoundError(f"Missing preprocessed patches: {patch_store_path}")
    upload_dir = params.iodir() / "ciphertexts_upload"
    upload_dir.mkdir(parents=True, exist_ok=True)

    branch_count = len(load_circuit_manifest(cfg)["circuit"]["input_shapes"])
    encrypted_count = 0
    n_pairs = params.get_batch_size()
    with h5py.File(patch_store_path, "r") as patch_store:
        patches = patch_store["patches"]
        if patches.shape[:3] != (n_pairs, 2, branch_count):
            raise ValueError(
                f"Unexpected patch store shape: {patches.shape}"
            )
        for pair_index in range(n_pairs):
            for image_index in range(2):
                for branch_index in range(branch_count):
                    tensor = torch.from_numpy(
                        patches[pair_index, image_index, branch_index]
                    )
                    ciphertext = orion.encrypt(orion.encode(tensor, input_level))
                    output_path = upload_dir / (
                        f"{pair_stem(pair_index)}_i{image_index}_"
                        f"b{branch_index}.h5"
                    )
                    ciphertext.save(output_path)
                    release_file_cache(output_path)
                    encrypted_count += 1
                    del ciphertext
    release_file_cache(patch_store_path)
    patch_store_path.unlink()

    print(
        f"[client_encode_encrypt_input] Encrypted {encrypted_count} patches "
        f"-> {upload_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
