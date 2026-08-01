#!/usr/bin/env python3
"""
client_encode_encrypt_input.py — Encode and encrypt patches.

Initializes orion with io_mode=load (loads PK from keys.h5 for encryption),
reads input_level from public_keys/input_level.txt, then encodes and encrypts
each patch .npy file into Orion's versioned HDF5 ciphertext format.

Does NOT need fit/compile/preload_all — encryption only requires the public key.
"""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    parse_stage_args,
    init_orion_scheme, load_circuit_manifest, validate_key_cache,
)


def main():
    size, cfg, params = parse_stage_args(resolve_checkpoint=False)

    import orion

    # Encryption uses the public key only — load it (not the secret key).
    validate_key_cache(cfg, params)
    init_orion_scheme(
        cfg, params, "load", "none", load_secret_key=False
    )
    input_level = int((params.iodir() / "public_keys" / "input_level.txt").read_text().strip())

    inter_dir = params.io_intermediate_dir()  # io/<size>/intermediate/
    upload_dir = params.iodir() / "ciphertexts_upload"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Find all patch files, sorted for deterministic ordering.
    # b[0-9]* matches only numeric branch indices (avoids temp files).
    patch_files = sorted(inter_dir.glob("p????_i?_b[0-9]*.npy"))
    branch_count = len(load_circuit_manifest(cfg)["circuit"]["input_shapes"])
    expected_names = {
        f"p{pair:04d}_i{image}_b{branch}.npy"
        for pair in range(params.get_batch_size())
        for image in range(2)
        for branch in range(branch_count)
    }
    actual_names = {path.name for path in patch_files}
    if actual_names != expected_names:
        raise ValueError(
            f"Plaintext patch set mismatch: "
            f"{len(expected_names - actual_names)} missing, "
            f"{len(actual_names - expected_names)} unexpected"
        )

    count = 0
    for patch_path in patch_files:
        arr = np.load(patch_path)       # (1, 3, 32, 32) float32
        tensor = torch.from_numpy(arr)
        ctxt = orion.encrypt(orion.encode(tensor, input_level))
        out_path = upload_dir / (patch_path.stem + ".h5")
        ctxt.save(out_path)
        count += 1

    if count == 0:
        print(f"[client_encode_encrypt_input] ERROR: no patch files found in {inter_dir}", flush=True)
        sys.exit(1)
    print(f"[client_encode_encrypt_input] Encrypted {count} patches → {upload_dir}", flush=True)


if __name__ == "__main__":
    main()
