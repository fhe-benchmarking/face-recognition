#!/usr/bin/env python3
"""
client_key_generation.py — FHE key generation + model preprocessing (client).

Runs client-side (holds the secret key): generates a fresh secret key and the
full evaluation-key set (relin + all galois/bootstrapping keys) and compiles the
model, persisting everything with io_mode=save. The secret key stays private in
secret_key/; the evaluation keys a client would upload go to public_keys/; the
plaintext model diagonals go to model_data/. The server never sees the secret
key — it loads only the evaluation keys.
"""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    parse_stage_args,
    load_detector, preprocess_one_image, build_pipeline_save,
    decode_master_image,
)


def main():
    size, cfg, params = parse_stage_args()
    t0 = time.time()

    keys_dir = params.iodir() / "public_keys"
    keys_dir.mkdir(parents=True, exist_ok=True)

    # Write current_size.txt for server_preprocess_model (which has no args)
    (params.rootdir / "io").mkdir(parents=True, exist_ok=True)
    (params.rootdir / "io" / "current_size.txt").write_text(str(size))

    dataset_path = params.rootdir / "datasets" / "face_dataset.npy"
    if not dataset_path.exists():
        print(f"[client_key_generation] ERROR: master dataset not found: {dataset_path}", flush=True)
        sys.exit(1)

    print("[client_key_generation] Loading master dataset for fit sample...", flush=True)
    dataset = np.load(dataset_path, allow_pickle=True)
    # orion.fit() only needs the tensor shape, not specific values; one image is sufficient.
    # Master dataset stores JPEG bytes (see decode_master_image); decode to (3, H, W) uint8 RGB.
    img0 = decode_master_image(dataset[0][0])  # first image of first pair

    print("[client_key_generation] Detecting + aligning face for fit sample...", flush=True)
    detector = load_detector()
    patches = preprocess_one_image(detector, img0, cfg["input_size"])

    fit_arr = np.stack([p.numpy() for p in patches], axis=0)  # (N, 1, 3, 32, 32)
    fit_path = keys_dir / "fit_sample.npy"
    np.save(fit_path, fit_arr)
    print(f"[client_key_generation] fit_sample.npy saved ({len(patches)} patches) → {fit_path}", flush=True)

    # Generate secret key + evaluation keys and compile the model (io_mode=save):
    # sk -> secret_key/, evaluation keys -> public_keys/, diagonals -> model_data/.
    t_keygen = time.time()
    input_level = build_pipeline_save(cfg, params)
    (keys_dir / "input_level.txt").write_text(str(input_level))
    elapsed_keygen = time.time() - t_keygen

    elapsed = time.time() - t0
    print(f"[client_key_generation] keys + model compiled in {elapsed_keygen:.1f}s  "
          f"input_level={input_level}  total={elapsed:.1f}s", flush=True)
    print(f"[client_key_generation] Eval keys → {keys_dir / 'keys.h5'}", flush=True)


if __name__ == "__main__":
    main()
