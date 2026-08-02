#!/usr/bin/env python3
"""
client_decrypt_decode.py — Decrypt score ciphertexts.

Initializes orion with io_mode=load (loads SK from secret_key/sk.h5), then decrypts
each globally indexed score ciphertext and writes one similarity float per line to
encrypted_model_predictions.txt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    init_orion_scheme, pair_stem, parse_stage_args, release_file_cache,
    validate_key_cache,
)


def main():
    _size, cfg, params = parse_stage_args(resolve_checkpoint=False)

    # Initialize orion with SK (io_mode=load) — only SK is needed for decryption.
    from orion.core import scheme as _scheme
    from orion.backend.python.tensors import CipherTensor

    print("[client_decrypt_decode] init_scheme(io_mode=load)...", flush=True)
    validate_key_cache(cfg, params, include_secret=True)
    init_orion_scheme(cfg, params, "load", "none")

    download_dir = params.iodir() / "ciphertexts_download"
    out_path = params.get_encrypted_model_predictions_file()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(".txt.tmp")
    n_pairs = params.get_batch_size()
    with temporary.open("w") as output:
        for pair_index in range(n_pairs):
            score_path = download_dir / f"{pair_stem(pair_index)}_score.h5"
            if not score_path.is_file():
                raise FileNotFoundError(f"Missing encrypted score: {score_path}")
            ciphertext = CipherTensor.load(_scheme, score_path)
            release_file_cache(score_path)
            score = float(ciphertext.decrypt().decode().flatten()[0])
            output.write(f"{score:.6f}\n")
            del ciphertext
    temporary.replace(out_path)

    print(
        f"[client_decrypt_decode] Decrypted {n_pairs} scores -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
