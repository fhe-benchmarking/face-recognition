#!/usr/bin/env python3
"""
client_decrypt_decode.py — Decrypt score ciphertexts.

Initializes orion with io_mode=load (loads SK from secret_key/sk.h5), then decrypts
each p{i:04d}_score.h5 and writes one similarity float per line to
encrypted_model_predictions.txt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    parse_stage_args,
    init_orion_scheme, validate_key_cache,
)


def main():
    size, cfg, params = parse_stage_args(resolve_checkpoint=False)

    # Initialize orion with SK (io_mode=load) — only SK is needed for decryption.
    from orion.core import scheme as _scheme
    from orion.backend.python.tensors import CipherTensor

    print("[client_decrypt_decode] init_scheme(io_mode=load)...", flush=True)
    validate_key_cache(cfg, params, include_secret=True)
    init_orion_scheme(cfg, params, "load", "none")

    download_dir = params.iodir() / "ciphertexts_download"
    score_files  = sorted(download_dir.glob("p????_score.h5"))

    if not score_files:
        print(f"[client_decrypt_decode] ERROR: no score files in {download_dir}", flush=True)
        sys.exit(1)
    indices = [int(path.name[1:5]) for path in score_files]
    expected = list(range(params.get_batch_size()))
    if indices != expected:
        raise ValueError(
            f"Encrypted score indices must be exactly 0..{len(expected) - 1}"
        )

    out_path = params.get_encrypted_model_predictions_file()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for score_path in score_files:
            try:
                ctxt = CipherTensor.load(_scheme, score_path)
                sim  = float(ctxt.decrypt().decode().flatten()[0])
            except Exception as e:
                print(f"[client_decrypt_decode] ERROR decrypting {score_path.name}: {e}", flush=True)
                raise
            f.write(f"{sim:.6f}\n")

    print(f"[client_decrypt_decode] Decrypted {len(score_files)} scores → {out_path}", flush=True)


if __name__ == "__main__":
    main()
