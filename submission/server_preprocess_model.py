#!/usr/bin/env python3
"""
server_preprocess_model.py — server-side model preprocessing (stub).

The client (client_key_generation) generates the evaluation keys and compiles
the model with io_mode=save — that step also produces the plaintext model
diagonals. The server therefore has nothing to compute here; it only needs
input_level.txt, which the client already wrote. This stub recomputes it from
the CKKS config for robustness. The heavy work (loading the evaluation keys and
diagonals) happens once in server_encrypted_compute via io_mode=load, without
ever touching the secret key.
"""
import sys
import time
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import load_submission_config, get_face_params, get_repo_root, mute_logs


def main():
    mute_logs()
    t0 = time.time()
    cfg = load_submission_config()

    size_file = get_repo_root() / "io" / "current_size.txt"
    if not size_file.exists():
        print(f"[server_preprocess_model] ERROR: {size_file} not found. "
              f"Run client_key_generation first.", flush=True)
        sys.exit(1)
    size = int(size_file.read_text().strip())
    params = get_face_params(size)
    keys_dir = params.iodir() / "public_keys"

    # Derive input_level from the CKKS config: len(LogQ) - 1.
    try:
        with open(cfg["orion_config"]) as f:
            logq = yaml.safe_load(f)["ckks_params"]["LogQ"]
    except KeyError as e:
        print(f"[server_preprocess_model] ERROR: missing key {e} in {cfg['orion_config']}", flush=True)
        sys.exit(1)
    input_level = len(logq) - 1
    keys_dir.mkdir(parents=True, exist_ok=True)
    (keys_dir / "input_level.txt").write_text(str(input_level))

    print(f"[server_preprocess_model] stub: input_level={input_level}  "
          f"total={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
