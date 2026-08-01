#!/usr/bin/env python3
"""Generate client keys from the public circuit manifest, without model access."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    parse_stage_args, init_orion_scheme, load_circuit_manifest,
    sha256_file, write_cache_manifest,
)


def main():
    size, cfg, params = parse_stage_args(resolve_checkpoint=False)
    t0 = time.time()

    keys_dir = params.iodir() / "public_keys"
    keys_dir.mkdir(parents=True, exist_ok=True)

    (params.rootdir / "io").mkdir(parents=True, exist_ok=True)
    (params.rootdir / "io" / "current_size.txt").write_text(str(size))
    print("[keygen] Generating keys from circuit manifest...", flush=True)
    import orion
    init_orion_scheme(cfg, params, "save", "none")
    circuit_manifest = load_circuit_manifest(cfg)
    orion.generate_keys_from_manifest(circuit_manifest)
    input_level = int(circuit_manifest["circuit"]["input_level"])
    (keys_dir / "input_level.txt").write_text(str(input_level))
    metadata = {
        "circuit_manifest_sha256": sha256_file(Path(cfg["circuit_manifest"])),
        "orion_commit": cfg["orion_commit"],
        "input_level": input_level,
    }
    print("[keygen] Hashing and validating generated key artifacts...", flush=True)
    write_cache_manifest(keys_dir, ["keys.h5", "input_level.txt"], metadata)
    write_cache_manifest(
        params.iodir() / "secret_key", ["sk.h5"],
        {"orion_commit": cfg["orion_commit"]},
    )
    print(
        f"[keygen] Complete in {time.time()-t0:.1f}s "
        f"(input_level={input_level})",
        flush=True,
    )


if __name__ == "__main__":
    main()
