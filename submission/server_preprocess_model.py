#!/usr/bin/env python3
"""Server-owned checkpoint compilation and persistent packed-model caching."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    build_pipeline_server_save, get_face_params, get_repo_root,
    get_server_model_dir, load_submission_config, mute_logs, sha256_file,
    validate_cache_manifest, validate_key_cache, write_cache_manifest,
    write_server_model_reference,
)


def _dataset_provenance(params, cfg):
    """Hash the indexed benchmark source, with legacy source hashes when present."""
    dataset_dir = params.rootdir / "datasets"
    indexed_path = dataset_dir / "face_dataset.h5"
    value = {
        "hf_repo": cfg["dataset_hf_repo"],
        "indexed_data_sha256": sha256_file(indexed_path),
    }
    legacy_data = dataset_dir / "face_dataset.npy"
    legacy_labels = dataset_dir / "face_dataset_labels.txt"
    if legacy_data.is_file():
        value["legacy_data_sha256"] = sha256_file(legacy_data)
    if legacy_labels.is_file():
        value["legacy_labels_sha256"] = sha256_file(legacy_labels)
    return value


def main():
    mute_logs()
    t0 = time.time()
    cfg = load_submission_config(resolve_checkpoint=True)

    size_file = get_repo_root() / "io" / "current_size.txt"
    if not size_file.exists():
        print(f"[server_preprocess_model] ERROR: {size_file} not found. "
              f"Run client_key_generation first.", flush=True)
        sys.exit(1)
    size = int(size_file.read_text().strip())
    params = get_face_params(size)
    validate_key_cache(cfg, params)
    model_dir, identity = get_server_model_dir(cfg, params)
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_hit = False
    try:
        cache = validate_cache_manifest(model_dir, verify_hashes=True)
        cache_hit = cache["metadata"].get("identity") == identity
    except (FileNotFoundError, ValueError):
        cache = None
    if not cache_hit:
        stale = model_dir / "cache_manifest.json"
        if stale.exists():
            stale.unlink()
        print("[model-prep] Compiling checkpoint into server-owned diagonals...", flush=True)
        input_level = build_pipeline_server_save(cfg, params, model_dir)
        cache = write_cache_manifest(
            model_dir,
            ["diagonals.h5"],
            {"identity": identity, "input_level": input_level},
        )
    else:
        input_level = int(cache["metadata"]["input_level"])
        print("[model-prep] Validated persistent packed-model cache.", flush=True)

    client_level = int(
        (params.iodir() / "public_keys" / "input_level.txt").read_text()
    )
    if input_level != client_level:
        raise ValueError(
            f"Client/server input-level mismatch: {client_level} != {input_level}"
        )
    write_server_model_reference(params, model_dir, cache)
    (params.iodir() / "submission_reported.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "Bandwidth": {
                    "Packed model weights": cache["files"]["diagonals.h5"]["size_bytes"],
                },
            },
            indent=2,
        )
        + "\n"
    )

    provenance = {
        "schema_version": 1,
        "model": {
            "hf_repo": cfg["ckpt_hf_repo"],
            "hf_file": cfg["ckpt_hf_file"],
            "sha256": identity["checkpoint_sha256"],
        },
        "dataset": _dataset_provenance(params, cfg),
        "orion_commit": cfg["orion_commit"],
        "orion_config_sha256": identity["orion_config_sha256"],
        "circuit_manifest_sha256": identity["circuit_manifest_sha256"],
        "pair_slots": cfg["pair_slots"],
        "stage_chunk_pairs": cfg["stage_chunk_pairs"],
        "aggregator_max_pairs": cfg["aggregator_max_pairs"],
        "packed_model_sha256": cache["files"]["diagonals.h5"]["sha256"],
        "packed_model_size_bytes": cache["files"]["diagonals.h5"]["size_bytes"],
    }
    (params.iodir() / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(
        f"[model-prep] Ready in {time.time()-t0:.1f}s "
        f"(cache={'hit' if cache_hit else 'created'})",
        flush=True,
    )


if __name__ == "__main__":
    main()
