# FHE Benchmarking Suite - Face Verification

This repository contains the harness for the face verification workload of the FHE benchmarking suite of [HomomorphicEncryption.org].

The repository includes a CryptoFace reference submission under `submission/`,
implemented with Orion and RNS-CKKS.

Submitters clone this repository and replace the contents of `submission/` with their own implementation. The stage scripts must accept a single positional argument (instance size 0–3) and follow the file I/O contract described below.

## Prerequisites

### System dependencies

`scripts/install_system_deps.sh` installs the OS-level packages via `apt`:

- `build-essential`, `python3-dev` — compilers/headers for building native extensions
- `golang-go` — required by Orion's build
- `libgraphviz-dev` — pipeline graph visualization
- `libgl1-mesa-glx`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender-dev` — OpenCV/OpenGL runtime libs (InsightFace)

```console
bash scripts/install_system_deps.sh
```

### Python dependencies

`requirements.txt` contains the harness and submission dependencies. Orion is
pinned to the revision used by the validated CryptoFace environment.

```console
bash scripts/install_python_deps.sh
```

### Dataset and model (Hugging Face)

The benchmark dataset and the reference model checkpoint are hosted on Hugging
Face and downloaded automatically on the first run (cached under
`~/.cache/huggingface`). No manual step is required — `huggingface_hub` is
installed with the Python dependencies above.

| Artifact | Hugging Face repo | Pulled by |
|---|---|---|
| `face_dataset.npy` + `face_dataset_labels.txt` | [`halmsu/celeba-1024-pairs`](https://huggingface.co/datasets/halmsu/celeba-1024-pairs) (dataset) | harness `generate_dataset.py` → `datasets/` |
| `backbone-64x64.ckpt` | [`halmsu/cryptoface-v1`](https://huggingface.co/halmsu/cryptoface-v1) (model) | reference submission `common.load_submission_config` |

The dataset contains 1,024 screened CelebA pairs: 512 genuine and 512 impostor
pairs. It preserves the original variable-size JPEG images. The input generator
decodes them without resizing, and the reference submission performs face
detection, landmark alignment, cropping, and only then resizes the aligned crop
for CryptoFace. The four benchmark variants sample 1, 128, 256, or all 1,024
pairs from this master set.

**Offline / local override.** To run without network access, place the two
dataset files in `datasets/` and the checkpoint at the `ckpt_path` in
`submission/config.yml` (default `submission/checkpoints/backbone-64x64.ckpt`);
existing local files are always used in preference to the download.

## Running the benchmark

```console
uv run python harness/run_submission.py -h
```

```
usage: run_submission.py [-h] [--num_runs NUM_RUNS] [--seed SEED]
                         [--clrtxt CLRTXT]
                         {0,1,2,3}

Run Face Verification FHE benchmark.

positional arguments:
  {0,1,2,3}            Instance size (0-single/1-small/2-medium/3-large)

options:
  --num_runs NUM_RUNS  Number of times to run stages 4-10 (default: 1)
  --seed SEED          Random seed for reproducible pair sampling
  --clrtxt CLRTXT      Set to 1 to force rerun of cleartext reference
```

### Example: single-pair smoke test

```console
uv run python harness/run_submission.py 0 --seed 42
```

### Example: small size, two runs

```console
uv run python harness/run_submission.py 1 --seed 3 --num_runs 2
```

The four variants contain 1, 128, 256, and 1024 face pairs. Batched variants
report EER and TAR at FAR=1%/0.1% for both the encrypted CryptoFace model and
the included ArcFace baseline, together with their paired metric differences.

Results are written to `measurements/` as JSON files (`results-1.json`, `results-2.json`, …).

## Pipeline stages

The harness drives the following sequence. Stages 2, 3, and 5–9 invoke the submission's scripts via `utils.run_exe_or_python()`, which runs `submission/<stage>.py` if present, otherwise `submission/build/<stage>`.

| Stage | Script | Description |
|-------|--------|-------------|
| 0 | harness | Remove and re-create `io/<size>/` |
| 1 | harness | Download (from Hugging Face if absent) and validate `datasets/face_dataset.npy` |
| 2 | submission | `client_key_generation` — generate CKKS keys and persist circuit-specific evaluation keys/model data |
| 3 | submission | `server_preprocess_model` — validate the persisted input level |
| 4 | harness | `generate_input.py` — sample face pairs into `datasets/<size>/intermediate/` |
| 5 | submission | `client_preprocess_input` — face alignment and patch extraction |
| 6 | submission | `client_encode_encrypt_input` — encode and encrypt patches |
| 7 | submission | `server_encrypted_compute` - five-slot encrypted face verification |
| 8 | submission | `client_decrypt_decode` — decrypt similarity scores |
| 9 | submission | `client_postprocess` — optional postprocessing |
| 10 | harness | ArcFace baseline, EER/TAR@FAR metrics, and paired comparison |

Stages 4–10 repeat for each `--num_runs` iteration.

## File I/O contract

| Path | Written by | Read by |
|------|-----------|---------|
| `datasets/face_dataset.npy` | Hugging Face (`halmsu/celeba-1024-pairs`) | harness stage 1, 4 |
| `datasets/<size>/intermediate/test_pairs.npz` | harness stage 4 | submission stage 5 |
| `datasets/<size>/intermediate/test_labels.txt` | harness stage 4 | harness stage 10 |
| `io/<size>/secret_key/sk.h5` | submission stage 2 | client stage 8 only |
| `io/<size>/public_keys/keys.h5` | submission stage 2 | submission stages 6, 7, 8 |
| `io/<size>/model_data/diagonals.h5` | submission stage 2 | server stage 7 |
| `io/<size>/public_keys/fit_sample.npy` | submission stage 2 | submission stage 7 |
| `io/<size>/public_keys/input_level.txt` | submission stage 2 (validated by stage 3) | submission stage 6 |
| `io/<size>/intermediate/*.npy` | submission stage 5 | submission stage 6 |
| `io/<size>/ciphertexts_upload/*.bin` | submission stage 6 | submission stage 7 |
| `io/<size>/ciphertexts_download/*.bin` | submission stage 7 | submission stage 8 |
| `io/<size>/encrypted_model_predictions.txt` | submission stage 8 | harness stage 10 |
| `io/<size>/harness_model_predictions.txt` | harness stage 10 | harness stage 10 |

## Directory structure

```
├── README.md
├── LICENSE.md
├── harness/
│   ├── run_submission.py       # Main harness orchestrator
│   ├── params.py               # InstanceParams and batch sizes
│   ├── utils.py                # Logging, timing, run_exe_or_python
│   ├── metrics.py              # EER and TAR@FAR calculation
│   ├── generate_dataset.py     # Download (from HF) + validate dataset
│   ├── generate_input.py       # Sample face pairs per run
│   ├── cleartext_impl.py       # ArcFace plaintext reference
│   └── verify_result.py        # Standalone metric verification
├── datasets/                   # Populated on first run from HF (halmsu/celeba-1024-pairs)
│   ├── face_dataset.npy        # Benchmark dataset (1024 CelebA pairs)
│   └── face_dataset_labels.txt # Ground-truth labels (0=different, 1=same)
├── submission/                 # Reference submission (CryptoFace)
│   ├── config.yml
│   ├── common.py
│   ├── client_key_generation.py
│   ├── server_preprocess_model.py
│   ├── client_preprocess_input.py
│   ├── client_encode_encrypt_input.py
│   ├── server_encrypted_compute.py
│   ├── client_decrypt_decode.py
│   ├── client_postprocess.py
│   ├── models/
│   ├── utils/
│   ├── checkpoints/            # backbone-64x64.ckpt downloaded from HF (halmsu/cryptoface-v1)
│   └── orion_configs/
├── scripts/
│   ├── install_system_deps.sh
│   └── install_python_deps.sh
├── io/                         # Client↔server communication (generated)
└── measurements/               # Per-run JSON results (generated)
```
