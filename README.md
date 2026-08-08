# FHE Benchmarking Suite - Face Verification

This repository contains the harness for the face-verification workload of the
FHE benchmarking suite from
[HomomorphicEncryption.org](https://homomorphicencryption.org/).

The repository includes a
[CryptoFace](https://openaccess.thecvf.com/content/CVPR2025/html/Ao_CryptoFace_End-to-End_Encrypted_Face_Recognition_CVPR_2025_paper.html)
reference submission under `submission/`, implemented with
[Orion](https://github.com/vboddeti/orion) and
[RNS-CKKS](https://eprint.iacr.org/2018/931).

Submitters clone this repository and replace the contents of `submission/` with
their own implementation. This README defines the normative workload interface.

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

The setup uses Python 3.12 and [`uv`](https://docs.astral.sh/uv/). Install
[`uv`](https://docs.astral.sh/uv/getting-started/installation/) and ensure it
is on `PATH` before running the installation script.

`requirements.txt` contains the harness and reference-submission dependencies.
Orion is pinned to the revision used by the validated CryptoFace environment.

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
| `face_dataset.npy` + `face_dataset_labels.txt` | [`halmsu/celeba-1024-pairs`](https://huggingface.co/datasets/halmsu/celeba-1024-pairs) (dataset) | harness `generate_dataset.py` -> indexed `datasets/face_dataset.h5` |
| `backbone-64x64.ckpt` | [`halmsu/cryptoface-v1`](https://huggingface.co/halmsu/cryptoface-v1) (model) | reference submission `common.load_submission_config` |

The dataset contains 1,024 screened CelebA pairs: 512 genuine and 512 impostor
pairs. It preserves the original variable-size JPEG images. The input generator
decodes them without resizing, and the reference submission performs face
detection, landmark alignment, cropping, and only then resizes the aligned crop
for CryptoFace. The four benchmark variants sample 1, 128, 256, or all 1,024
pairs from this master set.

Evaluation uses `datasets/face_dataset.h5` as a random-access store with
`image0`, `image1`, and `labels` datasets and schema version 1. The current
hosted NPY dataset is migrated once on first use. Larger datasets should be
provided directly in this indexed format; the harness then reads only selected
labels and the current image chunk instead of loading the source dataset.

**Offline / local override.** To run without network access, place either an
indexed `face_dataset.h5` or the two legacy dataset files in `datasets/`, and
place the checkpoint at the `ckpt_path` in `submission/config.yml` (default
`submission/checkpoints/backbone-64x64.ckpt`). Existing local files are used
in preference to the download.

## Running the benchmark

For every submission-owned stage, the harness runs the first available entry
point: `submission/<stage>.py` with the active Python interpreter, or
`submission/build/<stage>` as a native executable. Commands run from the
repository root. Stages that take a size argument receive `0`, `1`, `2`, or
`3`, corresponding to 1, 128, 256, or 1,024 pairs. A stage must exit nonzero
on failure and must finish writing its output before it reports success.

```console
uv run python harness/run_submission.py -h
```

```
usage: run_submission.py [-h] [--num_runs NUM_RUNS] [--seed SEED]
                         {0,1,2,3}

Run Face Verification FHE benchmark.

positional arguments:
  {0,1,2,3}            Instance size (0-single/1-small/2-medium/3-large)

options:
  --num_runs NUM_RUNS  Number of times to run stages 4-10 (default: 1)
  --seed SEED          Random seed for reproducible pair sampling (default: 42).
                       Fixed by default so all submissions sample identical pairs.
```

### Example: single-pair smoke test

```console
uv run python harness/run_submission.py 0 --seed 42
```

### Example: small size, two runs

```console
uv run python harness/run_submission.py 1 --seed 3 --num_runs 2
```

The four variants contain 1, 128, 256, and 1,024 face pairs. Batched variants
report EER and TAR at FAR=1%/0.1% for both the encrypted CryptoFace model and
the included [ArcFace](https://arxiv.org/abs/1801.07698) baseline, together
with their paired metric differences.

Results are written to `measurements/` as JSON files (`results-1.json`,
`results-2.json`, …).

## Pipeline stages

The harness drives the following sequence. It owns stages 0, 1, 4, and 10;
stages 2, 3, and 5–9 belong to the submission.

| Stage | Script | Description |
|-------|--------|-------------|
| 0 | harness | Remove and re-create `io/<size>/` |
| 1 | harness | Provision and validate indexed `datasets/face_dataset.h5` |
| 2 | submission | `client_key_generation` — generate client secret and public/evaluation key material |
| 3 | submission | `server_preprocess_model` — compile/cache packed model weights and validate the persisted input level |
| 4 | harness | `generate_input.py` — sample row indices and create an indexed input store |
| 5 | submission | `client_preprocess_input` — face alignment and patch extraction |
| 6 | submission | `client_encode_encrypt_input` — encode and encrypt patches |
| 7 | submission | `server_encrypted_compute` — bounded parallel encrypted face verification |
| 8 | submission | `client_decrypt_decode` — decrypt similarity scores |
| 9 | submission | `client_postprocess` — optional postprocessing |
| 10 | harness | ArcFace baseline, EER/TAR@FAR metrics, and paired comparison |

Stages 4–10 repeat for each `--num_runs` iteration.

The required final output is
`io/<size>/encrypted_model_predictions.txt`: exactly one finite floating-point
similarity score per input pair, in input order, with one score per line and a
final newline. A submission may choose its other intermediate filenames under
`io/<size>/`.

Client secret material must not be loaded by stages 3 or 7. Cleartext images,
plaintext features, and decrypted scores must not be made available to the
server stage. Public/evaluation keys, encrypted inputs, packed model weights,
and encrypted results may cross the client/server boundary.

## File I/O contract

| Path | Written by | Read by |
|------|-----------|---------|
| `datasets/face_dataset.h5` | harness stage 1 migration, or supplied directly | random-access source for harness stage 4 |
| `datasets/<size>/intermediate/test_selection.npz` | harness stage 4 | bounded chunk materializer |
| `datasets/<size>/intermediate/test_pairs.h5` | conventional input materializer | submission stage 5 |
| `datasets/<size>/intermediate/test_labels.txt` | harness stage 4 | harness stage 10 |
| `io/<size>/secret_key/sk.h5` | submission stage 2 | client stage 8 only |
| `io/<size>/public_keys/keys.h5` | submission stage 2 | submission stages 6, 7, 8 |
| `submission/circuit_manifest.json` | submission | client stage 2, server stages 3 and 7 |
| `io/<size>/public_keys/input_level.txt` | client stage 2 | server stage 3, client stage 6 |
| `io/server_data/<cache-key>/diagonals.h5` | server stage 3 | server stage 7 |
| `io/<size>/server_model.json` | server stage 3 | server stage 7 |
| `io/<size>/submission_reported.json` | submission (optional) | harness |
| `io/<size>/server_reported.json` | submission stage 7 (optional) | harness |
| `io/<size>/provenance.json` | reference submission stage 3 | local audit only; not copied into measurement JSON |
| `io/<size>/intermediate/preprocessed_patches.h5` | submission stage 5 | submission stage 6 |
| `io/<size>/ciphertexts_upload/*.h5` | client stage 6 | server stage 7 |
| `io/<size>/ciphertexts_download/*.h5` | server stage 7 | client stage 8 |
| `io/<size>/encrypted_model_predictions.txt` | submission stage 8 or 9 | harness stage 10 |
| `io/<size>/harness_model_predictions.txt` | harness stage 10 | harness stage 10 |

The stage-4 HDF5 input contains equally sized `image0` and `image1`
variable-length `uint8` datasets. Each element is an encoded RGB image, and
rows retain benchmark input order. The labels file contains one `0` (impostor)
or `1` (genuine) label per pair in the same order.

The harness treats `submission_reported.json` generically. Its optional
`Bandwidth` object maps artifact labels to non-negative integer byte counts; it
does not interpret submission-specific paths or cache layouts. A submission may
also write `server_reported.json`, mapping timing labels to numeric seconds with
nested metadata allowed. These reports supplement rather than replace the
harness's own wall-time and artifact-size measurements.

## Performance and quality measurement

The harness measures elapsed wall-clock time between stage completion markers,
including any intervening harness artifact-size collection. Stages 1–3 run once
and form `Offline setup total`; stages 4–10 run once per requested iteration and
form `Online evaluation total`. `Timing["Total"]` is their sum. Values under
`Server Reported` are optional submission diagnostics and may include both wall
time and summed worker-seconds; they are not added to the harness total.

`Bandwidth` reports serialized artifact sizes. The harness measures public and
evaluation keys, encrypted inputs, and encrypted results from disk. Submissions
can report additional serialized artifacts, such as packed model weights,
through `submission_reported.json`.

Quality is face-verification quality, not classification accuracy. Stage 10
first verifies the exact score count and rejects non-finite scores. For each
batched variant, it evaluates the encrypted model and the included ArcFace
baseline on the same sampled pairs, then sweeps every distinct similarity
threshold over all pairs. It reports:

- equal error rate (EER), interpolated where false-accept and false-reject rates
  meet; lower is better;
- true-accept rate (TAR) at false-accept rate (FAR) at most 1% and 0.1%; higher
  is better; and
- encrypted-minus-ArcFace gaps for all three metrics.

A batched run passes when its encrypted EER is no more than 0.15 above the
ArcFace EER on the same pairs. A single-pair smoke test reports only its score
and ground-truth label because EER and TAR are not meaningful for one sample.
Formal batched benchmark numbers are the average of three runs. The single-pair
variant is a smoke test and is reported from one run, as described in
[`measurements/README.md`](measurements/README.md).

The same score validation and metric calculation are available independently:

```console
python3 harness/verify_result.py <labels-file> <scores-file> [tag]
```

## Directory structure

```
├── README.md
├── LICENSE.md
├── harness/
│   ├── run_submission.py       # Main harness orchestrator
│   ├── params.py               # InstanceParams and batch sizes
│   ├── utils.py                # Logging, timing, run_exe_or_python
│   ├── metrics.py              # EER and TAR@FAR calculation
│   ├── verify_result.py        # Standalone score/label quality verifier
│   ├── generate_dataset.py     # Download (from HF) + validate dataset
│   ├── generate_input.py       # Sample face-pair row indices per run
│   ├── face_dataset_store.py   # Indexed dataset access and legacy migration
│   ├── materialize_input_store.py # Create the indexed run input
│   └── cleartext_impl.py       # ArcFace plaintext reference
├── datasets/                   # Populated on first run from HF (halmsu/celeba-1024-pairs)
│   ├── .gitkeep                # Keep the initially empty directory in Git
│   ├── face_dataset.npy        # Downloaded legacy dataset (1,024 CelebA pairs)
│   ├── face_dataset_labels.txt # Downloaded ground-truth labels
│   └── face_dataset.h5         # Generated indexed random-access store
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
│   └── orion_configs/
├── scripts/
│   ├── install_system_deps.sh
│   └── install_python_deps.sh
├── io/                         # Client↔server communication (generated)
└── measurements/               # Per-run JSON results (generated)
```
