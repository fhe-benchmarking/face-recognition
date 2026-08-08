# Measurements

`harness/run_submission.py` writes one JSON measurement file per run into a
sub-directory named for the instance size: `single`, `small`, `medium`, or
`large`.

Running with `--num_runs <n>` produces `results-1.json` … `results-<n>.json` in
the corresponding sub-directory.

This repository includes one end-to-end validated single-pair run and three
validated runs for each batched size. Formal batched reporting uses the average
of those three runs; the single-pair variant is a one-run smoke test.

## Submitting

Before submitting, run the single-pair smoke test once and each batched variant
three times, then commit the resulting files to your fork:

```console
uv run python harness/run_submission.py 0                # 1 pair
uv run python harness/run_submission.py 1 --num_runs 3   # 128 pairs
uv run python harness/run_submission.py 2 --num_runs 3   # 256 pairs
uv run python harness/run_submission.py 3 --num_runs 3   # 1024 pairs
```

The average of the three batched runs is the number reported for each batched
submission.

## File schema

Each `results-*.json` follows the FHE-benchmarking measurement schema:

- **`Timing`** — wall-clock latency per stage (e.g. `Key Generation`,
  `Encrypted model preprocessing`, `Input encryption`, `Encrypted computation`,
  `Result decryption`, …), plus `Total`.
- **`Bandwidth`** — sizes of `Public and evaluation keys`, `Packed model
  weights`, `Encrypted input`, and `Encrypted results`.
- **`Quality`** — for single inference, the encrypted similarity `score` and
  ground-truth `label`; for batched inference, `Encrypted model quality` and
  `Harness model quality`, each reporting `eer`, `tar_at_far_1pct`, and
  `tar_at_far_01pct`.
- **`Server Reported`** — the server's own timing, isolating the pure
  `Encrypted computation` from pipeline/key-loading setup and reporting the
  backbone, normalization, and inner-product breakdown.

The `Quality` block also contains the encrypted-minus-ArcFace metric gaps. A
batched run passes the quality criterion when its encrypted EER is no more than
0.15 above the ArcFace EER on the same sampled pairs.

Published workload pages and result tables live in the separate
[`fhe-benchmarking.github.io`](https://github.com/fhe-benchmarking/fhe-benchmarking.github.io/tree/main/face-recognition)
repository.
