# Measurements

`harness/run_submission.py` writes one JSON measurement file per run into a
sub-directory named for the instance size: `single`, `small`, `medium`, or
`large`.

Running with `--num_runs <n>` produces `results-1.json` … `results-<n>.json` in
the corresponding sub-directory.

This repository includes one end-to-end validated `results-1.json` for each of
the four sizes. Formal benchmark reporting should still use three runs as
described below.

## Submitting

Before submitting, run each variant you intend to submit with `--num_runs 3` and
commit the resulting files to your fork:

```console
uv run python harness/run_submission.py 0 --num_runs 3   # 1 pair
uv run python harness/run_submission.py 1 --num_runs 3   # 128 pairs
uv run python harness/run_submission.py 2 --num_runs 3   # 256 pairs
uv run python harness/run_submission.py 3 --num_runs 3   # 1024 pairs
```

The average of the three runs is the number reported for your submission.

## File schema

Each `results-*.json` follows the FHE-benchmarking measurement schema:

- **`Timing`** — wall-clock latency per stage (e.g. `Key Generation`,
  `Encrypted model preprocessing`, `Input encryption`, `Encrypted computation`,
  `Result decryption`, …), plus `Total`.
- **`Bandwidth`** — sizes of `Public and evaluation keys`, `Encrypted input`,
  `Encrypted results`.
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

Run `uv run python website/generate_results.py` after measurements change.
The generator averages timing and batched quality fields across all
`results-*.json` files and publishes every field described above.
