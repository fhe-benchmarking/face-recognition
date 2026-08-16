# Understanding the CryptoFace Quality and Latency Results

This document explains the quality and latency columns reported for the
stage-by-stage CryptoFace submission to the FHE face-recognition benchmark. It
describes what each number measures, how the harness calculates it, how the
three measurement runs are combined, and which qualifications matter when
interpreting the results.

## Executive summary

For every face pair, CryptoFace computes a scalar similarity score under
homomorphic encryption. Only that score is decrypted. The harness compares the
decrypted scores with genuine/impostor labels and calculates verification
metrics by sweeping score thresholds. It also evaluates an ArcFace plaintext
reference on the same original image pairs.

The principal quality criterion is the absolute Equal Error Rate (EER) gap:

```text
CryptoFace EER - ArcFace EER <= 0.15
```

The value `0.15` means 15 absolute percentage points, not a 15% relative
increase. The observed gaps are approximately 1.1--1.5 percentage points, so
all three batched workloads pass comfortably.

For latency, the two most important interpretations are:

- **Harness Total/Online** is the benchmark's end-to-end stage-by-stage wall
  time.
- **Server Compute (wall)** isolates the encrypted server computation after
  server pipeline/key setup and is the appropriate value for FHE-server
  throughput.

Server measurements are nested inside the harness's stage-7 measurement. They
must not be added to the harness total. Likewise, fields labeled `worker` are
summed work across concurrent processes, not elapsed latency.

## Workloads and reporting convention

The benchmark has four workload sizes:

| Workload | Face pairs | Reporting |
|---|---:|---|
| Single | 1 | One functional smoke-test run |
| Small | 128 | Arithmetic mean of three run-level measurements |
| Medium | 256 | Arithmetic mean of three run-level measurements |
| Large | 1,024 | Arithmetic mean of three run-level measurements |

The fixed master dataset contains 1,024 screened CelebA pairs: 512 genuine
pairs and 512 impostor pairs. Small and Medium sample from that master set
without replacement. Large uses every pair.

The harness starts with seed 42 and deterministically generates per-run seeds
`191664963`, `1662057957`, and `1405681631`. Each run's quality metrics are
calculated independently; the website then averages those three metric values.
Scores from different runs are not pooled into one ROC curve. Pair selection is
implemented in [`harness/generate_input.py`](../harness/generate_input.py).

## Quality results

The submitted quality values, expressed as percentages, are:

| Size | CryptoFace EER | ArcFace EER | EER gap | CryptoFace TAR@1% | ArcFace TAR@1% | CryptoFace TAR@0.1% | ArcFace TAR@0.1% | Result |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Small | 7.45% | 6.31% | +1.14 pp | 83.99% | 93.69% | 83.99% | 93.69% | PASS |
| Medium | 8.68% | 7.21% | +1.46 pp | 83.95% | 92.56% | 82.08% | 92.09% | PASS |
| Large | 7.23% | 6.05% | +1.17 pp | 84.77% | 93.55% | 82.03% | 93.16% | PASS |

The Single result contains only the decrypted score `0.588810` and label `1`
(genuine). One pair cannot define an ROC curve, EER, or TAR, so Single is a
functional correctness and score-format smoke test rather than an accuracy
estimate.

### What the quality columns mean

| Column | Definition | Preferred direction |
|---|---|---|
| FHE EER | EER calculated from decrypted CryptoFace scores | Lower |
| FHE TAR@1% | Highest empirical TAR at FAR no greater than 1% | Higher |
| FHE TAR@0.1% | Highest empirical TAR at FAR no greater than 0.1% | Higher |
| ArcFace columns | The corresponding metrics from ArcFace on the same pairs | Reference |
| EER gap | `FHE EER - ArcFace EER` | Lower |
| TAR gap | `FHE TAR - ArcFace TAR` | Higher |
| Max EER gap | Allowed absolute EER gap (`0.15`) | Fixed criterion |
| Result | PASS when the EER gap is no greater than `0.15` | PASS |

A positive EER gap means CryptoFace has a higher error rate than ArcFace. A
negative TAR gap means CryptoFace accepts fewer genuine pairs at that FAR
constraint. TAR gaps are diagnostic; the current PASS/FAIL rule uses only the
EER gap.

### Scores and labels

The harness expects exactly one finite floating-point similarity score per
input pair, in input order:

- label `1`: genuine pair, meaning the same identity;
- label `0`: impostor pair, meaning different identities; and
- a higher score indicates greater similarity.

Before calculating metrics, the harness rejects a score-count mismatch,
invalid labels, and NaN or infinite scores. These checks and the metric
implementation are in [`harness/metrics.py`](../harness/metrics.py).

### EER calculation

For threshold `t`, scores greater than or equal to `t` are accepted as
genuine. At each distinct score threshold, the harness calculates:

```text
FAR = false accepts / number of impostor pairs
FRR = false rejects / number of genuine pairs
TAR = true accepts / number of genuine pairs = 1 - FRR
```

The harness orders distinct scores from highest to lowest and lowers the
threshold through the complete set. EER is the point where `FAR = FRR`. Since
the empirical ROC curve is discrete, it linearly interpolates between the two
adjacent points surrounding the crossover.

EER is sometimes described as threshold-free. More precisely, it does not
require a deployment threshold to be selected in advance; it is still derived
from a threshold sweep over the evaluation set.

### TAR at fixed FAR

For TAR@1% and TAR@0.1%, the harness selects the largest observed TAR among all
thresholds satisfying:

```text
FAR <= 0.01   for TAR@1%
FAR <= 0.001  for TAR@0.1%
```

TAR is not interpolated. It uses attainable empirical thresholds, including
all tied scores at the same threshold.

This creates an important finite-sample effect. The smallest nonzero FAR is
one divided by the number of impostor pairs:

| Workload/run composition | Smallest nonzero FAR | Consequence |
|---|---:|---|
| Small: 53--71 impostors | 1.41--1.89% | Both 1% and 0.1% require zero false accepts |
| Medium: 112--135 impostors | 0.74--0.89% | 1% may allow one false accept; 0.1% allows none |
| Large: 512 impostors | 0.195% | 1% allows at most five false accepts; 0.1% allows none |

This is why the two Small TAR columns are identical. With so few impostors,
the first nonzero empirical FAR already exceeds 1%.

### Sampling and class balance

The master set is balanced, but Small and Medium sampling is uniform rather
than class-stratified. The actual genuine/impostor counts were:

| Size | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Small | 57 / 71 | 60 / 68 | 75 / 53 |
| Medium | 121 / 135 | 127 / 129 | 144 / 112 |
| Large | 512 / 512 | 512 / 512 | 512 / 512 |

Consequently, Small and Medium quality values vary somewhat between runs. Large
contains the full master set each time, so its three quality measurements are
identical apart from input order. Repeating Large is useful for reproducibility
and timing variance but does not create three independent quality samples.

For example, Small run 1 has CryptoFace EER 7.02%, whereas the submitted table
shows 7.45%, the arithmetic mean of all three stage runs. Medium run 1 has
CryptoFace EER 9.92%, while the submitted three-run average is 8.68%.

### What “FHE quality” includes

The label `FHE EER` means EER calculated from the complete encrypted
CryptoFace pipeline's decrypted output scores. It reflects all of the
following:

- the trained, FHE-friendly CryptoFace model;
- 64x64 aligned inputs and the four-patch architecture;
- polynomial activations and polynomial L2 normalization;
- CKKS numerical approximation; and
- final encrypted inner-product scores.

It is **not** an isolated measurement of encryption-induced numerical error.
Isolating that effect would require comparing encrypted CryptoFace against the
identical CryptoFace graph evaluated in plaintext. The reported comparison
instead answers whether the complete encrypted submission remains acceptably
close to the benchmark's stronger plaintext ArcFace reference.

Both paths process the same original image pairs and use InsightFace's
Buffalo-L detector with one explicit 640x640 detector pass. CryptoFace uses the
detector landmarks to align at 112x112 and then downsamples to its 64x64 model
input. ArcFace performs its own aligned recognition crop and produces a
512-dimensional embedding. See
[`submission/common.py`](../submission/common.py) and
[`harness/cleartext_impl.py`](../harness/cleartext_impl.py).

### Why the TAR gap is larger than the EER gap

EER describes the central ROC crossover, which occurs around 6--9% error for
these results. TAR@1%, and particularly TAR@0.1%, probes the extreme tail of
the impostor-score distribution. A handful of poorly ranked pairs can
therefore change low-FAR TAR substantially while changing EER only slightly.

CryptoFace can consequently trail ArcFace by only about one percentage point
in EER while trailing by roughly 9--11 percentage points in low-FAR TAR. The
small sample counts also make the low-FAR values coarse.

## Latency results

The submitted stage-by-stage timing values are:

| Workload | Pairs | Total | Offline setup | Online evaluation | Harness Compute | Server Compute (wall) |
|---|---:|---:|---:|---:|---:|---:|
| Single | 1 | 47.710 min | 19.391 min | 28.319 min | 24.623 min | 4.183 min |
| Small | 128 | 2.240 h | 19.243 min | 1.919 h | 1.656 h | 1.245 h |
| Medium | 256 | 3.418 h | 21.038 min | 3.067 h | 2.670 h | 2.294 h |
| Large | 1,024 | 12.507 h | 17.689 min | 12.213 h | 10.688 h | 10.270 h |

The corresponding throughput is:

| Workload | End-to-end online throughput | Pure encrypted-server throughput |
|---|---:|---:|
| Small | 66.70 pairs/hour | 102.82 pairs/hour |
| Medium | 83.47 pairs/hour | 111.62 pairs/hour |
| Large | 83.85 pairs/hour | 99.71 pairs/hour |

The batched values are arithmetic means across three result JSONs. They are
not the cumulative duration of all three runs.

## Harness timing columns

The harness records observable wall time between stage-completion markers. The
strict stage-by-stage contract means these stages execute sequentially; client
preprocessing, encryption, server computation, and decryption do not overlap.
The timing mechanism is implemented in
[`harness/utils.py`](../harness/utils.py).

### Total

```text
Total = Offline setup total + Online evaluation total
```

It represents a cold end-to-end benchmark run that includes dataset
validation, key generation, and model preparation. It does not mean that
offline setup must be paid for every request in a deployed system.

### Offline setup

Offline setup contains stages 1--3:

- **Dataset:** provision, migrate if necessary, and validate the indexed
  dataset.
- **Keygen:** create the client secret key and public, relinearization,
  rotation, and bootstrapping evaluation keys.
- **Model prep:** compile or validate the Orion model representation and its
  packed-model cache.

When the harness executes `--num_runs 3`, these stages run once before the
per-run loop. Their measured values are copied into every result JSON so that
each file describes one cold end-to-end execution. The website therefore
reports the same one-time setup contribution alongside the average online run;
it neither sums setup three times nor amortizes it over three runs.

### Online evaluation

Online evaluation is the sum of stages 4--10:

1. Input generation
2. Input preprocessing
3. Input encryption
4. Encrypted computation
5. Result decryption
6. Result postprocessing
7. ArcFace baseline inference
8. Quality calculation

The exact stage sequence is in
[`harness/run_submission.py`](../harness/run_submission.py).

### Input generation

The harness samples pair indices, writes the labels in matching order, and
materializes the selected encoded images in the run's HDF5 input store.

### Input preprocessing

This is conventional client-side work: face detection, landmark alignment,
resizing to 64x64, normalization, and extraction of four 32x32 patches from
each face image.

### Input encryption

The client encodes and encrypts eight patch tensors per pair: two images times
four patches.

### Harness Compute

The harness `Compute` column is the wall time surrounding the complete stage-7
server command. It includes server process launch, server pipeline/key setup,
input validation, encrypted inference, worker management, encrypted-result
writing, and small amounts of adjacent harness bookkeeping such as artifact
size collection.

It is therefore larger than both the server's `Total (wall)` and its pure
`Compute (wall)`.

### Result decryption

This is client-side loading, decryption, and decoding of the encrypted scalar
scores. The marker-to-marker harness value also includes a small amount of
adjacent encrypted-result size collection.

### Result postprocessing

This validates and formats the decrypted scores into the benchmark's required
one-score-per-line output. It is negligible in these measurements.

### ArcFace

This runs the plaintext ArcFace quality reference on the same pairs. It is
included in the harness's formal online evaluation time, but it is not part of
a production encrypted CryptoFace request. For example, ArcFace accounts for
about 18.95 minutes of the Large online result.

### Quality

This validates the reference scores and calculates the ArcFace metrics and
paired gaps. It takes milliseconds and has no material effect on latency. Most
of the encrypted-score validation sweep occurs immediately before the ArcFace
completion marker, but it is also negligible relative to ArcFace inference.

## Server timing columns

Server timing fields are submission-provided diagnostics within harness stage
7. Their definitions are written in
[`submission/server_encrypted_compute.py`](../submission/server_encrypted_compute.py).

For Large, the nested timing relationship is:

```text
Harness Compute                       10.688 h
└── Server persistent process         10.677 h
    └── Server Total                  10.644 h
        ├── Pipeline/key setup         0.375 h
        └── Pure encrypted compute    10.270 h
```

### Total (wall)

```text
Server Total = Pipeline and key setup + Encrypted computation
```

It excludes some process-level argument parsing, validation, worker lifecycle,
and report-writing overhead that appears in `Server process lifetime`.

### Server process lifetime

Elapsed time from the beginning of the stage-7 server process until its report
is written. It includes setup, input validation, worker creation and teardown,
encrypted processing, and server bookkeeping.

### Setup (wall)

Loads the packed model and evaluation keys and performs runtime Orion circuit
compilation. The table breaks out:

- **Model I/O:** reading the approximately 114.1 GiB packed-model artifact;
- **Runtime compile:** Orion's runtime circuit compilation; and
- remaining setup overhead, including key loading and validation.

### Compute (wall)

Pure elapsed wall time spent processing encrypted pair chunks after server
setup. This is the best column for discussing FHE-server throughput.

### Mean per pair

```text
Mean per pair = Server Compute (wall) / number of pairs
```

For batched workloads this is an inverse-throughput quantity, not the response
latency of one particular pair. For example, Large reports 36.105 seconds per
pair, equivalent to 99.71 pairs/hour, because ten pairs are processed
concurrently. An individual pair can remain in flight for several minutes
while the saturated server completes one pair every 36 seconds on average.

For Single there is no batching, so its 4.183-minute value is an actual
single-pair server latency.

### Worker-time columns

Fields explicitly labeled `worker` are summed work across concurrent branch
and aggregation processes:

- **Input I/O (worker):** summed ciphertext deserialization time;
- **Transport (worker):** summed feature/result serialization and interprocess
  transport time;
- **Inference (worker):** backbone + normalization + inner-product worker time;
- **Backbones (worker):** summed four-patch CNN branch evaluation time;
- **Normalize (worker):** summed polynomial embedding normalization time; and
- **Inner product (worker):** summed encrypted similarity computation time.

These values can be much larger than elapsed wall time. Large reports about
662.6 backbone worker-hours while completing pure encrypted computation in
10.27 wall hours. The ratio corresponds to approximately 64.5 concurrently
active backbone-worker equivalents on average. The configured maximum is 80:
ten concurrent pair slots times two images times four patch backbones.

Worker time answers how much aggregate parallel work was performed. Wall time
answers how long the caller waited. Worker-time values must not be added to
wall-time values.

## Why Medium appears faster than Large

Large contains four times as many pairs as Medium:

- Medium: 256 pairs, 3.418 hours total;
- Large: 1,024 pairs, 12.507 hours total.

Large is approximately 3.66 times slower for four times the work, so scaling is
close to linear once fixed setup is considered. The pure server mean values
are also in the same range:

| Size | Server mean per pair |
|---|---:|
| Small | 35.014 s |
| Medium | 32.254 s |
| Large | 36.105 s |

Medium obtained the best sustained throughput in these measurements. The
longer Large execution experiences more memory pressure, disk traffic, process
churn, and normal host/run variance. It does not use a different model or
algorithm.

## Interpretation cautions

- Quality is face-verification ranking quality, not classification accuracy at
  one predetermined threshold.
- The FHE-versus-ArcFace gap compares different recognizers and does not isolate
  CKKS numerical error.
- Small and Medium samples are not class-stratified.
- Low-FAR TAR is statistically coarse with only 53--512 impostor pairs.
- The harness uses one global threshold sweep per run with no separate
  calibration set or cross-validation.
- No confidence intervals are currently reported.
- Batched table entries are three-run arithmetic means, not sums.
- Large's three quality runs contain the same complete pair set and therefore
  test reproducibility rather than independent quality samples.
- Offline setup runs once per harness invocation but is included in every
  result JSON as a cold-run contribution.
- Harness `Online` includes the ArcFace reference pass, which is benchmark
  validation work rather than production FHE inference.
- Server timing is already contained in harness timing; adding the two double
  counts stage 7.
- Worker-hours are aggregate parallel work, not latency.
- Batched `Mean per pair` is inverse throughput, not individual-pair response
  time.
- Measurements are local to the reported AMD EPYC 7502 host and do not include
  network transfer latency.
- Harness timing is observable marker-to-marker wall time and intentionally
  includes small adjacent bookkeeping operations; it is not isolated
  instruction-level kernel profiling.

## Source artifacts

The exact unrounded inputs to the submitted tables are retained in:

- [`measurements/single/results-1.json`](../measurements/single/results-1.json)
- [`measurements/small/`](../measurements/small/)
- [`measurements/medium/`](../measurements/medium/)
- [`measurements/large/`](../measurements/large/)

The primary calculation and orchestration sources are:

- [`harness/metrics.py`](../harness/metrics.py) — EER, TAR@FAR, metric gaps, and
  acceptance;
- [`harness/run_submission.py`](../harness/run_submission.py) — stage order and
  quality evaluation;
- [`harness/utils.py`](../harness/utils.py) — harness timing boundaries and
  result serialization;
- [`harness/cleartext_impl.py`](../harness/cleartext_impl.py) — ArcFace
  reference scoring;
- [`submission/common.py`](../submission/common.py) — CryptoFace detection and
  alignment; and
- [`submission/server_encrypted_compute.py`](../submission/server_encrypted_compute.py)
  — server wall-time and worker-time definitions.
