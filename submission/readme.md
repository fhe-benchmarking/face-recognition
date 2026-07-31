# CryptoFace - FHE Face-Verification Reference Submission

This is the reference submission for the FHE face-verification benchmark. It runs
**CryptoFace** (patch-based CNN face recognition, CVPR 2025) fully homomorphically
under **RNS-CKKS**, using the [Orion](https://github.com/vboddeti/orion) compiler
with the [Lattigo](https://github.com/tuneinsight/lattigo) backend.

The server evaluates CryptoFace on encrypted patches and returns an encrypted
similarity score. Key material is separated by role: the client keeps
`secret_key/sk.h5`, while the server receives `public_keys/keys.h5` containing
only the public, relinearization, rotation, and bootstrapping evaluation keys.
The server process initializes Orion with `load_secret_key=False`.

---

## 1. Model architecture

CryptoFace is a **patch-based CNN (PCNN)** designed so that every operation has a
low-degree polynomial (CKKS-friendly) form.

```
aligned 64×64 RGB face crop
        │
        ├── split into a 2×2 grid → 4 non-overlapping 32×32 patches
        │
        ▼  (one independent branch per patch)
   ┌───────────────────────────────────────────────┐
   │  Backbone (per patch)                          │   ×4
   │    conv stack with HerPN polynomial activations│
   │    → 64 channels × 2×2 = 256 features          │
   │  Linear (256 → 256)   (patch BN fused in)      │
   └───────────────────────────────────────────────┘
        │
        ▼  sum the 4 per-patch outputs
   256-dimensional embedding
        │
        ▼  L2 normalization via polynomial approximation
   L2NormPoly:  a·y² + b·y + c ≈ 1/√y,   y = ‖embedding‖²
        │
        ▼
   unit-normalized 256-d embedding
```

Key design choices that make the network FHE-evaluable:

- **Patching.** Splitting the 64×64 face into four 32×32 patches keeps each
  backbone shallow, bounding multiplicative depth. Input size is configurable
  (64 → 4 patches, 96 → 9, 128 → 16); this submission uses **64×64 / 4 patches**.
- **HerPN activations.** Hermite-polynomial normalization replaces ReLU+BatchNorm
  with a low-degree polynomial, which CKKS can evaluate directly. BatchNorm is
  fused into the following `Linear` at compile time to save one level of depth.
- **Polynomial L2 normalization.** Exact `1/√y` is not available under CKKS, so it
  is approximated by a quadratic `a·y² + b·y + c` fitted to the empirical
  sum-of-squares (SOS) range of the embeddings on the benchmark data
  (see `config.yml: l2_poly_coeffs`).
- **Verification score.** The similarity of two faces is the inner product of
  their unit-normalized embeddings (cosine similarity), computed homomorphically;
  only this scalar is decrypted.

Weights come from `backbone-64x64.ckpt`, hosted on Hugging Face
([`halmsu/cryptoface-v1`](https://huggingface.co/halmsu/cryptoface-v1)) and
downloaded automatically on first use (see `common.load_submission_config`).

---

## 2. Encrypted inference pipeline

The submission implements the harness's client/server stages (`submission/*.py`):

| Stage | Script | Role |
|------:|--------|------|
| 2 | `client_key_generation` | Generates the private secret key, circuit-specific public/evaluation keys, and persisted model diagonals. |
| 3 | `server_preprocess_model` | Harness-compatible validation of the persisted input level. |
| 5 | `client_preprocess_input` | Client aligns each face (InsightFace) and extracts the 32×32 patches. |
| 6 | `client_encode_encrypt_input` | Client CKKS-encodes and encrypts the patch tensors. |
| 7 | `server_encrypted_compute` | Server loads the Orion circuit and evaluation keys without the secret key, evaluates encrypted pairs with bounded process lifetimes, and returns encrypted scores. |
| 8 | `client_decrypt_decode` | Client decrypts the scalar similarity scores. |

`server_encrypted_compute` compiles the pipeline once and pre-forks five
FHE-quiescent slot managers. Each active slot forks eight one-shot workers (two
images times four backbones) for one pair, then reaps them before accepting the
next pair. A separate quiescent manager creates aggregation processes in
bounded generations of ten pairs. This avoids forking from a process that has
already executed Go/FHE code and releases retained memory regularly.

Five slots, or at most 40 simultaneous backbone workers, maximize throughput on
an otherwise idle 1 TB machine. Four slots is the safer setting when additional
memory headroom is required. The setting is `cryptoface.pair_slots` in
`config.yml`.

The stage writes `io/<size>/server_reported.json` with encrypted-compute wall
time, pipeline setup, and backbone/normalization/inner-product timing.

---

## 3. CKKS parameters and 128-bit security

Parameters are defined in `orion_configs/cryptoface_net4.yml` and follow Lattigo's
**128-bit-secure** bootstrapping preset.

| Parameter | Value |
|-----------|-------|
| Scheme | RNS-CKKS (full-RNS Cheon–Kim–Kim–Song), Lattigo backend |
| Ring degree `N` | `2^16 = 65536` (power-of-two cyclotomic, `RingType: standard`) |
| Secret distribution | sparse ternary, Hamming weight `H = 192` |
| Computation modulus `logQ` | 16 levels: `{55, 15×46}` ≈ **745 bits** |
| Key-switching aux primes `logP` | `3×55` ≈ 165 bits |
| Bootstrap circuit primes | `14×55` ≈ 770 bits |
| Scale | `2^46` |
| Max ciphertext modulus `log(Q·P)` | ≈ **1515 bits** at `N = 2^16` |

**Why this is ≥128-bit secure.** The hardness of RNS-CKKS reduces to Ring-LWE,
whose security is governed by the ring degree `N` and the largest ciphertext
modulus `Q·P` that appears during evaluation (the bootstrapping/key-switching
modulus is the binding case). Per the
[Homomorphic Encryption Standard](https://homomorphicencryption.org/standard/)
and the [LWE estimator](https://github.com/malb/lattice-estimator), for
`N = 2^16` a ternary secret admits a maximum `log(Q·P)` of roughly **1550 bits**
at the 128-bit classical security level. This submission's worst-case
`log(Q·P) ≈ 1515 bits < 1550`, so it stays within the 128-bit bound. The sparse
secret (`H = 192`) is the Lattigo bootstrapping default and is accounted for in
Lattigo's security estimate (which considers sparse-secret / hybrid attacks);
the resulting parameter set targets **≥128 bits** of classical security.

The serialized secret key is confined to `io/<size>/secret_key/`; it is not
stored in `public_keys/keys.h5` or loaded by the encrypted server stage.

---

## 4. Running the reference submission

1. Install dependencies with `scripts/install_system_deps.sh` and
   `scripts/install_python_deps.sh`.
2. The model checkpoint and dataset are fetched from Hugging Face on first run;
   no manual placement is needed. To run offline, see the "Dataset and model"
   section of the top-level `README.md`.
3. From the repo root:

   ```console
   uv run python harness/run_submission.py 0 --seed 42
   uv run python harness/run_submission.py 1 --num_runs 3
   ```

The four benchmark sizes are 1, 128, 256, and 1024 pairs. Batched runs compare
CryptoFace EER and TAR@FAR against the included ArcFace baseline. The acceptance
criterion allows at most a 0.15 absolute EER increase over ArcFace on the same
pairs.

Configuration knobs live in `config.yml` (`input_size`, `l2_poly_coeffs`,
`pair_slots`, and aggregator lifetime) and
`orion_configs/cryptoface_net4.yml` (CKKS parameters).
