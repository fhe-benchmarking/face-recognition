# Submission interface

This document defines the normative stage-by-stage face-verification benchmark
interface.

## Entry points

For each stage name, the harness runs the first available entry point:

1. `submission/<stage>.py` with the active Python interpreter; or
2. `submission/build/<stage>` as a native executable.

Commands run from the repository root. A size argument is one of `0`, `1`,
`2`, or `3`, corresponding to 1, 128, 256, or 1,024 face pairs. A stage must
exit nonzero on failure and must not report success before its output is fully
written.

## Normative stages

The harness creates a fresh `io/<instance>/` directory, then invokes these
submission-owned stages in order:

| Stage | Invocation | Responsibility |
|---:|---|---|
| 2 | `client_key_generation <size>` | Generate client secret material and the public/evaluation material needed by the server. |
| 3 | `server_preprocess_model` | Prepare the encrypted server model without access to client secret material. |
| 5 | `client_preprocess_input <size>` | Read the harness input and perform submission-specific detection, alignment, cropping, and tensor preparation. |
| 6 | `client_encode_encrypt_input <size>` | Encrypt the prepared inputs using client-owned material. |
| 7 | `server_encrypted_compute <size>` | Read encrypted inputs and write encrypted verification scores without loading the secret key. |
| 8 | `client_decrypt_decode <size>` | Decrypt and decode the server results. |
| 9 | `client_postprocess <size>` | Write final scores in benchmark format. This may be a no-op when stage 8 already writes that format. |

The harness, not the submission, owns stages 1, 4, and 10. Before stage 5 it
writes:

- `datasets/<instance>/intermediate/test_pairs.h5`, containing equally sized
  `image0` and `image1` variable-length `uint8` datasets. Each element is an
  encoded RGB image and rows appear in benchmark input order;
- `datasets/<instance>/intermediate/test_labels.txt`, containing one `0` or
  `1` label per pair.

The required final output is
`io/<instance>/encrypted_model_predictions.txt`: one finite floating-point
similarity score per line, in input order, followed by a newline. It must
contain exactly the configured number of scores. The submission may choose its
own intermediate filenames under `io/<instance>/`.

Client secret material must not be loaded by stages 3 or 7. Cleartext images,
plaintext features, and decrypted scores must not be made available to the
server stage. Public/evaluation keys, encrypted inputs, the encrypted model,
and encrypted results may cross the client/server boundary.

## Optional reports

A submission may write these generic JSON artifacts under `io/<instance>/`:

- `submission_reported.json` with a `Bandwidth` object mapping labels to
  non-negative integer byte counts;
- `server_reported.json` mapping timing labels to numeric seconds, with nested
  metadata allowed;
- `provenance.json` containing any JSON object that identifies the evaluated
  software, model, parameters, and data.

The harness copies valid reports into the measurement JSON. These files do not
replace the harness's own wall-time or artifact-size measurements.

## Validation

At minimum, submitters should run:

```bash
uv run python harness/run_submission.py 0 --seed 42
```

The single-pair instance validates stage execution, score count, score
finiteness, reports, and artifact ownership. Quality metrics require at least
the 128-pair instance; larger runs should be attempted only after the single
and small instances pass.
