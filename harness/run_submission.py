#!/usr/bin/env python3
"""Run the stage-by-stage face-verification FHE benchmark."""

# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil
import subprocess
import sys

import numpy as np

import utils
from metrics import calculate_face_metrics, compare_to_arcface
from params import instance_name


def main():
    size, params, seed, num_runs = utils.parse_submission_arguments(
        "Run Face Verification FHE benchmark."
    )
    print(f"\n[harness] Running face verification for {instance_name(size)}")

    utils.ensure_directories(params.rootdir)
    harness_dir = params.rootdir / "harness"
    submission_dir = params.rootdir / "submission"
    io_dir = params.iodir()
    shutil.rmtree(io_dir, ignore_errors=True)
    io_dir.mkdir(parents=True)
    utils.log_step(0, "Init", True)

    dataset_npy = params.rootdir / "datasets" / "face_dataset.npy"
    subprocess.run(
        [sys.executable, harness_dir / "generate_dataset.py", str(dataset_npy)],
        check=True,
    )
    utils.log_step(1, "Test dataset generation")

    utils.run_exe_or_python(submission_dir, "client_key_generation", str(size))
    utils.log_step(2, "Key Generation")
    utils.run_exe_or_python(submission_dir, "server_preprocess_model")
    utils.log_step(3, "Encrypted model preprocessing")
    utils.log_size(io_dir / "public_keys", "Public and evaluation keys")
    utils.log_submission_reported(io_dir)

    rng = np.random.default_rng(seed)
    persistent_io = {path.name for path in io_dir.iterdir()}
    for run in range(num_runs):
        run_path = params.measuredir() / f"results-{run + 1}.json"
        for path in io_dir.iterdir():
            if path.name not in persistent_io:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        utils.reset_run_state()
        if num_runs > 1:
            print(f"\n         [harness] Run {run + 1} of {num_runs}")

        command = [sys.executable, harness_dir / "generate_input.py", str(size)]
        if seed is not None:
            command.extend([
                "--seed", str(int(rng.integers(0, 0x7FFFFFFF))),
            ])
        subprocess.run(command, check=True)
        subprocess.run(
            [sys.executable, harness_dir / "materialize_input_store.py", str(size)],
            check=True,
        )
        utils.log_step(4, "Input generation")

        utils.run_exe_or_python(
            submission_dir, "client_preprocess_input", str(size)
        )
        utils.log_step(5, "Input preprocessing")
        utils.run_exe_or_python(
            submission_dir, "client_encode_encrypt_input", str(size)
        )
        utils.log_step(6, "Input encryption")
        utils.log_size(io_dir / "ciphertexts_upload", "Encrypted input")
        utils.run_exe_or_python(
            submission_dir, "server_encrypted_compute", str(size)
        )
        utils.log_step(7, "Encrypted computation")
        utils.log_size(io_dir / "ciphertexts_download", "Encrypted results")
        utils.run_exe_or_python(
            submission_dir, "client_decrypt_decode", str(size)
        )
        utils.log_step(8, "Result decryption")
        utils.run_exe_or_python(submission_dir, "client_postprocess", str(size))
        utils.log_step(9, "Result postprocessing")

        labels = params.get_ground_truth_labels_file()
        encrypted_scores = params.get_encrypted_model_predictions_file()
        harness_scores = params.get_harness_model_predictions_file()
        if not encrypted_scores.exists():
            sys.exit(f"[harness] Error: result file not found: {encrypted_scores}")

        encrypted_metrics = calculate_face_metrics(
            labels,
            encrypted_scores,
            "Encrypted model quality",
            expected_count=params.get_batch_size(),
        )
        utils.log_quality(encrypted_metrics, "Encrypted model quality")
        if size == 0:
            utils.log_step(10, "Harness: Run quality check")
            run_path.parent.mkdir(parents=True, exist_ok=True)
            utils.save_run(run_path, iodir=io_dir)
            continue

        subprocess.run(
            [
                sys.executable,
                harness_dir / "cleartext_impl.py",
                str(params.get_test_input_file()),
                str(harness_scores),
            ],
            check=True,
        )
        utils.log_step(10.1, "Harness: Run inference for harness plaintext model")

        harness_metrics = calculate_face_metrics(
            labels,
            harness_scores,
            "Harness model quality",
            expected_count=params.get_batch_size(),
        )
        utils.log_quality(harness_metrics, "Harness model quality")
        comparison = compare_to_arcface(encrypted_metrics, harness_metrics)
        utils.log_quality_comparison(comparison)
        if comparison:
            print(
                "[harness] Encrypted vs ArcFace: "
                f"EER gap={comparison['eer_gap']:+.4f}, "
                f"TAR@FAR=1% gap={comparison['tar_at_far_1pct_gap']:+.4f}, "
                f"acceptance={'PASS' if comparison['passed'] else 'FAIL'}"
            )
        utils.log_step(10.2, "Harness: Run quality check")

        run_path.parent.mkdir(parents=True, exist_ok=True)
        utils.save_run(run_path, iodir=io_dir)

    print(f"\nAll steps completed for face verification ({instance_name(size)})!")


if __name__ == "__main__":
    main()
