#!/usr/bin/env python3
"""
utils.py - Harness utilities for argument parsing, logging, and results saving.

Provides:
  - parse_submission_arguments(): CLI argument parsing
  - ensure_directories():         validate required repo subdirectories
  - build_submission():           build submission via its build_task.sh
  - run_exe_or_python():          run a stage as Python script or compiled binary
  - log_step():                   print per-stage elapsed time
  - log_size():                   print and record directory sizes
  - log_quality():                record a single score or EER/TAR@FAR metrics
  - save_run():                   write per-run JSON results to measurements/
  - reset_run_state():            clear accumulated per-run state between runs
"""
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

import sys
import subprocess
import argparse
import json
from datetime import datetime
from pathlib import Path
from params import InstanceParams, SINGLE, LARGE
from typing import Tuple

# Global variable to track the last timestamp
_last_timestamp: datetime = None
# Global variable to store measured times
_timestamps = {}
_timestampsStr = {}
# One-time stage timings (key generation, model preprocessing, ...) measured
# before the per-run loop. These survive reset_run_state() so they can be
# reported in every run's results file, matching the ml-inference schema.
_onetime_timestamps = {}
_onetime_timestampsStr = {}
# Global variable to store measured sizes
_bandwidth = {}
# One-time bandwidth, such as public and evaluation keys.
_onetime_bandwidth = {}
# Global variable to store model quality metrics
_model_quality = {}

def parse_submission_arguments(workload: str) -> Tuple[int, InstanceParams, int, int, int]:
    """
    Get the arguments of the submission. Populate arguments as needed for the workload.
    """
    # Parse arguments using argparse
    parser = argparse.ArgumentParser(description=workload)
    parser.add_argument('size', type=int, choices=range(SINGLE, LARGE+1),
                        help='Instance size (0-single/1-small/2-medium/3-large)')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of times to run stages 4-10 (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible pair sampling (default: 42). '
                             'Fixed by default so all submissions sample identical pairs.')
    parser.add_argument('--clrtxt', type=int,
                        help='Set to 1 to force rerun of cleartext reference')
    args = parser.parse_args()
    size = args.size
    seed = args.seed
    num_runs = args.num_runs
    clrtxt = args.clrtxt

    # Use params.py to get instance parameters
    params = InstanceParams(size)
    return size, params, seed, num_runs, clrtxt

def ensure_directories(rootdir: Path):
    """ Check that the current directory has sub-directories
    'harness', 'scripts', and 'submission' """
    required_dirs = ['harness', 'scripts', 'submission']
    for dir_name in required_dirs:
        if not (rootdir / dir_name).exists():
            print(f"Error: Required directory '{dir_name}'",
                  f"not found in {rootdir}")
            sys.exit(1)

def build_submission(script_dir: Path):
    """
    Build the submission. Fetching dependencies and compiling is 
    delegated entirely to the submission's build_task.sh.
    """
    subprocess.run([script_dir / "build_task.sh", "./submission"], check=True)

def log_step(step_num: int, step_name: str, start: bool = False):
    """
    Print a timestamped completion message and record elapsed time for a stage.
    If start=True, records the start time without printing (used for step 0).
    """
    global _last_timestamp
    global _timestamps
    global _timestampsStr
    now = datetime.now()
    timestamp = now.strftime("%H:%M:%S")

    # Calculate elapsed time if this isn't the first call
    elapsed_str = ""
    elapsed_seconds = 0
    if _last_timestamp is not None:
        elapsed_seconds = (now - _last_timestamp).total_seconds()
        elapsed_str = f" (elapsed: {round(elapsed_seconds, 4)}s)"

    # Update the last timestamp for the next call
    _last_timestamp = now

    if (not start):
        print(f"{timestamp} [harness] {step_num}: {step_name} completed{elapsed_str}")
        _timestampsStr[step_name] = f"{round(elapsed_seconds, 4)}s"
        _timestamps[step_name] = elapsed_seconds

def log_size(path: Path, object_name: str, flag: bool = False, previous: int = 0):
    """Print and record the disk size of a directory. If flag=True, subtracts previous bytes."""
    global _bandwidth
    
    # Check if the path exists before trying to calculate size
    if not path.exists():
        print(f"         [harness] Warning: {object_name} path does not exist: {path}")
        _bandwidth[object_name] = "0B"
        return 0
    
    size = int(subprocess.run(["du", "-sb", path], check=True,
                           capture_output=True, text=True).stdout.split()[0])
    if(flag):
        size -= previous
    
    print("         [harness]", object_name, "size:", human_readable_size(size))

    _bandwidth[object_name] = human_readable_size(size)
    return size

def run_exe_or_python(base, file_name, *args, check=True):
    """
    If {base}/{file_name}.py exists, run it with the current Python interpreter.
    Otherwise, run {base}/build/{file_name} as a compiled executable.
    """
    py  = base / f"{file_name}.py"
    exe = base / "build" / file_name

    if py.exists():
        cmd = [sys.executable, str(py), *args]
    elif exe.exists():
        cmd = [str(exe), *args]
    else:
        print(f"[harness] Error: neither {py} nor {exe} found")
        sys.exit(1)
    subprocess.run(cmd, check=check)


def human_readable_size(n: int) -> str:
    """Convert a byte count to a human-readable string (e.g. 1.4G, 358.8K)."""
    for unit in ["B","K","M","G","T"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}P"

def _read_server_reported(iodir: Path) -> dict:
    """
    Read the server's self-reported timing, written by server_encrypted_compute
    to io/<size>/server_reported.json. Returns {} if the file is absent.

    Expected schema (all values in seconds, floats):
      {
        "Encrypted computation": <float>,          # pure encrypted compute (excl. setup)
        "Total": <float>,                          # server wall time for stage 7
        "additional_measurements": { <label>: <float>, ... }   # optional fine-grained breakdown
      }
    """
    f = iodir / "server_reported.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        print(f"         [harness] Warning: could not parse {f}")
        return {}


def save_run(path: Path, size: int = 0, iodir: Path = None):
    """
    Write per-run timing, bandwidth, and quality metrics
    to a JSON file at the given path, using the ml-inference measurement schema:
    top-level Timing / Bandwidth / Quality / Server Reported keys.
    One-time stage timings (key generation, model preprocessing) captured before
    the per-run loop are included in each run's Timing block. Timing["Total"] is
    the sum of one-time and per-run stage latencies.
    """
    global _timestamps
    global _timestampsStr
    global _onetime_timestamps
    global _onetime_timestampsStr
    global _bandwidth
    global _onetime_bandwidth
    global _model_quality

    total = round(sum(_onetime_timestamps.values()) + sum(_timestamps.values()), 4)
    timing = {**_onetime_timestampsStr, **_timestampsStr, "Total": f"{total}s"}

    data = {
        "Timing": timing,
        "Bandwidth": {**_onetime_bandwidth, **_bandwidth},
    }
    if _model_quality:
        data["Quality"] = _model_quality

    # Server-reported timing (fine-grained breakdown of stage 7) when available.
    server_reported = _read_server_reported(iodir) if iodir is not None else {}
    if server_reported:
        print(f"         [submission] Server reported steps: {server_reported}")
        for step_name, seconds in server_reported.items():
            print(f"         [submission] {step_name}: {seconds}s")
        data["Server Reported"] = {
            k: (f"{v}s" if isinstance(v, (int, float)) else v)
            for k, v in server_reported.items()
        }
    elif iodir is not None:
        print(
            "         [harness] Note: submitters can provide server timings at "
            f"{iodir / 'server_reported.json'}"
        )
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print("[total latency]", f"{total}s")

def log_quality(metrics: dict, tag: str):
    """Store a single score or verification metrics in the quality block."""
    global _model_quality
    if not metrics:
        return
    if "score" in metrics:
        _model_quality[tag] = {
            "score": metrics["score"],
            "label": metrics["label"],
        }
        return
    _model_quality[tag] = {
        "eer":              metrics["eer"],
        "tar_at_far_1pct":  metrics["tar_far_1_percent"],
        "tar_at_far_01pct": metrics["tar_far_01_percent"],
    }

def log_quality_comparison(comparison: dict):
    """Store encrypted-versus-ArcFace deltas in the standard Quality block."""
    if comparison:
        _model_quality["Comparison to ArcFace baseline"] = comparison

def reset_run_state():
    """
    Reset per-run accumulated timing, bandwidth, and quality state.
    Call at the start of each run so that save_run() writes only that run's data.

    On the first call, the stage timings accumulated before the per-run loop
    (dataset validation, key generation, model preprocessing) are moved into the
    one-time store so they persist across runs and appear in every results file.
    """
    global _timestamps, _timestampsStr, _bandwidth, _model_quality
    global _onetime_timestamps, _onetime_timestampsStr
    global _onetime_bandwidth
    if not _onetime_timestamps and _timestamps:
        _onetime_timestamps    = dict(_timestamps)
        _onetime_timestampsStr = dict(_timestampsStr)
        _onetime_bandwidth     = dict(_bandwidth)
    _timestamps    = {}
    _timestampsStr = {}
    _bandwidth     = {}
    _model_quality = {}
