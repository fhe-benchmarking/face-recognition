#!/usr/bin/env python3
"""Generate complete benchmark result tables from measurement JSON files."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MEASUREMENTS = ROOT / "measurements"
WEBSITE = ROOT / "website"
SIZES = {
    "single": ("Single", "Single - 1"),
    "small": ("Small", "Small Batch - 128"),
    "medium": ("Medium", "Medium Batch - 256"),
    "large": ("Large", "Large Batch - 1,024"),
}

TIMING_COLUMNS = [
    ("Total", "Total"),
    ("Offline setup total", "Offline setup"),
    ("Online evaluation total", "Online evaluation"),
    ("Test dataset generation", "Dataset"),
    ("Key Generation", "Keygen"),
    ("Encrypted model preprocessing", "Model prep"),
    ("Input generation", "Input gen"),
    ("Input preprocessing", "Input prep"),
    ("Input encryption", "Input enc"),
    ("Encrypted computation", "Compute"),
    ("Result decryption", "Decrypt"),
    ("Result postprocessing", "Postprocess"),
    ("Harness: Run inference for harness plaintext model", "ArcFace"),
    ("Harness: Run quality check", "Quality"),
]

SERVER_COLUMNS = [
    ("Total", "Total (wall)"),
    ("Persistent process lifetime", "Process lifetime"),
    ("Pipeline load and key setup", "Setup (wall)"),
    ("Packed model I/O", "Model I/O"),
    ("Runtime circuit compilation", "Runtime compile"),
    ("Ciphertext input I/O worker-seconds", "Input I/O (worker)"),
    ("Ciphertext transport worker-seconds", "Transport (worker)"),
    ("Encrypted inference worker-seconds", "Inference (worker)"),
    ("Encrypted computation", "Compute (wall)"),
    ("Backbone forward worker-seconds", "Backbones (worker)"),
    ("Normalization worker-seconds", "Normalize (worker)"),
    ("Inner product worker-seconds", "Inner product (worker)"),
    ("Mean encrypted wall time per pair", "Mean / pair"),
]


def _seconds(value: str) -> float:
    if not value.endswith("s"):
        raise ValueError(f"Expected seconds value, got {value!r}")
    return float(value[:-1])


def _mean_path(runs: list[dict], *path: str) -> float | None:
    values = []
    for run in runs:
        value = run
        for key in path:
            if key not in value:
                break
            value = value[key]
        else:
            values.append(value)
    if not values:
        return None
    return statistics.fmean(values)


def _mean_seconds(runs: list[dict], section: str, key: str) -> float | None:
    values = [
        _seconds(run[section][key])
        for run in runs
        if section in run and key in run[section]
    ]
    return statistics.fmean(values) if values else None


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.3f}s"
    if seconds < 3600:
        return f"{seconds / 60:.3f}m"
    return f"{seconds / 3600:.3f}h"


def _format_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _format_bytes(value: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024


def _same_value(runs: list[dict], section: str, key: str) -> str:
    values = {run.get(section, {}).get(key) for run in runs}
    values.discard(None)
    if not values:
        return "-"
    if len(values) != 1:
        raise ValueError(f"{section}.{key} differs across runs: {sorted(values)}")
    return str(values.pop())


def _same_path_value(runs: list[dict], *path: str) -> str:
    values = []
    for run in runs:
        value = run
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            values.append(value)
    if not values:
        return "-"
    serialized = {json.dumps(value, sort_keys=True) for value in values}
    if len(serialized) != 1:
        raise ValueError(f"{'.'.join(path)} differs across runs")
    return str(values[0])


def _quality_columns(size: str, runs: list[dict]) -> list[tuple[str, str]]:
    if size == "single":
        scores = [
            run.get("Quality", {}).get("Encrypted model quality", {}).get("score")
            for run in runs
        ]
        labels = [
            run.get("Quality", {}).get("Encrypted model quality", {}).get("label")
            for run in runs
        ]
        valid_scores = [score for score in scores if score is not None]
        valid_labels = [label for label in labels if label is not None]
        return [
            ("Score(s)", ", ".join(f"{score:.6f}" for score in valid_scores) or "-"),
            ("Label(s)", ", ".join(str(label) for label in valid_labels) or "-"),
        ]

    encrypted = "Encrypted model quality"
    arcface = "Harness model quality"
    comparison = "Comparison to ArcFace baseline"
    passed = [
        run.get("Quality", {}).get(comparison, {}).get("passed")
        for run in runs
    ]
    verdict = "PASS" if passed and all(value is True for value in passed) else "FAIL"
    return [
        ("FHE EER", _format_metric(_mean_path(runs, "Quality", encrypted, "eer"))),
        ("FHE TAR@1%", _format_metric(_mean_path(runs, "Quality", encrypted, "tar_at_far_1pct"))),
        ("FHE TAR@0.1%", _format_metric(_mean_path(runs, "Quality", encrypted, "tar_at_far_01pct"))),
        ("ArcFace EER", _format_metric(_mean_path(runs, "Quality", arcface, "eer"))),
        ("ArcFace TAR@1%", _format_metric(_mean_path(runs, "Quality", arcface, "tar_at_far_1pct"))),
        ("ArcFace TAR@0.1%", _format_metric(_mean_path(runs, "Quality", arcface, "tar_at_far_01pct"))),
        ("EER gap", _format_metric(_mean_path(runs, "Quality", comparison, "eer_gap"))),
        ("TAR@1% gap", _format_metric(_mean_path(runs, "Quality", comparison, "tar_at_far_1pct_gap"))),
        ("TAR@0.1% gap", _format_metric(_mean_path(runs, "Quality", comparison, "tar_at_far_01pct_gap"))),
        ("Max EER gap", _format_metric(_mean_path(runs, "Quality", comparison, "maximum_eer_gap"))),
        ("Result", verdict),
    ]


def _render(size: str, display_name: str, runs: list[dict], date: str) -> str:
    quality = _quality_columns(size, runs)
    memory = _same_path_value(
        runs, "Provenance", "Harness environment", "memory_bytes"
    )
    groups = [
        ("Submitter", "submitter", [
            ("Name", '<a href="https://github.com/fhe-benchmarking/face-recognition">CryptoFace</a>'),
            ("Date", html.escape(date)),
            ("CPU", _same_path_value(
                runs, "Provenance", "Harness environment", "cpu_model"
            )),
            ("RAM", _format_bytes(int(memory)) if memory != "-" else "-"),
            ("Orion", _same_path_value(
                runs, "Provenance", "Submission", "orion_commit"
            )[:12]),
            ("Slots", _same_path_value(
                runs, "Provenance", "Submission", "pair_slots"
            )),
            ("Chunk", _same_path_value(
                runs, "Provenance", "Submission", "stage_chunk_pairs"
            )),
            ("R/L", "L"),
            ("Runs", str(len(runs))),
        ]),
        ("Bandwidth", "bandwidth", [
            ("Keys", _same_value(runs, "Bandwidth", "Public and evaluation keys")),
            ("Model", _same_value(runs, "Bandwidth", "Packed model")),
            ("Input", _same_value(runs, "Bandwidth", "Encrypted input")),
            ("Result", _same_value(runs, "Bandwidth", "Encrypted results")),
        ]),
        ("Quality", "quality", quality),
        ("Timing (harness)", "harness-timing", [
            (label, _format_duration(_mean_seconds(runs, "Timing", key)))
            for key, label in TIMING_COLUMNS
        ]),
        ("Timing (server)", "server-timing", [
            (label, _format_duration(_mean_seconds(runs, "Server Reported", key)))
            for key, label in SERVER_COLUMNS
        ]),
    ]
    group_headers = "\n".join(
        f'                    <th colspan="{len(columns)}">{html.escape(name)}</th>'
        for name, _, columns in groups
    )
    column_headers = "\n".join(
        f'                    <th class="{css}-header">{html.escape(label)}</th>'
        for _, css, columns in groups
        for label, _ in columns
    )
    cells = "\n".join(
        f"                    <td>{value}</td>"
        for _, _, columns in groups
        for _, value in columns
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>FHE Benchmarking Results - Face Recognition ({html.escape(display_name)})</title>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.colVis.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 16px; color: #17202a; }}
        h1 {{ font-size: 1.5rem; letter-spacing: 0; }}
        .table-wrap {{ width: 100%; overflow-x: auto; }}
        #resultsTable {{ border-collapse: collapse; width: 100%; }}
        #resultsTable th, #resultsTable td {{ padding: 4px 8px; white-space: nowrap; }}
        #resultsTable thead tr:first-child th {{ background: #263645; color: white; text-align: center; }}
        #resultsTable thead tr:nth-child(2) th {{ color: white; font-size: 0.85rem; }}
        .submitter-header {{ background: #2471a3 !important; }}
        .bandwidth-header {{ background: #1e8449 !important; }}
        .quality-header {{ background: #7d3c98 !important; }}
        .harness-timing-header {{ background: #b05d0b !important; }}
        .server-timing-header {{ background: #2874a6 !important; }}
        #resultsTable tbody tr:nth-child(odd) {{ background: #edf2f4; }}
        .dt-buttons {{ margin-bottom: 10px; }}
    </style>
</head>
<body>
    <h1>FHE Benchmarking Results - Face Recognition ({html.escape(display_name)})</h1>
    <p>Timing and batched quality values are averages across {len(runs)} measurement run(s). Single-pair scores and labels are listed per run. Durations are wall time unless labeled as worker time.</p>
    <div class="table-wrap">
        <table id="resultsTable" class="display">
            <thead>
                <tr>
{group_headers}
                </tr>
                <tr>
{column_headers}
                </tr>
            </thead>
            <tbody>
                <tr>
{cells}
                </tr>
            </tbody>
        </table>
    </div>
    <script>
        $(document).ready(function() {{
            $('#resultsTable').DataTable({{
                dom: 'Bfrtip',
                buttons: [{{ extend: 'colvis' }}],
                paging: false,
                searching: false,
                info: false,
                scrollX: true
            }});
        }});
    </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        help="Submission date shown in every table (default: newest measurement mtime)",
    )
    args = parser.parse_args()

    for size, (filename, display_name) in SIZES.items():
        paths = sorted((MEASUREMENTS / size).glob("results-*.json"))
        if not paths:
            raise FileNotFoundError(f"No measurements found for {size}")
        runs = [json.loads(path.read_text()) for path in paths]
        newest = max(path.stat().st_mtime for path in paths)
        date = args.date or datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M:%S")
        output = WEBSITE / f"{filename}.html"
        output.write_text(_render(size, display_name, runs, date))
        print(f"Wrote {output.relative_to(ROOT)} from {len(runs)} run(s)")


if __name__ == "__main__":
    main()
