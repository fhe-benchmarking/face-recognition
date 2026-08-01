#!/usr/bin/env python3
"""Evaluate encrypted face pairs with bounded process and memory lifetimes."""

import os

os.environ.setdefault("GOGC", "20")
os.environ.setdefault("GODEBUG", "madvdontneed=1")

import gc
import json
import multiprocessing
import sys
import time
import traceback
from collections import deque
from multiprocessing.connection import wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import build_pipeline_load, parse_stage_args


_pipeline = None
_embedding_dim = None
_n_patches = None
_upload_dir = None


def _error(exc):
    return {
        "status": "fail",
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _branch_entry(pair_idx, image_idx, branch_idx, connection):
    """Run one image/backbone branch and exit after returning its ciphertext."""
    try:
        from orion.backend.python.tensors import CipherTensor
        from orion.core import scheme

        total_t0 = time.time()
        ciphertext = CipherTensor.load(
            scheme,
            _upload_dir / f"p{pair_idx:04d}_i{image_idx}_b{branch_idx}.h5",
        )
        deserialize_s = time.time() - total_t0

        forward_t0 = time.time()
        feature = getattr(_pipeline, f"linear{branch_idx}")(
            getattr(_pipeline, f"backbone{branch_idx}")(ciphertext)
        )
        forward_s = time.time() - forward_t0

        serialize_t0 = time.time()
        feature_serialized = feature.serialize()
        connection.send({
            "status": "pass",
            "pair_idx": pair_idx,
            "image_idx": image_idx,
            "branch_idx": branch_idx,
            "deserialize_s": deserialize_s,
            "forward_s": forward_s,
            "serialize_s": time.time() - serialize_t0,
            "total_s": time.time() - total_t0,
            "feature_serialized": feature_serialized,
        })
    except BaseException as exc:
        try:
            connection.send(_error(exc))
        except BaseException:
            pass
        raise
    finally:
        connection.close()


def _terminate(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.kill()
            process.join()


def _run_branches(ctx, pair_idx, timeout_s):
    processes, pending, results = [], {}, []
    started = time.time()
    try:
        for image_idx in range(2):
            for branch_idx in range(_n_patches):
                receive, send = ctx.Pipe(duplex=False)
                process = ctx.Process(
                    target=_branch_entry,
                    args=(pair_idx, image_idx, branch_idx, send),
                )
                process.start()
                send.close()
                processes.append(process)
                pending[receive] = process

        deadline = time.monotonic() + timeout_s
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Pair {pair_idx} branch generation timed out")
            for connection in wait(list(pending), timeout=min(5.0, remaining)):
                process = pending.pop(connection)
                try:
                    result = connection.recv()
                except EOFError as exc:
                    raise RuntimeError(
                        f"Branch worker {process.pid} exited without a result"
                    ) from exc
                finally:
                    connection.close()
                if result["status"] != "pass":
                    raise RuntimeError(
                        f"Pair {pair_idx} branch failed: {result.get('type')}: "
                        f"{result.get('message')}\n{result.get('traceback')}"
                    )
                results.append(result)

        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Branch worker {process.pid} exited with {process.exitcode}"
                )
    except BaseException:
        _terminate(processes)
        for connection in pending:
            connection.close()
        raise

    results.sort(key=lambda item: (item["image_idx"], item["branch_idx"]))
    return time.time() - started, results


def _slot_loop(connection, timeout_s):
    """Remain FHE-quiescent and fork a clean generation for each pair."""
    ctx = multiprocessing.get_context("fork")
    try:
        while True:
            command = connection.recv()
            if command["op"] == "stop":
                connection.send({"status": "stopped"})
                return
            try:
                wall_s, results = _run_branches(ctx, command["pair_idx"], timeout_s)
                connection.send({
                    "status": "pass",
                    "pair_idx": command["pair_idx"],
                    "branch_wall_s": wall_s,
                    "branch_results": results,
                })
                del results
                gc.collect()
            except BaseException as exc:
                connection.send(_error(exc))
                return
    finally:
        connection.close()


def _aggregate(command, output_dir):
    from orion.backend.python.tensors import CipherTensor
    from orion.core import scheme
    from utils.he_operations import compute_inner_product_encrypted, tree_reduce_add

    pair_idx = command["pair_idx"]
    started = time.time()
    by_image = {0: {}, 1: {}}
    for result in command["branch_results"]:
        by_image[result["image_idx"]][result["branch_idx"]] = (
            result["feature_serialized"]
        )
    features = [
        [
            CipherTensor.deserialize(scheme, by_image[image][branch])
            for branch in range(_n_patches)
        ]
        for image in range(2)
    ]
    deserialize_s = time.time() - started

    normalization_t0 = time.time()
    embeddings = [
        _pipeline.normalization(tree_reduce_add(image_features))
        for image_features in features
    ]
    normalization_s = time.time() - normalization_t0

    inner_product_t0 = time.time()
    score = compute_inner_product_encrypted(
        embeddings[0], embeddings[1], _embedding_dim
    )
    inner_product_s = time.time() - inner_product_t0
    serialize_t0 = time.time()
    score.save(output_dir / f"p{pair_idx:04d}_score.h5")
    serialize_s = time.time() - serialize_t0

    del by_image, features, embeddings, score
    gc.collect()
    return {
        "pair_idx": pair_idx,
        "feature_deserialize_s": deserialize_s,
        "normalization_s": normalization_s,
        "inner_product_s": inner_product_s,
        "serialize_s": serialize_s,
        "total_s": time.time() - started,
    }


def _aggregator_generation_loop(connection, output_dir):
    try:
        while True:
            command = connection.recv()
            if command["op"] == "stop":
                connection.send({"status": "stopped"})
                return
            try:
                connection.send({
                    "status": "pass",
                    "result": _aggregate(command, output_dir),
                })
            except BaseException as exc:
                connection.send(_error(exc))
                return
    finally:
        connection.close()


def _start(ctx, target, *args):
    parent, child = ctx.Pipe(duplex=True)
    process = ctx.Process(target=target, args=(child, *args))
    process.start()
    child.close()
    return process, parent


def _stop(process, connection):
    if process.is_alive() and not connection.closed:
        try:
            connection.send({"op": "stop"})
            if connection.poll(30):
                connection.recv()
        except BaseException:
            pass
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=30)
    if process.is_alive():
        process.kill()
        process.join()
    if not connection.closed:
        connection.close()


def _aggregator_manager_loop(connection, output_dir, max_pairs):
    """Create bounded aggregator children from a permanently quiescent parent."""
    ctx = multiprocessing.get_context("fork")
    process = child_connection = None
    generation = generation_pairs = 0
    try:
        while True:
            command = connection.recv()
            if command["op"] == "stop":
                if process is not None:
                    _stop(process, child_connection)
                connection.send({"status": "stopped"})
                return
            if process is None:
                process, child_connection = _start(
                    ctx, _aggregator_generation_loop, output_dir
                )
                generation += 1
                generation_pairs = 0
            child_connection.send(command)
            response = child_connection.recv()
            response["generation"] = generation
            connection.send(response)
            generation_pairs += 1
            del command, response
            gc.collect()
            if generation_pairs >= max_pairs:
                _stop(process, child_connection)
                process = child_connection = None
    finally:
        if process is not None:
            _stop(process, child_connection)
        connection.close()


def _pair_indices(upload_dir):
    indices = sorted(
        int(path.name[1:5]) for path in upload_dir.glob("p????_i0_b0.h5")
    )
    if not indices:
        raise FileNotFoundError(f"No encrypted pairs found in {upload_dir}")
    if indices != list(range(len(indices))):
        raise ValueError("Encrypted pair indices must be contiguous from zero")
    missing = [
        (pair, image, branch)
        for pair in indices
        for image in range(2)
        for branch in range(_n_patches)
        if not (upload_dir / f"p{pair:04d}_i{image}_b{branch}.h5").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} encrypted branch inputs")
    return indices


def _write_report(
    params, total_s, setup_s, setup_details, compute_s, results, pair_slots
):
    branches = [item for result in results for item in result["branch_results"]]
    aggregations = [result["aggregation"] for result in results]
    backbone_s = sum(item["forward_s"] for item in branches)
    normalization_s = sum(item["normalization_s"] for item in aggregations)
    inner_product_s = sum(item["inner_product_s"] for item in aggregations)
    input_io_s = sum(item["deserialize_s"] for item in branches)
    transport_s = (
        sum(item["serialize_s"] for item in branches)
        + sum(item["feature_deserialize_s"] for item in aggregations)
        + sum(item["serialize_s"] for item in aggregations)
    )
    report = {
        "Encrypted computation": round(compute_s, 4),
        "Total": round(total_s, 4),
        "Pipeline load and key setup": round(setup_s, 4),
        "Packed model I/O": round(setup_details.get("model_io_s", 0.0), 4),
        "Runtime circuit compilation": round(
            setup_details.get("compile_s", 0.0), 4
        ),
        "Ciphertext input I/O worker-seconds": round(input_io_s, 4),
        "Ciphertext transport worker-seconds": round(transport_s, 4),
        "Encrypted inference worker-seconds": round(
            backbone_s + normalization_s + inner_product_s, 4
        ),
        "Backbone forward worker-seconds": round(backbone_s, 4),
        "Normalization worker-seconds": round(normalization_s, 4),
        "Inner product worker-seconds": round(inner_product_s, 4),
        "Mean encrypted wall time per pair": round(compute_s / len(results), 4)
        if results else 0.0,
        "Timing semantics": {
            "wall_time": [
                "Encrypted computation",
                "Total",
                "Pipeline load and key setup",
                "Packed model I/O",
                "Runtime circuit compilation",
                "Mean encrypted wall time per pair",
            ],
            "summed_worker_time": [
                "Ciphertext input I/O worker-seconds",
                "Ciphertext transport worker-seconds",
                "Encrypted inference worker-seconds",
                "Backbone forward worker-seconds",
                "Normalization worker-seconds",
                "Inner product worker-seconds",
            ],
        },
    }
    path = params.iodir() / "server_reported.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def main():
    global _pipeline, _embedding_dim, _n_patches, _upload_dir

    stage_t0 = time.time()
    _size, cfg, params = parse_stage_args()
    pair_slots = int(cfg.get("pair_slots", 5))
    aggregator_max_pairs = int(cfg.get("aggregator_max_pairs", 10))
    timeout_s = int(cfg.get("pair_timeout_s", 3600))
    if min(pair_slots, aggregator_max_pairs, timeout_s) < 1:
        raise ValueError("Server concurrency settings must be positive")

    setup_t0 = time.time()
    (
        _pipeline, _level, _embedding_dim, _n_patches, setup_details
    ) = build_pipeline_load(cfg, params)
    setup_s = time.time() - setup_t0
    _upload_dir = params.iodir() / "ciphertexts_upload"
    output_dir = params.iodir() / "ciphertexts_download"
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = _pair_indices(_upload_dir)
    if len(indices) != params.get_batch_size():
        raise ValueError(
            f"Expected {params.get_batch_size()} encrypted pairs, "
            f"found {len(indices)}"
        )
    pending = deque(
        pair for pair in indices
        if not (output_dir / f"p{pair:04d}_score.h5").exists()
    )
    slot_count = min(pair_slots, len(pending))
    print(
        f"[server_encrypted_compute] pairs={len(indices)} pending={len(pending)} "
        f"pair_slots={slot_count} max_branch_workers={2 * _n_patches * slot_count}",
        flush=True,
    )
    if not pending:
        _write_report(
            params, time.time() - stage_t0, setup_s, setup_details,
            0.0, [], slot_count,
        )
        return

    ctx = multiprocessing.get_context("fork")
    slots = []
    for slot_id in range(slot_count):
        process, connection = _start(ctx, _slot_loop, timeout_s)
        slots.append({
            "id": slot_id,
            "process": process,
            "connection": connection,
            "pair": None,
            "started": None,
        })
    aggregator, aggregator_connection = _start(
        ctx, _aggregator_manager_loop, output_dir, aggregator_max_pairs
    )

    ready, in_aggregation, results = deque(), None, []
    compute_t0 = time.time()
    try:
        while pending or any(s["pair"] is not None for s in slots) or ready or in_aggregation:
            for slot in slots:
                if slot["pair"] is None and pending:
                    slot["pair"] = pending.popleft()
                    slot["started"] = time.time()
                    slot["connection"].send({"op": "run", "pair_idx": slot["pair"]})
                    print(
                        f"[server] pair {slot['pair'] + 1}/{len(indices)} -> slot {slot['id']}",
                        flush=True,
                    )

            for connection in wait([s["connection"] for s in slots], timeout=0.25):
                slot = next(s for s in slots if s["connection"] is connection)
                response = connection.recv()
                if response["status"] != "pass":
                    raise RuntimeError(
                        f"Slot {slot['id']} failed pair {slot['pair']}: "
                        f"{response.get('message')}\n{response.get('traceback')}"
                    )
                ready.append({
                    "pair_idx": slot["pair"],
                    "pair_started": slot["started"],
                    "branch_wall_s": response["branch_wall_s"],
                    "branch_results": response["branch_results"],
                })
                print(
                    f"[server] pair {slot['pair'] + 1}/{len(indices)} backbones complete "
                    f"[{response['branch_wall_s']:.1f}s]", flush=True
                )
                slot["pair"] = slot["started"] = None

            if in_aggregation is None and ready:
                in_aggregation = ready.popleft()
                aggregator_connection.send({
                    "op": "aggregate",
                    "pair_idx": in_aggregation["pair_idx"],
                    "branch_results": in_aggregation["branch_results"],
                })
                for branch in in_aggregation["branch_results"]:
                    del branch["feature_serialized"]

            if in_aggregation is not None and aggregator_connection.poll():
                response = aggregator_connection.recv()
                if response["status"] != "pass":
                    raise RuntimeError(
                        f"Aggregator failed: {response.get('message')}\n"
                        f"{response.get('traceback')}"
                    )
                in_aggregation["aggregation"] = response["result"]
                in_aggregation["aggregator_generation"] = response["generation"]
                in_aggregation["pair_wall_s"] = time.time() - in_aggregation["pair_started"]
                results.append(in_aggregation)
                print(
                    f"[server] pair {in_aggregation['pair_idx'] + 1}/{len(indices)} saved "
                    f"[{in_aggregation['pair_wall_s']:.1f}s]", flush=True
                )
                in_aggregation = None

            now = time.time()
            for slot in slots:
                if slot["pair"] is not None and now - slot["started"] > timeout_s:
                    raise TimeoutError(f"Pair {slot['pair']} timed out")
                if not slot["process"].is_alive():
                    raise RuntimeError(f"Slot manager {slot['id']} exited unexpectedly")
            if not aggregator.is_alive():
                raise RuntimeError("Aggregator manager exited unexpectedly")
    finally:
        for slot in slots:
            _stop(slot["process"], slot["connection"])
        _stop(aggregator, aggregator_connection)

    compute_s = time.time() - compute_t0
    results.sort(key=lambda item: item["pair_idx"])
    _write_report(
        params, time.time() - stage_t0, setup_s, setup_details,
        compute_s, results, slot_count,
    )
    print(
        f"[server_encrypted_compute] Done: {len(results)} pairs in {compute_s:.1f}s "
        f"({len(results) / compute_s * 3600:.2f} pairs/hour)", flush=True
    )


if __name__ == "__main__":
    main()
