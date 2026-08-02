#!/usr/bin/env python3
"""Materialize the selected run as a bounded-memory indexed input store."""

import sys

import numpy as np

from face_dataset_store import STORE_NAME, materialize_selection_store
from params import InstanceParams


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: materialize_input_store.py <size>")
    params = InstanceParams(int(sys.argv[1]))
    with np.load(params.get_selection_file()) as selection:
        indices = selection["indices"]
        if len(indices) != params.get_batch_size():
            raise ValueError(
                f"Expected {params.get_batch_size()} selected pairs, "
                f"found {len(indices)}"
            )
        materialize_selection_store(
            params.rootdir / "datasets" / STORE_NAME,
            indices,
            params.get_test_input_file(),
        )
    print(
        f"[harness] Materialized {len(indices)} pairs -> "
        f"{params.get_test_input_file()}"
    )


if __name__ == "__main__":
    main()
