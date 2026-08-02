"""Random-access encoded face-pair storage for bounded-memory evaluation."""

import io
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


STORE_NAME = "face_dataset.h5"
_SCHEMA_VERSION = 1


def _encoded_bytes(value) -> bytes:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return bytes(value)
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"Unsupported face image shape: {array.shape}")
    if array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _release_file_cache(path: Path) -> None:
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or dontneed is None:
        return
    try:
        with path.open("rb", buffering=0) as stream:
            fadvise(stream.fileno(), 0, 0, dontneed)
    except OSError:
        pass


def validate_pair_store(store_path: Path) -> tuple[int, int]:
    """Validate the indexed dataset and return pair and genuine-pair counts."""
    with h5py.File(store_path, "r") as store:
        if store.attrs.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(f"Unsupported indexed dataset schema: {store_path}")
        if not {"image0", "image1", "labels"}.issubset(store):
            raise ValueError(f"Indexed dataset is missing required arrays: {store_path}")
        pair_count = len(store["labels"])
        if pair_count < 1:
            raise ValueError("Face dataset is empty")
        if len(store["image0"]) != pair_count or len(store["image1"]) != pair_count:
            raise ValueError("Indexed dataset image and label counts do not match")
        genuine_count = 0
        labels = store["labels"]
        for offset in range(0, pair_count, 1_000_000):
            block = labels[offset:offset + 1_000_000]
            invalid = sorted(set(int(value) for value in block) - {0, 1})
            if invalid:
                raise ValueError(f"Face labels must be 0 or 1, got {invalid}")
            genuine_count += int(np.sum(block))
    return pair_count, genuine_count


def ensure_pair_store(npy_path: Path, labels_path: Path) -> Path:
    """Create an indexed HDF5 store once from the legacy object-array format."""
    store_path = npy_path.with_name(STORE_NAME)
    source_size = npy_path.stat().st_size
    labels_size = labels_path.stat().st_size
    if store_path.is_file():
        try:
            with h5py.File(store_path, "r") as store:
                matches_source = (
                    store.attrs.get("schema_version") == _SCHEMA_VERSION
                    and store.attrs.get("source_size") == source_size
                    and store.attrs.get("labels_size") == labels_size
                    and len(store["labels"]) > 0
                )
            if matches_source:
                validate_pair_store(store_path)
                return store_path
        except (OSError, KeyError):
            pass

    data = np.load(npy_path, allow_pickle=True)
    labels = np.asarray(
        [int(line) for line in labels_path.read_text().splitlines() if line],
        dtype=np.int8,
    )
    if len(data) != len(labels) or len(data) == 0:
        raise ValueError(
            f"Invalid face dataset: pairs={len(data)}, labels={len(labels)}"
        )

    temporary = store_path.with_suffix(".h5.tmp")
    vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
    with h5py.File(temporary, "w") as store:
        image0 = store.create_dataset("image0", (len(data),), dtype=vlen_uint8)
        image1 = store.create_dataset("image1", (len(data),), dtype=vlen_uint8)
        store.create_dataset("labels", data=labels)
        for index, pair in enumerate(data):
            image0[index] = np.frombuffer(_encoded_bytes(pair[0]), dtype=np.uint8)
            image1[index] = np.frombuffer(_encoded_bytes(pair[1]), dtype=np.uint8)
        store.attrs["schema_version"] = _SCHEMA_VERSION
        store.attrs["source_size"] = source_size
        store.attrs["labels_size"] = labels_size
    temporary.replace(store_path)
    del data
    _release_file_cache(npy_path)
    _release_file_cache(store_path)
    validate_pair_store(store_path)
    return store_path


def decode_image(encoded) -> np.ndarray:
    """Decode one encoded image into CHW uint8 RGB form."""
    image = Image.open(io.BytesIO(np.asarray(encoded, dtype=np.uint8).tobytes()))
    return np.asarray(image.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)


def materialize_selection_store(
    store_path: Path, indices: np.ndarray, output_path: Path,
) -> None:
    """Copy a run selection into an indexed, label-free input store."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".h5.tmp")
    vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
    with h5py.File(store_path, "r") as source, h5py.File(temporary, "w") as output:
        image0 = output.create_dataset("image0", (len(indices),), dtype=vlen_uint8)
        image1 = output.create_dataset("image1", (len(indices),), dtype=vlen_uint8)
        for output_index, source_index in enumerate(indices):
            image0[output_index] = source["image0"][int(source_index)]
            image1[output_index] = source["image1"][int(source_index)]
        output.attrs["schema_version"] = _SCHEMA_VERSION
        output.attrs["pair_count"] = len(indices)
    temporary.replace(output_path)
    _release_file_cache(store_path)
    _release_file_cache(output_path)
