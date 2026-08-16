import os
import io
import sys
import yaml
import time
import hashlib
import json
import logging
import warnings
import numpy as np
import torch
from contextlib import contextmanager
from pathlib import Path
from PIL import Image
from utils.preprocessing import extract_patches

# Computed once at import time; submission/ is one level below the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PAIR_INDEX_WIDTH = 12


def get_repo_root() -> Path:
    """Return the repository root directory."""
    return _REPO_ROOT


def pair_stem(index: int) -> str:
    """Return a stable pair identifier that scales beyond four-digit batches."""
    if index < 0:
        raise ValueError("Pair indices must be non-negative")
    return f"p{index:0{_PAIR_INDEX_WIDTH}d}"


def decode_image(encoded) -> np.ndarray:
    """Decode an encoded image into CHW uint8 RGB form."""
    payload = np.asarray(encoded, dtype=np.uint8).tobytes()
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1).copy()


def mute_logs() -> None:
    """Reduce third-party logging while keeping submission milestones visible."""
    for name in ("matplotlib", "onnxruntime", "insightface", "PIL"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\..*")


@contextmanager
def suppress_third_party_output():
    """Temporarily silence native/Python chatter around noisy library calls."""
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.close(devnull_fd)


def _resolve_checkpoint(cfg: dict) -> str:
    """
    Return a local path to the model checkpoint.

    If the file at cfg['ckpt_path'] already exists it is used as-is (offline /
    local override). Otherwise the checkpoint is downloaded from the Hugging
    Face repo named in cfg['ckpt_hf_repo'] and cached under ~/.cache/huggingface.
    """
    local = cfg.get("ckpt_path")
    if local and Path(local).exists():
        return local

    repo = cfg.get("ckpt_hf_repo")
    fname = cfg.get("ckpt_hf_file")
    if not repo or not fname:
        raise FileNotFoundError(
            f"Checkpoint not found at {local!r} and no ckpt_hf_repo/ckpt_hf_file "
            f"configured to download it from Hugging Face."
        )
    from huggingface_hub import hf_hub_download
    print(f"[submission] Downloading checkpoint {fname} from HF repo {repo} ...",
          flush=True)
    return hf_hub_download(repo_id=repo, filename=fname)


def load_submission_config(resolve_checkpoint: bool = True) -> dict:
    """
    Reads submission/config.yml relative to repo root.
    Repo root is Path(__file__).parents[1] (submission/ → repo_root/).
    Returns the 'cryptoface' sub-dict from the yaml.
    Relative paths (orion_config) are resolved against repo_root; the checkpoint
    is resolved locally-or-from-Hugging-Face via _resolve_checkpoint.
    """
    config_path = _REPO_ROOT / "submission" / "config.yml"
    with open(config_path) as f:
        full_cfg = yaml.safe_load(f)
    cfg = full_cfg["cryptoface"]
    if "ckpt_path" in cfg and not Path(cfg["ckpt_path"]).is_absolute():
        cfg["ckpt_path"] = str((_REPO_ROOT / cfg["ckpt_path"]).resolve())
    for key in ("orion_config", "circuit_manifest"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((_REPO_ROOT / cfg[key]).resolve())
    if resolve_checkpoint:
        cfg["ckpt_path"] = _resolve_checkpoint(cfg)
    return cfg


def get_face_params(size: int):
    """
    Returns InstanceParams(size, rootdir=repo_root).
    Adds <repo_root>/harness/ to sys.path so params.py is importable.
    """
    harness_dir = str(_REPO_ROOT / "harness")
    if harness_dir not in sys.path:
        sys.path.insert(0, harness_dir)
    from params import InstanceParams
    return InstanceParams(size, rootdir=_REPO_ROOT)


def parse_stage_args(resolve_checkpoint: bool = True) -> tuple:
    """
    Parse the size argument common to all submission stage scripts.
    Reads sys.argv[1] as the instance size, loads config, and constructs params.

    Returns: (size, cfg, params)
    Exits with a usage message if the size argument is missing.
    """
    if len(sys.argv) < 2:
        script = Path(sys.argv[0]).stem
        print(f"Usage: {script} <size>", flush=True)
        sys.exit(1)
    mute_logs()
    size = int(sys.argv[1])
    cfg = load_submission_config(resolve_checkpoint=resolve_checkpoint)
    params = get_face_params(size)
    return size, cfg, params


def align_face(detector, img_chw_rgb: np.ndarray, output_size: int) -> np.ndarray | None:
    """
    Detect and align one face using InsightFace norm_crop, then resize to
    output_size x output_size. For output_size <= 112 the crop is always done
    at 112 first (better landmark precision), then downsampled. For larger
    sizes norm_crop is called directly at output_size.

    Args:
        detector: InsightFace FaceAnalysis app (already prepare()d)
        img_chw_rgb: (3, H, W) uint8 RGB
        output_size: target crop size in pixels (e.g., 64)

    Returns:
        (output_size, output_size, 3) uint8 RGB, or None if no face detected
    """
    import cv2
    from insightface.utils.face_align import norm_crop
    img_hwc_bgr = img_chw_rgb.transpose(1, 2, 0)[:, :, ::-1]
    faces = detector.get(img_hwc_bgr)
    if not faces:
        return None
    if output_size > 112:
        aligned_bgr = norm_crop(img_hwc_bgr, faces[0].kps, image_size=output_size)
    else:
        aligned_bgr = norm_crop(img_hwc_bgr, faces[0].kps, image_size=112)
        if output_size != 112:
            aligned_bgr = cv2.resize(aligned_bgr, (output_size, output_size),
                                     interpolation=cv2.INTER_LINEAR)
    return aligned_bgr[:, :, ::-1].copy()  # return RGB


def _center_crop_resize(img_chw_rgb: np.ndarray, output_size: int) -> np.ndarray:
    """Fallback: center crop to square then resize to output_size x output_size."""
    import cv2
    hwc = img_chw_rgb.transpose(1, 2, 0)
    h, w = hwc.shape[:2]
    side = min(h, w)
    top  = (h - side) // 2
    left = (w - side) // 2
    cropped = hwc[top:top+side, left:left+side]
    return cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def to_tensor(img_hwc_rgb: np.ndarray) -> torch.Tensor:
    """uint8 HWC RGB -> float32 (1, 3, H, W) normalized to [-1, 1]."""
    t = torch.from_numpy(img_hwc_rgb).permute(2, 0, 1).float()
    t = t / 255.0
    t = (t - 0.5) / 0.5
    return t.unsqueeze(0)


def preprocess_one_image(detector, img_chw_uint8: np.ndarray, input_size: int) -> list:
    """
    Preprocess one image into a list of patch tensors for Orion/CryptoFace inference.

    Args:
        detector: InsightFace FaceAnalysis app (already prepare()d)
        img_chw_uint8: (3, H, W) uint8 RGB
        input_size: target aligned face size in pixels (e.g., 64)

    Returns:
        list of N patch tensors, each (1, 3, 32, 32) float32
    """
    aligned = align_face(detector, img_chw_uint8, input_size)
    if aligned is None:
        print("[common] Warning: no face detected — falling back to center crop", flush=True)
        aligned = _center_crop_resize(img_chw_uint8, input_size)

    tensor = to_tensor(aligned)  # (1, 3, input_size, input_size) float32
    patches = extract_patches(tensor)  # list of N tensors, each (1, 3, 32, 32)
    return patches


def load_detector():
    """Load InsightFace FaceAnalysis for face detection and alignment.

    Runs on CPU (onnxruntime) so the submission stays portable on machines
    without a GPU.
    """
    from insightface.app import FaceAnalysis
    with suppress_third_party_output():
        app = FaceAnalysis(name='buffalo_l', providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def _release_fd_cache(fd: int) -> None:
    """Release completed sequential file I/O from the kernel page cache."""
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or dontneed is None:
        return
    try:
        fadvise(fd, 0, 0, dontneed)
    except OSError:
        # Cache eviction is an optimization and may be unsupported by the FS.
        pass


def release_file_cache(path: Path) -> None:
    """Advise the kernel that a fully consumed artifact need not stay cached."""
    try:
        with Path(path).open("rb", buffering=0) as stream:
            _release_fd_cache(stream.fileno())
    except OSError:
        pass


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
        _release_fd_cache(stream.fileno())
    return digest.hexdigest()


def load_circuit_manifest(cfg: dict) -> dict:
    with open(cfg["circuit_manifest"], encoding="utf-8") as stream:
        return json.load(stream)


def _canonical_hash(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_cache_manifest(directory: Path, files: list[str], metadata: dict) -> dict:
    records = {}
    for name in files:
        path = directory / name
        records[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": 1,
        "complete": True,
        "metadata": metadata,
        "files": records,
    }
    path = directory / "cache_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)
    return manifest


def validate_cache_manifest(directory: Path, verify_hashes: bool = True) -> dict:
    path = directory / "cache_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cache manifest: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("complete") is not True:
        raise ValueError(f"Incomplete or unsupported cache manifest: {path}")
    for name, expected in manifest.get("files", {}).items():
        artifact = directory / name
        if not artifact.is_file():
            raise FileNotFoundError(f"Missing cached artifact: {artifact}")
        if artifact.stat().st_size != expected.get("size_bytes"):
            raise ValueError(f"Cached artifact size mismatch: {artifact}")
        if verify_hashes and sha256_file(artifact) != expected.get("sha256"):
            raise ValueError(f"Cached artifact hash mismatch: {artifact}")
    if not manifest.get("files"):
        raise ValueError(f"Cache manifest contains no files: {path}")
    return manifest


def validate_key_cache(cfg: dict, params, include_secret: bool = False) -> None:
    manifest_hash = sha256_file(Path(cfg["circuit_manifest"]))
    public = validate_cache_manifest(params.iodir() / "public_keys")
    if set(public["files"]) != {"keys.h5", "input_level.txt"}:
        raise ValueError("Evaluation-key cache has an unexpected file manifest")
    if public["metadata"].get("circuit_manifest_sha256") != manifest_hash:
        raise ValueError("Evaluation keys were generated for another circuit manifest")
    if include_secret:
        secret = validate_cache_manifest(params.iodir() / "secret_key")
        if set(secret["files"]) != {"sk.h5"}:
            raise ValueError("Secret-key cache has an unexpected file manifest")


def model_cache_identity(cfg: dict) -> dict:
    with open(cfg["orion_config"]) as stream:
        orion_config = yaml.safe_load(stream)
    return {
        "checkpoint_sha256": sha256_file(Path(cfg["ckpt_path"])),
        "orion_config_sha256": _canonical_hash(orion_config),
        "circuit_manifest_sha256": sha256_file(Path(cfg["circuit_manifest"])),
        "orion_commit": cfg["orion_commit"],
    }


def get_server_model_dir(cfg: dict, params) -> tuple[Path, dict]:
    identity = model_cache_identity(cfg)
    cache_key = _canonical_hash(identity)[:24]
    return params.rootdir / "io" / "server_data" / cache_key, identity


def write_server_model_reference(params, model_dir: Path, cache_manifest: dict) -> None:
    reference = {
        "schema_version": 1,
        "model_dir": str(model_dir.resolve()),
        "cache_manifest_sha256": sha256_file(model_dir / "cache_manifest.json"),
        "packed_model_size_bytes": cache_manifest["files"]["diagonals.h5"]["size_bytes"],
        "packed_model_sha256": cache_manifest["files"]["diagonals.h5"]["sha256"],
    }
    path = params.iodir() / "server_model.json"
    path.write_text(json.dumps(reference, indent=2) + "\n")


def read_server_model_reference(params) -> tuple[Path, dict]:
    reference_path = params.iodir() / "server_model.json"
    reference = json.loads(reference_path.read_text())
    model_dir = Path(reference["model_dir"])
    manifest_path = model_dir / "cache_manifest.json"
    if sha256_file(manifest_path) != reference["cache_manifest_sha256"]:
        raise ValueError("Server model cache manifest changed after preprocessing")
    manifest = validate_cache_manifest(model_dir, verify_hashes=False)
    packed = manifest["files"]["diagonals.h5"]
    if (packed["size_bytes"] != reference["packed_model_size_bytes"] or
            packed["sha256"] != reference["packed_model_sha256"]):
        raise ValueError("Server model reference does not match packed model cache")
    return model_dir, manifest


def init_orion_scheme(
    cfg: dict,
    params,
    key_io_mode: str,
    diags_io_mode: str = "none",
    load_secret_key: bool = True,
    model_dir: Path | None = None,
):
    """Initialize Orion with independent client-key and server-model modes."""
    import orion
    keys_dir = params.iodir() / "public_keys"
    sk_dir = params.iodir() / "secret_key"
    model_dir = model_dir or (params.rootdir / "io" / "server_data" / "unused")
    keys_dir.mkdir(parents=True, exist_ok=True)
    sk_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg["orion_config"]) as f:
        config = yaml.safe_load(f)
    config["orion"]["io_mode"] = "none"
    config["orion"]["key_io_mode"] = key_io_mode
    config["orion"]["diags_io_mode"] = diags_io_mode
    config["orion"]["diags_path"] = str((model_dir / "diagonals.h5").resolve())
    config["orion"]["keys_path"]  = str((keys_dir / "keys.h5").resolve())
    config["orion"]["sk_path"]    = str((sk_dir / "sk.h5").resolve())
    config["orion"]["load_secret_key"] = load_secret_key
    with suppress_third_party_output():
        scheme = orion.init_scheme(config)
    if key_io_mode == "load":
        release_file_cache(keys_dir / "keys.h5")
        if load_secret_key:
            release_file_cache(sk_dir / "sk.h5")
    return scheme


def deterministic_fit_patches(cfg: dict) -> list[torch.Tensor]:
    """Return deterministic shape inputs; no benchmark image enters compilation."""
    manifest = load_circuit_manifest(cfg)
    patches = []
    for index, shape in enumerate(manifest["circuit"]["input_shapes"]):
        values = torch.linspace(-1.0, 1.0, int(np.prod(shape)), dtype=torch.float32)
        patches.append(torch.roll(values.reshape(shape), shifts=index, dims=-1))
    return patches


def _build_model_pipeline(cfg: dict):
    """Construct the CryptoFace model and per-image Orion pipeline (shared by the
    save and load paths). Returns (model, pipeline)."""
    from models.cryptoface_pcnn import CryptoFaceNet
    from models.pipeline import PerImagePipeline
    from models.weight_loader import load_cryptoface_checkpoint

    a, b, c = cfg["l2_poly_coeffs"]
    model = CryptoFaceNet(cfg["input_size"], l2_norm_coeffs=(a, b, c))
    load_cryptoface_checkpoint(model, cfg["ckpt_path"])
    for net in model.nets:
        net.init_orion_params()
    pipeline = PerImagePipeline(
        backbones=model.nets,
        linears=model.linear,
        normalization=model.normalization,
    )
    pipeline.eval()
    return model, pipeline


def build_pipeline_server_save(cfg: dict, params, model_dir: Path) -> int:
    """Server-owned checkpoint loading, circuit compilation, and model packing."""
    import orion

    t0 = time.time()
    with suppress_third_party_output():
        _model, pipeline = _build_model_pipeline(cfg)
        init_orion_scheme(
            cfg, params, "load", "save",
            load_secret_key=False, model_dir=model_dir,
        )
        orion.fit(pipeline, deterministic_fit_patches(cfg))
        input_level = orion.compile(pipeline)
        orion.validate_compiled_manifest(load_circuit_manifest(cfg))
    print(
        f"[model-prep] Packed server model in {time.time()-t0:.1f}s "
        f"(input_level={input_level})",
        flush=True,
    )
    return input_level


def build_pipeline_load(cfg: dict, params) -> tuple:
    """
    Load the Orion pipeline from server-owned packed model data and uploaded
    public/evaluation keys. The secret key is never loaded.

    Returns the pipeline, dimensions, and setup timing breakdown.
    """
    import orion
    from orion.core import scheme as _scheme

    t0 = time.time()
    model_dir, _cache = read_server_model_reference(params)

    with suppress_third_party_output():
        model, pipeline = _build_model_pipeline(cfg)
        init_orion_scheme(
            cfg, params, "load", "load",
            load_secret_key=False, model_dir=model_dir,
        )
        orion.fit(pipeline, deterministic_fit_patches(cfg))
        compile_t0 = time.time()
        orion.compile(pipeline)
        compile_s = time.time() - compile_t0
        orion.validate_compiled_manifest(load_circuit_manifest(cfg))
        preload_t0 = time.time()
        _scheme.lt_evaluator.preload_all(pipeline)
        model_io_s = time.time() - preload_t0
        release_file_cache(model_dir / "diagonals.h5")

    pipeline.he()

    n_patches = model.N
    embedding_dim = model.embedding_dim
    setup_s = time.time() - t0
    print(
        f"[server-setup] Pipeline ready in {setup_s:.1f}s "
        f"(compile={compile_s:.1f}s, model_io={model_io_s:.1f}s)",
        flush=True,
    )
    return (
        pipeline, embedding_dim, n_patches,
        {"compile_s": compile_s, "model_io_s": model_io_s},
    )
