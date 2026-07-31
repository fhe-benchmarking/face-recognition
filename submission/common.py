import os
import sys
import yaml
import time
import numpy as np
import torch
from pathlib import Path

from models.cryptoface_pcnn import CryptoFaceNet
from models.weight_loader import load_cryptoface_checkpoint
from models.pipeline import PerImagePipeline
from utils.preprocessing import extract_patches

# Computed once at import time; submission/ is one level below the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]


def get_repo_root() -> Path:
    """Return the repository root directory."""
    return _REPO_ROOT


def mute_logs() -> None:
    """Silence a submission stage's output, matching the ml-inference reference.

    Redirects stdout and stderr (file descriptors 1 and 2) to /dev/null, which
    also suppresses native-library chatter and Python warnings. Harness logging
    remains visible because the stage runs in a separate process.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
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


def load_submission_config() -> dict:
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
    if "orion_config" in cfg and not Path(cfg["orion_config"]).is_absolute():
        cfg["orion_config"] = str((_REPO_ROOT / cfg["orion_config"]).resolve())
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


def parse_stage_args() -> tuple:
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
    cfg = load_submission_config()
    params = get_face_params(size)
    return size, cfg, params


def decode_master_image(elem) -> np.ndarray:
    """Return a (3, H, W) uint8 RGB array from a master-dataset element.

    The master dataset (datasets/face_dataset.npy) stores original JPEG file
    bytes to keep the committed file small; this decodes those bytes. Raw uint8
    arrays are passed through unchanged. Mirrors the harness's decode in
    generate_input._to_chw_uint8 so the fit sample matches per-run inputs.
    """
    import io as _io
    from PIL import Image
    if isinstance(elem, (bytes, bytearray, np.bytes_)):
        img = Image.open(_io.BytesIO(bytes(elem))).convert("RGB")
        return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
    return np.asarray(elem)


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
    app = FaceAnalysis(name='buffalo_l', providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def init_orion_scheme(cfg: dict, params, io_mode: str,
                      load_secret_key: bool = True) -> None:
    """
    Load orion config yaml, set io_mode and key/diag paths, call orion.init_scheme().

    Key material is split so it mirrors a practical client/server split:
      - secret_key/sk.h5     — the secret key (private; client only)
      - public_keys/keys.h5  — the evaluation keys (relin + galois) a client
                               uploads to the server
      - model_data/diagonals.h5 — the plaintext model diagonals

    Args:
        cfg:     submission config dict (from load_submission_config)
        params:  InstanceParams (provides iodir())
        io_mode: "save" (generates + persists keys) or "load" (reads them)
        load_secret_key: if False (server side), the secret key is neither loaded
                 nor used — only the evaluation keys are.
    """
    import orion
    keys_dir  = params.iodir() / "public_keys"       # evaluation keys (relin + galois)
    sk_dir    = params.iodir() / "secret_key"        # private secret key (client only)
    model_dir = params.iodir() / "model_data"        # plaintext model diagonals
    keys_dir.mkdir(parents=True, exist_ok=True)
    sk_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg["orion_config"]) as f:
        config = yaml.safe_load(f)
    config["orion"]["io_mode"]  = io_mode
    config["orion"]["diags_path"] = str((model_dir / "diagonals.h5").resolve())
    config["orion"]["keys_path"]  = str((keys_dir / "keys.h5").resolve())
    config["orion"]["sk_path"]    = str((sk_dir / "sk.h5").resolve())
    config["orion"]["load_secret_key"] = load_secret_key
    orion.init_scheme(config)


def load_fit_patches(params) -> list:
    """
    Load the fit sample patches saved by client_key_generation.

    Loads params.iodir() / "public_keys" / "fit_sample.npy"
    Shape: (N, 1, 3, 32, 32) float32

    Returns list of N tensors, each (1, 3, 32, 32) float32.
    """
    fit_path = params.iodir() / "public_keys" / "fit_sample.npy"
    arr = np.load(fit_path)  # (N, 1, 3, 32, 32) float32
    return [torch.from_numpy(arr[k].copy()) for k in range(arr.shape[0])]


def _build_model_pipeline(cfg: dict):
    """Construct the CryptoFace model and per-image Orion pipeline (shared by the
    save and load paths). Returns (model, pipeline)."""
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


def build_pipeline_save(cfg: dict, params) -> int:
    """
    Client key generation + model preprocessing (run once): generate a fresh
    secret key and the full evaluation-key set (relin + all galois/rotation keys,
    including the bootstrapping keys), and compile with io_mode=save so those keys
    plus the plaintext model diagonals are persisted. The secret key is written
    to the private secret_key/ file; the evaluation keys (what a client uploads)
    go to public_keys/; the diagonals go to model_data/.

    Runs client-side because generating the rotation keys requires the secret key.
    server_preprocess_model / build_pipeline_load then load the evaluation keys
    without ever touching the secret key.

    Returns: input_level
    """
    import orion

    t0 = time.time()
    model, pipeline = _build_model_pipeline(cfg)

    # io_mode=save: generate + persist sk, relin, galois, and diagonals.
    init_orion_scheme(cfg, params, "save")

    fit_patches = load_fit_patches(params)
    orion.fit(pipeline, fit_patches)
    input_level = orion.compile(pipeline)
    print(f"[common] build_pipeline_save (compile io_mode=save) done in "
          f"{time.time()-t0:.1f}s  input_level={input_level}", flush=True)
    return input_level


def build_pipeline_load(cfg: dict, params) -> tuple:
    """
    Load the Orion pipeline entirely from disk (io_mode=load), using only the
    public/evaluation keys and plaintext diagonals persisted by
    build_pipeline_save. compile() reads them from HDF5 instead of recomputing;
    preload_all() brings them into memory. The secret key is never loaded.

    Returns: (pipeline, input_level, embedding_dim, n_patches)
    """
    import orion
    from orion.core import scheme as _scheme

    t0 = time.time()

    model, pipeline = _build_model_pipeline(cfg)

    # Server side: load only the evaluation keys (relin + galois) and plaintext
    # diagonals from disk — never the secret key (load_secret_key=False). compile()
    # reads them from HDF5 instead of recomputing (~25 min) on every run.
    init_orion_scheme(cfg, params, "load", load_secret_key=False)

    fit_patches = load_fit_patches(params)
    orion.fit(pipeline, fit_patches)
    input_level = orion.compile(pipeline)
    _scheme.lt_evaluator.preload_all(pipeline)

    pipeline.he()

    n_patches = model.N
    embedding_dim = model.embedding_dim
    print(f"[common] build_pipeline_load done in {time.time()-t0:.1f}s  "
          f"input_level={input_level}  n_patches={n_patches}", flush=True)
    return pipeline, input_level, embedding_dim, n_patches
