"""Reproducibility helpers: one call seeds every RNG, and a record of where a run came from."""
import os
import platform
import random
import subprocess

import numpy as np
import torch as th


def set_seed(seed, deterministic=False):
    """Seed Python, NumPy and PyTorch (CPU + all GPUs).

    `deterministic=True` also forces deterministic cuDNN kernels: bit-identical
    reruns on the same GPU, at some speed cost. Seeds alone already make runs
    statistically reproducible, which is what mean +/- std over seeds needs.
    """
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


def pick_device(device="auto"):
    if device == "auto":
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    return th.device(device)


def git_commit(path="."):
    """Commit hash of the code that produced a result ('unknown' outside a git checkout)."""
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run_info(seed=None):
    """Everything needed to trace a result back: code version, seed, libraries, hardware."""
    return {
        "commit": git_commit(os.path.dirname(os.path.abspath(__file__))),
        "seed": seed,
        "python": platform.python_version(),
        "torch": th.__version__,
        "numpy": np.__version__,
        "device": th.cuda.get_device_name(0) if th.cuda.is_available() else "cpu",
    }


def record_environment(out_dir, name):
    """Write the exact software/hardware environment of a notebook run (the locked environment).

    <out_dir>/<name>_pip_freeze.txt and <out_dir>/<name>_env.json
    """
    import json
    import sys
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (out / f"{name}_pip_freeze.txt").write_text(freeze)
    info = run_info()
    info.update(python_full=sys.version, cuda=th.version.cuda, cudnn=th.backends.cudnn.version(),
                platform=platform.platform())
    (out / f"{name}_env.json").write_text(json.dumps(info, indent=2))
    return info
