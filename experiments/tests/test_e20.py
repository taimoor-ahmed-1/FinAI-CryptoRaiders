"""E20: cross-fitted factors (training windows avoid the held-out block) and the entry-gated environment."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rltrade.rnn import crossfit as CF                    # noqa: E402
from rltrade.rnn.train import RnnConfig, SeqData, train_rnn  # noqa: E402


def _seq_prepared(n=3000, f=6, seed=0):
    rng = np.random.default_rng(seed)
    d = Path(tempfile.mkdtemp())
    for s in ("train", "val", "test"):
        np.save(d / f"alpha101_{s}.npy", rng.normal(size=(n, f)).astype(np.float32))
        np.save(d / f"labels_{s}.npy", rng.uniform(-1, 1, size=(n - 100, 8)).astype(np.float32))
    return d


def test_training_windows_avoid_excluded_rows():
    d = _seq_prepared()
    data = SeqData(d, th.device("cpu"))
    starts = data.allowed_starts(64, 0, ((1000, 1500),)).numpy()
    assert len(starts) > 0 and np.all((starts + 64 <= 1000) | (starts >= 1500))
    assert starts.min() == 0 and starts.max() == data.labelled("train") - 64
    th.manual_seed(0)
    x, _ = data.sample_train(256, 64, 0, 0.0, 0.0, th.from_numpy(starts))
    first = x[:, 0, 0].numpy()
    rows = {float(v): i for i, v in enumerate(data.x_all[:, 0].numpy())}
    i0 = np.array([rows[float(v)] for v in first])
    assert np.all((i0 + 64 <= 1000) | (i0 >= 1500))


def test_crossfit_blocks_and_stitching():
    assert CF.blocks(10, 4) == [(0, 2), (2, 5), (5, 8), (8, 10)]
    assert CF.exclusion((2, 5), 10, embargo=3) == (0, 8)
    d = _seq_prepared()
    cfg = RnnConfig(arch="gru", hp=dict(mid_dim=8, num_layers=1, dropout=0.0), seq_len=64, wup_dim=16, batch_size=8,
                    max_steps=3, valid_windows=8, device="cpu", amp=False, skip_head=0,
                    train_exclude=((1000, 1500),))
    out = Path(tempfile.mkdtemp())
    train_rnn(cfg, d, out / "r", verbose=False, save_splits=("train",))
    assert (out / "r" / "factors_train.npy").exists() and not (out / "r" / "factors_val.npy").exists()
    assert json.loads((out / "r" / "config.json").read_text())["train_exclude"] == [[1000, 1500]]
    # stitching: block k must come from fold k's models only
    n, base = 3000, out / "base"
    base.mkdir()
    for s in ("train", "val"):
        np.save(base / f"factors_{s}.npy", np.full((n, 8), -9.0, np.float32))
    fold_dirs = {}
    for f in range(4):
        fold_dirs[f] = []
        for s in range(2):
            r = out / f"f{f}s{s}"
            r.mkdir()
            np.save(r / "factors_train.npy", np.full((n, 8), 10 * f + s, np.float32))
            fold_dirs[f].append(r)
    fd = CF.write_factors(fold_dirs, n, base, out / "cf", ("train", "val"), k=4)
    tr = np.load(fd / "factors_train.npy")
    for f, (lo, hi) in enumerate(CF.blocks(n, 4)):
        assert np.all(tr[lo:hi] == 10 * f + 0.5)
    assert np.all(np.load(fd / "factors_val.npy") == -9.0)
