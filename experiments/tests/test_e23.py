"""E23: extra factor-model inputs are causal and scaled with train statistics only."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rltrade import enrich                                  # noqa: E402


def _frame(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    d = pd.DataFrame(index=idx)
    d["midpoint"] = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    d["buys"], d["sells"] = rng.uniform(1, 2, n), rng.uniform(1, 2, n)
    for s in ("bids", "asks"):
        for i in range(5):
            d[f"{s}_notional_{i}"] = rng.uniform(1e6, 2e6, n)
    for c in ("oi_btc", "top_ls_accounts", "top_ls_positions", "global_ls_accounts", "taker_ls_vol_ratio"):
        d[c] = rng.uniform(0.5, 2, n)
    d["funding_rate"] = rng.normal(0, 1e-4, n)
    d["eth_mid"] = 50 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    d["eth_volume"] = rng.uniform(1, 2, n)
    d["eth_taker_buy"] = d.eth_volume * rng.uniform(0.3, 0.7, n)
    for c in enrich.SENTIMENT_COLS:
        d[c] = np.repeat(rng.uniform(1, 5, n // 1440 + 1), 1440)[:n]
    return d


def test_extra_features_are_causal():
    d = _frame()
    a = enrich.extra_features(d)
    cut = 3000
    d2 = d.copy()
    d2.iloc[cut:] = d2.iloc[cut:] * 1.7                        # change the future only
    b = enrich.extra_features(d2)
    pd.testing.assert_frame_equal(a.iloc[:cut], b.iloc[:cut])
    assert not a.iloc[cut:].equals(b.iloc[cut:])
    assert np.isfinite(a.iloc[2000:].to_numpy()).all()         # after the 1-day warm-up every column is defined


def test_build_scales_on_train_only_and_appends_columns():
    src, ex, dst = (Path(tempfile.mkdtemp()) for _ in range(3))
    rng = np.random.default_rng(1)
    sizes = {"train": 400, "val": 200, "test": 100}             # the factor model always reads train / val / test
    for s, n in sizes.items():
        np.save(src / f"alpha101_{s}.npy", rng.uniform(-1, 1, (n, 3)).astype(np.float16))
        np.save(src / f"labels_{s}.npy", np.zeros((n, 8), np.float32))
        np.savez(src / f"market_{s}.npz", midpoint=np.ones(n))
        x = rng.normal(size=(n, 2)).astype(np.float32)
        if s == "val":
            x[:, 0] += 100.0                                     # far outside train: must clip to +1
        x[:5, 1] = np.nan                                       # leading gap: train median, then ffill
        np.save(ex / f"extra_{s}.npy", x)
    (src / "meta.json").write_text(json.dumps({"inputs": ["a", "b", "c"], "set": "toy"}))
    (ex / "extra.json").write_text(json.dumps({"columns": ["x0", "x1"]}))
    m = enrich.build(src, ex, dst, verbose=False)
    assert m["inputs"][-2:] == ["x0", "x1"] and m["n_inputs"] == 5
    tr, va = np.load(dst / "alpha101_train.npy"), np.load(dst / "alpha101_val.npy")
    assert tr.shape == (400, 5) and va.shape == (200, 5)
    assert np.array_equal(tr[:, :3], np.load(src / "alpha101_train.npy"))
    assert np.all(va[:, 3] == 1.0) and np.isfinite(tr).all() and np.abs(tr[:, 3:]).max() <= 1.0
