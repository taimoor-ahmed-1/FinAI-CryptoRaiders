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


# ---------------------------------------------------------------- E20b: gated environment and folds
from rltrade import factor_rule as FR                      # noqa: E402
from rltrade import walkforward as WF                      # noqa: E402
from rltrade.baselines import buy_and_hold_policy          # noqa: E402
from rltrade.env import EnvConfig, Market, run_policy      # noqa: E402
from tests.test_e16 import _prepared                       # noqa: E402


def test_rule_trades_the_same_inside_its_own_gate():
    d = _prepared(splits=("train", "val"), n=4000)
    cfg = EnvConfig(state="A")
    sig = FR.signals(np.load(d / "factors_train.npy"))
    grid = FR.make_grid(sig, quantiles=(0.7, 0.95))
    m = Market(d, d, "val", cfg, "cpu")
    for g in [grid[i] for i in (0, 3, 9, 14, 30, len(grid) - 1)]:
        pol = FR.rule_policy([g], cfg, "cpu")
        free = run_policy(pol, m, cfg)
        gcfg = EnvConfig(state="A", **WF.gate_config(g))
        gated = run_policy(pol, m, gcfg)
        assert np.array_equal(free["level"], gated["level"]) and np.allclose(free["equity"], gated["equity"]), g


def test_gate_blocks_entries_and_min_hold_locks():
    d = _prepared(splits=("train", "val"), n=4000)
    k = 0.8
    cfg = EnvConfig(state="A", entry_signal=2, entry_threshold=k)
    m = Market(d, d, "val", cfg, "cpu")
    bt = run_policy(buy_and_hold_policy, m, cfg)
    f2 = np.load(d / "factors_val.npy")[:, 2]
    first = int(np.argmax(bt["level"] > 0))
    assert bt["level"][first] == 1 and f2[bt["bar"][first]] > k
    assert np.all(f2[bt["bar"][:first]] <= k)
    toggle = lambda o: th.where(o[:, 0] > 0, th.zeros(len(o), dtype=th.long), th.full((len(o),), 2, dtype=th.long))  # noqa: E731
    lv = run_policy(toggle, m, EnvConfig(state="A", min_hold_bars=50))["level"]
    runs, cur = [], 0
    for x in lv[:-20]:
        if x > 0:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    assert runs and min(runs) == 10                       # 50 bars / 5 bars per decision
    assert np.all(run_policy(toggle, m, EnvConfig(state="A"))["level"][:20] == np.tile([1, 0], 10))


def test_write_fold_slices_and_stitch():
    d = _prepared(splits=("train", "val"), n=4000)
    w = WF.windows(4000, 4000)
    assert w["T3"] == ("train", 2000, 3000) and w["V2"] == ("val", 2000, 4000)
    out = WF.write_fold(d, d, w, 2, Path(tempfile.mkdtemp()))            # fold 3: train T1-T3, ckpt T4, judged V1
    src_v, src_t = np.load(d / "market_val.npz"), np.load(d / "market_train.npz")
    test = np.load(out / "market_test.npz")
    assert np.array_equal(test["midpoint"], src_v["midpoint"][:2000])
    assert np.array_equal(np.load(out / "factors_test.npy"), np.load(d / "factors_val.npy")[:2000])
    hist = np.load(out / "market_hist.npz")
    assert np.array_equal(hist["midpoint"], src_t["midpoint"])           # T1-T4 = all of train
    assert len(np.load(out / "factors_train.npy")) == 3000
    cfg = EnvConfig(state="A")
    bts = [run_policy(buy_and_hold_policy, Market(out, out, s, cfg, "cpu"), cfg) for s in ("val", "test")]
    st = WF.stitch(bts)
    assert len(st["returns"]) == sum(len(b["returns"]) for b in bts)
    assert np.isclose(st["equity"][-1], bts[0]["equity"][-1] * bts[1]["equity"][-1])
    res = {"num_trades": np.array([0, 9, 9]), "sharpe": np.array([5.0, 1.0, 2.0])}
    assert WF.select_rule(res) == 2
