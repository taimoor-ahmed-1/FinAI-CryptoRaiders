"""E21: factor variants for the learnability check, training with train + validation curves, Q preference."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rltrade import learnability as LB                      # noqa: E402
from rltrade.env import EnvConfig, Market                   # noqa: E402
from rltrade.factor_rule import forward_returns             # noqa: E402
from tests.test_e16 import TINY_DQN, _prepared              # noqa: E402


def test_shuffle_days_moves_whole_days():
    f = np.arange(10 * 7 + 3, dtype=np.float32)[:, None].repeat(8, 1)
    g = LB.shuffle_days(f, seed=0, block=10)
    assert sorted(g[:, 0]) == sorted(f[:, 0]) and not np.array_equal(f, g)
    assert np.array_equal(g[-3:], f[-3:])                         # the partial block stays last
    for i in range(7):                                            # every piece is one intact day
        d = g[i * 10:(i + 1) * 10, 0]
        assert d[0] % 10 == 0 and np.array_equal(d, d[0] + np.arange(10))


def test_planted_signal_hits_the_chosen_correlation():
    rng = np.random.default_rng(1)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, 200_000)))
    fwd = np.log1p(forward_returns(price, LB.PLANT_H))
    ok = np.isfinite(fwd)
    for rho in (0.0, 0.1, 0.4):
        s = LB.planted_signal(price, rho, seed=3)
        assert abs(np.corrcoef(s[ok], fwd[ok])[0, 1] - rho) < 0.03
        assert abs(s.std() - 1) < 1e-6
    a, b = LB.planted_signal(price, 0.2, seed=3), LB.planted_signal(price, 0.2, seed=3)
    assert np.array_equal(a, b)


def test_write_variant_keeps_other_factors_and_scale():
    d = _prepared(n=6000)
    out, info = LB.write_variant(d, d, Path(tempfile.mkdtemp()) / "p", "planted", rho=0.4)
    src, new = np.load(d / "factors_train.npy"), np.load(out / "factors_train.npy")
    keep = [j for j in range(8) if j != LB.PLANT_FACTOR]
    assert np.array_equal(src[:, keep], new[:, keep]) and not np.array_equal(src[:, 5], new[:, 5])
    assert abs(new[:, 5].std() - src[:, 5].std()) < 1e-3 * src[:, 5].std() + 1e-6
    assert info["ic"]["train"] > 0.2 and json.loads((out / "variant.json").read_text())["rho"] == 0.4
    out2, _ = LB.write_variant(d, d, Path(tempfile.mkdtemp()) / "s", "shuffled")
    assert np.allclose(np.sort(np.load(out2 / "factors_val.npy"), 0), np.sort(np.load(d / "factors_val.npy"), 0))


def test_train_run_scores_both_splits_and_resumes():
    d, out = _prepared(n=4000), Path(tempfile.mkdtemp())
    kw = dict(seed=0, agent_hp=TINY_DQN, env_over={"episode_bars": 300, "start_skip": 10}, total_steps=2_000,
              eval_every=1_000, device="cpu", train_chunks=2, val_chunks=2, verbose=False)
    res = LB.train_run("real_s0", d, d, out, **kw)
    assert [r["steps"] for r in res["curve"]][0] == 0 and res["curve"][-1]["steps"] >= 2_000
    assert {"train_sharpe", "val_sharpe", "train_gross_pct"} <= set(res["curve"][0])
    assert {"attr_timing_pct", "daily_ic", "sharpe"} <= set(res["final"]["val"])
    again = LB.train_run("real_s0", d, d, out, **kw)              # resumes from result.json
    assert again["curve"] == json.loads(json.dumps(res["curve"], default=float))
    agent, cfg = LB.load_agent(out / "runs" / "real_s0", "cpu")
    m = Market(d, d, "val", cfg, "cpu")
    pref, act = LB.q_preference(agent, m, batch=1000)
    assert pref.shape == (m.n,) and set(np.unique(act)) <= {0, 1, 2}
    row = LB.preference_table(pref, act, np.load(d / "factors_val.npy"), m.price.numpy(), horizons=(60,), every=10)
    assert "corr_f5" in row and "ic_60" in row
    assert abs(row["from_flat_long_pct"] + row["from_flat_short_pct"] + row["from_flat_stay_pct"] - 100) < 1e-9


def test_planted_only_keeps_one_input_and_matches_planted():
    d = _prepared(n=6000)
    out = Path(tempfile.mkdtemp())
    _, a = LB.write_variant(d, d, out / "p", "planted", rho=0.2, seed=0)
    _, b = LB.write_variant(d, d, out / "po", "planted_only", rho=0.2, seed=0)
    for s in ("train", "val"):
        fp, fo = np.load(out / "p" / f"factors_{s}.npy"), np.load(out / "po" / f"factors_{s}.npy")
        assert np.array_equal(fp[:, LB.PLANT_FACTOR], fo[:, LB.PLANT_FACTOR])       # the same planted signal
        others = [j for j in range(8) if j != LB.PLANT_FACTOR]
        assert np.all(fo[:, others] == 0) and a["ic"][s] == b["ic"][s]
