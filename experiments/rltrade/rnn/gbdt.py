"""
Gradient-boosted trees (LightGBM) as an alternative factor model to the GRU (experiment 14).

Same job, same interface as `train.train_rnn`: read a prepared folder (`alpha101_<split>.npy` inputs,
`labels_<split>.npy` 8-horizon trend targets), and write `factors_<split>.npy` (one prediction per bar per
horizon, in [-1, 1]) plus `metrics.json` / `config.json`, so the trading agent can use either model unchanged.

Trees see one row at a time, so the recent past is given to them as features. For every input x:
    x_t, its exponential moving average over ~15 bars and over ~60 bars (pandas ewm, adjust=False).
All three are causal (row t uses rows <= t only). They are computed once on the chronological series of all
splits joined end to end, as the GRU's windows are.

One LGBMRegressor per horizon (L2 loss, the GRU's MSE). Training rows: the labelled train rows after
`skip_head`, every `train_stride`-th row (neighbouring labels overlap heavily, so this loses little and is
several times faster). Early stopping on the labelled validation rows (same stride), as the GRU stops on
validation loss. Test and later splits are only predicted. Predictions are clipped to [-1, 1].
"""
import argparse
import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .train import available_splits, label_metrics

EMA_SPANS = (15, 60)


@dataclass
class GbdtConfig:
    ema_spans: tuple = EMA_SPANS
    train_stride: int = 3
    skip_head: int = 1024
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 500         # large leaves: the labels are noisy and strongly autocorrelated
    feature_fraction: float = 0.5
    bagging_fraction: float = 0.7
    bagging_freq: int = 1
    lambda_l2: float = 10.0
    n_estimators: int = 3000
    early_stopping: int = 100
    num_threads: int = 0                 # 0 = all cores
    device: str = "cpu"                  # "gpu" needs a GPU build of LightGBM
    seed: int = 0
    params: dict = field(default_factory=dict)   # any other LightGBM parameter


def load_inputs(prepared_dir):
    d = Path(prepared_dir)
    splits = available_splits(d)
    xs = [np.nan_to_num(np.load(d / f"alpha101_{s}.npy").astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
          for s in splits]
    length = {s: len(x) for s, x in zip(splits, xs)}
    offset = dict(zip(splits, np.cumsum([0] + [len(x) for x in xs[:-1]]).tolist()))
    y = {s: np.load(d / f"labels_{s}.npy").astype(np.float32) for s in splits}
    names = None
    meta = d / "meta.json"
    if meta.exists():
        names = json.loads(meta.read_text()).get("inputs")
    x = np.concatenate(xs)
    if not names or len(names) != x.shape[1]:
        names = [f"x{j:03d}" for j in range(x.shape[1])]
    return x, y, splits, offset, length, names


def make_features(x, spans=EMA_SPANS, block=32):
    """[x, ema_span1(x), ema_span2(x), ...] as float32, built in column blocks to keep memory down."""
    n, f = x.shape
    out = np.empty((n, f * (1 + len(spans))), dtype=np.float32)
    out[:, :f] = x
    for k, span in enumerate(spans, start=1):
        for j in range(0, f, block):
            blk = pd.DataFrame(x[:, j:j + block]).ewm(span=span, adjust=False).mean().to_numpy(np.float32)
            out[:, k * f + j: k * f + j + blk.shape[1]] = blk
    return out


def feature_names(names, spans=EMA_SPANS):
    return list(names) + [f"{n}__ema{s}" for s in spans for n in names]


def train_gbdt(cfg, prepared_dir, out_dir, verbose=True, save_factors=True):
    import lightgbm as lgb

    cfg = cfg if isinstance(cfg, GbdtConfig) else GbdtConfig(**cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    x, y, splits, offset, length, names = load_inputs(prepared_dir)
    feats = make_features(x, cfg.ema_spans)
    del x
    fnames = feature_names(names, cfg.ema_spans)
    tr_rows = np.arange(min(cfg.skip_head, len(y["train"]) // 4), len(y["train"]), cfg.train_stride)
    va_rows = np.arange(0, len(y["val"]), cfg.train_stride)
    X_tr, X_va = feats[offset["train"] + tr_rows], feats[offset["val"] + va_rows]
    if verbose:
        print(f"[gbdt] {feats.shape[1]} features ({len(names)} inputs x {1 + len(cfg.ema_spans)}) | "
              f"train rows {len(tr_rows):,} | val rows {len(va_rows):,} | {time.time() - t0:.0f}s to build")
    params = dict(objective="regression", learning_rate=cfg.learning_rate, num_leaves=cfg.num_leaves,
                  min_child_samples=cfg.min_child_samples, colsample_bytree=cfg.feature_fraction,
                  subsample=cfg.bagging_fraction, subsample_freq=cfg.bagging_freq, reg_lambda=cfg.lambda_l2,
                  n_estimators=cfg.n_estimators, random_state=cfg.seed, n_jobs=cfg.num_threads or -1,
                  device_type=cfg.device, verbose=-1, **cfg.params)
    n_out = y["train"].shape[1]
    pred = np.empty((len(feats), n_out), dtype=np.float32)
    gain = np.zeros(len(fnames))
    horizons = []
    for k in range(n_out):
        t1 = time.time()
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y["train"][tr_rows, k], eval_set=[(X_va, y["val"][va_rows, k])], eval_metric="l2",
              callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False)])
        pred[:, k] = np.clip(m.predict(feats, num_iteration=m.best_iteration_), -1.0, 1.0)
        g = m.booster_.feature_importance(importance_type="gain")
        gain += g / max(g.sum(), 1e-12)
        m.booster_.save_model(str(out / f"model_h{k}.txt"), num_iteration=m.best_iteration_)
        horizons.append({"horizon": k, "best_iteration": int(m.best_iteration_),
                         "val_l2": float(m.best_score_["valid_0"]["l2"]), "seconds": round(time.time() - t1, 1)})
        if verbose:
            print(f"  horizon {k}: {m.best_iteration_} trees | val l2 {horizons[-1]['val_l2']:.5f} | "
                  f"{horizons[-1]['seconds']:.0f}s")
    metrics = {"arch": "gbdt", "features": len(fnames), "horizons": horizons,
               "minutes": round((time.time() - t0) / 60, 2), "seed": cfg.seed}
    for s in splits:
        f = pred[offset[s]: offset[s] + length[s]]
        if save_factors:
            np.save(out / f"factors_{s}.npy", f)
        metrics[s] = label_metrics(f, y[s])
    # importance: mean share of gain over the horizons, per feature and summed per input
    imp = pd.DataFrame({"feature": fnames, "gain_share": gain / n_out})
    imp["input"] = imp.feature.str.split("__").str[0]
    imp.sort_values("gain_share", ascending=False).to_csv(out / "importance_features.csv", index=False)
    imp.groupby("input").gain_share.sum().sort_values(ascending=False).to_csv(out / "importance_inputs.csv")
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if verbose:
        print(f"[gbdt] done ({metrics['minutes']} min) | "
              + " | ".join(f"{s}: mse {metrics[s]['mse']:.4f} ic {metrics[s]['ic_mean']:+.3f} "
                           f"acc {metrics[s]['dir_acc_mean']:.3f}" for s in splits))
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Train the LightGBM factor model on prepared data.")
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--train-stride", type=int, default=3)
    a = ap.parse_args()
    train_gbdt(GbdtConfig(seed=a.seed, n_estimators=a.n_estimators, train_stride=a.train_stride), a.prepared, a.out)


if __name__ == "__main__":
    main()
