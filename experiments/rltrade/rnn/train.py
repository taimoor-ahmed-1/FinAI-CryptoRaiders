"""
Stage 2 - train a sequence model that reads the 101 Alpha101 factors and predicts
the 8 multi-horizon trend labels, then write its predictions for every bar of
every split. Those 8 numbers per bar ("RNN factors") are part of the trading
agent's state.

What is the same as 4_rnn_trainer.ipynb (these are the defaults)
    optimiser (AdamW, lr 3e-4, wd 1e-3, betas 0.9/0.999), ReduceLROnPlateau
    (x0.5, patience 4, min 1e-6), gradient clip 1.0, MSE loss after a 128-step
    warm-up, window 512, batch 64, input noise 0.01, label noise 0.01, validation
    every 32 steps, early stopping after 16 validations without improvement, and
    the step budget  train_rows / 512 / 64 * 1024.

What is different (see CHANGES.md)
    - validation / test targets are that split's own labels (the notebook scored
      validation inputs against the first rows of the TRAINING labels);
    - the models are causal (see rnn/__init__.py);
    - the last `max_offset` rows of a split have no label; they are left out of
      the loss instead of being trained towards zero-padded targets;
    - validation during training uses 256 fixed windows spread across the whole
      validation period (fast and representative); final metrics use every row;
    - factors for the trading agent are predicted with windows that start from a
      fresh state and have at least `wup_dim` bars of past context, the same
      condition the model was trained under. The notebook carried the leaky
      model's hidden state from chunk to chunk.

Test data is never used to choose anything here. Test metrics are computed at the
end, with the checkpoint chosen on validation, and are only reported.
"""
import argparse
import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch as th

from .. import seeding
from . import build_rnn

SPLITS = ("train", "val", "test")


@dataclass
class RnnConfig:
    arch: str = "enhanced"
    hp: dict = field(default_factory=dict)     # overrides rnn.DEFAULTS[arch] (mid_dim, num_layers, ...)
    seq_len: int = 512
    wup_dim: int = 128
    batch_size: int = 64
    optimizer: str = "adamw"       # adamw (notebook) | adam | rmsprop
    lr: float = 3e-4
    weight_decay: float = 1e-3
    clip_grad_norm: float = 1.0
    epoch: int = 1024              # notebook budget: steps = train_rows / seq_len / batch_size * epoch
    max_steps: int = 0             # > 0 caps the step budget (tuning, smoke tests)
    max_minutes: float = 0.0       # > 0 stops training after this wall-clock time
    valid_gap: int = 32
    patience: int = 16
    valid_windows: int = 256
    input_noise: float = 0.01
    label_noise: float = 0.01      # notebook: smoothing 0.1 * 0.1
    skip_head: int = 1024          # notebook never sampled the first 1024 train rows (factor warm-up)
    amp: bool = True               # mixed precision on GPU
    seed: int = 0
    device: str = "auto"


class SeqData:
    """Alpha101 inputs as one chronological series + per-split labels."""

    def __init__(self, prepared_dir, device):
        d = Path(prepared_dir)
        xs = [np.nan_to_num(np.load(d / f"alpha101_{s}.npy").astype(np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0) for s in SPLITS]
        self.length = {s: len(x) for s, x in zip(SPLITS, xs)}
        self.offset = dict(zip(SPLITS, np.cumsum([0] + [len(x) for x in xs[:-1]]).tolist()))
        self.x_all = th.from_numpy(np.concatenate(xs)).to(device)
        self.y = {s: th.from_numpy(np.load(d / f"labels_{s}.npy").astype(np.float32)).to(device)
                  for s in SPLITS}
        self.inp_dim = self.x_all.shape[1]
        self.out_dim = self.y["train"].shape[1]
        self.device = device

    def labelled(self, split):
        return self.y[split].shape[0]

    def sample_train(self, batch_size, seq_len, skip_head, input_noise, label_noise):
        hi = self.labelled("train") - seq_len
        lo = min(skip_head, max(hi - 1, 0))
        i0 = th.randint(lo, hi + 1, (batch_size,), device=self.device)
        ids = i0[:, None] + th.arange(seq_len, device=self.device)[None, :]
        x = self.x_all[ids]                               # train rows start at offset 0
        y = self.y["train"][ids]
        if input_noise:
            x = x + th.randn_like(x) * input_noise
        if label_noise:
            y = (y + th.randn_like(y) * label_noise).clamp(-1.0, 1.0)
        return x, y

    def valid_starts(self, split, seq_len, wup_dim, count):
        """Evenly spaced windows whose scored part lies in the split's labelled rows.

        A window may begin up to `wup_dim` bars before the split: that is past data,
        used only as warm-up context, never scored.
        """
        o = self.offset[split]
        lo = max(o - wup_dim, 0)
        hi = o + self.labelled(split) - seq_len
        if hi < lo:
            raise ValueError(f"{split} split too short for seq_len={seq_len}")
        return np.unique(np.linspace(lo, hi, num=count).astype(np.int64))


@th.no_grad()
def _run(net, x, amp):
    with th.autocast(device_type=x.device.type, dtype=th.float16, enabled=amp):
        out, _ = net(x)
    return out.float()


@th.no_grad()
def predict_series(net, x_all, seq_len, wup_dim, amp=False, batch=64):
    """Causal prediction for every row of `x_all` (T, F) -> (T, out_dim) float32.

    Row t is predicted by a window that ends at or after t and starts from a fresh
    state; the model is causal, so only rows <= t influence it. Apart from the first
    `wup_dim` rows of the series, every row has at least `wup_dim` bars of context.
    """
    net.eval()
    n = x_all.shape[0]
    seq_len = min(seq_len, n)
    stride = seq_len - wup_dim if seq_len > wup_dim else seq_len
    starts = list(range(0, n - seq_len + 1, stride))
    if starts[-1] != n - seq_len:
        starts.append(n - seq_len)
    out = None
    ar = th.arange(seq_len, device=x_all.device)
    for b in range(0, len(starts), batch):
        ss = th.tensor(starts[b:b + batch], device=x_all.device)
        y = _run(net, x_all[ss[:, None] + ar[None, :]], amp).cpu().numpy()
        if out is None:
            out = np.empty((n, y.shape[-1]), dtype=np.float32)
        for k, s in enumerate(ss.tolist()):
            keep = 0 if s == 0 else wup_dim
            out[s + keep: s + seq_len] = y[k, keep:]
    return out


def label_metrics(pred, lab):
    """MSE, per-horizon information coefficient (Pearson) and direction accuracy."""
    n = min(len(pred), len(lab))
    p, y = pred[:n].astype(np.float64), lab[:n].astype(np.float64)
    ic, acc = [], []
    for k in range(y.shape[1]):
        pk, yk = p[:, k], y[:, k]
        ic.append(float(np.corrcoef(pk, yk)[0, 1]) if pk.std() > 0 and yk.std() > 0 else 0.0)
        moved = yk != 0
        acc.append(float((np.sign(pk[moved]) == np.sign(yk[moved])).mean()) if moved.any() else float("nan"))
    return {"mse": float(((p - y) ** 2).mean()), "ic_mean": float(np.mean(ic)),
            "dir_acc_mean": float(np.nanmean(acc)), "ic": ic, "dir_acc": acc, "rows": int(n)}


def _optimizer(cfg, params):
    if cfg.optimizer == "adamw":
        return th.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    if cfg.optimizer == "adam":
        return th.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "rmsprop":
        return th.optim.RMSprop(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def train_rnn(cfg, prepared_dir, out_dir, trial=None, verbose=True, save_factors=True):
    """Train one model; write checkpoint, factors and metrics to `out_dir`.

    `trial` is an optional Optuna trial: validation loss is reported to it and the
    run is pruned when Optuna says so. `save_factors=False` skips writing the
    factor arrays (tuning trials - they are ~40 MB each). Returns the metrics dict.
    """
    cfg = cfg if isinstance(cfg, RnnConfig) else RnnConfig(**cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeding.set_seed(cfg.seed)
    device = seeding.pick_device(cfg.device)
    amp = bool(cfg.amp and device.type == "cuda")

    data = SeqData(prepared_dir, device)
    net = build_rnn(cfg.arch, data.inp_dim, data.out_dim, **cfg.hp).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    opt = _optimizer(cfg, net.parameters())
    sched = th.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4, min_lr=1e-6)
    scaler = th.amp.GradScaler("cuda", enabled=amp)

    steps = int(data.length["train"] / cfg.seq_len / cfg.batch_size * cfg.epoch)
    if cfg.max_steps > 0:
        steps = min(steps, cfg.max_steps)
    steps = max(steps, 1)
    vstarts = th.tensor(data.valid_starts("val", cfg.seq_len, cfg.wup_dim, cfg.valid_windows), device=device)
    ar = th.arange(cfg.seq_len, device=device)
    if verbose:
        print(f"[rnn] {cfg.arch} | {n_params:,} params | {device} | amp={amp} | "
              f"up to {steps:,} steps | {len(vstarts)} validation windows")

    def valid_loss():
        net.eval()
        tot, cnt = 0.0, 0
        for b in range(0, len(vstarts), 64):
            ss = vstarts[b:b + 64]
            ids = ss[:, None] + ar[None, :]
            pred = _run(net, data.x_all[ids], amp)[:, cfg.wup_dim:]
            lab = data.y["val"][ids[:, cfg.wup_dim:] - data.offset["val"]]
            tot += float(((pred - lab) ** 2).sum())
            cnt += lab.numel()
        net.train()
        return tot / cnt

    history, best, bad, t0 = [], float("inf"), 0, time.time()
    ckpt = out / "best_model.pt"
    stop_reason = "step budget"
    net.train()
    for step in range(steps):
        x, y = data.sample_train(cfg.batch_size, cfg.seq_len, cfg.skip_head, cfg.input_noise, cfg.label_noise)
        with th.autocast(device_type=device.type, dtype=th.float16, enabled=amp):
            pred, _ = net(x)
        loss = ((pred.float() - y) ** 2)[:, cfg.wup_dim:].mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        th.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip_grad_norm)
        scaler.step(opt)
        scaler.update()

        time_up = bool(cfg.max_minutes) and (time.time() - t0) / 60 > cfg.max_minutes
        if step % cfg.valid_gap == 0 or step == steps - 1 or time_up:   # time up: score latest weights first
            vl = valid_loss()
            sched.step(vl)
            lr = opt.param_groups[0]["lr"]
            history.append({"step": step, "train_loss": loss.item(), "val_loss": vl, "lr": lr,
                            "minutes": round((time.time() - t0) / 60, 2)})
            if vl < best:
                best, bad = vl, 0
                th.save(net.state_dict(), ckpt)
            else:
                bad += 1
            if verbose and (len(history) % 10 == 1 or bad == 0):
                print(f"  step {step:>6} | train {loss.item():.5f} | val {vl:.5f} | best {best:.5f} "
                      f"| lr {lr:.1e} | patience {bad}/{cfg.patience}")
            if trial is not None:
                trial.report(vl, step)
                if trial.should_prune():
                    import optuna
                    (out / "history.json").write_text(json.dumps(history))
                    raise optuna.TrialPruned()
            if bad >= cfg.patience:
                stop_reason = "early stopping"
                break
        if time_up:
            stop_reason = "time budget"
            break

    net.load_state_dict(th.load(ckpt, map_location=device))
    factors = predict_series(net, data.x_all, cfg.seq_len, cfg.wup_dim, amp=amp)
    metrics = {"arch": cfg.arch, "params": n_params, "steps_run": step + 1, "stop_reason": stop_reason,
               "best_val_loss_windows": best, "minutes": round((time.time() - t0) / 60, 2)}
    for s in SPLITS:
        f = factors[data.offset[s]: data.offset[s] + data.length[s]]
        if save_factors:
            np.save(out / f"factors_{s}.npy", f)
        metrics[s] = label_metrics(f, data.y[s].cpu().numpy())
    metrics["run"] = seeding.run_info(cfg.seed)

    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "history.json").write_text(json.dumps(history))
    if verbose:
        print(f"[rnn] done ({stop_reason}, {metrics['minutes']} min) | "
              + " | ".join(f"{s}: mse {metrics[s]['mse']:.4f} ic {metrics[s]['ic_mean']:+.3f} "
                           f"acc {metrics[s]['dir_acc_mean']:.3f}" for s in SPLITS))
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Train an RNN factor model on prepared data.")
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="enhanced")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--hp", default="{}", help='JSON architecture overrides, e.g. \'{"mid_dim": 256}\'')
    a = ap.parse_args()
    train_rnn(RnnConfig(arch=a.arch, seed=a.seed, max_steps=a.max_steps, max_minutes=a.max_minutes,
                        hp=json.loads(a.hp)), a.prepared, a.out)


if __name__ == "__main__":
    main()
