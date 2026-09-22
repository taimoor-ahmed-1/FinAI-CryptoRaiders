"""
RNN training targets: multi-horizon trend labels.

`seq_to_label` reproduces the notebook's label definition exactly. What changes
is how it is APPLIED - which is the leakage fix.

The bug (4_rnn_trainer.ipynb)
-----------------------------
Labels were generated only from the training split. The data loader was never
given validation or test labels, so it fell into this branch:

    self.valid_label_seq = self.train_label_seq[:self.valid_seq_len]
    self.test_label_seq  = self.train_label_seq[:self.test_seq_len]

Validation and test *targets* were the first rows of the TRAINING labels, while
the *inputs* came from the validation / test periods. Inputs and targets did not
correspond in time, so validation loss, test loss and the "best model" chosen by
validation loss were all meaningless.

The fix
-------
Labels are built separately for every split from that split's own prices. The
quantile thresholds that discretise price moves into classes are fitted on the
TRAIN split and reused for validation and test, so every split's targets mean
the same thing and no validation/test statistic leaks into the definition.

Alignment
---------
`label[t]` compares smoothed price at t with smoothed price at t + offset, where
offset is a cumulative sum of the window sizes. It therefore needs `max_offset`
future rows, and the last `max_offset` rows of each split have no label. Those
rows are still used as model *inputs*; they are just excluded from the loss.
"""
import numpy as np

# The notebook's 1-minute configuration -> 8 horizons (8 RNN outputs = 8 state factors)
WIN_SIZES = (10, 20, 30, 60, 80, 100, 200, 400)
LT_Q = (0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40)
GT_Q = (0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.96, 0.98, 0.99)


def _normal_moving_average(ary, win_size=5):
    avg = ary.copy()
    avg[win_size - 1:] = np.convolve(ary, np.ones(win_size) / win_size, mode="valid")
    return avg


def _px_diffs(ary, win_sizes):
    ary = np.asarray(ary, dtype=np.float64)
    offsets = np.cumsum(win_sizes)
    px_avg_0 = _normal_moving_average(ary, win_size=5)
    for win_size, offset in zip(win_sizes, offsets):
        px_avg_i = _normal_moving_average(ary, win_size=win_size)
        yield px_avg_i[offset:] - px_avg_0[:-offset]


def seq_to_label(ary, win_sizes=WIN_SIZES, thresholds=None):
    """Notebook label definition, with optionally supplied (train-fitted) thresholds.

    Returns (labels, thresholds). `labels` has shape (len(ary) - max_offset, len(win_sizes)),
    values in [-1, 1]. With `thresholds=None` they are fitted on `ary` itself, which is
    exactly the notebook's behaviour - use that for the TRAIN split only.
    """
    labels, fitted = [], []
    for k, px_diff in enumerate(_px_diffs(ary, win_sizes)):
        if thresholds is None:
            lt = np.quantile(px_diff, q=LT_Q, axis=0)
            gt = np.quantile(px_diff, q=GT_Q, axis=0)
        else:
            lt, gt = thresholds[k]
        fitted.append((lt, gt))
        lt_ary = np.less(px_diff[:, None], lt[None, :]).astype(np.float32).mean(axis=1)
        gt_ary = np.greater(px_diff[:, None], gt[None, :]).astype(np.float32).mean(axis=1)
        labels.append(gt_ary - lt_ary)
    n = min(l.shape[0] for l in labels)
    return np.concatenate([l[:n, None] for l in labels], axis=1).astype(np.float32), fitted


def labels_for_splits(mid_by_split, win_sizes=WIN_SIZES):
    """Build labels for train/val/test; thresholds fitted on train only.

    mid_by_split: {"train": array, "val": array, "test": array} of midpoint prices.
    Returns {split: labels}.
    """
    out = {}
    out["train"], thr = seq_to_label(mid_by_split["train"], win_sizes)
    for split in ("val", "test"):
        out[split], _ = seq_to_label(mid_by_split[split], win_sizes, thresholds=thr)
    return out


def max_offset(win_sizes=WIN_SIZES):
    return int(np.sum(win_sizes))
