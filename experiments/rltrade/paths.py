"""One place for the Google-Drive layout every notebook reads and writes.

    <root>/data/BTC_1min_with_sentiment_risk_train.csv.gz   input (upload once; never committed)
    <root>/prepared/        01  factors, labels, market arrays
    <root>/rnn/default/     02  causal RNN with the notebook's hyperparameters
    <root>/rnn/tuning/      04  Optuna studies, final runs, selected.json
    <root>/runs/            03, 05-07  results.csv + models/<run_id>/
    <root>/report/          tables and figures for the thesis
"""
import json
from pathlib import Path
from types import SimpleNamespace

DATA_FILE = "BTC_1min_with_sentiment_risk_train.csv.gz"


def layout(root):
    root = Path(root)
    L = SimpleNamespace(root=root, data=root / "data" / DATA_FILE, prepared=root / "prepared",
                        rnn_default=root / "rnn" / "default", rnn_tuning=root / "rnn" / "tuning",
                        runs=root / "runs", report=root / "report")
    for p in (L.prepared, L.rnn_default, L.rnn_tuning, L.runs, L.report):
        p.mkdir(parents=True, exist_ok=True)
    return L


def factor_dir(L, prefer_selected=True):
    """RNN factors for the RL stage: the model selected in 04 if it exists, else the 02 default."""
    sel = L.rnn_tuning / "selected.json"
    if prefer_selected and sel.exists():
        s = json.loads(sel.read_text())
        d = L.rnn_tuning / "final" / f"{s['arch']}_s{s['seed']}"
        print(f"RNN factors: selected model {s['arch']} seed {s['seed']} ({d})")
        return d
    if not (L.rnn_default / "factors_train.npy").exists():
        raise FileNotFoundError("no RNN factors yet - run 02_rnn_train (and optionally 04_rnn_tuning) first")
    print(f"RNN factors: default model ({L.rnn_default})")
    return L.rnn_default
