"""
Experiment 10 - the FinAI Contest 2025 Task 1 starter kit with its bugs fixed, on the contest's 1-second data,
price only (no DeepSeek signals). Every step skips work that is already on disk, so a re-run after a Colab
disconnect resumes.

    python kit_pipeline.py prepare  --csv BTC_1sec.csv --data DATA     Alpha101, labels, 70/15/15 row splits
    python kit_pipeline.py rnn      --data DATA                        RNN factor model -> 8 factors per second
    python kit_pipeline.py ensemble --data DATA --run RUN              4 DQN members, selected on validation
    python kit_pipeline.py evaluate --data DATA --run RUN --split test every 79-minute episode of the split
    python kit_pipeline.py news --csv BTC_1sec.csv --news news_train.csv --data DATA
                                                                       E11: DeepSeek news scores per second

Options of `ensemble`: --break-step (320000, the kit's TODO value), --num-sims (auto: 4096 as shipped when
the replay buffer fits in GPU memory, else halved), --slippage (7e-7 as shipped), --seed (0), --signals
(E11: add the 2 news inputs to the state).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MEMBERS = ("AgentD3QN", "AgentDoubleDQN", "AgentDoubleDQN", "AgentTwinD3QN")  # task1_ensemble.py's list
MAX_STEP = (4800 - 60) // 2


def state_dim(signals=False):
    return 2 + 8 + (2 if signals else 0)


def _use(data):
    os.environ["KIT_DATA"] = data  # read by ConfigData() inside the kit's functions


def prepare(csv, data):
    _use(data)
    from data_config import ConfigData
    from seq_data import convert_btc_csv_to_btc_npy
    if os.path.exists(ConfigData(data).input_ary_path):
        print("| prepare: done already")
        return
    t = time.time()
    convert_btc_csv_to_btc_npy(csv, ConfigData(data))
    print(f"| prepare: {time.time() - t:.0f} s")


def rnn(data, gpu_id=0):
    _use(data)
    from data_config import ConfigData
    import seq_run
    cfg = ConfigData(data)
    if not os.path.exists(cfg.predict_net_path):
        seq_run.train_model(gpu_id=gpu_id)
    if not os.path.exists(cfg.predict_ary_path):
        seq_run.valid_model(gpu_id=gpu_id)
    print("| rnn: done")


def replay_gb(num_sims, signals=False):
    """GPU memory the kit's off-policy training needs: the replay buffer (max_step * 32 steps per env), the
    warm-up batch (3 horizons) and one exploration batch (1 horizon), float32 states."""
    per_step = num_sims * (state_dim(signals) + 3) * 4
    return (MAX_STEP * 32 * per_step + 4 * MAX_STEP * 4 * per_step) / 1e9


def auto_num_sims(gpu_id, signals=False):
    import torch
    if gpu_id < 0 or not torch.cuda.is_available():
        return 2 ** 12
    total = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
    for n in (2 ** 12, 2 ** 11, 2 ** 10, 2 ** 9):
        if replay_gb(n, signals) + 1.5 < 0.92 * total:
            return n
    return 2 ** 9


def news(csv, news_csv, data):
    _use(data)
    from data_config import ConfigData
    from news_signals import build
    if os.path.exists(ConfigData(data).signals_ary_path):
        print("| news: done already")
        return
    build(csv, news_csv, data)


def ensemble(data, run, gpu_id=0, break_step=int(32e4), num_sims="auto", slippage=7e-7, seed=0,
             stop_loss_rate=1e-3, signals=False):
    _use(data)
    import torch
    import erl_agent
    import task1_ensemble
    if signals:
        from data_config import ConfigData
        assert os.path.exists(ConfigData(data).signals_ary_path), "run `kit_pipeline.py news` first"
    num_sims = auto_num_sims(gpu_id, signals) if num_sims == "auto" else int(num_sims)
    cfg = {"break_step": int(break_step), "num_sims": num_sims, "slippage": slippage, "seed": seed,
           "stop_loss_rate": stop_loss_rate, "signals": bool(signals), "members": list(MEMBERS),
           "replay_gb": round(replay_gb(num_sims, signals), 1),
           "device": torch.cuda.get_device_name(gpu_id) if gpu_id >= 0 and torch.cuda.is_available() else "cpu"}
    os.makedirs(run, exist_ok=True)
    path = os.path.join(run, "config.json")
    if os.path.exists(path):  # a resumed run must keep its settings
        old = json.load(open(path))
        keep = ("break_step", "num_sims", "slippage", "seed", "stop_loss_rate", "signals", "members")
        assert all(old.get(k, False) == cfg[k] for k in keep), f"{path} has different settings: {old}"
        cfg = old
    json.dump(cfg, open(path, "w"), indent=1)
    print("| ensemble:", cfg)
    t = time.time()
    task1_ensemble.run(run, [getattr(erl_agent, m) for m in MEMBERS], gpu_id=gpu_id, break_step=break_step,
                       num_sims=num_sims, slippage=slippage, seed=seed, data_dir=data, stop_loss_rate=stop_loss_rate,
                       signals=signals)
    print(f"| ensemble: {time.time() - t:.0f} s")


def evaluate(data, run, split="test", gpu_id=0):
    _use(data)
    import task1_eval
    cfg = json.load(open(os.path.join(run, "config.json")))
    return task1_eval.run_evaluation(os.path.join(run, "ensemble_models"), gpu_id=gpu_id, data_dir=data,
                                     split=split, out_dir=os.path.join(run, f"eval_{split}"),
                                     slippage=cfg["slippage"], stop_loss_rate=cfg["stop_loss_rate"],
                                     signals=cfg.get("signals", False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("prepare", "rnn", "news", "ensemble", "evaluate"))
    ap.add_argument("--csv")
    ap.add_argument("--news")
    ap.add_argument("--data", required=True)
    ap.add_argument("--run")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--break-step", type=int, default=int(32e4))
    ap.add_argument("--num-sims", default="auto")
    ap.add_argument("--slippage", type=float, default=7e-7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--signals", action="store_true")
    a = ap.parse_args(argv)
    if a.step == "prepare":
        prepare(a.csv, a.data)
    elif a.step == "rnn":
        rnn(a.data, a.gpu)
    elif a.step == "news":
        news(a.csv, a.news, a.data)
    elif a.step == "ensemble":
        ensemble(a.data, a.run, a.gpu, a.break_step, a.num_sims, a.slippage, a.seed, signals=a.signals)
    else:
        evaluate(a.data, a.run, a.split, a.gpu)


if __name__ == "__main__":
    main()
