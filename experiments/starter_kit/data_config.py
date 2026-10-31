import json
import os


class ConfigData:
    """[fix] One prepared folder for the whole contest file. The splits are row ranges (splits.json), so
    factors and prices are computed once, causally, and every split reads the same arrays."""

    def __init__(self, data_dir: str = None):
        data_dir = data_dir or os.environ.get("KIT_DATA", "./data")
        self.data_dir = data_dir
        self.alpha_raw_path = f"{data_dir}/alpha101_raw.npy"  # Alpha101 before normalisation, all rows
        self.input_ary_path = f"{data_dir}/BTC_1sec_input.npy"  # normalised with train quantiles, all rows
        self.label_ary_path = f"{data_dir}/BTC_1sec_label.npy"  # train rows only
        self.price_ary_path = f"{data_dir}/BTC_1sec_price.npy"  # bid, ask, mid, all rows
        self.predict_ary_path = f"{data_dir}/BTC_1sec_predict.npy"  # RNN factors, all rows
        self.predict_net_path = f"{data_dir}/BTC_1sec_predict.pth"
        self.splits_path = f"{data_dir}/splits.json"  # {"train": [r0, r1], "val": ..., "test": ...}

    def split_rows(self, split: str):
        with open(self.splits_path) as f:
            return tuple(json.load(f)[split])
