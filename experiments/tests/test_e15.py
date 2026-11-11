"""E15: the slow context inputs are causal, and the GRU shortcut starts as the plain GRU."""
import numpy as np
import torch as th

from rltrade.long_context import slow_features
from rltrade.rnn import build_rnn


def test_slow_features_are_causal():
    rng = np.random.default_rng(0)
    mid = 50_000 * np.exp(np.cumsum(rng.normal(0, 1e-3, 4000)))
    names = ["flow_imbalance", "ret_1m", "other"]
    x = rng.normal(size=(4000, 3)).astype(np.float32)
    f, cols = slow_features(mid, x, names, spans=(240, 1440))
    assert f.shape == (4000, len(cols)) and len(cols) == 6 + 2 * 2
    mid2, x2 = mid.copy(), x.copy()
    mid2[3000:] *= 1.5
    x2[3000:] = 0.0
    f2, _ = slow_features(mid2, x2, names, spans=(240, 1440))
    assert np.allclose(f[:3000], f2[:3000])
    assert not np.allclose(f[3000:], f2[3000:])


def test_skip_starts_as_plain_gru_and_is_causal():
    th.manual_seed(0)
    plain = build_rnn("gru", 7, 8, mid_dim=16, num_layers=2, dropout=0.0)
    th.manual_seed(0)
    skip = build_rnn("gru", 7, 8, mid_dim=16, num_layers=2, dropout=0.0, skip=True)
    assert set(plain.state_dict()) < set(skip.state_dict())          # old checkpoints still load into skip=False
    skip.load_state_dict({**skip.state_dict(), **plain.state_dict()})
    plain.eval(); skip.eval()
    x = th.randn(2, 30, 7)
    assert th.allclose(plain(x)[0], skip(x)[0], atol=1e-6)        # zero-initialised shortcut
    with th.no_grad():
        skip.skip.weight.normal_()
    x2 = x.clone(); x2[:, 20:] = 0
    y, y2 = skip(x)[0], skip(x2)[0]
    assert th.allclose(y[:, :20], y2[:, :20], atol=1e-6) and not th.allclose(y[:, 20:], y2[:, 20:])
