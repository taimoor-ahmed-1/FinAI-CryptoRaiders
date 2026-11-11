"""The tree model's history features must be causal: row t may depend on rows <= t only."""
import numpy as np

from rltrade.rnn.gbdt import feature_names, make_features


def test_features_are_causal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 7)).astype(np.float32)
    f = make_features(x, spans=(15, 60), block=3)
    x2 = x.copy()
    x2[300:] = rng.normal(size=(200, 7))          # change the future only
    f2 = make_features(x2, spans=(15, 60), block=3)
    assert f.shape == (500, 21)
    assert np.array_equal(f[:300], f2[:300])
    assert not np.array_equal(f[300:], f2[300:])


def test_feature_layout():
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    f = make_features(x, spans=(3,))
    assert np.array_equal(f[:, :2], x)                         # raw inputs first
    ema = np.empty_like(x)
    a = 2 / (3 + 1)
    ema[0] = x[0]
    for t in range(1, 6):
        ema[t] = a * x[t] + (1 - a) * ema[t - 1]
    assert np.allclose(f[:, 2:], ema, atol=1e-5)
    assert feature_names(["a", "b"], (3,)) == ["a", "b", "a__ema3", "b__ema3"]
