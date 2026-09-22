"""Risk-adjusted rewards for the reward-function comparison (brief section 3.4 d)."""
import torch as th

from .rewards import Reward


class DifferentialSharpe(Reward):
    """Differential Sharpe ratio (Moody & Saffell, 2001).

    Keeps exponential moving estimates of the first and second moment of step
    returns (A, B) and rewards each step by its marginal effect on the Sharpe ratio:

        D_t = (B_{t-1} dA - 0.5 A_{t-1} dB) / (B_{t-1} - A_{t-1}^2)^{3/2}
        dA = r_t - A_{t-1},  dB = r_t^2 - B_{t-1}

    `eta` sets the memory (~1/eta steps). The moments start from a small prior
    variance so the first steps are not divided by ~0; D is clipped for stability.
    D is already scale-free (roughly "return in units of its own volatility"), so
    its default scale is 1, not the 100 used by the equity-based rewards.
    """

    def __init__(self, num_envs, device, scale=1.0, eta=0.01, prior_std=1e-3, clip=5.0):
        super().__init__(num_envs, device, scale)
        self.eta, self.var0, self.clip = eta, prior_std ** 2, clip
        self.A = th.zeros(num_envs, device=device, dtype=th.float64)
        self.B = th.full((num_envs,), self.var0, device=device, dtype=th.float64)

    def reset(self, mask=None):
        mask = slice(None) if mask is None else mask
        self.A[mask] = 0.0
        self.B[mask] = self.var0

    def __call__(self, eq_prev, eq_next, level_prev, level_new):
        r = eq_next / eq_prev - 1.0
        dA, dB = r - self.A, r * r - self.B
        # variance floored at the prior: after long flat spells B decays towards 0 and the
        # unfloored ratio would explode on the next non-zero return
        var = (self.B - self.A ** 2).clamp(min=self.var0)
        d = ((var + self.A ** 2) * dA - 0.5 * self.A * dB) / var ** 1.5
        self.A = self.A + self.eta * dA
        self.B = self.B + self.eta * dB
        return (d * self.scale).clamp(-self.clip, self.clip)


class DrawdownPenalized(Reward):
    """Net asset change minus `lam` x every INCREASE in drawdown from the running peak.

    With lam=1 a loss that deepens the drawdown counts double; recovering from a
    drawdown is not penalised.
    """

    def __init__(self, num_envs, device, scale=100.0, lam=1.0):
        super().__init__(num_envs, device, scale)
        self.lam = lam
        self.peak = th.ones(num_envs, device=device, dtype=th.float64)

    def reset(self, mask=None):
        mask = slice(None) if mask is None else mask
        self.peak[mask] = 1.0

    def __call__(self, eq_prev, eq_next, level_prev, level_new):
        dd_prev = 1.0 - eq_prev / self.peak
        self.peak = th.maximum(self.peak, eq_next)
        dd_next = 1.0 - eq_next / self.peak
        return ((eq_next - eq_prev) - self.lam * (dd_next - dd_prev).clamp(min=0)) * self.scale
