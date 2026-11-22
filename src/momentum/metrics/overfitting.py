"""Is the best backtest just the luckiest of many tries?

Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012). This is the
probability that the true Sharpe beats ``benchmark``, given the observed Sharpe
``SR`` (per period, not annualised), the number of returns ``T``, and their skew
``g3`` and kurtosis ``g4``:

    PSR = Phi( (SR - benchmark) * sqrt(T - 1) / sqrt(1 - g3 * SR + (g4 - 1) / 4 * SR^2) )

Fat tails and negative skew widen the denominator, so they lower the PSR.

Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014). This is the PSR with the
benchmark raised to the Sharpe you'd expect from the *best* of N strategies that
are all worthless:

    SR0 = sqrt(Var[SR_n]) * ((1 - e) * Phi^-1(1 - 1/N) + e * Phi^-1(1 - 1/(N * exp(1))))

where ``Var[SR_n]`` is the variance of the Sharpes across the N trials and
``e`` is the Euler-Mascheroni constant (0.5772). A DSR above 0.95 means the best
strategy is unlikely to be a fluke of the search. Below 0.5 means picking the best
of N is about as good as luck.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


def probabilistic_sharpe(returns: pd.Series, benchmark: float = 0.0) -> float:
    """PSR of per-period returns against a per-period Sharpe ``benchmark``."""
    r = returns.dropna().to_numpy(dtype=float)
    t = len(r)
    sr = r.mean() / r.std(ddof=1)
    s = pd.Series(r)
    g3 = float(s.skew())  # type: ignore[arg-type]
    g4 = float(s.kurt()) + 3.0  # type: ignore[arg-type]  # pandas gives excess kurtosis
    denom = 1 - g3 * sr + (g4 - 1) / 4 * sr**2
    return _N.cdf((sr - benchmark) * math.sqrt(t - 1) / math.sqrt(denom))


def expected_max_sharpe(trial_sharpes: np.ndarray) -> float:
    """``SR0``: the best Sharpe expected from N zero-skill trials (per period)."""
    n = len(trial_sharpes)
    if n < 2:
        return 0.0
    sd = float(np.std(trial_sharpes, ddof=1))
    return sd * (
        (1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n) + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n * math.e))
    )


@dataclass(frozen=True)
class DeflatedSharpe:
    n_trials: int
    best_sharpe: float
    """Annualised Sharpe of the best trial."""
    hurdle: float
    """Annualised SR0: what luck alone would deliver as the best of N."""
    psr: float
    """Probability that the best trial's true Sharpe is above 0."""
    dsr: float
    """Probability that it is above the luck hurdle."""


def deflated_sharpe(trial_returns: pd.DataFrame, periods_per_year: int = 12) -> DeflatedSharpe:
    """DSR of the best column of ``trial_returns`` (per-period excess returns)."""
    per_period = (trial_returns.mean() / trial_returns.std(ddof=1)).to_numpy(dtype=float)
    best = int(np.nanargmax(per_period))
    sr0 = expected_max_sharpe(per_period[np.isfinite(per_period)])
    best_returns = trial_returns.iloc[:, best]
    scale = math.sqrt(periods_per_year)
    return DeflatedSharpe(
        n_trials=trial_returns.shape[1],
        best_sharpe=per_period[best] * scale,
        hurdle=sr0 * scale,
        psr=probabilistic_sharpe(best_returns, 0.0),
        dsr=probabilistic_sharpe(best_returns, sr0),
    )
