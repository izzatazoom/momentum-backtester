"""Performance statistics on a daily value series.

CAGR          = (V_end / V_start) ** (1 / years) - 1,   years = calendar days / 365.25
volatility    = stdev(daily returns) * sqrt(252)
max drawdown  = min over t of  V_t / max(V_s, s <= t) - 1
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def years(nav: pd.Series) -> float:
    span = pd.Timestamp(nav.index[-1]) - pd.Timestamp(nav.index[0])
    return span.days / 365.25


def cagr(nav: pd.Series) -> float:
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years(nav)) - 1)


def volatility(nav: pd.Series) -> float:
    return float(nav.pct_change().std(ddof=1) * np.sqrt(TRADING_DAYS))


def drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1


def max_drawdown(nav: pd.Series) -> float:
    return float(drawdown(nav).min())
