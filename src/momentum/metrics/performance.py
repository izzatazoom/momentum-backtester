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


def monthly_returns(nav: pd.Series) -> pd.Series:
    """Return for each calendar month, measured close-to-close between month-ends.

    The first month runs from the first value in ``nav`` to that month's last day,
    so a series starting on the 1st loses nothing.
    """
    ends = nav.groupby(pd.DatetimeIndex(nav.index).to_period("M")).last()
    base = pd.concat([pd.Series([nav.iloc[0]]), ends.iloc[:-1].reset_index(drop=True)])
    out = pd.Series(ends.to_numpy() / base.to_numpy() - 1, index=ends.index)
    return out.rename("return")


def monthly_risk_free(rf_daily: pd.Series, months: pd.PeriodIndex) -> pd.Series:
    """Compound a daily risk-free return into one rate per month."""
    grown = (1 + rf_daily).groupby(pd.DatetimeIndex(rf_daily.index).to_period("M")).prod() - 1
    return grown.reindex(months).fillna(0.0)


def sharpe(excess: pd.Series, periods_per_year: int = 12) -> float:
    """Annualised Sharpe ratio of periodic excess returns:

    Sharpe = mean(excess) / stdev(excess) * sqrt(periods_per_year)
    """
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


def excess_monthly(nav: pd.Series, rf_daily: pd.Series | None) -> pd.Series:
    """Monthly returns of ``nav`` minus the monthly T-bill return."""
    r = monthly_returns(nav)
    if rf_daily is None:
        return r
    out: pd.Series = r - monthly_risk_free(rf_daily, pd.PeriodIndex(r.index)).to_numpy()
    return out
