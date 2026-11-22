"""Cross-sectional momentum scores.

At each month-end ``t`` (the last trading day of the month):

    score(t) = P(t - skip) / P(t - lookback) - 1

where ``P(t - k)`` is the close on the month-end ``k`` months before ``t``. With the
defaults (12, 1) this is the classic 12-1 momentum. It is the return over the past
year, leaving out the latest month, which tends to reverse.

Risk-adjusted momentum divides the score by the annualised volatility of daily
returns over the same window, so a steady 30% rise ranks above a jumpy one:

    vol(t) = stdev(daily returns from t - lookback to t - skip) * sqrt(252)
    risk_adjusted_score(t) = score(t) / vol(t)

Every value at ``t`` uses prices dated ``t`` or earlier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class MomentumSpec:
    lookback: int = 12
    """Months back to the start of the window."""
    skip: int = 1
    """Most recent months left out."""
    risk_adjusted: bool = False
    """Divide by window volatility."""
    min_coverage: float = 0.9
    """Share of trading days in the window that must have a price."""

    def __post_init__(self) -> None:
        if not 0 <= self.skip < self.lookback:
            raise ValueError(f"need 0 <= skip < lookback, got {self.skip}, {self.lookback}")

    @property
    def label(self) -> str:
        return f"{self.lookback}-{self.skip}{' risk-adj' if self.risk_adjusted else ''}"


@dataclass(frozen=True)
class Signal:
    spec: MomentumSpec
    scores: pd.DataFrame
    """Month-ends x tickers. NaN where it can't be computed."""
    eligible: pd.DataFrame
    """True where the stock may be ranked on that month-end."""
    coverage: pd.DataFrame
    """Share of trading days in the window with a price."""
    volatility: pd.DataFrame
    """Annualised volatility of daily returns over the window."""


def month_ends(index: pd.DatetimeIndex, complete_only: bool = True) -> pd.DatetimeIndex:
    """Last date in ``index`` for each calendar month.

    If the data stops before the final month's last business day, that month is
    unfinished and its last date isn't a real month-end, so it's dropped (unless
    ``complete_only`` is False).
    """
    last = pd.DatetimeIndex(pd.Series(index, index=index).groupby(index.to_period("M")).max())
    if complete_only and len(last) and last[-1] < pd.offsets.BMonthEnd().rollforward(last[-1]):
        last = last[:-1]
    return pd.DatetimeIndex(last, name="date")


def raw_momentum(prices: pd.DataFrame, ends: pd.DatetimeIndex, spec: MomentumSpec) -> pd.DataFrame:
    """``P(t - skip) / P(t - lookback) - 1`` on each month-end."""
    monthly = prices.loc[ends]
    return monthly.shift(spec.skip) / monthly.shift(spec.lookback) - 1


def window_stats(
    prices: pd.DataFrame, ends: pd.DatetimeIndex, spec: MomentumSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Price coverage and annualised volatility over each month-end's window.

    The window is the trading days after ``t - lookback`` up to and including
    ``t - skip``, the same span the score's return covers.
    """
    values = prices.to_numpy(dtype=float)
    daily_ret = prices.pct_change(fill_method=None).to_numpy(dtype=float)
    pos = prices.index.get_indexer(ends)
    cov = np.full((len(ends), prices.shape[1]), np.nan)
    vol = np.full_like(cov, np.nan)
    for i in range(spec.lookback, len(ends)):
        lo, hi = pos[i - spec.lookback] + 1, pos[i - spec.skip] + 1
        cov[i] = np.isfinite(values[lo:hi]).mean(axis=0)
        window = daily_ret[lo:hi]
        n = np.isfinite(window).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = np.nanstd(window, axis=0, ddof=1)
        vol[i] = np.where(n >= 2, sd * np.sqrt(TRADING_DAYS), np.nan)
    return (
        pd.DataFrame(cov, index=ends, columns=prices.columns),
        pd.DataFrame(vol, index=ends, columns=prices.columns),
    )


def compute_signal(
    prices: pd.DataFrame, membership: pd.DataFrame, spec: MomentumSpec | None = None
) -> Signal:
    """Scores and eligibility on every month-end.

    A stock is eligible on month-end ``t`` when it is an index member on ``t``, has
    a price on ``t``, has a score (so it was priced at both ends of the window), and
    has prices on at least ``spec.min_coverage`` of the window's trading days.
    """
    spec = spec or MomentumSpec()
    ends = month_ends(pd.DatetimeIndex(prices.index))
    scores = raw_momentum(prices, ends, spec)
    coverage, vol = window_stats(prices, ends, spec)
    if spec.risk_adjusted:
        scores = scores / vol.where(vol > 0)

    member = membership.reindex(index=ends, columns=prices.columns, fill_value=False)
    eligible = (
        member.astype(bool)
        & prices.loc[ends].notna()
        & scores.notna()
        & (coverage >= spec.min_coverage)
    )
    return Signal(spec, scores, eligible, coverage, vol)
