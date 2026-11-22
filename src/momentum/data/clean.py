"""Clean raw price data.

Rules, in order:

1. Sort dates, drop duplicate dates (keep the last), keep only benchmark trading days.
2. Treat zero, negative and infinite prices as missing.
3. Fill gaps of up to ``max_gap`` trading days with the last price. Longer gaps stay
   missing, since a stock that hasn't traded for weeks shouldn't look tradeable.
4. Never fill before a stock's first price or after its last one. After the last
   price the stock is treated as delisted and the backtest exits at that price.

Splits and dividends are handled upstream by using adjusted closes. One-day moves
bigger than ``jump_threshold`` are flagged, not changed, since they are usually real
(earnings, takeovers) but occasionally a bad split adjustment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fill_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    """Forward-fill interior runs of NaN no longer than ``max_gap``."""
    s = series.copy()
    valid = s.notna()
    if not valid.any():
        return s
    first, last = valid.idxmax(), valid[::-1].idxmax()
    interior = (s.index >= first) & (s.index <= last)
    run_id = valid.cumsum()
    run_len = (~valid).groupby(run_id).transform("sum")
    fillable = (~valid) & interior & (run_len <= max_gap)
    s[fillable] = s.ffill()[fillable]
    return s


def clean_prices(
    raw: pd.DataFrame,
    calendar: pd.DatetimeIndex | None = None,
    *,
    max_gap: int = 5,
    jump_threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean a (dates x tickers) price frame.

    Args:
        raw: adjusted closes, possibly with gaps and bad values.
        calendar: trading days to keep (normally the benchmark's). Defaults to the
            dates where any ticker has a price.
        max_gap: longest run of missing days to fill.
        jump_threshold: absolute one-day return above which a move is flagged.

    Returns:
        ``(prices, report)``. ``report`` has one row per ticker with its first and
        last date, days filled, days still missing between first and last, and
        number of flagged jumps.
    """
    df = raw.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.mask(~np.isfinite(df) | (df <= 0))
    if calendar is None:
        calendar = pd.DatetimeIndex(df.index[df.notna().any(axis=1)])
    df = df.reindex(calendar)

    cleaned = df.apply(fill_short_gaps, max_gap=max_gap)
    returns = cleaned.pct_change(fill_method=None)

    rows = []
    for t in cleaned.columns:
        before, after = df[t], cleaned[t]
        valid = after.dropna()
        if valid.empty:
            rows.append({"ticker": t, "first": pd.NaT, "last": pd.NaT})
            continue
        span = after.loc[valid.index[0] : valid.index[-1]]
        rows.append(
            {
                "ticker": t,
                "first": valid.index[0],
                "last": valid.index[-1],
                "filled": int((before.isna() & after.notna()).sum()),
                "missing": int(span.isna().sum()),
                "jumps": int((returns[t].abs() > jump_threshold).sum()),
            }
        )
    report = pd.DataFrame(rows).set_index("ticker")
    for col in ("filled", "missing", "jumps"):
        report[col] = report.get(col, pd.Series(dtype=float)).fillna(0).astype(int)
    return cleaned, report
