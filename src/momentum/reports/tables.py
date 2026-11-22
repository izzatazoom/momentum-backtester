"""Text tables for the command line."""

from __future__ import annotations

import pandas as pd

from momentum.portfolio.selection import one_class_per_company, rank_scores
from momentum.signals.momentum import Signal


def _row(frame: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    return frame.loc[[date]].iloc[0]


def resolve_month_end(signal: Signal, when: str) -> pd.Timestamp:
    """Last signal month-end on or before ``when`` (``YYYY-MM`` means that month's end)."""
    ends = signal.scores.index
    target = pd.Period(when, freq="M").end_time if len(when) == 7 else pd.Timestamp(when)
    candidates = ends[ends <= target]
    if candidates.empty:
        raise ValueError(f"no month-end on or before {when}")
    return pd.Timestamp(candidates[-1])


def picks_table(
    signal: Signal,
    prices: pd.DataFrame,
    date: pd.Timestamp,
    n: int,
    share_classes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Ranked picks on ``date``, with the two prices behind each score."""
    ends = list(pd.DatetimeIndex(signal.scores.index))
    i = ends.index(date)
    spec = signal.spec
    if i < spec.lookback:
        raise ValueError(f"{date.date()} is less than {spec.lookback} months into the data")
    recent, base = ends[i - spec.skip], ends[i - spec.lookback]

    eligible = signal.eligible
    if share_classes is not None:
        eligible = one_class_per_company(eligible, share_classes)
    ranks = _row(rank_scores(signal.scores, eligible), date).dropna().sort_values()
    top = list(ranks.index[:n])

    p_base = _row(prices, base)[top]
    p_recent = _row(prices, recent)[top]
    table = pd.DataFrame(
        {
            "rank": ranks.loc[top].astype(int),
            f"price {base.date()}": p_base,
            f"price {recent.date()}": p_recent,
            "return": p_recent / p_base - 1,
        }
    )
    if spec.risk_adjusted:
        table["volatility"] = _row(signal.volatility, date)[top]
        table["score"] = _row(signal.scores, date)[top]
    table.index.name = "ticker"
    return table


def format_picks(table: pd.DataFrame, signal: Signal, date: pd.Timestamp) -> str:
    n_eligible = int(_row(signal.eligible, date).sum())
    header = (
        f"Momentum {signal.spec.label} on {date.date()}: "
        f"top {len(table)} of {n_eligible} eligible stocks\n"
        "Trades would execute at the next trading day's close.\n"
    )
    fmt = table.copy()
    for col in fmt.columns:
        if col.startswith("price"):
            fmt[col] = fmt[col].map("{:,.2f}".format)
        elif col in ("return", "volatility"):
            fmt[col] = fmt[col].map("{:+.1%}".format if col == "return" else "{:.1%}".format)
        elif col == "score":
            fmt[col] = fmt[col].map("{:.2f}".format)
    return header + fmt.to_string()
