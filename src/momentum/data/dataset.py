"""Assemble everything the backtest needs into one ``Dataset``.

Raw downloads are cached, cleaning is not, so changing a cleaning rule never needs
a fresh download. Delete ``data/cache`` or pass ``refresh=True`` to download again.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from momentum.data.cache import Cache
from momentum.data.clean import clean_prices
from momentum.data.download import Fetcher, _yf_download, download_closes
from momentum.data.rates import TBILL_SYMBOL, annual_pct_to_period
from momentum.data.universe import (
    build_membership,
    load_changes,
    load_current,
    load_yahoo_symbols,
    membership_mask,
)

BENCHMARK = "QQQ"
DEFAULT_START = "2015-01-01"
DEFAULT_CACHE = Path("data/cache")


@dataclass(frozen=True)
class Dataset:
    prices: pd.DataFrame
    """Cleaned adjusted closes, trading days x tickers."""
    membership: pd.DataFrame
    """True where a ticker was in the Nasdaq-100 on that day (same shape as prices)."""
    benchmark: pd.Series
    """Cleaned adjusted closes of QQQ."""
    risk_free: pd.Series
    """Daily risk-free return from the 13-week T-bill yield."""
    intervals: pd.DataFrame
    """Membership intervals: ticker, start, end."""
    quality: pd.DataFrame
    """Per-ticker data report: dates, fills, jumps, coverage and flags."""


def coverage(prices: pd.DataFrame, membership: pd.DataFrame) -> pd.Series:
    """Share of each ticker's membership days on which it has a price.

    NaN for tickers that were never members inside the date range.
    """
    member_days = membership.sum()
    priced_member_days = (membership & prices.notna()).sum()
    return (priced_member_days / member_days.where(member_days > 0)).rename("coverage")


def late_start(prices: pd.DataFrame, membership: pd.DataFrame, tolerance: int = 5) -> pd.Series:
    """True where prices begin more than ``tolerance`` days after membership does.

    Usually a reused ticker (Yahoo's history belongs to a newer company) or a
    spin-off that kept its parent's index slot. Worth a manual look.
    """
    out = {}
    for t in prices.columns:
        member_pos = membership[t].to_numpy().nonzero()[0]
        priced_pos = prices[t].notna().to_numpy().nonzero()[0]
        out[t] = (
            len(member_pos) > 0
            and len(priced_pos) > 0
            and priced_pos[0] - member_pos[0] > tolerance
        )
    return pd.Series(out, name="late_start")


def load_dataset(
    start: str = DEFAULT_START,
    end: str | None = None,
    *,
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
    fetch: Fetcher = _yf_download,
) -> Dataset:
    """Build the dataset, downloading only if the cache is missing or stale.

    Members with no price on a given day simply can't be selected that day. Members
    Yahoo has no history for at all are listed as ``no_data`` in the quality report;
    they are the survivorship bias that free data can't remove.
    """
    intervals = build_membership(load_current(), load_changes(), start)
    tickers = sorted(intervals["ticker"].unique())
    aliases = load_yahoo_symbols()
    symbols = [aliases.get(t, t) for t in tickers] + [BENCHMARK, TBILL_SYMBOL]

    cache = Cache(cache_dir)
    meta = {"start": start, "end": end, "symbols": symbols}
    fresh = False
    if cache.exists("raw_closes") and not refresh:
        raw, saved = cache.load("raw_closes")
        fresh = all(saved.get(k) == v for k, v in meta.items())
    if not fresh:
        raw = download_closes(symbols, start, end, fetch=fetch)
        cache.save(
            "raw_closes",
            raw,
            {**meta, "downloaded": date.today().isoformat(), "yfinance": _yf_version()},
        )
    raw = raw.rename(columns={v: k for k, v in aliases.items()})

    calendar = pd.DatetimeIndex(raw.index[raw[BENCHMARK].notna()])
    prices, quality = clean_prices(raw[tickers], calendar)
    benchmark, _ = clean_prices(raw[[BENCHMARK]], calendar)
    tbill = raw[TBILL_SYMBOL].reindex(calendar).ffill().bfill()

    member = membership_mask(intervals, calendar).reindex(columns=tickers)
    cov = coverage(prices, member)
    quality = quality.join(cov).join(late_start(prices, member))
    quality["no_data"] = cov == 0

    return Dataset(
        prices=prices,
        membership=member,
        benchmark=benchmark[BENCHMARK],
        risk_free=annual_pct_to_period(tbill).rename("rf"),
        intervals=intervals,
        quality=quality,
    )


def summarize(ds: Dataset) -> str:
    """Plain-text data quality summary."""
    q = ds.quality
    members = q[q["coverage"].notna()]
    member_days = int(ds.membership.sum().sum())
    priced = int((ds.membership & ds.prices.notna()).sum().sum())
    no_data = members.index[members["no_data"]].tolist()
    late = members.index[members["late_start"] & ~members["no_data"]].tolist()
    per_day = ds.membership.sum(axis=1)
    lines = [
        f"Dates            {ds.prices.index[0].date()} to {ds.prices.index[-1].date()}"
        f" ({len(ds.prices)} trading days)",
        f"Index members    {len(members)} tickers over the period,"
        f" {per_day.min()}-{per_day.max()} on any one day",
        f"Price coverage   {priced / member_days:.1%} of member-days have a price",
        f"No data          {len(no_data)} tickers (acquired/delisted): {', '.join(no_data) or '-'}",
        f"Late start       {len(late)} tickers, prices begin after membership: "
        f"{', '.join(late) or '-'}",
        f"Gaps filled      {int(q['filled'].sum())} ticker-days",
        f"Flagged jumps    {int(q['jumps'].sum())} one-day moves above 50%"
        + (f" ({', '.join(q.index[q['jumps'] > 0])})" if q["jumps"].any() else ""),
        f"Benchmark        {BENCHMARK}, {ds.benchmark.notna().sum()} days",
        f"Risk-free        13-week T-bill, latest {(1 + ds.risk_free.iloc[-1]) ** 252 - 1:.2%}"
        " a year",
    ]
    return "\n".join(lines)


def _yf_version() -> str:
    try:
        return version("yfinance")
    except Exception:
        return "unknown"
