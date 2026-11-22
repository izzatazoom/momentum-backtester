"""Download adjusted daily closes from Yahoo Finance via yfinance."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

Fetcher = Callable[..., pd.DataFrame]


def _yf_download(**kwargs: Any) -> pd.DataFrame:
    import yfinance as yf

    result: pd.DataFrame = yf.download(**kwargs)
    return result


def download_closes(
    tickers: Sequence[str],
    start: str,
    end: str | None = None,
    *,
    batch_size: int = 50,
    fetch: Fetcher = _yf_download,
) -> pd.DataFrame:
    """Return a (dates x tickers) frame of split- and dividend-adjusted closes.

    Every requested ticker gets a column, all-NaN if Yahoo has no data for it
    (typically a company that was acquired or delisted).

    ``fetch`` is injectable so tests can run without network access.
    """
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = list(tickers[i : i + batch_size])
        raw = fetch(
            tickers=batch,
            start=start,
            end=end,
            auto_adjust=True,
            actions=False,
            progress=False,
            group_by="column",
            threads=True,
        )
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw.xs("Close", axis=1, level=0)
        else:  # single ticker: flat columns
            closes = raw[["Close"]].set_axis(batch[:1], axis=1)
        frames.append(closes)

    out = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    out = out.reindex(columns=list(tickers))
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(None).normalize()
    out.index.name = "date"
    out.columns.name = None
    return out.sort_index().astype(float)
