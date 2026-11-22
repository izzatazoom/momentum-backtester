"""Rates, cache, download and the full dataset pipeline, all offline."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from momentum.data.cache import Cache
from momentum.data.dataset import BENCHMARK, coverage, late_start, load_dataset, summarize
from momentum.data.download import download_closes
from momentum.data.rates import TBILL_SYMBOL, annual_pct_to_period


def test_rate_compounds_back_to_annual() -> None:
    daily = annual_pct_to_period(pd.Series([5.0]))
    assert (1 + daily.iloc[0]) ** 252 == pytest.approx(1.05)
    monthly = annual_pct_to_period(pd.Series([12.0]), periods_per_year=12)
    assert monthly.iloc[0] == pytest.approx(1.12 ** (1 / 12) - 1)  # ~0.949%, not 1%


def test_cache_round_trip(tmp_path: Path) -> None:
    df = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    cache = Cache(tmp_path)
    assert not cache.exists("x")
    cache.save("x", df, {"start": "2024-01-01"})
    got, meta = cache.load("x")
    pd.testing.assert_frame_equal(got, df, check_freq=False)
    assert meta == {"start": "2024-01-01"}


def fake_yahoo(missing: set[str] = frozenset()) -> Any:  # type: ignore[assignment]
    """Imitate yf.download: MultiIndex (field, ticker) columns, no data for `missing`."""
    calls: list[list[str]] = []

    def fetch(tickers: list[str], start: str, end: str | None, **_: Any) -> pd.DataFrame:
        calls.append(tickers)
        idx = pd.bdate_range(start, end or "2016-12-30", tz="America/New_York")
        rng = np.random.default_rng(len(calls))
        data = {}
        for t in tickers:
            if t in missing:
                continue
            level = 3.0 if t == TBILL_SYMBOL else 100.0
            walk = level * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
            data[("Close", t)] = data[("Open", t)] = walk
        return pd.DataFrame(data, index=idx)

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


def test_download_batches_and_keeps_missing_tickers() -> None:
    fetch = fake_yahoo(missing={"GONE"})
    out = download_closes(["A", "B", "GONE"], "2024-01-01", "2024-01-31", batch_size=2, fetch=fetch)
    assert fetch.calls == [["A", "B"], ["GONE"]]
    assert list(out.columns) == ["A", "B", "GONE"]
    assert out["GONE"].isna().all()
    assert out.index.tz is None


def test_coverage() -> None:
    idx = pd.bdate_range("2024-01-01", periods=4)
    member = pd.DataFrame({"A": [True, True, False, False], "B": [True] * 4}, index=idx)
    prices = pd.DataFrame({"A": [1, np.nan, np.nan, np.nan], "B": [1.0] * 4}, index=idx)
    assert coverage(prices, member).to_dict() == {"A": 0.5, "B": 1.0}


def test_late_start() -> None:
    idx = pd.bdate_range("2024-01-01", periods=10)
    member = pd.DataFrame({"A": [True] * 10, "B": [True] * 10}, index=idx)
    prices = pd.DataFrame({"A": [1.0] * 10, "B": [np.nan] * 7 + [1.0] * 3}, index=idx)
    assert late_start(prices, member, tolerance=5).to_dict() == {"A": False, "B": True}


def test_load_dataset_offline(tmp_path: Path) -> None:
    fetch = fake_yahoo(missing={"BRCM", "NLOK"})  # NLOK is fetched as GEN
    ds = load_dataset("2015-01-01", "2016-12-30", cache_dir=tmp_path, fetch=fetch)

    assert ds.prices.shape == ds.membership.shape
    assert ds.prices.index.equals(ds.benchmark.index)
    assert ds.quality.loc["BRCM", "no_data"]
    assert not ds.quality.loc["AAPL", "no_data"]
    assert ds.prices["NLOK"].notna().all()  # renamed column came back as NLOK
    assert "GEN" in fetch.calls[0] + fetch.calls[-1] or any("GEN" in c for c in fetch.calls)
    assert ds.risk_free.notna().all()
    assert BENCHMARK not in ds.prices.columns
    assert "No data          1 tickers (acquired/delisted): BRCM" in summarize(ds)

    # second call reads the cache instead of downloading
    n_calls = len(fetch.calls)
    load_dataset("2015-01-01", "2016-12-30", cache_dir=tmp_path, fetch=fetch)
    assert len(fetch.calls) == n_calls


def test_full_pipeline_offline(tmp_path: Path) -> None:
    from momentum.backtest.strategy import (
        StrategyConfig,
        buy_and_hold,
        run_equal_weight_universe,
        run_momentum,
    )

    ds = load_dataset("2015-01-01", "2016-12-30", cache_dir=tmp_path, fetch=fake_yahoo())
    config = StrategyConfig(top_n=20)
    strat = run_momentum(ds, config)
    universe = run_equal_weight_universe(ds, config)

    assert strat.nav.index[0] == pd.Timestamp("2016-02-01")  # first signal Jan 2016, +1 day
    assert (strat.trades["holdings"] == 20).all()
    assert (universe.trades["holdings"] > 20).all()
    assert (strat.cash.abs() < 1e-9).all()
    assert not (strat.weights[["GOOG", "GOOGL"]] > 0).all(axis=1).any()
    assert buy_and_hold(ds.benchmark, strat.nav.index[0]).iloc[0] == 1.0
