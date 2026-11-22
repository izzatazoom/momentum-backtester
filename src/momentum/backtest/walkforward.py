"""Walk-forward testing: choose settings on the past, trade them on the future.

Every January:

1. Score every setting in the grid on the training months before that January.
   The score is the Sharpe ratio of monthly excess returns, *smoothed*: averaged
   with the settings next to it in the grid (one step of lookback or of stocks
   held, same skip and risk adjustment). A setting that only works on its own,
   surrounded by bad neighbours, is probably luck.
2. Trade the winning setting for the next 12 months.

The test years run as one continuous portfolio. When the setting changes,
existing holdings carry over and only the trades needed are made and paid for.

No look-ahead: the setting for test year Y is chosen from returns up to the
December close of Y - 1. The December month-end signal, whose trades execute in
early January, already uses the new setting.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from momentum.backtest.engine import BacktestResult, run_backtest
from momentum.backtest.strategy import StrategyConfig, momentum_targets
from momentum.data.dataset import Dataset
from momentum.data.universe import load_share_classes
from momentum.metrics.performance import excess_monthly, sharpe
from momentum.portfolio.costs import CostModel, FlatCost
from momentum.signals import MomentumSpec, Signal, compute_signal


@dataclass(frozen=True)
class Grid:
    lookback: Sequence[int] = (3, 6, 9, 12)
    skip: Sequence[int] = (0, 1)
    top_n: Sequence[int] = (10, 20, 30)
    risk_adjusted: Sequence[bool] = (False, True)

    def configs(self, cost: CostModel | None = None, lag: int = 1) -> list[StrategyConfig]:
        cost = cost or FlatCost()
        return [
            StrategyConfig(MomentumSpec(lb, sk, ra), n, cost, lag)
            for ra, sk, lb, n in itertools.product(
                self.risk_adjusted, self.skip, self.lookback, self.top_n
            )
        ]


@dataclass(frozen=True)
class GridRun:
    """Every grid setting backtested over the full period."""

    configs: list[StrategyConfig]
    targets: dict[str, pd.DataFrame]
    navs: pd.DataFrame
    """Daily values from the common start, all starting at 1."""
    excess: pd.DataFrame
    """Monthly excess returns (over T-bills) from the common start."""

    @property
    def labels(self) -> list[str]:
        return [c.label for c in self.configs]

    def by_label(self, label: str) -> StrategyConfig:
        return next(c for c in self.configs if c.label == label)


def run_grid(ds: Dataset, configs: list[StrategyConfig]) -> GridRun:
    share_classes = load_share_classes()
    signals: dict[MomentumSpec, Signal] = {}
    targets, navs = {}, {}
    for config in configs:
        if config.spec not in signals:
            signals[config.spec] = compute_signal(ds.prices, ds.membership, config.spec)
        w = momentum_targets(signals[config.spec], config.top_n, share_classes)
        result = run_backtest(ds.prices, w, ds.risk_free, config.cost, lag=config.lag)
        targets[config.label] = w
        navs[config.label] = result.nav

    # Compare everyone over the same dates: from the latest first trade.
    start = max(nav.index[0] for nav in navs.values())
    nav = pd.DataFrame({k: v.loc[start:] / v.loc[start] for k, v in navs.items()})
    excess = pd.DataFrame({k: excess_monthly(nav[k], ds.risk_free) for k in nav})
    return GridRun(configs, targets, nav, excess)


def smoothed_scores(scores: pd.Series, configs: list[StrategyConfig]) -> pd.Series:
    """Average each setting's score with its grid neighbours.

    Neighbours share skip and risk adjustment and are one step away in lookback
    or in stocks held (not diagonally).
    """
    lookbacks = sorted({c.spec.lookback for c in configs})
    tops = sorted({c.top_n for c in configs})
    key = {
        (
            c.spec.risk_adjusted,
            c.spec.skip,
            lookbacks.index(c.spec.lookback),
            tops.index(c.top_n),
        ): c
        for c in configs
    }
    out = {}
    for (ra, sk, i, j), c in key.items():
        around = [(i, j), (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        vals = [scores[key[(ra, sk, a, b)].label] for a, b in around if (ra, sk, a, b) in key]
        out[c.label] = float(np.nanmean(vals))
    return pd.Series(out).reindex(scores.index)


@dataclass(frozen=True)
class WalkForwardResult:
    window: str
    windows: pd.DataFrame
    """One row per test year: training span, chosen setting, its training Sharpe."""
    oos: BacktestResult
    """The continuous out-of-sample portfolio."""
    grid: GridRun = field(repr=False)


def walk_forward(
    ds: Dataset,
    grid: GridRun,
    *,
    window: str = "expanding",
    rolling_months: int = 60,
    min_train_months: int = 35,
    smooth: bool = True,
) -> WalkForwardResult:
    """Run the yearly choose-then-trade loop and the stitched out-of-sample portfolio."""
    if window not in ("expanding", "rolling"):
        raise ValueError("window must be 'expanding' or 'rolling'")
    months = pd.PeriodIndex(grid.excess.index)
    years = sorted({p.year for p in months})

    rows = []
    for year in years:
        train = grid.excess[months < pd.Period(f"{year}-01", "M")]
        if window == "rolling":
            train = train.iloc[-rolling_months:]
        if len(train) < min_train_months:
            continue
        raw = train.apply(sharpe)
        score = smoothed_scores(raw, grid.configs) if smooth else raw
        chosen = str(score.idxmax())
        rows.append(
            {
                "test year": year,
                "train from": str(train.index[0]),
                "train to": str(train.index[-1]),
                "months": len(train),
                "chosen": chosen,
                "train Sharpe": raw[chosen],
                "smoothed": score[chosen],
            }
        )
    if not rows:
        raise ValueError("not enough history for a single training window")
    windows = pd.DataFrame(rows).set_index("test year")

    oos = run_backtest(
        ds.prices,
        _stitched_targets(ds, grid, windows),
        ds.risk_free,
        grid.configs[0].cost,
        lag=grid.configs[0].lag,
    )
    return WalkForwardResult(window, windows, oos, grid)


def _stitched_targets(ds: Dataset, grid: GridRun, windows: pd.DataFrame) -> pd.DataFrame:
    """Month-end targets, each taken from the setting chosen for its execution year."""
    index = pd.DatetimeIndex(ds.prices.index)
    lag = grid.configs[0].lag
    signal_dates = sorted(set().union(*(t.index for t in grid.targets.values())))
    rows = {}
    for t in signal_dates:
        pos = int(index.get_indexer(pd.DatetimeIndex([t]))[0]) + lag
        if pos >= len(index):
            continue
        year = index[pos].year
        if year not in windows.index:
            continue
        w = grid.targets[str(windows.loc[year, "chosen"])]
        rows[t] = w.loc[t] if t in w.index else pd.Series(0.0, index=w.columns)
    return pd.DataFrame(rows).T.fillna(0.0)
