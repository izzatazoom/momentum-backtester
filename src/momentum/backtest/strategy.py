"""Run the momentum strategy and its benchmarks on a ``Dataset``."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from momentum.backtest.engine import BacktestResult, run_backtest
from momentum.data.dataset import Dataset
from momentum.data.universe import load_share_classes
from momentum.portfolio.costs import CostModel, FlatCost
from momentum.portfolio.selection import equal_weights, one_class_per_company, select_top
from momentum.signals import MomentumSpec, Signal, compute_signal


@dataclass(frozen=True)
class StrategyConfig:
    spec: MomentumSpec = field(default_factory=MomentumSpec)
    top_n: int = 20
    cost: CostModel = field(default_factory=FlatCost)
    lag: int = 1

    @property
    def label(self) -> str:
        return f"Momentum {self.spec.label}, top {self.top_n}"


def _first_active(weights: pd.DataFrame) -> pd.DataFrame:
    """Drop the leading rows where nothing can be held yet (the warm-up year)."""
    active = weights.sum(axis=1) > 0
    if not active.any():
        raise ValueError("no month-end has any eligible stock")
    first = active.to_numpy().argmax()
    return weights.iloc[first:]


def momentum_targets(signal: Signal, top_n: int, share_classes: pd.DataFrame) -> pd.DataFrame:
    """Equal weights in the ``top_n`` highest-scoring eligible stocks each month-end."""
    picked = select_top(signal.scores, signal.eligible, top_n, share_classes)
    return _first_active(equal_weights(picked))


def universe_targets(signal: Signal, share_classes: pd.DataFrame) -> pd.DataFrame:
    """Equal weights in every eligible stock: the index without stock-picking."""
    eligible = one_class_per_company(signal.eligible, share_classes)
    return _first_active(equal_weights(eligible))


def run_momentum(ds: Dataset, config: StrategyConfig | None = None) -> BacktestResult:
    config = config or StrategyConfig()
    signal = compute_signal(ds.prices, ds.membership, config.spec)
    targets = momentum_targets(signal, config.top_n, load_share_classes())
    return run_backtest(ds.prices, targets, ds.risk_free, config.cost, lag=config.lag)


def run_equal_weight_universe(ds: Dataset, config: StrategyConfig | None = None) -> BacktestResult:
    """Benchmark with the same eligibility, lag and costs as the strategy, minus selection."""
    config = config or StrategyConfig()
    signal = compute_signal(ds.prices, ds.membership, config.spec)
    targets = universe_targets(signal, load_share_classes())
    return run_backtest(ds.prices, targets, ds.risk_free, config.cost, lag=config.lag)


def buy_and_hold(prices: pd.Series, start: pd.Timestamp) -> pd.Series:
    """Value of 1 dollar put into ``prices`` at the close of ``start``. No costs."""
    p = prices.loc[start:].ffill()
    nav: pd.Series = p / float(p.iloc[0])
    return nav.rename(str(prices.name))
