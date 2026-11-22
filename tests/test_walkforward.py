"""Walk-forward selection, stitching, and the overfitting statistics."""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from momentum.backtest.strategy import StrategyConfig
from momentum.backtest.walkforward import Grid, run_grid, smoothed_scores, walk_forward
from momentum.data.dataset import Dataset
from momentum.metrics.overfitting import (
    EULER_GAMMA,
    deflated_sharpe,
    expected_max_sharpe,
    probabilistic_sharpe,
)
from momentum.metrics.performance import monthly_returns, sharpe
from momentum.portfolio.costs import FlatCost
from momentum.signals import MomentumSpec

SMALL = Grid(lookback=(3, 6), skip=(1,), top_n=(1, 2), risk_adjusted=(False,))


def synthetic_dataset(seed: int = 0, years: int = 6) -> Dataset:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(
        "2015-07-01", periods=252 * years
    )  # 6-month lookback trades from Jan 2016
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    drift = np.array([0.0008, 0.0004, 0.0, -0.0002, 0.0006])
    rets = rng.normal(drift, 0.015, (len(days), len(tickers)))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=days, columns=tickers)
    members = pd.DataFrame(True, index=days, columns=tickers)
    bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(days)))), days)
    empty = pd.DataFrame(columns=["ticker", "start", "end"])
    return Dataset(prices, members, bench.rename("QQQ"), pd.Series(0.0, days), empty, empty)


@pytest.fixture(scope="module")
def ds() -> Dataset:
    return synthetic_dataset()


# --- the grid -----------------------------------------------------------------------


def test_default_grid_has_48_settings() -> None:
    configs = Grid().configs()
    assert len(configs) == 48
    assert len({c.label for c in configs}) == 48


def test_smoothing_averages_with_grid_neighbours() -> None:
    """2 x 2 grid (lookback 3/6, top 1/2). Each cell has two neighbours, no diagonals.

                top 1   top 2
    lookback 3   1.0     2.0
    lookback 6   3.0     6.0

    lookback 3, top 1 -> mean(1, 3, 2) = 2.0
    lookback 6, top 2 -> mean(6, 3, 2) = 3.67
    """
    configs = SMALL.configs()
    raw = {
        StrategyConfig(MomentumSpec(3, 1), 1).label: 1.0,
        StrategyConfig(MomentumSpec(3, 1), 2).label: 2.0,
        StrategyConfig(MomentumSpec(6, 1), 1).label: 3.0,
        StrategyConfig(MomentumSpec(6, 1), 2).label: 6.0,
    }
    out = smoothed_scores(pd.Series(raw), configs)
    assert out[StrategyConfig(MomentumSpec(3, 1), 1).label] == pytest.approx(2.0)
    assert out[StrategyConfig(MomentumSpec(6, 1), 2).label] == pytest.approx(11 / 3)


def test_lone_spike_loses_to_a_good_plateau() -> None:
    """Raw winner sits among bad neighbours, so smoothing prefers the steady block."""
    configs = Grid(lookback=(3, 6, 9), skip=(1,), top_n=(1,), risk_adjusted=(False,)).configs()
    labels = [c.label for c in configs]  # lookback 3, 6, 9
    raw = pd.Series([0.2, 1.0, 0.1], index=labels)  # 6 spikes alone
    plateau = pd.Series([0.8, 0.7, 0.8], index=labels)
    assert smoothed_scores(raw, configs).max() < smoothed_scores(plateau, configs).max()


# --- walk-forward -------------------------------------------------------------------


def test_training_windows_and_first_test_year(ds: Dataset) -> None:
    grid = run_grid(ds, SMALL.configs(FlatCost(0)))
    wf = walk_forward(ds, grid, min_train_months=35)
    first = wf.windows.iloc[0]
    assert wf.windows.index[0] == 2019
    assert first["train to"] == "2018-12"
    assert (wf.windows["months"].diff().dropna() == 12).all()  # expanding: +12 a year


def test_rolling_window_is_capped(ds: Dataset) -> None:
    grid = run_grid(ds, SMALL.configs(FlatCost(0)))
    wf = walk_forward(ds, grid, window="rolling", rolling_months=36, min_train_months=24)
    assert (wf.windows["months"] <= 36).all()


def test_oos_portfolio_holds_what_the_chosen_setting_picked(ds: Dataset) -> None:
    grid = run_grid(ds, SMALL.configs(FlatCost(0)))
    wf = walk_forward(ds, grid)
    for exec_day, trade in wf.oos.trades.iterrows():
        chosen = wf.windows.loc[exec_day.year, "chosen"]
        target = grid.targets[chosen].loc[trade["signal_date"]]
        held = wf.oos.weights.loc[exec_day]
        assert set(held.index[held > 1e-12]) == set(target.index[target > 0])


def test_choice_uses_only_the_past(ds: Dataset) -> None:
    """Wreck every price after 2019: the 2019 and 2020 choices can't change."""
    configs = SMALL.configs(FlatCost(0))
    before = walk_forward(ds, run_grid(ds, configs))

    prices = ds.prices.copy()
    later = prices.index > pd.Timestamp("2019-12-31")
    prices.loc[later] *= np.linspace(1, 0.2, later.sum())[:, None] * [1, 3, 1, 3, 1]
    ds2 = Dataset(prices, ds.membership, ds.benchmark, ds.risk_free, ds.intervals, ds.quality)
    after = walk_forward(ds2, run_grid(ds2, configs))

    assert before.windows.loc[[2019, 2020], "chosen"].equals(
        after.windows.loc[[2019, 2020], "chosen"]
    )
    cut = pd.Timestamp("2019-12-31")
    pd.testing.assert_series_equal(before.oos.nav.loc[:cut], after.oos.nav.loc[:cut])


# --- performance helpers ------------------------------------------------------------


def test_monthly_returns_hand_checked() -> None:
    nav = pd.Series(
        [100, 105, 110, 121],
        index=pd.to_datetime(["2024-01-15", "2024-01-20", "2024-01-31", "2024-02-29"]),
    )
    assert monthly_returns(nav).tolist() == pytest.approx([0.10, 0.10])


def test_sharpe_hand_checked() -> None:
    # mean 2%, stdev 1.414% -> 0.02 / 0.01414 * sqrt(12) = 4.899
    assert sharpe(pd.Series([0.01, 0.03])) == pytest.approx(0.02 / math.sqrt(2e-4) * math.sqrt(12))


# --- overfitting statistics ---------------------------------------------------------


def test_expected_max_sharpe_two_trials() -> None:
    """Two trials with Sharpes 0 and sqrt(2) have stdev 1.

    SR0 = 1 * ((1 - g) * Phi^-1(1/2) + g * Phi^-1(1 - 1/(2e))) = g * Phi^-1(0.816) = about 0.52
    """
    got = expected_max_sharpe(np.array([0.0, math.sqrt(2)]))
    expected = EULER_GAMMA * NormalDist().inv_cdf(1 - 1 / (2 * math.e))
    assert got == pytest.approx(expected)
    assert got == pytest.approx(0.52, abs=0.01)


def test_hurdle_grows_with_more_trials() -> None:
    rng = np.random.default_rng(0)
    few = expected_max_sharpe(rng.normal(0, 0.1, 5))
    many = expected_max_sharpe(rng.normal(0, 0.1, 500))
    assert many > few


def test_psr_is_one_half_at_its_own_sharpe() -> None:
    r = pd.Series(np.random.default_rng(1).normal(0.01, 0.04, 120))
    assert probabilistic_sharpe(r, float(r.mean() / r.std())) == pytest.approx(0.5)


def test_psr_matches_formula() -> None:
    r = pd.Series(np.random.default_rng(2).normal(0.008, 0.05, 96))
    sr = r.mean() / r.std()
    g3, g4 = r.skew(), r.kurt() + 3
    z = sr * math.sqrt(len(r) - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    assert probabilistic_sharpe(r) == pytest.approx(NormalDist().cdf(z))


def test_deflated_sharpe_punishes_a_lucky_search() -> None:
    """200 strategies of pure noise: the best looks good, but DSR sees through it."""
    rng = np.random.default_rng(4)
    noise = pd.DataFrame(rng.normal(0, 0.04, (120, 200)))
    d = deflated_sharpe(noise)
    assert d.best_sharpe > 0.5  # the luckiest one looks decent...
    assert d.psr > 0.85  # ...and naive PSR is fooled...
    assert d.dsr < 0.6  # ...but deflating for 200 tries is not
