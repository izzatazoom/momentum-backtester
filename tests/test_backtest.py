"""Backtest engine on tiny datasets you can follow with a calculator."""

import numpy as np
import pandas as pd
import pytest

from momentum.backtest.engine import run_backtest
from momentum.backtest.strategy import buy_and_hold
from momentum.metrics.performance import cagr, max_drawdown
from momentum.portfolio.costs import FlatCost, VolatilityCost

DAYS = pd.bdate_range("2024-01-29", periods=8)  # Mon 29 Jan .. Wed 7 Feb
T0 = DAYS[3]  # Thu 1 Feb: signal date (pretend it's a month-end)
NO_COST = FlatCost(0)


def prices(**cols: list[float]) -> pd.DataFrame:
    return pd.DataFrame(cols, index=DAYS[: len(next(iter(cols.values())))], dtype=float)


def targets(date: pd.Timestamp, **weights: float) -> pd.DataFrame:
    return pd.DataFrame(weights, index=[date])


def test_one_day_lag_misses_the_signal_day_jump() -> None:
    """Signal on day 3. A jumps 100 -> 200 on day 4, the execution day itself.

    Buying at day 4's close (200) means the jump is NOT earned; only the move
    from 200 to 220 is. Trading at day 3's close would have doubled the money:
    that's look-ahead.
    """
    p = prices(A=[100, 100, 100, 100, 200, 220, 220, 220])
    r = run_backtest(p, targets(T0, A=1.0), cost=NO_COST)
    assert r.nav.index[0] == DAYS[4]
    assert r.nav.iloc[0] == pytest.approx(1.0)
    assert r.nav.loc[DAYS[5]] == pytest.approx(1.1)  # 220 / 200


def test_longer_lag() -> None:
    p = prices(A=[100, 100, 100, 100, 200, 220, 242, 242])
    r = run_backtest(p, targets(T0, A=1.0), cost=NO_COST, lag=2)
    assert r.nav.index[0] == DAYS[5]
    assert r.nav.loc[DAYS[6]] == pytest.approx(1.1)  # 242 / 220


def test_lag_zero_is_refused() -> None:
    p = prices(A=[100.0] * 8)
    with pytest.raises(ValueError, match="lag"):
        run_backtest(p, targets(T0, A=1.0), lag=0)


def test_first_purchase_cost() -> None:
    """$10,000 into two stocks at 10 bps.

    cost = 0.001 * (amount bought) and amount bought = 10,000 - cost, so
    cost = 10,000 * 0.001 / 1.001 = $9.99001.
    """
    p = prices(A=[50.0] * 8, B=[20.0] * 8)
    r = run_backtest(p, targets(T0, A=0.5, B=0.5), cost=FlatCost(10), initial=10_000)
    expected_cost = 10_000 * 0.001 / 1.001
    assert r.nav.iloc[0] == pytest.approx(10_000 - expected_cost)
    assert r.trades["cost"].iloc[0] * 10_000 == pytest.approx(expected_cost)
    assert r.trades["turnover"].iloc[0] == pytest.approx(0.5 * (10_000 - expected_cost) / 10_000)


def test_rebalance_back_to_equal_weight() -> None:
    """Hold A and B 50/50 with $100. A doubles, so: A = $100, B = $50, V = $150.

    Drifted weights are 2/3 and 1/3. Back to 1/2 each means selling $25 of A and
    buying $25 of B. Turnover = 1/2 * (25 + 25) / 150 = 1/6. At 10 bps that costs
    about 0.001 * 50 = $0.05.
    """
    p = prices(A=[10, 10, 10, 10, 10, 20, 20, 20], B=[10.0] * 8)
    t = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]}, index=[DAYS[3], DAYS[5]])
    r = run_backtest(p, t, cost=FlatCost(10), initial=100)

    second = r.trades.iloc[1]
    # the first trade cost 100 * 0.001 / 1.001, so values are a hair under 100 and 50
    v_before = 150 * (1 - 0.001 / 1.001)
    assert second["turnover"] == pytest.approx(1 / 6, rel=1e-6)
    assert second["cost"] * v_before == pytest.approx(0.05, rel=1e-3)
    end = r.weights.loc[DAYS[6]]
    assert end["A"] == pytest.approx(0.5) and end["B"] == pytest.approx(0.5)


def test_weights_drift_between_rebalances() -> None:
    p = prices(A=[10, 10, 10, 10, 10, 20, 20, 20], B=[10.0] * 8)
    r = run_backtest(p, targets(T0, A=0.5, B=0.5), cost=NO_COST)
    assert r.weights.loc[DAYS[5], "A"] == pytest.approx(2 / 3)  # not reset to 1/2


def test_full_replacement_is_100_percent_turnover() -> None:
    p = prices(A=[10.0] * 8, B=[10.0] * 8)
    t = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=[DAYS[3], DAYS[5]])
    r = run_backtest(p, t, cost=NO_COST)
    assert r.trades["turnover"].iloc[1] == pytest.approx(1.0)


def test_delisted_stock_is_sold_at_last_price_and_cash_earns_rf() -> None:
    """A is bought at 100, last trades at 80 on day 5, then disappears.

    From day 6 the $0.80 sits in cash earning 1% a day (a silly rate so it shows).
    """
    p = prices(A=[100, 100, 100, 100, 100, 80, np.nan, np.nan])
    rf = pd.Series(0.01, index=DAYS)
    r = run_backtest(p, targets(T0, A=1.0), risk_free=rf, cost=NO_COST)
    assert r.nav.loc[DAYS[5]] == pytest.approx(0.8)
    assert r.cash.loc[DAYS[6]] == pytest.approx(1.0)
    assert r.nav.loc[DAYS[7]] == pytest.approx(0.8 * 1.01**2)


def test_short_gap_holds_last_value_instead_of_delisting() -> None:
    p = prices(A=[100, 100, 100, 100, 100, np.nan, 110, 110])
    r = run_backtest(p, targets(T0, A=1.0), cost=NO_COST)
    assert r.nav.loc[DAYS[5]] == pytest.approx(1.0)
    assert r.nav.loc[DAYS[6]] == pytest.approx(1.1)
    assert r.cash.max() == pytest.approx(0.0)


def test_unpriced_target_stays_in_cash_and_portfolio_never_borrows() -> None:
    p = prices(A=[10.0] * 8, B=[10, 10, 10, 10, np.nan, 10, 10, 10])
    r = run_backtest(p, targets(T0, A=0.5, B=0.5), cost=NO_COST)
    assert r.cash.iloc[0] == pytest.approx(0.5)
    assert (r.cash >= -1e-12).all()


def test_rf_uses_previous_days_rate() -> None:
    p = prices(A=[10.0] * 8)
    rf = pd.Series([0.0] * 5 + [0.5, 0.0, 0.0], index=DAYS)  # rate printed on day 5
    r = run_backtest(p, targets(T0, A=0.0), risk_free=rf, cost=NO_COST)
    assert r.nav.loc[DAYS[5]] == pytest.approx(1.0)  # not earned yet
    assert r.nav.loc[DAYS[6]] == pytest.approx(1.5)  # earned the next day


def test_invalid_targets() -> None:
    p = prices(A=[10.0] * 8)
    with pytest.raises(ValueError):
        run_backtest(p, targets(T0, A=1.5))
    with pytest.raises(ValueError):
        run_backtest(p, targets(T0, A=-0.1))


def test_no_look_ahead_in_engine() -> None:
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2023-01-02", periods=300)
    p = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.02, (300, 4)), axis=0)), days, list("ABCD")
    )
    ends = days[[20, 80, 140, 200, 260]]
    w = pd.DataFrame(rng.dirichlet(np.ones(4), len(ends)), index=ends, columns=p.columns)
    before = run_backtest(p, w, cost=VolatilityCost())

    cut = days[150]
    p2, w2 = p.copy(), w.copy()
    p2.loc[days > cut] *= 3.0
    w2.loc[w2.index > cut] = np.roll(w2.loc[w2.index > cut].to_numpy(), 1, axis=1)
    after = run_backtest(p2, w2, cost=VolatilityCost())
    pd.testing.assert_series_equal(before.nav.loc[:cut], after.nav.loc[:cut])


# --- cost models --------------------------------------------------------------------


def test_volatility_cost_rate() -> None:
    """Daily returns alternate +2%/-2%, so sigma is about 2%: rate = 5 bps + 0.05 * 2%."""
    days = pd.bdate_range("2024-01-01", periods=80)
    path = 100 * np.cumprod(np.where(np.arange(80) % 2, 1.02, 1 / 1.02))
    model = VolatilityCost(base_bps=5, k=0.05, window=60)
    model.prepare(pd.DataFrame({"X": path}, index=days))
    sigma = pd.Series(path).pct_change().iloc[19:79].std()
    assert model.rates(79)[0] == pytest.approx(5e-4 + 0.05 * sigma)
    assert 0.0014 < model.rates(79)[0] < 0.0016  # about 15 bps


def test_flat_cost_rate() -> None:
    model = FlatCost(25)
    model.prepare(pd.DataFrame({"A": [1.0], "B": [1.0]}))
    assert model.rates(0).tolist() == [0.0025, 0.0025]


# --- helpers ------------------------------------------------------------------------


def test_buy_and_hold_and_basic_metrics() -> None:
    days = pd.to_datetime(["2020-01-01", "2020-07-01", "2021-01-01", "2022-01-01"])
    nav = buy_and_hold(pd.Series([50.0, 75.0, 60.0, 72.0], index=days, name="QQQ"), days[0])
    assert nav.tolist() == pytest.approx([1.0, 1.5, 1.2, 1.44])
    assert max_drawdown(nav) == pytest.approx(1.2 / 1.5 - 1)  # -20%
    assert cagr(nav) == pytest.approx(1.44 ** (365.25 / 731) - 1)  # about +20% a year
