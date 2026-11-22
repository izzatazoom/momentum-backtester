"""Momentum scores and eligibility.

The hand-checkable dataset: three stocks, one price per month (each month's trading
days all share the same close). On the 2024-01 month-end, 12-1 momentum compares
December 2023 (m11) with January 2023 (m0):

    A: 150 / 100 - 1 = +50%     (A's jump to 300 in January is the skipped month)
    B: 120 / 100 - 1 = +20%
    C:  60 /  80 - 1 = -25%
"""

import numpy as np
import pandas as pd
import pytest

from momentum.signals import MomentumSpec, compute_signal, month_ends
from momentum.signals.momentum import window_stats

MONTHLY = {
    "A": [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 148, 150, 300, 310],
    "B": [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 119, 120, 120, 121],
    "C": [80, 78, 76, 74, 72, 70, 68, 66, 64, 63, 62, 60, 61, 62],
}


def step_prices(monthly: dict[str, list[float]], start: str = "2023-01-01") -> pd.DataFrame:
    """Daily prices that stay at each month's value for the whole month."""
    n = len(next(iter(monthly.values())))
    days = pd.bdate_range(start, periods=n * 23)
    days = days[days.to_period("M") < pd.Period(start, "M") + n]
    month_no = (days.to_period("M") - pd.Period(start, "M")).map(lambda x: x.n)
    return pd.DataFrame({t: np.asarray(v, float)[month_no] for t, v in monthly.items()}, index=days)


@pytest.fixture
def prices() -> pd.DataFrame:
    return step_prices(MONTHLY)


@pytest.fixture
def members(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(True, index=prices.index, columns=prices.columns)


JAN_2024 = pd.Timestamp("2024-01-31")
FEB_2024 = pd.Timestamp("2024-02-29")


def test_hand_checked_12_1_scores(prices: pd.DataFrame, members: pd.DataFrame) -> None:
    s = compute_signal(prices, members).scores
    assert s.loc[JAN_2024].to_dict() == pytest.approx({"A": 0.5, "B": 0.2, "C": -0.25})
    # one month later the window moves on: m12 / m1
    assert s.loc[FEB_2024].to_dict() == pytest.approx(
        {"A": 300 / 105 - 1, "B": 120 / 102 - 1, "C": 61 / 78 - 1}
    )


def test_first_eleven_month_ends_have_no_score(prices: pd.DataFrame, members: pd.DataFrame) -> None:
    s = compute_signal(prices, members).scores
    assert s.loc[:"2023-12-31"].isna().all().all()


def test_skip_zero_includes_latest_month(prices: pd.DataFrame, members: pd.DataFrame) -> None:
    s = compute_signal(prices, members, MomentumSpec(lookback=12, skip=0)).scores
    assert s.loc[JAN_2024, "A"] == pytest.approx(300 / 100 - 1)


def test_shorter_lookback(prices: pd.DataFrame, members: pd.DataFrame) -> None:
    s = compute_signal(prices, members, MomentumSpec(lookback=6, skip=1)).scores
    assert s.loc[JAN_2024, "B"] == pytest.approx(120 / 112 - 1)  # m11 / m6


def test_spec_validation() -> None:
    with pytest.raises(ValueError):
        MomentumSpec(lookback=12, skip=12)


def test_month_ends_use_last_trading_day() -> None:
    days = pd.bdate_range("2024-03-01", "2024-04-30").drop(pd.Timestamp("2024-03-29"))  # holiday
    assert list(month_ends(days)) == [pd.Timestamp("2024-03-28"), pd.Timestamp("2024-04-30")]


def test_unfinished_last_month_is_not_a_month_end() -> None:
    days = pd.bdate_range("2024-03-01", "2024-04-19")  # data stops mid-April
    assert list(month_ends(days)) == [pd.Timestamp("2024-03-29")]
    assert month_ends(days, complete_only=False)[-1] == pd.Timestamp("2024-04-19")


# --- look-ahead ---------------------------------------------------------------------


def test_no_look_ahead() -> None:
    """Scrambling every price and membership after date t changes nothing at or before t."""
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2020-01-01", "2022-12-31")
    tickers = list("ABCDEFGH")
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.02, (len(days), len(tickers))), axis=0)),
        index=days,
        columns=tickers,
    )
    members = pd.DataFrame(rng.random(prices.shape) > 0.2, index=days, columns=tickers)
    spec = MomentumSpec(risk_adjusted=True)
    before = compute_signal(prices, members, spec)

    cutoff = pd.Timestamp("2021-09-30")
    future = days > cutoff
    prices2, members2 = prices.copy(), members.copy()
    prices2.loc[future] *= rng.uniform(0.2, 5.0, (future.sum(), len(tickers)))
    members2.loc[future] = ~members2.loc[future]
    after = compute_signal(prices2, members2, spec)

    upto = before.scores.index <= cutoff
    pd.testing.assert_frame_equal(before.scores[upto], after.scores[upto])
    pd.testing.assert_frame_equal(before.eligible[upto], after.eligible[upto])
    assert not before.scores[~upto].equals(after.scores[~upto])  # the scramble did bite


# --- eligibility --------------------------------------------------------------------


def test_must_be_member_on_signal_date(prices: pd.DataFrame, members: pd.DataFrame) -> None:
    members.loc["2024-01-15":, "B"] = False
    e = compute_signal(prices, members).eligible.loc[JAN_2024]
    assert e.to_dict() == {"A": True, "B": False, "C": True}


def test_needs_price_at_start_of_window(prices: pd.DataFrame, members: pd.DataFrame) -> None:
    prices.loc[:"2023-01-31", "C"] = np.nan  # listed in February 2023
    sig = compute_signal(prices, members)
    assert np.isnan(sig.scores.loc[JAN_2024, "C"])
    assert not sig.eligible.loc[JAN_2024, "C"]


def test_needs_price_on_signal_date(prices: pd.DataFrame, members: pd.DataFrame) -> None:
    prices.loc[JAN_2024, "A"] = np.nan
    assert not compute_signal(prices, members).eligible.loc[JAN_2024, "A"]


def test_needs_enough_prices_inside_window(prices: pd.DataFrame, members: pd.DataFrame) -> None:
    window = prices.loc["2023-02-01":"2023-12-29"].index  # 230 trading days
    prices.loc[window[1:45], "A"] = np.nan  # ~19% missing -> out
    prices.loc[window[1:10], "B"] = np.nan  # ~4% missing  -> still in
    sig = compute_signal(prices, members)
    assert sig.coverage.loc[JAN_2024, "A"] < 0.9 < sig.coverage.loc[JAN_2024, "B"]
    assert sig.eligible.loc[JAN_2024].to_dict() == {"A": False, "B": True, "C": True}


# --- risk-adjusted ------------------------------------------------------------------


def test_risk_adjusted_prefers_the_smoother_path() -> None:
    """Same start and end prices, but Y wobbles: same return, lower risk-adjusted score."""
    days = pd.bdate_range("2023-01-01", "2024-01-31")
    trend = 100 * np.exp(np.linspace(0, np.log(1.3), len(days)))
    wobble = np.where(np.arange(len(days)) % 2 == 0, 1.03, 1 / 1.03)
    wobble[days.isin(month_ends(days))] = 1.0  # identical month-end prices
    prices = pd.DataFrame({"X": trend, "Y": trend * wobble}, index=days)
    members = pd.DataFrame(True, index=days, columns=prices.columns)

    raw = compute_signal(prices, members).scores.loc[JAN_2024]
    adj = compute_signal(prices, members, MomentumSpec(risk_adjusted=True)).scores.loc[JAN_2024]
    assert raw["X"] == pytest.approx(raw["Y"])
    assert adj["X"] > adj["Y"]


def test_volatility_matches_direct_calculation() -> None:
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2023-01-01", "2024-01-31")
    prices = pd.DataFrame({"Z": 50 * np.exp(np.cumsum(rng.normal(0, 0.015, len(days))))}, days)
    ends = month_ends(days)
    _, vol = window_stats(prices, ends, MomentumSpec())

    start, stop = ends[0], ends[-2]  # 2023-01-31 to 2023-12-29
    window = prices["Z"].pct_change().loc[start:stop].iloc[1:]  # returns after the start date
    expected = window.std(ddof=1) * np.sqrt(252)
    assert vol.loc[ends[-1], "Z"] == pytest.approx(expected)


def test_picks_table_shows_the_prices_behind_each_score(
    prices: pd.DataFrame, members: pd.DataFrame
) -> None:
    from momentum.reports.tables import picks_table, resolve_month_end

    sig = compute_signal(prices, members)
    date = resolve_month_end(sig, "2024-01")
    assert date == JAN_2024
    table = picks_table(sig, prices, date, n=2)
    assert list(table.index) == ["A", "B"]
    assert table.loc["A"].to_dict() == pytest.approx(
        {"rank": 1, "price 2023-01-31": 100, "price 2023-12-29": 150, "return": 0.5}
    )
