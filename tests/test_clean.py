import numpy as np
import pandas as pd
import pytest

from momentum.data.clean import clean_prices, fill_short_gaps

nan = np.nan


def series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def test_short_gap_is_filled_with_last_price() -> None:
    s = series([10, nan, nan, 13])
    assert fill_short_gaps(s, max_gap=5).tolist() == [10, 10, 10, 13]


def test_long_gap_is_left_empty_not_partly_filled() -> None:
    s = series([10, nan, nan, nan, 14])
    out = fill_short_gaps(s, max_gap=2)
    assert out.tolist()[0] == 10 and out.tolist()[-1] == 14
    assert out.iloc[1:4].isna().all()


def test_no_fill_before_listing_or_after_delisting() -> None:
    s = series([nan, nan, 10, 11, nan, nan])
    out = fill_short_gaps(s, max_gap=5)
    assert out.iloc[:2].isna().all()  # not listed yet
    assert out.iloc[4:].isna().all()  # delisted: exit at last price (11)


def test_clean_prices_end_to_end() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    raw = pd.DataFrame(
        {
            "GOOD": [100, 101, nan, 103, 104, 105],
            "BAD": [50, 0, -1, 52, np.inf, 54],  # 0, negative and inf are not prices
            "JUMP": [10, 10, 30, 30, 30, 30],  # +200% in a day
        },
        index=idx,
        dtype=float,
    )
    raw = pd.concat([raw, raw.iloc[[3]]])  # duplicated date

    prices, report = clean_prices(raw, max_gap=5)

    assert len(prices) == 6
    assert prices["GOOD"].tolist() == [100, 101, 101, 103, 104, 105]
    assert prices["BAD"].tolist() == [50, 50, 50, 52, 52, 54]
    assert report.loc["GOOD", "filled"] == 1
    assert report.loc["BAD", "filled"] == 3
    assert report.loc["JUMP", "jumps"] == 1
    assert report.loc["GOOD", "jumps"] == 0


def test_calendar_drops_non_trading_days() -> None:
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-06"])  # 6th is a Saturday
    raw = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    prices, _ = clean_prices(raw, calendar=pd.DatetimeIndex(idx[:2]))
    assert prices.index.tolist() == list(idx[:2])


def test_ticker_with_no_data() -> None:
    raw = pd.DataFrame(
        {"A": [1.0, 2.0], "GONE": [nan, nan]}, index=pd.bdate_range("2024-01-01", periods=2)
    )
    prices, report = clean_prices(raw)
    assert prices["GONE"].isna().all()
    assert pd.isna(report.loc["GONE", "first"])


@pytest.mark.parametrize("max_gap", [0, 1])
def test_max_gap_zero_or_one(max_gap: int) -> None:
    out = fill_short_gaps(series([1, nan, 3]), max_gap=max_gap)
    assert out.isna().sum() == (1 if max_gap == 0 else 0)
