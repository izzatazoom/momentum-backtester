"""Point-in-time membership.

Hand-made history (today's members are A, B, C):

    2020-03-02  C added, D removed
    2020-06-01  B added

So before March the index was {A, D}; from March {A, C}; from June {A, B, C}.
"""

import pandas as pd
import pytest

from momentum.data.universe import (
    MembershipError,
    apply_corrections,
    build_membership,
    load_changes,
    load_current,
    members_on,
    membership_mask,
)


@pytest.fixture
def changes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-02", "2020-06-01"]),
            "added": ["C", "B"],
            "removed": ["D", ""],
        }
    )


@pytest.fixture
def intervals(changes: pd.DataFrame) -> pd.DataFrame:
    return build_membership(["A", "B", "C"], changes, start="2020-01-01")


def test_intervals(intervals: pd.DataFrame) -> None:
    got = {r.ticker: (r.start, r.end) for r in intervals.itertuples()}
    nat = pd.NaT
    assert got["A"] == (nat, nat)
    assert got["B"] == (pd.Timestamp("2020-06-01"), nat)
    assert got["C"] == (pd.Timestamp("2020-03-02"), nat)
    assert got["D"] == (nat, pd.Timestamp("2020-03-02"))


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        ("2020-02-28", ["A", "D"]),
        ("2020-03-02", ["A", "C"]),  # change takes effect on its own date
        ("2020-05-29", ["A", "C"]),
        ("2020-06-01", ["A", "B", "C"]),
    ],
)
def test_members_on(intervals: pd.DataFrame, day: str, expected: list[str]) -> None:
    assert members_on(intervals, day) == expected


def test_mask_matches_members_on(intervals: pd.DataFrame) -> None:
    dates = pd.bdate_range("2020-02-25", "2020-06-05")
    mask = membership_mask(intervals, dates)
    for d in dates:
        assert sorted(mask.columns[mask.loc[d]]) == members_on(intervals, d)


def test_changes_before_start_are_ignored(changes: pd.DataFrame) -> None:
    got = build_membership(["A", "B", "C"], changes, start="2020-04-01")
    assert members_on(got, "2020-04-01") == ["A", "C"]
    assert "D" not in set(got["ticker"])


def test_ticker_that_leaves_and_rejoins_gets_two_intervals() -> None:
    changes = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-02-03", "2020-04-01"]),
            "added": ["", "X"],
            "removed": ["X", ""],
        }
    )
    got = build_membership(["X"], changes, start="2020-01-01")
    assert len(got[got["ticker"] == "X"]) == 2
    assert members_on(got, "2020-03-02") == []
    assert members_on(got, "2020-04-01") == ["X"]


def test_inconsistent_log_raises() -> None:
    bad = pd.DataFrame({"date": pd.to_datetime(["2020-03-02"]), "added": ["Z"], "removed": [""]})
    with pytest.raises(MembershipError, match="Z added but not a member"):
        build_membership(["A"], bad, start="2020-01-01")


def test_corrections() -> None:
    log = pd.DataFrame(
        {
            "date": pd.to_datetime(["2015-11-11", "2016-02-01"]),
            "added": ["PYPL", "AVGO"],
            "removed": ["AVGO", ""],
        }
    )
    fixes = pd.DataFrame(
        {
            "date": ["2015-11-11", "2016-02-01"],
            "action": ["rename_removed", "drop"],
            "ticker": ["AVGO", "AVGO"],
            "new_ticker": ["BRCM", None],
        }
    )
    out = apply_corrections(log, fixes)
    assert out["removed"].tolist() == ["BRCM"]


def test_real_nasdaq100_history_is_consistent() -> None:
    """The shipped Wikipedia data replays back to 2015 with ~100 members every day."""
    intervals = build_membership(load_current(), load_changes(), start="2015-01-01")
    mask = membership_mask(intervals, pd.bdate_range("2015-01-01", "2026-11-20"))
    per_day = mask.sum(axis=1)
    assert per_day.between(100, 110).all()
    assert "BRCM" in members_on(intervals, "2015-06-01")
    assert "AVGO" in members_on(intervals, "2015-12-01")
