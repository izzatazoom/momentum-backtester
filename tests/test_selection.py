import numpy as np
import pandas as pd
import pytest

from momentum.portfolio.selection import (
    equal_weights,
    one_class_per_company,
    rank_scores,
    select_top,
)

D1, D2 = pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")


def frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=[D1, D2][: len(rows)])


def test_top_n_by_score() -> None:
    scores = frame([{"A": 0.5, "B": 0.2, "C": -0.25}])
    picked = select_top(scores, scores.notna(), n=2)
    assert picked.loc[D1].to_dict() == {"A": True, "B": True, "C": False}


def test_ineligible_stocks_are_never_picked() -> None:
    scores = frame([{"A": 0.5, "B": 0.2, "C": -0.25}])
    eligible = scores.notna()
    eligible["A"] = False
    picked = select_top(scores, eligible, n=2)
    assert picked.loc[D1].to_dict() == {"A": False, "B": True, "C": True}


def test_ties_go_to_first_ticker_alphabetically() -> None:
    scores = frame([{"Z": 0.3, "M": 0.3, "A": 0.3}])
    ranks = rank_scores(scores, scores.notna())
    assert ranks.loc[D1].to_dict() == {"A": 1, "M": 2, "Z": 3}


def test_n_larger_than_universe_takes_all_eligible() -> None:
    scores = frame([{"A": 0.1, "B": np.nan}])
    picked = select_top(scores, scores.notna(), n=10)
    assert picked.loc[D1].to_dict() == {"A": True, "B": False}


def test_n_must_be_positive() -> None:
    scores = frame([{"A": 0.1}])
    with pytest.raises(ValueError):
        select_top(scores, scores.notna(), n=0)


SHARE_CLASSES = pd.DataFrame(
    {"company": ["Alphabet", "Alphabet"], "ticker": ["GOOGL", "GOOG"], "priority": [1, 2]}
)


def test_one_share_class_per_company() -> None:
    # GOOG scores a hair higher but GOOGL is the preferred class.
    scores = frame([{"GOOGL": 0.40, "GOOG": 0.41, "AAPL": 0.30}])
    picked = select_top(scores, scores.notna(), n=2, share_classes=SHARE_CLASSES)
    assert picked.loc[D1].to_dict() == {"GOOGL": True, "GOOG": False, "AAPL": True}


def test_falls_back_to_other_class_when_preferred_is_ineligible() -> None:
    eligible = frame([{"GOOGL": 0.0, "GOOG": 1.0}, {"GOOGL": 1.0, "GOOG": 1.0}]).astype(bool)
    out = one_class_per_company(eligible, SHARE_CLASSES)
    assert out.loc[D1].to_dict() == {"GOOGL": False, "GOOG": True}
    assert out.loc[D2].to_dict() == {"GOOGL": True, "GOOG": False}


def test_equal_weights() -> None:
    picked = frame([{"A": 1, "B": 1, "C": 0}, {"A": 0, "B": 0, "C": 0}]).astype(bool)
    w = equal_weights(picked)
    assert w.loc[D1].to_dict() == {"A": 0.5, "B": 0.5, "C": 0.0}
    assert w.loc[D2].sum() == 0.0  # nothing eligible -> all cash
