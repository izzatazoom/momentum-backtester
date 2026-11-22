"""Turn scores into holdings: one share class per company, top N, equal weights."""

from __future__ import annotations

import pandas as pd


def one_class_per_company(eligible: pd.DataFrame, share_classes: pd.DataFrame) -> pd.DataFrame:
    """Keep only the highest-priority eligible share class of each company.

    ``share_classes`` has columns ``company``, ``ticker``, ``priority``. If the
    preferred class isn't eligible on a date, the next one may be used instead.
    """
    out = eligible.copy()
    for _, group in share_classes.groupby("company"):
        taken = pd.Series(False, index=out.index)
        for ticker in group.sort_values("priority")["ticker"]:
            if ticker in out.columns:
                out[ticker] = out[ticker] & ~taken
                taken |= out[ticker]
    return out


def rank_scores(scores: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """Rank eligible stocks on each date, 1 = highest score. NaN if not eligible.

    Ties go to the alphabetically first ticker, so results never depend on
    column order.
    """
    cols = sorted(scores.columns)
    masked = scores[cols].where(eligible[cols])
    return masked.rank(axis=1, ascending=False, method="first")


def select_top(
    scores: pd.DataFrame,
    eligible: pd.DataFrame,
    n: int,
    share_classes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Boolean frame: True for the ``n`` best-ranked eligible stocks on each date."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if share_classes is not None:
        eligible = one_class_per_company(eligible, share_classes)
    ranks = rank_scores(scores, eligible)
    return (ranks <= n).reindex(columns=scores.columns)


def equal_weights(selected: pd.DataFrame) -> pd.DataFrame:
    """1/k for each of the k selected stocks on a date, 0 elsewhere."""
    count = selected.sum(axis=1)
    return selected.astype(float).div(count.where(count > 0), axis=0).fillna(0.0)
