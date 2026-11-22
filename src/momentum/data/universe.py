"""Point-in-time Nasdaq-100 membership.

Wikipedia gives today's constituents plus a dated log of additions and removals.
Starting from today's list and undoing each change in reverse date order recovers
the membership on any past date, so a backtest only ever picks stocks that were
actually in the index at the time.

A change dated ``D`` takes effect at the start of trading on ``D``. A stock is a
member on date ``d`` when ``start <= d < end`` for one of its intervals.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

UNIVERSE_DIR = Path(__file__).resolve().parents[3] / "data" / "universe"


class MembershipError(ValueError):
    """The change log contradicts the reconstructed membership."""


def load_current(path: Path = UNIVERSE_DIR / "nasdaq100_current.csv") -> list[str]:
    return pd.read_csv(path)["ticker"].astype(str).str.strip().tolist()


def load_yahoo_symbols(path: Path = UNIVERSE_DIR / "yahoo_symbols.csv") -> dict[str, str]:
    """Index ticker -> Yahoo symbol, for renamed companies."""
    if not path.exists():
        return {}
    table = pd.read_csv(path, dtype=str)
    return dict(zip(table["ticker"], table["yahoo"], strict=True))


def load_share_classes(path: Path = UNIVERSE_DIR / "share_classes.csv") -> pd.DataFrame:
    """Columns ``company``, ``ticker``, ``priority`` (1 = preferred class)."""
    return pd.read_csv(path, dtype={"company": str, "ticker": str, "priority": int})


def load_changes(
    path: Path = UNIVERSE_DIR / "nasdaq100_changes.csv",
    corrections: Path | None = UNIVERSE_DIR / "corrections.csv",
) -> pd.DataFrame:
    """Load the change log as rows of (date, added, removed), with corrections applied.

    Blank added/removed cells become empty strings.
    """
    changes = pd.read_csv(path, dtype=str, keep_default_na=False)
    changes["date"] = pd.to_datetime(changes["date"])
    if corrections is not None and corrections.exists():
        changes = apply_corrections(changes, pd.read_csv(corrections, dtype=str))
    return changes[["date", "added", "removed"]].reset_index(drop=True)


def apply_corrections(changes: pd.DataFrame, corrections: pd.DataFrame) -> pd.DataFrame:
    """Apply manual fixes to the change log.

    Supported actions:
        ``drop``            remove the row on ``date`` that adds or removes ``ticker``
        ``rename_removed``  on ``date``, replace removed ``ticker`` with ``new_ticker``
    """
    out = changes.copy()
    for fix in corrections.to_dict("records"):
        date, action, ticker = pd.Timestamp(str(fix["date"])), fix["action"], fix["ticker"]
        on_date = out["date"] == date
        if action == "drop":
            hit = on_date & ((out["added"] == ticker) | (out["removed"] == ticker))
            out = out[~hit]
        elif action == "rename_removed":
            hit = on_date & (out["removed"] == ticker)
            out["removed"] = out["removed"].mask(hit, str(fix["new_ticker"]))
        else:
            raise ValueError(f"unknown correction action: {action!r}")
        if not hit.any():
            raise ValueError(f"correction matched no rows: {fix}")
    return out


def build_membership(
    current: Iterable[str], changes: pd.DataFrame, start: str | pd.Timestamp
) -> pd.DataFrame:
    """Reconstruct membership intervals from ``start`` to today.

    Returns a DataFrame with columns ``ticker``, ``start``, ``end``. ``start`` is NaT
    when the stock was already a member on ``start``. ``end`` is NaT when it still is.
    A ticker can have several intervals if it left and later rejoined.

    Raises:
        MembershipError: if a change contradicts the reconstructed state.
    """
    later = changes[changes["date"] > pd.Timestamp(start)].sort_values("date", ascending=False)
    members = set(current)
    open_end: dict[str, pd.Timestamp | None] = dict.fromkeys(members)
    rows: list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]] = []

    dates: list[pd.Timestamp] = list(later["date"])
    added: list[str] = list(later["added"])
    removed: list[str] = list(later["removed"])
    for date, add, rem in zip(dates, added, removed, strict=True):
        if add:  # before `date` it was not a member
            if add not in members:
                raise MembershipError(f"{date.date()}: {add} added but not a member")
            members.remove(add)
            rows.append((add, date, open_end.pop(add)))
        if rem:  # before `date` it was a member
            if rem in members:
                raise MembershipError(f"{date.date()}: {rem} removed but still a member")
            members.add(rem)
            open_end[rem] = date

    rows.extend((t, None, open_end[t]) for t in members)
    out = pd.DataFrame(rows, columns=["ticker", "start", "end"])
    out["start"] = pd.to_datetime(out["start"])
    out["end"] = pd.to_datetime(out["end"])
    return out.sort_values(["ticker", "start"], na_position="first").reset_index(drop=True)


def membership_mask(intervals: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean frame (dates x tickers): True where the ticker was an index member."""
    tickers = sorted(intervals["ticker"].unique())
    columns = {t: np.zeros(len(dates), dtype=bool) for t in tickers}
    values = dates.to_numpy()
    starts = intervals["start"].to_numpy()
    ends = intervals["end"].to_numpy()
    for ticker, lo, hi in zip(intervals["ticker"], starts, ends, strict=True):
        inside = np.ones(len(dates), dtype=bool)
        if not np.isnat(lo):
            inside &= values >= lo
        if not np.isnat(hi):
            inside &= values < hi
        columns[str(ticker)] |= inside
    return pd.DataFrame(columns, index=dates)


def members_on(intervals: pd.DataFrame, date: str | pd.Timestamp) -> list[str]:
    """Sorted list of tickers that were index members on ``date``."""
    d = pd.Timestamp(date)
    started = intervals["start"].isna() | (intervals["start"] <= d)
    not_ended = intervals["end"].isna() | (intervals["end"] > d)
    return sorted(intervals.loc[started & not_ended, "ticker"].unique())
