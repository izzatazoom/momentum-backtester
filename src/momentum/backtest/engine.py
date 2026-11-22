"""Daily backtest engine with periodic rebalancing.

Timeline for one rebalance:

    t        signal date (a month-end close). Target weights use data up to t.
    t + lag  execution day. Trades happen at this day's close.
    after    holdings drift with their prices until the next execution day.

Accounting, in dollars, each day ``d``:

1. Holdings are marked to the close of ``d``. A stock with no price on ``d`` keeps
   its last value. A stock past its last price ever is treated as delisted: it is
   sold at that last price, with no trading cost (a takeover pays out in cash), and
   the money waits in cash.
2. Cash earns the risk-free rate.
3. On an execution day the portfolio trades to its targets:

       V            = cash + sum(holdings)                portfolio value
       target_i     = w_i * (V - cost)                    after paying for the trade
       cost         = sum(rate_i * |target_i - holding_i|)
       turnover     = 1/2 * sum(|target_i - holding_i|) / V

   ``cost`` depends on the targets and the targets depend on ``cost``, so the pair
   is solved by fixed-point iteration (it converges in a few steps because rates
   are tiny). Stocks with no price on the execution day can't be traded: a held
   one is kept as it is, a new one is left in cash. If kept positions leave less
   room than the targets need, the buys are scaled down, so the portfolio never
   borrows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from momentum.portfolio.costs import CostModel, FlatCost


@dataclass(frozen=True)
class BacktestResult:
    nav: pd.Series
    """Portfolio value at each close, starting at ``initial`` minus the first trade's cost."""
    weights: pd.DataFrame
    """End-of-day stock weights (value / NAV)."""
    cash: pd.Series
    """End-of-day cash weight."""
    trades: pd.DataFrame
    """One row per rebalance: signal date, turnover, cost (share of NAV), holdings."""

    @property
    def returns(self) -> pd.Series:
        return self.nav.pct_change().dropna()


def _execution_days(index: pd.DatetimeIndex, signal_dates: pd.Index, lag: int) -> dict[int, int]:
    """Map row of ``index`` where trades happen -> row of the target frame."""
    out = {}
    positions = index.get_indexer(pd.DatetimeIndex(signal_dates))
    if (positions < 0).any():
        raise ValueError("every signal date must be a trading day in the price data")
    for row, p in enumerate(positions):
        pos = int(p) + lag
        if pos < len(index):
            out[pos] = row
    return out


def run_backtest(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    risk_free: pd.Series | None = None,
    cost: CostModel | None = None,
    *,
    lag: int = 1,
    initial: float = 1.0,
) -> BacktestResult:
    """Simulate trading to ``targets`` and holding between rebalances.

    Args:
        prices: daily closes, trading days x tickers (NaN where not trading).
        targets: target weights indexed by signal date. Each date must be in
            ``prices.index``. Rows sum to at most 1, and the rest is cash.
        risk_free: daily risk-free return. The return credited for day ``d`` is
            the rate known on ``d - 1``.
        cost: transaction cost model (default 10 bps flat).
        lag: trading days between signal and execution (1 = next day's close).
        initial: starting capital.
    """
    if lag < 1:
        raise ValueError("lag must be at least 1: you can't trade at the close you just saw")
    targets = targets.reindex(columns=prices.columns, fill_value=0.0).fillna(0.0)
    if (targets < 0).to_numpy().any() or (targets.sum(axis=1) > 1 + 1e-9).any():
        raise ValueError("target weights must be non-negative and sum to at most 1")

    index = pd.DatetimeIndex(prices.index)
    cost = cost or FlatCost()
    cost.prepare(prices)
    schedule = _execution_days(index, targets.index, lag)
    if not schedule:
        raise ValueError("no execution day falls inside the price data")
    start = min(schedule)

    px = prices.to_numpy(dtype=float)
    valid = np.isfinite(px)
    last_valid = np.where(valid.any(axis=0), len(px) - 1 - np.argmax(valid[::-1], axis=0), -1)
    rf = np.zeros(len(index))
    if risk_free is not None:
        rf = risk_free.reindex(index).shift(1).fillna(0.0).to_numpy()
    w_target = targets.to_numpy(dtype=float)

    n = px.shape[1]
    hold = np.zeros(n)
    mark = np.full(n, np.nan)
    cash = initial
    days = range(start, len(index))
    nav = np.empty(len(days))
    weights = np.empty((len(days), n))
    cash_w = np.empty(len(days))
    log = []

    for k, d in enumerate(days):
        if d > start:
            held = hold != 0
            move = held & valid[d]
            hold[move] *= px[d, move] / mark[move]
            mark[move] = px[d, move]
            gone = held & (d > last_valid)
            cash += hold[gone].sum()
            hold[gone] = 0.0
            cash *= 1 + rf[d]

        if d in schedule:
            row = schedule[d]
            value = cash + hold.sum()
            tradable = valid[d]
            rates = cost.rates(d)
            frozen = float(hold[~tradable].sum())  # held but can't trade today
            paid = 0.0
            for _ in range(50):
                budget = max(value - paid - frozen, 0.0)
                want = w_target[row] * (value - paid) * tradable
                scale = min(1.0, budget / want.sum()) if want.sum() > 0 else 0.0
                target = np.where(tradable, want * scale, hold)
                new_paid = float(np.sum(rates * np.abs(target - hold)))
                if abs(new_paid - paid) <= 1e-15 * max(value, 1.0):
                    break
                paid = new_paid
            log.append(
                {
                    "date": index[d],
                    "signal_date": targets.index[row],
                    "turnover": 0.5 * np.abs(target - hold).sum() / value,
                    "cost": paid / value,
                    "holdings": int((target > 0).sum()),
                }
            )
            cash = value - paid - target.sum()
            hold = target
            mark[tradable] = px[d, tradable]

        total = cash + hold.sum()
        nav[k] = total
        weights[k] = hold / total
        cash_w[k] = cash / total

    dates = index[start:]
    return BacktestResult(
        nav=pd.Series(nav, index=dates, name="nav"),
        weights=pd.DataFrame(weights, index=dates, columns=prices.columns),
        cash=pd.Series(cash_w, index=dates, name="cash"),
        trades=pd.DataFrame(log).set_index("date"),
    )
