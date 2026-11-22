"""Text report for a walk-forward run."""

from __future__ import annotations

import pandas as pd

from momentum.backtest.strategy import (
    StrategyConfig,
    buy_and_hold,
    run_equal_weight_universe,
)
from momentum.backtest.walkforward import WalkForwardResult
from momentum.data.dataset import BENCHMARK, Dataset
from momentum.metrics.overfitting import deflated_sharpe, probabilistic_sharpe
from momentum.metrics.performance import excess_monthly, monthly_returns, sharpe
from momentum.reports.tables import format_summary, summary_row

DEFAULT_LABEL = StrategyConfig().label


def _row(nav: pd.Series, rf: pd.Series, trades: pd.DataFrame | None = None) -> dict[str, float]:
    row = summary_row(nav, trades)
    row["Sharpe"] = sharpe(excess_monthly(nav, rf))
    return row


def comparison(wf: WalkForwardResult, ds: Dataset) -> pd.DataFrame:
    """Out-of-sample portfolio against the alternatives, over the same dates."""
    start = wf.oos.nav.index[0]
    navs = wf.grid.navs.loc[start:]
    full_sharpe = wf.grid.excess.apply(sharpe)
    hindsight = str(full_sharpe.idxmax())
    universe = run_equal_weight_universe(ds, wf.grid.configs[0])

    rows = {
        f"Walk-forward ({wf.window}), out-of-sample": _row(wf.oos.nav, ds.risk_free, wf.oos.trades),
        f"Hindsight best: {hindsight}": _row(navs[hindsight], ds.risk_free),
        f"Fixed default: {DEFAULT_LABEL}": _row(navs[DEFAULT_LABEL], ds.risk_free),
        "Equal-weight Nasdaq-100": _row(universe.nav.loc[start:], ds.risk_free),
        f"{BENCHMARK} (buy and hold)": _row(buy_and_hold(ds.benchmark, start), ds.risk_free),
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    return table[["final value", "CAGR", "volatility", "Sharpe", "max drawdown"]]


def yearly(wf: WalkForwardResult, ds: Dataset) -> pd.DataFrame:
    """Chosen setting, its training Sharpe, and what happened in the test year."""
    nav = wf.oos.nav
    qqq = buy_and_hold(ds.benchmark, nav.index[0])
    ends = nav.groupby(pd.DatetimeIndex(nav.index).year).last()
    q_ends = qqq.groupby(pd.DatetimeIndex(qqq.index).year).last()
    oos_ret = ends / ends.shift(1, fill_value=nav.iloc[0]) - 1
    q_ret = q_ends / q_ends.shift(1, fill_value=qqq.iloc[0]) - 1

    out = wf.windows[["train from", "train to", "chosen", "train Sharpe"]].copy()
    out["test return"] = oos_ret.reindex(out.index)
    out[f"{BENCHMARK} return"] = q_ret.reindex(out.index)
    return out


def format_report(wf: WalkForwardResult, ds: Dataset, robustness: WalkForwardResult | None) -> str:
    start, end = wf.oos.nav.index[0].date(), wf.oos.nav.index[-1].date()
    grid = wf.grid
    lines = [
        f"Walk-forward test: {len(grid.configs)} settings, {wf.window} training window,",
        "chosen each January by smoothed training Sharpe, traded for the next 12 months.",
        f"Out-of-sample period {start} to {end}, costs {grid.configs[0].cost}.",
        "",
        "Year by year (test return = the out-of-sample portfolio)",
        "",
    ]
    y = yearly(wf, ds)
    fmt = y.copy().astype(object)
    fmt["train Sharpe"] = y["train Sharpe"].map("{:.2f}".format)
    for c in ("test return", f"{BENCHMARK} return"):
        fmt[c] = y[c].map("{:+.1%}".format)
    lines += [fmt.to_string(), ""]

    table = comparison(wf, ds)
    shown = format_summary(table.drop(columns="Sharpe")).splitlines()
    sharpe_col = ["Sharpe"] + [f"{v:.2f}" for v in table["Sharpe"]]
    width = max(len(s) for s in shown)
    lines += ["Same dates, side by side", ""]
    lines += [f"{s.ljust(width)}  {sh:>6}" for s, sh in zip(shown, sharpe_col, strict=True)]

    def item(label: str, value: str, indent: int = 4) -> str:
        return f"{' ' * indent}{label:<56}{value:>6}"

    promised = float(wf.windows["train Sharpe"].mean())
    delivered = sharpe(excess_monthly(wf.oos.nav, ds.risk_free))
    lines += [
        "",
        "In-sample vs out-of-sample",
        item("Average training Sharpe of the chosen settings", f"{promised:.2f}", 2),
        item("Out-of-sample Sharpe actually earned", f"{delivered:.2f}", 2),
        item("Shortfall", f"{promised - delivered:.2f}", 2),
    ]

    n = len(grid.configs)
    dsr = deflated_sharpe(grid.excess)
    oos_psr = probabilistic_sharpe(excess_monthly(wf.oos.nav, ds.risk_free))
    dsr_active = deflated_sharpe(active_returns(grid.navs, ds))
    oos_active = active_returns(wf.oos.nav.to_frame(), ds).iloc[:, 0]
    lines += [
        "",
        "Overfitting check (Deflated Sharpe Ratio, Bailey & Lopez de Prado 2014)",
        "",
        "  vs T-bills: does it make money at all?",
        item(f"Best Sharpe of {n} settings, whole period", f"{dsr.best_sharpe:.2f}"),
        item(f"Best expected from {n} settings by luck alone", f"{dsr.hurdle:.2f}"),
        item("Probability the best setting's true Sharpe > 0", f"{dsr.psr:.0%}"),
        item("Deflated Sharpe: probability it beats the luck hurdle", f"{dsr.dsr:.0%}"),
        item("Out-of-sample: probability true Sharpe > 0", f"{oos_psr:.0%}"),
        "",
        f"  vs {BENCHMARK}: does it beat just buying the index? (information ratio)",
        item(f"Best information ratio of {n} settings", f"{dsr_active.best_sharpe:.2f}"),
        item(f"Best expected from {n} settings by luck alone", f"{dsr_active.hurdle:.2f}"),
        item("Deflated Sharpe: probability it beats the luck hurdle", f"{dsr_active.dsr:.0%}"),
        item("Out-of-sample information ratio", f"{sharpe(oos_active):.2f}"),
        "",
        "Sharpe of every setting over the whole period (rows: skip, lookback; cols: stocks held)",
        "",
        grid_table(wf).to_string(),
    ]

    if robustness is not None:
        r_sharpe = sharpe(excess_monthly(robustness.oos.nav, ds.risk_free))
        r_cagr = summary_row(robustness.oos.nav)["CAGR"]
        picks = ", ".join(f"{y}: {c}" for y, c in robustness.windows["chosen"].items())
        lines += [
            "",
            f"Robustness: {robustness.window} 5-year window",
            f"  Out-of-sample CAGR {r_cagr:.1%}, Sharpe {r_sharpe:.2f}",
            f"  Chosen: {picks}",
        ]
    return "\n".join(lines)


def active_returns(navs: pd.DataFrame, ds: Dataset) -> pd.DataFrame:
    """Monthly return of each column minus QQQ's over the same months."""
    qqq = monthly_returns(buy_and_hold(ds.benchmark, navs.index[0]))
    return pd.DataFrame({c: monthly_returns(navs[c].dropna()) - qqq for c in navs.columns}).dropna()


def grid_table(wf: WalkForwardResult) -> pd.DataFrame:
    full = wf.grid.excess.apply(sharpe)
    rows = []
    for c in wf.grid.configs:
        rows.append(
            {
                "signal": "risk-adj" if c.spec.risk_adjusted else "plain",
                "skip": c.spec.skip,
                "lookback": c.spec.lookback,
                "top": c.top_n,
                "Sharpe": round(full[c.label], 2),
            }
        )
    df = pd.DataFrame(rows)
    return df.pivot_table(
        index=["signal", "skip", "lookback"], columns="top", values="Sharpe"
    ).rename(columns=lambda n: f"top {n}")
