"""Command-line entry point.

Run ``python -m momentum --help`` (or ``momentum --help`` once installed).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from momentum import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="momentum",
        description="Walk-forward backtester for a US equity momentum strategy.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    data = sub.add_parser("data", help="download/clean price data and print a quality report")
    data.add_argument("--start", default="2015-01-01", help="first date (default 2015-01-01)")
    data.add_argument("--end", default=None, help="last date (default today)")
    data.add_argument("--cache", type=Path, default=Path("data/cache"), help="cache folder")
    data.add_argument("--refresh", action="store_true", help="ignore the cache and re-download")

    sig = sub.add_parser("signal", help="show the stocks momentum would pick on a month-end")
    sig.add_argument("--date", default=None, help="YYYY-MM or YYYY-MM-DD (default: latest)")
    sig.add_argument("--top", type=int, default=20, help="number of stocks (default 20)")
    sig.add_argument("--lookback", type=int, default=12, help="months (default 12)")
    sig.add_argument("--skip", type=int, default=1, help="most recent months skipped (default 1)")
    sig.add_argument("--risk-adjusted", action="store_true", help="divide by volatility")

    bt = sub.add_parser("backtest", help="run one fixed-setting backtest against benchmarks")
    bt.add_argument("--top", type=int, default=20, help="number of stocks held (default 20)")
    bt.add_argument("--lookback", type=int, default=12, help="months (default 12)")
    bt.add_argument("--skip", type=int, default=1, help="most recent months skipped (default 1)")
    bt.add_argument("--risk-adjusted", action="store_true", help="divide score by volatility")
    bt.add_argument("--cost-bps", type=float, default=10.0, help="flat cost per side (default 10)")
    bt.add_argument("--vol-cost", action="store_true", help="volatility-scaled costs instead")
    bt.add_argument("--lag", type=int, default=1, help="days from signal to trade (default 1)")
    bt.add_argument("--cost-sweep", action="store_true", help="also show results at 0-50 bps")
    bt.add_argument("--save", type=Path, default=None, help="write daily values to this CSV")

    wf = sub.add_parser("walkforward", help="choose settings on past data, test on the next year")
    wf.add_argument("--window", choices=["expanding", "rolling"], default="expanding")
    wf.add_argument("--no-smooth", action="store_true", help="pick by raw, not smoothed, Sharpe")
    wf.add_argument("--cost-bps", type=float, default=10.0, help="flat cost per side (default 10)")
    wf.add_argument("--vol-cost", action="store_true", help="volatility-scaled costs instead")
    wf.add_argument("--save", type=Path, default=None, help="folder for CSV output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "data":
        from momentum.data.dataset import load_dataset, summarize

        ds = load_dataset(args.start, args.end, cache_dir=args.cache, refresh=args.refresh)
        print(summarize(ds))
        return 0

    if args.command == "signal":
        from momentum.data.dataset import load_dataset
        from momentum.data.universe import load_share_classes
        from momentum.reports.tables import format_picks, picks_table, resolve_month_end
        from momentum.signals import MomentumSpec, compute_signal

        ds = load_dataset()
        spec = MomentumSpec(args.lookback, args.skip, args.risk_adjusted)
        signal = compute_signal(ds.prices, ds.membership, spec)
        when = args.date or str(signal.scores.index[-1].date())
        date = resolve_month_end(signal, when)
        table = picks_table(signal, ds.prices, date, args.top, load_share_classes())
        print(format_picks(table, signal, date))
        return 0

    if args.command == "backtest":
        return _backtest(args)

    if args.command == "walkforward":
        return _walkforward(args)

    parser.print_help()
    return 0


def _backtest(args: argparse.Namespace) -> int:
    import pandas as pd

    from momentum.backtest.strategy import (
        StrategyConfig,
        buy_and_hold,
        run_equal_weight_universe,
        run_momentum,
    )
    from momentum.data.dataset import BENCHMARK, load_dataset
    from momentum.portfolio.costs import CostModel, FlatCost, VolatilityCost
    from momentum.reports.tables import format_summary, summary_row
    from momentum.signals import MomentumSpec

    ds = load_dataset()
    cost: CostModel = VolatilityCost() if args.vol_cost else FlatCost(args.cost_bps)
    spec = MomentumSpec(args.lookback, args.skip, args.risk_adjusted)
    config = StrategyConfig(spec, args.top, cost, args.lag)

    strat = run_momentum(ds, config)
    universe = run_equal_weight_universe(ds, config)
    start = strat.nav.index[0]
    qqq = buy_and_hold(ds.benchmark, start)
    universe_nav = universe.nav.loc[start:]

    table = pd.DataFrame.from_dict(
        {
            config.label: summary_row(strat.nav, strat.trades),
            "Equal-weight Nasdaq-100": summary_row(universe_nav, universe.trades.loc[start:]),
            f"{BENCHMARK} (buy and hold)": summary_row(qqq),
        },
        orient="index",
    )
    print(
        f"{config.label}, costs {cost}, trades {args.lag} day after signal\n"
        f"{start.date()} to {strat.nav.index[-1].date()}, "
        f"{len(strat.trades)} rebalances, average cash {strat.cash.mean():.1%}\n"
    )
    print(format_summary(table))

    if args.cost_sweep:
        rows = {}
        for bps in (0, 5, 10, 25, 50):
            sweep = StrategyConfig(spec, args.top, FlatCost(bps), args.lag)
            result = run_momentum(ds, sweep)
            rows[f"{bps} bps"] = summary_row(result.nav, result.trades)
        print("\nCost sensitivity (flat cost per side)\n")
        print(format_summary(pd.DataFrame.from_dict(rows, orient="index")))

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        out = pd.DataFrame({"momentum": strat.nav, "equal_weight": universe_nav, "qqq": qqq})
        out.to_csv(args.save)
        print(f"\nDaily values written to {args.save}")
    return 0


def _walkforward(args: argparse.Namespace) -> int:
    from momentum.backtest.walkforward import Grid, run_grid, walk_forward
    from momentum.data.dataset import load_dataset
    from momentum.portfolio.costs import CostModel, FlatCost, VolatilityCost
    from momentum.reports.walkforward import comparison, format_report, yearly

    ds = load_dataset()
    cost: CostModel = VolatilityCost() if args.vol_cost else FlatCost(args.cost_bps)
    grid = run_grid(ds, Grid().configs(cost))
    smooth = not args.no_smooth
    main_run = walk_forward(ds, grid, window=args.window, smooth=smooth)
    other = "rolling" if args.window == "expanding" else "expanding"
    check = walk_forward(ds, grid, window=other, smooth=smooth)
    print(format_report(main_run, ds, check))

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        main_run.oos.nav.to_csv(args.save / "walkforward_nav.csv")
        yearly(main_run, ds).to_csv(args.save / "walkforward_years.csv")
        comparison(main_run, ds).to_csv(args.save / "walkforward_comparison.csv")
        grid.navs.to_csv(args.save / "grid_navs.csv")
        print(f"\nCSV files written to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
