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

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
