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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "data":
        from momentum.data.dataset import load_dataset, summarize

        ds = load_dataset(args.start, args.end, cache_dir=args.cache, refresh=args.refresh)
        print(summarize(ds))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
