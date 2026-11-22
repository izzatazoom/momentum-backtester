"""Command-line entry point.

Run ``python -m momentum --help`` (or ``momentum --help`` once installed).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from momentum import __version__


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="momentum",
        description="Walk-forward backtester for a US equity momentum strategy.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the requested command. Returns a process exit code."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
