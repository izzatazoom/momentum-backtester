"""Build the Nasdaq-100 universe files from Wikipedia.

Parses two Wikipedia pages (raw wikitext) into CSVs under data/universe/:

    nasdaq100_current.csv   ticker, name
    nasdaq100_changes.csv   date, added, added_name, removed, removed_name, reason

Usage:
    python scripts/build_universe.py                # download pages and rebuild
    python scripts/build_universe.py --current a.wiki --history b.wiki

Only the standard library is used so this can run before the package is installed.
"""

from __future__ import annotations

import argparse
import csv
import re
import urllib.request
from datetime import datetime
from pathlib import Path

CURRENT_PAGE = "List_of_NASDAQ-100_companies"
HISTORY_PAGE = "Historical_components_of_the_Nasdaq-100"
RAW_URL = "https://en.wikipedia.org/w/index.php?title={page}&action=raw"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "universe"


def fetch(page: str) -> str:
    req = urllib.request.Request(
        RAW_URL.format(page=page), headers={"User-Agent": "momentum-backtester/0.1"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return str(resp.read().decode("utf-8"))


def strip_markup(text: str) -> str:
    """Turn wiki markup into plain text: drop refs/templates, unwrap [[links]]."""
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    return " ".join(text.split())


def table_rows(wikitext: str, table_id: str) -> list[list[str]]:
    """Return the data rows of the wikitable with the given id, as lists of raw cells."""
    start = wikitext.index(f'id="{table_id}"')
    end = wikitext.index("\n|}", start)
    body = wikitext[start:end]
    rows: list[list[str]] = []
    for chunk in body.split("\n|-")[1:]:
        cells: list[str] = []
        for line in chunk.split("\n"):
            if line.startswith("!") or line.strip() in ("", "-"):
                continue
            if line.startswith("|"):
                cells.extend(part.strip() for part in line[1:].split("||"))
            elif cells:  # continuation of the previous cell
                cells[-1] += " " + line.strip()
        if cells:
            rows.append(cells)
    return rows


def parse_current(wikitext: str) -> list[tuple[str, str]]:
    return [
        (strip_markup(cells[0]), strip_markup(cells[1]))
        for cells in table_rows(wikitext, "constituents")
        if len(cells) >= 2
    ]


def parse_history(wikitext: str) -> list[dict[str, str]]:
    out = []
    for cells in table_rows(wikitext, "changes"):
        if len(cells) < 5:
            raise ValueError(f"unexpected row: {cells}")
        date = datetime.strptime(strip_markup(cells[0]), "%B %d, %Y").date()
        out.append(
            {
                "date": date.isoformat(),
                "added": strip_markup(cells[1]),
                "added_name": strip_markup(cells[2]),
                "removed": strip_markup(cells[3]),
                "removed_name": strip_markup(cells[4]),
                "reason": strip_markup(" ".join(cells[5:])),
            }
        )
    return out


def write_csv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--current", type=Path, help="local wikitext of the current list")
    parser.add_argument("--history", type=Path, help="local wikitext of the change history")
    args = parser.parse_args()

    current_text = args.current.read_text("utf-8") if args.current else fetch(CURRENT_PAGE)
    history_text = args.history.read_text("utf-8") if args.history else fetch(HISTORY_PAGE)

    current = parse_current(current_text)
    changes = parse_history(history_text)

    write_csv(
        OUT_DIR / "nasdaq100_current.csv",
        ["ticker", "name"],
        [{"ticker": t, "name": n} for t, n in current],
    )
    write_csv(
        OUT_DIR / "nasdaq100_changes.csv",
        ["date", "added", "added_name", "removed", "removed_name", "reason"],
        sorted(changes, key=lambda r: r["date"]),
    )
    print(f"{len(current)} current constituents, {len(changes)} changes -> {OUT_DIR}")


if __name__ == "__main__":
    main()
