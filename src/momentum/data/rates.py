"""Risk-free rate from the 13-week US T-bill yield (Yahoo symbol ``^IRX``).

``^IRX`` is quoted as an annualised percentage, e.g. 5.2 means 5.2% a year.
The per-period rate is the one that compounds back to the annual figure:

    r_period = (1 + y / 100) ** (1 / periods_per_year) - 1

``^IRX`` is technically a discount yield, slightly below the true investment
yield. The difference is a few basis points and is ignored.
"""

from __future__ import annotations

import pandas as pd

TBILL_SYMBOL = "^IRX"


def annual_pct_to_period(yield_pct: pd.Series, periods_per_year: int = 252) -> pd.Series:
    """Convert an annualised percentage yield into a per-period simple return."""
    return (1 + yield_pct / 100) ** (1 / periods_per_year) - 1
