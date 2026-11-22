"""Transaction cost models.

Costs are a fraction of each dollar traded, charged on buys and sells alike:

    cost = sum over stocks of  rate_i * |dollars traded in stock i|

``FlatCost`` uses one rate for every stock (10 bps = 0.10% by default).

``VolatilityCost`` charges jumpier stocks more, since their bid-ask spreads tend to
be wider:

    rate_i = base + k * sigma_i

where ``sigma_i`` is the stock's daily return volatility over the previous
``window`` days. With the defaults (5 bps + 0.05 * sigma), a stock moving 2% a day
costs 5 + 10 = 15 bps. This is a rough proxy. Without volume or quote data the
scaling can't be calibrated, so the flat model is the default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

BPS = 1e-4


class CostModel(Protocol):
    def prepare(self, prices: pd.DataFrame) -> None:
        """Precompute anything that depends on the price history."""

    def rates(self, day: int) -> np.ndarray:
        """Cost per dollar traded for each column, on row ``day`` of the prices."""
        ...


@dataclass
class FlatCost:
    bps: float = 10.0

    def prepare(self, prices: pd.DataFrame) -> None:
        self._n = prices.shape[1]

    def rates(self, day: int) -> np.ndarray:
        return np.full(self._n, self.bps * BPS)

    def __str__(self) -> str:
        return f"{self.bps:g} bps flat"


@dataclass
class VolatilityCost:
    base_bps: float = 5.0
    k: float = 0.05
    window: int = 60

    def prepare(self, prices: pd.DataFrame) -> None:
        daily = prices.pct_change(fill_method=None)
        # shift(1): the rate charged on day d only uses returns up to d - 1
        sigma = daily.rolling(self.window, min_periods=20).std().shift(1)
        median = sigma.median(axis=1)
        self._sigma = sigma.apply(lambda col: col.fillna(median)).fillna(0.0).to_numpy()

    def rates(self, day: int) -> np.ndarray:
        result: np.ndarray = self.base_bps * BPS + self.k * self._sigma[day]
        return result

    def __str__(self) -> str:
        return f"{self.base_bps:g} bps + {self.k:g} x daily vol"
