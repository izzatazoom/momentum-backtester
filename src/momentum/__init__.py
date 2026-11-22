"""Walk-forward backtesting engine for a US equity cross-sectional momentum strategy.

Subpackages
-----------
data       Download, clean and cache price data.
signals    Compute momentum scores and rank stocks.
portfolio  Turn rankings into target weights; track turnover and costs.
backtest   Monthly rebalance loop and walk-forward testing.
metrics    Performance statistics (CAGR, Sharpe, drawdown, alpha/beta, ...).
reports    Charts and result tables.
"""

__version__ = "0.1.0"
