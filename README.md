# Momentum Backtester

A walk-forward backtesting engine for a US equity cross-sectional momentum strategy (12-1 month),
with monthly rebalancing, a one-day execution lag, proportional transaction costs, and separate
in-sample / out-of-sample reporting against SPY.

Work in progress.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"
pytest
```

## Project layout

```
src/momentum/
  data/        download, clean and cache prices
  signals/     12-1 momentum score and ranking
  portfolio/   selection, weights, turnover, costs
  backtest/    monthly rebalance loop and walk-forward testing
  metrics/     CAGR, Sharpe, Sortino, drawdown, alpha/beta, ...
  reports/     charts and tables
tests/         pytest suite
```

## License

MIT. See [LICENSE](LICENSE).
