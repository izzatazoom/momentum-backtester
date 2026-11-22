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
python -m momentum data         # download prices once, then print a data quality report
```

## Data

- Universe: point-in-time Nasdaq-100 membership, rebuilt from the published history
  of index changes ([data/universe](data/universe)). A 2016 rebalance can only pick
  stocks that were in the index in 2016.
- Prices: daily split- and dividend-adjusted closes from Yahoo Finance, cached in
  `data/cache/` so every run uses the same data.
- Benchmark: QQQ. Risk-free rate: 13-week T-bill yield (`^IRX`).
- Known gap: Yahoo has no history for companies that were acquired or delisted.
  About 12% of member-days can't be priced, which biases results upward. The
  universe README lists the affected names.

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
