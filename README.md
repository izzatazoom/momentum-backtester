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
python -m momentum signal --date 2020-12          # top 20 picks on a month-end
python -m momentum signal --date 2020-12 --risk-adjusted
```

## Signal

On the last trading day of each month, every eligible stock gets a 12-1 momentum score:

```
score = P(1 month ago) / P(12 months ago) - 1
```

The most recent month is skipped because one-month returns tend to reverse. The
risk-adjusted variant divides the score by the annualised volatility of daily returns
over the same window. A stock is eligible if it is an index member on the signal
date, priced that day and 12 months earlier, and priced on at least 90% of the
trading days in between. Only one share class per company is held (GOOGL, not
GOOGL and GOOG). The top 20 are held in equal weights. Ties go to the alphabetically
first ticker.

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
