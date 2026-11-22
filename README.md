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
python -m momentum backtest --cost-sweep          # strategy vs benchmarks, 0-50 bps costs
python -m momentum backtest --vol-cost --save output/backtest.csv
python -m momentum walkforward --save output      # choose settings on the past, test on the future
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

## Backtest engine

Daily accounting with monthly rebalancing:

- Signals use the month-end close. Trades fill at the next trading day's close.
- Holdings drift with prices between rebalances. Turnover is measured against the
  drifted weights: `turnover = 1/2 * sum |target - current| / portfolio value`.
- Costs are charged on every dollar bought or sold: 10 bps flat by default, or
  `5 bps + 0.05 * daily volatility` per stock with `--vol-cost`. The trade and its
  cost are solved together, so the portfolio is fully invested *after* paying.
- A stock that stops trading is sold at its last price and the money is held in
  cash, which earns the 13-week T-bill rate. The portfolio never borrows.

Two benchmarks run over the same dates: QQQ bought and held, and an equal-weight
portfolio of every eligible Nasdaq-100 member with the same lag and costs. The
second shows what momentum *selection* adds beyond holding the index equally.

## Walk-forward testing

Settings are never chosen with hindsight. Every January:

1. All 48 settings in a grid fixed in advance (lookback 3/6/9/12 months, skip 0/1,
   hold 10/20/30 stocks, plain or risk-adjusted score) are scored on the training
   months before that January. The expanding window starts in February 2016, with
   a minimum of 35 months.
2. Each setting's training Sharpe is averaged with its grid neighbours, and the
   best smoothed score wins. A setting that only works on its own is treated as
   noise.
3. The winner trades for the next 12 months.

The test years (2019 onwards) run as one continuous portfolio. Holdings carry over
when the setting changes, and only the trades needed are made and paid for. A
5-year rolling window is run alongside as a robustness check.

Overfitting is measured two ways:

- **In-sample vs out-of-sample Sharpe.** The training Sharpe of the chosen
  settings (what the past promised) against what the next year delivered.
- **Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2014). The probability that
  the best of the 48 settings beats the Sharpe that luck alone would produce from
  48 tries. It's computed against T-bills (does it make money?) and against QQQ
  (does it beat the index?).

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
