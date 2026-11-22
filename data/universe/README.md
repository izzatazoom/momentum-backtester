# Nasdaq-100 universe

| File | Contents |
|---|---|
| `nasdaq100_current.csv` | Current constituents |
| `nasdaq100_changes.csv` | Every index addition and removal since 2007 |
| `corrections.csv` | Manual fixes applied on load, each with a note |
| `yahoo_symbols.csv` | Index tickers whose price history Yahoo files under a newer symbol |
| `share_classes.csv` | Companies with more than one listed class. Only the highest-priority eligible class can be held |

The first two are parsed from Wikipedia ("List of Nasdaq-100 companies" and
"Historical components of the Nasdaq-100", retrieved 2026-11-23) by
`scripts/build_universe.py`. They are kept as parsed. Fixes live in
`corrections.csv` so every deviation from the source is visible.

`momentum.data.universe` replays the changes backwards from the current list to get
point-in-time membership. On a fresh rebuild it checks that no change contradicts
the reconstructed state (for example removing a stock that wasn't a member).

## Known gaps

Yahoo deletes the history of companies that were acquired, taken private or went
bankrupt (CELG, XLNX, ATVI, EA, ...). Those members can't be priced, so the backtest
can never pick them. This is the survivorship bias that remains with free data, and
`python -m momentum data` reports how many member-days it affects.

## Reviewed flags (data downloaded 2026-11-23)

Late start, where prices begin after membership does:
- FOX, FOXA: Yahoo's history starts with Fox Corp (March 2019). The earlier 21st
  Century Fox years can't be priced. Fox Corp itself was the member from 2019.
- SNDK: the ticker was reused. Old SanDisk (left 2016) has no data; the 2025
  SanDisk spin-off, a member since 2026, is priced normally.
- WBD: a single price on 2026-10-05, just before its acquisition. It can never
  build the history the signal needs.

One-day moves above 50%: only PDD (+56% on 2022-03-16, a real rally in Chinese
tech stocks) happened while the stock was a member. The rest fall outside
membership. LILAK (2015-07-01, the day the tracking stock was created) and VIP
(2021-2025, share-ratio changes) look like artifacts, but neither stock was a member
afterwards long enough for them to affect a signal. MRNA's +177% on 2026-08-19,
inside the lookback for its October 2026 return, is unverified.

## Not mapped

Liberty Media (LMCA/LMCK) is left unmapped. Its 2016 recapitalisation into tracking
stocks means Yahoo's FWONA/FWONK history may not be the security the index held.
