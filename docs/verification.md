# RL refactor verification — 2026-09-08

This is software validation, not evidence that a trading strategy is profitable.

## Scope

- Daily, single-asset, long-only target position decisions with PPO and SAC.
- Observe today's close, execute at the next available session's open, mark at that session's close.
- One environment handles training, saved evaluation and all four baselines.
- Train-only preprocessing, validation-only checkpoint selection, independent test reward intervals, and all-seed reporting.
- The old supervised prediction stack was removed. Existing market CSV/metadata, historical exports and the experiment database were preserved unchanged.

## End-to-end evidence

- Full test suite: **63 passed in 9.63s**, with no warnings, on Python 3.12 / Windows. Coverage includes hand-calculated account cases, same-day input rejection, future-data isolation, actual PPO/SAC learning, artifact overwrite prevention, moved-run replay and dashboard recovery.
- SAC: synthetic 252-session dataset, 128 steps per seed, seeds 42 and 43. Both runs completed with 96 gradient updates. Reloading both saved checkpoints reproduced every test metric and the entire daily NAV history exactly.
- PPO: the existing AAPL CSV, restricted to dates from 2023-01-01, completed 128 training steps and saved a validation-selected checkpoint and test results.
- Browser: loaded 756 synthetic daily bars, trained PPO for 16 steps, and displayed the saved NAV, actual allocation and four baseline trajectories.
- The dashboard's saved-run test moves a real experiment, confirms charts load, then removes its history file and confirms a recoverable error is displayed.
- The installed environment passes `python -m pip check`.

The command-line verification artifacts are local, ignored files under `outputs/verification`. Their summaries record data fingerprints, dates, configuration, seed, actual training steps, gradient updates and package versions.

## Existing data quality

All existing raw CSVs were checked without modifying them. AAPL, BABA, JD, MSFT, NVDA, TSLA and 000568.SZ passed the OHLCV validator. This only checks structural consistency, not economic accuracy or corporate-action treatment.

- `002594.SZ_raw.csv` has an invalid OHLC range on 2025-12-16.
- `399997.SZ_raw.csv` contains only one row, insufficient for an experiment.
- `399997_raw.csv` is empty.

The application rejects invalid or insufficient data rather than silently repairing it.

## Research limits

These short runs verify execution and reproducibility. They do not assess a policy's statistical significance, stability across market regimes, or suitability for real trading. Multi-asset allocation, walk-forward research, order-book matching, limit-order queues, corporate actions and broker execution require further work. T+1 follows from the daily one-rebalance schedule; this environment does not simulate intraday trading.
