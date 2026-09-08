# Task 1 implementation report

Implemented only the assigned core package and tests in `.worktrees/rl-core`; no commits, old-stack deletion, broker actions or external writes.

## Verification evidence

Shared Python: `C:/Users/111/Desktop/StockRLTrader/.venv/Scripts/python.exe`.

- Initial tests were written first; collection failed because the new `stockrl` package did not yet exist (three expected missing-package errors).
- Initial implementation: 32 tests passed, with two Gymnasium warnings about infinite observation bounds.
- Added regression cases before fixing: a passive baseline incorrectly sold 1 of 166 shares after a 60-to-100 opening gap, and observation-space limits were not finite. Both failed as expected, then passed after the fixes.
- Added CSV bit-exact roundtrip regression before changing parser: 46 of 100 numeric values differed under the default pandas parser. The `round_trip` parser makes all values identical.
- Final command: `python -m pytest tests/test_environment.py tests/test_data.py tests/test_evaluation.py -q` -> **40 passed, no warnings**. Includes Gymnasium and SB3 `check_env`.

Hand checks cover cash conservation, sale proceeds funding later purchases, next-open fills and overnight ownership, 50% action mapping, default integer shares/custom lots, all-in affordability, minimum commissions, directional slippage, sell tax, historical-volume sizing, execution-day zero volume, daily T+1 sale eligibility, actual executed-quantity turnover, reward penalties, random seeded episode boundaries, final mark-to-market truncation, causal features, train-only normalization, exact data reload and baseline NAV.

## Public contracts

- `data.load_csv(path)` selects Date/OHLCV from extra-column CSVs; returns float OHLCV indexed by strictly increasing unique DatetimeIndex named Date. Rejects invalid dates, duplicates, missing columns, nonfinite/nonpositive prices, invalid OHLC ranges and negative volume. Prices are preserved as supplied; adjustment/corporate-action consistency remains a source responsibility.
- `data.make_demo_data(n=756, seed=42)` returns deterministic synthetic OHLCV. These are demonstration data, not investment evidence.
- `features.build_features(bars)` returns eight finite, same-index, causal features. Warm-up uses only existing history, never backward-fill from future rows.
- `ObservationNormalizer.fit(frame)` is a classmethod returning a fitted normalizer. `transform(frame)` returns a same-index DataFrame and rejects changed columns/nonfinite input. Constants scale by one; transformed values clip to +/-10. JSON `save(path)` / `load(path)` preserve parameters. Consumers must supply training rows only to fit.
- `TradingConfig` exactly follows plan constructor and validates rates, cash, lot size and penalties.
- `TradingEnv(bars, config=None, start=None, end=None, normalizer=None, random_start=False, episode_length=None)` accepts integer inclusive boundaries or exact date boundaries. First decision observes start; first execution/reward is start+1; final mark is end. Optional episode length counts reward transitions; random episodes stay entirely in the provided bounds.
- `reset(seed=...)` returns float32 observation and initial info. `step(scalar_or_single_element_array)` maps [-1,1] to target [0,1]; finite inputs outside limits clip. Nonfinite/multi-element actions reject without advancing.
- Public attributes: `bars`, `features` (raw market features), `config`, `start`, `end`, `current_step`, `cash`, `shares`, `nav`, `history`, `trades`, `feature_names` (eight market and three account observation names).
- Observations concatenate market features, cash fraction, current holding weight, current drawdown; declared bounds +/-1e6, market values clipped before float32 conversion.
- History includes initial snapshot; each transition has date/nav/cash/shares/weight/turnover/cost/requested_weight/executed_weight/reward. Initial snapshot has no reward. Dates are ISO date strings. Trades record decision/execution date, side, share quantity, execution price, open price, notional, commission, tax, slippage cost and total cost.
- `evaluate_policy(env, policy)` resets and calls `policy(obs, env)` through the shared environment. Returns `metrics`, `history`, `trades` (the latter two are lists of JSON-safe dictionaries).
- Metrics: **total_return**, annualized_return, volatility, sharpe, max_drawdown, total_turnover, total_cost, average_weight, trade_count. Annualization assumes 252 daily periods, simple-return Sharpe assumes zero risk-free rate; insufficient observations/zero-volatility Sharpe and nonfinite results use null. Maximum drawdown is positive loss magnitude. Average weight excludes initial snapshot.
- `baseline_policy(name)` supports cash, buy_hold, half, trend. Trend compares current close to trailing 20-day mean. Buy_hold is a persistent full-allocation request: it retains shares, continues liquidity-limited entry and can invest residual rounding cash after a decline. This deliberately retains the same scalar target-weight API and shared execution engine for every policy.

## Accounting definitions and boundaries

Target shares are computed from pre-trade open NAV, then restricted by whole lots, historical-volume participation and available cash including both proportional/minimum fees. The target weight is a request, not a guaranteed fill. `executed_weight` is post-fill weight at the execution open; `weight` is closing weight. Turnover is actual share quantity times unadjusted opening price divided by pre-trade opening NAV. `cost` includes commissions, sale tax and directional slippage; fees/slippage already flow through cash/NAV, so base reward is log NAV return without subtracting them twice. Optional reward penalties use actual turnover and incremental drawdown only.

Capacity uses the trailing 20 volumes through decision close. Next day's volume is consulted only for the zero-volume suspension flag, never for fill sizing. At most one rebalance per trading row means prior-day purchases are unlocked at the next open; daily T+1 has no distinct intraday branch. Data and time-limit termination is truncation with outstanding holdings marked, never forced liquidation.

No changes to assigned core files are pending. Parent may review/integrate these files; no claim of profitable strategy performance is made.
