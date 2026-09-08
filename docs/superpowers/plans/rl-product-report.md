# Task 3 product entry points report

Date: 2026-09-08
Worktree: `C:\Users\111\Desktop\StockRLTrader\.worktrees\rl-core`
Scope owner: product entry points only; migration deletions were performed separately by the parent task.

## Delivered behavior

- Added `python -m stockrl` and the `stockrl` console entry point with `demo`, `train`, and `evaluate` workflows.
- `demo` generates `synthetic_demo` data and calls the real `run_experiment` API.
- `train` accepts exactly one of a validated local CSV or an optional Yahoo ticker download. Algorithm, timesteps, seeds, artifact root, chronological split ratios, episode length, cash, costs, lot size, and participation rate are configurable.
- `evaluate` calls `evaluate_saved_run` and replays the persisted bars, configuration, split, normalizer, and checkpoint without training or tuning.
- Replaced the Streamlit landing page with a local experiment dashboard. It loads synthetic data, in-memory CSV uploads, or preserved `stock_data/*_raw.csv` files; runs real PPO/SAC experiments; and displays test NAV, actual position, four baseline paths, metrics, and downloadable daily history.
- Upload names are reduced to a display-only basename. Uploaded bytes are parsed in memory and never used as a filesystem path.
- Replaced `run.py` with a current-Python Streamlit launcher and made `run_pipeline.py` a thin CLI delegate. Neither invokes a shell.
- Added Python packaging, dependency ranges, pytest configuration, safe artifact ignores, and the final Streamlit theme. Removed the prior CORS/XSRF overrides.
- Rewrote README around the implemented RL system, including exact chronology, accounting and data assumptions, precise baseline behavior, artifacts, entry points, limitations, and preserved historical files.

## Interface design

The UI follows the approved restrained experiment-record direction: `#101820` background, paper-white text, aqua RL trace, amber buy-and-hold reference, local Georgia/Segoe UI/Microsoft YaHei/Consolas font stack, dense but responsive controls, visible focus rings, 44 px actions, and reduced-motion handling. The sole high-emphasis visual is a shared-time-axis chart combining NAV paths with actual RL position. The interface contains no forecast probability, confidence score, trade recommendation, broker action, remote font, or decorative icon dependency.

The generic enterprise-gateway pattern returned by the UI guidance search was rejected because it did not match a local research tool. Accessibility, chart labeling, input feedback, dark contrast, and data-table fallback guidance were retained.

## TDD and verification evidence

Initial product test run failed 6 tests for the expected missing behavior: no module entry point, no CLI module, and the legacy Streamlit page importing the removed supervised stack.

After implementation:

```text
python -m pytest tests/test_cli.py tests/test_frontend.py -q
6 passed
```

These tests include a real four-timestep PPO optimization, saved model artifact, saved-run evaluation replay, Streamlit AppTest execution, real UI-triggered PPO training, and an upload named `../../escaped.csv` parsed without creating that path. No training or Streamlit mock is used.

Fresh integrated verification after core review fixes:

```text
python -m pytest -q
62 passed in 19.10s
```

Additional checks completed:

- `python -m stockrl --help` exits successfully and lists `demo`, `train`, and `evaluate` without importing legacy predictors.
- `python run_pipeline.py --help` delegates to the same parser.
- `python -m compileall -q stockrl frontend run.py run_pipeline.py` exits successfully.
- `pyproject.toml` and `.streamlit/config.toml` parse successfully with Python `tomllib`.
- `git diff --check` reported no whitespace errors (only the repository's expected LF-to-CRLF conversion notices).
- Source scan found no legacy predictor, supervised-learning, confidence, XGBoost, LightGBM, pandas-ta, or unsafe Streamlit server override references in active product entry points.

## Files owned

- `stockrl/cli.py`
- `stockrl/__main__.py`
- `frontend/app.py`
- `run.py`
- `run_pipeline.py`
- `requirements.txt`
- `pyproject.toml`
- `README.md`
- `.gitignore`
- `.streamlit/config.toml`
- `tests/test_cli.py`
- `tests/test_frontend.py`

No commit was created, as requested. The parent task owns final integration and any additional browser-verification notes.

## Bounded dashboard fix round

The final reviewer identified one important portability defect: loading a copied experiment folder in the dashboard reused absolute artifact paths from its original location. A real tiny experiment reproduced the resulting `FileNotFoundError` after the original folder was moved.

The saved-result loader now reconstructs each seed and baseline artifact path from the selected `summary.json` parent. Live results returned by the current process remain unchanged. Chart, comparison-table, and download preparation now share guarded artifact loading; a missing or malformed file produces a visible recovery message instead of terminating the page.

The same round corrected three browser findings: successful data loads immediately rerun so the sidebar reflects current state, primary aqua buttons use dark text, and five result metrics use a three-card plus two-card layout to prevent values from clipping at typical desktop widths.

Regression cycle:

```text
python -m pytest tests/test_frontend.py -q
# RED: 2 failed, 2 passed (stale sidebar and moved-run FileNotFoundError)
# GREEN: 4 passed in 6.17s
```

The moved-folder regression creates and trains a real four-timestep PPO experiment, moves its complete run directory, removes the original location, loads the copied result through Streamlit, verifies the rebased chart/history path, then removes `history.csv` and verifies a visible page error without an uncaught exception.
