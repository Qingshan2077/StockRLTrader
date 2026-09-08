# Task 2 implementation report

Implemented in `stockrl/training.py`, `stockrl/experiments.py`, and `tests/test_training.py`.

## Public entry points

- `run_experiment(...)` follows the plan and additionally accepts `progress_callback`. It creates a unique child run directory under the supplied artifact parent; its returned `output_dir` identifies that child.
- `load_agent(model_path, algorithm='PPO')` loads the selected SB3 checkpoint on CPU.
- `load_bundle(model_path)` returns `AgentBundle(agent, normalizer, config, metadata)` and a deterministic `.policy(observation, env=None)` callable. The seed folder is independently movable.
- `evaluate_saved_run(run_dir, output_dir=None)` reloads data, rules, normalizer and models, validates the saved data fingerprint, and evaluates the original test interval without retraining or fitting. It resolves source artifacts relative to the current run folder, so moving the run does not break replay. Default replay output is a separate child folder; overwriting the source run is rejected.
- `split_intervals(bars, train_ratio=.6, val_ratio=.2)` records integer bounds and observation/reward dates. Adjacent intervals share a boundary observation, never a reward. Minimum partition sizes are 20 training transitions and five each for validation and test.

Each baseline entry contains `metrics`, `history_path`, and `trades_path`. Aggregate entries contain `mean`, population `std` (`ddof=0`), and `count`; undefined metrics remain null. No seed winner is selected using test results.

## Training and artifacts

Both algorithms use actual Stable-Baselines3 optimizers, CPU seeds, deterministic PyTorch operations, a small two-layer 32-unit policy, bounded rollout/batch sizes for smoke runs, and fresh train/validation environments. SAC warmup scales down for short runs so smoke checkpoints have real gradient updates.

The validation callback extends SB3 `EvalCallback`. It excludes policies with zero gradient updates and always evaluates after the final optimization. This handles the first-rollout boundary where standard step-only evaluation could otherwise save an untrained policy. The checkpoint is selected exclusively by fixed validation mean reward. The learner has no test environment argument.

Normalizer fitting uses only the training observation prefix. Saved models are evaluated once per seed on test; all four baselines use identical boundaries/configuration and independent fresh accounts. Saved run artifacts include OHLCV snapshot, SHA256, account rules, package/Python versions, data label, dates, seeds, split boundaries, selected models, normalizers, validation result traces, learning settings, actual training steps/updates, and policy/baseline histories and trades. History includes requested/executed weight and realized account state.

Progress callbacks receive `training`, `seed_complete`, and `complete` event dictionaries. The training event includes seed and actual elapsed timesteps.

## Test-first evidence

1. Initial eight tests were written before these modules existed. The red run produced **8 failures**, all for the missing `stockrl` implementation; no collection failure or mocked training was used.
2. After implementation and agreement on core metrics, **8 tests passed in 7.21 seconds**.
3. Added an independent replay-portability/source-protection regression. It first exposed floating point CSV reload differences and then **DID NOT RAISE ValueError** for attempted source overwrite. Core CSV loading now uses round-trip precision, and replay now rejects source overwrite.
4. Final training-only run: **9 passed in 6.92 seconds**.
5. Final combined data/environment/evaluation/training integration: **49 passed in 7.17 seconds**, no warnings.

Verification command (from the worktree):

```text
C:/Users/111/Desktop/StockRLTrader/.venv/Scripts/python.exe -m pytest tests/test_data.py tests/test_environment.py tests/test_evaluation.py tests/test_training.py -q --noconftest
```

The tests cover real PPO and SAC gradient updates/save/load, validation-selected checkpoint evidence, exact moved-directory replay metrics, same-policy reload NAV, cash baseline and test dates, immutable preprocessing under validation/test changes, identical trained policies and validation traces under future-test perturbation, three-seed aggregate inclusion, tiny/invalid split rejection, and saved market tamper detection. All artifacts use isolated pytest temporary output directories.

Smoke runs demonstrate software operation only; they provide no evidence of profitability. No commit, frontend/CLI edit, or external action was made by this task.

## Round 1 review corrections

Two P2 regressions were independently reproduced with real integrations before fixes:

- A PPO budget of 148 produced a raw rollout of 37, which has no batch divisor in 2..32. The red regression failed with `ValueError: max() iterable argument is empty`. Rollouts are now rounded down to an even number, retaining the 2..128 rollout and 2..32 batch bounds while guaranteeing a valid divisor. The regression trains for the requested budget, checks actual gradient updates, and loads a trained selected checkpoint. Its isolated green run passed in 7.04 seconds.
- A replay pointed at another existing experiment directory overwrote its summary and evaluation files. The red regression failed with `DID NOT RAISE ValueError`. Replay now creates a fresh destination atomically with `exist_ok=False` and translates any collision into a clear `ValueError` before artifact writes. The regression snapshots every file in an actual second experiment and verifies all bytes and paths remain unchanged. Its isolated green run passed in 7.03 seconds.

Final covering command: `C:/Users/111/Desktop/StockRLTrader/.venv/Scripts/python.exe -m pytest tests/test_training.py -q --noconftest` — **11 passed in 12.61 seconds**, with no warnings. These corrections touched only the two owned modules, training tests, and this report; no commit was made. Explicit replay output now requires a previously nonexistent directory.
