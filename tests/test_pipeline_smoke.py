"""
pipeline 冒烟测试 — 用合成数据验证完整管线是否可运行

用法:
    python tests/test_pipeline_smoke.py

输出:
    - 各阶段耗时
    - 回测结果摘要
    - 错误位置（如果有）

注意:
    - 使用合成数据，不需要 yfinance
    - 跳过 RL（需要 PyTorch）
    - 返回 exit code 0 表示成功
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
import json


def make_synthetic_data(n_days: int = 504) -> pd.DataFrame:
    """生成合成日线 OHLCV 数据"""
    rng = np.random.RandomState(42)
    t = np.arange(n_days)
    trend = np.sin(2 * np.pi * t / 252) * 0.3
    trend[:126] = np.linspace(0, 0.4, 126)
    noise = rng.randn(n_days) * 0.015
    ret = np.diff(np.concatenate([[0], trend])) * 0.02 + noise
    price = 100 * np.cumprod(1 + ret)
    dates = [datetime(2020, 1, 1) + timedelta(days=int(i)) for i in range(n_days)]
    df = pd.DataFrame({
        'Open': price * (1 + rng.randn(n_days) * 0.005),
        'High': price * (1 + abs(rng.randn(n_days)) * 0.01),
        'Low': price * (1 - abs(rng.randn(n_days)) * 0.01),
        'Close': price,
        'Volume': rng.randint(1_000_000, 10_000_000, n_days),
    }, index=pd.DatetimeIndex(dates, name='Date'))
    return df.sort_index()


def test_pipeline_smoke():
    """
    完整管线冒烟测试。

    步骤:
      1. 创建临时目录存放数据
      2. 生成合成数据
      3. 加载配置
      4. 特征工程
      5. 数据切分 + 标准化
      6. 模型训练 + OOF 预测
      7. Alpha 评估
      8. 风控 + 回测
      9. 结果验证
    """
    import time
    t_start = time.time()

    print("=" * 70)
    print("  StockTrader v3 Pipeline 冒烟测试")
    print("=" * 70)

    # ── 1. 准备临时环境 ──
    tmpdir = Path(tempfile.mkdtemp(prefix="stocktrader_test_"))
    data_dir = tmpdir / "stock_data"
    data_dir.mkdir()

    # 生成合成数据
    df = make_synthetic_data()
    df.to_csv(data_dir / "SYNTH_raw.csv")
    print(f"  [1/9] 合成数据: {len(df)} 行 | 临时目录: {tmpdir}")

    # ── 2. 加载配置 ──
    from utils.config_loader import ConfigLoader
    from utils.logging_system import LoggingSystem
    from utils.helpers import set_seed

    # 使用默认配置 + 覆盖
    os.environ["QUANT_system_data_dir"] = str(data_dir)
    os.environ["QUANT_system_start_date"] = "2020-01-01"
    os.environ["QUANT_data_split_train_ratio"] = "0.6"
    os.environ["QUANT_data_split_val_ratio"] = "0.2"

    cfg = ConfigLoader()
    log = LoggingSystem()
    set_seed(42)
    print(f"  [2/9] 配置已加载")

    # ── 3. 特征工程 ──
    from layers.features import FeaturePipeline, FeatureRegistry, FeatureCache, LabelBuilder

    registry = FeatureRegistry()
    cache = FeatureCache(data_dir=str(data_dir))
    pipeline = FeaturePipeline(registry=registry, cache=cache, benchmark_data=None)

    features = pipeline.build_features(df, ticker="SYNTH", use_cache=False)
    print(f"  [3/9] 特征: {features.shape}, 因子数: {len(registry)}")

    horizon = 5
    lb = LabelBuilder(horizon=horizon, label_type="regression")
    labels = lb.build_labels(features)
    features, labels = lb.align(features, labels)
    print(f"  [4/9] 标签: {len(labels)} 条, 范围 [{labels.min():.4f}, {labels.max():.4f}]")

    # ── 4. 数据切分 ──
    feat_cols = [c for c in features.columns
                 if c not in ('Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close')]
    X_all = features[feat_cols].values.astype(np.float32)
    y_all = labels.values.astype(np.float32)

    n = len(X_all)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    print(f"  [5/9] 切分: train={train_end}({60}%) val={val_end - train_end}({20}%) test={n - val_end}({20}%)")

    # ── 5. 模型训练 + OOF ──
    from run_pipeline import _oof_predict
    from layers.signals.models.linear_model import LinearModel
    from layers.signals import ModelTrainer
    from sklearn.preprocessing import StandardScaler

    X_tr, y_tr = X_all[:train_end], y_all[:train_end]
    X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test, y_test = X_all[val_end:], y_all[val_end:]

    scaler_tr = StandardScaler().fit(X_tr)
    X_tr_s = scaler_tr.transform(X_tr)
    X_val_s = scaler_tr.transform(X_val)

    # OOF
    oof_preds = _oof_predict(lambda config=None: LinearModel(name="ridge", config=config),
                             X_tr, y_tr, n_folds=5, config={"alpha": 1.0})
    print(f"  [6/9] OOF 预测: {np.sum(np.abs(oof_preds) > 1e-8)}/{len(oof_preds)} 非零")

    # 主模型
    model = LinearModel(name="ridge", config={"alpha": 1.0})
    trainer = ModelTrainer(model)
    metrics = trainer.train(X_tr_s, y_tr, X_val_s, y_val)
    print(f"  [6/9] 模型: val_r2={metrics.get('val_r2', 0):.4f}")

    val_preds = model.predict(X_val_s)

    # 最终模型
    scaler_full = StandardScaler().fit(np.vstack([X_tr, X_val]))
    X_train_val_s = scaler_full.transform(np.vstack([X_tr, X_val]))
    X_test_s = scaler_full.transform(X_test)
    final_model = LinearModel(name="ridge_final", config={"alpha": 1.0})
    final_model.fit(X_train_val_s, np.concatenate([y_tr, y_val]))
    test_preds = final_model.predict(X_test_s)

    # 合成 signal_scores
    signal_scores = np.zeros(n, dtype=np.float64)
    signal_scores[:train_end] = oof_preds
    signal_scores[train_end:val_end] = val_preds
    signal_scores[val_end:] = test_preds
    print(f"  [7/9] 信号: min={signal_scores.min():.4f}, max={signal_scores.max():.4f}, 非零={(signal_scores != 0).mean():.1%}")

    # ── 6. Alpha 评估 ──
    from layers.evaluation.alpha_eval import AlphaEvaluator
    alphas = AlphaEvaluator(test_preds, y_test)
    ic = alphas.compute_ic()
    rank_ic = alphas.compute_rank_ic()
    quantile = alphas.compute_quantile_analysis()
    print(f"  [8/9] IC={ic.get('IC', 0):.4f}, RankIC={rank_ic.get('RankIC', 0):.4f}, "
          f"spread={quantile.get('top_bottom_spread', 0):.4f}")

    # ── 7. 波动率计算 ──
    from run_pipeline import compute_rolling_volatility
    prices_full = features['Close'].values.astype(np.float64)
    vol_full = compute_rolling_volatility(prices_full, window=20)
    print(f"  [8/9] 波动率: mean={vol_full.mean():.3f}, [{vol_full.min():.3f}, {vol_full.max():.3f}]")

    # ── 8. 组合构建 + 风控 ──
    from layers.portfolio import PortfolioConstructor
    from layers.risk.risk_engine import RiskEngine

    train_signals = signal_scores[:train_end]
    constructor = PortfolioConstructor(method="sigmoid", alpha=3.0,
                                        max_weight=1.0, min_weight=-1.0)
    if np.any(train_signals != 0):
        constructor.fit(train_signals)
    print(f"  [8/9] 组合构建: method=sigmoid")

    # 使用 DummyConfig 风控
    class DummyConfig:
        def get(self, key, default=None):
            cfg = {
                "risk": {
                    "position": {"max_gross_exposure": 1.0, "max_net_exposure": 1.0,
                                 "max_single_asset_weight": 1.0},
                    "volatility": {"enabled": True, "target_vol": 0.25,
                                   "vol_window": 20, "vol_threshold": 1.5},
                    "drawdown": {"enabled": True, "circuit_breaker": -0.20,
                                 "deleverage_factor": 0.5},
                    "liquidity": {"enabled": True, "adv_limit_ratio": 0.01,
                                  "minimum_volume": 10000},
                    "turnover": {"max_daily_turnover": 0.3},
                    "stop_loss": {"threshold": -0.05, "reentry_penalty": 0.5},
                },
                "backtest": {
                    "commission": 0.001, "slippage": 0.0005, "tax": 0.0,
                    "market_impact": {"model": "sqrt", "k": 0.1},
                },
            }
            keys = key.split(".")
            v = cfg
            for k in keys:
                v = v.get(k, {}) if isinstance(v, dict) else default
            return v if v != {} else default

    risk_engine = RiskEngine(DummyConfig())
    print(f"  [8/9] 风控引擎已初始化")

    # ── 9. 回测 ──
    from layers.backtest.backtest_engine import BacktestEngine, BacktestConfig

    bt_config = BacktestConfig(
        initial_balance=10000.0, commission=0.001, slippage=0.0005,
        max_position=1.0, max_single=1.0,
    )
    engine = BacktestEngine(bt_config)

    test_features = features.iloc[val_end:]
    test_signal = signal_scores[val_end:]

    results = {}

    # signal_only
    risk_engine.reset()
    r1 = engine.run("signal_only", test_features, test_signal)
    results["signal_only"] = r1["metrics"]
    print(f"  [9/9] signal_only: return={r1['metrics'].get('annualized_return', 0)*100:.2f}%, "
          f"sharpe={r1['metrics'].get('sharpe_ratio', 0):.3f}, "
          f"dd={r1['metrics'].get('max_drawdown', 0)*100:.2f}%")

    # signal_risk
    risk_engine.reset()
    r2 = engine.run("signal_risk", test_features, test_signal, risk_manager=risk_engine)
    results["signal_risk"] = r2["metrics"]
    print(f"  [9/9] signal_risk: return={r2['metrics'].get('annualized_return', 0)*100:.2f}%, "
          f"sharpe={r2['metrics'].get('sharpe_ratio', 0):.3f}, "
          f"dd={r2['metrics'].get('max_drawdown', 0)*100:.2f}%")

    n_trades = sum(len(r.get('trades', [])) for r in [r1, r2])
    print(f"  [9/9] 交易总数: {n_trades}")

    # ── 10. 清理 ──
    shutil.rmtree(tmpdir)
    elapsed = time.time() - t_start

    print(f"\n{'='*70}")
    print(f"  冒烟测试完成 ✅  |  耗时: {elapsed:.1f}s")
    print(f"{'='*70}")

    # 基本验证
    checks = [
        ("特征工程有输出", len(registry) > 10),
        ("OOF 有非零预测", np.any(np.abs(oof_preds) > 1e-8)),
        ("IC 计算正常", ic.get("IC", -999) > -2),
        ("RankIC 计算正常", rank_ic.get("RankIC", -999) > -2),
        ("signal_only 回测有结果", r1['metrics'].get('n_days', 0) > 0),
        ("signal_risk 回测有结果", r2['metrics'].get('n_days', 0) > 0),
        ("波动率无 NaN", not np.any(np.isnan(vol_full))),
    ]

    all_pass = True
    for name, ok in checks:
        status = "✅" if ok else "❌"
        if not ok:
            all_pass = False
        print(f"  {status} {name}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(test_pipeline_smoke())
