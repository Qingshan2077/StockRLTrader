"""
A 股全流程管线入口

用法:
    python run_ashare.py                              # 默认 000001 (平安银行)
    python run_ashare.py --ticker 000568              # 指定股票
    python run_ashare.py --ticker 000568 --no-cache   # 强制重新下载
    python run_ashare.py --universe                   # 运行截面批量模式（实验性）

依赖:
    pip install akshare>=1.16.0
"""

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from utils.config_loader import ConfigLoader
from utils.logging_system import LoggingSystem
from utils.helpers import set_seed

from layers.data.ashare_provider import AShareDataProvider, CrossSectionalLoader
from layers.features import FeaturePipeline, FeatureRegistry, FeatureCache, LabelBuilder
from layers.signals import ModelTrainer
from layers.signals.models.lightgbm_model import LightGBMModel
from layers.signals.models.linear_model import LinearModel
from layers.signals.models.mlp_model import MLPSignalModel as MLPModel
from layers.evaluation.alpha_eval import AlphaEvaluator
from layers.evaluation.decay_monitor import SignalDecayMonitor
from layers.portfolio import PortfolioConstructor
from layers.risk.risk_engine import RiskEngine
from layers.backtest import BacktestEngine
from layers.experiments import ExperimentManager

PRICE_COLS = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'}


def main():
    parser = argparse.ArgumentParser(description="StockTrader — A 股管线")
    parser.add_argument("--ticker", default="000001", help="A 股 6 位数字代码")
    parser.add_argument("--start", default="2018-01-01", help="开始日期")
    parser.add_argument("--no-cache", action="store_true", help="强制重新下载")
    parser.add_argument("--universe", action="store_true", help="批量下载股票池数据（不下管线）")
    parser.add_argument("--universe-size", type=int, default=50, help="股票池大小")
    args = parser.parse_args()

    cfg = ConfigLoader()
    log = LoggingSystem()
    set_seed(cfg.get("system.seed", 42))

    ticker = args.ticker.strip().zfill(6)
    log.info(f"🚀 StockTrader A 股管线 — {ticker}")

    # 1. 数据
    log.info("[1/6] 加载 A 股数据...")
    provider = AShareDataProvider(
        ticker=ticker,
        start_date=args.start,
    )
    df = provider.load_or_download()
    if df is None or len(df) < 100:
        log.error(f"数据量不足: {len(df) if df is not None else 0}")
        return
    log.info(f"  数据: {len(df)} 行, {df.index[0].date()} ~ {df.index[-1].date()}")
    log.info(f"  列: {df.columns.tolist()}")

    # 2. 特征工程
    log.info("[2/6] 特征工程...")
    bench = provider.get_benchmark()
    registry = FeatureRegistry()
    cache = FeatureCache(data_dir=cfg.get("system.data_dir", "stock_data"))
    pipeline = FeaturePipeline(registry=registry, cache=cache, benchmark_data=bench)
    features = pipeline.build_features(df, ticker=ticker, use_cache=not args.no_cache)

    lb = LabelBuilder(horizon=5, label_type="regression")
    labels = lb.build_labels(features)
    features, labels = lb.align(features, labels)
    log.info(f"  特征: {features.shape}, 标签: {labels.shape}")

    # 3. 数据切分
    log.info("[3/6] 数据切分...")
    feat_cols = [c for c in features.columns if c not in PRICE_COLS]
    X_all = features[feat_cols].values.astype(np.float32)
    y_all = labels.values.astype(np.float32)
    n = len(X_all)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_tr, y_tr = X_all[:train_end], y_all[:train_end]
    X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test, y_test = X_all[val_end:], y_all[val_end:]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    log.info(f"  切分: train={len(X_tr)}, val={len(X_val)}, test={len(X_test)}")

    # 4. 模型训练
    log.info("[4/6] 模型训练...")
    active_models = ["lightgbm", "ridge"]
    all_test_preds = []

    for name in active_models:
        log.info(f"  训练: {name}")
        model_map = {
            "lightgbm": LightGBMModel,
            "ridge": lambda **kw: LinearModel(name="ridge", **kw),
            "mlp": MLPModel,
        }
        cls = model_map.get(name)
        if cls is None:
            continue

        m = cls(name=name, config=cfg.get(name, {}))
        trainer = ModelTrainer(m)
        _ = trainer.train(X_tr_s, y_tr, X_val_s, y_val)

        # 最终版本 (train+val)
        m_final = cls(name=f"{name}_final", config=cfg.get(name, {}))
        m_final.fit(np.vstack([X_tr_s, X_val_s]), np.concatenate([y_tr, y_val]))
        test_pred = m_final.predict(X_test_s)
        all_test_preds.append(test_pred)

    # 等权集成
    test_preds = np.mean(all_test_preds, axis=0)

    # 5. 评估
    log.info("[5/6] Alpha 评估...")
    if len(test_preds) >= 10:
        alphas = AlphaEvaluator(test_preds, y_test, dates=features.index[val_end:])
        ic = alphas.compute_ic()
        rank_ic = alphas.compute_rank_ic()
        ic_series = alphas.compute_ic_series(window=21)
        quantile = alphas.compute_quantile_analysis(n_buckets=5)

        log.info(f"  IC={ic.get('IC', 0):.4f}(p={ic.get('IC_pvalue', 1):.4f})")
        log.info(f"  RankIC={rank_ic.get('RankIC', 0):.4f}")
        log.info(f"  IC_IR={ic_series.get('IC_IR', 0):.3f}")
        if quantile:
            log.info(f"  top-bottom spread={quantile.get('top_bottom_spread', 0):.4f}")
            log.info(f"  Q1~Q5 收益: {[quantile.get(f'Q{i}_return', 0) for i in range(1, 6)]}")

    # 6. 回测
    log.info("[6/6] 回测...")
    test_features = features.iloc[val_end:]
    test_scores = test_preds[:len(test_features)]

    engine = BacktestEngine(type('C', (), {
        'initial_balance': 10000, 'commission': 0.001,
        'slippage': 0.0005, 'max_position': 1.0,
        'max_single': 1.0, 'tax': 0.0,
    })())
    r = engine.run("signal_only", test_features, test_scores)
    m = r["metrics"]
    log.info(f"  年化收益: {m.get('annualized_return', 0)*100:.2f}%")
    log.info(f"  夏普比率: {m.get('sharpe_ratio', 0):.2f}")
    log.info(f"  最大回撤: {m.get('max_drawdown', 0)*100:.2f}%")


if __name__ == "__main__":
    main()
