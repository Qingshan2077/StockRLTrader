"""
全流程编排 — 一键执行 pipeline v3
带 OOF 预测、真实波动率、严格的数据泄露防护

生产级设计原则:
  - 测试集在任何阶段都不可见（包括 RL 训练）
  - 所有信号分数通过时序 OOF (out-of-fold) 或递进训练产生，无前视偏差
  - 波动率使用真实滚动计算，不硬编码
  - 回测支持三种模式对比（可在 signal_risk 和 full 之间选基准模式）

用法:
    python run_pipeline.py                              # 默认 AAPL + LightGBM
    python run_pipeline.py --ticker NVDA                # 指定股票
    python run_pipeline.py --ticker 000568.SZ           # A 股（需 AKShare 后端）
    python run_pipeline.py --skip-rl                    # 跳过 RL 训练
    python run_pipeline.py --walk-forward               # 启用 Walk-Forward 回测
    python run_pipeline.py --multi-model                # 启用多模型集成
    python run_pipeline.py --baseline-only              # 只跑 signal_only 快速验证
"""
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

from utils.config_loader import ConfigLoader
from utils.logging_system import LoggingSystem
from utils.helpers import set_seed

from layers.data.data_provider import MarketDataProvider
from layers.features import FeaturePipeline, FeatureRegistry, FeatureCache, LabelBuilder
from layers.signals import ModelTrainer, LabelFactory
from layers.signals.models.lightgbm_model import LightGBMModel
from layers.signals.models.xgboost_model import XGBoostModel
from layers.signals.models.linear_model import LinearModel
from layers.signals.models.mlp_model import MLPSignalModel as MLPModel
from layers.ensemble import EnsembleManager
from layers.evaluation.alpha_eval import AlphaEvaluator
from layers.portfolio import PortfolioConstructor
from layers.risk.risk_engine import RiskEngine
from layers.rl.trainer import RLTrainer
from layers.backtest import BacktestEngine, BacktestReport
from layers.backtest.walk_forward import WalkForwardBacktest
from layers.evaluation.metrics import MetricsCalculator
from layers.experiments import ExperimentManager


# ─────────────────────────────────────────────
#  全局常量
# ─────────────────────────────────────────────
PRICE_COLS = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """返回 DataFrame 中的特征列（排除了 OHLCV 价格列）"""
    return [c for c in df.columns if c not in PRICE_COLS]


def compute_rolling_volatility(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    从价格序列计算滚动年化波动率。

    返回和输入等长的数组，前 window-1 个元素用全局均值填充，
    避免回测因前导 NaN 缩短。
    """
    ret = np.diff(prices) / prices[:-1]
    vol = np.full(len(prices), np.nan)
    # 滚动标准差 → 年化
    for i in range(window, len(prices)):
        vol[i] = np.std(ret[i - window:i]) * np.sqrt(252)
    # 前导填充：用第一个有效值向前填充
    first_valid = np.where(~np.isnan(vol))[0]
    if len(first_valid) > 0:
        vol[:first_valid[0]] = vol[first_valid[0]]
    # 仍有 NaN → 兜底 0.2
    vol = np.nan_to_num(vol, nan=0.2)
    return vol.astype(np.float64)


def _expand_predict(model_class, X_train, y_train, X_val, y_val, config):
    """
    递进训练 + 预测: train → predict val。

    适用场景：从 train 集训练，预测 val 集。
    """
    m = model_class(config=config)
    trainer = ModelTrainer(m)
    metrics = trainer.train(X_train, y_train, X_val, y_val)
    preds = m.predict(X_val)
    return preds, metrics


def _oof_predict(
    model_class, X_all: np.ndarray, y_all: np.ndarray,
    n_folds: int = 5, config: dict = None
) -> np.ndarray:
    """
    时序 OOF (out-of-fold) 预测。

    使用 expanding window 方式切分：
      fold 0: train [0:fold1], predict [fold1:fold2]
      fold 1: train [0:fold2], predict [fold2:fold3]
      ...
    确保每个预测点只使用其之前的数据，无任何未来信息泄露。

    返回：
        和 X_all 等长的 ndarray，前 fold1 个元素为 0（不可预测）。
    """
    n = len(X_all)
    if n < 100:
        return np.zeros(n)
    fold_size = n // (n_folds + 1)
    oof = np.zeros(n, dtype=np.float64)

    for i in range(n_folds):
        train_end = (i + 1) * fold_size
        val_end = min(train_end + fold_size, n)
        if val_end - train_end < 20:
            break

        X_tr, y_tr = X_all[:train_end], y_all[:train_end]
        X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_val_s = scaler.transform(X_val)

        m = model_class(config=config or {})
        trainer = ModelTrainer(m, verbose=False)
        trainer.train(X_tr_s, y_tr, None, None)
        preds = m.predict(X_val_s)
        oof[train_end:val_end] = preds

    return oof


def main():
    parser = argparse.ArgumentParser(description="机构级量化交易系统 — 全流程 v3")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--config", default=None, help="自定义 YAML 配置")
    parser.add_argument("--stage", default="single_asset",
                        choices=["single_asset", "multi_asset"])
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--multi-model", action="store_true",
                        help="启用多模型集成（默认只用 LightGBM）")
    parser.add_argument("--baseline-only", action="store_true",
                        help="只跑 signal_only 回测，跳过风控和 RL（快速验证用）")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    import pandas as pd  # 延迟导入避免 argparse 阻塞

    # ====== 0. 初始化 ======
    cfg = ConfigLoader(override_file=args.config)
    log = LoggingSystem()
    set_seed(cfg.get("system.seed", 42))

    ticker = args.ticker
    exp_mgr = ExperimentManager()
    exp_id = exp_mgr.create_experiment(
        name=f"pipeline_{ticker}_{datetime.now():%Y%m%d_%H%M}",
        config=cfg.to_dict(),
        tags=[ticker, args.stage, "pipeline"],
    )
    log.info(f"Pipeline started: exp_id={exp_id}, ticker={ticker}")

    # ====== 1. 数据 ======
    log.info("[1/9] 加载数据...")
    provider = MarketDataProvider(
        ticker,
        data_dir=cfg.get("system.data_dir", "stock_data"),
        start_date=cfg.get("system.start_date", "2015-01-01"),
        cache_format=cfg.get("market_data.cache_format", "csv"),
    )
    df = provider.load_or_download()
    if df is None or len(df) < 100:
        log.error(f"数据量不足: {len(df) if df is not None else 0} 行")
        return
    log.info(f"数据: {len(df)} 行, {df.index[0].date()} ~ {df.index[-1].date()}")

    # ====== 2. 特征工程 ======
    log.info("[2/9] 特征工程...")
    bench = provider.get_benchmark()
    registry = FeatureRegistry()
    cache = FeatureCache(data_dir=cfg.get("system.data_dir", "stock_data"))
    pipeline = FeaturePipeline(registry=registry, cache=cache, benchmark_data=bench)
    features = pipeline.build_features(df, ticker=ticker, use_cache=not args.no_cache)

    # 标签构造
    horizon = cfg.get("label.horizons", [5])[0]
    lb = LabelBuilder(horizon=horizon, label_type=cfg.get("label.type", "regression"),
                      classification_threshold=cfg.get("label.classification_threshold", 0.005))
    labels = lb.build_labels(features)
    features, labels = lb.align(features, labels)
    log.info(f"特征: {features.shape}, 标签: {labels.shape}, "
             f"缓存命中率: {cache.hit_rate:.0%}")

    # ====== 3. 数据切分与标准化 ======
    log.info("[3/9] 数据切分...")
    feat_cols = get_feature_columns(features)
    X_all = features[feat_cols].values.astype(np.float32)
    y_all = labels.values.astype(np.float32)
    dates_all = features.index.values  # 保留日期索引

    n = len(X_all)
    train_ratio = cfg.get("data_split.train_ratio", 0.6)
    val_ratio = cfg.get("data_split.val_ratio", 0.2)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_tr, y_tr = X_all[:train_end], y_all[:train_end]
    X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test, y_test = X_all[val_end:], y_all[val_end:]

    log.info(f"切分: train={train_end}({train_ratio:.0%}) "
             f"val={val_end - train_end}({val_ratio:.0%}) "
             f"test={n - val_end}({1 - train_ratio - val_ratio:.0%})")

    # ====== 4. Alpha 模型训练 & 全量预测 ======
    #   — 训练集做 OOF 预测（不泄露未来）
    #   — 验证集用 train 训练后预测
    #   — 测试集用 train+val 训练后预测（完全 OOD）
    log.info("[4/9] 训练 Alpha 模型...")

    active_models = cfg.get("active_models", ["lightgbm"])
    model_map = {
        "lightgbm": LightGBMModel,
        "xgboost": XGBoostModel,
        "ridge": lambda **kw: LinearModel(name="ridge", **kw),
        "lasso": lambda **kw: LinearModel(name="lasso", **kw),
        "mlp": MLPModel,
    }

    # 标准化器：只在训练集上 fit
    from sklearn.preprocessing import StandardScaler
    scaler_tr = StandardScaler().fit(X_tr)
    X_tr_s = scaler_tr.transform(X_tr)
    scaler_full = StandardScaler().fit(np.vstack([X_tr, X_val]))
    X_train_val_s = scaler_full.transform(np.vstack([X_tr, X_val]))

    # ── 主模型：在 train 上训练，预测 val + test ──
    model_cls = model_map.get(active_models[0], LightGBMModel)
    model_cfg = cfg.get(active_models[0], {})
    log.info(f"主模型: {active_models[0]}")

    # 标准化 val 和 test（用相应的 scaler）
    X_val_s = scaler_tr.transform(X_val)
    X_test_s_for_test = scaler_full.transform(X_test)  # 主模型用 train+val 重新训练后推理

    # 训练主模型
    import lightgbm as lgb
    main_model = model_cls(name=active_models[0], config=model_cfg)
    main_trainer = ModelTrainer(main_model)
    train_metrics = main_trainer.train(X_tr_s, y_tr, X_val_s, y_val)
    log.info(f"主模型: val_r2={train_metrics.get('val_r2', 0):.4f}, "
             f"val_rmse={train_metrics.get('val_rmse', 0):.4f}")

    # 再训练一个 train+val 版本用于 test 集推理（这是正确的做法：用最多可用数据训练，再在未见过的测试集上评估）
    final_model = model_cls(name=f"{active_models[0]}_final", config=model_cfg)
    final_trainer = ModelTrainer(final_model)
    final_metrics = final_trainer.train(X_train_val_s, np.concatenate([y_tr, y_val]))
    log.info(f"最终模型 (train+val): 训练完成")

    # ── 生成全量 signal_scores ──
    #   约束：RL 可能看到的数据不能包含 test 集的信息。
    #   方案：
    #     - train 段: OOF 预测（仅用历史数据）
    #     - val 段:   主模型预测（train→val）
    #     - test 段:  最终模型预测（train+val→test）
    log.info("生成全量信号分数...")

    signal_scores = np.zeros(n, dtype=np.float64)

    # train 段: OOF
    oof_preds = _oof_predict(model_cls, X_tr, y_tr, n_folds=5, config=model_cfg)
    signal_scores[:train_end] = oof_preds

    # val 段: 主模型预测（train → val）
    val_preds = main_model.predict(X_val_s)
    signal_scores[train_end:val_end] = val_preds

    # test 段: 最终模型预测（train+val → test）
    test_preds = final_model.predict(X_test_s_for_test)
    signal_scores[val_end:] = test_preds

    log.info(f"信号范围: [{signal_scores.min():.4f}, {signal_scores.max():.4f}], "
             f"非零比例: {(signal_scores != 0).mean():.1%}")

    # ====== 5. Alpha 评估 ======
    log.info("[5/9] Alpha 评估...")
    if len(test_preds) >= 10:
        test_labels = y_test
        eval_signal = test_preds[:len(test_labels)]
        alphas = AlphaEvaluator(eval_signal, test_labels, dates=features.index[val_end:val_end + len(test_labels)])
        ic = alphas.compute_ic()
        rank_ic = alphas.compute_rank_ic()
        ic_series = alphas.compute_ic_series(window=21)
        quantile = alphas.compute_quantile_analysis(n_buckets=5)
        log.info(f"IC={ic.get('IC', 0):.4f}(p={ic.get('IC_pvalue', 1):.4f}), "
                 f"RankIC={rank_ic.get('RankIC', 0):.4f}, "
                 f"IC_IR={ic_series.get('IC_IR', 0):.3f}")
        if quantile:
            spread = quantile.get('top_bottom_spread', 0)
            log.info(f"分位数 spread={spread:.4f}")
    else:
        ic, rank_ic = {"IC": 0}, {"RankIC": 0}
        quantile = {}
        log.warn("测试集样本不足，跳过 Alpha 评估")

    # ====== 6. 多模型集成（可选） ======
    if args.multi_model:
        log.info("[5b/9] 多模型集成...")
        models = []
        for name in active_models:
            if name == active_models[0]:
                models.append(final_model)
                continue
            cls = model_map.get(name)
            if cls is None:
                continue
            m = cls(name=name, config=cfg.get(name, {}))
            m.fit(X_train_val_s, np.concatenate([y_tr, y_val]))
            models.append(m)

        ensemble_mgr = EnsembleManager(method=cfg.get("ensemble.method", "weighted"))
        ensemble_mgr.add_models(models)
        ensemble_preds = ensemble_mgr.predict(X_test_s_for_test)
        # 用集成信号覆盖测试段（如果在 baseline-mode 则跳过）
        if len(ensemble_preds) == len(test_preds):
            signal_scores[val_end:] = ensemble_preds
            log.info(f"集成模型 ({len(models)} 个) 已覆盖测试段信号")

    # ====== 7. 真实波动率计算 ======
    log.info("[6/9] 计算真实波动率...")
    prices_full = features['Close'].values.astype(np.float64)
    volumes_full = features['Volume'].values.astype(np.float64) if 'Volume' in features.columns else np.ones(n) * 1e6
    vol_full = compute_rolling_volatility(prices_full, window=20)

    test_prices = prices_full[val_end:]
    test_volumes = volumes_full[val_end:]
    test_volatility = vol_full[val_end:]

    log.info(f"测试期波动率: mean={test_volatility.mean():.3f}, "
             f"min={test_volatility.min():.3f}, max={test_volatility.max():.3f}")

    # ====== 8. Portfolio + Risk ======
    log.info("[7/9] 组合构建 + 风控...")
    # PortfolioConstructor: signal_score → portfolio weight
    # 在训练集上 fit（学习信号分布参数），在测试集上 transform
    train_signals = signal_scores[:train_end]
    constructor = PortfolioConstructor(
        method=cfg.get("portfolio.method", "sigmoid"),
        alpha=cfg.get("portfolio.alpha", 3.0),
        max_weight=cfg.get("position.max_single_asset_weight", 1.0),
        min_weight=-cfg.get("position.max_single_asset_weight", 1.0),
    )
    if len(train_signals) > 0 and np.any(train_signals != 0):
        constructor.fit(train_signals)

    risk_engine = RiskEngine(cfg)

    # ====== 9. RL 执行优化（在 train+val 上训练，test 上推理） ======
    rl_exec = None
    if not args.skip_rl and not args.baseline_only:
        log.info("[8/9] 训练 RL 执行优化器...")
        rl_trainer = RLTrainer(
            algorithm=cfg.get("algorithm", "PPO"),
            config=cfg.to_dict(),
        )

        # ── RL 训练数据：train+val 段（绝不使用 test 段） ──
        rl_train_sig = signal_scores[:val_end]
        rl_train_prices = prices_full[:val_end]
        rl_train_vols = volumes_full[:val_end]
        rl_train_volat = vol_full[:val_end]

        # 训练有效数据检查
        if len(rl_train_sig) >= 100 and np.any(rl_train_sig != 0):
            from layers.env.execution_env import ExecutionEnv
            rl_env = ExecutionEnv(
                signal_scores=rl_train_sig,
                target_positions=np.clip(rl_train_sig, -1, 1),
                prices=rl_train_prices,
                volumes=rl_train_vols,
                volatilities=rl_train_volat,
                config=type('Cfg', (), {
                    'initial_balance': cfg.get("initial_balance", 10000),
                    'commission': cfg.get("commission", 0.001),
                    'slippage': cfg.get("slippage", 0.0005),
                    'lambda_cost': cfg.get("reward.cost_penalty", 0.1),
                    'lambda_turnover': cfg.get("reward.turnover_penalty", 0.05),
                    'lambda_drawdown': cfg.get("reward.drawdown_penalty", 0.1),
                    'drawdown_threshold': cfg.get("reward.drawdown_threshold", 0.02),
                    'rl_window_size': cfg.get("training.window_size", 10),
                })(),
            )
            from stable_baselines3.common.vec_env import DummyVecEnv
            rl_metrics = rl_trainer.train(
                DummyVecEnv([lambda: rl_env]),
                total_timesteps=cfg.get("training.total_timesteps", 100000),
            )
            log.info(f"RL 训练完成: 最终净值={rl_metrics.get('final_portfolio_value', 0):.0f}, "
                     f"episodes={len(rl_metrics.get('episode_rewards', []))}")
            rl_exec = rl_trainer
        else:
            log.warn("RL 训练数据不足或无有效信号，跳过 RL")

    # ====== 10. 回测 ======
    log.info("[9/9] 回测对比...")

    # 测试段数据
    test_features = features.iloc[val_end:]
    test_signal_scores = signal_scores[val_end:]

    bt_config = type('Cfg', (), {
        'initial_balance': cfg.get("initial_balance", 10000),
        'commission': cfg.get("commission", 0.001),
        'slippage': cfg.get("slippage", 0.0005),
        'max_position': cfg.get("position.max_gross_exposure", 1.0),
        'max_single': cfg.get("position.max_single_asset_weight", 1.0),
        'tax': cfg.get("tax", 0.0),
    })
    engine = BacktestEngine(bt_config)

    results = {}

    # 模式 1: 纯信号
    risk_engine.reset()
    r1 = engine.run("signal_only", test_features, test_signal_scores)
    results["signal_only"] = r1["metrics"]

    if not args.baseline_only:
        # 模式 2: 信号+风控
        risk_engine.reset()
        r2 = engine.run("signal_risk", test_features, test_signal_scores,
                         risk_manager=risk_engine)
        results["signal_risk"] = r2["metrics"]

        # 模式 3: 信号+风控+RL
        if rl_exec is not None:
            risk_engine.reset()
            r3 = engine.run(
                "full", test_features, test_signal_scores,
                risk_manager=risk_engine,
                rl_executor=rl_exec,
                prices=test_prices,
                volumes=test_volumes,
                volatilities=test_volatility,
            )
            results["full_layered"] = r3["metrics"]

    # ====== 11. Walk-Forward 回测（可选） ======
    if args.walk_forward:
        log.info("[9b/9] Walk-Forward 回测...")
        wf_engine = WalkForwardBacktest(
            n_folds=cfg.get("walk_forward.n_folds", 5),
            train_window_years=cfg.get("walk_forward.train_window_years", 3),
            test_window_years=cfg.get("walk_forward.test_window_years", 1),
            mode=cfg.get("walk_forward.mode", "expanding"),
        )

        def _wf_train(train_data):
            """Walk-Forward 的训练回调"""
            train_feat = train_data[feat_cols].values.astype(np.float32)
            train_lbl = LabelBuilder(horizon=horizon).build_labels(train_data).values.astype(np.float32)
            if len(train_feat) != len(train_lbl):
                train_lbl = train_lbl[:len(train_feat)]
            s = StandardScaler().fit(train_feat)
            m = model_cls(config=model_cfg)
            m.fit(s.transform(train_feat), train_lbl)
            path = f"{cfg.get('system.model_dir', 'models')}/wf_{ticker}.pkl"
            m.save(path)
            return path

        def _wf_test(test_data, model_path):
            """Walk-Forward 的测试回调"""
            from layers.signals.base_model import BaseAlphaModel
            test_feat = test_data[feat_cols].values.astype(np.float32)
            m = model_cls()
            m.load(model_path)
            preds = m.predict(test_feat)
            ret = test_data['Close'].pct_change().shift(-horizon).values[:len(preds)]
            valid = ~(np.isnan(preds) | np.isnan(ret))
            if valid.sum() < 10:
                return {"sharpe": 0, "return": 0, "n": 0}
            from utils.helpers import annualized_sharpe, annualized_return
            nav = np.cumprod(1 + preds[:len(ret)] * ret / 10)  # 简化模拟
            return {
                "sharpe": annualized_sharpe(preds[valid] * ret[valid]),
                "return": float(nav[-1] / nav[0] - 1),
                "n": int(valid.sum()),
            }

        wf_results = wf_engine.run(features, _wf_train, _wf_test)
        wf_summary = wf_engine.summary()
        log.info(f"Walk-Forward: {wf_summary}")
        # 记录到实验管理器
        exp_mgr.update_experiment(exp_id, {"walk_forward": wf_summary})

    # ====== 12. 生成报告 ======
    log.info("[10/9] 生成报告...")
    report_gen = BacktestReport()
    report = report_gen.generate(results, ticker=ticker)

    # ====== 13. 记录实验 ======
    log.info("[11/9] 记录实验...")
    final_metrics = {}
    best_mode = "signal_risk"
    best_sharpe = -float("inf")
    for mode, m in results.items():
        final_metrics[f"{mode}_sharpe"] = m.get("sharpe_ratio", 0)
        final_metrics[f"{mode}_return"] = m.get("annualized_return", 0)
        final_metrics[f"{mode}_dd"] = m.get("max_drawdown", 0)
        if m.get("sharpe_ratio", -float("inf")) > best_sharpe:
            best_sharpe = m["sharpe_ratio"]
            best_mode = mode

    exp_mgr.complete_experiment(exp_id, {
        **final_metrics,
        "best_mode": best_mode,
        "best_sharpe": best_sharpe,
        "IC": ic.get("IC", 0),
        "RankIC": rank_ic.get("RankIC", 0),
        "top_bottom_spread": quantile.get("top_bottom_spread", 0),
        "n_train": train_end,
        "n_val": val_end - train_end,
        "n_test": n - val_end,
        "n_features": len(feat_cols),
        "horizon": horizon,
        "model": active_models[0],
    })

    # ====== 输出汇总 ======
    print("\n" + "=" * 90)
    print(f"  {ticker} 全流程回测对比 (pipeline v3)")
    print("=" * 90)
    if results:
        cols = list(results.keys())
        print(f"{'指标':<18s} " + " ".join(f"{k:>18s}" for k in cols))
        print("-" * 90)

        metric_list = [
            ("total_return", "总收益率"),
            ("annualized_return", "年化收益率"),
            ("sharpe_ratio", "夏普比率"),
            ("max_drawdown", "最大回撤"),
            ("sortino_ratio", "Sortino"),
            ("calmar_ratio", "Calmar"),
            ("win_rate", "胜率"),
            ("turnover_rate", "换手率"),
            ("total_trades", "交易次数"),
        ]
        for key, name in metric_list:
            vals = []
            for m in results.values():
                v = m.get(key, 0)
                if key in ("total_return", "annualized_return", "max_drawdown",
                           "win_rate", "turnover_rate"):
                    vals.append(f"{v*100:>17.2f}%")
                elif key == "total_trades":
                    vals.append(f"{int(v):>18d}")
                else:
                    vals.append(f"{v:>18.3f}")
            print(f"{name:<18s} " + " ".join(vals))

        print("=" * 90)
        print(f"实验 ID: {exp_id}")
        print(f"IC: {ic.get('IC', 0):.4f}  |  RankIC: {rank_ic.get('RankIC', 0):.4f}")
        print(f"最优模式: {best_mode} (Sharpe={best_sharpe:.3f})")
    else:
        print("  无回测结果")
    print("=" * 90)
    log.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
