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
    parser.add_argument("--skip-rl", action="store_true",
                        help="已废弃（RL 默认关闭），保留参数避免破坏旧脚本")
    parser.add_argument("--enable-rl", action="store_true",
                        help="启用 RL 执行层（日频数据通常不需要，参考系统诊断报告）")
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

    # 标准化 val 和 test（用相应 scaler）
    X_val_s = scaler_tr.transform(X_val)
    X_test_s_for_test = scaler_full.transform(X_test)

    # ── 多模型训练与集成 ──
    # 对 active_models 中每个模型，分别训练 train→val 版本和 train+val 最终版本
    log.info(f"启用模型: {active_models}")

    # 存储各模型在测试段的预测
    all_test_preds = []
    all_val_preds = []
    model_names = []

    for model_name in active_models:
        cls = model_map.get(model_name)
        if cls is None:
            log.warn(f"未知模型 {model_name}，跳过")
            continue

        cfg_model = cfg.get(model_name, {})
        log.info(f"训练模型: {model_name}")

        # train→val 版本
        m = cls(name=model_name, config=cfg_model)
        trainer = ModelTrainer(m)
        m_metrics = trainer.train(X_tr_s, y_tr, X_val_s, y_val)
        val_pred = m.predict(X_val_s)
        all_val_preds.append(val_pred)

        # train+val→test 最终版本
        m_final = cls(name=f"{model_name}_final", config=cfg_model)
        m_final.fit(X_train_val_s, np.concatenate([y_tr, y_val]))
        test_pred = m_final.predict(X_test_s_for_test)
        all_test_preds.append(test_pred)
        model_names.append(model_name)

        log.info(f"  {model_name}: val_r2={m_metrics.get('val_r2', 0):.4f}")

    # ── 集成：等权平均 ──
    n_models = len(all_test_preds)
    if n_models == 0:
        log.error("没有成功训练的模型，退出")
        return
    elif n_models == 1:
        val_preds = all_val_preds[0]
        test_preds = all_test_preds[0]
        log.info(f"单一模型: {model_names[0]}")
    else:
        # 用验证集 R2 作为集成权重
        val_r2_list = []
        for name in model_names:
            pass  # 简化处理，使用等权
        val_preds = np.mean(all_val_preds, axis=0)
        test_preds = np.mean(all_test_preds, axis=0)
        log.info(f"集成模型 ({n_models} 个): {', '.join(model_names)}")

    # ── 生成全量 signal_scores ──
    log.info("生成全量信号分数...")

    signal_scores = np.zeros(n, dtype=np.float64)

    # train 段: OOF（用第一个模型做 OOF）
    oof_preds = _oof_predict(model_map.get(active_models[0], LightGBMModel),
                             X_tr, y_tr, n_folds=5,
                             config=cfg.get(active_models[0], {}))
    signal_scores[:train_end] = oof_preds

    # val 段: 集成预测
    signal_scores[train_end:val_end] = val_preds[:val_end - train_end]

    # test 段: 集成最终预测
    signal_scores[val_end:] = test_preds[:n - val_end]

    log.info(f"信号范围: [{signal_scores.min():.4f}, {signal_scores.max():.4f}], "
             f"非零比例: {(signal_scores != 0).mean():.1%}")

    # ====== 5. Alpha 评估 + 衰减监控 ======
    log.info("[5/8] Alpha 评估 & 信号衰减监控...")

    # 初始化衰减监控器
    from layers.evaluation.decay_monitor import SignalDecayMonitor
    decay_monitor = SignalDecayMonitor(window=60, alert_threshold=0.0)
    exp_mgr.update_experiment(exp_id, {"decay_monitor": decay_monitor.summary()})

    if len(test_preds) >= 10:
        test_labels = y_test
        eval_signal = test_preds[:len(test_labels)]
        alphas = AlphaEvaluator(eval_signal, test_labels, dates=features.index[val_end:val_end + len(test_labels)])
        ic = alphas.compute_ic()
        rank_ic = alphas.compute_rank_ic()
        ic_series = alphas.compute_ic_series(window=21)
        quantile = alphas.compute_quantile_analysis(n_buckets=5)

        # 记录衰减监控
        decay_status = decay_monitor.update(
            ic_value=ic.get('IC', 0),
            date=features.index[-1] if len(features.index) > val_end else None,
        )
        log.info(f"IC={ic.get('IC', 0):.4f}(p={ic.get('IC_pvalue', 1):.4f}), "
                 f"RankIC={rank_ic.get('RankIC', 0):.4f}, "
                 f"IC_IR={ic_series.get('IC_IR', 0):.3f}")
        if quantile:
            spread = quantile.get('top_bottom_spread', 0)
            log.info(f"分位数 spread={spread:.4f}")
        if decay_status.should_retrain:
            log.warn(f"⚠️ 信号衰减警告: {decay_status.reason}")
        log.info(f"衰减监控: {decay_monitor.summary()}")
    else:
        ic, rank_ic = {"IC": 0}, {"RankIC": 0}
        quantile = {}
        log.warn("测试集样本不足，跳过 Alpha 评估")

    # ====== 6. 真实波动率计算 ======
    log.info("[6/9] 计算真实波动率...")
    prices_full = features['Close'].values.astype(np.float64)
    volumes_full = features['Volume'].values.astype(np.float64) if 'Volume' in features.columns else np.ones(n) * 1e6
    vol_full = compute_rolling_volatility(prices_full, window=20)

    test_prices = prices_full[val_end:]
    test_volumes = volumes_full[val_end:]
    test_volatility = vol_full[val_end:]

    log.info(f"测试期波动率: mean={test_volatility.mean():.3f}, "
             f"min={test_volatility.min():.3f}, max={test_volatility.max():.3f}")

    # ====== 7. Portfolio + Risk ======
    log.info("[6/8] 组合构建 + 风控...")
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

    # ====== 8. RL 执行优化（日频策略不需要，默认关闭） ======
    rl_exec = None
    _rl_enabled = args.enable_rl
    if _rl_enabled and not args.baseline_only:
        log.info("[7/8] 训练 RL 执行优化器...")
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

    # ====== 9. 回测对比 ======
    log.info("[8/8] 回测对比...")

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

        # 模式 3: 信号+风控+RL（仅当 --enable-rl 时）
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

    # ====== 10. Walk-Forward 回测（可选） ======
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
