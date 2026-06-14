"""
A 股截面多因子选股管线 — Phase 2 入口

流程:
  1. 加载股票池 (300 只 A 股, 含行业)
  2. 批量下载数据 + 计算特征 → MultiIndex 面板
  3. 截面因子处理 (winsorize → 中性化 → z-score)
  4. 训练截面排名模型 (LightGBM/Ridge)
  5. Alpha 评估 (IC, 分位数)
  6. 截面多空回测

用法:
    python run_ashare_cross.py                              # 200 只, 默认参数
    python run_ashare_cross.py --quick                       # 50 只, ridge (快速验证)
    python run_ashare_cross.py --size 300 --model lightgbm   # 完整版
    python run_ashare_cross.py --skip-train                  # 跳过训练, 加载已有模型

依赖:
    pip install akshare lightgbm scikit-learn pandas numpy
"""

import argparse
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from utils.config_loader import ConfigLoader
from utils.logging_system import LoggingSystem
from utils.helpers import set_seed

from layers.data.ashare_provider import AShareDataProvider
from layers.data.ashare_universe import get_sw_sector, _SECTOR_MAP
from layers.features import FeaturePipeline, FeatureRegistry, FeatureCache
from layers.features.cross_sectional import process_panel_factors
from layers.signals.cross_sectional_model import CrossSectionalRanker
from layers.evaluation.factor_analyzer import FactorAnalyzer
from layers.backtest.cross_sectional_backtest import CrossSectionalBacktest

import warnings
warnings.filterwarnings("ignore")

PRICE_COLS = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
CACHE_DIR = "stock_data/ashare_cache"
RESULT_DIR = "stock_data/cross_sectional_results"


def build_panel(universe_size=200, start_date="2018-01-01",
                force_download=False, verbose=True):
    """构建截面面板数据 (date, code) × features"""
    log = LoggingSystem()
    log.info("📋 获取股票池...")
    all_codes = sorted(_SECTOR_MAP.keys())
    universe = all_codes[:universe_size]
    log.info(f"  股票池: {len(universe)} 只 (静态池)")

    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)

    all_chunks, failed = [], 0
    benchmark_data = None
    t0 = time.time()

    for i, code in enumerate(universe):
        try:
            provider = AShareDataProvider(
                ticker=code, data_dir=str(cache_dir),
                start_date=start_date, cache_format="parquet",
            )
            df = provider.load_or_download()
            if df is None or len(df) < 120:
                failed += 1; continue

            if benchmark_data is None:
                benchmark_data = provider.get_benchmark()
            registry = FeatureRegistry()
            cache = FeatureCache(data_dir=str(cache_dir))
            pipeline = FeaturePipeline(
                registry=registry, cache=cache, benchmark_data=benchmark_data,
            )
            features = pipeline.build_features(df, ticker=code, use_cache=not force_download)
            if features is None or len(features) < 60:
                failed += 1; continue

            features["code"] = code
            features["sector"] = get_sw_sector(code)
            features = features.reset_index()
            features["date"] = pd.to_datetime(features["Date"])
            features.set_index(["date", "code"], inplace=True)
            features.drop(columns=["Date"], inplace=True, errors="ignore")
            all_chunks.append(features)

        except Exception:
            failed += 1; continue

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            log.info(f"  [{i+1}/{len(universe)}] 已加载 {len(all_chunks)} 只, 跳过 {failed}")

    if not all_chunks:
        raise RuntimeError("没有成功加载任何股票数据")
    panel = pd.concat(all_chunks).sort_index()
    log.info(f"✅ 面板完成: {len(panel)} 行, "
             f"{len(panel.index.get_level_values('date').unique())} 交易日, "
             f"{len(panel.index.get_level_values('code').unique())} 只股票")
    return panel


def prepare_labels(panel, horizon=5):
    """准备截面标签: label_{horizon}d + label_1d + label_rank"""
    result = panel.copy()
    for code in result.index.get_level_values("code").unique():
        mask = result.index.get_level_values("code") == code
        close = result.loc[mask, "Close"].sort_index()
        if len(close) < horizon + 5:
            continue
        # 5 日 forward return (训练目标)
        fwd_ret = close.shift(-horizon) / close - 1
        result.loc[mask, f"label_{horizon}d"] = fwd_ret.values
        # 1 日 forward return (回测收益)
        ret_1d = close.shift(-1) / close - 1
        result.loc[mask, "label_1d"] = ret_1d.values
    result = result.dropna(subset=[f"label_{horizon}d"])

    for dt in result.index.get_level_values("date").unique():
        mask = result.index.get_level_values("date") == dt
        sub = result.loc[mask, f"label_{horizon}d"]
        rnk = sub.rank(method="average", ascending=True)
        result.loc[mask, "label_rank"] = (rnk - 1) / max(len(rnk) - 1, 1)
    return result


def time_split(panel, train_ratio=0.6, val_ratio=0.2):
    dates = panel.index.get_level_values("date").unique().sort_values()
    n = len(dates)
    te, ve = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    train = panel.loc[panel.index.get_level_values("date").isin(dates[:te])]
    val = panel.loc[panel.index.get_level_values("date").isin(dates[te:ve])]
    test = panel.loc[panel.index.get_level_values("date").isin(dates[ve:])]
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="StockTrader — A 股截面多因子选股管线")
    parser.add_argument("--size", type=int, default=200, help="股票池大小")
    parser.add_argument("--start", default="2018-01-01", help="数据起始日期")
    parser.add_argument("--horizon", type=int, default=5, help="预测窗口 (天)")
    parser.add_argument("--model", default="lightgbm", choices=["lightgbm", "ridge"], help="模型类型")
    parser.add_argument("--long", type=int, default=30, help="做多股票数量")
    parser.add_argument("--short", type=int, default=15, help="做空股票数量 (0=纯多)")
    parser.add_argument("--force-download", action="store_true", help="强制重新下载")
    parser.add_argument("--skip-data", action="store_true", help="跳过数据加载 (用缓存)")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练, 加载已保存模型")
    parser.add_argument("--quick", action="store_true", help="快速模式: 50 只, ridge")
    args = parser.parse_args()

    if args.quick:
        args.size = min(args.size, 50)
        args.model = "ridge"
        args.long = 10
        args.short = 5

    cfg = ConfigLoader()
    log = LoggingSystem()
    set_seed(cfg.get("system.seed", 42))

    log.info("🚀 StockTrader — 截面多因子选股管线 (Phase 2)")
    log.info(f"  股票池: {args.size} 只, 窗口: {args.horizon} 日, "
             f"模型: {args.model}, 多空: {args.long}/{args.short}")

    model_path = f"{RESULT_DIR}/model.pkl"

    # ════════ 1. 面板数据 ════════
    panel_file = f"{RESULT_DIR}/panel_{args.size}_{args.start[:4]}.pkl"
    if args.skip_data and Path(panel_file).exists():
        log.info("[1/5] 从缓存加载面板...")
        panel = pd.read_pickle(panel_file)
    else:
        log.info("[1/5] 构建面板数据...")
        panel = build_panel(args.size, args.start, args.force_download)
        pd.to_pickle(panel, panel_file)

    n_stocks = len(panel.index.get_level_values("code").unique())
    n_dates = len(panel.index.get_level_values("date").unique())
    log.info(f"  股票数: {n_stocks}, 交易日: {n_dates}")

    # ════════ 2. 标签 ════════
    log.info("[2/5] 准备截面标签...")
    panel = prepare_labels(panel, horizon=args.horizon)

    # ════════ 3. 截面因子处理 ════════
    log.info("[3/5] 截面因子处理...")
    meta_cols = {"Open", "Close", "High", "Low", "Volume", "Amount",
                 "sector", f"label_{args.horizon}d", "label_1d", "label_rank", "Adj Close"}
    factor_cols = [c for c in panel.columns if c not in meta_cols]
    panel = process_panel_factors(panel, factor_cols, sector_col="sector",
                                  do_sector_neutral=True, method="mad", n_mad=5.0)
    feat_cols = factor_cols
    log.info(f"  有效因子: {len(feat_cols)}")

    # ════════ 4. 模型 ════════
    if args.skip_train:
        log.info("[4/5] 加载已训练的模型...")
        if not Path(model_path).exists():
            log.error(f"  模型文件不存在: {model_path}, 请先跑一次完整训练")
            return
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        model = saved["model"]
        feat_cols = saved.get("feat_cols", feat_cols)
    else:
        log.info("[4/5] 训练截面模型...")
        model = CrossSectionalRanker(model_type=args.model, label_horizon=args.horizon)

    train, val, test = time_split(panel)
    log.info(f"  切分: train={len(train)}, val={len(val)}, test={len(test)}")

    if not args.skip_train:
        metrics = model.fit(train, feat_cols, label_col="label_rank", val_panel=val)
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "feat_cols": feat_cols}, f)

    test_pred = model.predict_rank(test, feat_cols)
    imp = model.feature_importance()
    if imp is not None and len(imp) > 0:
        log.info("  Top 10 因子:")
        for _, row in imp.head(10).iterrows():
            log.info(f"    {row['feature']}: {row['importance']:.2f}")

    # ════════ 5. Alpha 评估 ════════
    log.info("[5/5] 分析评估...\n")
    analyzer = FactorAnalyzer(test_pred)
    ics = analyzer.compute_factor_ics(["prediction"], f"label_{args.horizon}d")
    ic_summary = analyzer.factor_ic_summary()
    log.info(f"📈 预测得分 IC 分析 (测试集)")
    log.info(f"  IC 均值:   {ic_summary.loc['prediction', 'IC_mean']:.4f}")
    log.info(f"  IC 标准差: {ic_summary.loc['prediction', 'IC_std']:.4f}")
    log.info(f"  IC_IR:     {ic_summary.loc['prediction', 'IC_IR']:.3f}")
    log.info(f"  IC 胜率:   {ic_summary.loc['prediction', 'IC_positive_pct']*100:.1f}%")

    try:
        quantile = analyzer.quantile_analysis("prediction", f"label_{args.horizon}d")
        log.info(f"\n📊 分位数分层分析:")
        for b in quantile.index:
            log.info(f"  {b}: 收益={quantile.loc[b,'mean_return']*100:+.2f}%, "
                     f"夏普={quantile.loc[b,'sharpe']:.2f}")
    except Exception as e:
        log.warn(f"  分位数分析跳过: {e}")

    # ════════ 6. 截面多空回测 ════════
    log.info(f"\n{'='*50}\n📊 截面多空回测\n{'='*50}")
    bt = CrossSectionalBacktest(
        long_n=args.long, short_n=args.short, weight_scheme="equal",
        commission=0.0003, slippage=0.0005, max_sector_weight=0.3,
    )
    result = bt.run(test_pred, rank_col="rank_pct",
                    return_col=f"label_1d", sector_col="sector")

    # 保存结果 (供前端看板使用)
    result.nav.to_frame(name="nav").to_parquet(f"{RESULT_DIR}/nav.parquet")
    result.daily_returns.to_frame(name="daily_return").to_parquet(f"{RESULT_DIR}/daily_returns.parquet")
    result.positions.to_parquet(f"{RESULT_DIR}/positions.parquet")
    if imp is not None and len(imp) > 0:
        imp.to_parquet(f"{RESULT_DIR}/feature_importance.parquet")
    if ic_summary is not None and len(ic_summary) > 0:
        ic_summary.to_parquet(f"{RESULT_DIR}/ic_summary.parquet")

    # 保存完整预测结果 (给看板用)
    test_pred.to_parquet(f"{RESULT_DIR}/test_pred_full.parquet")

    log.info(f"\n✅ 截面管线结束! 结果目录: {RESULT_DIR}/")
    log.info(f"  前端看板: streamlit run frontend/app.py → 第12/13页")
    log.info(f"  下次加 --skip-data --skip-train 跳过数据加载和训练")


if __name__ == "__main__":
    main()
