"""
v3 · 全流程管线 — 一键执行 + 进度可视化
"""
import numpy as np
import pandas as pd
import streamlit as st

from frontend.v3_utils import (
    init_session, apply_theme, load_v3_config, get_available_stocks,
    load_stock_data, load_features,
    metric_tile, section_header, empty_state, comparison_table,
)

init_session()
apply_theme()

st.set_page_config(page_title="全流程管线", page_icon="▸", layout="wide")
st.markdown('<h1 style="font-family:Noto Serif SC,serif;font-weight:600;color:#e8e4d9;border-bottom:1px solid #252a35;padding-bottom:0.6rem"><span style="color:#c9a84c">▸</span> 全流程管线</h1>', unsafe_allow_html=True)
st.caption("数据 → 特征 → 模型 → 回测 → 评估 — 一键执行")

c1, c2 = st.columns([1, 2])

with c1:
    st.markdown('<div style="font-size:0.7rem;color:#7a7570;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:0.5rem">· 配置</div>', unsafe_allow_html=True)
    stocks = get_available_stocks()
    ticker = st.text_input("股票代码", value=stocks[0] if stocks else "AAPL")

    st.markdown('<div style="font-size:0.7rem;color:#7a7570;letter-spacing:0.8px;text-transform:uppercase;margin:1rem 0 0.5rem">· 步骤</div>', unsafe_allow_html=True)
    do_features = st.checkbox("特征工程", value=True)
    do_signal = st.checkbox("Alpha 模型 (LightGBM)", value=True)
    do_backtest = st.checkbox("风控回测对比", value=True)
    do_rl = st.checkbox("RL 执行（已废弃，需 --enable-rl）", value=False)

    horizon = st.slider("预测天数", 1, 20, 5, key="pipe_h")
    run_btn = st.button("▸ 执行全流程", use_container_width=True, type="primary")

if run_btn:
    steps = []
    if do_features: steps.append("特征工程")
    if do_signal: steps.append("Alpha模型")
    if do_backtest: steps.append("风控回测")
    if do_rl: steps.append("RL训练")
    total = len(steps)
    progress = st.progress(0)
    log = st.empty()
    completed = 0
    pipeline_results = {}

    if do_features:
        log.info(f"[{completed+1}/{total}] 特征工程...")
        df = load_stock_data(ticker)
        if df is None or df.empty:
            log.error(f"无法加载 {ticker}"); st.stop()
        result = load_features(ticker, df)
        if result is None: st.stop()
        features, labels, _ = result
        st.session_state.v3_features = features
        st.session_state.v3_labels = labels
        completed += 1; progress.progress(completed/total)
        log.success(f"[{completed}/{total}] 特征工程完成: {features.shape}")

    if do_signal:
        log.info(f"[{completed+1}/{total}] 训练 LightGBM...")
        features = st.session_state.v3_features
        labels = st.session_state.v3_labels
        fc = [c for c in features.columns if c not in ('Open','High','Low','Close','Volume','Adj Close')]
        X_all = features[fc].values.astype(np.float32)
        y_all = labels.values.astype(np.float32)
        n = len(X_all); tr_e = int(n*0.6); vl_e = int(n*0.8)
        from sklearn.preprocessing import StandardScaler
        scl = StandardScaler().fit(X_all[:tr_e])
        Xtr = scl.transform(X_all[:tr_e])
        Xva = scl.transform(X_all[tr_e:vl_e])
        Xte = scl.transform(X_all[vl_e:])
        from layers.signals.models.lightgbm_model import LightGBMModel
        m = LightGBMModel(name="lightgbm", config={})
        met = m.fit(Xtr, y_all[:tr_e], Xva, y_all[tr_e:vl_e])
        preds = m.predict(Xte)
        st.session_state.v3_models["lightgbm"] = {
            "model": m, "metrics": met, "test_preds": preds,
            "train_time": 0.0, "scaler": scl,
        }
        st.session_state.v3_test_features = features.iloc[vl_e:]
        st.session_state.v3_test_labels = y_all[vl_e:]
        completed += 1; progress.progress(completed/total)
        log.success(f"[{completed}/{total}] R² = {met.get('val_r2',0):.4f}")

    if do_backtest:
        log.info(f"[{completed+1}/{total}] 回测对比...")
        test_preds = st.session_state.v3_models["lightgbm"]["test_preds"]
        test_features = st.session_state.v3_test_features
        test_signals = test_preds[:len(test_features)]
        from layers.risk.risk_engine import RiskEngine
        from layers.backtest.backtest_engine import BacktestEngine
        class BtCfg: initial_balance=10000.0; commission=0.001; slippage=0.0005; max_position=1.0
        re = RiskEngine(); eng = BacktestEngine(BtCfg())
        results = {}
        re.reset(); results["signal"] = eng.run("signal_only",test_features,test_signals,risk_manager=re)["metrics"]
        re.reset(); results["+risk"] = eng.run("signal_risk",test_features,test_signals,risk_manager=re)["metrics"]
        pipeline_results = results
        completed += 1; progress.progress(completed/total)
        log.success(f"[{completed}/{total}] 回测完成")

    if do_rl:
        log.info(f"[{completed+1}/{total}] RL (简化)...")
        completed += 1; progress.progress(completed/total)
        log.success(f"[{completed}/{total}] 请用「交易执行」页面完整训练")

    progress.empty(); log.empty()

    if pipeline_results:
        st.success(f"全流程完成 — {completed}/{total} 步")
        section_header("回测对比")
        st.dataframe(comparison_table(pipeline_results), use_container_width=True, hide_index=True)
        best = max(pipeline_results, key=lambda m: pipeline_results[m].get("sharpe_ratio",0))
        st.info(f"最优模式: **{best}** (夏普={pipeline_results[best].get('sharpe_ratio',0):.3f})")
        if st.button("▸ 保存实验"):
            try:
                from layers.experiments.manager import ExperimentManager
                mgr = ExperimentManager()
                eid = mgr.create_experiment(name=f"pipeline_{ticker}", config={"ticker":ticker,"horizon":horizon}, tags=[ticker,"pipeline"])
                mgr.complete_experiment(eid, {f"{m}_sharpe":pipeline_results[m].get("sharpe_ratio",0) for m in pipeline_results})
                st.success(f"已保存: {eid}")
            except Exception as e:
                st.warning(f"保存失败: {e}")

with c2:
    if not run_btn:
        st.markdown('<div style="font-size:0.7rem;color:#7a7570;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:0.5rem">· 流程说明</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="color:#b0aa9e;font-family:Inter,sans-serif;font-size:0.9rem;line-height:1.8">
        <strong style="color:#c9a84c">特征工程</strong> ── 原始OHLCV → 50+ 技术指标<br>
        <strong style="color:#c9a84c">Alpha模型</strong> ── LightGBM 预测未来收益<br>
        <strong style="color:#c9a84c">风控回测</strong> ── 三组对比 (纯信号 vs +风控)<br>
        <strong style="color:#c9a84c">RL执行</strong> ── 强化学习优化调仓节奏
        </div>""", unsafe_allow_html=True)

        section_header("可用股票")
        available = get_available_stocks()
        if available:
            badges = " ".join(
                f'<span style="display:inline-block;background:#1e2330;color:#b0aa9e;padding:3px 10px;border-radius:2px;font-family:JetBrains Mono,monospace;font-size:0.72rem;margin:2px">{t}</span>'
                for t in available[:20]
            )
            st.markdown(badges, unsafe_allow_html=True)
            if len(available) > 20:
                st.caption(f"... 共 {len(available)} 只")
