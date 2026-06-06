"""
v3 · 风控系统 — 三组回测 + 风险报告 + Walk-Forward
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from frontend.v3_utils import (
    init_session, apply_theme, load_v3_config,
    metric_tile, section_header, empty_state, dark_figure, comparison_table,
)

init_session()
apply_theme()

st.set_page_config(page_title="风控系统", page_icon="▸", layout="wide")
st.markdown('<h1 style="font-family:Noto Serif SC,serif;font-weight:600;color:#e8e4d9;border-bottom:1px solid #252a35;padding-bottom:0.6rem"><span style="color:#c9a84c">▸</span> 风控系统 & 回测</h1>', unsafe_allow_html=True)

cfg = load_v3_config()
model_names = list(st.session_state.v3_models.keys())

with st.sidebar:
    st.markdown('<div style="font-size:0.7rem;color:#7a7570;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:0.5rem">· 模型</div>', unsafe_allow_html=True)
    if not model_names:
        empty_state("请先训练模型")
        st.stop()
    sel_model = st.selectbox("信号模型", model_names)

    st.markdown("<hr style='border-color:#252a35;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;color:#7a7570;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:0.5rem">· 风控参数</div>', unsafe_allow_html=True)
    risk_cfg = cfg.get("risk", {})
    max_pos = st.slider("最大仓位", 0.1, 2.0, float(risk_cfg.get("position",{}).get("max_gross_exposure",1.0)), 0.1)
    stop_loss = st.slider("止损阈值 (%)", -20.0, -1.0, float(risk_cfg.get("stop_loss",{}).get("threshold",-0.05))*100, 0.5)/100
    target_vol = st.slider("目标波动率", 0.05, 1.0, float(risk_cfg.get("volatility",{}).get("target_vol",0.25)), 0.05)
    max_to = st.slider("最大换手率", 0.05, 1.0, float(risk_cfg.get("turnover",{}).get("max_daily_turnover",0.3)), 0.05)
    k_impact = st.slider("冲击系数 k", 0.01, 0.5, 0.1, 0.01)

    st.markdown("<hr style='border-color:#252a35;margin:1rem 0'>", unsafe_allow_html=True)
    run_btn = st.button("▸ 运行回测对比", use_container_width=True)

if run_btn:
    with st.spinner("运行三组回测..."):
        md = st.session_state.v3_models[sel_model]
        test_preds = md["test_preds"]
        features = st.session_state.v3_features
        if features is None: st.error("数据丢失"); st.stop()
        n = len(features); vl_end = int(n*0.8)
        test_features = features.iloc[vl_end:]
        test_signals = test_preds[:len(test_features)]

        from layers.risk.risk_engine import RiskEngine
        from layers.backtest.backtest_engine import BacktestEngine

        class BtCfg: initial_balance=10000.0; commission=0.001; slippage=0.0005; max_position=max_pos

        risk_engine = RiskEngine()
        risk_engine.stop_loss = stop_loss
        risk_engine.volatility.target_vol = target_vol
        risk_engine.max_turnover = max_to
        risk_engine.cost_model.k = k_impact

        engine = BacktestEngine(BtCfg())
        results = {}

        risk_engine.reset()
        results["signal"] = engine.run("signal_only", test_features, test_signals, risk_manager=risk_engine)["metrics"]
        risk_engine.reset()
        results["+risk"] = engine.run("signal_risk", test_features, test_signals, risk_manager=risk_engine)["metrics"]

        st.session_state.v3_backtest_results = results
        st.success("回测完成")
        st.rerun()

result_data = st.session_state.v3_backtest_results

if result_data is None:
    st.info("请配置参数并点击「运行回测对比」。")
else:
    tab1, tab2 = st.tabs(["回测对比", "风险报告"])

    with tab1:
        section_header("三组对比")
        df_cmp = comparison_table(result_data)
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)

        from frontend.v3_utils import benchmark_rating
        modes = list(result_data.keys())
        c1, c2, c3 = st.columns(3)
        for i, col in enumerate([c1,c2,c3]):
            if i < len(modes):
                m = result_data[modes[i]]
                sh = m.get('sharpe_ratio', 0)
                sh_label, _ = benchmark_rating("sharpe", sh)
                dd = m.get('max_drawdown', 0)
                dd_label, _ = benchmark_rating("max_drawdown", dd)
                with col:
                    metric_tile(modes[i].upper(),
                                f"Sharpe {sh:.3f}",
                                f"{sh_label} · 收益 {m.get('annualized_return',0)*100:.2f}% · DD {dd*100:.1f}% {dd_label}",
                                accent="gold" if sh>0.8 else "steel")

        section_header("指标对比")
        fig = make_subplots(rows=1, cols=2, subplot_titles=("最大回撤 (%)", "换手率 (%)"))
        dd_vals = [result_data[m]["max_drawdown"]*100 for m in modes]
        to_vals = [result_data[m]["turnover_rate"]*100 for m in modes]
        fig.add_trace(go.Bar(x=modes, y=dd_vals, marker_color=["#c9a84c","#6b8ba4"][:len(modes)],
                              text=[f"{v:.1f}%" for v in dd_vals], textposition="auto"), row=1, col=1)
        fig.add_trace(go.Bar(x=modes, y=to_vals, marker_color=["#c9a84c","#6b8ba4"][:len(modes)],
                              text=[f"{v:.1f}%" for v in to_vals], textposition="auto"), row=1, col=2)
        st.plotly_chart(dark_figure(fig, 350), use_container_width=True)

    with tab2:
        section_header("风控设置")
        params_df = pd.DataFrame([
            {"参数":"最大仓位","值":f"{max_pos*100:.0f}%"},
            {"参数":"止损阈值","值":f"{stop_loss*100:.1f}%"},
            {"参数":"目标波动率","值":f"{target_vol*100:.0f}%"},
            {"参数":"最大换手率","值":f"{max_to*100:.0f}%"},
            {"参数":"冲击系数 k","值":f"{k_impact:.3f}"},
        ])
        st.dataframe(params_df, use_container_width=True, hide_index=True)

        section_header("成本估算")
        c1,c2,c3 = st.columns(3)
        for i,m in enumerate(modes):
            met = result_data[m]
            sh = met.get('sharpe_ratio', 0)
            sh_label, _ = benchmark_rating("sharpe", sh)
            with [c1,c2,c3][i]:
                metric_tile(m.upper(),
                            f"Sharpe {sh:.3f}",
                            f"{sh_label} · DD: {met.get('max_drawdown',0)*100:.1f}%",
                            accent="bullish" if sh>1 else ("bearish" if sh<0 else "steel"))
