"""
v3 · Alpha 信号评估 — IC / RankIC / 分位数 / 因子分析
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from frontend.v3_utils import (
    init_session, apply_theme, load_v3_config,
    metric_tile, section_header, empty_state, dark_figure,
)

init_session()
apply_theme()

st.set_page_config(page_title="信号评估", page_icon="▸", layout="wide")
st.markdown('<h1 style="font-family:Noto Serif SC,serif;font-weight:600;color:#e8e4d9;border-bottom:1px solid #252a35;padding-bottom:0.6rem"><span style="color:#c9a84c">▸</span> Alpha 信号评估</h1>', unsafe_allow_html=True)

model_names = list(st.session_state.v3_models.keys())
if not model_names:
    empty_state("尚未训练模型 — 请先前往「模型训练」页面")
    st.stop()

with st.sidebar:
    st.markdown('<div style="font-size:0.7rem;color:#7a7570;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:0.5rem">· 模型</div>', unsafe_allow_html=True)
    sel = st.selectbox("已训练模型", model_names)
    ic_window = st.slider("滚动 IC 窗口 (天)", 5, 60, 21)
    st.markdown("<hr style='border-color:#252a35;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;color:#7a7570;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:0.5rem">· IC Decay Horizon</div>', unsafe_allow_html=True)
    horizons = st.multiselect("Horizon", [1,3,5,10,20], default=[1,3,5,10,20])

# ---- 数据准备 ----
model_data = st.session_state.v3_models[sel]
test_preds = model_data["test_preds"]
features = st.session_state.v3_features
labels = st.session_state.v3_labels
if features is None or labels is None:
    empty_state("数据丢失，请重新训练模型")
    st.stop()

test_labels = st.session_state.v3_test_labels
if test_labels is None:
    n = len(labels)
    test_labels = labels.values.astype(np.float32)[int(n*0.8):]

min_len = min(len(test_preds), len(test_labels))
test_preds, test_labels = test_preds[:min_len], test_labels[:min_len]

from layers.evaluation.alpha_eval import AlphaEvaluator
alphas = AlphaEvaluator(test_preds, test_labels)

tab1, tab2, tab3 = st.tabs(["IC 分析", "分位数", "信号分析"])

# ---- Tab 1 ----
with tab1:
    section_header("IC / RankIC")
    ic = alphas.compute_ic()
    ric = alphas.compute_rank_ic()
    ics = alphas.compute_ic_series(window=ic_window)

    from frontend.v3_utils import benchmark_rating
    cols = st.columns(4)
    with cols[0]:
        ic_val = ic.get('IC', 0)
        ic_label, _ = benchmark_rating("IC", ic_val)
        ic_t = ic.get('IC_tstat', 0)
        ic_t_label, _ = benchmark_rating("IC_tstat", ic_t)
        metric_tile("IC", f"{ic_val:.4f}",
                    f"{ic_label} · t={ic_t:.1f} · p={ic.get('IC_pvalue',1):.3f}", "gold")
    with cols[1]:
        rolling_t = ics.get('IC_tstat', 0)
        rt_label, _ = benchmark_rating("IC_tstat", rolling_t)
        metric_tile("滚动 IC t-stat", f"{rolling_t:.2f}",
                    f"{rt_label} · 窗口={ic_window}天", "steel")
    with cols[2]:
        ric_val = ric.get('RankIC', 0)
        ric_label, _ = benchmark_rating("RankIC", ric_val)
        metric_tile("RankIC", f"{ric_val:.4f}",
                    f"{ric_label} · p={ric.get('RankIC_pvalue',1):.3f}", "gold")
    with cols[3]:
        ir_val = ics.get('IC_IR', 0)
        ir_label, _ = benchmark_rating("IC_IR", ir_val)
        metric_tile("IC IR", f"{ir_val:.3f}",
                    f"{ir_label} · IC均值/标准差", "steel")

    if "IC_series" in ics and ics["IC_series"]:
        ic_arr = np.array(ics["IC_series"])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=("滚动 IC 时序", "IC 分布"),
                             vertical_spacing=0.12, row_heights=[0.6,0.4])
        fig.add_trace(go.Scatter(y=ic_arr, mode="lines", name="IC",
                                  line=dict(color="#c9a84c", width=1)), row=1, col=1)
        ma = pd.Series(ic_arr).rolling(20).mean()
        fig.add_trace(go.Scatter(y=ma, mode="lines", name="MA20",
                                  line=dict(color="#e74c3c", width=2)), row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="#3d5363", row=1, col=1)
        fig.add_trace(go.Histogram(x=ic_arr, nbinsx=30, marker_color="#6b8ba4"), row=2, col=1)
        st.plotly_chart(dark_figure(fig, 500), use_container_width=True)

    # IC Decay
    if features is not None and 'Close' in features.columns:
        close = features['Close']
        test_idx = features.index[int(len(features)*0.8):]
        decay_data = {}
        for h in [1,3,5,10,20]:
            fwd = close.shift(-h)/close - 1
            fwd_test = fwd.loc[test_idx]
            min_l = min(len(test_preds), len(fwd_test.dropna()))
            if min_l > 5:
                from scipy.stats import pearsonr
                c,_ = pearsonr(test_preds[:min_l], fwd_test.dropna().values[:min_l])
                decay_data[h] = c
        if decay_data:
            section_header("IC Decay")
            fig2 = go.Figure(go.Scatter(
                x=list(decay_data.keys()), y=list(decay_data.values()),
                mode="lines+markers", marker=dict(size=8, color="#c9a84c"),
                line=dict(width=2)))
            fig2.add_hline(y=0, line_dash="dash", line_color="#3d5363")
            fig2.update_layout(xaxis_title="预测天数", yaxis_title="IC")
            st.plotly_chart(dark_figure(fig2, 350), use_container_width=True)

# ---- Tab 2 ----
with tab2:
    section_header("分位数分析")
    qa = alphas.compute_quantile_analysis(n_buckets=5)
    if qa:
        buckets = [qa.get(f"Q{i+1}_return",0) for i in range(5)]
        labels_q = ["Q1 最弱","Q2","Q3","Q4","Q5 最强"]
        colors_bar = ["#e74c3c","#e67e22","#f39c12","#2ecc71","#27ae60"]
        fig = go.Figure(go.Bar(x=labels_q, y=buckets, marker_color=colors_bar,
                                text=[f"{v*100:.2f}%" for v in buckets], textposition="auto"))
        fig.add_hline(y=0, line_dash="dash", line_color="#3d5363")
        st.plotly_chart(dark_figure(fig, 380), use_container_width=True)
        c1,c2 = st.columns(2)
        with c1: metric_tile("Top-Bottom Spread", f"{qa.get('top_bottom_spread',0)*100:.2f}%",
                              "多空收益差", "bullish" if qa.get('top_bottom_spread',0)>0.01 else "neutral")
        with c2: metric_tile("信号强度", "强" if abs(qa.get('top_bottom_spread',0))>0.02 else "弱",
                              f"|spread| = {abs(qa.get('top_bottom_spread',0))*100:.2f}%")

# ---- Tab 3 ----
with tab3:
    section_header("信号分析")
    to = alphas.compute_turnover()
    c1,c2 = st.columns(2)
    with c1: metric_tile("信号换手率", f"{to:.4f}", "日均变化率")
    with c2: metric_tile("样本数", str(alphas.n))

    fig = go.Figure(go.Histogram(x=test_preds, nbinsx=50, marker_color="#6b8ba4"))
    fig.add_vline(x=0, line_dash="dash", line_color="#3d5363")
    fig.update_layout(title="信号分布")
    st.plotly_chart(dark_figure(fig, 350), use_container_width=True)

    from scipy import stats as sp_stats
    slope, intercept, r, _, _ = sp_stats.linregress(test_preds, test_labels)
    x_line = np.linspace(test_preds.min(), test_preds.max(), 100)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=test_preds, y=test_labels, mode="markers",
                               marker=dict(size=4, color="#6b8ba4", opacity=0.4)))
    fig2.add_trace(go.Scatter(x=x_line, y=slope*x_line+intercept,
                               mode="lines", name=f"r={r:.3f}",
                               line=dict(color="#c9a84c", width=2)))
    fig2.update_layout(title="信号 vs 前向收益", xaxis_title="signal_score", yaxis_title="前向收益")
    st.plotly_chart(dark_figure(fig2, 400), use_container_width=True)
