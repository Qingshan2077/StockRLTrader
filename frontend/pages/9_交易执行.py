"""
v3 · RL 交易执行（已废弃）

⚠️ 日频策略不需要 RL 执行层（PPO/SAC）。
   需 10,000+ episodes 才能收敛，日频数据只有 ~2500 个点。
   保留此页面用于实验性探索，需要通过 --enable-rl 启用。

参考: docs/系统诊断报告.md
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.v3_utils import (
    init_session, apply_theme, load_v3_config,
    metric_tile, section_header, empty_state, dark_figure,
)

init_session()
apply_theme()

st.set_page_config(page_title="交易执行", page_icon="▸", layout="wide")

st.markdown("""
<div style="background:#2d2121;border:1px solid #5a3535;border-radius:6px;padding:1rem 1.2rem;margin-bottom:1.5rem">
<p style="margin:0;color:#e8b4b4;font-size:0.9rem">
<strong>⚠️ RL 执行层已废弃</strong> — PPO/SAC 在日频策略上科学上不可行（需 10,000+ episodes，日频仅 ~2,500 点）。
此页面仅作实验保留，修改信号模型请使用「模型训练」页面。
</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<h1 style="font-family:Noto Serif SC,serif;font-weight:600;color:#e8e4d9;border-bottom:1px solid #252a35;padding-bottom:0.6rem"><span style="color:#c9a84c">▸</span> 交易执行 <span style="font-size:0.6rem;color:#7a7570;background:#1c2028;padding:0.15rem 0.6rem;border-radius:3px;margin-left:0.6rem">已废弃</span></h1>', unsafe_allow_html=True)

model_names = list(st.session_state.v3_models.keys())
with st.sidebar:
    st.markdown('<div style="font-size:0.7rem;color:#7a7570;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:0.5rem">· 信号模型</div>', unsafe_allow_html=True)
    if not model_names:
        empty_state("请先训练信号模型")
        st.stop()
    sel_model = st.selectbox("信号模型", model_names)

    st.markdown("<hr style='border-color:#252a35;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;color:#7a7570;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:0.5rem">· RL 参数（实验性）</div>', unsafe_allow_html=True)
    algorithm = st.selectbox("算法", ["PPO","SAC"], index=0)
    timesteps = st.slider("训练步数", 5000, 200000, 50000, 5000)

    st.markdown("<hr style='border-color:#252a35;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;color:#7a7570;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:0.5rem">· 奖励权重</div>', unsafe_allow_html=True)
    w_pnl = st.slider("λ PnL", 0.0, 3.0, 1.0, 0.1)
    w_cost = st.slider("λ Cost", 0.0, 1.0, 0.1, 0.01)
    w_to = st.slider("λ Turnover", 0.0, 1.0, 0.05, 0.01)
    w_dd = st.slider("λ Drawdown", 0.0, 1.0, 0.1, 0.01)

    train_btn = st.button("▸ 训练 RL Agent", use_container_width=True, disabled=True)
    if train_btn:
        st.warning("RL 训练已禁用。请使用命令行运行: python run_pipeline.py --enable-rl")

tab1, tab2, tab3 = st.tabs(["训练监控", "奖励分析", "交易信号"])

# 检查是否有已训练好的 RL 执行器（通过命令行 --enable-rl 传入）
rl_available = st.session_state.get("v3_rl_executor") is not None

with tab1:
    st.info("""
    RL 训练已默认禁用。如需实验 RL 执行层：
    1. `python run_pipeline.py --ticker AAPL --enable-rl`
    2. 回到此页面查看训练结果
    """)

with tab2:
    section_header("奖励权重配置")
    w_df = pd.DataFrame([
        {"分量":"PnL","权重":w_pnl},
        {"分量":"Cost","权重":w_cost},
        {"分量":"Turnover","权重":w_to},
        {"分量":"Drawdown","权重":w_dd},
    ])
    st.dataframe(w_df, use_container_width=True, hide_index=True)
    st.caption("奖励 = PnL - λ₁·Cost - λ₂·Turnover - λ₃·Drawdown")

with tab3:
    section_header("当前交易信号")
    if sel_model in st.session_state.v3_models:
        preds = st.session_state.v3_models[sel_model]["test_preds"]
        latest = float(preds[-1]) if len(preds)>0 else 0.0
        c1,c2,c3 = st.columns(3)
        direction = "看涨" if latest>0.05 else ("看跌" if latest<-0.05 else "观望")
        daccent = "bullish" if latest>0.05 else ("bearish" if latest<-0.05 else "neutral")
        with c1: metric_tile("Signal Score", f"{latest:.4f}", direction, daccent)
        with c2:
            tp = float(np.clip(latest*10,-1,1))
            metric_tile("目标仓位", f"{tp*100:.1f}%",
                        "全仓" if tp>0.8 else ("清仓" if tp<-0.8 else ""),
                        "bullish" if tp>0 else "bearish")
        with c3:
            if rl_available:
                metric_tile("执行比例", "RL 已就绪", "RL 优化中", "gold")
            else:
                metric_tile("执行比例", "默认100%", "RL 未启用(默认跳过)", "steel")
    else:
        st.info("需要先训练信号模型。")
