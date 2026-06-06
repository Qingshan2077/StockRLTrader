"""
v3 · RL 交易执行 — 训练监控 + 奖励分析 + 信号
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
st.markdown('<h1 style="font-family:Noto Serif SC,serif;font-weight:600;color:#e8e4d9;border-bottom:1px solid #252a35;padding-bottom:0.6rem"><span style="color:#c9a84c">▸</span> RL 交易执行</h1>', unsafe_allow_html=True)

model_names = list(st.session_state.v3_models.keys())
with st.sidebar:
    st.markdown('<div style="font-size:0.7rem;color:#7a7570;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:0.5rem">· 信号模型</div>', unsafe_allow_html=True)
    if not model_names:
        empty_state("请先训练信号模型")
        st.stop()
    sel_model = st.selectbox("信号模型", model_names)

    st.markdown("<hr style='border-color:#252a35;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;color:#7a7570;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:0.5rem">· RL 参数</div>', unsafe_allow_html=True)
    algorithm = st.selectbox("算法", ["PPO","SAC"], index=0)
    timesteps = st.slider("训练步数", 5000, 200000, 50000, 5000)

    st.markdown("<hr style='border-color:#252a35;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;color:#7a7570;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:0.5rem">· 奖励权重</div>', unsafe_allow_html=True)
    w_pnl = st.slider("λ PnL", 0.0, 3.0, 1.0, 0.1)
    w_cost = st.slider("λ Cost", 0.0, 1.0, 0.1, 0.01)
    w_to = st.slider("λ Turnover", 0.0, 1.0, 0.05, 0.01)
    w_dd = st.slider("λ Drawdown", 0.0, 1.0, 0.1, 0.01)

    train_btn = st.button("▸ 训练 RL Agent", use_container_width=True)

tab1, tab2, tab3 = st.tabs(["训练监控", "奖励分析", "交易信号"])

if train_btn:
    with st.spinner("训练 RL Agent (可能需要几分钟)..."):
        md = st.session_state.v3_models[sel_model]
        test_preds = md["test_preds"]
        features = st.session_state.v3_features
        if features is None: st.error("数据丢失"); st.stop()
        n = len(features); vl_end = int(n*0.8)
        test_features = features.iloc[vl_end:]
        test_signals = test_preds[:len(test_features)]

        prices = test_features['Close'].values
        volumes = test_features['Volume'].values if 'Volume' in test_features.columns else np.ones(len(test_features))*1e6
        volatilities = test_features['vol_20d'].values if 'vol_20d' in test_features.columns else np.full(len(test_features),0.2)

        from layers.rl.trainer import RLTrainer
        from layers.env.execution_env import ExecutionEnv
        from stable_baselines3.common.vec_env import DummyVecEnv

        class EnvCfg:
            initial_balance=10000.0; commission=0.001; slippage=0.0005
            lambda_cost=w_cost; lambda_turnover=w_to; lambda_drawdown=w_dd
            drawdown_threshold=0.02; rl_window_size=10

        env = ExecutionEnv(test_signals, np.clip(test_signals,-1,1),
                            prices, volumes, volatilities, EnvCfg())
        trainer = RLTrainer(algorithm=algorithm, config={"ppo":{},"sac":{}})
        vec_env = DummyVecEnv([lambda: env])

        try:
            metrics = trainer.train(vec_env, total_timesteps=timesteps)
            st.session_state.v3_rl_executor = trainer
            with tab1:
                st.success(f"训练完成")
                if "episode_rewards" in metrics and metrics["episode_rewards"]:
                    er = metrics["episode_rewards"]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=er, mode="lines", name="Reward",
                                              line=dict(color="#c9a84c",width=1)))
                    ma = pd.Series(er).rolling(max(1,len(er)//10)).mean()
                    fig.add_trace(go.Scatter(y=ma, mode="lines", name="MA",
                                              line=dict(color="#e74c3c",width=2)))
                    st.plotly_chart(dark_figure(fig, 400), use_container_width=True)
                c1,c2 = st.columns(2)
                with c1: metric_tile("平均执行比例", f"{metrics.get('action_mean',0):.3f}", "execution_ratio")
                with c2: metric_tile("最终资产", f"${metrics.get('final_portfolio_value','--')}", accent="gold")
        except Exception as e:
            st.error(f"训练失败: {e}")

with tab2:
    if st.session_state.v3_rl_executor is None:
        st.info("请先训练 RL 模型。奖励分解将在训练时记录。")
    else:
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
            rl = st.session_state.v3_rl_executor
            metric_tile("执行比例", "RL 已就绪" if rl else "--",
                        "RL 优化中" if rl else "需训练 RL", "gold" if rl else "steel")
    else:
        st.info("需要先训练信号模型。")
