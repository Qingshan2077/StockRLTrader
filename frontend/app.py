import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from improved_data_engine import DataEngine, BatchDataEngine
from predictor import ProbabilityPredictor
import json
from datetime import datetime

# 页面配置
st.set_page_config(
    page_title="AI股票交易助手",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 600;
    }
    .stButton>button:hover {
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }
</style>
""", unsafe_allow_html=True)

# 初始化 session state
if 'data_engine' not in st.session_state:
    st.session_state.data_engine = None
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = None


# 加载配置
@st.cache_data
def load_config():
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {"data": {"directory": "stock_data"}}


config = load_config()

# 侧边栏
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/stocks.png", width=80)
    st.markdown("## 🤖 AI股票交易助手")
    st.markdown("---")

    # 股票选择
    st.markdown("### 📊 选择股票")

    # 获取可用股票列表
    batch_engine = BatchDataEngine(data_dir=config['data']['directory'])
    available_tickers = batch_engine.list_available_data()

    if available_tickers:
        # 创建显示选项（包含自定义名称）
        ticker_options = {}
        ticker_display_list = []

        for t in available_tickers:
            engine_temp = DataEngine(t, data_dir=config['data']['directory'])
            custom_name = engine_temp.get_custom_name()
            if custom_name:
                display_name = f"{custom_name} ({t})"
            else:
                display_name = t
            ticker_options[display_name] = t
            ticker_display_list.append(display_name)

        selected_display = st.selectbox(
            "股票代码",
            ticker_display_list,
            index=0,
            key="ticker_select"
        )
        ticker = ticker_options[selected_display]
    else:
        st.warning("暂无数据，请先下载股票数据")
        ticker = st.text_input("输入股票代码", "AAPL")

    # 快速操作按钮
    st.markdown("### ⚡ 快速操作")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 加载数据", use_container_width=True):
            with st.spinner(f"正在加载 {ticker} 的数据..."):
                try:
                    engine = DataEngine(ticker, data_dir=config['data']['directory'])
                    df = engine.load_processed_data()
                    if df is not None and not df.empty:
                        st.session_state.data_engine = engine
                        st.session_state.current_ticker = ticker
                        st.session_state.data = df
                        st.success(f"✅ 成功加载 {len(df)} 条数据")
                    else:
                        st.error("数据为空，请先下载")
                except Exception as e:
                    st.error(f"加载失败: {e}")

    with col2:
        if st.button("🔄 更新数据", use_container_width=True):
            with st.spinner(f"正在更新 {ticker}..."):
                try:
                    engine = DataEngine(ticker, data_dir=config['data']['directory'])
                    engine.fetch_data()
                    df = engine.add_technical_indicators()
                    st.session_state.data_engine = engine
                    st.session_state.current_ticker = ticker
                    st.session_state.data = df
                    st.success("✅ 更新完成")
                except Exception as e:
                    st.error(f"更新失败: {e}")

    if st.button("🎯 训练模型", use_container_width=True):
        if st.session_state.data is not None:
            with st.spinner("正在训练预测模型..."):
                try:
                    predictor = ProbabilityPredictor(st.session_state.data)
                    predictor.create_targets()
                    predictor.train()
                    st.session_state.predictor = predictor
                    st.success("✅ 模型训练完成")
                except Exception as e:
                    st.error(f"训练失败: {e}")
        else:
            st.warning("请先加载数据")

    st.markdown("---")
    st.markdown("### 📈 系统状态")

    status_data = st.session_state.current_ticker is not None
    status_model = st.session_state.predictor is not None

    st.markdown(f"**数据状态:** {'🟢 已加载' if status_data else '🔴 未加载'}")
    st.markdown(f"**模型状态:** {'🟢 已训练' if status_model else '🔴 未训练'}")

    if status_data:
        st.markdown(f"**当前股票:** {st.session_state.current_ticker}")
        st.markdown(f"**数据点数:** {len(st.session_state.data)}")

# 主页面
st.markdown('<div class="main-header">📈 AI股票交易助手</div>', unsafe_allow_html=True)

# 顶部指标卡片
if st.session_state.data is not None and not st.session_state.data.empty:
    latest = st.session_state.data.iloc[-1]
    prev = st.session_state.data.iloc[-2]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        price_change = latest['Close'] - prev['Close']
        price_change_pct = (price_change / prev['Close']) * 100
        st.metric(
            label="💰 当前价格",
            value=f"${latest['Close']:.2f}",
            delta=f"{price_change_pct:.2f}%"
        )

    with col2:
        st.metric(
            label="📊 成交量",
            value=f"{latest['Volume'] / 1e6:.1f}M",
            delta=f"{((latest['Volume'] - prev['Volume']) / prev['Volume'] * 100):.1f}%"
        )

    with col3:
        rsi = latest.get('RSI', 0)
        rsi_signal = "超买" if rsi > 70 else "超卖" if rsi < 30 else "中性"
        st.metric(
            label="🎯 RSI指标",
            value=f"{rsi:.1f}",
            delta=rsi_signal
        )

    with col4:
        if st.session_state.predictor:
            probs = st.session_state.predictor.predict_future(st.session_state.data.iloc[-1:])
            prob_1d = probs.get(1, 0.5)
            signal = "看涨" if prob_1d > 0.55 else "看跌" if prob_1d < 0.45 else "观望"
            st.metric(
                label="🔮 AI预测",
                value=f"{prob_1d * 100:.1f}%",
                delta=signal
            )
        else:
            st.metric(
                label="🔮 AI预测",
                value="--",
                delta="未训练"
            )

    st.markdown("---")

    # 标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📈 价格走势", "🔮 AI预测", "📊 技术分析", "📋 数据表"])

    with tab1:
        st.subheader("股票价格走势图")

        # 创建 K 线图
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('价格走势', '成交量'),
            vertical_spacing=0.05
        )

        # K线图
        fig.add_trace(
            go.Candlestick(
                x=st.session_state.data.index,
                open=st.session_state.data['Open'],
                high=st.session_state.data['High'],
                low=st.session_state.data['Low'],
                close=st.session_state.data['Close'],
                name='K线'
            ),
            row=1, col=1
        )

        # 添加移动平均线
        if 'SMA_10' in st.session_state.data.columns:
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.data.index,
                    y=st.session_state.data['SMA_10'],
                    name='SMA 10',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )

        if 'SMA_50' in st.session_state.data.columns:
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.data.index,
                    y=st.session_state.data['SMA_50'],
                    name='SMA 50',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )

        # 成交量
        colors = ['red' if close < open else 'green'
                  for close, open in zip(st.session_state.data['Close'], st.session_state.data['Open'])]

        fig.add_trace(
            go.Bar(
                x=st.session_state.data.index,
                y=st.session_state.data['Volume'],
                name='成交量',
                marker_color=colors
            ),
            row=2, col=1
        )

        fig.update_layout(
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("AI预测分析")

        if st.session_state.predictor:
            latest_data = st.session_state.data.iloc[-1:]
            probs = st.session_state.predictor.predict_future(latest_data)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### 📊 预测概率")
                for days, prob in probs.items():
                    direction = "📈 看涨" if prob > 0.5 else "📉 看跌"
                    confidence = abs(prob - 0.5) * 200

                    st.markdown(f"**未来 {days} 天**")
                    st.progress(prob)
                    st.markdown(f"{direction} - 概率: {prob * 100:.1f}% (置信度: {confidence:.1f}%)")
                    st.markdown("---")

                # 综合建议
                avg_prob = sum(probs.values()) / len(probs)
                if avg_prob > 0.6:
                    st.success("🟢 **综合建议: 强烈看涨，建议买入**")
                elif avg_prob > 0.55:
                    st.info("🔵 **综合建议: 温和看涨，可考虑买入**")
                elif avg_prob > 0.45:
                    st.warning("🟡 **综合建议: 中性，建议观望**")
                elif avg_prob > 0.4:
                    st.warning("🟠 **综合建议: 温和看跌，可考虑减仓**")
                else:
                    st.error("🔴 **综合建议: 强烈看跌，建议卖出**")

            with col2:
                # 预测可视化
                days_list = list(probs.keys())
                prob_list = list(probs.values())

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=[f"{d}天" for d in days_list],
                    y=prob_list,
                    text=[f"{p * 100:.1f}%" for p in prob_list],
                    textposition='auto',
                    marker_color=['green' if p > 0.5 else 'red' for p in prob_list]
                ))

                fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                              annotation_text="中性线")

                fig.update_layout(
                    title="未来涨跌概率预测",
                    xaxis_title="时间窗口",
                    yaxis_title="上涨概率",
                    yaxis_range=[0, 1],
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("请先点击侧边栏的 '🎯 训练模型' 按钮")

    with tab3:
        st.subheader("技术指标分析")

        col1, col2 = st.columns(2)

        with col1:
            # RSI
            if 'RSI' in st.session_state.data.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.data.index[-100:],
                    y=st.session_state.data['RSI'].iloc[-100:],
                    name='RSI',
                    line=dict(color='purple', width=2)
                ))
                fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买")
                fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖")
                fig.update_layout(title="RSI指标", height=300)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # MACD
            if 'MACD_12_26_9' in st.session_state.data.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.data.index[-100:],
                    y=st.session_state.data['MACD_12_26_9'].iloc[-100:],
                    name='MACD',
                    line=dict(color='blue', width=2)
                ))
                if 'MACDs_12_26_9' in st.session_state.data.columns:
                    fig.add_trace(go.Scatter(
                        x=st.session_state.data.index[-100:],
                        y=st.session_state.data['MACDs_12_26_9'].iloc[-100:],
                        name='Signal',
                        line=dict(color='orange', width=1)
                    ))
                fig.update_layout(title="MACD指标", height=300)
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("数据详情")

        # 显示最近的数据
        display_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD_12_26_9']
        available_cols = [col for col in display_cols if col in st.session_state.data.columns]

        st.dataframe(
            st.session_state.data[available_cols].tail(20).sort_index(ascending=False),
            use_container_width=True
        )

        # 下载按钮
        csv = st.session_state.data.to_csv()
        st.download_button(
            label="📥 下载完整数据 (CSV)",
            data=csv,
            file_name=f"{st.session_state.current_ticker}_data.csv",
            mime="text/csv"
        )

else:
    # 欢迎页面
    st.info("👈 请在左侧选择股票并加载数据")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📥 数据管理")
        st.markdown("- 下载股票历史数据")
        st.markdown("- 自动增量更新")
        st.markdown("- 计算技术指标")

    with col2:
        st.markdown("### 🔮 AI预测")
        st.markdown("- XGBoost模型")
        st.markdown("- 多时间窗口预测")
        st.markdown("- 概率可视化")

    with col3:
        st.markdown("### 🤖 智能交易")
        st.markdown("- 强化学习策略")
        st.markdown("- 自动交易信号")
        st.markdown("- 风险管理")

# 底部信息
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: gray;">AI股票交易助手 v1.0 | 仅供学习研究使用</div>',
    unsafe_allow_html=True
)