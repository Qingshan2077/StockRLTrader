import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from improved_data_engine import DataEngine, BatchDataEngine
from advanced_predictor import AdvancedPredictor

from frontend.v3_utils import apply_theme

st.set_page_config(
    page_title="预测分析",
    page_icon="▸",
    layout="wide"
)
apply_theme()
st.markdown('<h1 style="font-family:Noto Serif SC,serif;font-weight:600;color:#e8e4d9;border-bottom:1px solid #252a35;padding-bottom:0.6rem"><span style="color:#c9a84c">▸</span> 智能预测分析</h1>', unsafe_allow_html=True)

# 初始化 session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None

# 获取可用股票
batch_engine = BatchDataEngine()
available_tickers = batch_engine.list_available_data()

if not available_tickers:
    st.warning("⚠️ 暂无本地数据，请先在'数据管理'页面下载股票数据")
    st.stop()

# 创建股票选项（带自定义名称）
ticker_options = {}
ticker_display_list = []

for t in available_tickers:
    engine_temp = DataEngine(t)
    custom_name = engine_temp.get_custom_name()
    if custom_name:
        display_name = f"{custom_name} ({t})"
    else:
        display_name = t
    ticker_options[display_name] = t
    ticker_display_list.append(display_name)

# 侧边栏配置
with st.sidebar:
    st.markdown("## 🔮 预测配置")

    # 选择股票
    selected_display = st.selectbox(
        "选择股票",
        ticker_display_list,
        key="pred_ticker"
    )
    selected_ticker = ticker_options[selected_display]

    st.markdown("---")

    # 预测参数
    st.markdown("### 📅 预测范围")

    prediction_days = st.slider(
        "预测天数",
        min_value=7,
        max_value=60,
        value=30,
        step=1,
        help="选择要预测未来多少天的走势"
    )

    show_confidence = st.checkbox(
        "显示置信区间",
        value=True,
        help="在图表中显示预测的置信区间"
    )

    st.markdown("---")

    # 训练按钮
    if st.button("🎯 训练预测模型", type="primary", use_container_width=True):
        with st.spinner("正在加载数据和训练模型..."):
            try:
                # 加载数据
                engine = DataEngine(selected_ticker)
                df = engine.load_processed_data()

                if df is None or df.empty:
                    st.error("数据加载失败")
                else:
                    # 创建预测器
                    predictor = AdvancedPredictor(df)

                    # 创建目标
                    predictor.create_price_targets([10, 30])
                    predictor.create_trend_targets([1, 5, 10, 20, 30])

                    # 训练模型
                    st.info("训练价格预测模型...")
                    price_metrics = predictor.train_price_model(horizon=10, test_size=0.2)

                    st.info("训练趋势预测模型...")
                    trend_metrics = predictor.train_trend_models(
                        horizons=[1, 5, 10, 20, 30],
                        test_size=0.2
                    )

                    # 保存到 session state
                    st.session_state.predictor = predictor
                    st.session_state.current_pred_ticker = selected_ticker
                    st.session_state.price_metrics = price_metrics
                    st.session_state.trend_metrics = trend_metrics

                    st.success("✅ 模型训练完成！")

            except Exception as e:
                st.error(f"训练失败: {e}")
                import traceback

                st.error(traceback.format_exc())

    st.markdown("---")

    # 模型状态
    st.markdown("### 📊 模型状态")

    if st.session_state.predictor is not None:
        st.success("🟢 模型已就绪")
        current_ticker = st.session_state.get('current_pred_ticker', 'Unknown')
        st.markdown(f"**当前股票**: {current_ticker}")
    else:
        st.info("🔴 未训练模型")
        st.markdown("请点击上方按钮训练模型")

# 主内容区
if st.session_state.predictor is None:
    # 欢迎页面
    st.markdown("""
    ## 欢迎使用智能预测分析系统

    ### 功能特点

    - 📈 **价格预测**: 预测未来股价走势
    - 📊 **趋势分析**: 判断未来涨跌趋势
    - 🎯 **置信区间**: 显示预测的不确定性
    - 📅 **灵活时间**: 支持 7-60 天预测

    ### 使用步骤

    1. 在左侧选择要分析的股票
    2. 点击 "🎯 训练预测模型" 按钮
    3. 等待模型训练完成
    4. 查看预测结果和走势图

    ### 预测原理

    系统使用 **XGBoost 机器学习算法**，基于以下特征预测未来走势：

    - 技术指标（RSI、MACD、布林带等）
    - 历史价格模式
    - 成交量变化
    - 市场动量

    ### 注意事项

    ⚠️ **免责声明**: 
    - 预测仅供参考，不构成投资建议
    - 股市有风险，投资需谨慎
    - 历史表现不代表未来收益
    """)

    # 显示示例图
    st.markdown("---")
    st.markdown("### 📊 预测示例")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **短期预测 (1-10天)**
        - 适合日内交易和短线操作
        - 预测准确率相对较高
        - 受短期新闻影响大
        """)

    with col2:
        st.info("""
        **中长期预测 (10-60天)**
        - 适合波段操作和中线持有
        - 反映大趋势
        - 不确定性增加
        """)

else:
    # 预测结果页面
    current_ticker = st.session_state.get('current_pred_ticker', 'Unknown')

    # 检查是否是当前股票
    if current_ticker != selected_ticker:
        st.warning(f"⚠️ 当前模型是为 {current_ticker} 训练的，但你选择的是 {selected_ticker}")
        st.info("请重新训练模型或选择对应的股票")
        st.stop()

    # 加载数据
    engine = DataEngine(selected_ticker)
    df = engine.load_processed_data()

    if df is None or df.empty:
        st.error("数据加载失败")
        st.stop()

    # 标签页
    tab1, tab2, tab3 = st.tabs(["📈 走势预测", "🎯 趋势分析", "📊 模型性能"])

    with tab1:
        st.subheader(f"📈 {selected_ticker} 未来走势预测")

        # 生成预测
        with st.spinner("正在生成预测..."):
            try:
                # 生成未来趋势线
                trend_line = st.session_state.predictor.generate_future_trend_line(
                    df.tail(100),
                    days=prediction_days,
                    method='mixed'
                )

                # 创建图表
                fig = go.Figure()

                # 历史价格（最近90天）
                hist_days = min(90, len(df))
                hist_data = df.tail(hist_days)

                fig.add_trace(go.Scatter(
                    x=hist_data.index,
                    y=hist_data['Close'],
                    name='历史价格',
                    line=dict(color='blue', width=2),
                    hovertemplate='<b>日期</b>: %{x}<br><b>价格</b>: $%{y:.2f}<extra></extra>'
                ))

                # 预测价格
                fig.add_trace(go.Scatter(
                    x=trend_line['dates'],
                    y=trend_line['prices'],
                    name='预测价格',
                    line=dict(color='red', width=2, dash='dash'),
                    hovertemplate='<b>日期</b>: %{x}<br><b>预测</b>: $%{y:.2f}<extra></extra>'
                ))

                # 置信区间
                if show_confidence:
                    # 上界
                    fig.add_trace(go.Scatter(
                        x=trend_line['dates'],
                        y=trend_line['upper_bound'],
                        name='置信上界 (+5%)',
                        line=dict(color='rgba(255,0,0,0.2)', width=0),
                        showlegend=True,
                        hoverinfo='skip'
                    ))

                    # 下界
                    fig.add_trace(go.Scatter(
                        x=trend_line['dates'],
                        y=trend_line['lower_bound'],
                        name='置信下界 (-5%)',
                        line=dict(color='rgba(255,0,0,0.2)', width=0),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.1)',
                        showlegend=True,
                        hoverinfo='skip'
                    ))

                # ⚠️ 关键：强制转成 Python datetime
                last_date = hist_data.index[-1].to_pydatetime()
                last_price = hist_data['Close'].iloc[-1]

                # 只画线，不要 annotation
                fig.add_vline(
                    x=last_date,
                    line_dash="dot",
                    line_color="gray"
                )

                # 手动加 annotation（完全绕过 Plotly bug）
                fig.add_annotation(
                    x=last_date,
                    y=1,
                    yref="paper",
                    text="今天",
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(color="gray")
                )

                # # 添加分界线
                # last_date = hist_data.index[-1]
                # last_price = hist_data['Close'].iloc[-1]
                #
                # fig.add_vline(
                #     x=last_date,
                #     line_dash="dot",
                #     line_color="gray",
                #     annotation_text="今天",
                #     annotation_position="top"
                # )

                fig.update_layout(
                    title=f"{selected_ticker} 价格走势与预测",
                    xaxis_title="日期",
                    yaxis_title="价格 ($)",
                    height=600,
                    hovermode='x unified',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

                # 预测摘要
                st.markdown("---")
                st.markdown("### 📊 预测摘要")

                col1, col2, col3, col4 = st.columns(4)

                current_price = hist_data['Close'].iloc[-1]
                predicted_price = trend_line['prices'][-1]
                price_change = predicted_price - current_price
                price_change_pct = (price_change / current_price) * 100

                with col1:
                    st.metric(
                        "当前价格",
                        f"${current_price:.2f}"
                    )

                with col2:
                    st.metric(
                        f"{prediction_days}天后预测价格",
                        f"${predicted_price:.2f}",
                        delta=f"{price_change_pct:+.2f}%"
                    )

                with col3:
                    st.metric(
                        "预测变化",
                        f"${abs(price_change):.2f}",
                        delta=f"{price_change:+.2f}"
                    )

                with col4:
                    trend_text = "📈 看涨" if price_change > 0 else "📉 看跌"
                    st.metric(
                        "预测趋势",
                        trend_text
                    )

                # 详细分析
                st.markdown("---")
                st.markdown("### 📝 详细分析")

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("#### 短期预测 (7天)")
                    week_price = trend_line['prices'][min(6, len(trend_line['prices']) - 1)]
                    week_change = ((week_price - current_price) / current_price) * 100

                    if week_change > 2:
                        st.success(f"📈 预计上涨 {week_change:.2f}% 至 ${week_price:.2f}")
                    elif week_change < -2:
                        st.error(f"📉 预计下跌 {week_change:.2f}% 至 ${week_price:.2f}")
                    else:
                        st.info(f"⏸️ 预计波动 {week_change:.2f}% 至 ${week_price:.2f}")

                with col_b:
                    st.markdown(f"#### 长期预测 ({prediction_days}天)")

                    if price_change_pct > 5:
                        st.success(f"📈 强势上涨趋势，预计涨幅 {price_change_pct:.2f}%")
                    elif price_change_pct > 0:
                        st.info(f"📈 温和上涨趋势，预计涨幅 {price_change_pct:.2f}%")
                    elif price_change_pct > -5:
                        st.warning(f"📉 温和下跌趋势，预计跌幅 {abs(price_change_pct):.2f}%")
                    else:
                        st.error(f"📉 明显下跌趋势，预计跌幅 {abs(price_change_pct):.2f}%")

            except Exception as e:
                st.error(f"预测生成失败: {e}")
                import traceback

                st.error(traceback.format_exc())

    with tab2:
        st.subheader("🎯 多时间窗口趋势分析")

        # 预测不同时间窗口
        latest_data = df.iloc[-1:]
        horizons = [1, 5, 10, 20, 30]

        predictions = []
        for h in horizons:
            try:
                pred = st.session_state.predictor.predict_future_trend(latest_data, h)
                predictions.append({
                    '时间窗口': f"{h} 天",
                    '预测趋势': pred['trend'],
                    '上涨概率': f"{pred['probability'] * 100:.1f}%",
                    '置信度': f"{pred['confidence']:.1f}%"
                })
            except:
                pass

        if predictions:
            df_pred = pd.DataFrame(predictions)
            st.dataframe(df_pred, use_container_width=True, hide_index=True)

            # 可视化
            st.markdown("---")
            st.markdown("### 📊 概率可视化")

            probs = [float(p['上涨概率'].rstrip('%')) for p in predictions]
            horizons_text = [p['时间窗口'] for p in predictions]

            fig = go.Figure()

            # 添加概率柱状图
            colors = ['green' if p > 50 else 'red' for p in probs]

            fig.add_trace(go.Bar(
                x=horizons_text,
                y=probs,
                text=[f"{p:.1f}%" for p in probs],
                textposition='auto',
                marker_color=colors,
                name='上涨概率'
            ))

            # 添加50%中性线
            fig.add_hline(
                y=50,
                line_dash="dash",
                line_color="gray",
                annotation_text="中性线 (50%)"
            )

            fig.update_layout(
                title="不同时间窗口的上涨概率",
                xaxis_title="时间窗口",
                yaxis_title="上涨概率 (%)",
                yaxis_range=[0, 100],
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # 综合建议
            st.markdown("---")
            st.markdown("### 💡 综合投资建议")

            avg_prob = sum(probs) / len(probs)

            if avg_prob > 60:
                st.success("""
                🟢 **强烈看涨** - 建议买入
                - 多个时间窗口显示上涨趋势
                - 可以考虑建仓或加仓
                - 设置止损点以控制风险
                """)
            elif avg_prob > 55:
                st.info("""
                🔵 **温和看涨** - 可考虑买入
                - 总体趋势偏向上涨
                - 建议小仓位试探
                - 注意观察市场变化
                """)
            elif avg_prob > 45:
                st.warning("""
                🟡 **中性观望** - 建议观望
                - 市场方向不明确
                - 等待更明确的信号
                - 保持现有仓位
                """)
            elif avg_prob > 40:
                st.warning("""
                🟠 **温和看跌** - 可考虑减仓
                - 趋势偏向下跌
                - 建议减少仓位
                - 关注支撑位
                """)
            else:
                st.error("""
                🔴 **强烈看跌** - 建议卖出
                - 多个时间窗口显示下跌趋势
                - 建议及时止损
                - 等待更好的买入机会
                """)

    with tab3:
        st.subheader("📊 模型性能评估")

        # 显示训练指标
        if 'price_metrics' in st.session_state:
            st.markdown("### 📈 价格预测模型")

            metrics = st.session_state.price_metrics

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("均方误差 (MSE)", f"{metrics['mse']:.2f}")
            with col2:
                st.metric("平均绝对误差 (MAE)", f"{metrics['mae']:.2f}")
            with col3:
                st.metric("方向准确率", f"{metrics['direction_accuracy'] * 100:.1f}%")

            st.markdown("---")

        if 'trend_metrics' in st.session_state:
            st.markdown("### 🎯 趋势预测模型")

            metrics = st.session_state.trend_metrics

            # 创建DataFrame
            trend_data = []
            for horizon, metric in metrics.items():
                trend_data.append({
                    '时间窗口': f'{horizon} 天',
                    '准确率': f"{metric['accuracy'] * 100:.1f}%"
                })

            df_metrics = pd.DataFrame(trend_data)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

            # 可视化
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=[d['时间窗口'] for d in trend_data],
                y=[float(d['准确率'].rstrip('%')) for d in trend_data],
                text=[d['准确率'] for d in trend_data],
                textposition='auto',
                marker_color='lightblue'
            ))

            fig.update_layout(
                title="不同时间窗口的预测准确率",
                xaxis_title="时间窗口",
                yaxis_title="准确率 (%)",
                yaxis_range=[0, 100],
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📝 模型说明")

        st.info("""
        **使用的算法**: XGBoost (Extreme Gradient Boosting)

        **特点**:
        - 🎯 高准确率 - 工业界广泛使用
        - 🚀 快速训练 - 支持并行计算
        - 📊 特征重要性 - 可解释性强
        - 🛡️ 防止过拟合 - 内置正则化

        **输入特征**:
        - 技术指标 (RSI, MACD, 布林带等)
        - 价格动量
        - 成交量变化
        - 趋势强度

        **注意事项**:
        - 模型基于历史数据训练
        - 无法预测突发事件影响
        - 建议结合基本面分析
        - 定期重新训练模型
        """)

# 底部提示
st.markdown("---")
st.markdown('<div style="text-align:center;color:#7a7570;font-size:0.75rem;letter-spacing:1px;font-family:JetBrains Mono,monospace">预测结果仅供参考 · 不构成投资建议 · 股市有风险，投资需谨慎</div>', unsafe_allow_html=True)