import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import numpy as np

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from improved_data_engine import DataEngine, BatchDataEngine
from macd_strategy import MACDStrategy

from frontend.v3_utils import apply_theme

st.set_page_config(
    page_title="短线交易",
    page_icon="▸",
    layout="wide"
)
apply_theme()
st.markdown('<h1 style="font-family:Noto Serif SC,serif;font-weight:600;color:#e8e4d9;border-bottom:1px solid #252a35;padding-bottom:0.6rem"><span style="color:#c9a84c">▸</span> MACD 短线交易策略</h1>', unsafe_allow_html=True)

# 初始化 session state
if 'strategy_result' not in st.session_state:
    st.session_state.strategy_result = None
if 'strategy_obj' not in st.session_state:
    st.session_state.strategy_obj = None

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
    st.markdown("## ⚡ 策略配置")

    # 选择股票
    selected_display = st.selectbox(
        "选择股票",
        ticker_display_list,
        key="short_ticker"
    )
    selected_ticker = ticker_options[selected_display]

    st.markdown("---")

    # 策略参数
    st.markdown("### 📊 策略参数")

    initial_balance = st.number_input(
        "初始资金 ($)",
        min_value=1000,
        max_value=1000000,
        value=10000,
        step=1000,
        help="回测的起始资金"
    )

    commission = st.slider(
        "手续费率 (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="每笔交易的手续费率"
    ) / 100

    smooth_threshold = st.slider(
        "平滑阈值",
        min_value=0.01,
        max_value=0.10,
        value=0.02,
        step=0.01,
        help="MACD变化小于此值时判定为平滑"
    )

    st.markdown("---")

    # 回测按钮
    if st.button("🚀 执行策略回测", type="primary", use_container_width=True):
        with st.spinner("正在加载数据和执行回测..."):
            try:
                # 加载数据
                engine = DataEngine(selected_ticker)
                df = engine.load_processed_data()

                if df is None or df.empty:
                    st.error("数据加载失败")
                else:
                    # 创建策略
                    strategy = MACDStrategy(
                        df,
                        initial_balance=initial_balance,
                        commission=commission,
                        smooth_threshold=smooth_threshold
                    )

                    # 执行回测
                    metrics = strategy.backtest()

                    # 保存到 session state
                    st.session_state.strategy_result = metrics
                    st.session_state.strategy_obj = strategy
                    st.session_state.current_short_ticker = selected_ticker

                    st.success("✅ 回测完成！")

            except Exception as e:
                st.error(f"回测失败: {e}")
                import traceback

                st.error(traceback.format_exc())

    st.markdown("---")

    # 策略说明
    st.markdown("### 📖 策略说明")

    with st.expander("查看详细说明"):
        st.markdown("""
        **买入信号（三个条件同时满足）**:
        1. ✅ MACD快线由下跌变为平滑
           - 前一天：MACD下降（变化 < 0）
           - 当天：MACD平滑（|变化| ≤ 阈值）
        2. ✅ 绿柱缩短
           - 柱状图为负值（绿柱）
           - 柱状图增加（负值变小）
        3. 💡 含义：下跌动能减弱，可能反转上涨

        **卖出信号（三个条件同时满足）**:
        1. ✅ MACD快线由上升变为平滑
           - 前一天：MACD上升（变化 > 0）
           - 当天：MACD平滑（|变化| ≤ 阈值）
        2. ✅ 红柱缩短
           - 柱状图为正值（红柱）
           - 柱状图减少（正值变小）
        3. 💡 含义：上涨动能减弱，可能反转下跌

        **交易周期**:
        - 📅 按日线交易（每日收盘后判断信号）
        - ⏰ 持仓时间：通常3-15个交易日
        - 🎯 目标：捕捉短期趋势反转

        **适用场景**:
        - 📈 趋势明显的股票
        - 💹 波段操作
        - 🔄 短线进出
        - 📊 技术面主导的市场

        **策略优势**:
        - ⚡ 反应灵敏：及时捕捉反转信号
        - 🛡️ 双重确认：MACD + 柱状图双重验证
        - 📉 风险可控：及时止盈止损
        - 📊 逻辑清晰：规则明确，易于执行

        **注意事项**:
        - ⚠️ 震荡市场可能产生假信号
        - ⚠️ 需要设置止损保护
        - ⚠️ 建议结合成交量分析
        - ⚠️ 不同股票需要不同参数
        """)

        st.markdown("---")
        st.markdown("#### 📊 图示说明")

        st.markdown("""
        **MACD指标组成**:
        - **蓝线（MACD）**: 快线，DIF线
        - **橙线（Signal）**: 慢线，DEA线
        - **红/绿柱**: 柱状图 = MACD - Signal
          - 红柱（正值）: MACD > Signal，多头强势
          - 绿柱（负值）: MACD < Signal，空头强势

        **买入时机示意**:
        ```
        Day 1: MACD ↓↓↓ (下跌) | 绿柱 ||||||||
        Day 2: MACD →  (平滑) | 绿柱 ||||||   ← 买入信号
        Day 3: MACD ↑  (上升) | 绿柱 ||||
        ```

        **卖出时机示意**:
        ```
        Day 1: MACD ↑↑↑ (上升) | 红柱 ||||||||
        Day 2: MACD →  (平滑) | 红柱 ||||||   ← 卖出信号
        Day 3: MACD ↓  (下降) | 红柱 ||||
        ```
        """)

        st.markdown("---")
        st.markdown("#### 🎯 参数调优指南")

        st.markdown("""
        **平滑阈值（Smooth Threshold）**:
        - **0.01**: 非常敏感，信号多，可能有假信号
        - **0.02**: 推荐值，平衡灵敏度和可靠性
        - **0.05**: 较保守，信号少，可靠性高

        **建议**:
        - 波动大的股票：用较大阈值（0.03-0.05）
        - 稳定的股票：用较小阈值（0.01-0.02）
        - 多次测试找到最优值
        """)

# 主内容区
if st.session_state.strategy_result is None:
    # 欢迎页面
    st.markdown("""
    ## 欢迎使用MACD短线交易策略

    ### 策略原理

    本策略基于 **MACD（移动平均收敛散度）指标**，捕捉趋势反转的早期信号。

    #### 🔍 核心逻辑

    **买入时机**：
    - 📉 MACD快线刚从下跌变为平滑（动能减弱）
    - 📊 绿柱开始缩短（空头力量减弱）
    - ⏰ 这通常是下跌趋势即将反转的信号

    **卖出时机**：
    - 📈 MACD快线刚从上升变为平滑（动能减弱）
    - 📊 红柱开始缩短（多头力量减弱）
    - ⏰ 这通常是上涨趋势即将结束的信号

    ### 策略特点

    ✅ **反应灵敏** - 能够快速捕捉趋势变化  
    ✅ **风险可控** - 及时止盈止损  
    ✅ **适合短线** - 持仓时间短，资金周转快  
    ✅ **逻辑清晰** - 规则明确，易于执行  

    ### 使用步骤

    1. **选择股票** - 在左侧选择要回测的股票
    2. **设置参数** - 配置初始资金、手续费等
    3. **执行回测** - 点击"执行策略回测"按钮
    4. **查看结果** - 分析收益率、交易记录等

    ### 注意事项

    ⚠️ **回测不等于实盘**  
    - 回测是理想化的，实盘有滑点
    - 需要考虑流动性影响
    - 市场环境会变化

    ⚠️ **风险控制**  
    - 设置合理的仓位
    - 使用止损保护
    - 不要过度交易

    ⚠️ **参数调优**  
    - 不同股票适合不同参数
    - 建议多次测试找最优参数
    - 避免过度优化（过拟合）
    """)

    # 示例展示
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **适合的股票**
        - 趋势明显
        - 波动适中
        - 流动性好
        - 科技股、大盘股
        """)

    with col2:
        st.info("""
        **不适合的股票**
        - 长期横盘
        - 极度震荡
        - 成交量小
        - 概念炒作股
        """)

    with col3:
        st.info("""
        **建议参数**
        - 初始资金: $10,000
        - 手续费: 0.1%
        - 平滑阈值: 0.02
        - 根据回测调整
        """)

else:
    # 结果页面
    current_ticker = st.session_state.get('current_short_ticker', 'Unknown')
    metrics = st.session_state.strategy_result
    strategy = st.session_state.strategy_obj

    # 检查是否是当前股票
    if current_ticker != selected_ticker:
        st.warning(f"⚠️ 当前结果是 {current_ticker} 的回测，但你选择的是 {selected_ticker}")
        st.info("请重新执行回测")
        st.stop()

    # 标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📊 回测概览", "📈 图表分析", "📋 交易记录", "⚙️ 策略优化"])

    with tab1:
        st.subheader(f"📊 {selected_ticker} 回测概览")

        # 关键指标
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "总收益率",
                f"{metrics['total_return_pct']:.2f}%",
                delta=f"{metrics['total_return_pct']:.2f}%"
            )

        with col2:
            profit = metrics['final_value'] - initial_balance
            st.metric(
                "最终资产",
                f"${metrics['final_value']:,.2f}",
                delta=f"${profit:+,.2f}"
            )

        with col3:
            st.metric(
                "最大回撤",
                f"{metrics['max_drawdown_pct']:.2f}%"
            )

        with col4:
            st.metric(
                "夏普比率",
                f"{metrics['sharpe_ratio']:.3f}"
            )

        with col5:
            st.metric(
                "交易次数",
                f"{metrics['total_trades']}"
            )

        st.markdown("---")

        # 详细指标
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### 📈 收益分析")

            subcol1, subcol2 = st.columns(2)

            with subcol1:
                st.metric("盈利交易", f"{metrics['profitable_trades']}")
                st.metric("平均盈利", f"${metrics['avg_profit']:,.2f}")
                st.metric("策略收益", f"{metrics['total_return_pct']:.2f}%")

            with subcol2:
                st.metric("亏损交易", f"{metrics['losing_trades']}")
                st.metric("平均亏损", f"${metrics['avg_loss']:,.2f}")
                st.metric("买入持有", f"{metrics['buy_hold_return_pct']:.2f}%")

            # 胜率
            win_rate = metrics['win_rate'] * 100
            st.progress(metrics['win_rate'])
            st.markdown(f"**胜率**: {win_rate:.1f}%")

            # 盈亏比
            st.markdown(f"**盈亏比**: {metrics['profit_loss_ratio']:.2f}")

        with col_b:
            st.markdown("### 📊 策略对比")

            # 对比图
            comparison_data = {
                '策略': ['MACD策略', '买入持有'],
                '收益率': [metrics['total_return_pct'], metrics['buy_hold_return_pct']]
            }

            fig = go.Figure()

            colors = ['green' if x > 0 else 'red' for x in comparison_data['收益率']]

            fig.add_trace(go.Bar(
                x=comparison_data['策略'],
                y=comparison_data['收益率'],
                text=[f"{x:.2f}%" for x in comparison_data['收益率']],
                textposition='auto',
                marker_color=colors
            ))

            fig.update_layout(
                title="策略收益对比",
                yaxis_title="收益率 (%)",
                height=300,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # 超额收益
            outperformance = metrics['outperformance']
            if outperformance > 0:
                st.success(f"🎉 策略跑赢买入持有 **{outperformance:.2f}%**")
            elif outperformance < 0:
                st.error(f"📉 策略跑输买入持有 **{abs(outperformance):.2f}%**")
            else:
                st.info("⚖️ 策略与买入持有持平")

        st.markdown("---")

        # 综合评价
        st.markdown("### 💡 策略评价")

        # 评分逻辑
        score = 0
        comments = []

        # 收益率评分
        if metrics['total_return_pct'] > 20:
            score += 30
            comments.append("✅ 收益率优秀")
        elif metrics['total_return_pct'] > 10:
            score += 20
            comments.append("✅ 收益率良好")
        elif metrics['total_return_pct'] > 0:
            score += 10
            comments.append("⚠️ 收益率一般")
        else:
            comments.append("❌ 策略亏损")

        # 胜率评分
        if metrics['win_rate'] > 0.6:
            score += 25
            comments.append("✅ 胜率很高")
        elif metrics['win_rate'] > 0.5:
            score += 15
            comments.append("✅ 胜率过半")
        else:
            comments.append("⚠️ 胜率偏低")

        # 盈亏比评分
        if metrics['profit_loss_ratio'] > 2:
            score += 25
            comments.append("✅ 盈亏比优秀")
        elif metrics['profit_loss_ratio'] > 1:
            score += 15
            comments.append("✅ 盈亏比合格")
        else:
            comments.append("⚠️ 盈亏比偏低")

        # 超额收益评分
        if outperformance > 10:
            score += 20
            comments.append("✅ 显著跑赢大盘")
        elif outperformance > 0:
            score += 10
            comments.append("✅ 跑赢大盘")
        else:
            comments.append("⚠️ 未能跑赢大盘")

        # 显示评分
        col_score1, col_score2 = st.columns([1, 3])

        with col_score1:
            st.metric("策略评分", f"{score}/100")

            if score >= 80:
                st.success("🌟 优秀")
            elif score >= 60:
                st.info("👍 良好")
            elif score >= 40:
                st.warning("😐 一般")
            else:
                st.error("👎 较差")

        with col_score2:
            for comment in comments:
                st.markdown(f"- {comment}")

    with tab2:
        st.subheader("📈 图表分析")

        # 加载数据
        engine = DataEngine(selected_ticker)
        df = engine.load_processed_data()

        # 获取交易记录
        trades = strategy.get_trade_details()

        # 创建图表
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('价格与买卖点', 'MACD指标', '资产曲线'),
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": True}],
                   [{"secondary_y": False}]]
        )

        # 1. 价格曲线
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                name='价格',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )

        # 买入点
        if not trades.empty:
            buy_trades = trades[trades['type'] == 'BUY']
            if not buy_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_trades['date'],
                        y=buy_trades['price'],
                        mode='markers',
                        name='买入',
                        marker=dict(
                            color='green',
                            size=12,
                            symbol='triangle-up',
                            line=dict(color='white', width=1)
                        )
                    ),
                    row=1, col=1
                )

            # 卖出点
            sell_trades = trades[trades['type'] == 'SELL']
            if not sell_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_trades['date'],
                        y=sell_trades['price'],
                        mode='markers',
                        name='卖出',
                        marker=dict(
                            color='red',
                            size=12,
                            symbol='triangle-down',
                            line=dict(color='white', width=1)
                        )
                    ),
                    row=1, col=1
                )

        # 2. MACD指标
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_12_26_9'],
                name='MACD',
                line=dict(color='blue', width=1)
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACDs_12_26_9'],
                name='Signal',
                line=dict(color='orange', width=1)
            ),
            row=2, col=1
        )

        # MACD柱状图
        colors = ['red' if val >= 0 else 'green' for val in df['MACDh_12_26_9']]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['MACDh_12_26_9'],
                name='Histogram',
                marker_color=colors,
                opacity=0.5
            ),
            row=2, col=1,
            secondary_y=True
        )

        # 3. 资产曲线
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=strategy.portfolio_values,
                name='策略资产',
                line=dict(color='green', width=2),
                fill='tonexty'
            ),
            row=3, col=1
        )

        # 添加初始资金线
        fig.add_hline(
            y=initial_balance,
            line_dash="dash",
            line_color="gray",
            annotation_text="初始资金",
            row=3, col=1
        )

        # 买入持有曲线
        buy_hold_values = [initial_balance * (1 + (price - df['Close'].iloc[0]) / df['Close'].iloc[0])
                           for price in df['Close']]
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=buy_hold_values,
                name='买入持有',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=3, col=1
        )

        # 更新布局
        fig.update_layout(
            height=900,
            showlegend=True,
            hovermode='x unified',
            title_text=f"{selected_ticker} MACD策略回测分析"
        )

        fig.update_xaxes(title_text="日期", row=3, col=1)
        fig.update_yaxes(title_text="价格 ($)", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="资产 ($)", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # 信号统计
        signals = strategy.get_signals_summary()

        st.markdown("---")
        st.markdown("### 📊 信号统计")

        col_sig1, col_sig2 = st.columns(2)

        with col_sig1:
            st.metric("总买入信号", signals['total_buy_signals'])
            st.metric("实际买入", metrics['total_trades'])

        with col_sig2:
            st.metric("总卖出信号", signals['total_sell_signals'])
            execution_rate = (metrics['total_trades'] / signals['total_buy_signals'] * 100) if signals[
                                                                                                   'total_buy_signals'] > 0 else 0
            st.metric("执行率", f"{execution_rate:.1f}%")

    with tab3:
        st.subheader("📋 交易明细")

        trades = strategy.get_trade_details()

        if trades.empty:
            st.info("没有交易记录")
        else:
            # 格式化交易记录
            trades_display = trades.copy()
            trades_display['date'] = pd.to_datetime(trades_display['date']).dt.strftime('%Y-%m-%d')
            trades_display['price'] = trades_display['price'].apply(lambda x: f"${x:.2f}")

            # 添加盈亏信息
            if 'profit' in trades_display.columns:
                trades_display['profit'] = trades_display['profit'].apply(
                    lambda x: f"${x:+,.2f}" if pd.notna(x) else ""
                )
                trades_display['profit_pct'] = trades_display['profit_pct'].apply(
                    lambda x: f"{x:+.2f}%" if pd.notna(x) else ""
                )

            # 重命名列
            column_mapping = {
                'date': '日期',
                'type': '类型',
                'price': '价格',
                'shares': '股数',
                'cost': '成本',
                'revenue': '收入',
                'balance': '余额',
                'profit': '盈亏',
                'profit_pct': '盈亏率'
            }

            trades_display = trades_display.rename(columns=column_mapping)

            # 显示表格
            st.dataframe(
                trades_display,
                use_container_width=True,
                hide_index=True
            )

            # 下载按钮
            csv = trades.to_csv(index=False)
            st.download_button(
                label="📥 下载交易记录 (CSV)",
                data=csv,
                file_name=f"{selected_ticker}_trades_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    with tab4:
        st.subheader("⚙️ 策略优化建议")

        st.markdown("""
        ### 💡 优化方向

        根据当前回测结果，以下是一些优化建议：
        """)

        col_opt1, col_opt2 = st.columns(2)

        with col_opt1:
            st.markdown("#### 📈 如果收益不理想")

            if metrics['total_return_pct'] < 5:
                st.warning("""
                **问题**: 收益率较低

                **可能原因**:
                - 平滑阈值设置不当
                - 市场不适合此策略
                - 交易频率过高

                **优化建议**:
                1. 调整平滑阈值（试试 0.01-0.05）
                2. 添加额外过滤条件
                3. 增加止损止盈规则
                4. 更换测试股票
                """)
            else:
                st.success("✅ 收益率表现良好，继续保持！")

        with col_opt2:
            st.markdown("#### 📊 如果胜率不高")

            if metrics['win_rate'] < 0.5:
                st.warning("""
                **问题**: 胜率低于50%

                **可能原因**:
                - 假信号过多
                - 止损设置问题
                - 市场震荡

                **优化建议**:
                1. 增加信号确认条件
                2. 结合成交量指标
                3. 添加趋势过滤
                4. 优化出场时机
                """)
            else:
                st.success("✅ 胜率良好，策略有效！")

        st.markdown("---")

        st.markdown("#### 🔧 参数调优建议")

        # 参数建议表
        param_suggestions = pd.DataFrame({
            '参数': ['初始资金', '手续费率', '平滑阈值'],
            '当前值': [f"${initial_balance:,}", f"{commission * 100:.2f}%", f"{smooth_threshold:.2f}"],
            '建议范围': ['$5,000 - $50,000', '0.05% - 0.2%', '0.01 - 0.05'],
            '调整建议': [
                '根据实际资金调整',
                '参考实际券商费率',
                '小值=敏感，大值=稳定'
            ]
        })

        st.table(param_suggestions)

        st.markdown("---")

        st.markdown("#### 📝 下一步行动")

        st.info("""
        1. **多股票测试**: 在不同股票上测试，找出最适合的
        2. **参数扫描**: 系统性地测试不同参数组合
        3. **时间周期**: 测试不同的历史时期
        4. **组合策略**: 结合其他指标（RSI、均线等）
        5. **风险管理**: 添加止损、仓位管理规则
        6. **实盘验证**: 小资金实盘测试
        """)

# 底部提示
st.markdown("---")
st.markdown('<div style="text-align:center;color:#7a7570;font-size:0.75rem;letter-spacing:1px;font-family:JetBrains Mono,monospace">回测结果仅供参考 · 实盘交易需谨慎 · 历史表现不代表未来收益</div>', unsafe_allow_html=True)