import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from improved_data_engine import DataEngine, BatchDataEngine
from rl_agent import RLTradingAgent
from data_validator import prepare_for_training, split_train_test

st.set_page_config(
    page_title="AI交易Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 强化学习交易 Agent")

# 初始化 session state
if 'rl_agent' not in st.session_state:
    st.session_state.rl_agent = None
if 'training_metrics' not in st.session_state:
    st.session_state.training_metrics = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

# 获取可用股票列表
batch_engine = BatchDataEngine()
available_tickers = batch_engine.list_available_data()

if not available_tickers:
    st.warning("⚠️ 暂无本地数据，请先在'数据管理'页面下载股票数据")
    st.stop()


# 创建带自定义名称的选项
def create_ticker_options(tickers):
    """创建包含自定义名称的股票选项"""
    options = {}
    display_list = []
    for t in tickers:
        engine_temp = DataEngine(t)
        custom_name = engine_temp.get_custom_name()
        if custom_name:
            display_name = f"{custom_name} ({t})"
        else:
            display_name = t
        options[display_name] = t
        display_list.append(display_name)
    return options, display_list


# 标签页
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 训练 Agent", "📂 模型管理", "📊 回测分析", "🎮 实时交易", "📈 性能对比"])

with tab1:
    st.subheader("训练强化学习交易 Agent")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📋 训练配置")

        # 选择股票（带自定义名称）
        ticker_options, ticker_display_list = create_ticker_options(available_tickers)

        selected_display = st.selectbox(
            "选择股票",
            ticker_display_list,
            key="train_ticker"
        )
        selected_ticker = ticker_options[selected_display]

        col_a, col_b = st.columns(2)

        with col_a:
            initial_balance = st.number_input(
                "初始资金 ($)",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000
            )

            train_steps = st.select_slider(
                "训练步数",
                options=[1000,10000, 25000, 50000, 100000, 200000,500000,1000000],
                value=50000
            )

        with col_b:
            model_type = st.selectbox(
                "模型类型",
                ["PPO", "A2C"],
                help="PPO: 策略优化算法，适合大多数场景\nA2C: 行动者评论家算法，训练速度快"
            )

            learning_rate = st.select_slider(
                "学习率",
                options=[0.0001, 0.0003, 0.001, 0.003],
                value=0.0003,
                format_func=lambda x: f"{x:.4f}"
            )

        # 数据分割比例
        train_ratio = st.slider(
            "训练集比例",
            min_value=0.5,
            max_value=0.9,
            value=0.8,
            step=0.05,
            format="%d%%"
        )

        st.markdown("---")

        # 开始训练
        if st.button("🚀 开始训练", type="primary", use_container_width=True):
            with st.spinner("正在加载数据..."):
                # 加载数据
                engine = DataEngine(selected_ticker)
                df = engine.load_processed_data()

                if df is None or df.empty:
                    st.error("数据加载失败，请先下载数据")
                else:
                    # 数据验证和清理
                    st.info("🔍 验证数据质量...")
                    df_clean, is_ready = prepare_for_training(df, min_length=100, verbose=False)

                    if not is_ready:
                        st.error("❌ 数据质量不符合训练要求，请检查数据")
                        st.warning(
                            "建议：\n1. 确保至少有 2-3 年的历史数据\n2. 检查是否包含技术指标\n3. 尝试重新下载数据")
                    else:
                        st.success("✅ 数据验证通过")

                        # 分割数据
                        train_data, test_data = split_train_test(df_clean, train_ratio=train_ratio, verbose=False)

                        st.info(f"📊 训练集: {len(train_data)} 天 | 测试集: {len(test_data)} 天")

                        # 创建进度条
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # 创建实时监控面板
                        st.markdown("### 📊 训练实时监控")

                        # 指标显示区域
                        metric_cols = st.columns(5)
                        metric_placeholders = {
                            'timesteps': metric_cols[0].empty(),
                            'episodes': metric_cols[1].empty(),
                            'portfolio': metric_cols[2].empty(),
                            'actions': metric_cols[3].empty(),
                            'reward': metric_cols[4].empty()
                        }

                        # 图表显示区域
                        chart_placeholder = st.empty()

                        # 详细信息区域
                        details_expander = st.expander("📈 查看详细训练指标", expanded=False)
                        with details_expander:
                            details_placeholder = st.empty()


                        # 定义更新函数
                        def update_progress(callback):
                            """更新训练进度"""
                            stats = callback.get_statistics()
                            progress = callback.n_calls / train_steps

                            # 更新进度条
                            progress_bar.progress(min(progress, 1.0))
                            status_text.text(f"训练进度: {callback.n_calls}/{train_steps} 步 ({progress * 100:.1f}%)")

                            # 更新指标卡片
                            metric_placeholders['timesteps'].metric(
                                "训练步数",
                                f"{callback.n_calls:,}",
                                delta=f"{stats['episode_count']} episodes"
                            )

                            metric_placeholders['episodes'].metric(
                                "平均奖励",
                                f"{stats['avg_episode_reward']:.2f}",
                                delta=f"长度: {stats['avg_episode_length']:.0f}"
                            )

                            portfolio_value = stats['current_portfolio_value']
                            if portfolio_value > 0:
                                pnl = (portfolio_value - initial_balance) / initial_balance * 100
                                metric_placeholders['portfolio'].metric(
                                    "当前资产",
                                    f"${portfolio_value:,.0f}",
                                    delta=f"{pnl:+.2f}%"
                                )

                            total_actions = stats['buy_count'] + stats['sell_count'] + stats['hold_count']
                            if total_actions > 0:
                                metric_placeholders['actions'].metric(
                                    "交易动作",
                                    f"{stats['buy_count'] + stats['sell_count']}",
                                    delta=f"买{stats['buy_count']}/卖{stats['sell_count']}"
                                )

                            metric_placeholders['reward'].metric(
                                "累计奖励",
                                f"{stats['total_rewards']:.1f}",
                                delta="总收益"
                            )

                            # 更新图表
                            if len(callback.portfolio_values) > 10:
                                import plotly.graph_objects as go
                                from plotly.subplots import make_subplots

                                fig = make_subplots(
                                    rows=2, cols=2,
                                    subplot_titles=('资产变化', 'Episode奖励', '交易动作分布', '奖励趋势'),
                                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                           [{"type": "pie"}, {"secondary_y": False}]]
                                )

                                # 1. 资产变化
                                portfolio_values = callback.portfolio_values[-4000:]
                                # steps = list(range(len(portfolio_values)))[-1000:]
                                fig.add_trace(
                                    go.Scatter(
                                        #x=steps,
                                        y=portfolio_values,
                                        name='资产净值',
                                        line=dict(color='blue', width=2)
                                    ),
                                    row=1, col=1
                                )
                                fig.add_hline(
                                    y=initial_balance,
                                    line_dash="dash",
                                    line_color="gray",
                                    row=1, col=1
                                )

                                # 2. Episode 奖励
                                episode_rewards=callback.episode_rewards[-50:]
                                if len(callback.episode_rewards) > 0:
                                    fig.add_trace(
                                        go.Scatter(
                                            y=episode_rewards,
                                            name='Episode奖励',
                                            line=dict(color='green', width=2)
                                        ),
                                        row=1, col=2
                                    )

                                # 3. 交易动作分布
                                action_counts = [stats['hold_count'], stats['buy_count'], stats['sell_count']]
                                fig.add_trace(
                                    go.Pie(
                                        labels=['持有', '买入', '卖出'],
                                        values=action_counts,
                                        marker=dict(colors=['gray', 'green', 'red'])
                                    ),
                                    row=2, col=1
                                )

                                # 4. 奖励趋势（最近100步）
                                if len(callback.rewards_history) > 0:
                                    recent_rewards = callback.rewards_history[-100:]
                                    fig.add_trace(
                                        go.Scatter(
                                            y=recent_rewards,
                                            name='即时奖励',
                                            line=dict(color='orange', width=1)
                                        ),
                                        row=2, col=2
                                    )

                                fig.update_layout(
                                    height=500,
                                    showlegend=False,
                                    title_text="训练过程监控"
                                )

                                chart_placeholder.plotly_chart(
                                    fig,
                                    use_container_width=True,
                                    key=f"train_chart_{callback.n_calls}"
                                )

                            # 更新详细信息
                            with details_placeholder.container():
                                col_a, col_b, col_c = st.columns(3)

                                with col_a:
                                    st.markdown("**📊 资金状况**")
                                    st.markdown(f"- 现金余额: ${stats['current_balance']:,.2f}")
                                    st.markdown(f"- 持仓数量: {stats['current_shares']:.0f} 股")
                                    st.markdown(f"- 总资产: ${stats['current_portfolio_value']:,.2f}")

                                with col_b:
                                    st.markdown("**📈 交易统计**")
                                    st.markdown(f"- 买入次数: {stats['buy_count']}")
                                    st.markdown(f"- 卖出次数: {stats['sell_count']}")
                                    st.markdown(f"- 持有次数: {stats['hold_count']}")

                                with col_c:
                                    st.markdown("**🎯 训练指标**")
                                    st.markdown(f"- Episode数: {stats['episode_count']}")
                                    st.markdown(f"- 平均奖励: {stats['avg_episode_reward']:.2f}")
                                    st.markdown(f"- 平均长度: {stats['avg_episode_length']:.0f}")


                        try:
                            # 创建 Agent
                            status_text.text("初始化 Agent...")
                            agent = RLTradingAgent(train_data, model_type=model_type)
                            progress_bar.progress(10)

                            # 创建自定义回调
                            from stable_baselines3.common.callbacks import BaseCallback


                            class StreamlitCallback(BaseCallback):
                                def __init__(self, update_freq=100):
                                    super().__init__()
                                    self.update_freq = update_freq
                                    self.training_callback = None

                                def _on_step(self):
                                    # 每隔一定步数更新界面
                                    if self.n_calls % self.update_freq == 0:
                                        if self.training_callback:
                                            try:
                                                update_progress(self.training_callback)
                                            except:
                                                pass  # 忽略更新错误，继续训练
                                    return True


                            # 训练（使用自定义回调）
                            status_text.text(f"开始训练 {model_type} Agent（实时监控中）...")
                            progress_bar.progress(20)

                            # 修改 agent.train 以支持实时更新
                            from rl_agent import TrainingCallback

                            # 创建环境
                            agent.create_environment(initial_balance=initial_balance)

                            # 创建训练回调
                            training_cb = TrainingCallback(verbose=0, update_freq=50)
                            streamlit_cb = StreamlitCallback(update_freq=100)
                            streamlit_cb.training_callback = training_cb

                            # 创建模型
                            if model_type == 'PPO':
                                from stable_baselines3 import PPO
                                import torch as th

                                agent.model = PPO(
                                    'MlpPolicy',
                                    agent.env,
                                    learning_rate=learning_rate,
                                    n_steps=2048,
                                    batch_size=64,
                                    n_epochs=10,
                                    gamma=0.99,
                                    gae_lambda=0.95,
                                    clip_range=0.2,
                                    ent_coef=0.01,
                                    vf_coef=0.5,
                                    max_grad_norm=0.5,
                                    policy_kwargs=dict(
                                        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                                        activation_fn=th.nn.Tanh
                                    ),
                                    verbose=0
                                )

                            # 开始训练（带实时监控）
                            agent.model.learn(
                                total_timesteps=train_steps,
                                callback=[training_cb, streamlit_cb],
                                progress_bar=False
                            )

                            # 最后一次更新
                            update_progress(training_cb)

                            progress_bar.progress(80)
                            status_text.text("计算最终指标...")
                            # 获取最后的状态统计
                            final_stats = training_cb.get_statistics()
                            # 获取资产历史曲线
                            portfolio_values = training_cb.portfolio_values
                            # 1. 计算总收益率
                            final_value = final_stats['current_portfolio_value']
                            total_return = (final_value - initial_balance) / initial_balance

                            # 2. 计算最大回撤 (Max Drawdown)
                            # 将列表转为 Series 以便计算
                            s_portfolio = pd.Series(portfolio_values)
                            # 计算累计最大值
                            running_max = s_portfolio.cummax()
                            # 计算回撤
                            drawdown = (s_portfolio - running_max) / running_max
                            max_drawdown = drawdown.min()  # 这是一个负数，显示时取绝对值即可

                            # 3. 计算夏普比率 (Sharpe Ratio)
                            # 假设无风险利率为 0，基于每个 step 的收益率计算
                            returns = s_portfolio.pct_change().dropna()
                            if returns.std() != 0:
                                # 简单的夏普比率估算 (假设年化步数为 252*交易时长，这里简化处理)
                                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 4)  # 这里的系数根据你的数据频率调整
                            else:
                                sharpe_ratio = 0.0

                            # 手动构建 metrics 字典
                            metrics = {
                                'total_return': total_return,
                                'final_value': final_value,
                                'max_drawdown': abs(max_drawdown),  # 确保是正数
                                'sharpe_ratio': sharpe_ratio
                            }

                            progress_bar.progress(80)
                            status_text.text("保存模型...")

                            # 保存模型
                            model_path = Path(f"data/models/rl_agent_{selected_ticker}_{model_type}.zip")
                            model_path.parent.mkdir(parents=True, exist_ok=True)
                            agent.save_model(str(model_path))

                            progress_bar.progress(100)
                            status_text.empty()
                            progress_bar.empty()

                            # 保存到 session state
                            st.session_state.rl_agent = agent
                            st.session_state.training_metrics = metrics
                            st.session_state.test_data = test_data
                            st.session_state.current_model = model_path.name  # 保存当前模型名称
                            st.session_state.current_ticker = selected_ticker  # 保存当前股票

                            st.success("✅ 训练完成！")

                            # 显示训练结果
                            st.markdown("### 📊 训练结果")

                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric(
                                    "总收益率",
                                    f"{metrics['total_return'] * 100:.2f}%",
                                    delta=f"{metrics['total_return'] * 100:.2f}%"
                                )

                            with col2:
                                st.metric(
                                    "最终资产",
                                    f"${metrics['final_value']:.2f}",
                                    delta=f"${metrics['final_value'] - initial_balance:.2f}"
                                )

                            with col3:
                                st.metric(
                                    "最大回撤",
                                    f"{metrics['max_drawdown'] * 100:.2f}%"
                                )

                            with col4:
                                st.metric(
                                    "夏普比率",
                                    f"{metrics['sharpe_ratio']:.3f}"
                                )

                        except Exception as e:
                            st.error(f"训练失败: {e}")
                            with st.expander("查看详细错误信息"):
                                import traceback

                                st.code(traceback.format_exc())

    with col2:
        st.markdown("### 📚 训练说明")

        st.info("""
        **什么是强化学习 Agent？**

        RL Agent 是一个会通过不断试错来学习交易策略的 AI。它会：

        1. 观察市场状态（价格、指标）
        2. 做出决策（买入/持有/卖出）
        3. 获得奖励（根据收益）
        4. 优化策略（变得更聪明）
        """)

        st.markdown("---")

        st.markdown("**🎯 训练目标**")
        st.markdown("- ✅ 最大化收益")
        st.markdown("- ✅ 控制风险（最大回撤）")
        st.markdown("- ✅ 识别趋势")
        st.markdown("- ✅ 减少过度交易")

        st.markdown("---")

        st.markdown("**⚙️ 参数说明**")

        with st.expander("模型类型"):
            st.markdown("""
            - **PPO**: 策略优化算法，稳定性好，推荐
            - **A2C**: 训练速度快，但可能不够稳定
            """)

        with st.expander("训练步数"):
            st.markdown("""
            - **10K-25K**: 快速测试
            - **50K**: 推荐用于日常训练
            - **100K+**: 深度训练，需要更长时间
            """)

        with st.expander("学习率"):
            st.markdown("""
            - **0.0001**: 慢但稳定
            - **0.0003**: 推荐值（默认）
            - **0.001+**: 快但可能不稳定
            """)

with tab2:
    st.subheader("📂 模型管理")

    model_dir = Path("data/models")

    if not model_dir.exists():
        st.warning("模型目录不存在，请先训练一个模型")
        model_dir.mkdir(parents=True, exist_ok=True)
    else:
        # 查找所有模型文件
        model_files = list(model_dir.glob("*.zip"))

        if not model_files:
            st.info("📭 暂无已保存的模型，请先在'训练 Agent'页面训练模型")
        else:
            st.markdown(f"### 📋 已保存的模型 ({len(model_files)} 个)")

            # 解析模型信息
            model_info = []
            for model_file in model_files:
                # 文件名格式: rl_agent_TICKER_PPO.zip
                parts = model_file.stem.split('_')
                if len(parts) >= 4:
                    ticker = parts[2]
                    model_type = parts[3]
                else:
                    ticker = "Unknown"
                    model_type = "Unknown"

                # 获取文件信息
                file_size = model_file.stat().st_size / 1024  # KB
                file_time = datetime.fromtimestamp(model_file.stat().st_mtime)

                # 获取股票自定义名称
                try:
                    engine = DataEngine(ticker)
                    custom_name = engine.get_custom_name()
                except:
                    custom_name = ""

                model_info.append({
                    "模型名称": model_file.name,
                    "股票": f"{custom_name} ({ticker})" if custom_name else ticker,
                    "算法": model_type,
                    "大小": f"{file_size:.1f} KB",
                    "创建时间": file_time.strftime("%Y-%m-%d %H:%M"),
                    "路径": str(model_file)
                })

            # 显示模型列表
            df_models = pd.DataFrame(model_info)
            st.dataframe(
                df_models.drop(columns=['路径']),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("---")

            # 模型操作
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("### 🔧 模型操作")

                # 选择模型
                selected_model = st.selectbox(
                    "选择要操作的模型",
                    [m["模型名称"] for m in model_info],
                    key="selected_model"
                )

                # 找到对应的模型信息
                selected_info = next(m for m in model_info if m["模型名称"] == selected_model)

                # 显示详细信息
                st.markdown("#### 📄 模型详情")
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown(f"**股票**: {selected_info['股票']}")
                    st.markdown(f"**算法**: {selected_info['算法']}")
                with col_b:
                    st.markdown(f"**大小**: {selected_info['大小']}")
                    st.markdown(f"**创建时间**: {selected_info['创建时间']}")

                st.markdown("---")

                # 操作按钮
                col_x, col_y, col_z = st.columns(3)

                with col_x:
                    if st.button("📥 加载模型", type="primary", use_container_width=True):
                        with st.spinner("正在加载模型..."):
                            try:
                                # 解析股票代码
                                ticker = selected_info['模型名称'].split('_')[2]

                                # 加载数据
                                engine = DataEngine(ticker)
                                df = engine.load_processed_data()

                                if df is None or df.empty:
                                    st.error(f"无法加载 {ticker} 的数据")
                                else:
                                    # 创建 Agent 并加载模型
                                    model_path = selected_info['路径']
                                    agent = RLTradingAgent(
                                        df,
                                        model_type=selected_info['算法'],
                                        model_path=model_path
                                    )

                                    # 保存到 session state
                                    st.session_state.rl_agent = agent
                                    st.session_state.current_model = selected_model
                                    st.session_state.current_ticker = ticker

                                    # 准备测试数据（使用最后20%作为测试）
                                    split = int(len(df) * 0.8)
                                    st.session_state.test_data = df.iloc[split:]

                                    st.success(f"✅ 已加载模型: {selected_model}")
                                    st.info(f"现在可以在'回测分析'和'实时交易'中使用该模型")

                            except Exception as e:
                                st.error(f"加载失败: {e}")
                                with st.expander("查看详细错误"):
                                    import traceback

                                    st.code(traceback.format_exc())

                with col_y:
                    if st.button("📝 重命名", use_container_width=True):
                        st.session_state.show_rename_dialog = True

                with col_z:
                    if st.button("🗑️ 删除", use_container_width=True):
                        st.session_state.show_delete_confirm = True

                # 重命名对话框
                if st.session_state.get('show_rename_dialog', False):
                    st.markdown("---")
                    new_name = st.text_input(
                        "输入新的模型名称（不含扩展名）",
                        value=selected_info['模型名称'].replace('.zip', ''),
                        key="new_model_name"
                    )

                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        if st.button("确认重命名", type="primary"):
                            try:
                                old_path = Path(selected_info['路径'])
                                new_path = old_path.parent / f"{new_name}.zip"

                                if new_path.exists():
                                    st.error("该名称已存在")
                                else:
                                    old_path.rename(new_path)

                                    # 同时重命名历史文件
                                    old_history = old_path.parent / f"{old_path.stem}_history.pkl"
                                    if old_history.exists():
                                        new_history = old_path.parent / f"{new_name}_history.pkl"
                                        old_history.rename(new_history)

                                    st.success(f"✅ 已重命名为: {new_name}.zip")
                                    st.session_state.show_rename_dialog = False
                                    st.rerun()
                            except Exception as e:
                                st.error(f"重命名失败: {e}")

                    with col_r2:
                        if st.button("取消"):
                            st.session_state.show_rename_dialog = False
                            st.rerun()

                # 删除确认对话框
                if st.session_state.get('show_delete_confirm', False):
                    st.markdown("---")
                    st.error(f"⚠️ 确认要删除模型 **{selected_model}** 吗？")
                    st.warning("此操作不可恢复！")

                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        if st.button("确认删除", type="secondary"):
                            try:
                                model_path = Path(selected_info['路径'])
                                model_path.unlink()

                                # 删除历史文件
                                history_path = model_path.parent / f"{model_path.stem}_history.pkl"
                                if history_path.exists():
                                    history_path.unlink()

                                st.success(f"✅ 已删除模型: {selected_model}")
                                st.session_state.show_delete_confirm = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"删除失败: {e}")

                    with col_d2:
                        if st.button("取消删除"):
                            st.session_state.show_delete_confirm = False
                            st.rerun()

            with col2:
                st.markdown("### 📊 当前加载的模型")

                if st.session_state.rl_agent is not None:
                    current_model = st.session_state.get('current_model', '训练的模型')
                    current_ticker = st.session_state.get('current_ticker', 'Unknown')

                    st.success("✅ 已加载模型")
                    st.markdown(f"**模型**: {current_model}")
                    st.markdown(f"**股票**: {current_ticker}")
                    st.markdown(f"**状态**: 🟢 就绪")

                    st.markdown("---")

                    st.markdown("**可用功能**:")
                    st.markdown("- 📊 回测分析")
                    st.markdown("- 🎮 实时交易信号")
                    st.markdown("- 📈 性能对比")
                else:
                    st.info("ℹ️ 未加载模型")
                    st.markdown("请选择一个模型并点击'加载模型'")

                st.markdown("---")

                st.markdown("### 💡 使用提示")
                st.markdown("""
                **模型命名规范**:
                - 格式: `rl_agent_股票_算法.zip`
                - 示例: `rl_agent_AAPL_PPO.zip`

                **操作流程**:
                1. 选择要使用的模型
                2. 点击'加载模型'
                3. 前往'回测'或'交易'页面

                **注意事项**:
                - 模型需要对应股票的数据
                - 删除操作不可恢复
                - 建议定期备份重要模型
                """)

with tab3:
    st.subheader("回测分析")

    if st.session_state.rl_agent is None:
        st.warning("⚠️ 请先在'训练 Agent'标签页训练模型")
    else:
        col1, col2 = st.columns([3, 1])

        with col2:
            st.markdown("### 回测设置")

            backtest_balance = st.number_input(
                "初始资金",
                min_value=1000,
                value=10000,
                step=1000
            )

            if st.button("🔍 开始回测", use_container_width=True):
                with st.spinner("正在回测..."):
                    try:
                        results = st.session_state.rl_agent.backtest(
                            st.session_state.test_data,
                            initial_balance=backtest_balance
                        )
                        st.session_state.backtest_results = results
                        st.success("✅ 回测完成")
                    except Exception as e:
                        st.error(f"回测失败: {e}")

        with col1:
            if st.session_state.backtest_results:
                results = st.session_state.backtest_results
                metrics = results['metrics']

                # 显示指标
                st.markdown("### 📊 回测指标")

                col_a, col_b, col_c, col_d, col_e = st.columns(5)

                with col_a:
                    st.metric(
                        "总收益率",
                        f"{metrics['total_return'] * 100:.2f}%"
                    )

                with col_b:
                    st.metric(
                        "最终资产",
                        f"${metrics['final_value']:.2f}"
                    )

                with col_c:
                    st.metric(
                        "最大回撤",
                        f"{metrics['max_drawdown'] * 100:.2f}%"
                    )

                with col_d:
                    st.metric(
                        "夏普比率",
                        f"{metrics['sharpe_ratio']:.3f}"
                    )

                with col_e:
                    st.metric(
                        "交易次数",
                        f"{metrics['num_trades']}"
                    )

                st.markdown("---")

                # 资产曲线
                st.markdown("### 📈 资产曲线")

                fig = go.Figure()

                # 资产净值
                fig.add_trace(go.Scatter(
                    y=results['net_worths'],
                    name='资产净值',
                    line=dict(color='blue', width=2)
                ))

                # 标记买卖点
                buy_points = [i for i, a in enumerate(results['actions']) if a == 1]
                sell_points = [i for i, a in enumerate(results['actions']) if a == 2]

                if buy_points:
                    fig.add_trace(go.Scatter(
                        x=buy_points,
                        y=[results['net_worths'][i] for i in buy_points],
                        mode='markers',
                        name='买入',
                        marker=dict(color='green', size=10, symbol='triangle-up')
                    ))

                if sell_points:
                    fig.add_trace(go.Scatter(
                        x=sell_points,
                        y=[results['net_worths'][i] for i in sell_points],
                        mode='markers',
                        name='卖出',
                        marker=dict(color='red', size=10, symbol='triangle-down')
                    ))

                # 添加初始资产线
                fig.add_hline(
                    y=backtest_balance,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="初始资产"
                )

                fig.update_layout(
                    title="资产净值变化",
                    xaxis_title="交易日",
                    yaxis_title="资产 ($)",
                    height=400,
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True,  key="backtest_fig")
                # 交易记录
                st.markdown("### 📋 交易记录")

                if results['trades']:
                    trades_df = pd.DataFrame(results['trades'])
                    trades_df['step'] = trades_df['step'].astype(int)
                    trades_df['type'] = trades_df['type'].map({'buy': '买入', 'sell': '卖出'})

                    st.dataframe(
                        trades_df,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("未发生任何交易")

with tab4:
    st.subheader("实时交易信号")

    if st.session_state.rl_agent is None:
        st.warning("⚠️ 请先训练或加载一个模型")
        st.info("👉 可以在'训练 Agent'页面训练新模型，或在'模型管理'页面加载已有模型")
    else:
        # 显示当前使用的模型
        current_model = st.session_state.get('current_model', '刚训练的模型')
        current_ticker = st.session_state.get('current_ticker', 'Unknown')

        st.info(f"🤖 当前模型: **{current_model}** | 股票: **{current_ticker}**")

        # 选择股票（带自定义名称）
        ticker_options, ticker_display_list = create_ticker_options(available_tickers)

        selected_display = st.selectbox(
            "选择股票查看交易信号",
            ticker_display_list,
            key="signal_ticker"
        )
        signal_ticker = ticker_options[selected_display]

        if st.button("🔍 获取信号", use_container_width=True):
            with st.spinner("正在分析..."):
                try:
                    # 加载数据
                    engine = DataEngine(signal_ticker)
                    df = engine.load_processed_data()

                    if df is None:
                        st.error("数据加载失败")
                    else:
                        # 获取最近的数据用于预测
                        recent_data = df.tail(50)

                        # 获取交易信号
                        signal = st.session_state.rl_agent.get_trading_signals(
                            recent_data,
                            initial_balance=10000
                        )

                        # 显示信号
                        st.markdown("---")

                        col1, col2, col3 = st.columns([1, 2, 1])

                        with col1:
                            st.markdown("### 🎯 交易信号")

                            action = signal['action']
                            if action == 'BUY':
                                st.success(f"### 📈 {action}")
                                st.markdown("**建议：买入**")
                            elif action == 'SELL':
                                st.error(f"### 📉 {action}")
                                st.markdown("**建议：卖出**")
                            else:
                                st.info(f"### ⏸️ {action}")
                                st.markdown("**建议：持有/观望**")

                            st.metric("当前价格", f"${signal['price']:.2f}")
                            st.metric("置信度", signal['confidence'])

                        with col2:
                            st.markdown("### 📝 分析理由")
                            st.info(signal['reasoning'])

                            st.markdown("### 📊 当前市场状态")

                            latest = recent_data.iloc[-1]

                            indicators = {}
                            if 'RSI' in recent_data.columns:
                                indicators['RSI'] = f"{latest['RSI']:.2f}"
                            if 'MACD_12_26_9' in recent_data.columns:
                                indicators['MACD'] = f"{latest['MACD_12_26_9']:.4f}"
                            if 'SMA_10' in recent_data.columns:
                                indicators['SMA_10'] = f"${latest['SMA_10']:.2f}"
                            if 'SMA_50' in recent_data.columns:
                                indicators['SMA_50'] = f"${latest['SMA_50']:.2f}"

                            st.json(indicators)

                        with col3:
                            st.markdown("### ⚠️ 风险提示")
                            st.warning("""
                            - 仅供参考
                            - 不构成投资建议
                            - 请结合自己判断
                            - 注意风险控制
                            """)

                except Exception as e:
                    st.error(f"获取信号失败: {e}")

with tab5:
    st.subheader("性能对比")

    st.info("🚧 此功能正在开发中...")

    st.markdown("""
    未来版本将包含：
    - 📊 与买入持有策略对比
    - 📈 与其他 RL 算法对比
    - 🎯 不同参数设置的对比
    - 📉 多个股票的性能对比
    - 📋 多模型横向比较
    """)

    # 如果有多个模型，显示简单对比
    model_dir = Path("data/models")
    if model_dir.exists():
        model_files = list(model_dir.glob("*.zip"))
        if len(model_files) > 1:
            st.markdown("---")
            st.markdown(f"### 📊 已有 {len(model_files)} 个模型可用于对比")
            st.markdown("敬请期待完整的对比功能！")

# 底部说明
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
💡 提示: RL Agent 需要大量数据和训练时间才能学到有效策略<br>
建议使用至少2-3年的历史数据进行训练
</div>
""", unsafe_allow_html=True)