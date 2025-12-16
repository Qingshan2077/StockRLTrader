import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import json
from datetime import datetime

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from improved_data_engine import DataEngine, BatchDataEngine

st.set_page_config(
    page_title="数据管理",
    page_icon="📊",
    layout="wide"
)

st.title("📊 数据管理中心")


# 加载配置
@st.cache_data
def load_config():
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {"data": {"directory": "stock_data"}}


config = load_config()
data_dir = config['data']['directory']

# 标签页
tab1, tab2, tab3, tab4, tab5 = st.tabs(["➕ 添加股票", "🔄 更新数据", "📋 数据列表", "✏️ 股票命名", "🗑️ 数据管理"])

with tab1:
    st.subheader("添加新股票数据")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 单个添加")
        ticker_input = st.text_input(
            "输入股票代码",
            placeholder="例如: AAPL, MSFT, NVDA",
            key="single_ticker"
        )

        start_date = st.date_input(
            "起始日期",
            value=pd.to_datetime("2015-01-01"),
            key="start_date"
        )

        if st.button("📥 下载数据", type="primary", use_container_width=True):
            if ticker_input:
                ticker = ticker_input.upper().strip()
                with st.spinner(f"正在下载 {ticker} 的数据..."):
                    try:
                        engine = DataEngine(
                            ticker,
                            data_dir=data_dir,
                            start_date=str(start_date)
                        )
                        df = engine.fetch_data()
                        df = engine.add_technical_indicators()

                        st.success(f"✅ 成功下载 {ticker} 的 {len(df)} 条数据")

                        # 显示预览
                        with st.expander("数据预览"):
                            st.dataframe(df.tail(10))
                    except Exception as e:
                        st.error(f"下载失败: {e}")
            else:
                st.warning("请输入股票代码")

    with col2:
        st.markdown("### 快速添加预设")

        presets = {
            "科技巨头": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "AI芯片": ["NVDA", "AMD", "INTC", "QCOM"],
            "电动车": ["TSLA", "NIO", "XPEV", "LI"],
            "中概股": ["BABA", "BIDU", "JD", "PDD"]
        }

        selected_preset = st.selectbox("选择预设列表", list(presets.keys()))

        st.info(f"包含: {', '.join(presets[selected_preset])}")

        if st.button("📥 批量下载", use_container_width=True):
            tickers = presets[selected_preset]
            batch_engine = BatchDataEngine(data_dir=data_dir)

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, ticker in enumerate(tickers):
                status_text.text(f"正在处理 {ticker}... ({i + 1}/{len(tickers)})")
                try:
                    batch_engine.process_ticker(ticker)
                except Exception as e:
                    st.error(f"{ticker} 失败: {e}")
                progress_bar.progress((i + 1) / len(tickers))

            status_text.text("✅ 批量下载完成！")

    st.markdown("---")
    st.markdown("### 🔤 批量添加（多个股票）")

    batch_input = st.text_area(
        "输入多个股票代码（用空格、逗号或换行分隔）",
        placeholder="AAPL MSFT NVDA\n或\nAAPL, MSFT, NVDA",
        height=100
    )

    if st.button("📥 批量下载自定义列表", type="primary"):
        if batch_input:
            # 处理输入，支持多种分隔符
            import re

            tickers = re.split(r'[,\s\n]+', batch_input.upper().strip())
            tickers = [t for t in tickers if t]  # 过滤空字符串

            if tickers:
                st.info(f"准备下载 {len(tickers)} 个股票: {', '.join(tickers)}")

                batch_engine = BatchDataEngine(data_dir=data_dir)

                progress_bar = st.progress(0)
                status_text = st.empty()
                results = {"成功": [], "失败": []}

                for i, ticker in enumerate(tickers):
                    status_text.text(f"正在处理 {ticker}... ({i + 1}/{len(tickers)})")
                    try:
                        if batch_engine.process_ticker(ticker):
                            results["成功"].append(ticker)
                        else:
                            results["失败"].append(ticker)
                    except Exception as e:
                        results["失败"].append(f"{ticker} ({e})")

                    progress_bar.progress((i + 1) / len(tickers))

                # 显示结果
                status_text.empty()
                progress_bar.empty()

                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"✅ 成功: {len(results['成功'])} 个")
                    if results["成功"]:
                        st.write(", ".join(results["成功"]))

                with col2:
                    if results["失败"]:
                        st.error(f"❌ 失败: {len(results['失败'])} 个")
                        st.write(", ".join(results["失败"]))
        else:
            st.warning("请输入股票代码")

with tab2:
    st.subheader("更新现有数据")

    batch_engine = BatchDataEngine(data_dir=data_dir)
    available_tickers = batch_engine.list_available_data()

    if not available_tickers:
        st.warning("暂无本地数据，请先添加股票")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"### 本地股票列表 ({len(available_tickers)} 个)")

            # 显示所有股票的最后更新时间
            update_info = []
            for ticker in available_tickers:
                engine = DataEngine(ticker, data_dir=data_dir)
                meta = engine._load_metadata()
                custom_name = meta.get("custom_name", "")
                update_info.append({
                    "股票名称": custom_name if custom_name else "-",
                    "股票代码": ticker,
                    "最后更新": meta.get("last_update", "未知"),
                    "数据点数": meta.get("data_points", 0),
                    "日期范围": f"{meta.get('date_range', {}).get('start', '')[:10]} ~ {meta.get('date_range', {}).get('end', '')[:10]}"
                })

            df_info = pd.DataFrame(update_info)
            st.dataframe(df_info, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("### 更新操作")

            if st.button("🔄 更新所有股票", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, ticker in enumerate(available_tickers):
                    status_text.text(f"正在更新 {ticker}... ({i + 1}/{len(available_tickers)})")
                    try:
                        batch_engine.process_ticker(ticker)
                    except Exception as e:
                        st.error(f"{ticker} 更新失败: {e}")
                    progress_bar.progress((i + 1) / len(available_tickers))

                status_text.text("✅ 全部更新完成！")
                st.rerun()

            st.markdown("---")

            selected_tickers = st.multiselect(
                "选择要更新的股票",
                available_tickers
            )

            if st.button("🔄 更新选中股票", use_container_width=True):
                if selected_tickers:
                    for ticker in selected_tickers:
                        with st.spinner(f"正在更新 {ticker}..."):
                            try:
                                batch_engine.process_ticker(ticker)
                                st.success(f"✅ {ticker} 更新完成")
                            except Exception as e:
                                st.error(f"❌ {ticker} 更新失败: {e}")
                    st.rerun()
                else:
                    st.warning("请先选择股票")

with tab3:
    st.subheader("本地数据详情")

    batch_engine = BatchDataEngine(data_dir=data_dir)
    available_tickers = batch_engine.list_available_data()

    if not available_tickers:
        st.warning("暂无本地数据")
    else:
        # 创建显示选项（包含自定义名称）
        ticker_options = {}
        for ticker in available_tickers:
            engine = DataEngine(ticker, data_dir=data_dir)
            meta = engine._load_metadata()
            custom_name = meta.get('custom_name', '')
            display_name = f"{custom_name} ({ticker})" if custom_name else ticker
            ticker_options[display_name] = ticker

        selected_display = st.selectbox(
            "选择股票查看详情",
            list(ticker_options.keys())
        )

        selected_ticker = ticker_options[selected_display]

        if selected_ticker:
            engine = DataEngine(selected_ticker, data_dir=data_dir)
            meta = engine.get_info()

            # 显示元数据
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                custom_name = meta.get("custom_name", "")
                st.metric("股票名称", custom_name if custom_name else "未设置")
            with col2:
                st.metric("股票代码", selected_ticker)
            with col3:
                st.metric("数据点数", meta.get("data_points", 0))
            with col4:
                date_range = meta.get("date_range", {})
                st.metric("起始日期", date_range.get("start", "")[:10] if date_range.get("start") else "未知")
            with col5:
                st.metric("结束日期", date_range.get("end", "")[:10] if date_range.get("end") else "未知")

            st.markdown("---")

            # 加载并显示数据
            df = engine.load_processed_data()

            if df is not None:
                # 数据统计
                st.markdown("### 📈 数据统计")

                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("平均价格", f"${df['Close'].mean():.2f}")
                with col2:
                    st.metric("最高价", f"${df['Close'].max():.2f}")
                with col3:
                    st.metric("最低价", f"${df['Close'].min():.2f}")
                with col4:
                    st.metric("价格波动", f"{df['Close'].std():.2f}")
                with col5:
                    avg_volume = df['Volume'].mean() / 1e6
                    st.metric("平均成交量", f"{avg_volume:.1f}M")

                st.markdown("---")

                # 数据预览
                st.markdown("### 📋 数据预览（最近20条）")

                display_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if 'RSI' in df.columns:
                    display_cols.append('RSI')
                if 'MACD_12_26_9' in df.columns:
                    display_cols.append('MACD_12_26_9')

                st.dataframe(
                    df[display_cols].tail(20).sort_index(ascending=False),
                    use_container_width=True
                )

                # 下载按钮
                csv = df.to_csv()
                st.download_button(
                    label=f"📥 下载 {selected_ticker} 完整数据",
                    data=csv,
                    file_name=f"{selected_ticker}_complete_data.csv",
                    mime="text/csv"
                )

with tab4:
    st.subheader("✏️ 股票命名管理")

    st.markdown("""
    为股票添加自定义名称，方便记忆和管理。

    **示例**:
    - AAPL → 苹果公司
    - NVDA → 英伟达
    - BABA → 阿里巴巴
    """)

    batch_engine = BatchDataEngine(data_dir=data_dir)
    available_tickers = batch_engine.list_available_data()

    if not available_tickers:
        st.warning("暂无本地数据，请先添加股票")
    else:
        st.markdown("---")

        # 显示当前所有股票的命名情况
        st.markdown("### 📋 当前命名列表")

        name_info = []
        for ticker in available_tickers:
            engine = DataEngine(ticker, data_dir=data_dir)
            meta = engine._load_metadata()
            custom_name = meta.get("custom_name", "")
            name_info.append({
                "股票代码": ticker,
                "自定义名称": custom_name if custom_name else "❌ 未设置",
                "状态": "✅ 已命名" if custom_name else "⚠️ 未命名"
            })

        df_names = pd.DataFrame(name_info)
        st.dataframe(df_names, use_container_width=True, hide_index=True)

        st.markdown("---")

        # 设置/修改名称
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### ✏️ 设置股票名称")

            # 选择要命名的股票
            ticker_to_name = st.selectbox(
                "选择股票",
                available_tickers,
                key="ticker_to_name",
                format_func=lambda x: f"{x} - {DataEngine(x, data_dir=data_dir).get_custom_name() or '未命名'}"
            )

            # 获取当前名称
            current_engine = DataEngine(ticker_to_name, data_dir=data_dir)
            current_name = current_engine.get_custom_name()

            # 输入新名称
            new_name = st.text_input(
                "输入自定义名称",
                value=current_name,
                placeholder="例如：苹果公司、英伟达、阿里巴巴",
                key="new_custom_name"
            )

            col_a, col_b = st.columns(2)

            with col_a:
                if st.button("💾 保存名称", type="primary", use_container_width=True):
                    if new_name and new_name.strip():
                        try:
                            current_engine.set_custom_name(new_name.strip())
                            st.success(f"✅ 已将 {ticker_to_name} 命名为: {new_name.strip()}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"保存失败: {e}")
                    else:
                        st.warning("请输入名称")

            with col_b:
                if st.button("🗑️ 清除名称", use_container_width=True):
                    try:
                        current_engine.set_custom_name("")
                        st.success(f"✅ 已清除 {ticker_to_name} 的自定义名称")
                        st.rerun()
                    except Exception as e:
                        st.error(f"清除失败: {e}")

        with col2:
            st.markdown("### 💡 命名建议")

            # 常见股票的默认名称
            common_names = {
                "AAPL": "苹果公司",
                "MSFT": "微软",
                "GOOGL": "谷歌",
                "AMZN": "亚马逊",
                "META": "Meta",
                "NVDA": "英伟达",
                "TSLA": "特斯拉",
                "AMD": "超微半导体",
                "INTC": "英特尔",
                "BABA": "阿里巴巴",
                "BIDU": "百度",
                "JD": "京东",
                "PDD": "拼多多",
                "NIO": "蔚来",
                "XPEV": "小鹏汽车",
                "LI": "理想汽车",
                "JPM": "摩根大通",
                "BAC": "美国银行",
                "V": "Visa",
                "MA": "万事达"
            }

            if ticker_to_name in common_names and not current_name:
                st.info(f"💡 推荐名称: **{common_names[ticker_to_name]}**")

                if st.button(f"使用推荐名称", key="use_suggested"):
                    try:
                        current_engine.set_custom_name(common_names[ticker_to_name])
                        st.success(f"✅ 已使用推荐名称")
                        st.rerun()
                    except Exception as e:
                        st.error(f"设置失败: {e}")

            st.markdown("---")
            st.markdown("**命名技巧**:")
            st.markdown("- 使用中文名称更直观")
            st.markdown("- 可以添加行业标签")
            st.markdown("- 例如：苹果-科技")
            st.markdown("- 例如：特斯拉-新能源")

        st.markdown("---")

        # 批量命名
        with st.expander("🔧 批量设置常见股票名称"):
            st.markdown("自动为常见股票设置中文名称")

            unnamed_common = [t for t in available_tickers if
                              t in common_names and not DataEngine(t, data_dir=data_dir).get_custom_name()]

            if unnamed_common:
                st.info(f"发现 {len(unnamed_common)} 个未命名的常见股票")

                for ticker in unnamed_common:
                    st.markdown(f"- {ticker} → {common_names[ticker]}")

                if st.button("🚀 一键批量命名", type="primary"):
                    success_count = 0
                    for ticker in unnamed_common:
                        try:
                            engine = DataEngine(ticker, data_dir=data_dir)
                            engine.set_custom_name(common_names[ticker])
                            success_count += 1
                        except Exception as e:
                            st.error(f"{ticker} 命名失败: {e}")

                    st.success(f"✅ 成功命名 {success_count} 个股票")
                    st.rerun()
            else:
                st.success("✅ 所有常见股票都已命名")

with tab5:
    st.subheader("数据管理")

    batch_engine = BatchDataEngine(data_dir=data_dir)
    available_tickers = batch_engine.list_available_data()

    if not available_tickers:
        st.warning("暂无本地数据")
    else:
        st.markdown("### 🗑️ 删除数据")
        st.warning("⚠️ 删除操作不可恢复，请谨慎操作！")

        col1, col2 = st.columns([2, 1])

        with col1:
            to_delete = st.multiselect(
                "选择要删除的股票",
                available_tickers
            )

        with col2:
            st.markdown("　")  # 占位
            st.markdown("　")
            if st.button("🗑️ 删除选中数据", type="secondary", use_container_width=True):
                if to_delete:
                    confirm = st.checkbox("我确认要删除这些数据")
                    if confirm:
                        for ticker in to_delete:
                            try:
                                data_path = Path(data_dir)
                                files = [
                                    data_path / f"{ticker}_raw.csv",
                                    data_path / f"{ticker}_processed.csv",
                                    data_path / f"{ticker}_meta.json"
                                ]
                                for f in files:
                                    if f.exists():
                                        f.unlink()
                                st.success(f"✅ 已删除 {ticker}")
                            except Exception as e:
                                st.error(f"❌ 删除 {ticker} 失败: {e}")
                        st.rerun()
                    else:
                        st.info("请勾选确认框")
                else:
                    st.warning("请先选择要删除的股票")

        st.markdown("---")

        # 清空所有数据
        st.markdown("### 🚨 危险操作")
        with st.expander("清空所有数据"):
            st.error("这将删除所有已下载的股票数据！")

            confirm_all = st.text_input(
                "输入 'DELETE ALL' 确认",
                key="confirm_delete_all"
            )

            if st.button("🚨 清空所有数据", type="secondary"):
                if confirm_all == "DELETE ALL":
                    try:
                        data_path = Path(data_dir)
                        for f in data_path.glob("*.csv"):
                            f.unlink()
                        for f in data_path.glob("*.json"):
                            f.unlink()
                        st.success("✅ 所有数据已清空")
                        st.rerun()
                    except Exception as e:
                        st.error(f"清空失败: {e}")
                else:
                    st.warning("请输入正确的确认文本")

# 页面底部
st.markdown("---")
st.info("💡 提示: 建议定期更新数据以保持最新状态")