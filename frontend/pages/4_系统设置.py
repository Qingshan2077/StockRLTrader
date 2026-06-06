import streamlit as st
from frontend.v3_utils import apply_theme, section_header, metric_tile

st.set_page_config(
    page_title="系统设置",
    page_icon="▸",
    layout="wide"
)
apply_theme()
st.markdown('<h1 style="font-family:Noto Serif SC,serif;font-weight:600;color:#e8e4d9;border-bottom:1px solid #252a35;padding-bottom:0.6rem"><span style="color:#c9a84c">▸</span> 系统设置</h1>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["系统信息", "配置管理", "关于"])

with tab1:
    section_header("系统概况")
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_tile("系统版本", "v3.0", "Trading Terminal", "gold")
    with col2:
        metric_tile("架构", "三层量化", "信号→风险→执行", "steel")
    with col3:
        metric_tile("状态", "运行中", "开发阶段", "bullish")

    section_header("项目路径")
    st.code("""
StockTrader/
├── frontend/          # Streamlit 前端
│   ├── app.py         # 主页面
│   ├── pages/         # 功能页面 (1-11)
│   └── v3_utils.py    # 主题引擎 + 组件库
├── layers/            # 新分层系统
│   ├── signals/       # 信号层 (LightGBM/XGBoost/MLP...)
│   ├── features/      # 特征工程
│   ├── evaluation/    # 信号评估
│   └── ensemble/      # 模型融合
├── stock_data/        # 股票数据存储
├── config/            # 配置文件
└── models/            # 模型文件
    """)

with tab2:
    section_header("配置文件")
    st.info("配置文件位于 config/ 目录，支持 YAML 格式的系统参数配置。")
    st.markdown("""
    **可配置项**:
    - 数据源参数（起始日期、缓存格式）
    - 模型超参数（学习率、隐藏层、epochs）
    - 标签参数（预测天数、分类/回归）
    - 回测参数（初始资金、手续费率）
    """)

with tab3:
    section_header("关于项目")
    st.markdown("""
    ### AI 股票交易系统 v3.0

    基于三层量化架构的 AI 股票交易系统：

    **层1 — 信号层**: 监督学习模型（LightGBM/XGBoost/MLP/LSTM/GRU/Transformer）
    预测未来收益率，生成 signal_score

    **层2 — 风险与约束层**: 仓位限制、止损、波动率降仓
    输出 target_position_after_risk

    **层3 — RL 执行优化层**: 强化学习优化调仓节奏
    输出 execution_ratio ∈ [0,1]

    ---
    *仅供学习研究使用，不构成投资建议*
    """)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#7a7570;font-size:0.75rem;letter-spacing:1px;font-family:JetBrains Mono,monospace">AI 股票交易系统 v3.0 · Trading Terminal Noir</div>', unsafe_allow_html=True)
