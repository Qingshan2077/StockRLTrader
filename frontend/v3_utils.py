"""
v3 前端核心 — 主题引擎 + 公共组件

设计方向: 「Trading Terminal Noir」
深色专业交易终端美学, 金属质感强调色, 几何精密排版
"""
import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==================== 路径设置 ====================

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ============================================================
#   主题引擎
# ============================================================

THEME = """
<style>
/* ---------- 导入字体 ---------- */
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=JetBrains+Mono:wght@300;400;600&family=Inter:wght@300;400;500;600&display=swap');

/* ============================================================
   根变量 — 暗黑交易终端美学
   参考: 股息投资计算器.html 的深色+金色设计语言
   ============================================================ */
:root {
    --bg-root: #0a0c10;
    --bg-surface: #111318;
    --bg-card: #181c24;
    --bg-elevated: #1e2330;
    --border-subtle: #252a35;
    --border-accent: #2e3545;
    --text-primary: #e8e4d9;
    --text-secondary: #b0aa9e;
    --text-muted: #7a7570;
    --accent-gold: #c9a84c;
    --accent-gold-light: #e8c97a;
    --accent-gold-dim: #8a6e2f;
    --accent-steel: #6b8ba4;
    --accent-steel-dim: #3d5363;
    --bullish: #4caf82;
    --bullish-dim: #2d6b4f;
    --bearish: #e07060;
    --bearish-dim: #5c2020;
    --neutral: #f39c12;
    --info: #5b8dd9;
    --radius-sm: 2px;
    --radius-md: 4px;
    --radius-lg: 8px;
    --shadow-card: 0 2px 12px rgba(0,0,0,0.4);
    --shadow-elevated: 0 4px 24px rgba(0,0,0,0.6);
    --font-serif: 'Noto Serif SC', 'Source Han Serif SC', serif;
    --font-mono: 'JetBrains Mono', 'Cascadia Code', monospace;
    --font-sans: 'Inter', -apple-system, sans-serif;
}

/* ---------- 全局背景 ---------- */
.stApp {
    background: var(--bg-root);
}
.stApp > header {
    background: transparent !important;
    backdrop-filter: none;
}

/* ---------- 主体排版 ---------- */
.stMarkdown, .stText, p, label, .stSelectbox label, .stSlider label {
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
}
h1, h2, h3, h4 {
    font-family: var(--font-serif) !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    letter-spacing: 0;
}
h1 {
    font-size: 1.8rem !important;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 0.6rem;
    margin-bottom: 1.5rem;
}
h2 {
    font-size: 1.2rem !important;
    color: var(--accent-gold) !important;
    font-weight: 400 !important;
    letter-spacing: 1px;
}
h3 {
    font-size: 1rem !important;
    color: var(--text-secondary) !important;
    font-weight: 400 !important;
}

/* ---------- 侧边栏 ---------- */
section[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
section[data-testid="stSidebar"] .stMarkdown {
    color: var(--text-secondary);
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    font-weight: 400 !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted) !important;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--border-subtle) !important;
}

/* ---------- 按钮 — 金色边框 + 滑入感 ---------- */
.stButton > button {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    transition: all 0.25s ease;
    padding: 0.45rem 1.2rem !important;
}
.stButton > button:hover {
    border-color: var(--accent-gold) !important;
    color: var(--accent-gold) !important;
    box-shadow: 0 0 16px rgba(201, 168, 76, 0.12);
}
.stButton > button[kind="primary"] {
    background: rgba(201, 168, 76, 0.08) !important;
    border-color: var(--accent-gold) !important;
    color: var(--accent-gold-light) !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--accent-gold) !important;
    color: var(--bg-root) !important;
}
.stButton > button[kind="secondary"] {
    border-color: var(--bearish-dim) !important;
    color: var(--bearish) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(224, 112, 96, 0.1) !important;
}

/* ---------- 输入控件 — 等宽字体数值感 ---------- */
.stSelectbox div[data-baseweb="select"] > div,
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: var(--bg-root) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stSelectbox div[data-baseweb="select"] > div:focus,
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--accent-gold-dim) !important;
    box-shadow: 0 0 0 3px rgba(201, 168, 76, 0.06) !important;
}
.stSlider > div > div > div {
    background: var(--accent-gold-dim) !important;
}

/* ---------- 数据表 — 交易终端风格 ---------- */
.stDataFrame {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
}
.stDataFrame th {
    background: var(--bg-elevated) !important;
    color: var(--text-muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border-subtle) !important;
}
.stDataFrame td {
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
}

/* ---------- Tabs — 金色下划线 ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid var(--border-subtle);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 1px;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.6rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-gold) !important;
    border-bottom: 2px solid var(--accent-gold) !important;
}

/* ---------- 进度条 ---------- */
.stProgress > div > div {
    background: var(--accent-gold-dim) !important;
}

/* ---------- 信息/警告/成功/错误 — 左侧色带 ---------- */
.stAlert {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-subtle) !important;
    background: var(--bg-card) !important;
    font-family: var(--font-sans) !important;
}
.stSuccess { border-left: 3px solid var(--bullish) !important; }
.stWarning { border-left: 3px solid var(--neutral) !important; }
.stError   { border-left: 3px solid var(--bearish) !important; }
.stInfo    { border-left: 3px solid var(--info) !important; }

/* ---------- Metric 原生组件 ---------- */
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}
[data-testid="stMetricDelta"] {
    font-family: var(--font-mono) !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted) !important;
}

/* ---------- Expander ---------- */
.streamlit-expanderHeader {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 1px;
    color: var(--text-muted) !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
}

/* ---------- 分割线 ---------- */
hr {
    border-color: var(--border-subtle) !important;
}

/* ---------- 多选下拉 ---------- */
.stMultiSelect div[data-baseweb="select"] > div {
    background: var(--bg-root) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
}

/* ---------- Checkbox ---------- */
.stCheckbox label {
    color: var(--text-secondary) !important;
    font-family: var(--font-sans) !important;
}

/* ---------- 滚动条 ---------- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-root); }
::-webkit-scrollbar-thumb {
    background: var(--border-accent);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--accent-steel-dim); }
</style>
"""


def apply_theme():
    """注入黑色主题 CSS"""
    st.markdown(THEME, unsafe_allow_html=True)


# ============================================================
#   Plotly 图表模板
# ============================================================

PLOTLY_DARK = go.layout.Template()
PLOTLY_DARK.layout.update(
    paper_bgcolor="#111318",
    plot_bgcolor="#111318",
    font=dict(family="Inter, sans-serif", color="#b0aa9e", size=12),
    title=dict(font=dict(family="Noto Serif SC, serif", size=16, color="#e8e4d9")),
    xaxis=dict(
        gridcolor="#252a35", linecolor="#2e3545",
        zerolinecolor="#252a35", color="#7a7570",
    ),
    yaxis=dict(
        gridcolor="#252a35", linecolor="#2e3545",
        zerolinecolor="#2e3545", color="#7a7570",
    ),
    legend=dict(font=dict(color="#7a7570")),
    colorway=["#c9a84c", "#6b8ba4", "#4caf82", "#e07060",
              "#f39c12", "#5b8dd9", "#9b6ddb", "#5bc4b8"],
    bargap=0.25,
)

PLOTLY_COLORS = {
    "gold": "#c9a84c",
    "steel": "#6b8ba4",
    "bullish": "#4caf82",
    "bearish": "#e07060",
    "neutral": "#f39c12",
    "info": "#5b8dd9",
}


def dark_figure(fig: go.Figure, height: int = 350) -> go.Figure:
    """统一图表暗色样式"""
    fig.update_layout(
        template=PLOTLY_DARK,
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
    )
    return fig


# ============================================================
#   UI 组件
# ============================================================

def metric_tile(label: str, value: str, sub: str = "",
                accent: str = "steel"):
    """
    暗黑金属质感指标卡片 — 左侧色带 + 深色面板
    accent: gold | steel | bullish | bearish | neutral
    """
    colors = {
        "gold": ("#c9a84c", "#8a6e2f"),
        "steel": ("#6b8ba4", "#3d5363"),
        "bullish": ("#4caf82", "#2d6b4f"),
        "bearish": ("#e07060", "#5c2020"),
        "neutral": ("#f39c12", "#6b4c10"),
    }
    c1, c2 = colors.get(accent, colors["steel"])
    sub_html = f'<div style="font-size:0.7rem;color:{c1};margin-top:5px;opacity:0.75;letter-spacing:0.3px">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div style="
      background:linear-gradient(165deg, {c2}18 0%, #181c24 100%);
      border:1px solid #252a35;
      border-left:3px solid {c1};
      border-radius:2px;
      padding:14px 16px;
      font-family:'JetBrains Mono',monospace;
    ">
      <div style="font-size:0.65rem;color:#7a7570;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px">{label}</div>
      <div style="font-size:1.3rem;font-weight:600;color:#e8e4d9;letter-spacing:0.5px">{value}</div>
      {sub_html}
    </div>""", unsafe_allow_html=True)


def section_header(title: str, subtitle: str = ""):
    """区域标题 — 金色文字 + 底部虚线"""
    sub = f'<span style="color:#7a7570;font-size:0.8rem;font-weight:300;margin-left:12px;font-family:Inter,sans-serif">{subtitle}</span>' if subtitle else ""
    st.markdown(f"""
    <div style="display:flex;align-items:baseline;margin:1.2rem 0 0.6rem 0;padding-bottom:0.5rem;border-bottom:1px solid #252a35">
      <span style="font-family:'JetBrains Mono',monospace;font-size:1.05rem;color:#c9a84c;letter-spacing:1px;font-weight:400">{title}</span>
      {sub}
    </div>""", unsafe_allow_html=True)


def empty_state(message: str = None, icon: str = "—"):
    """空状态引导"""
    st.markdown(f"""
    <div style="text-align:center;padding:3.5rem 1rem;color:#7a7570">
      <div style="font-size:2.5rem;margin-bottom:1rem;opacity:0.5">{icon}</div>
      <div style="font-family:'Noto Serif SC',serif;font-size:0.9rem;letter-spacing:0.5px">{message or '暂无数据'}</div>
    </div>""", unsafe_allow_html=True)


def status_badge(text: str, kind: str = "neutral"):
    """状态标签"""
    clr = {"ok": "#4caf82", "warn": "#f39c12", "err": "#e07060", "neutral": "#6b8ba4"}.get(kind, "#6b8ba4")
    bg = {"ok": "#2d6b4f", "warn": "#6b4c10", "err": "#5c2020", "neutral": "#3d5363"}.get(kind, "#3d5363")
    st.markdown(f'<span style="display:inline-block;background:{bg};color:{clr};padding:2px 10px;border-radius:2px;font-family:JetBrains Mono,monospace;font-size:0.7rem;letter-spacing:0.8px;text-transform:uppercase">{text}</span>', unsafe_allow_html=True)


def benchmark_rating(metric: str, value: float) -> str:
    """
    量化指标质量评级
    返回: (评级标签, 颜色)

    阈值参考:
      IC/RankIC — 量化圈通用标准 (Grinold & Kahn)
      Sharpe/Sortino — 投资行业惯例
      回撤/胜率 — 风控最佳实践
    """
    # ---- 信号质量 (IC 体系) ----
    if metric in ("IC", "RankIC"):
        if value >= 0.10:   return ("★ 顶级", "#c9a84c")
        elif value >= 0.05: return ("★ 优秀", "#4caf82")
        elif value >= 0.03: return ("▲ 可用", "#6b8ba4")
        elif value >= 0.01: return ("△ 弱", "#f39c12")
        elif value >= 0:     return ("— 微弱", "#7a7570")
        else:                return ("↓ 反向", "#e07060")
    # IC IR (Information Ratio)
    if metric == "IC_IR":
        if value >= 1.0:    return ("★ 顶级", "#c9a84c")
        elif value >= 0.5:  return ("★ 稳定", "#4caf82")
        elif value >= 0.3:  return ("▲ 一般", "#6b8ba4")
        else:               return ("△ 波动大", "#f39c12")
    # IC t-stat
    if metric == "IC_tstat":
        if value >= 3.0:    return ("★ 极显著", "#c9a84c")
        elif value >= 2.0:  return ("★ 显著", "#4caf82")
        elif value >= 1.2:  return ("▲ 弱显著", "#6b8ba4")
        else:               return ("△ 不显著", "#f39c12")
    # ---- 模型拟合 ----
    if metric == "R2":
        if value >= 0.10:    return ("★ 优秀", "#c9a84c")
        elif value >= 0.05:  return ("★ 良好", "#4caf82")
        elif value >= 0.01:  return ("▲ 微弱", "#6b8ba4")
        elif value >= 0:     return ("△ 几乎零", "#f39c12")
        else:                return ("↓ 负值", "#e07060")
    # ---- 回测收益 ----
    if metric == "annualized_return":
        if value >= 0.30:   return ("★ 优秀", "#c9a84c")
        elif value >= 0.15: return ("★ 良好", "#4caf82")
        elif value >= 0.05: return ("▲ 一般", "#6b8ba4")
        elif value >= 0:    return ("△ 偏低", "#f39c12")
        else:               return ("↓ 亏损", "#e07060")
    # ---- 夏普比率 ----
    if metric == "sharpe":
        if value >= 2.0:    return ("★ 顶级", "#c9a84c")
        elif value >= 1.0:  return ("★ 良好", "#4caf82")
        elif value >= 0.5:  return ("▲ 一般", "#6b8ba4")
        elif value >= 0:    return ("△ 偏低", "#f39c12")
        else:               return ("↓ 负值", "#e07060")
    # ---- Sortino ----
    if metric == "sortino":
        if value >= 2.0:    return ("★ 顶级", "#c9a84c")
        elif value >= 1.0:  return ("★ 良好", "#4caf82")
        elif value >= 0.5:  return ("▲ 一般", "#6b8ba4")
        else:               return ("△ 偏低", "#f39c12")
    # ---- 最大回撤 ----
    if metric == "max_drawdown":
        if value >= -0.05:  return ("★ 极低风险", "#c9a84c")
        elif value >= -0.10: return ("★ 低风险", "#4caf82")
        elif value >= -0.20: return ("▲ 中等风险", "#6b8ba4")
        elif value >= -0.50: return ("△ 高风险", "#f39c12")
        else:                return ("↓ 极高风险", "#e07060")
    # ---- Calmar ----
    if metric == "calmar":
        if value >= 1.0:    return ("★ 优秀", "#c9a84c")
        elif value >= 0.5:  return ("★ 良好", "#4caf82")
        elif value >= 0.3:  return ("▲ 一般", "#6b8ba4")
        else:               return ("△ 偏低", "#f39c12")
    # ---- 胜率 ----
    if metric == "win_rate":
        if value >= 0.60:   return ("★ 高胜率", "#c9a84c")
        elif value >= 0.50: return ("★ 良好", "#4caf82")
        elif value >= 0.40: return ("▲ 一般", "#6b8ba4")
        elif value >= 0.30: return ("△ 偏低", "#f39c12")
        else:               return ("↓ 很低", "#e07060")
    # ---- 默认 ----
    return ("", "#7a7570")


def comparison_table(results: dict) -> pd.DataFrame:
    """三组回测 → DataFrame (含质量评级)"""
    cols = {
        "annualized_return": "年化收益率",
        "annualized_volatility": "年化波动率",
        "sharpe_ratio": "夏普比率",
        "sortino_ratio": "Sortino",
        "max_drawdown": "最大回撤",
        "calmar_ratio": "Calmar",
        "win_rate": "胜率",
        "turnover_rate": "换手率",
        "final_value": "最终资产",
    }
    # 指标到评级 key 的映射
    rating_key = {
        "annualized_return": "annualized_return",
        "sharpe_ratio": "sharpe",
        "sortino_ratio": "sortino",
        "max_drawdown": "max_drawdown",
        "calmar_ratio": "calmar",
        "win_rate": "win_rate",
    }
    rows = []
    for k, name in cols.items():
        row = {"指标": name}
        # 取第一个模式的数值做评级参考
        first_mode = next(iter(results.values()), {})
        first_val = first_mode.get(k, 0) if isinstance(first_mode, dict) else 0
        rating = benchmark_rating(rating_key.get(k, ""), float(first_val))[0] if k in rating_key else ""
        if rating:
            row["指标"] = f"{name} {rating}"

        for mode, m in results.items():
            v = m.get(k, 0)
            if isinstance(v, float):
                if "return" in k or "drawdown" in k or "rate" in k:
                    row[mode] = f"{v*100:.2f}%"
                elif "ratio" in k and "sharpe" not in k.lower() and "sortino" not in k.lower():
                    row[mode] = f"{v*100:.2f}%"
                elif "value" in k:
                    row[mode] = f"${v:.2f}"
                else:
                    row[mode] = f"{v:.3f}"
            else:
                row[mode] = str(v)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
#   Session 初始化
# ============================================================

def init_session():
    defaults = {
        "v3_config": None, "v3_ticker": None,
        "v3_features": None, "v3_labels": None,
        "v3_test_features": None, "v3_test_labels": None,
        "v3_signal_scores": None, "v3_models": {},
        "v3_ensemble": None, "v3_risk_params": None,
        "v3_backtest_results": None, "v3_rl_executor": None,
        "v3_training_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================
#   配置 & 数据加载
# ============================================================

@st.cache_resource
def load_v3_config() -> dict:
    try:
        from utils.config_loader import ConfigLoader
        return ConfigLoader(config_dir=str(_PROJECT_ROOT / "config")).to_dict()
    except Exception:
        return {}


def get_available_stocks(data_dir: str = "stock_data") -> list[str]:
    p = Path(_PROJECT_ROOT / data_dir)
    if not p.exists():
        return []
    stocks = set()
    for f in list(p.glob("*_raw.csv")) + list(p.glob("*_raw.parquet")):
        ticker = f.stem.replace("_raw", "").upper()
        stocks.add(ticker)
    return sorted(stocks)


def load_stock_data(ticker: str) -> pd.DataFrame | None:
    from layers.data.data_provider import MarketDataProvider
    cfg = load_v3_config()
    provider = MarketDataProvider(
        ticker,
        data_dir=str(_PROJECT_ROOT / cfg.get("system", {}).get("data_dir", "stock_data")),
        start_date=cfg.get("system", {}).get("start_date", "2015-01-01"),
        cache_format=cfg.get("market_data", {}).get("cache_format", "csv"),
    )
    df = provider.load_or_download()
    return df if (df is not None and not df.empty) else None


def load_features(ticker: str, df: pd.DataFrame = None):
    from layers.features import FeaturePipeline, FeatureRegistry, FeatureCache
    from layers.features.label_builder import LabelBuilder
    if df is None:
        df = load_stock_data(ticker)
        if df is None:
            return None
    registry = FeatureRegistry()
    cache = FeatureCache(data_dir=str(_PROJECT_ROOT / "stock_data"))
    pipeline = FeaturePipeline(registry=registry, cache=cache)
    features = pipeline.build_features(df, ticker=ticker, use_cache=True)
    cfg = load_v3_config()
    horizon = (cfg.get("label", {}).get("horizons") or [5])[0]
    lb = LabelBuilder(horizon=horizon)
    labels = lb.build_labels(features)
    features, labels = lb.align(features, labels)
    return features, labels, pipeline.registry
