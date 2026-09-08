"""Local Streamlit dashboard for reproducible StockRL experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stockrl.data import load_csv, make_demo_data
from stockrl.env import TradingConfig
from stockrl.experiments import run_experiment


OUTPUT_ROOT = Path(os.environ.get("STOCKRL_OUTPUT_DIR", ROOT / "outputs" / "experiments")).expanduser()
BASELINES = {"cash": "现金", "buy_hold": "买入持有", "half": "固定半仓", "trend": "均线择时"}
METRICS = {
    "total_return": "净收益", "annualized_return": "年化收益", "volatility": "年化波动",
    "sharpe": "Sharpe", "max_drawdown": "最大回撤", "total_turnover": "累计换手",
    "total_cost": "累计费用", "average_weight": "平均仓位", "trade_count": "成交笔数",
}

st.set_page_config(page_title="StockRL 实验台", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    :root { --ink:#101820; --panel:#16232d; --paper:#e7edf2; --muted:#9aa9b8;
            --aqua:#61d3ba; --amber:#e5b567; --rule:#31424f; }
    .stApp { background:var(--ink); color:var(--paper); }
    h1,h2,h3 { font-family:Georgia,"Noto Serif SC",serif; letter-spacing:-.015em; }
    p,label,.stMarkdown { font-family:"Segoe UI","Microsoft YaHei",sans-serif; }
    [data-testid="stMetricValue"],code,.mono { font-family:Consolas,monospace; font-variant-numeric:tabular-nums; }
    [data-testid="stSidebar"] { background:#0c141b; border-right:1px solid var(--rule); }
    [data-testid="stMetric"] { background:var(--panel); border:1px solid var(--rule); border-radius:7px; padding:14px 16px; }
    .scope-line { color:var(--muted); max-width:70ch; margin:-.35rem 0 1.15rem; line-height:1.65; }
    .data-stamp { border-left:3px solid var(--aqua); background:var(--panel); padding:12px 16px; margin:8px 0 16px; }
    .demo-stamp { border-left-color:var(--amber); }
    .stButton>button,.stDownloadButton>button { min-height:44px; border-radius:5px; font-weight:600; }
    .stButton>button[kind="primary"],.stButton>button[kind="primary"] p,
    button[data-testid="stBaseButton-primary"],button[data-testid="stBaseButton-primary"] p { color:var(--ink)!important; }
    .stButton>button:focus-visible,.stDownloadButton>button:focus-visible { outline:3px solid var(--aqua); outline-offset:2px; }
    [data-baseweb="tab-list"] { gap:1.25rem; border-bottom:1px solid var(--rule); }
    [data-baseweb="tab"] { min-height:48px; }
    @media (prefers-reduced-motion:reduce) { * { scroll-behavior:auto!important; transition:none!important; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    st.session_state.setdefault("bars", None)
    st.session_state.setdefault("data_label", None)
    st.session_state.setdefault("summary", None)


def _clean_filename(name: str) -> str:
    return Path(name.replace("\\", "/")).name or "market.csv"


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.replace("，", ",").split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("随机种子请用逗号分隔整数，例如 42, 43, 44") from exc
    if not seeds:
        raise ValueError("至少填写一个随机种子")
    if len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError("随机种子必须互不重复，且在 0 到 2³²-1 之间")
    return seeds


def _data_stamp(label: str, rows: int) -> None:
    is_demo = label == "synthetic_demo"
    kind = "演示数据" if is_demo else "市场数据"
    css = "data-stamp demo-stamp" if is_demo else "data-stamp"
    note = "合成路径只能验证软件流程，不能证明策略盈利。" if is_demo else "请确认复权、分红和拆股口径一致。"
    st.markdown(
        f'<div class="{css}"><b>{kind}</b> · <span class="mono">{rows:,}</span> 行<br>'
        f'<span style="color:#9aa9b8">{note}</span></div>', unsafe_allow_html=True,
    )


def _metric_text(key: str, value) -> str:
    if value is None:
        return "未定义"
    if key in {"total_return", "annualized_return", "volatility", "max_drawdown", "total_turnover", "average_weight"}:
        return f"{float(value):.2%}"
    if key == "total_cost":
        return f"{float(value):,.2f}"
    if key == "trade_count":
        return f"{int(value):,}"
    return f"{float(value):.3f}"


def _history(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _trajectory_chart(run: dict) -> go.Figure:
    history = _history(run["history_path"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=history["date"], y=history["nav"], name="RL 净值",
                             line={"color": "#61d3ba", "width": 2.4}), secondary_y=False)
    colors = {"cash": "#778896", "buy_hold": "#e5b567", "half": "#8da0d8", "trend": "#c28fd7"}
    for name, artifact in run.get("baselines", {}).items():
        baseline = _history(artifact["history_path"])
        fig.add_trace(go.Scatter(x=baseline["date"], y=baseline["nav"], name=BASELINES.get(name, name),
                                 line={"color": colors.get(name, "#9aa9b8"), "width": 1.3, "dash": "dot"}),
                      secondary_y=False)
    fig.add_trace(go.Scatter(x=history["date"], y=history["weight"], name="RL 实际仓位",
                             line={"color": "#e7edf2", "width": 1.2}, fill="tozeroy",
                             fillcolor="rgba(231,237,242,.08)"), secondary_y=True)
    fig.update_yaxes(title_text="净值", tickformat=",.0f", gridcolor="#263642", secondary_y=False)
    fig.update_yaxes(title_text="实际仓位", tickformat=".0%", range=[0, 1.05], gridcolor="#263642", secondary_y=True)
    fig.update_xaxes(title_text="交易日", showgrid=False)
    fig.update_layout(height=510, hovermode="x unified", paper_bgcolor="#101820", plot_bgcolor="#101820",
                      font={"color": "#e7edf2", "family": "Segoe UI, Microsoft YaHei, sans-serif"},
                      legend={"orientation": "h", "y": 1.12, "x": 0},
                      margin={"l": 10, "r": 10, "t": 70, "b": 10})
    return fig


def _comparison_frame(run: dict) -> pd.DataFrame:
    rows = [{"策略": "RL", **run["metrics"]}]
    rows.extend({"策略": BASELINES.get(name, name), **artifact["metrics"]}
                for name, artifact in run.get("baselines", {}).items())
    frame = pd.DataFrame(rows).set_index("策略")
    for key in METRICS:
        if key not in frame:
            frame[key] = None
    return frame[list(METRICS)].rename(columns=METRICS)


def _saved_summaries() -> list[Path]:
    if not OUTPUT_ROOT.is_dir():
        return []
    return sorted(OUTPUT_ROOT.rglob("summary.json"), key=lambda path: path.stat().st_mtime, reverse=True)


def _load_saved_summary(path: Path) -> dict:
    """Load an experiment and rebase artifacts to its current directory."""
    summary = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent.resolve()
    runs = summary.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("summary.json 没有可显示的 seed 结果")
    for run in runs:
        seed_dir = root / f"seed_{int(run['seed'])}"
        for key, name in (("history_path", "history.csv"), ("trades_path", "trades.csv"),
                          ("model_path", "model.zip"), ("normalizer_path", "normalizer.json"),
                          ("validation_path", "evaluations.npz")):
            if key in run:
                run[key] = str(seed_dir / name)
        baselines = run.get("baselines", {})
        if not isinstance(baselines, dict):
            raise ValueError("summary.json 的基准结果格式无效")
        for name, artifact in baselines.items():
            if not isinstance(artifact, dict):
                raise ValueError(f"summary.json 的 {name} 基准格式无效")
            artifact_dir = seed_dir / "baselines" / name
            artifact["history_path"] = str(artifact_dir / "history.csv")
            artifact["trades_path"] = str(artifact_dir / "trades.csv")
    summary["output_dir"] = str(root)
    summary["data_path"] = str(root / "bars.csv")
    return summary


_init_state()

with st.sidebar:
    st.subheader("研究边界")
    st.markdown("日线 · 单资产 · 只做多")
    st.caption("收盘观察，下一交易日开盘执行；测试区间不参与训练或选模。")
    st.divider()
    if st.session_state["bars"] is None:
        st.caption("尚未载入数据")
    else:
        st.caption(f"当前数据：{st.session_state['data_label']}")
        st.caption(f"{len(st.session_state['bars']):,} 行")

st.title("StockRL 实验台")
st.markdown('<p class="scope-line">把行情、账户约束和随机种子固定下来，训练自主仓位策略，并在完全隔离的测试区间与四个基准逐日比较。</p>',
            unsafe_allow_html=True)

data_tab, train_tab, results_tab = st.tabs(["数据与环境", "训练实验", "结果对比"])

with data_tab:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("行情来源")
        source = st.selectbox("选择数据来源", ["演示数据", "上传 CSV", "本地样例"], key="data_source")
        if source == "演示数据":
            demo_rows = st.slider("演示数据行数", 81, 2000, 756, 25, key="demo_rows")
            demo_seed = st.number_input("行情随机种子", min_value=0, max_value=2**31 - 1, value=42, key="demo_seed")
            if st.button("载入演示数据", type="primary", width="stretch", key="load_demo"):
                st.session_state["bars"] = make_demo_data(int(demo_rows), int(demo_seed))
                st.session_state["data_label"] = "synthetic_demo"
                st.session_state["summary"] = None
                st.rerun()
        elif source == "上传 CSV":
            uploaded = st.file_uploader("CSV 文件", type=["csv"], key="csv_upload",
                                        help="必须包含 Date、Open、High、Low、Close、Volume；文件只在内存中读取。")
            if st.button("载入上传数据", type="primary", width="stretch", key="load_upload"):
                if uploaded is None:
                    st.error("请先选择 CSV 文件。")
                else:
                    try:
                        uploaded.seek(0)
                        st.session_state["bars"] = load_csv(uploaded)
                        st.session_state["data_label"] = f"uploaded:{_clean_filename(uploaded.name)}"
                        st.session_state["summary"] = None
                        st.rerun()
                    except ValueError as exc:
                        st.error(f"CSV 无法载入：{exc}")
        else:
            samples = sorted((ROOT / "stock_data").glob("*_raw.csv"))
            if samples:
                sample = st.selectbox("本地行情文件", samples, format_func=lambda path: path.name, key="local_sample")
                if st.button("载入本地样例", type="primary", width="stretch", key="load_local"):
                    try:
                        st.session_state["bars"] = load_csv(sample)
                        st.session_state["data_label"] = f"local:{sample.name}"
                        st.session_state["summary"] = None
                        st.rerun()
                    except ValueError as exc:
                        st.error(f"本地文件无法载入：{exc}")
            else:
                st.info("stock_data 中没有 *_raw.csv，可改用上传或演示数据。")
    with right:
        st.subheader("账户与交易")
        initial_cash = st.number_input("初始现金", min_value=100.0, value=10_000.0, step=1_000.0, key="initial_cash")
        commission = st.number_input("单边手续费率", min_value=0.0, max_value=0.1, value=0.001, format="%.4f", key="commission")
        slippage = st.number_input("方向性滑点率", min_value=0.0, max_value=0.1, value=0.0005, format="%.4f", key="slippage")
        sell_tax = st.number_input("卖出税率", min_value=0.0, max_value=0.1, value=0.0, format="%.4f", key="sell_tax")
        min_commission = st.number_input("每笔最低手续费", min_value=0.0, value=0.0, key="min_commission")
        lot_size = st.number_input("最小成交股数", min_value=1, value=1, step=1, key="lot_size")
        max_participation = st.number_input("最大历史成交量参与率", min_value=0.0001, max_value=1.0,
                                            value=0.01, format="%.4f", key="max_participation")

    bars = st.session_state["bars"]
    if bars is not None:
        _data_stamp(st.session_state["data_label"], len(bars))
        st.dataframe(bars.tail(12), width="stretch")

with train_tab:
    st.subheader("训练设置")
    col1, col2, col3 = st.columns(3)
    with col1:
        algorithm = st.selectbox("算法", ["PPO", "SAC"], key="algorithm")
        timesteps = st.number_input("训练步数", min_value=2, value=10_000, step=1_000, key="timesteps")
    with col2:
        seeds_text = st.text_input("随机种子", value="42", key="seeds", help="多个种子用逗号分隔。")
        episode_length = st.number_input("每个回合最大步数", min_value=1, value=126, key="episode_length")
    with col3:
        train_ratio = st.slider("训练区间占比", 0.40, 0.80, 0.60, 0.05, key="train_ratio")
        validation_ratio = st.slider("验证区间占比", 0.10, 0.35, 0.20, 0.05, key="validation_ratio")
    st.caption("剩余日期自动作为测试区间；训练、验证、测试只共享边界观察，不共享收益区间。")

    can_train = st.session_state["bars"] is not None
    if st.button("运行实验", type="primary", disabled=not can_train, key="run_experiment"):
        try:
            seeds = _parse_seeds(seeds_text)
            if train_ratio + validation_ratio >= 1:
                raise ValueError("训练与验证区间占比之和必须小于 1")
            config = TradingConfig(initial_cash=float(initial_cash), commission=float(commission),
                                   slippage=float(slippage), sell_tax=float(sell_tax),
                                   min_commission=float(min_commission), lot_size=int(lot_size),
                                   max_participation=float(max_participation))
            progress = st.progress(0, text="准备实验")

            def on_progress(event: dict) -> None:
                if event.get("event") == "training":
                    ratio = min(0.9, float(event.get("timesteps", 0)) / max(float(timesteps), 1.0) / len(seeds))
                    progress.progress(ratio, text=f"训练 seed {event.get('seed')}")
                elif event.get("event") == "seed_complete":
                    done = seeds.index(event["seed"]) + 1 if event.get("seed") in seeds else 1
                    progress.progress(min(0.95, done / len(seeds)), text=f"已完成 {done}/{len(seeds)} 个种子")
                elif event.get("event") == "complete":
                    progress.progress(1.0, text="实验完成")

            st.session_state["summary"] = run_experiment(
                st.session_state["bars"], OUTPUT_ROOT, algorithm=algorithm, timesteps=int(timesteps),
                seeds=seeds, config=config, train_ratio=float(train_ratio), val_ratio=float(validation_ratio),
                data_label=st.session_state["data_label"], episode_length=int(episode_length),
                progress_callback=on_progress)
            st.success(f"实验已保存：{st.session_state['summary']['run_id']}")
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(f"实验未运行：{exc}")
    if not can_train:
        st.info("先在“数据与环境”中载入一份行情。")

with results_tab:
    st.subheader("样本外结果")
    saved = _saved_summaries()
    if saved:
        selected = st.selectbox("已保存实验", saved,
                                format_func=lambda path: str(path.parent.relative_to(OUTPUT_ROOT)), key="saved_summary")
        if st.button("载入已保存结果", key="load_saved"):
            try:
                st.session_state["summary"] = _load_saved_summary(selected)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                st.error(f"结果无法载入：{exc}")

    summary = st.session_state["summary"]
    if summary is None:
        st.info("运行实验或载入已有实验后，这里会显示测试区间结果。")
    else:
        if summary.get("data_label") == "synthetic_demo":
            st.warning("当前为合成演示数据。收益数字只验证流程，不代表任何市场表现。")
        try:
            runs = summary["runs"]
            if not isinstance(runs, list) or not runs:
                raise ValueError("没有可显示的 seed 结果")
            seed_options = [run["seed"] for run in runs]
            selected_seed = st.selectbox("查看随机种子", seed_options, key="result_seed")
            run = next(item for item in runs if item["seed"] == selected_seed)
            metrics = run["metrics"]
            trajectory = _trajectory_chart(run)
            comparison = _comparison_frame(run)
            csv_data = _history(run["history_path"]).to_csv(index=False).encode("utf-8-sig")
        except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
            st.error(f"结果文件无法读取：{exc}。请恢复该实验的完整产物目录后重试。")
        else:
            first_row = st.columns(3)
            second_row = st.columns(2)
            cards = [*first_row, *second_row]
            for column, key in zip(cards, ("total_return", "annualized_return", "sharpe", "max_drawdown", "total_cost")):
                column.metric(METRICS[key], _metric_text(key, metrics.get(key)))
            st.plotly_chart(trajectory, width="stretch", key=f"trajectory_{selected_seed}")
            st.caption("净值和实际仓位使用同一时间轴；所有基准复用相同测试区间、费用和成交约束。")
            st.dataframe(comparison, width="stretch")
            st.download_button("下载 RL 逐日记录", csv_data,
                               file_name=f"{summary['run_id']}-seed-{selected_seed}-history.csv",
                               mime="text/csv", key="download_history")

st.divider()
st.markdown("本工具仅用于强化学习交易研究与软件验证，不连接券商，不构成投资建议。")
