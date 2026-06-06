"""
v3 · 实验管理 — 历史实验 + 对比 + 报告
"""
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from frontend.v3_utils import (
    init_session, apply_theme,
    metric_tile, section_header, empty_state, dark_figure,
)

init_session()
apply_theme()

st.set_page_config(page_title="实验管理", page_icon="▸", layout="wide")
st.markdown('<h1 style="font-family:Noto Serif SC,serif;font-weight:600;color:#e8e4d9;border-bottom:1px solid #252a35;padding-bottom:0.6rem"><span style="color:#c9a84c">▸</span> 实验管理</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div style="font-size:0.7rem;color:#7a7570;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:0.5rem">· 筛选</div>', unsafe_allow_html=True)
    status_filter = st.selectbox("状态", ["全部","running","completed","failed"])
    if st.button("▸ 刷新列表", use_container_width=True):
        st.rerun()

tab1, tab2 = st.tabs(["实验列表", "对比分析"])

with tab1:
    section_header("实验历史")
    try:
        from layers.experiments.manager import ExperimentManager
        mgr = ExperimentManager()
        exps = mgr.list_experiments(status=status_filter if status_filter!="全部" else None)
        if not exps:
            empty_state("暂无实验 — 运行 pipeline 后自动记录")
        else:
            rows = []
            for exp in exps:
                fm = exp.get("final_metrics",{}) or {}
                rows.append({
                    "ID": exp.get("exp_id","")[:20],
                    "名称": exp.get("name",""),
                    "状态": exp.get("status",""),
                    "时间": (exp.get("created_at","") or "")[:19],
                    "夏普": f"{fm.get('best_sharpe',0):.3f}" if fm.get('best_sharpe') else "--",
                    "IC": f"{fm.get('IC',0):.4f}" if fm.get('IC') else "--",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(f"共 {len(exps)} 个实验")
    except Exception as e:
        st.info(f"数据库尚未初始化，运行 pipeline 后自动创建。")

with tab2:
    section_header("报告对比")
    result_dir = Path("results")
    if result_dir.exists():
        json_files = sorted(result_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if json_files:
            selected = st.multiselect(
                "选择报告文件",
                [f.name for f in json_files[:10]],
                default=[f.name for f in json_files[:3]] if len(json_files)>=3 else [f.name for f in json_files],
            )
            if selected and len(selected)>=2:
                all_m = {}
                for fn in selected:
                    with open(result_dir/fn,"r") as f:
                        data = json.load(f)
                    raw = data.get("raw_metrics",{})
                    for mode,metrics in raw.items():
                        all_m[f"{fn[:12]}_{mode}"] = metrics

                names = list(all_m.keys())
                sharpe_vals = [all_m[n].get("sharpe_ratio",0) for n in names]
                fig = go.Figure(go.Bar(
                    x=[n[:20] for n in names], y=sharpe_vals,
                    marker_color="#c9a84c",
                    text=[f"{v:.3f}" for v in sharpe_vals], textposition="auto"))
                st.plotly_chart(dark_figure(fig, 420), use_container_width=True)
        else:
            empty_state("暂无报告文件")
    else:
        empty_state("results/ 目录不存在")
