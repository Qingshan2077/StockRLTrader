"""
v3 · Alpha 模型训练 — 8种模型 + Ensemble
"""
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.preprocessing import StandardScaler

from frontend.v3_utils import (
    init_session, apply_theme, load_v3_config, get_available_stocks,
    load_stock_data, load_features,
    metric_tile, section_header, empty_state, status_badge, dark_figure,
    benchmark_rating, comparison_table,
    PLOTLY_DARK,
)

init_session()
apply_theme()

st.set_page_config(page_title="模型训练", page_icon="▸", layout="wide")
st.markdown('<h1 style="font-family:Noto Serif SC,serif;font-weight:600;color:#e8e4d9;border-bottom:1px solid #252a35;padding-bottom:0.6rem"><span style="color:#c9a84c">▸</span> Alpha 模型训练</h1>', unsafe_allow_html=True)

# ==================== 侧边栏 ====================
cfg = load_v3_config()
stocks = get_available_stocks()
if "v3_ticker" not in st.session_state or st.session_state.v3_ticker is None:
    st.session_state.v3_ticker = stocks[0] if stocks else "AAPL"

with st.sidebar:
    st.markdown('<div style="font-size:0.7rem;color:#7a7570;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:0.5rem">· 数据选择</div>', unsafe_allow_html=True)
    ticker = st.selectbox("股票代码", stocks if stocks else ["AAPL"],
                           index=stocks.index(st.session_state.v3_ticker)
                           if st.session_state.v3_ticker in stocks else 0)
    if ticker != st.session_state.v3_ticker:
        st.session_state.v3_ticker = ticker
        st.session_state.v3_models = {}
        st.session_state.v3_signal_scores = None

    st.markdown("<hr style='border-color:#252a35;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;color:#7a7570;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:0.5rem">· 模型选择</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.8rem;color:#8a8580;margin-bottom:0.5rem">推荐: LightGBM + Ridge（日频数据不需要复杂模型）</div>', unsafe_allow_html=True)
    model_options = {
        "lightgbm":  "LightGBM ★ 推荐",
        "ridge":     "Ridge ★ 推荐",
        "xgboost":   "XGBoost（可选，与LightGBM功能重叠）",
        "mlp":       "MLP（需torch，可选）",
        "lstm":      "LSTM（需torch，日频易过拟合）",
        "gru":       "GRU（需torch，日频易过拟合）",
        "tcn":       "TCN（需torch，日频易过拟合）",
        "transformer":"Transformer（需torch，日频易过拟合）",
    }
    selected = st.multiselect(
        "选择模型", list(model_options.keys()),
        default=["lightgbm", "ridge"],
        format_func=lambda x: model_options[x],
    )

    st.markdown("<hr style='border-color:#252a35;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;color:#7a7570;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:0.5rem">· 参数</div>', unsafe_allow_html=True)
    label_type = st.selectbox("标签类型", ["regression", "classification"], index=0)
    horizon = st.slider("预测天数", 1, 20, 5)

    st.markdown("<hr style='border-color:#252a35;margin:1rem 0'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        train_btn = st.button("▸ 训练选中", use_container_width=True)
    with c2:
        train_all_btn = st.button("▸ 训练全部", use_container_width=True)
    if st.button("× 清除模型"):
        st.session_state.v3_models = {}
        st.rerun()

# ==================== 主区域 ====================
tab1, tab2, tab3 = st.tabs(["训练进度", "模型对比", "Ensemble"])

# --- Tab 1 ---
with tab1:
    if train_btn or train_all_btn:
        models_to_train = list(model_options.keys()) if train_all_btn else selected
        if len(models_to_train) > 3 and not train_all_btn:
            st.warning("建议使用「训练全部」按钮批量对比")

        st.info(f"开始训练 {len(models_to_train)} 个模型: {', '.join(models_to_train)}")

        with st.spinner("加载数据 + 特征工程..."):
            df = load_stock_data(ticker)
            if df is None or df.empty:
                st.error(f"无法加载 {ticker} 数据，请先在「数据管理」页面下载。")
                st.stop()
            result = load_features(ticker, df)
            if result is None: st.stop()
            features, labels, registry = result
            st.session_state.v3_features = features
            st.session_state.v3_labels = labels

        feat_cols = [c for c in features.columns
                      if c not in ('Open','High','Low','Close','Volume','Adj Close')]
        X_all = features[feat_cols].values.astype(np.float32)
        y_all = labels.values.astype(np.float32)
        n = len(X_all)
        tr_end, vl_end = int(n*0.6), int(n*0.8)

        scaler = StandardScaler().fit(X_all[:tr_end])
        X_train_s = scaler.transform(X_all[:tr_end])
        X_val_s = scaler.transform(X_all[tr_end:vl_end])
        X_test_s = scaler.transform(X_all[vl_end:])

        st.session_state.v3_test_features = features.iloc[vl_end:]
        st.session_state.v3_test_labels = y_all[vl_end:]

        progress = st.progress(0)
        status_text = st.empty()

        for i, model_name in enumerate(models_to_train):
            status_text.text(f"训练 {model_name}... ({i+1}/{len(models_to_train)})")
            progress.progress((i+1)/len(models_to_train))
            t0 = time.time()

            try:
                if model_name == "lightgbm":
                    from layers.signals.models.lightgbm_model import LightGBMModel
                    m = LightGBMModel(name="lightgbm", config=cfg.get("lightgbm",{}))
                elif model_name == "xgboost":
                    from layers.signals.models.xgboost_model import XGBoostModel
                    m = XGBoostModel(name="xgboost", config=cfg.get("xgboost",{}))
                elif model_name == "ridge":
                    from layers.signals.models.linear_model import LinearModel
                    m = LinearModel(name="ridge", config=cfg.get("ridge",{}))
                elif model_name == "mlp":
                    from layers.signals.models.mlp_model import MLPSignalModel
                    m = MLPSignalModel(name="mlp", config=cfg.get("mlp", {}))
                elif model_name == "lstm":
                    from layers.signals.models.lstm_model import LSTMSignalModel
                    m = LSTMSignalModel(name="lstm", config={
                        "hidden_size": cfg.get("lstm",{}).get("hidden_size",64),
                        "num_layers": cfg.get("lstm",{}).get("num_layers",2),
                        "dropout": cfg.get("lstm",{}).get("dropout",0.3),
                        "sequence_length": cfg.get("lstm",{}).get("sequence_length",20),
                        "learning_rate": cfg.get("lstm",{}).get("learning_rate",0.001),
                        "batch_size": cfg.get("lstm",{}).get("batch_size",64),
                        "epochs": cfg.get("lstm",{}).get("epochs",100),
                        "patience": cfg.get("lstm",{}).get("patience",15),
                    })
                elif model_name == "gru":
                    from layers.signals.models.gru_model import GRUSignalModel
                    m = GRUSignalModel(name="gru", config={
                        "hidden_size": cfg.get("gru",{}).get("hidden_size",64),
                        "num_layers": cfg.get("gru",{}).get("num_layers",2),
                        "dropout": cfg.get("gru",{}).get("dropout",0.3),
                        "sequence_length": cfg.get("gru",{}).get("sequence_length",20),
                        "learning_rate": cfg.get("gru",{}).get("learning_rate",0.001),
                        "batch_size": cfg.get("gru",{}).get("batch_size",64),
                        "epochs": cfg.get("gru",{}).get("epochs",100),
                        "patience": cfg.get("gru",{}).get("patience",15),
                    })
                elif model_name == "tcn":
                    from layers.signals.models.tcn_model import TCNSignalModel
                    m = TCNSignalModel(name="tcn", config={
                        "num_channels": cfg.get("tcn",{}).get("num_channels",[64,64,64,64]),
                        "kernel_size": cfg.get("tcn",{}).get("kernel_size",3),
                        "dropout": cfg.get("tcn",{}).get("dropout",0.3),
                        "sequence_length": cfg.get("tcn",{}).get("sequence_length",20),
                        "learning_rate": cfg.get("tcn",{}).get("learning_rate",0.001),
                        "batch_size": cfg.get("tcn",{}).get("batch_size",64),
                        "epochs": cfg.get("tcn",{}).get("epochs",100),
                        "patience": cfg.get("tcn",{}).get("patience",15),
                    })
                elif model_name == "transformer":
                    from layers.signals.models.transformer_model import TransformerSignalModel
                    m = TransformerSignalModel(name="transformer", config={
                        "d_model": cfg.get("transformer",{}).get("d_model",64),
                        "nhead": cfg.get("transformer",{}).get("nhead",4),
                        "num_layers": cfg.get("transformer",{}).get("num_layers",2),
                        "dim_feedforward": cfg.get("transformer",{}).get("dim_feedforward",128),
                        "dropout": cfg.get("transformer",{}).get("dropout",0.3),
                        "sequence_length": cfg.get("transformer",{}).get("sequence_length",20),
                        "learning_rate": cfg.get("transformer",{}).get("learning_rate",0.001),
                        "batch_size": cfg.get("transformer",{}).get("batch_size",64),
                        "epochs": cfg.get("transformer",{}).get("epochs",100),
                        "patience": cfg.get("transformer",{}).get("patience",15),
                    })
                else:
                    status_text.text(f"未知模型: {model_name}")
                    continue

                metrics = m.fit(X_train_s, y_all[:tr_end], X_val_s, y_all[tr_end:vl_end])
                preds = m.predict(X_test_s)
                st.session_state.v3_models[model_name] = {
                    "model": m, "metrics": metrics, "test_preds": preds,
                    "train_time": time.time()-t0, "scaler": scaler,
                }
            except Exception as e:
                st.error(f"{model_name} 失败: {e}")

        progress.empty(); status_text.empty()
        st.success(f"完成 — 已训练 {len(st.session_state.v3_models)} 个模型")
        st.rerun()

    if st.session_state.v3_models:
        section_header("已训练模型")
        cols = st.columns(min(len(st.session_state.v3_models), 4))
        for i, (name, data) in enumerate(st.session_state.v3_models.items()):
            r2 = data["metrics"].get("val_r2", 0)
            rmse = data["metrics"].get("val_rmse", 0)
            r2_label, r2_color = benchmark_rating("R2", r2)
            with cols[i%4]:
                metric_tile(name.upper(), f"{r2:.4f}",
                            f"{r2_label} · RMSE {rmse:.4f} · {data.get('train_time',0):.1f}s",
                            accent="gold" if r2 > 0.01 else ("steel" if r2 > 0 else "bearish"))
    else:
        empty_state("尚未训练模型", "—")

# --- Tab 2 ---
with tab2:
    if len(st.session_state.v3_models) < 2:
        st.info("需要至少 2 个模型对比，请使用「训练全部」按钮。")
    else:
        section_header("模型指标对比")
        md = st.session_state.v3_models
        names, r2v, times = list(md.keys()), [], []
        for n in names:
            r2v.append(md[n]["metrics"].get("val_r2",0))
            times.append(md[n].get("train_time",0))

        fig = make_subplots(rows=1, cols=2, subplot_titles=("验证集 R²", "训练时间 (s)"))
        fig.add_trace(go.Bar(x=names, y=r2v, marker_color="#c9a84c", text=[f"{v:.4f}" for v in r2v], textposition="auto"), row=1, col=1)
        fig.add_trace(go.Bar(x=names, y=times, marker_color="#6b8ba4", text=[f"{v:.1f}s" for v in times], textposition="auto"), row=1, col=2)
        st.plotly_chart(dark_figure(fig, 380), use_container_width=True)

        rows = [{"模型": n, "R²": f"{md[n]['metrics'].get('val_r2',0):.4f}",
                 "RMSE": f"{md[n]['metrics'].get('val_rmse',0):.4f}",
                 "耗时": f"{md[n].get('train_time',0):.1f}s"} for n in names]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- Tab 3 ---
with tab3:
    if len(st.session_state.v3_models) < 2:
        st.info("需要至少 2 个模型构建 Ensemble。")
    else:
        section_header("模型融合 Ensemble")
        ens_method = st.selectbox("融合方法", ["weighted","stacking","regime"], index=0,
                                  format_func=lambda x: {"weighted":"加权平均","stacking":"Stacking","regime":"Regime切换"}[x])
        if st.button("▸ 构建 Ensemble", use_container_width=True):
            with st.spinner("构建中..."):
                from layers.ensemble import EnsembleManager
                em = EnsembleManager(method=ens_method)
                em.add_models([d["model"] for d in st.session_state.v3_models.values()])
                ft = st.session_state.v3_features
                if ft is not None:
                    fc = [c for c in ft.columns if c not in ('Open','High','Low','Close','Volume','Adj Close')]
                    Xa = ft[fc].values.astype(np.float32)
                    ya = st.session_state.v3_labels.values.astype(np.float32)
                    n2 = len(Xa); te2 = int(n2*0.6); ve2 = int(n2*0.8)
                    sc = StandardScaler().fit(Xa[:te2])
                    mets = em.fit(sc.transform(Xa[:te2]), ya[:te2], sc.transform(Xa[te2:ve2]), ya[te2:ve2])
                st.session_state.v3_ensemble = em
                st.success("构建完成")
                if ens_method == "weighted" and "weights" in mets:
                    w = mets["weights"]
                    fig = go.Figure(go.Pie(labels=list(w.keys()), values=list(w.values()),
                                           hole=0.4, marker_colors=["#c9a84c","#6b8ba4","#2ecc71","#e74c3c","#f39c12"]))
                    st.plotly_chart(dark_figure(fig, 350), use_container_width=True)
