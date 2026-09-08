"""Streamlit smoke tests use its real application runner and real PPO training."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


APP = Path(__file__).parents[1] / "frontend" / "app.py"


def _csv_bytes(rows=81):
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    close = pd.Series(range(100, 100 + rows), dtype=float)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": close + 0.1,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 100_000,
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


def test_dashboard_renders_research_scope_and_three_work_areas():
    app = AppTest.from_file(str(APP)).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "StockRL 实验台"
    assert [tab.label for tab in app.tabs] == ["数据与环境", "训练实验", "结果对比"]
    copy = " ".join(item.value for item in app.markdown)
    assert "单资产" in copy
    assert "不构成投资建议" in copy


def test_uploaded_csv_ignores_unsafe_filename_and_loads_in_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKRL_OUTPUT_DIR", str(tmp_path / "runs"))
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app.selectbox(key="data_source").set_value("上传 CSV").run(timeout=30)
    app.file_uploader(key="csv_upload").set_value(
        ("../../escaped.csv", _csv_bytes(), "text/csv")
    ).run(timeout=30)
    app.button(key="load_upload").click().run(timeout=30)

    assert not app.exception
    assert app.session_state["data_label"] == "uploaded:escaped.csv"
    assert len(app.session_state["bars"]) == 81
    assert not (tmp_path / "escaped.csv").exists()


def test_demo_can_train_real_ppo_and_show_saved_result(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKRL_OUTPUT_DIR", str(tmp_path / "runs"))
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app.button(key="load_demo").click().run(timeout=30)
    app.number_input(key="timesteps").set_value(4).run(timeout=30)
    app.number_input(key="episode_length").set_value(8).run(timeout=30)
    app.button(key="run_experiment").click().run(timeout=120)

    assert not app.exception
    summary = app.session_state["summary"]
    assert summary["data_label"] == "synthetic_demo"
    assert Path(summary["runs"][0]["model_path"]).is_file()
    assert len(app.get("plotly_chart")) >= 1
    assert len(app.dataframe) >= 1
