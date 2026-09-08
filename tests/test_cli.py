"""Product entry points exercise the real experiment pipeline."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_module_help_is_lightweight_and_lists_the_three_workflows():
    result = subprocess.run(
        [sys.executable, "-m", "stockrl", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0
    assert all(command in result.stdout for command in ("demo", "train", "evaluate"))
    assert "predictor" not in (result.stdout + result.stderr).lower()


def test_train_requires_one_data_source(capsys):
    from stockrl.cli import main

    exit_code = main(["train", "--timesteps", "4"])

    assert exit_code == 2
    assert "--csv" in capsys.readouterr().err


def test_demo_and_saved_evaluation_run_real_ppo(tmp_path, capsys):
    from stockrl.cli import main

    exit_code = main(
        [
            "demo",
            "--rows",
            "81",
            "--timesteps",
            "4",
            "--seeds",
            "7",
            "--episode-length",
            "8",
            "--output",
            str(tmp_path / "experiments"),
        ]
    )
    assert exit_code == 0
    summaries = list((tmp_path / "experiments").glob("*/summary.json"))
    assert len(summaries) == 1
    trained = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert trained["data_label"] == "synthetic_demo"
    assert Path(trained["runs"][0]["model_path"]).is_file()

    exit_code = main(
        ["evaluate", str(summaries[0].parent), "--output", str(tmp_path / "replays")]
    )
    assert exit_code == 0
    assert (tmp_path / "replays" / "summary.json").is_file()
    assert "仅用于研究" in capsys.readouterr().out
