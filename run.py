"""Launch the local Streamlit experiment dashboard with this Python runtime."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
APP = ROOT / "frontend" / "app.py"


if __name__ == "__main__":
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(APP), *sys.argv[1:]],
        cwd=ROOT,
    ))
