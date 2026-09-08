"""Launch the local React/API application and its separately owned worker.

No installation or frontend build is performed automatically. The old UI is
an explicit optional entry while dynamic migration verification is deferred.
"""

import argparse
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from uuid import uuid4

from stockrl_app.errors import AppError


ROOT = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="StockRL 本地实验台")
    result.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    result.add_argument("--port", type=int, default=8000)
    result.add_argument("--legacy", action="store_true", help="显式启动旧 Streamlit 界面")
    return result


def child_options() -> dict:
    options = {"cwd": str(ROOT), "env": os.environ.copy()}
    if sys.platform == "win32":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    return options


def preflight(host: str, port: int) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("端口必须位于 1～65535。")
    missing = [name for name in ("fastapi", "uvicorn", "pydantic", "filelock", "multipart")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("缺少 Web 依赖，请先使用当前 Python 安装 requirements.txt。")
    if not (ROOT / "api" / "static" / "index.html").is_file():
        raise RuntimeError(
            "尚无 React 静态资源。请在准备运行时先安装前端依赖，"
            "再执行 npm --prefix frontend-react run build。启动器不会自动构建。"
        )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError as exc:
            raise RuntimeError(f"无法使用本地端口 {port}；请关闭占用程序或选择 --port。") from exc


def stop_owned(process: subprocess.Popen, timeout: float = 5) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.legacy:
        if importlib.util.find_spec("streamlit") is None:
            print('旧界面需要可选依赖：python -m pip install ".[legacy]"', file=sys.stderr)
            return 2
        return subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "frontend" / "app.py")],
            **child_options(),
        )
    api_process = worker_process = None
    stop_file = None
    try:
        preflight(args.host, args.port)
        from stockrl_app.settings import AppSettings
        from stockrl_app.storage.database import initialize_database

        settings = AppSettings.from_env(project_root=ROOT)
        initialize_database(settings.database_path)
        control_root = settings.app_dir / "control"
        control_root.mkdir(parents=True, exist_ok=True)
        stop_file = control_root / f"launcher-{uuid4().hex}.stop"
        options = child_options()
        options["env"].update({"STOCKRL_HOST": args.host, "STOCKRL_PORT": str(args.port)})
        worker_process = subprocess.Popen(
            [sys.executable, "-m", "stockrl_app.jobs.worker", "--stop-file", str(stop_file)],
            **options,
        )
        api_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.main:create_app", "--factory",
             "--host", args.host, "--port", str(args.port)],
            **options,
        )
        print(f"实验台：http://{args.host}:{args.port}；关闭浏览器不会停止任务，Ctrl+C 退出服务。")
        warned = False
        while api_process.poll() is None:
            if worker_process.poll() is not None and not warned:
                print("worker 已退出，界面仍可浏览。请查看运行日志并重新启动 worker。", file=sys.stderr)
                warned = True
            time.sleep(.3)
        return int(api_process.returncode or 0)
    except KeyboardInterrupt:
        return 0
    except (AppError, OSError, ValueError, RuntimeError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 2
    finally:
        if stop_file is not None:
            try:
                stop_file.touch(exist_ok=True)
            except OSError as exc:
                print(f"无法发送停止请求：{exc}", file=sys.stderr)
        if api_process is not None:
            try:
                stop_owned(api_process)
            except (OSError, subprocess.TimeoutExpired) as exc:
                print(f"API 未确认退出：{exc}", file=sys.stderr)
        if worker_process is not None:
            try:
                worker_process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                try:
                    stop_owned(worker_process)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    print(f"worker 未确认退出：{exc}", file=sys.stderr)
        if stop_file is not None:
            try:
                if worker_process is None or worker_process.poll() is not None:
                    stop_file.unlink(missing_ok=True)
            except OSError as exc:
                print(f"停止记录保留供排查：{exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
