# 本地开发与入口

项目使用 React + FastAPI。研究 v2 的数据、schema 2 升级、滚动评估和显式续跑说明见 [研究 v2](research-v2.md)。本次研究功能使用隔离数据进行了测试和短训练，没有生成生产前端构建；以下构建命令供用户后续自行执行。

## 安装与启动

建议 Python 3.12+、Node.js 22.12+。源码环境：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
npm --prefix frontend-react install --ignore-scripts
npm --prefix frontend-react run build
.venv\Scripts\python.exe run.py
```

Vite 输出到 `api/static/`。构建产物不提交 Git；打包配置会在实际构建后收集这些文件。当前没有生成静态交付包，缺少 `api/static/index.html` 时启动器会直接说明原因，不自动安装或构建。仅部署本机 loopback，不适用于公网或多用户共享。

分进程开发方式：

```powershell
python -m stockrl_app.maintenance init
python -m uvicorn api.main:create_app --factory --host 127.0.0.1 --port 8000
python -m stockrl_app.jobs.worker
npm --prefix frontend-react run dev
```

上述三个常驻服务分别运行；Vite 将 `/api` 代理到本地 8000 端口。API 不隐式初始化数据库、启动 worker 或载入 Torch/SB3。独立 worker 才负责计算。通过 `run.py` 启动时，启动器显式初始化新应用存储，并管理它自己启动的子进程。

如果后端使用其他端口（例如旧工程占用 8000，新工程使用 8081），在当前工作树的 `frontend-react/.env` 中设置：

```dotenv
STOCKRL_API_TARGET=http://127.0.0.1:8081
```

该文件已被 Git 忽略；未设置时仍使用 8000。修改后重启 Vite 开发服务。代理地址必须指向当前工作树启动的本机 HTTP 后端，否则新页面可能连接到缺少研究接口的旧服务并显示“资源不存在”。可访问 `http://127.0.0.1:5173/api/v1/researches` 检查代理是否返回研究列表。

`STOCKRL_APP_DIR` 默认 `outputs/app/`，`STOCKRL_OUTPUT_DIR` 默认 `outputs/experiments/`。这两个目录必须位于本机磁盘。`STOCKRL_HOST`、`STOCKRL_PORT` 可供分进程方式设置本地监听配置；直接使用 uvicorn 时绑定地址/端口仍需与配置一致。配置只在启动时读取，HTTP 请求不能修改存储目录。

## 检查与契约

允许独立进行的静态检查：

```powershell
python scripts/static_check.py
python scripts/generate_contracts.py --check
python -m ruff check stockrl_app api scripts run.py tests/app tests/test_launcher.py
npm --prefix frontend-react run typecheck
```

`generate_contracts.py` 仅通过 AST 读取 Python 模型注解，生成 TypeScript 传输类型，不导入应用。默认参数由 `/capabilities` 提供。它能检查源码类型同步，但不能证明 FastAPI 实际生成的 OpenAPI 或请求校验行为正确。

未来动态验收时，运行 `python scripts/export_openapi.py frontend-react/openapi.json`，再使用前端的 OpenAPI 类型生成命令核对契约；这个导出会导入应用模型，本轮未执行。API 错误体统一包含 `error` 与 `request_id`，浏览器显示中文错误和可定位编号。

## 旧入口

`python -m stockrl demo/train/evaluate` 与 `run_pipeline.py` 保持离线核心入口。行情网络下载需要可选 `market` 依赖。CLI 直接执行，不受 Web 队列锁管理，同机使用时自行避免同时启动多个重任务。

旧 Streamlit 暂留于 `frontend/`，安装 `.[legacy]` 后可通过 `python run.py --legacy` 使用。因本轮动态验收被明确禁止，暂不删除旧入口；这不是默认新前端的依赖。只有端到端与迁移验收得到实际证据后，才决定最终清理。

## 当前验证边界

本次不运行 pytest、Vitest、Playwright、Vite 构建、API/worker、训练或模型重放。测试文件是待执行的回归用例，不代表通过。尤其 Windows 进程退出、取消竞态、发布恢复、浏览器刷新、曲线展示、PPO/SAC 真实训练及前端交付包仍需后续动态验收。
