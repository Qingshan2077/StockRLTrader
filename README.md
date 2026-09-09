# StockRLTrader

独立的强化学习交易研究项目：智能体观察历史行情和账户状态，自主决定目标仓位。当前开发分支将 Web 工作流迁移为 **React + TypeScript + FastAPI + 独立 worker**，保留 Python 研究核心和离线 CLI。

> 此次迁移按用户要求仅编写代码和进行静态检查。没有运行构建、测试、训练或服务；运行兼容性与策略有效性均未验证。下面的命令供后续获得运行授权或用户自行使用，启动器不会自动构建。

## 代码结构

| 目录 | 职责 |
|---|---|
| `frontend-react/` | 中文实验界面、数据准备、任务状态、曲线和比较 |
| `api/` | `/api/v1` HTTP 契约、错误与本地访问限制 |
| `stockrl_app/` | 不可变数据集、SQLite 元数据、幂等请求、任务与产物管理 |
| `stockrl_app/jobs/` | 独立协调进程、一个计算子进程、取消和恢复 |
| `stockrl/` | 数据、因果特征、TradingEnv、PPO/SAC、评估与重放 |
| `frontend/` | 可选的旧 Streamlit 入口，等待动态迁移验收后退出 |

设计、架构决定和任务计划见 [协作文档](docs/README.md)，运行方式见 [本地开发](docs/operations/local-development.md)，数据迁移与故障处理见 [维护说明](docs/operations/migration-recovery.md)。

## 后续安装与启动

需要 Python 3.12+。构建前端需要 Node.js 22.12+；构建后的界面由 FastAPI 提供，无需常驻 Node 服务。

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
npm --prefix frontend-react install --ignore-scripts
npm --prefix frontend-react run build
.venv\Scripts\python.exe run.py
或者：
python -m uvicorn api.main:create_app --factory --host 127.0.0.1 --port 8081
npm --prefix frontend-react run dev
```

默认地址为 `http://127.0.0.1:8000`。`run.py` 检查前端构建并启动 API 和独立 worker；缺少构建会给出说明，绝不自动安装或构建。关闭浏览器不会取消任务；退出启动器会请求停止正在执行的任务。默认单机单用户，一个训练/重放任务执行，其余排队，不连接券商。

可分别启动 API、worker 和 Vite 进行开发，具体步骤见维护文档。Web 依赖位于 `.[web]`；Yahoo 下载位于 `.[market]`；旧界面位于 `.[legacy]`。仅使用核心 CLI 时可安装 `.`，不需要 Streamlit、API 或 worker。

## 离线 CLI

```powershell
python -m stockrl demo --timesteps 10000 --seeds 42 43 44 --output outputs/experiments
python -m stockrl train --csv stock_data/AAPL_raw.csv --algorithm PPO --timesteps 10000
python -m stockrl train --ticker AAPL --start 2018-01-01 --end 2025-01-01
python -m stockrl evaluate outputs/experiments/<run-id>
python run_pipeline.py --help
```

CLI 保持直接执行，不经过 Web 队列。不要把同时启动多个 CLI 重任务理解为队列能够控制的并发。Yahoo 下载为可选网络入口，CSV 与合成演示可离线使用。

CSV 必须包含 `Date,Open,High,Low,Close,Volume`，日期唯一且递增、每个日历日最多一行，价格有限正数且 OHLC 合法，成交量非负。程序不猜测复权、分红、拆股或计价单位，也不静默修复行情。

## 研究规则

- 日线、单资产、只做多、不融资、不卖空。观察 t 收盘，动作在下一交易行开盘执行，并按该行收盘估值。
- 动作 `[-1,1]` 映射到目标仓位 `[0,1]`。现金、整手、滑点、佣金、卖出税和历史成交量参与率由同一个 TradingEnv 执行；费用只计入净值一次。
- 基础奖励为扣费后的对数净值变化；保留已有可选回撤/换手惩罚，不新增奖励策略。数据尾部按市值截断，不虚构清仓。
- 默认训练/验证/测试比例为 60%/20%/20%，收益区间互不重叠；归一化只拟合训练段，验证选择检查点，测试不挑选模型或 seed。
- PPO 与 SAC 使用相同环境。多个 seed 都要报告；不同随机种子不能代替不同市场时期的检验。

## 结果与基准

保存模型、归一化、数据指纹、切分、版本、验证记录、逐日净值和成交。新应用增加不可变请求、manifest 和持久任务状态，核心目录继续兼容既有 CLI 格式。重放使用保存的模型与数据，生成独立结果，不是新的样本外证据。

四个基准为现金、持续满仓请求、固定半仓和均线择时。`buy_hold` 实际含义是持续请求 100% 仓位，受现金/参与率约束时后续继续买入，不主动卖出。所有基准与 RL 使用同一评估区间、费用和交易限制。

报表包括净收益、年化收益、年化波动、Sharpe、最大回撤、累计换手、费用、平均仓位和交易笔数。年化使用 252 个交易日、Sharpe 使用零无风险利率；未定义指标保存为 null。图表抽样不改变全量指标，CSV 下载保留原始行。

## 数据与版本管理

应用状态默认保存于 `outputs/app/state.sqlite3`，数据快照位于 `outputs/app/datasets/`，实验文件位于 `outputs/experiments/`。可通过 `STOCKRL_APP_DIR` 和 `STOCKRL_OUTPUT_DIR` 配置启动时根目录。只支持本机磁盘，不使用网络共享文件夹存 SQLite 或锁。

`stock_data/`、历史导出 CSV、根目录旧 `experiments.db` 和原实验均保留；旧数据库不会作为新应用数据库打开。历史 RL 目录通过显式维护命令只读导入，不自动将旧业务记录转为研究证据。

正式协作文档在 `docs/`；不希望上传的笔记在 `docs1/`，由 Git 忽略。此次代码保存在独立工作树和本地分支，未推送远端。

## 后续验证

```powershell
python -m pip install -r requirements-dev.txt
python -m ruff check stockrl stockrl_app api run.py tests
npm --prefix frontend-react run typecheck
python -m pytest
npm --prefix frontend-react test
```

上述动态验证命令本轮没有运行。浏览器端到端与故障注入场景仍需后续实施验收；当前没有自动化 e2e 启动脚本。静态检查不能验证训练、进程故障恢复、浏览器交互或打包后的行为；具体执行记录以新的验证报告为准，不能引用历史的 63 项测试结论覆盖本次重构。

## 研究限制

这是 RL 研究和软件验证工具。短训练与漂亮的收益曲线不证明策略有效；单资产、长仓收益可能来自市场上涨，较低回撤可能只是较低持仓。当前仍缺少完整市场机制、风险匹配基准、滚动样本外研究和跨资产泛化验证。日线 T+1/整手规则不是完整交易所撮合，Yahoo/CSV 的复权和数据质量也会影响结果。框架迁移不构成投资建议或实盘能力。
