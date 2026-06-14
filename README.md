# AI 量化交易系统 v3

基于三层量化架构的 AI 股票交易系统，集成数据管理、Alpha 信号挖掘、风险控制和强化学习执行优化。

## 架构概览

```
┌──────────────────────────────────────────────┐
│          Frontend (React) + Backend (FastAPI)  │
├──────────────────────────────────────────────┤
│  层1: 信号层 (Supervised Learning)             │
│  LightGBM / XGBoost / MLP / LSTM / GRU / TCN │
│  → signal_score (预测未来收益率)                │
├──────────────────────────────────────────────┤
│  层2: 风险与约束层                              │
│  仓位限制 / 止损 / 波动率降仓 / 回撤控制         │
│  → target_position_after_risk                 │
├──────────────────────────────────────────────┤
│  层3: RL 执行优化层 (PPO / SAC)                 │
│  优化调仓节奏，不预测涨跌                        │
│  → execution_ratio ∈ [0,1]                    │
├──────────────────────────────────────────────┤
│  Data Layer: yfinance + pandas-ta             │
│  Features: 50+ 技术因子 (严格无未来函数)         │
│  Eval: IC / RankIC / 分位数分析 / 归因          │
│  Backtest: Bar-based + Event-driven + Walk-Forward │
└──────────────────────────────────────────────┘
```

## 功能特性

### 数据管理
- 增量更新：智能识别本地数据，只下载新增部分
- 批量处理：支持同时处理多个股票代码
- 技术指标：自动计算 50+ 种常用技术指标
- 本地存储：数据保存在本地，避免重复下载
- 元数据管理：记录更新时间、数据范围等信息
- 特征缓存：基于 MD5 的特征缓存，加速迭代

### Alpha 信号模型（8 种）
- **树模型**: LightGBM、XGBoost
- **线性模型**: Ridge、Lasso
- **深度学习**: MLP (BatchNorm+Dropout)、LSTM、GRU、TCN、Transformer
- **集成学习**: 加权平均、Stacking、Voting、市场状态自适应

### 信号评估
- IC (Pearson) / Rank IC (Spearman) 分析
- 滚动 IC 序列与 IC_IR
- IC Decay 跨周期衰减
- 分位数分析与 Top-Bottom Spread
- 因子相关性矩阵
- 信号换手率监控

### 风险管理
- 仓位约束：最大总/净敞口、单一资产上限
- 止损控制：动态止损与重新入场机制
- 波动率缩放：目标波动率 + 高波动降仓
- 回撤控制：熔断机制 + 动态降仓
- 流动性过滤：ADV 限制、最小成交量
- 换手率限制与交易成本估算

### 回测系统
- 三种模式对比：纯信号 / 信号+风控 / 信号+风控+RL
- Walk-Forward 回测（扩展窗口 / 滚动窗口）
- 事件驱动回测基础设施
- 综合指标：年化收益、Sharpe、Sortino、Calmar、胜率、换手率
- 绩效归因：Alpha / 成本 / 换手 / 风险分解

### 实验管理
- SQLite 持久化实验记录
- 超参数网格/随机搜索
- 实验对比与最优选择

### Web 界面（Trading Terminal Noir 主题）
- 11 个功能页面，完整量化工作流
- ECharts / lightweight-charts 交互图表
- 专业暗色交易终端风格 UI

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动系统

```bash
# 方式一：启动 FastAPI 后端
python run_api.py

# 方式二：快速启动菜单
python scripts/quick_start.py

# 方式三：启动 React 前端（另开终端）
cd frontend-react
npm run dev
```

前端默认地址为 `http://127.0.0.1:5173`，后端默认地址为 `http://127.0.0.1:8000`。

### 3. 运行全流程管线

```bash
# 默认 AAPL + LightGBM + PPO
python run_pipeline.py

# 指定股票
python run_pipeline.py --ticker NVDA

# 跳过 RL 训练
python run_pipeline.py --skip-rl

# 启用 Walk-Forward 回测
python run_pipeline.py --walk-forward

# 多资产模式
python run_pipeline.py --stage multi_asset
```

### 4. 使用流程

#### 数据管理
1. 进入 **"数据管理"** 页面
2. 输入股票代码（如 AAPL, MSFT, NVDA）
3. 点击下载或选择预设列表批量下载
4. 数据会自动保存到本地

#### 模型训练
1. 进入 **"模型训练"** 页面
2. 选择 Alpha 模型类型（LightGBM、XGBoost、MLP 等）
3. 配置超参数并开始训练
4. 对比各模型的 R² / RMSE / IC

#### 信号评估
1. 进入 **"信号评估"** 页面
2. 查看 IC 序列和衰减曲线
3. 分位数分析验证信号单调性

#### 回测验证
1. 进入 **"风控系统"** 页面
2. 选择回测模式（signal_only / signal_risk / full）
3. 对比三种模式的表现差异

#### 全流程管线
1. 进入 **"全流程管线"** 页面
2. 一键运行：数据 → 特征 → 信号 → 回测 → 实验记录

## 数据存储结构

```
stock_data/
├── AAPL_raw.csv             # 原始 OHLCV 数据
├── AAPL_processed.csv       # 包含技术指标的数据
├── AAPL_meta.json           # 元数据信息
└── ...
```

## 项目结构

```
StockTrader/
├── frontend-react/           # React + Vite 前端
├── api/                      # FastAPI 后端
├── layers/                   # v3 分层量化系统
│   ├── data/                 # 数据加载 (yfinance + 缓存)
│   ├── features/             # 特征工程 (6组 50+ 因子)
│   ├── signals/              # Alpha 模型 (8种 + 集成)
│   ├── ensemble/             # 模型集成 (加权/Stacking/Voting)
│   ├── portfolio/            # 组合构建与优化
│   ├── risk/                 # 风控约束
│   ├── rl/                   # RL 训练与奖励
│   ├── env/                  # Gymnasium 交易环境
│   ├── backtest/             # 回测引擎 + Walk-Forward
│   ├── evaluation/           # Alpha评估 + 绩效归因
│   ├── experiments/          # 实验管理 (SQLite)
│   └── inference/            # 在线推理 + 信号监控
├── config/                   # YAML 配置文件
├── scripts/                  # 工具脚本
├── stock_data/               # 股票数据存储
├── models/                   # 训练好的模型
├── improved_data_engine.py   # v1 数据引擎 (兼容层)
├── predictor.py              # XGBoost 概率预测 (快速分析)
├── advanced_predictor.py     # 增强预测器 (价格+趋势+持久化)
├── macd_strategy.py          # MACD 短线策略
├── run_api.py                # FastAPI 后端启动脚本
└── run_pipeline.py           # v3 全流程管线
```

## 配置系统

系统使用 YAML 分层配置，支持命令行覆盖和环境变量：

```
config/
├── default.yaml    # 系统默认参数
├── data.yaml       # 数据源配置
├── model.yaml      # 模型超参数
├── risk.yaml       # 风控参数
├── rl.yaml         # RL 训练参数
└── backtest.yaml   # 回测参数
```

## 免责声明

本工具仅供学习和研究使用，不构成任何投资建议。股市有风险，投资需谨慎。
# React + FastAPI Migration

Developer marker: `zlh`.

The project now includes a front/back separated workbench:

- Backend: `api/main.py`
- Backend launcher: `python run_api.py`
- Frontend: `frontend-react`
- Migration status: `MIGRATION_STATUS.md`
- Final notes: `MIGRATION_FINAL_NOTES.md`

The legacy frontend has been removed. Use `python run_api.py` for the backend and `npm run dev` under `frontend-react` for the frontend.
