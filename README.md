# 股票数据管理系统 + AI 交易助手

一个完整的 AI 股票交易系统，集成数据管理、概率预测和强化学习交易 Agent。

## 📋 功能特性

### 数据管理
- ✅ **增量更新**：智能识别本地数据，只下载新增部分
- ✅ **批量处理**：支持同时处理多个股票代码
- ✅ **技术指标**：自动计算20+种常用技术指标
- ✅ **本地存储**：数据保存在本地，避免重复下载
- ✅ **元数据管理**：记录更新时间、数据范围等信息

### AI 预测
- ✅ **概率预测**：使用 XGBoost 预测未来 1/5/10 天涨跌概率
- ✅ **多时间窗口**：同时分析短期、中期、长期趋势
- ✅ **可视化分析**：交互式图表展示预测结果
- ✅ **综合建议**：基于多个时间窗口给出交易建议

### 强化学习 Agent
- ✅ **自动学习**：通过试错学习最优交易策略
- ✅ **趋势识别**：识别上涨、下跌、震荡行情
- ✅ **交易信号**：自动生成买入/卖出/持有信号
- ✅ **风险控制**：优化收益的同时控制最大回撤
- ✅ **回测分析**：在历史数据上验证策略效果

### Web 界面
- ✅ **零命令行**：完全图形化操作
- ✅ **实时图表**：K线图、技术指标图、资产曲线
- ✅ **多页面管理**：数据管理、AI预测、RL交易分离
- ✅ **响应式设计**：支持桌面和移动端

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Web 应用

```bash
# 方式一：使用启动脚本（推荐）
python run.py

# 方式二：直接启动 Streamlit
streamlit run frontend/app.py
```

浏览器会自动打开 `http://localhost:8501`

### 3. 使用流程

#### 📊 数据管理
1. 进入 **"📊 数据管理"** 页面
2. 输入股票代码（如 AAPL, MSFT, NVDA）
3. 点击下载或选择预设列表批量下载
4. 数据会自动保存到本地

#### 🔮 AI 预测
1. 在主页选择已下载的股票
2. 点击 **"📥 加载数据"**
3. 点击 **"🎯 训练模型"**
4. 查看 **"🔮 AI预测"** 标签页的预测结果

#### 🤖 RL 交易
1. 进入 **"🤖 AI交易"** 页面
2. 选择 **"🎯 训练 Agent"** 标签
3. 配置参数并开始训练（建议 50K 步）
4. 在 **"📊 回测分析"** 查看策略效果
5. 在 **"🎮 实时交易"** 获取交易信号

## 📁 数据存储结构

```
stock_data/
├── AAPL_raw.csv          # 原始价格数据
├── AAPL_processed.csv    # 包含技术指标的数据
├── AAPL_meta.json        # 元数据信息
├── MSFT_raw.csv
├── MSFT_processed.csv
├── MSFT_meta.json
└── ...
```

## 📊 包含的技术指标

### 趋势指标
- SMA (10, 50, 200日简单移动平均)
- EMA (12, 26日指数移动平均)
- 均线偏离率

### 动量指标
- RSI (相对强弱指数)
- MACD (指数平滑异同移动平均线)

### 波动率指标
- 布林带宽度和位置
- ATR (平均真实波幅)

### 成交量指标
- 成交量变化率
- 成交量比率

### 收益率
- 1日、5日、20日收益率
- 趋势强度

## 🔧 高级功能

### 自定义起始日期

```python
engine = DataEngine("AAPL", start_date="2020-01-01")
```

### 强制重新下载

```python
engine.fetch_data(force_update=True)
```

### 自定义数据目录

```python
batch = BatchDataEngine(data_dir="my_custom_folder")
```

## 📝 下一步：使用 RL Agent

RL Agent 训练完成后，你可以：

```python
from rl_agent import RLTradingAgent
from improved_data_engine import DataEngine

# 加载数据
engine = DataEngine("NVDA")
df = engine.load_processed_data()

# 加载训练好的模型
agent = RLTradingAgent(df, model_path="data/models/rl_agent_NVDA_PPO.zip")

# 获取交易信号
signal = agent.get_trading_signals(df.tail(50))
print(f"动作: {signal['action']}")
print(f"理由: {signal['reasoning']}")

# 或在 Web 界面中：
# 🤖 AI交易 → 🎮 实时交易 → 获取信号
```

## 📚 文档

- **[SETUP.md](SETUP.md)** - 详细部署指南
- **[RL_AGENT_GUIDE.md](RL_AGENT_GUIDE.md)** - RL Agent 使用教程

## 🎯 系统架构

```
Frontend (Streamlit)
    ↓
Data Engine (yfinance + pandas-ta)
    ↓
├─ Predictor (XGBoost) → 概率预测
└─ RL Agent (PPO/A2C) → 交易策略
    ↓
Trading Signals → 买/卖/持有
```

## ⚠️ 免责声明

本工具仅供学习和研究使用，不构成任何投资建议。股市有风险，投资需谨慎。

## 📞 问题反馈

如有问题或建议，欢迎反馈！