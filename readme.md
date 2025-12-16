# 股票数据管理系统

一个强大的股票数据采集、存储和管理工具，为后续的机器学习和强化学习模型提供数据支持。

## 📋 功能特性

- ✅ **增量更新**：智能识别本地数据，只下载新增部分
- ✅ **批量处理**：支持同时处理多个股票代码
- ✅ **技术指标**：自动计算20+种常用技术指标
- ✅ **本地存储**：数据保存在本地，避免重复下载
- ✅ **元数据管理**：记录更新时间、数据范围等信息
- ✅ **命令行工具**：提供简洁的CLI和交互式界面

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install yfinance pandas pandas-ta
```

### 2. 配置代理（可选）

如果需要代理访问，请编辑 `improved_data_engine.py` 开头的代理设置：

```python
proxy = 'http://127.0.0.1:7897'  # 修改为你的代理地址
```

如果不需要代理，将这两行注释掉：
```python
# os.environ['HTTP_PROXY'] = proxy
# os.environ['HTTPS_PROXY'] = proxy
```

### 3. 使用方法

#### 方式一：命令行模式

```bash
# 添加股票
python stock_data_manager.py --add AAPL MSFT NVDA TSLA

# 更新所有股票
python stock_data_manager.py --update

# 更新指定股票
python stock_data_manager.py --update AAPL NVDA

# 查看所有本地股票
python stock_data_manager.py --list

# 查看股票详情
python stock_data_manager.py --info AAPL
```

#### 方式二：交互式模式（推荐新手）

```bash
python stock_data_manager.py --interactive
```

然后按照菜单提示操作即可。

#### 方式三：在代码中使用

```python
from improved_data_engine import DataEngine, BatchDataEngine

# 单个股票
engine = DataEngine("AAPL")
engine.fetch_data()  # 下载或更新数据
df = engine.add_technical_indicators()  # 计算技术指标

# 获取最新数据
latest = engine.get_latest_data()
print(latest[['Close', 'RSI', 'MACD_12_26_9']])

# 批量处理
batch = BatchDataEngine()
tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
batch.process_batch(tickers)
```

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

## 📝 下一步：集成预测模型

数据准备好后，可以使用 `predictor.py` 进行预测：

```python
from improved_data_engine import DataEngine
from predictor import ProbabilityPredictor

# 加载数据
engine = DataEngine("NVDA")
df = engine.load_processed_data()

# 训练预测模型
predictor = ProbabilityPredictor(df)
predictor.create_targets()
predictor.train()

# 获取预测
latest = engine.get_latest_data()
probs = predictor.predict_future(latest)
print(probs)  # {1: 0.65, 5: 0.72, 10: 0.58}
```

## 🎯 后续开发计划

- [ ] 强化学习Agent模块
- [ ] 实时交易信号生成
- [ ] 回测系统
- [ ] Web可视化界面
- [ ] 风险管理模块

## ⚠️ 免责声明

本工具仅供学习和研究使用，不构成任何投资建议。股市有风险，投资需谨慎。

## 📞 问题反馈

如有问题或建议，欢迎反馈！