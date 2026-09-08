# StockRLTrader

StockRLTrader 是一个独立的强化学习交易研究项目。智能体直接根据截至当日收盘的行情和账户状态决定目标仓位，不需要监督学习预测值。第一版只处理日线、单资产、只做多、不融资的本地实验，不连接券商，也不执行实盘交易。

## 安装

需要 Python 3.12 或更高版本。建议使用虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

后续 Windows 示例中的 `python` 可以直接替换为 `.venv\Scripts\python.exe`，无需激活脚本。Linux 或 macOS 请使用 `.venv/bin/python`。

## 三种命令行工作流

下面的演示会生成明确标注的合成行情并运行一个 PPO 实验。短训练只用于确认软件能够运行，不能证明策略有效：

```bash
python -m stockrl demo --timesteps 10000 --seeds 42 43 44 --output outputs/experiments
```

使用本地 CSV 训练：

```bash
python -m stockrl train --csv stock_data/AAPL_raw.csv --algorithm PPO --timesteps 10000 --seeds 42 43 44
```

CSV 必须包含 `Date,Open,High,Low,Close,Volume`。日期应唯一且递增；日线输入的同一个日历日只能出现一行，即使时间戳不同也会拒绝。价格必须为有限正数，成交量不能为负。程序不会猜测复权方式；请确保开高低收使用一致的复权、分红和拆股口径。可用 `--start` 和 `--end` 截取日期。

Yahoo 下载是一个可选便利入口，需要网络连接，不是训练本身的依赖：

```bash
python -m stockrl train --ticker AAPL --start 2018-01-01 --end 2025-01-01 --timesteps 10000
```

对一个已完成实验进行原样重放，不再训练、归一化或选模：

```bash
python -m stockrl evaluate outputs/experiments/<run-id>
```

查看全部参数：

```bash
python -m stockrl --help
python -m stockrl demo --help
python -m stockrl train --help
python -m stockrl evaluate --help
```

旧的启动文件仍可作为薄入口使用，所有实验逻辑都由同一个 API 执行：

```bash
python run_pipeline.py demo --timesteps 10000
python run.py
```

`python run.py` 会用当前 Python 启动 Streamlit。界面包含“数据与环境”“训练实验”“结果对比”三个区域，可载入合成演示数据、上传 CSV 或读取 `stock_data` 中的本地原始行情。实验默认保存到 `outputs/experiments`；可在启动前设置 `STOCKRL_OUTPUT_DIR` 改变目录。

## 时间顺序与账户假设

时刻 t 的观察只包含截至 t 收盘的信息，动作在 t+1 开盘成交，并按 t+1 收盘估值。动作 `[-1, 1]` 线性映射为目标仓位 `[0, 1]`。交易环境统一记录现金、持股、滑点、手续费、卖出税和净值，费用只扣一次；现金不能透支，也不允许卖空。

数据按日期分为训练、验证和测试收益区间，默认比例为 60% / 20% / 20%。归一化参数只拟合训练区间，验证区间只选择检查点，测试区间只在选择完成后评估一次。三个区间可以共享边界观察行，但不共享任何成交或收益区间。数据尾部按市值截断，不虚构清仓交易。

流动性上限使用决策时已知的历史成交量。`lot_size`、最低佣金和次日可卖约束是简化的日线规则，不代表完整交易所撮合、盘口排队或市场冲击模型。

## 基准、指标与产物

每个随机种子的测试区间都会使用同一个交易环境运行四个基准：

- 现金：目标仓位始终为 0%。
- 买入持有：目标仓位持续为 100%；受参与率或现金限制时会在之后的交易日继续买入，不会主动卖出。
- 固定半仓：目标仓位持续为 50%。
- 均线择时：收盘价高于最近 20 个可用交易日均线时目标为 100%，否则为 0%。

结果报告净收益、年化收益、年化波动、Sharpe、最大回撤、累计换手、交易费用、平均仓位和成交笔数。Sharpe 使用 252 个交易日和 0 无风险利率；无波动或数据不足的指标保存为 `null`，不会伪装成 0。

每次实验都会保存：

```text
outputs/experiments/<run-id>/
├── summary.json
├── bars.csv
└── seed_<seed>/
    ├── model.zip
    ├── normalizer.json
    ├── history.csv
    ├── trades.csv
    └── baselines/
```

`summary.json` 还记录数据 SHA256、配置、日期切分、依赖版本、随机种子和验证检查点信息。多个种子只报告测试指标的均值和总体标准差，不根据测试成绩挑选“最佳种子”。

## Python API

```python
from stockrl.data import load_csv
from stockrl.experiments import run_experiment, evaluate_saved_run

bars = load_csv("stock_data/AAPL_raw.csv")
summary = run_experiment(
    bars,
    "outputs/experiments",
    algorithm="PPO",
    timesteps=10_000,
    seeds=(42, 43, 44),
    data_label="csv:AAPL_raw.csv",
)
replay = evaluate_saved_run(summary["output_dir"])
```

核心接口还包括 `TradingEnv`、`TradingConfig`、`ObservationNormalizer`、`evaluate_policy`、`baseline_policy` 和 `load_bundle`。PPO 是默认算法，SAC 用于对照；两者共用完全相同的数据、账户与评估路径。

## 验证

```bash
python -m pytest
```

测试覆盖资金守恒、下一开盘成交、隔夜收益归属、费用、卖出回款、整手和流动性限制、因果特征、环境截断、PPO/SAC 真实训练、保存加载一致性、CLI 及 Streamlit 启动。

## 研究限制

本项目用于研究和软件验证，不构成投资建议。回测收益不能代表未来表现。日线撮合忽略盘口、排队、精细市场冲击和许多交易所规则；Yahoo 或用户 CSV 的数据质量、复权方式和幸存者偏差会直接影响结果。训练步数很短的烟雾实验仅说明代码可执行。

当前版本用独立的 `stockrl/` 研究栈替代了旧的监督学习预测、分层信号、旧风控执行和多页面入口。仓库保留 `stock_data/` 中的市场文件、根目录历史导出 CSV、`experiments.db`、LICENSE 和 Git 历史；这些旧产物不会自动转化为新 RL 实验的证据。当前实现仅支持逐只股票的日线实验，不声称支持多资产组合或盘中交易。
