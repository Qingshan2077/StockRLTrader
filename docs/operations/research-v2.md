# 数据核算、滚动研究与行为诊断

研究入口是 `/researches`，市场数据登记入口是 `/market-datasets`。原有实验与重放入口继续使用原来的 v1 核算和产物；研究结果不能混入旧实验排名。

本次交付用于检验 RL 是否带来额外价值，没有改变两层 32 单元网络、增加自动调参或承诺盈利。PPO 默认、SAC 可选。单个模型只负责一只股票的多头仓位，动作范围为 0–100%。多标的研究分别训练，不是组合资产配置。

## 启动和迁移

在新工作树中使用 Python 3.12+ 安装 `requirements-dev.txt`，并在 `frontend-react` 下安装锁定依赖。开发入口沿用 FastAPI、独立 worker 和 Vite；本次没有生成生产前端构建，可以先使用开发服务器。

```powershell
python -m stockrl_app.maintenance init
python -m uvicorn api.main:create_app --factory --host 127.0.0.1 --port 8000
# 另开终端
python -m stockrl_app.jobs.worker
# 再开终端，浏览 Vite 打印的地址
npm --prefix frontend-react run dev
```

`init` 只创建新库或接受 schema 2。若指向已有 schema 1 库，先停止 API、worker 及其计算子进程，等待最后一次 worker 心跳超过 30 秒，再显式迁移：

```powershell
python -m stockrl_app.maintenance migrate outputs/backups/before-research-v2.sqlite3
python -m stockrl_app.maintenance check
```

迁移使用包含 WAL 的一致性备份、迁移锁和写事务；失败会回滚并保留备份。不要覆盖现有备份。回退必须停止所有写入进程，恢复整份备份及其对应旧代码；同时保留数据与产物目录。开发验证仅使用临时库，没有迁移用户现有数据库。

`STOCKRL_APP_DIR`、`STOCKRL_OUTPUT_DIR` 沿用原配置；`STOCKRL_RESEARCH_DIR` 默认是 `outputs/researches`。工作树没有共享 `.venv`、`outputs` 或忽略文档的机制，需要明确选择解释器和数据路径。

## 数据规则

上传六个命名文件，合计不超过 100 MiB，单文件不超过 64 MiB；不接受 ZIP。CSV 上限 100000 行，公司行动上限 10000 行。原文件按字节冻结并记录 SHA-256。

| 文件 | 关键字段 |
| --- | --- |
| metadata.json | dataset_schema_version=2、instrument_id、symbol、exchange、security_type、currency、timezone、source、source_version、retrieved_at、coverage_start/end、adjustment=raw、corporate_actions_complete、completeness_source、price_basis_source、volume_unit=shares、action_share_basis=old_shares |
| bars.csv | session, Open, High, Low, Close, Volume；原始价格、原始股数单位 |
| sessions.csv | session, open_at, close_at, is_open；覆盖自然日并声明开闭市，时间带时区 |
| actions.csv | action_id, kind, effective_session, available_at, pay_session, cash_per_old_share, split_ratio；不适用字段留空 |
| tradability.csv | session, available_at, can_buy_open, can_sell_open, limit_up, limit_down, reference_close, reason |
| market-profile.json | profile_id/version、exchange/security_type/currency、effective_start/end、buy_lot、sell_odd_lot、t_plus_one、commission_rate/min、sell_tax_rate、other_fee_rate、dividend_tax_rate、slippage、participation_cap、rule_sources、fee_model=proportional_minimum |

支持 `SSE` / `SZSE`、`common_stock`、六位股票代码。研究身份由交易所和代码产生，例如 `SZSE:000568`，不能通过另起数据集名称绕过暴露记录。港美股和未明确支持的证券类别拒绝套用大陆规则。

规则可以用 `fee_intervals` 记录日期不同的费用，区间必须完整连续，每段明确列出费用、税率、滑点、容量比例和来源。手续费、最低手续费及滑点在压力测试中乘以 2 或 3；税费和其他费用不变。无效压力参数在排队前阻断。

证券规则并非恒定：例如上交所 2026 年规则自 7 月 6 日生效，不能直接用于全部历史区间；2023 年 8 月 28 日证券交易印花税减半。规则的来源应指向对应历史公告，而佣金、分红税率和滑点必须注明实验假设。[上交所规则公告](https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/c_20260424_10816492.shtml)、[税务总局公告](https://shanghai.chinatax.gov.cn/gate/big5/shanghai.chinatax.gov.cn/tax/zcfw/zcfgk/yhs/202308/t468451.html)。仓库没有把未经核验的真实证券规则包作为默认配置；测试包明确为合成数据。

数据资格为 ready、incomplete 或 unsupported。缺资料的包可以登记供检查，格式或价格值损坏的包不能登记。只有 ready 能进入研究。来源声明的一致性可以检查，供应商是否如实提供完整公司行动仍需人工核验。前复权/后复权 CSV 不能改一个标签就成为原始交易数据。

## 账户与特征

观察发生在 t 收盘，成交发生在下一开市会话开盘。原始价格用于成交和账户，连续特征价格通过当时已知的分红和拆股因果构造。每个 fold 的 normalizer 只拟合训练数据，观察包含 8 个市场特征和现金、仓位、应收占比、回撤共 12 维。旧模型的 11 维观察不能作为新模型输入。

分红按前收盘持股先形成税后应收，再处理拆股；到账日前应收参与 NAV，但不能用于买入。只支持正整数倍拆股和现金分红。非整数送转、反向拆股、配股、合并和退市等事件保持阻断。分红税使用显式实验税率，不模拟每个人持有期限对应的递延补税。

成交受整手、T+1、方向可交易状态、涨跌停和过去 20 会话成交量约束；不使用成交日全日成交量决定开盘容量。停牌只能使用可信的历史估值；公司行动期间还需调整后的参考价及估值来源。最后一天不强制清仓。

## 预注册研究

默认训练 60 个月、验证 12 个月、测试 6 个月，每 6 个月滚动一次；不足完整尾窗会说明省略原因。测试会话互不重叠，各 fold 独立训练和现金起步，不拼接为一条连续实盘账户净值。

当前每个账户初始现金固定为 10000；高价股票或小目标仓位可能受整手限制而无法成交。请检查实际仓位与未成交原因，不要将资金不足解释为模型择时。

默认 seeds 为 42–46，每单元 100000 步、126 会话训练 episode。检查点最多 20 个，必须发生在参数更新后，包含最后一个。只按完整验证段累计对数净值选模，同分取较早检查点；测试结果不选择 seed 或模型。最多 250 单元、2500 万请求训练步；预览展示请求步数、rollout 上界、评估次数和时间上限。

全部比较对象共用账户内核：现金、买入持有、每日 25%/50%/75%、每周 50%、20 日趋势、10% 目标波动率，以及验证集校准的固定仓位。主要比较对象固定为每日半仓和风险匹配固定仓位；匹配权重只在验证集 0–100%（5% 步长）选择，同分取较低仓位。

## 如何读结果

技术状态、数据资格和经济结论分别显示。`candidate_edge` 表示值得继续验证的优势，不表示统计显著或可以实盘。

先要求所有计划单元完整、指纹与规则合格、每标的至少四个完整不重叠窗口、固定五个 seed、测试风险匹配率至少 80%、声明保留样本。再按 seed 中位数、fold 中位数逐层比较：相对两个主要参照的 CAGR 差都大于零，至少 60% 窗口同时胜出，回撤差不超过 2 个百分点，双倍执行费用下 CAGR 差非负。三倍费用只报告。多标的结论至少需要五个预先选定标的，每个合格，至少 60% 标的通过且等权标的中位差为正。

没有达到证据条件就是“证据不足”；合格但没有通过经济门槛就是“未发现额外价值”。探索性数据的门槛结果仅作为 provisional_outcome；父研究和单标的都不能显示已合格优势。

诊断包含原始与裁剪动作、仓位边界/近半仓比例、目标与实际仓位差、未成交原因、优化日志、验证及测试特征漂移、四种历史行情分组。`validation/` 前缀属于验证特征，其余特征行属于测试。缺失字段显示未记录，不假设为零；诊断不自动调参或解释因果。

## 暴露记录与续跑

预览会检查既有测试暴露记录。已经查看的测试区间不能通过复制 CSV、换 seed 或重新登记标的身份恢复为保留样本。进入执行单元后保守登记其计划区间；该研究已锁定的资格不被自身登记追溯改变，但后续研究会受影响。

旧实验没有可靠标的身份时应人工建立暴露记录，再显式导入；记录包含 instrument_id、start_session、end_session、scope、reason、recorded_at、protocol_id（可空）、source。`scope=exposed_test` 会阻止重叠区间获得保留样本资格。

```powershell
python -m stockrl_app.maintenance import-exposures docs1/research/exposure-records.json
```

用户已查看的京东和泸州老窖记录保存在当前工作树的忽略目录，未写入用户生产库。该本地文件不会随 Git 自动分发；迁移时需保留并导入。若没有完整暴露登记，应选择探索性研究，不能把“未查到记录”理解为绝对未见数据。

研究内部按锁定顺序串行执行标的/fold/seed。每单元 12 小时，父任务上限按单元数累加；取消与心跳沿用原 worker。成功单元封存后只读，失败、取消或中断必须显式续跑，续跑创建新 attempt/job，保存旧记录；只重做未完整发布的单元，不恢复半个优化器。续跑验证协议、数据、代码内容、提交和 Python/依赖版本；有变化就创建新研究。

产物位于 `outputs/researches/<research_id>/`，标的和 fold 目录使用稳定安全散列，具体身份查看 protocol.json 和单元 manifest。每个单元包含模型、normalizer、检查点选择、三个费用场景及诊断。父汇总只有完整封存和登记后才算发布成功。损坏文件不能靠手改数据库状态恢复。
