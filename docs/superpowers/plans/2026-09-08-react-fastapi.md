# React + FastAPI Refactor Implementation Plan

> **For agentic workers:** 实施时使用 `executing-plans` 技能按任务执行并在阶段边界审查。所有步骤用复选框追踪。本计划是用户要求预先准备的文档；不得因读取计划就开始实现、启动协作者或推送远端。

**Goal:** 将完整的本地 RL 实验工作流迁移到 React + FastAPI，支持独立任务生命周期、可追踪产物和历史兼容。

**Architecture:** React 消费版本化 HTTP API，FastAPI 调用应用服务，独立 worker 协调计算子进程。SQLite 保存任务与索引，文件保存不可变数据和实验，既有 stockrl 核心与离线 CLI 保持研究语义。

**Tech Stack:** Python 3.12+、React、TypeScript、Vite、React Router、TanStack Query、Plotly.js、FastAPI、Pydantic、SQLite、filelock、既有 Gymnasium/SB3/PyTorch、pytest、Vitest、Testing Library、Playwright。

**Spec:** [2026-09-08-react-fastapi-design.md](../specs/2026-09-08-react-fastapi-design.md)。执行者必须同时阅读 Spec；API 字段、默认值和状态表以 Spec 为准。

**Status:** Draft；所有实现步骤未开始。基线 `a66ea1bc6f1cf7a8c5512dc0454634b33304a0ab`，文档日期 2026-09-08。

## Global Constraints

- C-01：Python 3.12+；前端 Node.js 22.12+、npm、React、TypeScript、Vite；Windows 为首要验收平台，同时验证 Linux。
- C-02：日线、单资产、只做多、无融资无卖空；RL 直接决定目标仓位，不依赖 StockTrader 信号。
- C-03：观察 t 收盘、下一交易行开盘成交、同一行收盘估值；动作 [-1,1] 映射到目标仓位 [0,1]；费用只计入净值一次。
- C-04：训练、验证、测试的收益区间互不重叠；归一化只拟合训练段；验证选检查点；测试不参与选模或挑选 seed。
- C-05：训练、评估和基准共享 TradingEnv；数据末尾按市值截断；本次不改变算法超参数规则、观察特征、奖励和撮合语义。
- C-06：默认本机单用户、一个训练或重放任务执行；训练不运行在 HTTP 请求、API 进程或 FastAPI BackgroundTasks 中。
- C-07：保留用户行情、历史 CSV、旧 experiments.db 和既有实验；docs1/ 保持忽略；本轮只提交本地文档，不推送或编写项目代码。
- C-08：前端不接触服务器任意路径，不计算另一套绩效；API、CLI 和 worker 使用同一领域规则；合成演示和重放必须显式标注。

本计划依照“先不要编写代码”只给出文件职责、接口契约、实施步骤和可核验场景，不附实现代码或假装已经运行的测试输出。文中拟建文件和命令均供以后实施使用，当前并不存在这些新服务和测试。实施任务产生的提交均先留本地，推送按用户后续指令处理。

## 1. 任务依赖与交付阶段

| 阶段 | 任务 | 可审查交付物 | 阶段门槛 |
|---|---|---|---|
| M1：规则与应用基础 | P01 → P02 → P03 | 领域基线、元数据、数据集、预览和幂等提交服务 | 核心不变，输入和任务记录可验证 |
| M2：执行与产物 | P04 → P05 → P06 | 独立任务、真实训练、原子发布、旧产物与重放 | 取消、中断、重放和完整性通过 |
| M3：接口与完整界面 | P07 → P08 → P09 | 版本化 API、React 全流程和契约生成 | 浏览器恢复及全部现有能力覆盖 |
| M4：切换与交付 | P10 → P11 | 默认新入口、依赖清理、运维说明与验证报告 | 干净安装、故障恢复、回退、最终审查通过 |

P04 依赖 P02 的状态仓储；P05 还依赖 P01/P03。P06 依赖 P05 的产物协议。P07 在 P03～P06 服务完整后接入真实 API。P08/P09 消费 P07 的稳定类型，不能各自发明接口。首次开发默认按表顺序执行；如果以后拆给不同协作者，先合入共享契约再分工。

## 2. 拟定文件组织

| 路径 | 职责 |
|---|---|
| `stockrl/protocol.py` | 从 experiments 提取轻量切分规则，保留原导出 |
| `stockrl/control.py` | 可选取消信号、结构化执行事件和取消异常 |
| `stockrl_app/models.py` | 应用配置、请求、响应记录和状态枚举 |
| `stockrl_app/settings.py` | 固定根目录、本地限制和运行参数 |
| `stockrl_app/storage/database.py`、`migrations.py` | 连接、事务、版本、初始化和一致性备份 |
| `stockrl_app/storage/datasets.py`、`experiments.py`、`jobs.py`、`artifacts.py` | 分职责仓储和条件更新 |
| `stockrl_app/datasets.py`、`experiments.py` | 数据登记、配置预览与提交服务 |
| `stockrl_app/results.py`、`legacy.py` | 报表查询、兼容读取和历史导入 |
| `stockrl_app/artifacts.py` | manifest、白名单路径、指纹和发布恢复 |
| `stockrl_app/jobs/service.py`、`worker.py`、`runner.py`、`recovery.py` | 任务查询/取消服务、协调入口、计算子进程入口、重启核对 |
| `stockrl_app/jobs/adapter.py` | 对接核心同步实验和重放，转换事件 |
| `stockrl_app/maintenance.py` | 显式 schema 升级、备份、历史导入命令 |
| `api/main.py`、`dependencies.py`、`errors.py`、`security.py` | app factory、依赖注入、统一错误、本地访问边界 |
| `api/routes/health.py`、`datasets.py`、`experiments.py`、`jobs.py`、`artifacts.py` | 分资源 HTTP 路由 |
| `frontend-react/src/api/` | 生成类型、薄客户端、请求与错误处理 |
| `frontend-react/src/features/datasets/`、`experiment-form/`、`jobs/`、`results/` | 按工作流组织页面和组件 |
| `frontend-react/src/components/`、`app/` | 共享展示控件、路由和查询配置 |
| `tests/app/`、`tests/api/`、`frontend-react/e2e/` | 服务、契约、进程和浏览器行为验证 |

测试/构建配置随首次需要它的任务加入，不做一个无法独立验收的空脚手架提交。模块内部如进一步拆分，应同步本表和任务文件清单，不把所有应用逻辑堆入 main.py。

## 3. 共享接口契约

以下名称是拟定接口，实施任务必须使用相同名称或先更新本计划及 Spec。类型含义完整定义于 Spec 的字段表和 HTTP 契约，不在前端复制第二份约束。

| 定义任务 | 名称 | 消费者与含义 |
|---|---|---|
| P02 | `AppSettings` | 根目录、时限与容量；API/worker/维护入口共用 |
| P02 | `Dataset`、`ExperimentRecord`、`JobDetail`、`JobEvent`、`ArtifactRecord` | 持久记录的应用表示；job_id 在历史 ExperimentRecord 可为空 |
| P02 | `Page[T]` | items、next_cursor；稳定排序，供列表查询使用 |
| P03 | `ExperimentRequest`、`ExperimentPreview`、`SubmissionResult`、`Capabilities` | 请求、规范化预览、experiment_id/job_id、默认值与限制 |
| P03 | `DatasetService.import_csv/import_demo/import_local/get/preview/list` | 只接受文件流、受控 source ID 或生成参数，返回 Dataset/分页/样例 |
| P03 | `ExperimentService.preview(request)` | 返回 ExperimentPreview，不创建任务 |
| P03 | `ExperimentService.submit(request, idempotency_key)` | 返回 SubmissionResult；数据库原子登记实验/任务/幂等 |
| P04 | `JobRepository.claim_next(owner_token)` | 返回 JobDetail 或空；同时更新 revision 和 running |
| P04 | `JobService.cancel(job_id)` | 返回 JobDetail；幂等条件更新 |
| P04 | `JobRepository.append_event(job_id, owner_token, event)` | 返回 JobEvent；持久 seq、过滤过期 owner |
| P04 | `Worker.run()`、`run_job(job_id, owner_token, control_channel)` | 协调进程与可 spawn 的顶层子进程入口 |
| P05 | `ExperimentAdapter.execute(job, request, control, emit)` | worker 内调用核心；返回 staging 内的待发布产物描述 |
| P05 | `ArtifactStore.validate/publish/reconcile/resolve` | 核对、发布、重启核对与受控路径解析 |
| P06 | `ExperimentDetail`、`SeriesResponse`、`ResultService.detail/series/trades/artifacts` | 读取产物而不加载模型；返回 Spec 定义的详情/曲线/分页/产物对象 |
| P06 | `LegacyImporter.scan()` | 只读扫描固定根目录，返回新增/已存在/损坏/不支持计数 |
| P06 | `ReplayService.submit(source_experiment_id, idempotency_key)` | 新实验/任务与来源关联，返回 SubmissionResult |
| P07 | `create_app(settings)` | FastAPI factory，不启动 worker、不加载模型 |

## P01：锁定研究语义并提取轻量规则

**覆盖：** C-02～05；AC-02、AC-15；ADR-0001/0005。

**文件：** 新建 `stockrl/protocol.py`、`tests/test_protocol.py`、`tests/fixtures/migration/README.md`；按需修改 `stockrl/experiments.py` 和 `tests/test_training.py` 的导入。既有 data/env/features/evaluation 测试保留。

**输入/输出：** 输入基线 `split_intervals` 和现有核心测试；输出可独立导入的相同切分函数，原 `stockrl.experiments.split_intervals` 仍可调用。输出迁移对照说明与被忽略的本地基线实验。

- [ ] 在实施分支记录起始 SHA、用户数据及旧数据库 SHA256，确认未混入其他人的修改；运行基线 `python -m pytest`，记录实际结果。
- [ ] 在改动前用基线代码生成 PPO/SAC 各一份短实验及一个旧 replay，至少一份包含两个 seed；输出放 `outputs/verification/react-fastapi-baseline/`，记录环境、配置与文件指纹，不把模型塞入 Git。
- [ ] 添加轻量导入和边界测试：81 行数据得到 (0,48)、(48,64)、(64,80)，奖励日期无交集；非法比例及过短数据保持原错误；导入 protocol 不加载 Torch/SB3。
- [ ] 运行 `python -m pytest tests/test_protocol.py`，确认新增能力在提取前失败；移动切分规则并保留兼容导出，不改变计算。
- [ ] 运行 `python -m pytest tests/test_protocol.py tests/test_data.py tests/test_environment.py tests/test_evaluation.py tests/test_training.py`，核对资金手算、因果特征与选模测试。
- [ ] 审查差异只涉及规则位置与验证，提交一个本地变更；进入 P02 前关闭任何研究语义差异。

**验收：** 新旧导出一致，基线产物可供后续兼容测试；此任务不宣称任何策略有效。

## P02：应用模型、存储版本和元数据仓储

**覆盖：** FR-03/08/09；AC-03、AC-12、AC-13；ADR-0003。

**文件：** 新建 `stockrl_app/__init__.py`、`models.py`、`settings.py`、`storage/__init__.py`、`storage/database.py`、`storage/migrations.py`、`storage/datasets.py`、`storage/experiments.py`、`storage/jobs.py`、`storage/artifacts.py`、`maintenance.py`、`tests/app/test_storage.py`、`tests/app/test_settings.py`。在 `pyproject.toml` 添加本任务实际需要的 Pydantic/filelock 等应用依赖分组与包发现。

**输入/输出：** 输入 Spec 的实体、状态与根目录定义；输出 AppSettings、记录模型、Page、仓储事务和显式维护入口。新数据库只在明确初始化命令或启动器首次建立空应用目录时创建，未知 schema 不自动升级。

- [ ] 写仓储测试：两连接相同幂等 key 只登记一次；外键无效被拒绝；状态 revision 条件失败不覆盖原值；重新连接后记录存在。
- [ ] 写路径与配置测试：默认新数据库在 outputs/app，根目录 experiments.db 不被打开；根目录实际路径固定，网络共享根目录不作为支持场景。
- [ ] 运行 `python -m pytest tests/app/test_storage.py tests/app/test_settings.py`，先验证缺失行为，再实现 schema、唯一约束、WAL、foreign_keys、5 秒 busy timeout 和短事务。
- [ ] 实现显式版本检查、初始化和一致性备份；测试未知未来 schema 拒绝写入，升级失败事务回滚，WAL 下备份恢复含最近已提交记录。
- [ ] 复跑本任务测试与 `tests/test_protocol.py`；检查包导入不启动数据库迁移或进程。
- [ ] 审查表字段可覆盖 Spec 的实验来源、发布凭据和 owner/revision，提交本地变更。

**验收：** 状态持久化且历史文件无变化；不要求本任务已经能训练。

## P03：数据集、预览和幂等实验提交

**覆盖：** FR-01～03；AC-01～03；ADR-0001/0003/0005。

**文件：** 新建 `stockrl_app/datasets.py`、`stockrl_app/experiments.py`、`tests/app/test_datasets.py`、`tests/app/test_experiment_service.py`；扩充 `models.py` 与相应仓储。

**输入/输出：** 消费 P01 的轻量切分、现有 validate_bars/TradingConfig 和 P02 仓储；产出 DatasetService、ExperimentService、Capabilities 与请求/预览/提交类型。此任务创建 queued，不执行训练。

- [ ] 添加 CSV/演示/白名单本地来源测试，覆盖恶意文件名、符号链接/junction 越界、无效 OHLC、同日重复、空文件、20 MiB 和 100000 行限制；结构无效文件不能产生可训练快照。
- [ ] 添加不可变快照测试：登记后改动原 CSV 不改变数据集；价格口径 unknown 保留；时间过滤只作用于新实验快照，数据集本体不变。
- [ ] 添加默认值与配置测试：PPO/10000/[42]/126/0.6/0.2 及全部 TradingConfig 字段；拒绝 NaN、布尔整数、重复 seed 和未知字段；research purpose 必须有说明。
- [ ] 运行 `python -m pytest tests/app/test_datasets.py tests/app/test_experiment_service.py`，确认缺失行为后实现数据服务、规范化请求/hash、预览和原子提交。
- [ ] 用两个并发连接提交相同 key，相同请求返回同一结果，不同请求冲突；创建 20 个 queued 后新 key 被容量拒绝，旧 key 仍返回原任务。
- [ ] 核对 preview 不写实验/任务，submit 重验数据和限制；复跑上述测试及 P01/P02 覆盖，提交本地变更。

**验收：** 不依赖 FastAPI 就能完成数据准备和可靠排队；API/前端随后直接复用服务。

## P04：worker 生命周期、取消和恢复

**覆盖：** FR-04/05；AC-03、AC-05～07、AC-13；ADR-0002。

**文件：** 新建 `stockrl_app/jobs/__init__.py`、`service.py`、`worker.py`、`runner.py`、`recovery.py`、`tests/app/test_jobs.py`、`tests/app/test_worker_process.py`；扩充 jobs 仓储，在 service.py 定义 JobService。测试使用受控的短计算子进程，先不依赖真实训练。

**输入/输出：** 消费持久队列及记录类型；产出单实例协调进程、spawn 子进程入口、原子领取、取消、持久事件和恢复协议。计算入口接收可替换执行适配器供 P05 接入。

- [ ] 写状态表参数化测试覆盖全部合法转移，终态不可回退；同一任务被两次领取时仅一个成功；过期 owner 的事件和结果被拒绝。
- [ ] 写进程行为测试：启动两个 worker 只有一个持实例锁；计算子进程持执行锁；停止协调进程后新协调进程不在旧执行仍活跃时启动第二任务。
- [ ] 运行 `python -m pytest tests/app/test_jobs.py tests/app/test_worker_process.py`，确认缺失行为后实现协调、控制通道、seq 事件和实际进程退出检查。
- [ ] 验证 queued 取消不启动、running 先 cancelling、正常退出才 cancelled、重复取消幂等；注入成功/取消同时发生，断言仅一个合法终态且事件顺序可解释。
- [ ] 验证 5 秒心跳、30 秒失联提示、15 秒协作取消、5 秒终止确认、12 小时任务时限的逻辑；单元测试用可控时钟，另保留真实短进程退出测试，避免真的等待 12 小时。
- [ ] 验证 worker 崩溃、父通道关闭、锁仍被占用与锁释放四种恢复结果；不得按过期时间重跑或仅凭 PID 杀进程。
- [ ] 在 Windows 及 Linux 跑进程测试，记录并修复平台差异，提交本地变更。

**验收：** 尚无 RL 时也能证明任务管理正确；不能用模拟执行器替代后续真实训练验收。

## P05：真实 RL 适配、进度与可信产物发布

**覆盖：** FR-04～06/08；AC-04～08、AC-12、AC-15；ADR-0002/0003/0005。

**文件：** 新建 `stockrl/control.py`、`stockrl_app/jobs/adapter.py`、`stockrl_app/artifacts.py`、`tests/app/test_training_jobs.py`、`tests/app/test_artifact_publish.py`；按需修改 `stockrl/training.py`、`stockrl/experiments.py`、`stockrl/evaluation.py`，补 `tests/test_training.py`。

**输入/输出：** 消费 P03 请求和 P04 任务控制；产出 ExperimentAdapter、ArtifactStore、结构化核心事件与 manifest v1。原 run_experiment/evaluate_saved_run 的既有调用保持有效，新取消参数必须可选。

- [ ] 先写取消适配测试：训练步、验证/评估循环、seed 边界能响应；取消不触发成功汇总、测试选模或伪造最终模型；不传控制参数时旧行为一致。
- [ ] 写两个 seed 的进度测试，包括 PPO 实际步数超预算、进入验证/评估/发布阶段、进度不倒退，以及事件节流不改变验证频率。
- [ ] 运行 `python -m pytest tests/app/test_training_jobs.py tests/app/test_artifact_publish.py`，确认缺失行为后将核心接入计算子进程；适配器之外的 API/查询模块不导入训练。
- [ ] 用 PPO/SAC 各一个真实短任务验证 `_n_updates > 0`、验证检查点选择、各 seed 文件、四基准和聚合；保留同一模型加载重放的 NAV 对照。
- [ ] 实现 staging、请求快照、manifest/指纹、研究/指标版本、代码/依赖/CPU 元数据、发布意图及短条件事务；严格遵循 Spec 的先持久意图再移动/提交顺序，注入文件缺失、hash 不符、目标已存在、磁盘写失败及“目录移动后提交前崩溃”。
- [ ] 重启核对：合法凭据补记成功，冲突/部分产物隔离，已提交 cancelling 不发布为成功；测试原目录不会被覆盖。
- [ ] 运行本任务测试和既有 data/env/evaluation/training 回归，检查非取消路径的数学语义未变，提交本地变更。

**验收：** 真实实验完整成功才成为可比较结果；取消与中断不会污染成功实验列表。

## P06：结果服务、旧格式导入与重放

**覆盖：** FR-06～08；AC-08～12；ADR-0003/0005。

**文件：** 新建 `stockrl_app/results.py`、`stockrl_app/legacy.py`、`tests/app/test_results.py`、`tests/app/test_legacy_import.py`、`tests/app/test_replay_jobs.py`；扩充 `maintenance.py`、`experiments.py`、jobs adapter 和 models。

**输入/输出：** 消费 P01 基线产物和 P05 ArtifactStore；产出 ResultService、LegacyImporter、ReplayService、SeriesResponse。重放沿用队列，创建独立应用实验 ID 并保存 source_experiment_id。

- [ ] 用 P01 保存的真实旧训练和旧 replay 建兼容测试，并增加残缺、未知 schema、源目录缺失、绝对路径越界样例；导入不写源文件，重复导入不重复。
- [ ] 写结果测试：读取报表不加载 Torch/模型；null 保留；分页稳定；超过 5000 点的图表抽样有明确计数，指标仍来自全量报表，下载不抽样。
- [ ] 运行 `python -m pytest tests/app/test_results.py tests/app/test_legacy_import.py tests/app/test_replay_jobs.py`，确认缺失行为后实现适配、受控下载和数据完整性分类。
- [ ] 实现新 replay 请求、来源验证、新 ID、全新输出和 bars.csv/model.zip/normalizer.json/training.json 源文件副本；指纹/依赖不兼容时拒绝执行，报告允许浏览与允许重放的区别。
- [ ] 在临时目录搬移旧完整实验及新 replay，核对同一模型 NAV 绝对误差不超过 1e-8；旧 replay 缺源时不根据未经验证路径加载模型。
- [ ] 测试最多 4 个完整训练实验的比较条件：数据/测试期/交易配置/计价单位/核心语义版本/指标版本不同会返回差异，不生成合并排名，不按测试 seed 排名。
- [ ] 对比用户文件及旧数据库 SHA256，复跑本任务覆盖后提交本地变更。

**验收：** 新旧研究结果可解释、可浏览、可按条件重放；没有自动迁移旧业务数据库。

## P07：FastAPI 契约与本地访问边界

**覆盖：** FR-01～09；AC-01～06、AC-10、AC-13；ADR-0001/0004。

**文件：** 新建 `api/__init__.py`、`main.py`、`dependencies.py`、`errors.py`、`security.py`、`routes/__init__.py`、`routes/health.py`、`datasets.py`、`experiments.py`、`jobs.py`、`artifacts.py`、`tests/api/test_contract.py`、`test_workflows.py`、`test_security.py`、`contracts/openapi.json`；扩充 pyproject 中 web/test 依赖与包发现。

**输入/输出：** 消费 P03～P06 真实服务；产出 create_app(settings) 和 Spec 中全部 HTTP 路径。Pydantic 模型是 OpenAPI/TypeScript 的结构来源；任何契约变动先同步 Spec。

- [ ] 为全部路由添加请求/响应契约测试，覆盖默认值、HTTP 状态、分页、统一错误与 request_id；确定性的服务测试不启动真实训练。
- [ ] 添加安全边界测试：非法 Host/Origin、未知字段、错误 Content-Type、上传超限、路径/junction 越界、伪造产物 ID、模型上传和错误中的绝对路径泄漏。
- [ ] 运行 `python -m pytest tests/api`，确认失败后实现薄路由；禁止路由内运行模型、读任意路径或复制核心指标计算。
- [ ] 导出 contracts/openapi.json，建立可重复生成命令 `python -m api.export_openapi`（同任务新建 `api/export_openapi.py`），生成结果移除非确定性字段但不隐藏契约变化。
- [ ] 用真实服务提交任务、取消、下载和重放；验证 API 重启后查询/幂等仍有效，API 进程导入不加载 Torch/SB3，热重载不创建 worker。
- [ ] 一个真实短训练运行时，以 2 秒间隔访问状态和 live 各 20 次，记录硬件与响应时间，按 AC-04 的 2 秒目标验收；未达到时先定位阻塞点，不以增加后台线程掩盖训练仍在 API 中。
- [ ] 完成 API 与应用层测试，提交本地变更及生成契约。

**验收：** 无 React 也能完成完整 HTTP 工作流，前端协作者可直接消费固定契约。

## P08：React 数据准备与实验提交

**覆盖：** FR-01～03/09；AC-01～03、AC-13/14；ADR-0001/0004。

**文件：** 新建 `frontend-react/package.json`、`package-lock.json`、`index.html`、`tsconfig.json`、`vite.config.ts`、`vitest.config.ts`、`src/main.tsx`、`src/app/router.tsx`、`src/app/query-client.ts`、`src/api/schema.d.ts`、`src/api/client.ts`、`src/components/` 下实际需要的组件、`src/features/datasets/`、`src/features/experiment-form/`、`src/styles.css` 及对应 `.test.tsx`；新增 `tests/api/test_schema_export.py` 检查契约稳定。

**输入/输出：** 消费 P07 OpenAPI 和 capabilities；产出 `/datasets` 与 `/experiments/new`，成功提交导航到 job ID 对应页面。路由、服务器状态和草稿状态分开。

- [ ] 初始化与本任务页面所需的最小 React/TypeScript/Vite 配置，锁定依赖；定义 npm scripts：`dev`、`build`、`typecheck`、`test`、`api:generate`、`api:check`。
- [ ] 从 OpenAPI 生成 schema.d.ts；薄客户端统一处理错误和请求 ID，组件不得复制一份手写请求类型；接口生成差异应使 api:check 失败。
- [ ] 写行为测试：数据来源切换、无效 CSV、unknown 价格口径、capabilities 默认值、研究说明必填、服务端预览、保留校验失败后的输入。
- [ ] 运行 `npm --prefix frontend-react run test -- --run`，以失败的行为测试驱动页面实现；普通样式不添加镜像实现的测试。
- [ ] 写刷新/重试测试：提交网络结果未知后保留同一 key，再试得到同一 job；服务器缓存刷新不覆盖草稿；显式新建实验才换 key。
- [ ] 运行 `npm --prefix frontend-react run typecheck`、`npm --prefix frontend-react run api:check`、前端测试与 build；浏览器人工检查键盘、表单错误、两种桌面尺寸和窄屏主操作。
- [ ] 提交本地变更；此阶段不把占位任务页当作完整前端验收。

**验收：** 用户可以通过真实 API 登记数据并提交可追踪任务，表单与服务端规范一致。

## P09：React 任务跟踪、结果分析与重放

**覆盖：** FR-04～08；AC-05～11/14；ADR-0002/0004/0005。

**文件：** 新建 `frontend-react/src/features/jobs/`、`src/features/results/` 的页面、查询 hooks 和测试；修改 router；新建 `frontend-react/playwright.config.ts`、`e2e/experiment.spec.ts`、`e2e/recovery.spec.ts`、`e2e/results.spec.ts`；增加 npm `e2e` script。

**输入/输出：** 消费 P07 JobDetail、JobEvent、ExperimentDetail、SeriesResponse 等契约；产出 Spec 规定的队列、任务详情、历史、实验详情和 compare 全部页面。

- [ ] 写组件行为测试覆盖 queued、running、cancelling、各终态、worker 失联、网络错误和恢复等待；点击取消不能提前显示 cancelled。
- [ ] 实现两秒轮询、十秒上限退避、终态停止、seq 去重及刷新恢复；跨 seed 累积显示步进度，评估/保存阶段不显示整体成功。
- [ ] 写结果测试验证 null、全 seed 汇总、四基准语义、演示/replay 标记、抽样提示、原始下载和不同条件比较差异；图表直接展示服务端数据。
- [ ] 完成结果与重放入口；复制配置生成新草稿，原实验不可修改；源不兼容时禁用重放并说明原因。
- [ ] 用真实 API/worker 跑 Playwright：合成数据 → 短 PPO → 刷新/直接深链接 → 成功结果 → 原样重放；另一条覆盖取消、API 重启和数据损坏。仅在这类可控技术验收中使用合成数据。
- [ ] 运行前端 typecheck、api:check、单元测试、build 和 `npm --prefix frontend-react run e2e`；核对 1440×900、1280×720、窄屏和键盘交互。
- [ ] 对照现有 Streamlit 的所有能力与 Spec FR-01～08，记录差异并修复遗漏后提交本地变更。

**验收：** 完整产品工作流可通过新界面运行；任务仍由服务器拥有，刷新不会重跑。

## P10：启动、打包、迁移说明与 Streamlit 退出

**覆盖：** FR-09/10；AC-10/13/14；ADR-0004。

**文件：** 修改 `run.py`、`pyproject.toml`、`requirements.txt`、`.gitignore`、`README.md`、`frontend-react/vite.config.ts`；新建 `requirements.lock`、`requirements-dev.lock`、`docs/operations/local-development.md`、`docs/operations/migration-recovery.md`、`tests/test_launcher.py`、`tests/api/test_static_app.py`。M3 验收后删除活跃 `frontend/app.py`、`.streamlit/config.toml` 和 `tests/test_frontend.py`，移除 Streamlit/Python Plotly 的活跃依赖。

**输入/输出：** 消费已验收 React/API/worker；产出 Python 本地启动器、前端静态交付、CLI 独立安装和可回退的迁移说明。Git 历史保留旧界面。

P10 的入口替换先在实施分支形成候选版本，供 P11 验证；P11 全部门槛通过前不得集成到用户日常 main 或移除其回退能力。

- [ ] 写启动器测试：当前 Python 被使用、端口占用可解释、目录不可写/未知 schema/前端未构建时给出明确错误、模块导入无副作用、热重载不产生第二 worker。
- [ ] 将前端构建输出设为专用生成目录 `api/static/`，从 Git 忽略并纳入 Python 交付包的 package data；构建只清理该已验证生成目录。测试页面深链接可回退到入口，API 404 不返回 HTML。
- [ ] 实现 run.py 启动/关闭 API 与 worker 的进程管理，Windows 辅助进程隐藏窗口；关闭浏览器不停止服务，退出启动器受控停止任务。
- [ ] 分离 core/web/market/test 依赖，生成 Python 锁文件并验证 Windows/Linux 可安装；保留离线 CLI 及可选 Yahoo。源码使用说明给出先构建 React 的步骤，交付包测试确认无需 Node 常驻或 Streamlit。
- [ ] 编写初始化、升级、备份/恢复、历史导入、取消、失联、worker 恢复等待和磁盘错误说明；明确 CLI 直接训练与 Web 队列是不同资源入口。
- [ ] 保存切换代码点与一致性备份，在副本上演练旧入口回退及重新进入新应用；核对旧数据库与用户数据未变，禁止自动降级 schema 或删除产物。
- [ ] 运行 `python -m pytest tests/test_launcher.py tests/test_cli.py tests/api/test_static_app.py` 与打包安装验证，确认 M3 验收成立后移除活跃 Streamlit 文件和依赖，再运行覆盖测试并提交本地变更。

**验收：** 全流程默认通过 React，CLI 仍可离线运行，用户能按文档启动、停止和回退。

## P11：完整验证、实现审查与交付记录

**覆盖：** 全部 AC；ADR-0001～0005。

**文件：** 新建 `docs/verification/2026-09-08-react-fastapi.md`（实施日期变化时使用实际日期命名）、`docs/reviews/react-fastapi-implementation-review.md`、`.github/workflows/ci.yml`；更新本计划勾选、Spec/ADR 状态和 docs 索引。不得把用户尚未确认的 ADR 自动改为 Accepted。

**输入/输出：** 消费 P01～P10 的变更与证据；产出逐条需求对照、真实验证报告、审查发现与关闭记录。历史 verification.md 不改成新结果。

- [ ] 建 CI：Python 核心/应用/API 在 Windows 和 Linux 验证；前端类型、契约、测试和构建；关键 Playwright 流程。将慢的真实训练/故障注入与快速单元测试区分，但二者都是切换必需证据。
- [ ] 运行 `python -m pytest` 与前端 typecheck/api:check/test/build/e2e，记录实际命令、平台、版本、耗时和失败，不复制旧的 63 项通过结论。
- [ ] 在干净临时目录验证源码安装、带静态资源的包安装、独立 CLI 与完整 Web；真实 PPO/SAC 都有梯度更新，同一保存模型重放达到 NAV 容差。
- [ ] 重复执行任务取消/完成竞争、API 重启、worker 崩溃/孤儿锁、发布窗口崩溃、未知 schema、历史 replay 缺来源与无权限目录场景，逐条链接 AC 证据。
- [ ] 进行需求审查、工程审查和研究语义审查；记录具体问题、影响、对应修复与复验。审查者先核对最终差异，不能以计划完成度替代实现检查。
- [ ] 若存在 P1 问题、任何 AC 未验证或用户文件变化，停止默认入口切换并保留可运行旧版本；解决相关问题后只重跑受影响覆盖及必要集成检查。
- [ ] 完成后生成本地交付说明：行为变化、剩余限制、复现命令、回退点、当前分支和提交；只有实际完成的步骤才勾选。推送、合并或继续研究按用户的后续明确指令处理。

**验收：** 证据支持全部软件迁移要求；不把迁移完成写成策略盈利或实盘可用。

## 4. 需求覆盖矩阵

| Spec 需求 | 主要任务 | 验收 |
|---|---|---|
| FR-01 数据准备 | P02、P03、P07、P08 | AC-01、AC-13 |
| FR-02 配置预览 | P01、P03、P07、P08 | AC-02 |
| FR-03 提交幂等 | P02、P03、P04、P07、P08 | AC-03、AC-04 |
| FR-04 生命周期 | P04、P05、P07、P09 | AC-05、AC-06、AC-12 |
| FR-05 进度恢复 | P04、P05、P07、P09 | AC-06、AC-07 |
| FR-06 结果下载 | P05、P06、P07、P09 | AC-08、AC-13 |
| FR-07 研究比较 | P03、P06、P09 | AC-08、AC-09 |
| FR-08 版本兼容 | P02、P05、P06、P07 | AC-10～12 |
| FR-09 启动维护 | P02、P07、P08、P10 | AC-13、AC-14 |
| FR-10 切换清理 | P09、P10、P11 | AC-14、AC-15 |
| C-02～05 研究不变 | P01、P05、P06、P11 | AC-02、AC-04、AC-11、AC-15 |

## 5. 执行前状态

本计划已把用户指定的 React + FastAPI 方向转换为可审阅的工作拆分；尚未执行任何 P01～P11 步骤。下一步是用户审阅文档或指示推送文档分支。是否开始开发、是否调整范围，由后续指令决定。
