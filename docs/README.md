# 项目协作文档

研究 v2 已增加数据核算、滚动评估和基础诊断。当前操作见 [研究使用说明](operations/research-v2.md)，验证与限制见 [2026-09-09 交付验证](reviews/2026-09-09-research-v2-verification.md)。下文保留此前 React + FastAPI 重构的协作记录。

## 当前实施：React + FastAPI 整体重构

日期：2026-09-08。设计基线：`main` 的 `a66ea1bc6f1cf7a8c5512dc0454634b33304a0ab`。文档分支：`docs/react-fastapi-refactor`。

用户后续已明确授权开发，要求所有代码在一个新工作树、一个新分支下完成；开发过程不运行构建，只编写代码并静态检查，不得推送远端。实施分支是 `refactor/react-fastapi`。先前文档分支的推送授权不适用于本次代码。

执行调整见 [ADR-0006](adr/0006-static-only-implementation.md)。[本地开发](operations/local-development.md)和[维护恢复](operations/migration-recovery.md)记录后续使用步骤；这些动态命令本轮未执行。原设计的动态验收条件继续保留，不能以静态检查代替。

建议按以下顺序阅读：

| 文档 | 负责回答的问题 | 当前状态 |
|---|---|---|
| [Review：现状与重构边界](reviews/2026-09-08-react-fastapi-review.md) | 为什么改、哪些应保留、什么风险会阻碍迁移？ | 已完成文档自审，待用户审阅 |
| [Spec：目标行为与验收](superpowers/specs/2026-09-08-react-fastapi-design.md) | 系统应做到什么，接口和失败行为是什么？ | 已按后续指令实施，动态验收待执行 |
| [ADR-0001：技术栈与系统边界](adr/0001-react-fastapi-boundaries.md) | 为什么选择这个组合，哪些模块负责什么？ | Accepted for implementation |
| [ADR-0002：独立训练任务](adr/0002-local-job-lifecycle.md) | 长任务如何执行、取消和处理中断？ | Accepted for implementation |
| [ADR-0003：元数据与实验产物](adr/0003-experiment-storage.md) | 什么需要入库，什么需要不可变保存？ | Accepted for implementation |
| [ADR-0004：接口、部署与迁移](adr/0004-contract-deployment-migration.md) | 如何协作开发、上线切换与回退？ | Accepted；本轮执行调整见 ADR-0006 |
| [ADR-0005：RL 研究语义](adr/0005-preserve-research-semantics.md) | 如何避免把框架重构变成未经验证的策略改动？ | Accepted for implementation |
| [Plan：分阶段实施计划](superpowers/plans/2026-09-08-react-fastapi.md) | 按什么顺序实施，逐项怎样验证？ | 代码实施；动态验收待授权执行 |

## 文档职责与变更规则

- Review 记录基线事实、问题影响、优先级和待验证假设，不把计划中的能力描述为已有功能。
- Spec 是目标行为和验收条件的权威定义。需求编号使用 `FR-*`、约束编号使用 `C-*`、验收编号使用 `AC-*`，变更时保留追踪关系。
- ADR 记录一项重要决定的背景、备选方案、取舍与重新评估条件。`Proposed` 不等于已经获得实施许可；用户确认后记录确认日期再改为 `Accepted`。已经实施的重大决定发生改变时，新增 ADR 并标记替代关系，不抹掉历史。
- Plan 引用 Spec 和 ADR，记录任务依赖、拟改文件、接口、验证方法与验收条件。不以勾选计划代替运行证据。
- 本提案以 Spec 为行为准绳，ADR 解释选择，Plan 负责落实；发现矛盾先同步文档，再继续实现。旧 RL 核心文档仍是现有版本的历史记录，不直接改写为新方案。

## 后续协作交接

后续实施建议按“契约与数据 → 任务执行 → HTTP 接口 → React → 兼容与切换”拆分变更。每个变更附关联需求、涉及 ADR、验证结果与未解决问题。共享接口先形成可审查的契约，再由前后端分别消费；任务状态机和产物协议设单一负责人，避免多处自行定义。

本次遵循用户的一个工作树、一个分支要求；协作者按互不重叠的文件职责工作，主代理集成并核对 API 类型、状态枚举和存储版本。后续是否采用独立分支协作由下一次任务决定。

每阶段审查分为需求一致性、工程正确性和研究语义三部分；关键问题关闭后才能切换默认入口。实现阶段的新证据写入新的验证报告，不能引用旧的“63 项测试通过”作为新系统验收结论。

## 本地文档和版本管理

`docs/` 保存需要协作、审查和未来随代码版本管理的正式文档。`docs1/` 保存用户不希望上传的本地笔记，继续由根目录 `/docs1/` 忽略规则排除；正式文档不得依赖 `docs1/` 才能理解。训练产物仍留在被忽略的输出目录。

已有 [RL 核心设计](superpowers/specs/2026-09-08-rl-core-design.md)、[RL 核心计划](superpowers/plans/2026-09-08-rl-core.md) 和 [历史验证记录](verification.md) 保留不变。历史验证记录中的分支、推送状态只描述当时的事件，不代表此后的仓库状态。
