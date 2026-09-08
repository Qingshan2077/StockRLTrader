# ADR-0002：独立进程执行与持久任务生命周期

- 日期：2026-09-08；状态：Accepted for implementation，依据后续开发指令；动态验收未执行，见 [ADR-0006](0006-static-only-implementation.md)。
- 关联：[Spec FR-03～05](../superpowers/specs/2026-09-08-react-fastapi-design.md)、[ADR-0003](0003-experiment-storage.md)。

## 背景

RL 训练是长时间 CPU 计算，浏览器会话和 HTTP 请求均不适合作为其生命周期。接口快速返回任务 ID，只解决等待响应的问题，尚未解决任务是否入队、是否重复执行、如何取消以及进程退出后怎样解释状态。

FastAPI 的 BackgroundTasks 适合响应之后的进程内工作；官方对重计算给出独立任务工具的建议。无论采用什么框架，进程内回调都不能代替这里要求的持久状态协议。[FastAPI 后台任务说明](https://fastapi.tiangolo.com/tutorial/background-tasks/)

## 提议决定

独立 worker 协调进程消费 SQLite 中的队列，一次管理一个通过 `multiprocessing` 显式 `spawn` 创建的计算子进程。API 只提交/查询/取消，不启动训练，不在启动回调中隐式创建 worker。种子在一个任务中保持原有顺序执行。

显式选择 spawn，避免依赖操作系统或 Python 版本的默认启动方式。只传可序列化输入，所有进程入口必须可安全导入，不继承连接、DataFrame 或模型对象。Python 官方记录了 spawn 的导入与可序列化约束。[Python multiprocessing](https://docs.python.org/3/library/multiprocessing.html)

持久状态严格采用 `queued/running/cancelling/succeeded/failed/cancelled/interrupted`。领取、取消、完成使用数据库条件更新和 revision；幂等 key 与规范化请求哈希防止用户重复提交。新执行采用新 job ID，不把失败任务悄悄重置为 queued。

协调进程持实例锁，计算子进程持独立执行锁；拟使用 filelock 的系统文件锁，拒绝降级为只靠文件存在的软锁。该库提供跨平台进程锁，具体 Windows 和 Linux 行为必须纳入故障测试。[filelock 官方文档](https://py-filelock.readthedocs.io/en/latest/)

API 重启不影响 worker。worker 重启若发现旧子进程仍持锁，就显示恢复等待，暂不执行新任务；确认退出后核对完成凭据，不能确认成功的任务标为 interrupted。父通信通道断开时子进程在安全点退出；系统不声称可无条件回收所有阻塞的孤儿计算。

取消先记录 cancelling，协作停止优先，超时后仅终止协调进程确实拥有的子进程；确认退出才释放名额。完整的时限、竞争处理和重启规则由 Spec 定义。训练回调和评估循环提供可选取消检查，原 Python API 默认保持同步行为。

## 备选方案

| 方案 | 好处 | 代价与判断 |
|---|---|---|
| HTTP 同步训练 | 最少组件 | 长连接、资源占用和生命周期耦合，不满足要求 |
| API 线程池/BackgroundTasks | 容易接入 | API 重启丢执行上下文，CPU 与请求共享进程，仍需自行补任务协议 |
| 单机独立 worker + SQLite | 本地依赖少，可明确控制生命周期 | 要自己实现状态、取消、锁、恢复和发布测试；本提案采用 |
| Celery/RQ 等队列与外部 broker | 成熟的队列能力与分布式扩展 | 多一个常驻基础设施，仍需业务幂等、产物一致性和平台兼容验证；本次暂不引入 |

## 影响与代价

浏览器和 API 不再决定训练是否继续，失败具有可解释终态。代价是新增进程协调和故障测试，不能把它当成几个接口的附属小功能。进度先采用两秒轮询，省去推送连接恢复协议，但保留持久 seq 事件以便未来扩展。

首版没有自动续训、自动重试或多个训练并发。强制取消留下的部分文件仅供诊断，不能进入完整实验统计。用户直接运行 CLI 的任务独立于队列，必须在维护文档中说明资源关系。

## 验证与重审条件

AC-03～07、AC-12～13 验证并发提交、领取、取消竞态、进程中断和发布。需要多机、多 GPU、资源配额或并发任务时重新评估成熟队列与调度器；替换 worker 实现时保留 API 状态与幂等语义。
