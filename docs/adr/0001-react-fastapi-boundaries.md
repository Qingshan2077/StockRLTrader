# ADR-0001：React + FastAPI 与研究核心的边界

- 日期：2026-09-08。
- 状态：Proposed。用户已确定 React + FastAPI 方向；本文的组件划分与配套选型待审阅。
- 关联：[Spec §3](../superpowers/specs/2026-09-08-react-fastapi-design.md)、[Review R-04/R-06/R-09](../reviews/2026-09-08-react-fastapi-review.md)。

## 背景

现有 Streamlit 页面承担交互、部分数据装载、实验调用和文件读取。`stockrl/` 已独立提供研究功能；重写这些计算不能直接改善前端协作。下一步需要可独立演进的页面、接口和任务执行边界，同时保持一个容易本地运行的项目。

## 提议决定

采用单仓库分层应用：`frontend-react/` 负责 React SPA；`api/` 负责 FastAPI HTTP；`stockrl_app/` 负责应用模型、服务、仓储和任务；`stockrl/` 保留领域计算。API 不包含训练算法和交易规则，核心不导入 Web 或数据库模块。

React 配合 TypeScript、Vite、React Router、TanStack Query，服务器状态与表单草稿分开。图表优先沿用 Plotly 的表达方式，前端不重算指标。初期不用 Next.js/SSR：当前工具没有搜索引擎收录、服务端页面渲染或内容站需求。React 官方也明确，从构建工具起步需自己安排路由和数据获取，本决定承担这些责任。[React 官方说明](https://react.dev/learn/build-a-react-app-from-scratch)

FastAPI 使用 Pydantic 定义输入输出及生成 OpenAPI，训练执行另见 ADR-0002。自动验证和文档适合训练配置等结构化接口，但不会自动实现任务持久化、用户系统或业务权限。[FastAPI 官方功能](https://fastapi.tiangolo.com/features/)

Python 保持 3.12+，前端选择 Node.js 22.12+；实际安装需检查所选版本并形成锁文件。Vite 当前文档列出的 Node 支持下限包含 22.12，本项目选择这一分支作为最低要求。[Vite 环境要求](https://vite.dev/guide/)

## 备选方案

| 方案 | 好处 | 未采用原因 |
|---|---|---|
| Streamlit 增加 React 局部组件 | 改造较小，保留 Python 交互便利 | 不满足整个 Web 应用转为 React + FastAPI 的目标 |
| React + Flask | 完全可行，核心轻便 | 当前没有需要延续的 Flask 服务，需额外组合验证和契约文档 |
| React + Django | ORM、认证和管理后台完整 | 当前重点是实验接口和计算任务，未要求多用户业务管理 |
| 同时重写 RL 核心 | 可以统一风格 | 会失去行为基线，带来与界面目标无关的研究风险 |

## 影响与代价

前后端可围绕 HTTP 契约协作，CLI 不需要通过网络才能运行。代价是同时维护 Python 和 TypeScript 工具链、接口生成、状态同步和两个开发服务。依赖边界必须通过测试约束，不能只靠目录名称。

前后端与 StockTrader 使用一致技术栈有利于维护，但本次不合并仓库、不共享数据库、不复制其选股业务，也不承诺其组件可直接复用。

## 验证与重审条件

AC-02、AC-13、AC-14、AC-15 验证契约、导入边界和核心回归。若多人权限与管理后台成为主产品，或出现 SSR/公开内容需求，再比较 Django、全栈 React 框架或独立服务；不以“以后可能需要”预先增加这些系统。
