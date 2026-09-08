# React 研究工作台

React、TypeScript、React Router、TanStack Query 与 Plotly。数据集、实验预览与提交、任务/事件、策略轨迹、产物下载与最多四个实验的条件比较均使用 `/api/v1`。

本轮仅编写代码、安装静态工具并检查类型；**没有运行构建、单元测试、浏览器、前后端服务或训练**。以下运行命令留待后续验收授权：

```text
npm ci --ignore-scripts
npm run typecheck
npm run dev
npm run build
npm test
```

开发服务器绑定 `127.0.0.1:5173`，将 `/api` 代理到 `127.0.0.1:8000`。构建输出到 `../api/static`；构建会清理该静态产物目录，不应在其中保存手工源文件。Vite 不自动启动 API 或 worker。

`src/api/contracts.ts` 由仓库的 Python AST 契约脚本生成。运行时 OpenAPI 一致性仍需后续验收：先使用根目录的导出脚本生成 `frontend-react/openapi.json`，再运行 `npm run contracts:openapi`。当前 UI 使用 AST 契约，不能把静态类型检查当作运行时兼容验证。

提交前将不可变请求和 UUID 幂等键保存在浏览器会话存储；未确认结果时只允许重试同一请求，显式放弃本地记录不会取消服务器任务。训练表单从服务端 capabilities 初始化一次，后续刷新不会覆盖草稿。事件按游标补取，当前会话保留最近 1000 条显示记录。

颜色、布局与图表可读性尚未经过浏览器或截图验收。`src/display.test.ts` 已编写但未运行；不存在已通过的端到端测试声明。
