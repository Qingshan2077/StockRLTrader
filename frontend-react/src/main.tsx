import { StrictMode, Suspense, lazy } from "react";
import { createRoot } from "react-dom/client";
import {
  BrowserRouter,
  Link,
  NavLink,
  Navigate,
  Outlet,
  Route,
  Routes,
} from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ApiError, useCapabilities } from "./api/client";
import { ErrorBoundary, Loading } from "./components";
import "./styles.css";

const Datasets = lazy(() => import("./pages/Datasets"));
const NewExperiment = lazy(() => import("./pages/NewExperiment"));
const Jobs = lazy(() =>
  import("./pages/Jobs").then((module) => ({ default: module.Jobs })),
);
const JobDetail = lazy(() =>
  import("./pages/Jobs").then((module) => ({ default: module.JobDetail })),
);
const Experiments = lazy(() =>
  import("./pages/Experiments").then((module) => ({
    default: module.Experiments,
  })),
);
const ExperimentDetail = lazy(() =>
  import("./pages/Experiments").then((module) => ({
    default: module.ExperimentDetail,
  })),
);
const Compare = lazy(() =>
  import("./pages/Experiments").then((module) => ({ default: module.Compare })),
);
const client = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 10000,
      retry: (count, error) =>
        count < 2 &&
        (!(error instanceof ApiError) ||
          error.status === 0 ||
          error.status >= 500),
      retryDelay: (attempt) => Math.min(10000, 2000 * 2 ** attempt),
      refetchOnWindowFocus: true,
    },
    mutations: { retry: false },
  },
});
function Layout() {
  const caps = useCapabilities();
  return (
    <div className="shell">
      <a className="skip-link" href="#content">
        跳到主要内容
      </a>
      <aside className="sidebar">
        <Link className="brand" to="/experiments">
          <span>
            StockRL<span className="brand-dot">.</span>
          </span>
          <small>交易研究台</small>
        </Link>
        <nav aria-label="主导航">
          <NavLink to="/datasets">
            数据集<span>Data</span>
          </NavLink>
          <NavLink to="/experiments/new">
            创建实验<span>Research</span>
          </NavLink>
          <NavLink to="/jobs">
            任务<span>Execution</span>
          </NavLink>
          <NavLink to="/experiments" end>
            实验记录<span>Archive</span>
          </NavLink>
          <NavLink to="/compare">
            比较<span>Compare</span>
          </NavLink>
        </nav>
        <div className="sidebar-foot">
          <span
            className={`connection ${caps.data?.worker_available ? "available" : ""}`}
          />
          {caps.error
            ? "服务连接待确认"
            : caps.data?.worker_available
              ? "执行器在线"
              : "执行器未在线"}
          <p>单标的 · 日频 · 只做多</p>
          <small>买卖与仓位由策略决定</small>
        </div>
      </aside>
      <div className="workspace">
        <div className="topbar">
          <span>RL RESEARCH / 本地实验工作区</span>
          <span>观察 → 目标仓位 → 约束成交</span>
        </div>
        <main id="content">
          <Suspense fallback={<Loading />}>
            <Outlet />
          </Suspense>
        </main>
        <footer>
          实验结果用于策略研究。测试集不参与选模，重放不增加样本外证据。
        </footer>
      </div>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary>
      <QueryClientProvider client={client}>
        <BrowserRouter>
          <Routes>
            <Route element={<Layout />}>
              <Route index element={<Navigate replace to="/experiments" />} />
              <Route path="datasets" element={<Datasets />} />
              <Route path="experiments/new" element={<NewExperiment />} />
              <Route path="jobs" element={<Jobs />} />
              <Route path="jobs/:jobId" element={<JobDetail />} />
              <Route path="experiments" element={<Experiments />} />
              <Route
                path="experiments/:experimentId"
                element={<ExperimentDetail />}
              />
              <Route path="compare" element={<Compare />} />
              <Route
                path="*"
                element={
                  <section className="panel">
                    <h1>找不到这个页面</h1>
                    <Link to="/experiments">回到实验记录</Link>
                  </section>
                }
              />
            </Route>
          </Routes>
        </BrowserRouter>
      </QueryClientProvider>
    </ErrorBoundary>
  </StrictMode>,
);
