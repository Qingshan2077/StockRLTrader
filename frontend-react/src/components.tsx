import { Component, type ErrorInfo, type ReactNode, useState } from "react";
import { Link } from "react-router-dom";
import { ApiError } from "./api/client";
import type { Dataset, ExperimentDetail, JobDetail } from "./api/contracts";
import {
  experimentName,
  integrityNames,
  number,
  phaseNames,
  statusNames,
  timestamp,
} from "./display";

export class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: boolean }
> {
  state = { error: false };
  static getDerivedStateFromError() {
    return { error: true };
  }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("页面显示失败", error, info.componentStack);
  }
  render() {
    return this.state.error ? (
      <main className="fatal">
        <h1>页面暂时无法显示</h1>
        <p>已提交的任务仍由服务端管理。重新加载后按任务 ID 查询。</p>
        <button onClick={() => location.reload()}>重新加载</button>
      </main>
    ) : (
      this.props.children
    );
  }
}
export function ErrorBox({
  error,
  retry,
}: {
  error: unknown;
  retry?: () => void;
}) {
  if (!error) return null;
  return (
    <div className="notice error" role="alert">
      <strong>{error instanceof Error ? error.message : "请求失败。"}</strong>
      {error instanceof ApiError && (
        <>
          <small>
            {error.code}
            {error.requestId && ` · 请求编号 ${error.requestId}`}
          </small>
          {error.details != null && (
            <details>
              <summary>查看字段说明</summary>
              <pre>{JSON.stringify(error.details, null, 2)}</pre>
            </details>
          )}
        </>
      )}
      {retry && (
        <button className="secondary" onClick={retry}>
          重试
        </button>
      )}
    </div>
  );
}
export function Loading() {
  return (
    <p className="loading" role="status">
      正在读取…
    </p>
  );
}
export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}
export function PageTitle({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children?: ReactNode;
}) {
  return (
    <header className="page-title">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
      </div>
      <div>{children}</div>
    </header>
  );
}
export function Badge({
  children,
  tone = "",
}: {
  children: ReactNode;
  tone?: string;
}) {
  return <span className={`badge ${tone}`}>{children}</span>;
}
export function JobBadge({ job }: { job: JobDetail }) {
  return (
    <Badge
      tone={
        job.status === "succeeded"
          ? "good"
          : ["failed", "interrupted"].includes(job.status)
            ? "bad"
            : ""
      }
    >
      {statusNames[job.status]}
    </Badge>
  );
}
export function Provenance({ experiment }: { experiment: ExperimentDetail }) {
  return (
    <div className="badges">
      <Badge>
        {experiment.kind === "replay"
          ? "原样重放"
          : experiment.request?.purpose === "research"
            ? "研究实验"
            : experiment.request
              ? "技术验证"
              : "历史产物"}
      </Badge>
      {experiment.is_synthetic === true && (
        <Badge tone="warning">合成演示</Badge>
      )}
      {experiment.is_synthetic === null && <Badge>数据性质未知</Badge>}
      <Badge tone={experiment.integrity === "complete" ? "good" : "warning"}>
        {integrityNames[experiment.integrity]}
      </Badge>
    </div>
  );
}
export function DatasetFacts({ dataset }: { dataset: Dataset }) {
  return (
    <dl className="facts">
      <div>
        <dt>数据范围</dt>
        <dd>
          {dataset.first_date} — {dataset.last_date}
        </dd>
      </div>
      <div>
        <dt>数据行数</dt>
        <dd>{number(dataset.rows, 0)}</dd>
      </div>
      <div>
        <dt>计价单位</dt>
        <dd>
          {dataset.quote_unit === "unknown"
            ? "未声明计价单位"
            : dataset.quote_unit}
        </dd>
      </div>
      <div>
        <dt>复权方式</dt>
        <dd>
          {dataset.adjustment === "unknown" ? "未声明" : dataset.adjustment}
        </dd>
      </div>
      <div className="wide">
        <dt>数据指纹 SHA256</dt>
        <dd className="mono break">{dataset.snapshot_sha256}</dd>
      </div>
    </dl>
  );
}
export function JobProgress({ job }: { job: JobDetail }) {
  return (
    <section className="panel">
      <div className="section-heading">
        <h2>{phaseNames[job.phase ?? ""] ?? "等待开始"}</h2>
        <JobBadge job={job} />
      </div>
      {!job.worker_available &&
        !["succeeded", "failed", "cancelled", "interrupted"].includes(
          job.status,
        ) && (
          <p className="notice warning">
            执行器离线，任务状态待确认。排队任务将在执行器恢复后处理。
          </p>
        )}
      {job.recovery_waiting && (
        <p className="notice warning">
          正在等待旧执行进程退出，暂不领取新任务。
        </p>
      )}
      <dl className="facts">
        <div>
          <dt>当前 seed</dt>
          <dd>
            {job.seed ?? "—"}
            {job.seed_index !== null &&
              `（${job.seed_index + 1}/${job.seed_count}）`}
          </dd>
        </div>
        <div>
          <dt>{job.kind === "research" ? "已完成研究单元" : "已完成 seed"}</dt>
          <dd>
            {job.completed_seeds} / {job.seed_count}
          </dd>
        </div>
        {(job.kind === "train" || job.kind === "research") && (
          <div>
            <dt>实际 / 计划训练步数</dt>
            <dd>
              {number(job.actual_steps_total, 0)} /{" "}
              {number(job.requested_steps_total, 0)}
            </dd>
          </div>
        )}
        <div>
          <dt>最近心跳</dt>
          <dd>{timestamp(job.heartbeat_at)}</dd>
        </div>
      </dl>
      {job.kind === "train" && job.training_fraction !== null && (
        <>
          <label className="progress-label" htmlFor="training-progress">
            训练步进度 {number(job.training_fraction * 100, 1)}%
          </label>
          <progress
            id="training-progress"
            value={job.training_fraction}
            max={1}
          />
          <p className="hint">
            训练步数达到计划后仍需评估与保存；只有状态为“已完成”才表示结果发布成功。
          </p>
        </>
      )}
      {job.kind === "replay" && (
        <p className="hint">
          重放不重新训练，仅核验原模型与保存数据的可复现性。
        </p>
      )}
      {job.error && (
        <div role="alert" className="notice error">
          {job.error.message}
          <small>{job.error.code}</small>
        </div>
      )}
    </section>
  );
}
export function ExperimentLink({ item }: { item: ExperimentDetail }) {
  return (
    <Link to={`/experiments/${item.experiment_id}`}>
      <strong>{experimentName(item)}</strong>
      <small className="mono">{item.experiment_id.slice(0, 8)}</small>
    </Link>
  );
}
export function CursorButtons({
  hasMore,
  next,
  previous,
  canPrevious,
  busy = false,
}: {
  hasMore: boolean;
  next: () => void;
  previous: () => void;
  canPrevious: boolean;
  busy?: boolean;
}) {
  return (
    <nav className="pagination" aria-label="分页">
      <button
        className="secondary"
        disabled={!canPrevious || busy}
        onClick={previous}
      >
        上一页
      </button>
      <button className="secondary" disabled={!hasMore || busy} onClick={next}>
        下一页
      </button>
    </nav>
  );
}
export function useCursor() {
  const [cursors, setCursors] = useState<Array<string | null>>([null]);
  return {
    cursor: cursors[cursors.length - 1],
    canPrevious: cursors.length > 1,
    previous: () => setCursors((old) => old.slice(0, -1)),
    next: (value: string | null) => {
      if (value) setCursors((old) => [...old, value]);
    },
    reset: () => setCursors([null]),
  };
}
export function ConfirmAbandon({ onConfirm }: { onConfirm: () => void }) {
  const [open, setOpen] = useState(false);
  return open ? (
    <div className="notice warning">
      <p>
        放弃本地重试记录不会取消服务器上可能已创建的任务。请先在任务列表确认，避免重复运行。
      </p>
      <div className="actions">
        <Link to="/jobs">查看任务</Link>
        <button className="danger" onClick={onConfirm}>
          确认放弃重试记录
        </button>
        <button className="secondary" onClick={() => setOpen(false)}>
          保留
        </button>
      </div>
    </div>
  ) : (
    <button className="secondary" onClick={() => setOpen(true)}>
      放弃这次提交记录
    </button>
  );
}
