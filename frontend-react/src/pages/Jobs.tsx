import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  request,
  search,
  terminal,
  useJob,
  usePollInterval,
} from "../api/client";
import type {
  EventPage,
  JobDetail as Job,
  JobEvent,
  Page,
} from "../api/contracts";
import { readStored, saveStored } from "../api/submission";
import {
  CursorButtons,
  Empty,
  ErrorBox,
  JobBadge,
  JobProgress,
  Loading,
  PageTitle,
  useCursor,
} from "../components";
import { phaseNames, statusNames, timestamp } from "../display";

export function Jobs() {
  const cursor = useCursor();
  const [status, setStatus] = useState("");
  const [kind, setKind] = useState("");
  const interval = usePollInterval();
  const query = useQuery({
    queryKey: ["jobs", status, kind, cursor.cursor],
    queryFn: ({ signal }) =>
      request<Page<Job>>(
        `/jobs${search({ status, kind, cursor: cursor.cursor })}`,
        { signal },
      ),
    retry: false,
    refetchInterval: (q) => interval(q.state),
  });
  return (
    <>
      <PageTitle eyebrow="QUEUE / 任务队列" title="任务">
        <Link className="button" to="/experiments/new">
          创建实验
        </Link>
      </PageTitle>
      <div className="filters">
        <label>
          任务状态
          <select
            value={status}
            onChange={(event) => {
              setStatus(event.target.value);
              cursor.reset();
            }}
          >
            <option value="">全部状态</option>
            {Object.entries(statusNames).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label>
          任务类型
          <select
            value={kind}
            onChange={(event) => {
              setKind(event.target.value);
              cursor.reset();
            }}
          >
            <option value="">全部类型</option>
            <option value="train">训练</option>
            <option value="replay">重放</option>
            <option value="research">研究</option>
          </select>
        </label>
      </div>
      <ErrorBox error={query.error} retry={() => void query.refetch()} />
      {query.isPending && <Loading />}
      {query.data && !query.data.items.length && (
        <Empty>没有符合条件的任务。</Empty>
      )}
      {query.data && (
        <div className="panel table-scroll">
          <table>
            <thead>
              <tr>
                <th>任务</th>
                <th>类型</th>
                <th>状态</th>
                <th>当前阶段</th>
                <th>完成单元 / seed</th>
                <th>创建时间</th>
              </tr>
            </thead>
            <tbody>
              {query.data.items.map((job) => (
                <tr key={job.job_id}>
                  <td>
                    <Link
                      className="mono"
                      to={
                        job.kind === "research"
                          ? `/researches/${job.experiment_id}`
                          : `/jobs/${job.job_id}`
                      }
                    >
                      {job.job_id.slice(0, 8)}
                    </Link>
                  </td>
                  <td>
                    {job.kind === "research"
                      ? "研究"
                      : job.kind === "train"
                        ? "训练"
                        : "重放"}
                  </td>
                  <td>
                    <JobBadge job={job} />
                    {!job.worker_available && !terminal(job.status) && (
                      <small>执行器离线</small>
                    )}
                    {job.recovery_waiting && <small>恢复等待</small>}
                  </td>
                  <td>{phaseNames[job.phase ?? ""] ?? "—"}</td>
                  <td className="numeric">
                    {job.completed_seeds} / {job.seed_count}
                  </td>
                  <td>{timestamp(job.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <CursorButtons
        hasMore={query.data?.has_more ?? false}
        next={() => cursor.next(query.data?.next_cursor ?? null)}
        previous={cursor.previous}
        canPrevious={cursor.canPrevious}
        busy={query.isFetching}
      />
    </>
  );
}
export function mergeEvents(
  existing: JobEvent[],
  incoming: JobEvent[],
): JobEvent[] {
  const bySeq = new Map(existing.map((event) => [event.seq, event]));
  incoming.forEach((event) => bySeq.set(event.seq, event));
  return [...bySeq.values()].sort((a, b) => a.seq - b.seq).slice(-1000);
}
function Events({ id, done }: { id: string; done: boolean }) {
  const key = `stockrl.events.${id}`;
  const [events, setEvents] = useState<JobEvent[]>(
    () => readStored<{ events: JobEvent[] }>(key)?.events ?? [],
  );
  const [error, setError] = useState<Error | null>(null);
  const [restart, setRestart] = useState(0);
  const [busy, setBusy] = useState(true);
  useEffect(() => {
    const stored = readStored<{ events: JobEvent[]; cursor: number }>(key);
    let collected = stored?.events ?? [];
    let cursor = stored?.cursor ?? 0;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;
    async function poll() {
      try {
        const page = await request<EventPage>(
          `/jobs/${id}/events${search({ after_seq: cursor, limit: 100 })}`,
          { signal: controller.signal },
        );
        if (controller.signal.aborted) return;
        collected = mergeEvents(collected, page.items);
        cursor = page.next_seq;
        setEvents(collected);
        setError(null);
        setBusy(false);
        failures = 0;
        try {
          saveStored(key, { events: collected, cursor });
        } catch {
          /* The current page retains events if session storage is full. */
        }
        if (page.has_more || !done)
          timer = setTimeout(() => void poll(), page.has_more ? 50 : 2000);
      } catch (reason) {
        if (controller.signal.aborted) return;
        setBusy(false);
        setError(reason as Error);
        failures++;
        timer = setTimeout(
          () => void poll(),
          Math.min(10000, 2000 * 2 ** failures),
        );
      }
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [id, key, done, restart]);
  function description(event: JobEvent) {
    const payload = event.payload;
    if ("status" in payload) return statusNames[payload.status];
    if ("phase" in payload)
      return `${phaseNames[payload.phase]}${payload.seed !== null ? ` · seed ${payload.seed}` : ""}`;
    if ("actual_steps_total" in payload)
      return `实际 ${payload.actual_steps_total} 步 · 已完成 ${payload.completed_seeds} 个 seed`;
    return payload.message;
  }
  return (
    <section className="panel">
      <h2>事件记录</h2>
      <p className="hint">
        按序号补取并去重，展示最近 1000 条；刷新后继续读取。
      </p>
      {busy && <Loading />}
      <ErrorBox error={error} retry={() => setRestart((old) => old + 1)} />
      {!busy && !events.length && <Empty>还没有事件。</Empty>}
      <ol className="events">
        {events.map((event) => (
          <li key={event.seq}>
            <span className="mono muted">#{event.seq}</span>
            <time>{timestamp(event.occurred_at)}</time>
            <span>{description(event)}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}
export function JobDetail() {
  const { jobId = "" } = useParams();
  const query = useJob(jobId);
  const client = useQueryClient();
  const [cancelBusy, setCancelBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  async function cancel() {
    if (cancelBusy) return;
    setCancelBusy(true);
    setError(null);
    try {
      const job = await request<Job>(`/jobs/${jobId}/cancel`, {
        method: "POST",
        body: "{}",
      });
      client.setQueryData(["job", jobId], job);
      await client.invalidateQueries({ queryKey: ["jobs"] });
    } catch (reason) {
      setError(reason as Error);
    } finally {
      setCancelBusy(false);
    }
  }
  return (
    <>
      <PageTitle eyebrow="EXECUTION / 执行状态" title="任务详情">
        <Link to="/jobs">返回任务列表</Link>
      </PageTitle>
      <p className="mono break">{jobId}</p>
      {query.isPending && <Loading />}
      {query.error && (
        <p className="notice warning">
          连接中断，任务状态待确认。浏览器断线不会取消执行。
        </p>
      )}
      <ErrorBox error={query.error} retry={() => void query.refetch()} />
      <ErrorBox error={error} />
      {query.data && (
        <>
          <JobProgress job={query.data} />
          <div className="actions">
            <Link
              className="button secondary"
              to={
                query.data.kind === "research"
                  ? `/researches/${query.data.experiment_id}`
                  : `/experiments/${query.data.experiment_id}`
              }
            >
              {query.data.kind === "research" ? "查看研究" : "查看实验"}
            </Link>
            <button
              className="danger"
              disabled={
                terminal(query.data.status) ||
                query.data.status === "cancelling" ||
                cancelBusy
              }
              onClick={() => void cancel()}
            >
              {query.data.status === "cancelling"
                ? "等待确认计算停止…"
                : cancelBusy
                  ? "发送取消请求…"
                  : "取消任务"}
            </button>
          </div>
          <p className="hint">
            取消运行中的任务会先进入“正在取消”，确认计算停止后才进入终态。已结束任务不会重新入队。
          </p>
          <Events key={jobId} id={jobId} done={terminal(query.data.status)} />
        </>
      )}
    </>
  );
}
