import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { request, search, usePollInterval } from "../api/client";
import { qualificationNames, type ResearchPage } from "../api/research";
import {
  CursorButtons,
  Empty,
  ErrorBox,
  Loading,
  PageTitle,
  useCursor,
} from "../components";
import { statusNames, timestamp } from "../display";
export default function Researches() {
  const cursor = useCursor(),
    interval = usePollInterval();
  const query = useQuery({
    queryKey: ["researches", cursor.cursor],
    queryFn: ({ signal }) =>
      request<ResearchPage>(`/researches${search({ cursor: cursor.cursor })}`, {
        signal,
      }),
    refetchInterval: (q) => interval(q.state),
  });
  return (
    <>
      <PageTitle eyebrow="RESEARCH / 研究记录" title="研究">
        <div className="actions">
          <Link className="button" to="/researches/new">
            创建研究
          </Link>
          <Link to="/market-datasets">市场数据</Link>
        </div>
      </PageTitle>
      <p>
        执行状态、研究资格与经济结论分别判定。详情保留每个窗口和随机种子的结果。
      </p>
      <ErrorBox error={query.error} retry={() => void query.refetch()} />
      {query.isPending && <Loading />}
      {query.data?.items.length === 0 && (
        <Empty>还没有研究。登记数据后，先预览预算再提交。</Empty>
      )}
      {!!query.data?.items.length && (
        <div className="panel table-scroll">
          <table>
            <thead>
              <tr>
                <th>研究命题</th>
                <th>执行状态</th>
                <th>研究资格</th>
                <th>创建时间</th>
              </tr>
            </thead>
            <tbody>
              {query.data.items.map((r) => (
                <tr key={r.research_id}>
                  <td>
                    <Link to={`/researches/${r.research_id}`}>
                      {r.hypothesis}
                    </Link>
                  </td>
                  <td>
                    {statusNames[r.technical_status] ?? r.technical_status}
                  </td>
                  <td>{qualificationNames[r.qualification]}</td>
                  <td>{timestamp(r.created_at)}</td>
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
      <Link to="/experiments">查看旧版实验记录</Link>
    </>
  );
}
