import type { CandleRecord } from "../api/types";

interface CandlePreviewProps {
  records: CandleRecord[];
}

export function CandlePreview({ records }: CandlePreviewProps) {
  const points = records
    .filter((row) => typeof row.Close === "number")
    .slice(-120)
    .map((row) => row.Close as number);

  if (points.length < 2) {
    return <div className="empty-state">暂无可展示的价格数据</div>;
  }

  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = Math.max(max - min, 1e-9);
  const width = 900;
  const height = 320;
  const step = width / Math.max(points.length - 1, 1);
  const path = points
    .map((value, index) => {
      const x = index * step;
      const y = height - ((value - min) / span) * height;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");

  const last = points[points.length - 1];
  const first = points[0];
  const up = last >= first;

  return (
    <div className="chart-frame">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Close price preview">
        <defs>
          <linearGradient id="price-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={up ? "#36c48f" : "#ff6b5f"} stopOpacity="0.28" />
            <stop offset="100%" stopColor={up ? "#36c48f" : "#ff6b5f"} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={`${path} L ${width} ${height} L 0 ${height} Z`} fill="url(#price-fill)" />
        <path d={path} fill="none" stroke={up ? "#36c48f" : "#ff6b5f"} strokeWidth="2.5" />
      </svg>
    </div>
  );
}
