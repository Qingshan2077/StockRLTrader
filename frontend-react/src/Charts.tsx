import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import type { Data, Layout } from "plotly.js";
import type { Policy, SeriesResponse } from "./api/contracts";
import { policyNames } from "./display";

const Plot = createPlotlyComponent(Plotly);
const colors = ["#61d3ba", "#e5b567", "#8cb4f1", "#d99fcb", "#c7cfb6"];
const dashes = ["solid", "dash", "dot", "dashdot", "longdash"] as const;
export default function Charts({
  series,
  field,
  unit,
}: {
  series: SeriesResponse[];
  field: "nav" | "weights" | "cash" | "cost";
  unit: string;
}) {
  const traces: Data[] = [];
  series.forEach((item, index) => {
    const keys =
      field === "weights"
        ? (["weight", "requested_weight", "executed_weight"] as const)
        : [field];
    keys.forEach((key, keyIndex) =>
      traces.push({
        x: item.points.map((point) => point.date),
        y: item.points.map((point) => point[key]),
        type: "scatter",
        mode: "lines",
        connectgaps: false,
        name: `${policyNames[item.policy as Policy]}${field === "weights" ? ` · ${{ weight: "收盘实际仓位", requested_weight: "请求目标仓位", executed_weight: "开盘成交后仓位" }[key as "weight" | "requested_weight" | "executed_weight"]}` : ""}`,
        line: {
          color: colors[index % colors.length],
          width: 1.8,
          dash:
            field === "weights"
              ? dashes[keyIndex]
              : dashes[index % dashes.length],
        },
        hovertemplate: `%{x}<br>%{y${field === "weights" ? ":.2%" : ":,.3f"}}<extra>%{fullData.name}</extra>`,
      }),
    );
  });
  const layout: Partial<Layout> = {
    autosize: true,
    height: field === "weights" ? 420 : 360,
    paper_bgcolor: "#14212c",
    plot_bgcolor: "#14212c",
    font: {
      family: "Segoe UI, Microsoft YaHei, sans-serif",
      color: "#e7edf2",
      size: 12,
    },
    margin: { l: 70, r: 20, t: 16, b: 80 },
    hovermode: "x unified",
    legend: { orientation: "h", y: -0.2 },
    xaxis: { gridcolor: "#2b3d4b", zeroline: false, title: { text: "日期" } },
    yaxis: {
      gridcolor: "#2b3d4b",
      zerolinecolor: "#526170",
      tickformat: field === "weights" ? ".0%" : undefined,
      title: { text: field === "weights" ? "仓位比例" : unit },
    },
    uirevision: `${field}-${series.map((item) => item.seed + item.policy).join()}`,
  };
  return (
    <Plot
      data={traces}
      layout={layout}
      useResizeHandler
      style={{ width: "100%" }}
      config={{
        responsive: true,
        displaylogo: false,
        scrollZoom: false,
        modeBarButtonsToRemove: ["toImage"],
      }}
    />
  );
}
