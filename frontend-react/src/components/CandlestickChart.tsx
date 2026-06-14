import { useEffect, useMemo, useRef } from "react";
import { ColorType, createChart } from "lightweight-charts";
import type { CandleRecord } from "../api/types";

interface CandlestickChartProps {
  records: CandleRecord[];
}

function getRecordDate(record: CandleRecord): string | null {
  const value = record.Date ?? record.date;
  return typeof value === "string" ? value.slice(0, 10) : null;
}

export function CandlestickChart({ records }: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartData = useMemo(
    () =>
      records
        .map((record) => {
          const time = getRecordDate(record);
          if (
            !time ||
            typeof record.Open !== "number" ||
            typeof record.High !== "number" ||
            typeof record.Low !== "number" ||
            typeof record.Close !== "number"
          ) {
            return null;
          }
          return {
            time,
            open: record.Open,
            high: record.High,
            low: record.Low,
            close: record.Close,
            volume: typeof record.Volume === "number" ? record.Volume : 0
          };
        })
        .filter((item): item is NonNullable<typeof item> => item !== null),
    [records]
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container || chartData.length === 0) {
      return;
    }

    const chart = createChart(container, {
      width: container.clientWidth,
      height: 420,
      layout: {
        background: { type: ColorType.Solid, color: "#10141c" },
        textColor: "#9aa7b8"
      },
      grid: {
        vertLines: { color: "#1f2836" },
        horzLines: { color: "#1f2836" }
      },
      rightPriceScale: {
        borderColor: "#263040"
      },
      timeScale: {
        borderColor: "#263040",
        timeVisible: false
      }
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: "#36c48f",
      downColor: "#ff6b5f",
      borderUpColor: "#36c48f",
      borderDownColor: "#ff6b5f",
      wickUpColor: "#36c48f",
      wickDownColor: "#ff6b5f"
    });

    candleSeries.setData(
      chartData.map((item) => ({
        time: item.time,
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close
      }))
    );

    const volumeSeries = chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "",
      color: "#354357"
    });

    volumeSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.82,
        bottom: 0
      }
    });

    volumeSeries.setData(
      chartData.map((item) => ({
        time: item.time,
        value: item.volume,
        color: item.close >= item.open ? "rgba(54,196,143,0.35)" : "rgba(255,107,95,0.35)"
      }))
    );

    chart.timeScale().fitContent();

    const resizeObserver = new ResizeObserver(() => {
      chart.applyOptions({ width: container.clientWidth });
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
    };
  }, [chartData]);

  if (chartData.length === 0) {
    return <div className="empty-state">No candle data available.</div>;
  }

  return <div ref={containerRef} className="lw-chart" />;
}
