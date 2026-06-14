import { useEffect, useRef } from "react";
import * as echarts from "echarts";

interface EChartProps {
  option: Record<string, unknown>;
  height?: number;
}

export function EChart({ option, height = 320 }: EChartProps) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current) {
      return;
    }
    const chart = echarts.init(ref.current, "dark");
    chart.setOption(option as echarts.EChartsOption, true);
    const resize = () => chart.resize();
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      chart.dispose();
    };
  }, [option]);

  return <div className="echart" ref={ref} style={{ height }} />;
}
