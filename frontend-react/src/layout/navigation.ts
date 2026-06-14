import {
  Binary,
  Bot,
  Boxes,
  CandlestickChart,
  Database,
  FlaskConical,
  Gauge,
  GitCompare,
  LineChart,
  Radar,
  Settings,
  Workflow
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { TranslationKey } from "../i18n";

export const navItems = [
  { path: "/dashboard", labelKey: "dashboard", icon: CandlestickChart },
  { path: "/data", labelKey: "dataCenter", icon: Database },
  { path: "/forecast", labelKey: "forecastLab", icon: LineChart },
  { path: "/alpha", labelKey: "alphaLab", icon: FlaskConical },
  { path: "/signals", labelKey: "signals", icon: Radar },
  { path: "/risk", labelKey: "riskBacktest", icon: Gauge },
  { path: "/execution", labelKey: "execution", icon: Bot },
  { path: "/experiments", labelKey: "experiments", icon: GitCompare },
  { path: "/pipeline", labelKey: "pipeline", icon: Workflow },
  { path: "/cross-section", labelKey: "crossSection", icon: Boxes },
  { path: "/factor-monitor", labelKey: "factorMonitor", icon: Binary },
  { path: "/settings", labelKey: "settings", icon: Settings }
] satisfies Array<{ path: string; labelKey: TranslationKey; icon: LucideIcon }>;
