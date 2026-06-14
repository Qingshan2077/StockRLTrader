import { Navigate, Route, Routes } from "react-router-dom";
import { Shell } from "./layout/Shell";
import { AlphaLab } from "./pages/AlphaLab";
import { CrossSection } from "./pages/CrossSection";
import { Dashboard } from "./pages/Dashboard";
import { DataCenter } from "./pages/DataCenter";
import { ExecutionLab } from "./pages/ExecutionLab";
import { Experiments } from "./pages/Experiments";
import { FactorMonitor } from "./pages/FactorMonitor";
import { ForecastLab } from "./pages/ForecastLab";
import { Pipeline } from "./pages/Pipeline";
import { RiskBacktest } from "./pages/RiskBacktest";
import { Settings } from "./pages/Settings";
import { SignalEvaluation } from "./pages/SignalEvaluation";

export default function App() {
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/data" element={<DataCenter />} />
        <Route path="/forecast" element={<ForecastLab />} />
        <Route path="/alpha" element={<AlphaLab />} />
        <Route path="/signals" element={<SignalEvaluation />} />
        <Route path="/risk" element={<RiskBacktest />} />
        <Route path="/execution" element={<ExecutionLab />} />
        <Route path="/experiments" element={<Experiments />} />
        <Route path="/pipeline" element={<Pipeline />} />
        <Route path="/cross-section" element={<CrossSection />} />
        <Route path="/factor-monitor" element={<FactorMonitor />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Shell>
  );
}
