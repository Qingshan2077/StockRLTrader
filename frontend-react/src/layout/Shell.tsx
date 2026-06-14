import { useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { api } from "../api/client";
import { useT } from "../i18n";
import { navItems } from "./navigation";

interface ShellProps {
  children: ReactNode;
}

export function Shell({ children }: ShellProps) {
  const t = useT();
  const summary = useQuery({
    queryKey: ["system-summary"],
    queryFn: api.systemSummary
  });

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">ST</div>
          <div>
            <strong>StockTrader</strong>
            <span>Quant Workbench</span>
          </div>
        </div>
        <nav>
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink key={item.path} to={item.path} className={({ isActive }) => (isActive ? "active" : "")}>
                <Icon size={18} />
                <span>{t(item.labelKey)}</span>
              </NavLink>
            );
          })}
        </nav>
      </aside>
      <main className="workspace">
        <header className="topbar">
          <div className="status-pill">API {summary.isError ? "offline" : "online"}</div>
          <div className="topbar-meta">
            <span>{summary.data?.stage ?? "single_asset"}</span>
            <span>{summary.data?.available_tickers ?? 0} symbols</span>
            <span>v{summary.data?.version ?? "3.0"}</span>
          </div>
        </header>
        <section className="content">{children}</section>
      </main>
    </div>
  );
}
