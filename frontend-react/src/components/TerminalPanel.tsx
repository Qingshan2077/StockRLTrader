import type { ReactNode } from "react";

interface TerminalPanelProps {
  title: string;
  children: ReactNode;
  actions?: ReactNode;
}

export function TerminalPanel({ title, children, actions }: TerminalPanelProps) {
  return (
    <section className="terminal-panel">
      <header>
        <h2>{title}</h2>
        {actions ? <div>{actions}</div> : null}
      </header>
      {children}
    </section>
  );
}
