import { SectionHeader } from "../components/SectionHeader";
import { TerminalPanel } from "../components/TerminalPanel";

interface PagePlaceholderProps {
  title: string;
  eyebrow: string;
  scope: string[];
}

export function PagePlaceholder({ title, eyebrow, scope }: PagePlaceholderProps) {
  return (
    <>
      <SectionHeader title={title} eyebrow={eyebrow} />
      <TerminalPanel title="Migration Scope">
        <ul className="scope-list">
          {scope.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </TerminalPanel>
    </>
  );
}
