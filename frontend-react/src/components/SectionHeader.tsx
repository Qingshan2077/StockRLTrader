interface SectionHeaderProps {
  title: string;
  eyebrow?: string;
}

export function SectionHeader({ title, eyebrow }: SectionHeaderProps) {
  return (
    <div className="section-header">
      {eyebrow ? <span>{eyebrow}</span> : null}
      <h1>{title}</h1>
    </div>
  );
}
