interface MetricTileProps {
  label: string;
  value: string;
  detail?: string;
  tone?: "neutral" | "bull" | "bear" | "gold" | "cyan";
}

export function MetricTile({ label, value, detail, tone = "neutral" }: MetricTileProps) {
  return (
    <div className={`metric-tile tone-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {detail ? <small>{detail}</small> : null}
    </div>
  );
}
