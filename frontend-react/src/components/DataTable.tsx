interface DataTableProps {
  rows: Record<string, unknown>[];
  columns?: string[];
  maxRows?: number;
}

export function DataTable({ rows, columns, maxRows = 50 }: DataTableProps) {
  const visible = rows.slice(0, maxRows);
  const cols = columns ?? Array.from(new Set(visible.flatMap((row) => Object.keys(row)))).slice(0, 12);

  if (visible.length === 0) {
    return <div className="empty-state">No table data.</div>;
  }

  return (
    <div className="data-table">
      <table>
        <thead>
          <tr>
            {cols.map((col) => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {visible.map((row, index) => (
            <tr key={index}>
              {cols.map((col) => (
                <td key={col}>{formatCell(row[col])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatCell(value: unknown): string {
  if (typeof value === "number") {
    return Number.isInteger(value) ? `${value}` : value.toFixed(4);
  }
  if (value === null || value === undefined) {
    return "-";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}
