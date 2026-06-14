from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def dataframe_records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []

    frame = df.copy()
    if limit is not None:
        frame = frame.tail(limit)

    index_name = frame.index.name or "date"
    frame = frame.reset_index().rename(columns={frame.reset_index().columns[0]: index_name})
    records = frame.to_dict(orient="records")
    return [json_safe(record) for record in records]


def dataframe_payload(df: pd.DataFrame, limit: int | None = None) -> dict[str, Any]:
    records = dataframe_records(df, limit=limit)
    return {
        "columns": [str(c) for c in df.columns] if df is not None else [],
        "records": records,
        "row_count": int(len(df)) if df is not None else 0,
    }
