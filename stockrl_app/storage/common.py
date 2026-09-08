import base64
import json
import sqlite3
from typing import TypeVar

from ..errors import AppError
from ..models import Page, StrictModel

T = TypeVar('T', bound=StrictModel)


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def page_rows(connection: sqlite3.Connection, table: str, id_column: str, model: type[T], *,
              limit: int = 20, cursor: str | None = None,
              filters: dict[str, str] | None = None,
              filter_columns: dict[str, str] | None = None) -> Page[T]:
    """Identifiers are repository constants. Cursor is scoped to table and filters."""
    if type(limit) is not int or not 1 <= limit <= 100:
        raise AppError('INVALID_PAGINATION', '每页条数必须介于 1 和 100。')
    clauses, values = [], []
    scope = canonical_json({'table': table, 'filters': filters or {}})
    for name, value in (filters or {}).items():
        column = (filter_columns or {}).get(name, name)
        clauses.append(f'{column}=?')
        values.append(value)
    if cursor:
        try:
            if len(cursor) > 4096:
                raise ValueError('oversize cursor')
            decoded = json.loads(base64.urlsafe_b64decode(cursor + '=' * (-len(cursor) % 4)))
            if not isinstance(decoded, list) or len(decoded) != 3 or decoded[0] != scope:
                raise ValueError('cursor scope')
            if not all(isinstance(value, str) for value in decoded):
                raise ValueError('cursor type')
        except (ValueError, TypeError, UnicodeError) as exc:
            raise AppError('INVALID_CURSOR', '分页游标无效，请重新加载列表。') from exc
        clauses.append(f'(created_at < ? OR (created_at = ? AND {id_column} < ?))')
        values.extend((decoded[1], decoded[1], decoded[2]))
    where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
    rows = connection.execute(f'SELECT * FROM {table}{where} ORDER BY created_at DESC,{id_column} DESC LIMIT ?',
                              (*values, limit + 1)).fetchall()
    has_more = len(rows) > limit
    visible = rows[:limit]
    next_cursor = None
    if has_more:
        last = visible[-1]
        next_cursor = base64.urlsafe_b64encode(canonical_json([scope, last['created_at'], last[id_column]]).encode()).decode().rstrip('=')
    return Page[model](items=[model.model_validate_json(row['payload_json']) for row in visible],
                       next_cursor=next_cursor, has_more=has_more)
