"""Explicit import of prior research knowledge; no inference from a dataset hash."""
import hashlib
import json
from pathlib import Path

from stockrl.research.contracts import ExposureRecord

from .storage.common import canonical_json


def import_exposures(db, path: Path) -> int:
    values = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(values, list) or len(values) > 10000:
        raise ValueError('Exposure import must contain at most 10000 records')
    records = [ExposureRecord.model_validate(value).model_dump(mode='json') for value in values]
    inserted = 0
    with db.transaction() as connection:
        for record in records:
            encoded = canonical_json(record)
            result = connection.execute('''INSERT OR IGNORE INTO exposure_records
                (exposure_id,research_id,instrument_id,interval_start,interval_end,exposed_at,payload_json)
                VALUES(?,NULL,?,?,?,?,?)''', (hashlib.sha256(encoded.encode()).hexdigest(), record['instrument_id'],
                record['start_session'], record['end_session'], record['recorded_at'], encoded))
            inserted += result.rowcount
    return inserted
