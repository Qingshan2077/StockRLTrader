"""Offline immutable v2 bundles. Registration is independent from training eligibility."""
import json
import os
from pathlib import Path
from uuid import uuid4

from stockrl.market.contracts import to_dict
from stockrl.market.validation import validate_market_bundle, load_market_bundle

from .errors import AppError
from .models import utc_now
from .settings import confined_path
from .storage.common import canonical_json

FILE_NAMES = ('metadata.json', 'bars.csv', 'sessions.csv', 'actions.csv', 'tradability.csv', 'market-profile.json')


def public_preview(preview) -> dict:
    result = to_dict(preview)
    dataset = result.pop('dataset', None)
    if dataset:
        result['metadata'] = dataset['metadata']
        result['market_profile'] = dataset['market_profile']
        result['rows'] = len(dataset['bars'])
    else:
        result.update(metadata=None, market_profile=None, rows=0)
    return result


class MarketDatasetService:
    def __init__(self, settings, db):
        self.settings, self.db = settings, db

    def preview(self, files: dict[str, bytes]) -> dict:
        return public_preview(validate_market_bundle(files))

    def register(self, files: dict[str, bytes]) -> dict:
        preview = self.preview(files)
        if 'DATA_INVALID' in preview['blocking_reasons']:
            raise AppError('DATA_INVALID', '数据包格式或数据值无效。', 422, {'issues': preview['issues']})
        if 'BUNDLE_TOO_LARGE' in preview['blocking_reasons']:
            raise AppError('REQUEST_TOO_LARGE', '数据包超出大小限制。', 413)
        identifier = str(uuid4())
        relative = f'market-datasets/{identifier}'
        directory = confined_path(self.settings.app_dir, relative)
        directory.mkdir(parents=True, exist_ok=False)
        written = []
        keep = False
        try:
            for name in FILE_NAMES:
                if name not in files:
                    continue
                path = confined_path(directory, name)
                with path.open('xb') as stream:
                    stream.write(files[name])
                    stream.flush()
                    os.fsync(stream.fileno())
                written.append(path)
            record = {**preview, 'dataset_id': identifier, 'created_at': utc_now(), 'bundle_relative_path': relative}
            with self.db.transaction() as connection:
                connection.execute('''INSERT INTO market_datasets_v2
                    (dataset_id,created_at,snapshot_sha256,bundle_relative_path,payload_json) VALUES(?,?,?,?,?)''',
                    (identifier, record['created_at'], preview['fingerprint'], relative, canonical_json(record)))
            keep = True
            return self.public(record)
        finally:
            if not keep:
                for path in written:
                    path.unlink(missing_ok=True)
                directory.rmdir()

    @staticmethod
    def public(record: dict) -> dict:
        return {key: value for key, value in record.items() if key != 'bundle_relative_path'}

    def get(self, dataset_id: str) -> dict:
        with self.db.connection() as connection:
            row = connection.execute('SELECT payload_json FROM market_datasets_v2 WHERE dataset_id=?', (dataset_id,)).fetchone()
        if row is None:
            raise AppError('DATASET_NOT_FOUND', '市场数据包不存在。', 404)
        return json.loads(row[0])

    def list(self) -> dict:
        with self.db.connection() as connection:
            rows = connection.execute('SELECT payload_json FROM market_datasets_v2 ORDER BY created_at DESC LIMIT 100').fetchall()
        return {'items': [self.public(json.loads(row[0])) for row in rows]}

    def paths(self, dataset_id: str) -> dict[str, Path]:
        record = self.get(dataset_id)
        root = confined_path(self.settings.app_dir, record['bundle_relative_path'], must_exist=True)
        return {name: confined_path(root, name, must_exist=True) for name in FILE_NAMES if (root / name).is_file()}

    def load(self, dataset_id: str):
        from stockrl.market.validation import MarketDataError
        record = self.get(dataset_id)
        try:
            return load_market_bundle(self.paths(dataset_id), expected_hashes=record['file_hashes'])
        except MarketDataError as exc:
            raise AppError(getattr(exc, 'code', 'DATA_CONTRACT_INCOMPLETE'), str(exc), 422) from exc
