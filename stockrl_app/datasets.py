"""Immutable, validated daily bars. Input names never determine file paths."""
import builtins
from collections.abc import Iterable, Iterator
import hashlib
from io import BytesIO
import os
from pathlib import Path, PureWindowsPath
import tempfile
from typing import BinaryIO
from uuid import uuid4

import pandas as pd

from stockrl.data import validate_bars, make_demo_data
from stockrl.protocol import split_intervals

from .errors import AppError
from .models import (BarRow, Dataset, DatasetMetadata, DatasetPreview, DatasetRecord, DemoDatasetRequest,
                     LocalDatasetRequest, LocalSource, Page, utc_now)
from .settings import AppSettings, confined_path
from .storage.database import Database
from .storage.datasets import DatasetRepository


def bars_bytes(bars: pd.DataFrame) -> bytes:
    """One deterministic serialization shared by data and filtered experiment fingerprints."""
    return bars.to_csv(index=True, index_label='Date', lineterminator='\n', float_format='%.17g').encode('utf-8')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_csv(source: Path | BytesIO, max_rows: int) -> pd.DataFrame:
    try:
        frame = pd.read_csv(source, nrows=max_rows + 1, float_precision='round_trip')
        if len(frame) > max_rows:
            raise AppError('DATASET_ROW_LIMIT', f'数据最多允许 {max_rows} 行。', 413)
        if 'Date' not in frame:
            raise ValueError('CSV 缺少 Date 列')
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop('Date'), errors='raise'), name='Date')
        return validate_bars(frame)
    except AppError:
        raise
    except (ValueError, TypeError, UnicodeError, OverflowError) as exc:
        raise AppError('INVALID_DATASET', 'CSV 数据无效，请检查日期、重复交易日与 OHLCV 数值。', 422,
                       [{'field': 'file', 'reason': str(exc)[:500]}]) from exc


class DatasetService:
    def __init__(self, settings: AppSettings, db: Database) -> None:
        self.settings = settings
        self.db = db
        self.repository = DatasetRepository(db)

    def get(self, dataset_id: str) -> Dataset:
        return self.repository.get(dataset_id).public()

    def list(self, *, limit: int = 20, cursor: str | None = None) -> Page[Dataset]:
        page = self.repository.list(limit=limit, cursor=cursor)
        return Page[Dataset](items=[row.public() for row in page.items], next_cursor=page.next_cursor, has_more=page.has_more)

    def local_sources(self) -> builtins.list[LocalSource]:
        root = self.settings.local_source_dir
        if not root.is_dir():
            return []
        sources = []
        for candidate in sorted(root.rglob('*_raw.csv')):
            relative = candidate.relative_to(root)
            try:
                path = confined_path(root, relative, must_exist=True)
            except AppError:
                continue
            if path.is_file():
                source_id = hashlib.sha256(relative.as_posix().encode()).hexdigest()
                sources.append(LocalSource(source_id=source_id, display_name=relative.as_posix()))
        return sources

    def create_local(self, request: LocalDatasetRequest) -> Dataset:
        # Re-enumerate for each selection; no client-provided server path is accepted.
        for source in self.local_sources():
            if source.source_id == request.source_id:
                path = confined_path(self.settings.local_source_dir, source.display_name, must_exist=True)
                metadata = DatasetMetadata.model_validate(request.model_dump(exclude={'source_id'}))
                try:
                    with path.open('rb') as stream:
                        return self._create_stream(stream, source.display_name, metadata, 'local', source.source_id)
                except OSError as exc:
                    raise AppError('LOCAL_SOURCE_UNAVAILABLE', '本地行情文件已不可读取，请重新选择。', 409) from exc
        raise AppError('LOCAL_SOURCE_NOT_FOUND', '此本地行情来源不存在或已不可用。', 404)

    def create_csv(self, stream: BinaryIO | Iterable[bytes], filename: str,
                   metadata: DatasetMetadata | None = None) -> Dataset:
        return self._create_stream(stream, filename, metadata or DatasetMetadata(), 'csv', None)

    @staticmethod
    def _chunks(stream: BinaryIO | Iterable[bytes]) -> Iterator[bytes]:
        if hasattr(stream, 'read'):
            while chunk := stream.read(64 * 1024):
                yield chunk
        else:
            yield from stream

    def _create_stream(self, stream: BinaryIO | Iterable[bytes], filename: str, metadata: DatasetMetadata,
                       source_kind: str, source_reference: str | None) -> Dataset:
        # Check initialized storage before creating even temporary application files.
        with self.db.connection():
            pass
        incoming = confined_path(self.settings.app_dir, '.incoming')
        incoming.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix='upload-', suffix='.csv', dir=incoming)
        temporary = confined_path(incoming, Path(temporary_name).name, must_exist=True)
        digest, total = hashlib.sha256(), 0
        try:
            with os.fdopen(descriptor, 'wb') as destination:
                for chunk in self._chunks(stream):
                    if not isinstance(chunk, bytes):
                        raise AppError('INVALID_UPLOAD', '上传内容必须是文件字节。')
                    total += len(chunk)
                    if total > self.settings.upload_limit_bytes:
                        raise AppError('UPLOAD_TOO_LARGE', 'CSV 超过 20 MiB 上传限制。', 413)
                    digest.update(chunk)
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            bars = parse_csv(temporary, self.settings.max_dataset_rows)
            return self._register(bars, metadata, source_kind, filename=filename, original=temporary,
                                  original_sha256=digest.hexdigest(), source_reference=source_reference)
        finally:
            temporary.unlink(missing_ok=True)

    def create_demo(self, request: DemoDatasetRequest) -> Dataset:
        if request.rows > self.settings.max_dataset_rows:
            raise AppError('DATASET_ROW_LIMIT', '演示数据超过行数限制。', 413)
        try:
            bars = make_demo_data(request.rows, request.seed)
        except (ValueError, OverflowError) as exc:
            raise AppError('INVALID_DEMO', '无法生成此行数的日线演示数据。', 422) from exc
        metadata = DatasetMetadata.model_validate(request.model_dump(exclude={'rows', 'seed'}))
        metadata.quote_unit = 'synthetic_unit'
        return self._register(bars, metadata, 'demo', filename=f'合成演示 {request.rows} 行',
                              source_reference=f'demo:rows={request.rows};seed={request.seed}')

    def _register(self, bars: pd.DataFrame, metadata: DatasetMetadata, source_kind: str, *, filename: str,
                  original: Path | None = None, original_sha256: str | None = None,
                  source_reference: str | None = None) -> Dataset:
        with self.db.connection():
            pass
        bars = validate_bars(bars)
        if len(bars) > self.settings.max_dataset_rows:
            raise AppError('DATASET_ROW_LIMIT', '数据超过行数限制。', 413)
        content = bars_bytes(bars)
        dataset_id = str(uuid4())
        directory = confined_path(self.settings.datasets_dir, dataset_id)
        directory.mkdir(parents=True, exist_ok=False)
        snapshot = confined_path(directory, 'bars.csv')
        original_path = confined_path(directory, 'original.csv') if original else None
        try:
            with snapshot.open('xb') as destination:
                destination.write(content)
                destination.flush()
                os.fsync(destination.fileno())
            if original_path:
                # The bounded incoming file is moved only after validation succeeds.
                original.rename(original_path)
            safe_name = PureWindowsPath(filename).name.replace('\x00', '').strip()[:200] or 'CSV 数据集'
            record = DatasetRecord(dataset_id=dataset_id, source_kind=source_kind,
                display_name=metadata.display_name or safe_name, source_description=metadata.source_description,
                is_synthetic=source_kind == 'demo', adjustment=metadata.adjustment,
                quote_unit='synthetic_unit' if source_kind == 'demo' else metadata.quote_unit,
                rows=len(bars), first_date=bars.index[0].isoformat(), last_date=bars.index[-1].isoformat(),
                snapshot_sha256=hashlib.sha256(content).hexdigest(), original_sha256=original_sha256,
                created_at=utc_now(), snapshot_relative_path=snapshot.relative_to(self.settings.app_dir).as_posix(),
                original_relative_path=original_path.relative_to(self.settings.app_dir).as_posix() if original_path else None,
                source_reference=source_reference)
            self.repository.insert(record)
            return record.public()
        except BaseException:
            snapshot.unlink(missing_ok=True)
            if original_path:
                original_path.unlink(missing_ok=True)
            directory.rmdir()
            raise

    def read_bars(self, dataset_id: str) -> pd.DataFrame:
        record = self.repository.get(dataset_id)
        try:
            path = confined_path(self.settings.app_dir, record.snapshot_relative_path, must_exist=True)
            if sha256_file(path) != record.snapshot_sha256:
                raise AppError('DATASET_INTEGRITY', '数据集快照指纹不一致，不能继续使用。', 409)
            bars = parse_csv(path, self.settings.max_dataset_rows)
            if len(bars) != record.rows:
                raise AppError('DATASET_INTEGRITY', '数据集快照行数不一致。', 409)
            return bars
        except OSError as exc:
            raise AppError('DATASET_INTEGRITY', '数据集快照不可读取。', 409) from exc

    def preview(self, dataset_id: str, *, limit: int = 100) -> DatasetPreview:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise AppError('INVALID_PREVIEW_LIMIT', '预览行数必须介于 1 和 100。')
        dataset, bars = self.get(dataset_id), self.read_bars(dataset_id)
        warnings, trainable = [], True
        try:
            split_intervals(bars)
        except ValueError:
            trainable = False
            warnings.append('数据不足以切分为至少 20 个训练、5 个验证和 5 个测试收益区间。')
        if dataset.quote_unit == 'unknown':
            warnings.append('计价单位未声明。')
        if dataset.adjustment == 'unknown':
            warnings.append('价格调整口径未声明。')
        sample = [BarRow(Date=index.isoformat(), **{column: float(row[column]) for column in bars.columns})
                  for index, row in bars.head(limit).iterrows()]
        return DatasetPreview(dataset=dataset, total_rows=len(bars), sample=sample, trainable=trainable, warnings=warnings)
