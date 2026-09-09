"""Read indexed reports without importing or deserializing a learning model."""
import base64
import json
from pathlib import Path

from .artifact_protocol import BASELINES, hash_file, report_rows
from .errors import AppError
from .models import ArtifactStorageRecord, ExperimentDetail
from .result_models import ComparisonResponse, SeriesPoint, SeriesResponse, TradeRow, TradesResponse
from .settings import AppSettings, confined_path
from .storage.artifacts import ArtifactRepository
from .storage.database import Database
from .storage.experiments import ExperimentRepository
from .storage.jobs import JobRepository


class ResultService:
    def __init__(self, settings: AppSettings, db: Database):
        self.settings, self.db = settings, db
        self.experiments = ExperimentRepository(db)
        self.artifacts = ArtifactRepository(db)

    def artifact_path(self, record: ArtifactStorageRecord) -> Path:
        root = self.settings.output_dir if record.root_id == 'output' else self.settings.app_dir
        path = confined_path(root, record.relative_path, must_exist=True)
        try:
            if not path.is_file() or path.stat().st_size != record.size or hash_file(path) != record.sha256:
                raise AppError('ARTIFACT_CORRUPT', '实验文件已改变或不完整，无法读取。', 409)
        except OSError as exc:
            raise AppError('ARTIFACT_CORRUPT', '实验文件暂时无法读取。', 409) from exc
        return path

    def download(self, artifact_id: str) -> tuple[Path, ArtifactStorageRecord]:
        record = self.artifacts.get(artifact_id)
        return self.artifact_path(record), record

    def detail(self, experiment_id: str) -> ExperimentDetail:
        record = self.experiments.get(experiment_id)
        if record.kind == 'research':
            raise AppError('RESEARCH_ROUTE_REQUIRED', '请使用研究页面读取此结果。', 409)
        # Report integrity is independent of the immutable historical job outcome.
        if record.integrity == 'complete':
            for artifact in self.artifacts.list(experiment_id):
                try:
                    self.artifact_path(artifact)
                except AppError:
                    record.replayable = False
                    if artifact.filename.startswith('replay-source/'):
                        record.replay_block_reason = '历史重放的原始来源已改变或缺失；仍可浏览已保存报表。'
                    else:
                        record.integrity = 'corrupt'
                        record.replay_block_reason = '已登记产物缺失或指纹改变，请检查本地文件。'
        return record.public()

    def _csv(self, experiment_id: str, seed: int, policy: str, filename: str):
        experiment = self.experiments.get(experiment_id)
        if experiment.kind == 'research':
            raise AppError('RESEARCH_ROUTE_REQUIRED', '研究结果使用独立的指标与产物协议。', 409)
        if experiment.integrity != 'complete':
            raise AppError('RESULT_UNAVAILABLE', '此实验尚无完整可读取的结果。', 409)
        if type(seed) is not int or seed not in [run.get('seed') for run in experiment.runs]:
            raise AppError('SEED_NOT_FOUND', '此实验没有该 seed。', 404)
        if policy not in ('rl', *BASELINES):
            raise AppError('INVALID_POLICY', '基准名称无效。')
        suffix = f'seed_{seed}/' + ('' if policy == 'rl' else f'baselines/{policy}/') + filename
        matches = [item for item in self.artifacts.list(experiment_id)
                   if item.filename == suffix]
        if len(matches) != 1:
            raise AppError('ARTIFACT_CORRUPT', '缺少所选策略的结果文件。', 409)
        artifact = matches[0]
        return self.artifact_path(artifact), artifact

    def series(self, experiment_id: str, seed: int, policy: str = 'rl', max_points: int = 5000) -> SeriesResponse:
        if type(max_points) is not int or not 100 <= max_points <= 5000:
            raise AppError('INVALID_PAGINATION', '曲线点数必须介于 100 和 5000。')
        path, _ = self._csv(experiment_id, seed, policy, 'history.csv')
        rows = list(report_rows(path, history=True, max_rows=self.settings.max_dataset_rows))
        count = len(rows)
        if not count:
            raise AppError('ARTIFACT_CORRUPT', '历史曲线为空。', 409)
        # Deterministic even spacing, preserving both endpoints; metrics are never recomputed here.
        indices = range(count) if count <= max_points else [i * (count - 1) // (max_points - 1) for i in range(max_points)]
        points = []
        for index in indices:
            points.append(SeriesPoint(**rows[index]))
        return SeriesResponse(seed=seed, policy=policy, points=points, original_count=count,
                              returned_count=len(points), is_sampled=count > len(points))

    def trades(self, experiment_id: str, seed: int, policy: str = 'rl', *, cursor: str | None = None,
               limit: int = 20) -> TradesResponse:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise AppError('INVALID_PAGINATION', '每页条数必须介于 1 和 100。')
        path, artifact = self._csv(experiment_id, seed, policy, 'trades.csv')
        scope = [artifact.artifact_id, artifact.sha256]
        offset = 0
        if cursor:
            try:
                if len(cursor) > 1024:
                    raise ValueError('large cursor')
                decoded = json.loads(base64.urlsafe_b64decode(cursor + '=' * (-len(cursor) % 4)))
                if not isinstance(decoded, list) or len(decoded) != 3 or decoded[:2] != scope:
                    raise ValueError('cursor scope')
                offset = decoded[2]
                if type(offset) is not int or offset < 0:
                    raise ValueError('cursor offset')
            except (ValueError, TypeError, UnicodeError) as exc:
                raise AppError('INVALID_CURSOR', '成交分页游标无效。') from exc
        items, total = [], 0
        for index, row in enumerate(report_rows(path, history=False, max_rows=self.settings.max_dataset_rows)):
            total += 1
            if offset <= index < offset + limit:
                items.append(TradeRow(**row))
        more = offset + len(items) < total
        next_cursor = base64.urlsafe_b64encode(json.dumps([*scope, offset + len(items)]).encode()).decode().rstrip('=') if more else None
        return TradesResponse(seed=seed, policy=policy, items=items, total_rows=total,
                              next_cursor=next_cursor, has_more=more)

    def compare(self, ids: list[str]) -> ComparisonResponse:
        if not 1 <= len(ids) <= 4 or len(set(ids)) != len(ids):
            raise AppError('INVALID_COMPARISON', '请选择 1 到 4 个不同的完整训练实验。')
        details, signatures = [], []
        for experiment_id in ids:
            raw = self.experiments.get(experiment_id)
            detail = self.detail(experiment_id)
            if detail.kind != 'train' or detail.integrity != 'complete':
                raise AppError('INVALID_COMPARISON', '比较仅接受完整的训练实验。', 409)
            if detail.job_id and JobRepository(self.db).get(detail.job_id).status != 'succeeded':
                raise AppError('INVALID_COMPARISON', '比较实验的任务尚未成功。', 409)
            test = detail.splits.get('test')
            signatures.append({
                'filtered_data_sha256': raw.filtered_data_sha256,
                'test_interval': test.model_dump() if test else None,
                'trading_config': (detail.request.trading_config.model_dump() if detail.request else
                    detail.legacy_config.trading_config.model_dump() if detail.legacy_config and detail.legacy_config.trading_config else None),
                'quote_unit': detail.dataset.quote_unit if detail.dataset else 'unknown',
                'core_semantics_version': detail.versions.core_semantics_version,
                'metrics_version': detail.versions.metrics_version,
            })
            details.append(detail)
        differences = []
        for field in signatures[0]:
            values = [signature[field] for signature in signatures]
            if any(value is None or value == 'unknown' for value in values) or any(value != values[0] for value in values[1:]):
                differences.append({'field': field, 'values': values})
        return ComparisonResponse(experiments=details, same_conditions=not differences, differences=differences,
                                  notes=['展示所有 seed，不按测试成绩挑选最佳 seed。',
                                         '总体标准差使用 ddof=0；不同条件不生成合并排名。',
                                         '这些统计不构成风险匹配验证或新增样本外证据。'])
