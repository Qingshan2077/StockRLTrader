"""Explicit, read-only indexing of the fixed pre-web RL artifact layout."""
from pathlib import Path
from datetime import datetime
from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError

from .artifact_protocol import BASELINES, baseline_aggregate, hash_file, public_aggregate, public_runs, read_json, report_rows
from .errors import AppError
from .models import ArtifactStorageRecord, ExperimentRecord, LegacyConfiguration, SplitInterval, Versions, utc_now
from .settings import AppSettings, confined_path
from .storage.artifacts import ArtifactRepository
from .storage.database import Database
from .storage.experiments import ExperimentRepository

DEPENDENCIES = ('numpy', 'pandas', 'torch', 'gymnasium', 'stable-baselines3')
FEATURES = ['log_return', 'intraday_return', 'range', 'volatility_20', 'momentum_5',
            'momentum_20', 'volume_relative', 'trend_20']


def fixed_files(summary: dict, *, reports: bool = True, models: bool = True) -> list[str]:
    seeds = [run['seed'] for run in public_runs(summary)]
    files = ['summary.json', 'bars.csv'] if models else ['summary.json']
    for seed in seeds:
        prefix = f'seed_{seed}/'
        if models:
            files.extend(prefix + name for name in ('model.zip', 'normalizer.json', 'training.json'))
        if reports:
            files.extend(prefix + name for name in ('history.csv', 'trades.csv'))
            files.extend(prefix + f'baselines/{policy}/{name}' for policy in BASELINES
                         for name in ('history.csv', 'trades.csv'))
    return files


def known_protocol(summary: dict) -> bool:
    """Recognize this repository's validated baseline, not merely parseable JSON."""
    try:
        from .models import TradingConfigModel
        TradingConfigModel.model_validate(summary['config'])
        splits = {name: SplitInterval.model_validate(summary['splits'][name])
                  for name in ('train', 'validation', 'test')}
        fit = summary['normalizer_fit']
        train, validation, test = (splits[name] for name in ('train', 'validation', 'test'))
        boundaries = (train.start == 0 and validation.start == train.end and test.start == validation.end
                      and train.reward_count >= 20 and validation.reward_count >= 5 and test.reward_count >= 5
                      and all(interval.end - interval.start == interval.reward_count for interval in splits.values()))
        dates = all(datetime.fromisoformat(interval.observation_start_date)
                    < datetime.fromisoformat(interval.first_reward_date)
                    <= datetime.fromisoformat(interval.last_reward_date) for interval in splits.values())
        dates = (dates and train.last_reward_date == validation.observation_start_date
                 and validation.last_reward_date == test.observation_start_date)
        return (summary.get('algorithm') in ('PPO', 'SAC')
                and boundaries and dates
                and fit == {'start': 0, 'end': splits['train'].end}
                and summary.get('feature_names') == FEATURES
                and all(run.get('checkpoint_selection') == 'validation_mean_reward' for run in summary['runs'])
                and all(isinstance(summary.get('versions', {}).get(name), str) for name in DEPENDENCIES)
                and isinstance(summary.get('data_sha256'), str) and len(summary['data_sha256']) == 64)
    except (KeyError, TypeError, ValueError, AttributeError, ValidationError):
        return False


def safe_label(value) -> str:
    if not isinstance(value, str):
        return '历史实验'
    return value.replace('\\', '/').rsplit('/', 1)[-1][:200] or '历史实验'


class LegacyImporter:
    def __init__(self, settings: AppSettings, db: Database):
        self.settings, self.db = settings, db

    def _artifact(self, experiment_id: str, root: Path, filename: str, *, public_name: str | None = None):
        path = confined_path(root, filename, must_exist=True)
        relative = path.relative_to(self.settings.output_dir).as_posix()
        size = path.stat().st_size
        if not path.is_file():
            raise AppError('ARTIFACT_CORRUPT', '历史产物不是普通文件。', 409)
        name = public_name or filename
        artifact_id = str(uuid5(NAMESPACE_URL, f'stockrl:{experiment_id}:{name}'))
        return ArtifactStorageRecord(artifact_id=artifact_id, experiment_id=experiment_id, kind=path.suffix.lstrip('.'),
            filename=name, size=size, sha256=hash_file(path), root_id='output', relative_path=relative,
            download_url=f'/api/v1/artifacts/{artifact_id}/download', created_at=utc_now())

    def _source(self, root: Path, summary: dict) -> tuple[Path, dict]:
        if not summary.get('evaluation_only'):
            return root, summary
        value = summary.get('source_run_dir')
        if not isinstance(value, str):
            raise AppError('REPLAY_SOURCE_MISSING', '历史重放的原始来源无法确认。', 409)
        source = Path(value).expanduser()
        source = (source if source.is_absolute() else root / source).resolve()
        try:
            relative = source.relative_to(self.settings.output_dir)
        except ValueError as exc:
            raise AppError('REPLAY_SOURCE_MISSING', '历史重放来源不在允许的产物目录内。', 409) from exc
        source = confined_path(self.settings.output_dir, relative, must_exist=True)
        original = read_json(confined_path(source, 'summary.json', must_exist=True))
        if (original.get('evaluation_only') or not known_protocol(original)
                or original.get('data_sha256') != summary.get('data_sha256')
                or original.get('splits') != summary.get('splits')
                or [run['seed'] for run in public_runs(original)] != [run['seed'] for run in public_runs(summary)]):
            raise AppError('REPLAY_SOURCE_MISSING', '历史重放来源与报告不一致。', 409)
        return source, original

    def import_folder(self, relative: str) -> ExperimentRecord:
        root = confined_path(self.settings.output_dir, relative, must_exist=True)
        if not root.is_dir() or relative.startswith('.staging'):
            raise AppError('INVALID_IMPORT', '只能索引完成产物目录。')
        summary_path = confined_path(root, 'summary.json', must_exist=True)
        fingerprint = hash_file(summary_path)
        experiment_id = str(uuid5(NAMESPACE_URL, f'stockrl-legacy:{relative}:{fingerprint}'))
        repository = ExperimentRepository(self.db)
        try:
            return repository.get(experiment_id)
        except AppError as exc:
            if exc.code != 'EXPERIMENT_NOT_FOUND':
                raise
        record = ExperimentRecord(experiment_id=experiment_id, created_at=utc_now(),
            output_relative_path=relative, legacy_fingerprint=fingerprint, integrity='corrupt',
            versions=Versions(artifact_schema_version='legacy-v0', core_semantics_version='unknown', metrics_version='unknown'),
            replay_block_reason='历史产物不完整或无法识别。')
        artifacts = []
        try:
            if (root / 'manifest.json').exists():
                manifest = read_json(confined_path(root, 'manifest.json', must_exist=True))
                record.integrity = 'unsupported'
                record.versions.artifact_schema_version = manifest.get('schema_version') if type(manifest.get('schema_version')) is int else 'unknown'
                record.replay_block_reason = '此维护命令仅导入 legacy-v0；带清单产物需恢复原应用数据库。'
            else:
                summary = read_json(summary_path)
                record.kind = 'replay' if summary.get('evaluation_only') is True else 'train'
                record.run_id = safe_label(summary.get('run_id'))
                record.data_label = safe_label(summary.get('data_label'))
                # Do not infer real market data from an unfamiliar historic label.
                record.is_synthetic = True if 'demo' in str(summary.get('data_label', '')).lower() else None
                record.runs = public_runs(summary)
                try:
                    record.legacy_config = LegacyConfiguration.model_validate({
                        'algorithm': summary.get('algorithm'), 'trading_config': summary.get('config'),
                        'timesteps': summary.get('timesteps'), 'seeds': [run['seed'] for run in record.runs],
                        'episode_length': summary.get('episode_length'), 'train_ratio': summary.get('train_ratio'),
                        'val_ratio': summary.get('val_ratio')})
                except ValidationError:
                    record.legacy_config = None
                record.aggregate = public_aggregate(summary.get('aggregate'), runs=record.runs)
                record.baseline_aggregate = baseline_aggregate(record.runs)
                record.splits = {name: SplitInterval.model_validate(summary['splits'][name])
                                 for name in ('train', 'validation', 'test')}
                record.filtered_data_sha256 = summary.get('data_sha256')
                for name in fixed_files(summary, models=False):
                    if name.endswith('.csv'):
                        for _ in report_rows(confined_path(root, name, must_exist=True),
                                             history=name.endswith('history.csv'), max_rows=self.settings.max_dataset_rows,
                                             interval=record.splits['test'].model_dump()):
                            pass
                    artifacts.append(self._artifact(experiment_id, root, name))
                record.integrity = 'complete'
                if known_protocol(summary):
                    record.versions.core_semantics_version = 1
                    record.versions.metrics_version = 1
                try:
                    source, source_summary = self._source(root, summary)
                    if not known_protocol(source_summary):
                        raise AppError('UNKNOWN_PROTOCOL', '无法确认历史模型的训练与指标协议。', 409)
                    source_artifacts = []
                    for name in fixed_files(source_summary, reports=False):
                        public_name = f'replay-source/{name}' if source != root else name
                        source_artifacts.append(self._artifact(experiment_id, source, name, public_name=public_name))
                    bars = confined_path(source, 'bars.csv', must_exist=True)
                    if hash_file(bars) != source_summary['data_sha256']:
                        raise AppError('ARTIFACT_CORRUPT', '历史行情指纹不一致。', 409)
                    existing = {item.filename for item in artifacts}
                    artifacts.extend(item for item in source_artifacts if item.filename not in existing)
                    record.replayable, record.replay_block_reason = True, None
                except AppError as exc:
                    record.replay_block_reason = exc.message
        except (AppError, OSError, KeyError, ValueError, TypeError, AttributeError, ValidationError):
            record.integrity = 'corrupt'
            record.runs, record.aggregate = [], None
            record.baseline_aggregate = {}
            record.replayable = False
            record.replay_block_reason = '历史报表缺失、损坏或格式不符合旧协议。'
        record.artifacts = [item.public() for item in artifacts]
        with self.db.transaction() as connection:
            # Import commands may run concurrently; the deterministic identity is unique.
            row = connection.execute('SELECT experiment_id FROM experiments WHERE experiment_id=?', (experiment_id,)).fetchone()
            if row:
                return repository.get(experiment_id, connection)
            repository.insert(record, connection)
            for artifact in artifacts:
                ArtifactRepository(self.db).insert(artifact, connection)
        return record

    def scan(self) -> list[ExperimentRecord]:
        if not self.settings.output_dir.exists():
            return []
        results = []
        # Old CLI replay folders are one level beneath a run; no arbitrary recursive browsing.
        candidates = set(self.settings.output_dir.glob('*/summary.json')) | set(self.settings.output_dir.glob('*/*/summary.json'))
        for path in sorted(candidates):
            relative = path.parent.relative_to(self.settings.output_dir).as_posix()
            if relative.startswith('.'):
                continue
            try:
                results.append(self.import_folder(relative))
            except (AppError, OSError) as exc:
                if isinstance(exc, AppError) and exc.status_code >= 500:
                    raise
                # Unreadable/escaping directories have no trustworthy fingerprint to index.
                # Keep scanning other independent reports; never mutate the source.
                import logging
                logging.getLogger(__name__).warning('Skipped inaccessible legacy report: %s', relative)
        return results
