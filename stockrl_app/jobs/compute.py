"""Spawn-only computation entry point. Heavy learners are imported inside execute()."""
from importlib.metadata import PackageNotFoundError, version
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
from threading import Event, Thread

from filelock import FileLock, Timeout

from ..artifact_protocol import baseline_aggregate, hash_file, public_runs, read_json
from ..artifacts import seal_bundle
from ..errors import AppError
from ..settings import AppSettings, confined_path
from ..storage.database import Database
from ..storage.artifacts import ArtifactRepository
from ..storage.experiments import ExperimentRepository
from ..storage.jobs import JobRepository

DEPENDENCIES = ('numpy', 'pandas', 'torch', 'gymnasium', 'stable-baselines3')
LOG = logging.getLogger(__name__)


def dependency_versions() -> dict[str, str]:
    try:
        return {name: version(name) for name in DEPENDENCIES}
    except PackageNotFoundError as exc:
        raise AppError('DEPENDENCY_MISSING', '执行环境缺少训练依赖，请检查安装。', 409) from exc


def assert_replay_compatible(summary: dict, source_versions, installed: dict[str, str]) -> None:
    if source_versions.core_semantics_version != 1 or source_versions.metrics_version != 1:
        raise AppError('REPLAY_INCOMPATIBLE', '源实验研究协议无法确认，不能加载模型。', 409)
    saved = summary.get('versions')
    if not isinstance(saved, dict) or any(saved.get(name) != installed[name] for name in DEPENDENCIES):
        raise AppError('REPLAY_INCOMPATIBLE', '源实验依赖版本与当前环境不一致，请在匹配环境重放。', 409)


def execution_metadata(settings: AppSettings, versions: dict[str, str]) -> dict:
    commit, dirty = None, None
    try:
        options = {'cwd': settings.project_root, 'capture_output': True, 'text': True,
                   'timeout': 5, 'check': True}
        if os.name == 'nt':
            options['creationflags'] = subprocess.CREATE_NO_WINDOW
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], **options).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain'], **options).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    import torch
    return {'code_commit': commit, 'code_dirty': dirty, 'versions': versions,
            'python_version': platform.python_version(), 'device': 'cpu',
            'cpu_count': os.cpu_count(), 'torch_num_threads': torch.get_num_threads(),
            'torch_num_interop_threads': torch.get_num_interop_threads(),
            'thread_environment': {key: os.environ.get(key) for key in
                ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}}


def make_portable(root: Path, summary: dict) -> None:
    """Only storage references change; metrics, training protocol and model bytes do not."""
    import json
    summary['output_dir'], summary['data_path'] = '.', 'bars.csv'
    summary.pop('source_run_dir', None)
    for run in summary['runs']:
        prefix = f"seed_{run['seed']}"
        for field, filename in (('history_path', 'history.csv'), ('trades_path', 'trades.csv'),
                                ('model_path', 'model.zip'), ('normalizer_path', 'normalizer.json'),
                                ('validation_path', 'evaluations.npz')):
            if field in run:
                run[field] = f'{prefix}/{filename}'
        for name, baseline in run['baselines'].items():
            baseline['history_path'] = f'{prefix}/baselines/{name}/history.csv'
            baseline['trades_path'] = f'{prefix}/baselines/{name}/trades.csv'
    with (root / 'summary.json').open('w', encoding='utf-8', newline='\n') as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())


def canonicalize_training_bars(bundle: Path, snapshot: Path, summary: dict, expected_hash: str) -> None:
    """Normalize storage bytes, not values, after the unchanged core has completed."""
    import json
    shutil.copyfile(snapshot, bundle / 'bars.csv')
    fingerprint = hash_file(bundle / 'bars.csv')
    if fingerprint != expected_hash:
        raise AppError('DATASET_CHANGED', '训练期间冻结快照发生变化，结果未发布。', 409)
    summary['data_sha256'] = fingerprint
    for run in summary['runs']:
        path = bundle / f"seed_{run['seed']}/training.json"
        training = read_json(path)
        training['data_sha256'] = fingerprint
        with path.open('w', encoding='utf-8', newline='\n') as output:
            json.dump(training, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write('\n')


def freeze_replay_source(settings: AppSettings, db: Database, source_id: str,
                         verified_root: Path, destination: Path, control) -> Path:
    """Copy registered model inputs into private staging, verifying the copied bytes."""
    from ..legacy import fixed_files
    indexed = {item.filename: item for item in ArtifactRepository(db).list(source_id)}
    prefix = 'replay-source/' if 'replay-source/summary.json' in indexed else ''
    summary = read_json(verified_root / 'summary.json')
    destination.mkdir(exist_ok=False)
    for name in fixed_files(summary, reports=False):
        control.check()
        item = indexed.get(prefix + name)
        if item is None:
            raise AppError('ARTIFACT_CORRUPT', '重放输入未登记。', 409)
        source = confined_path(verified_root, name, must_exist=True)
        target = confined_path(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if target.stat().st_size != item.size or hash_file(target) != item.sha256:
            raise AppError('ARTIFACT_CORRUPT', '复制时来源实验文件发生变化。', 409)
    return destination


def execute(settings: AppSettings, job_id: str, owner_token: str, control) -> None:
    db = Database(settings.database_path, settings.busy_timeout_ms)
    job = JobRepository(db).get(job_id)
    if job.owner_token != owner_token or job.status != 'running':
        raise AppError('STATE_CONFLICT', '任务执行权已失效。', 409)
    if job.kind == 'research':
        from ..research_compute import execute_research
        return execute_research(settings, db, job, control)
    experiment = ExperimentRepository(db).get(job.experiment_id)
    stage = confined_path(settings.output_dir, f'.staging/{job_id}')
    stage.mkdir(parents=True, exist_ok=False)
    bundle = stage / 'bundle'
    versions = dependency_versions()
    control.check()
    if job.kind == 'train':
        from stockrl.data import load_csv
        from stockrl.experiments import run_experiment
        request = experiment.request
        if request is None or not experiment.bars_relative_path:
            raise AppError('INVALID_REQUEST', '训练任务缺少已冻结的数据和配置。', 409)
        bars_path = confined_path(settings.app_dir, experiment.bars_relative_path, must_exist=True)
        if hash_file(bars_path) != experiment.filtered_data_sha256:
            raise AppError('DATASET_CHANGED', '已提交的数据快照指纹不符。', 409)
        bars = load_csv(bars_path)
        control.check()
        summary = run_experiment(bars, stage, algorithm=request.algorithm, timesteps=request.timesteps,
            seeds=request.seeds, config=request.trading_config.to_core(), train_ratio=request.train_ratio,
            val_ratio=request.val_ratio, data_label=experiment.data_label or 'user_csv',
            episode_length=request.episode_length, control=control)
        core_root = Path(summary['output_dir']).resolve()
        if core_root.parent != stage or not core_root.is_dir():
            raise AppError('ARTIFACT_CORRUPT', '计算结果目录不符合约定。', 409)
        core_root.rename(bundle)
        # CSV newline differences across OSes must not sever the frozen input fingerprint.
        canonicalize_training_bars(bundle, bars_path, summary, experiment.filtered_data_sha256)
    else:
        from ..replays import resolve_replay_source
        source = ExperimentRepository(db).get(experiment.source_experiment_id)
        source_root = resolve_replay_source(settings, db, source.experiment_id)
        source_root = freeze_replay_source(settings, db, source.experiment_id, source_root,
                                          stage / 'source', control)
        source_summary = read_json(source_root / 'summary.json')
        assert_replay_compatible(source_summary, source.versions, versions)
        control.check()
        # Verification and version checks precede the first possible model deserialization.
        from stockrl.experiments import evaluate_saved_run
        summary = evaluate_saved_run(source_root, bundle, control=control)
        for name in ['bars.csv', *[f"seed_{run['seed']}/{filename}"
                                  for run in summary['runs']
                                  for filename in ('model.zip', 'normalizer.json', 'training.json')]]:
            control.check()
            source_path = confined_path(source_root, name, must_exist=True)
            destination = confined_path(bundle, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination)
    control.check()
    summary['experiment_id'] = experiment.experiment_id
    summary['source_experiment_id'] = experiment.source_experiment_id
    summary['is_synthetic'] = experiment.is_synthetic
    summary['purpose'] = experiment.request.purpose if experiment.request else 'legacy_replay'
    summary['baseline_aggregate'] = baseline_aggregate(public_runs(summary))
    make_portable(bundle, summary)
    seal_bundle(bundle, job=job, experiment=experiment,
                metadata={**execution_metadata(settings, versions), 'seeds': [run['seed'] for run in summary['runs']]})
    control.check()


def compute_entry(settings: AppSettings, job_id: str, owner_token: str, connection) -> None:
    """The OS lock lives in the computing child, including while its parent is gone."""
    from stockrl.control import ExecutionControl, ExperimentCancelled
    cancelled = Event()

    def receive_commands():
        try:
            while True:
                command = connection.recv()
                if command == 'cancel':
                    cancelled.set()
        except (EOFError, OSError):
            cancelled.set()

    def emit(event):
        connection.send({'type': 'event', 'event': event})

    Thread(target=receive_commands, name='parent-liveness', daemon=True).start()
    try:
        lock_path = confined_path(settings.app_dir, 'locks/execution.lock')
        control = ExecutionControl(cancel_requested=cancelled.is_set, event_callback=emit)
        execution_lock = FileLock(lock_path)
        while True:
            control.check()
            try:
                execution_lock.acquire(timeout=.2)
                break
            except Timeout:
                # A delayed orphan may briefly acquire the lock after recovery has
                # fenced its job. Wait cancellably instead of failing a fresh job.
                continue
        try:
            control.check()
            execute(settings, job_id, owner_token, control)
            connection.send({'type': 'done'})
        finally:
            execution_lock.release()
    except ExperimentCancelled:
        try:
            connection.send({'type': 'cancelled'})
        except (OSError, BrokenPipeError):
            pass
    except BaseException as exc:
        LOG.exception('Compute failed for job %s', job_id)
        error = {'code': exc.code, 'message': exc.message} if isinstance(exc, AppError) else {
            'code': 'COMPUTE_FAILED', 'message': '计算未完成，请使用任务 ID 查看本地日志。'}
        try:
            connection.send({'type': 'error', 'error': error})
        except (OSError, BrokenPipeError):
            pass
    finally:
        connection.close()
