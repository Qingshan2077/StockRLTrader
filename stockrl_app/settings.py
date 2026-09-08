"""Immutable process configuration and resolved local filesystem boundaries."""
from dataclasses import dataclass
import os
from pathlib import Path, PureWindowsPath

from .errors import AppError


def local_root(value: str | Path, base: Path) -> Path:
    raw = str(value)
    if raw.startswith(('\\\\', '//')) or PureWindowsPath(raw).drive.startswith('\\\\'):
        raise AppError('NETWORK_ROOT_UNSUPPORTED', '应用存储必须位于本机磁盘。', 422)
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else base / path).resolve()
    if str(resolved).startswith(('\\\\', '//')):
        raise AppError('NETWORK_ROOT_UNSUPPORTED', '应用存储必须位于本机磁盘。', 422)
    if os.name == 'nt':
        import ctypes
        if ctypes.windll.kernel32.GetDriveTypeW(str(resolved.anchor)) == 4:
            raise AppError('NETWORK_ROOT_UNSUPPORTED', '应用存储不支持映射网络驱动器。', 422)
    return resolved


def confined_path(root: Path, relative: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve symlinks/junctions before checking containment; never allow a root itself."""
    value = str(relative)
    candidate = Path(relative)
    anchored_root = Path(os.path.abspath(root))
    if anchored_root.resolve() != anchored_root:
        raise AppError('ROOT_CHANGED', '应用目录的实际位置已变更，请重新启动并检查配置。', 409)
    if candidate.is_absolute() or PureWindowsPath(value).is_absolute() or PureWindowsPath(value).drive:
        raise AppError('PATH_OUTSIDE_ROOT', '文件路径不在允许的目录内。', 409)
    try:
        resolved = (anchored_root / candidate).resolve(strict=must_exist)
        resolved.relative_to(anchored_root)
        if resolved == anchored_root:
            raise ValueError('root is not a file')
    except (ValueError, OSError, RuntimeError) as exc:
        raise AppError('PATH_OUTSIDE_ROOT', '文件不存在或不在允许的目录内。', 409) from exc
    return resolved


@dataclass(frozen=True)
class AppSettings:
    project_root: Path = Path(__file__).resolve().parents[1]
    app_dir: Path | str | None = None
    output_dir: Path | str | None = None
    local_source_dir: Path | str | None = None
    frontend_dir: Path | str | None = None
    upload_limit_bytes: int = 20 * 1024 * 1024
    max_dataset_rows: int = 100000
    queue_capacity: int = 20
    busy_timeout_ms: int = 5000
    heartbeat_interval_seconds: int = 5
    worker_unavailable_seconds: int = 30
    cancellation_grace_seconds: int = 15
    termination_wait_seconds: int = 5
    task_time_limit_seconds: int = 12 * 60 * 60
    host: str = '127.0.0.1'
    port: int = 8000
    allowed_hosts: tuple[str, ...] = ('127.0.0.1', 'localhost', '[::1]')
    allowed_origins: tuple[str, ...] = ('http://127.0.0.1:8000', 'http://localhost:8000', 'http://127.0.0.1:5173', 'http://localhost:5173')

    def __post_init__(self) -> None:
        root = local_root(self.project_root, Path.cwd())
        object.__setattr__(self, 'project_root', root)
        for name, default in (('app_dir', 'outputs/app'), ('output_dir', 'outputs/experiments'),
                              ('local_source_dir', 'stock_data'), ('frontend_dir', 'api/static')):
            object.__setattr__(self, name, local_root(getattr(self, name) or default, root))
        for name in ('upload_limit_bytes', 'max_dataset_rows', 'queue_capacity', 'busy_timeout_ms',
                     'heartbeat_interval_seconds', 'worker_unavailable_seconds', 'cancellation_grace_seconds',
                     'termination_wait_seconds', 'task_time_limit_seconds'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise AppError('INVALID_SETTINGS', '应用资源限制必须是正整数。')
        if self.host not in ('127.0.0.1', 'localhost', '::1'):
            raise AppError('INVALID_SETTINGS', '应用仅支持本机访问地址。')
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise AppError('INVALID_SETTINGS', '监听端口必须介于 1 和 65535。')
        origins = tuple(dict.fromkeys((*self.allowed_origins, f'http://127.0.0.1:{self.port}',
                                      f'http://localhost:{self.port}', f'http://[::1]:{self.port}')))
        object.__setattr__(self, 'allowed_origins', origins)

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> 'AppSettings':
        values: dict[str, object] = {}
        if project_root is not None:
            values['project_root'] = project_root
        for field, variable in (('app_dir', 'STOCKRL_APP_DIR'), ('output_dir', 'STOCKRL_OUTPUT_DIR')):
            if os.environ.get(variable):
                values[field] = os.environ[variable]
        if os.environ.get('STOCKRL_HOST'):
            values['host'] = os.environ['STOCKRL_HOST']
        if os.environ.get('STOCKRL_PORT'):
            try:
                values['port'] = int(os.environ['STOCKRL_PORT'])
            except ValueError as exc:
                raise AppError('INVALID_SETTINGS', '监听端口必须是整数。') from exc
        return cls(**values)

    @property
    def database_path(self) -> Path:
        return confined_path(self.app_dir, 'state.sqlite3')

    @property
    def datasets_dir(self) -> Path:
        return confined_path(self.app_dir, 'datasets')

    @property
    def staging_dir(self) -> Path:
        return confined_path(self.output_dir, '.staging')
