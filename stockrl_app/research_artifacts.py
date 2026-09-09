"""Independent v2 manifest reader; historical v1 bundles are never rewritten."""
from collections.abc import Callable
import json
import os
from pathlib import Path

from .artifact_protocol import hash_file
from .errors import AppError
from .settings import confined_path

VERSIONS = {'artifact_schema_version': 2, 'core_semantics_version': 2,
            'metrics_version': 2, 'research_protocol_version': 2}


def write_json_atomic(path: Path, value: dict) -> None:
    """Same-directory replacement: readers observe either the old or full new JSON."""
    from uuid import uuid4
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seal_research_bundle(root: Path, metadata: dict, *,
                         check: Callable[[], None] = lambda: None) -> str:
    reserved = {'schema_version', 'versions', 'files'}
    if reserved.intersection(metadata) or (root / 'manifest.json').exists():
        raise AppError('PUBLISH_CONFLICT', '清单保留字段或已有产物不可覆盖。', 409)
    entries = []
    for path in sorted(root.rglob('*')):
        check()
        relative = path.relative_to(root).as_posix()
        safe = confined_path(root, relative, must_exist=True)
        if path.is_symlink() or safe != path.absolute():
            raise AppError('ARTIFACT_CORRUPT', '产物不能包含文件链接。', 409)
        if not path.is_file():
            continue
        with safe.open('rb+') as stream:
            os.fsync(stream.fileno())
        entries.append({'path': relative, 'size': safe.stat().st_size, 'sha256': hash_file(safe)})
    write_json_atomic(root / 'manifest.json', {**metadata, 'schema_version': 2,
                                             'versions': VERSIONS, 'files': entries})
    return hash_file(root / 'manifest.json')


def verify_research_bundle(root: Path, *, expected_hash: str | None = None,
                           check: Callable[[], None] = lambda: None) -> dict:
    try:
        manifest_path = confined_path(root, 'manifest.json', must_exist=True)
        if expected_hash and hash_file(manifest_path) != expected_hash:
            raise ValueError('manifest digest')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('schema_version') != 2 or manifest.get('versions') != VERSIONS:
            raise AppError('UNSUPPORTED_ARTIFACT_VERSION', '研究产物版本不支持或版本字段不一致。', 409)
        expected = {'manifest.json'}
        for entry in manifest['files']:
            check()
            name = entry['path']
            if name in expected or not isinstance(name, str) or '\\' in name:
                raise ValueError('duplicate or invalid path')
            expected.add(name)
            safe = confined_path(root, name, must_exist=True)
            if not safe.is_file() or safe.stat().st_size != entry['size'] or hash_file(safe) != entry['sha256']:
                raise ValueError('content digest')
        actual = set()
        for path in root.rglob('*'):
            check()
            name = path.relative_to(root).as_posix()
            if path.is_symlink() or confined_path(root, name, must_exist=True) != path.absolute():
                raise ValueError('linked entry')
            if path.is_file():
                actual.add(name)
        if actual != expected:
            raise ValueError('unlisted files')
        return manifest
    except AppError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AppError('ARTIFACT_CORRUPT', '研究产物文件缺失、内容或清单指纹不符。', 409) from exc
